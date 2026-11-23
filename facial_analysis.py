import threading
import time
from collections import deque

import numpy as np

from face_dynamics import DynamicsTracker, smile_onset

OUTER_L, OUTER_R = 33, 263
INNER_L, INNER_R = 133, 362
MOUTH_L, MOUTH_R = 61, 291
BROW_L = [70, 63, 105, 66, 107]
BROW_R = [300, 293, 334, 296, 336]
LID_TOP_L, LID_TOP_R = 159, 386
EYE_TOP_L, EYE_BOT_L = [159, 158, 157, 160], [145, 144, 163, 153]
EYE_TOP_R, EYE_BOT_R = [386, 385, 384, 387], [374, 373, 390, 388]

CAMERA_MIRRORED = False

YAW_MAX, PITCH_MAX = 20.0, 25.0
MIN_OCW_PX = 40.0

REST_DROOP_NORMAL, REST_DROOP_SPAN = 0.025, 0.050
SMILE_MIN_EXC = 0.04
SMILE_ASYM_NORMAL, SMILE_ASYM_SPAN = 0.15, 0.30
BROW_REST_NORMAL, BROW_REST_SPAN = 0.020, 0.050
BROW_MIN_EXC = 0.015
BROW_ASYM_NORMAL, BROW_ASYM_SPAN = 0.15, 0.30
EYE_ATTEMPT_EAR = 0.12
EYE_ASYM_NORMAL, EYE_ASYM_SPAN = 0.03, 0.08
LOWER_MILD, LOWER_MOD, UPPER_FLAG = 0.25, 0.60, 0.50

MIN_STAGE_SAMPLES = 10
STAGES = ("IDLE", "REST", "SMILE", "BROWS", "EYES", "DONE")
DURATION = {"REST": 3.0, "SMILE": 5.0, "BROWS": 5.0, "EYES": 5.0}
PROMPT = {
    "IDLE":  "Press 'Start facial exam'",
    "REST":  "Relax face, look at camera",
    "SMILE": "Smile widely, show teeth",
    "BROWS": "Raise both eyebrows high",
    "EYES":  "Squeeze eyes tightly shut",
    "DONE":  "Exam complete",
}


def patient_side(image_side):
    if image_side is None:
        return None
    is_img_left = image_side == "L"
    return "patient_left" if (is_img_left == CAMERA_MIRRORED) else "patient_right"


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def _idx(value, normal, span):
    return _clip01((value - normal) / span)


class FaceFrame:

    def __init__(self, lms, w, h):
        self._lms, self.w, self.h = lms, w, h
        a, b = self.px(OUTER_L), self.px(OUTER_R)
        d = b - a
        self.ocw = float(np.linalg.norm(d))
        self.u = d / max(self.ocw, 1e-9)
        self.v = np.array([-self.u[1], self.u[0]])
        self.origin = (a + b) / 2.0

    def px(self, i):
        p = self._lms[i]
        return np.array([p.x * self.w, p.y * self.h], dtype=np.float64)

    def ff(self, i):
        d = self.px(i) - self.origin
        return np.array([d @ self.u, d @ self.v]) / max(self.ocw, 1e-9)

    def ff_mean(self, ids):
        return np.mean([self.ff(i) for i in ids], axis=0)


def measure(lms, w, h):
    f = FaceFrame(lms, w, h)

    def ear(top, bot, c0, c1):
        ap = f.ff_mean(bot)[1] - f.ff_mean(top)[1]
        return ap / (np.linalg.norm(f.ff(c0) - f.ff(c1)) + 1e-9)

    cL, cR = f.ff(MOUTH_L), f.ff(MOUTH_R)
    return {
        "cL": cL, "cR": cR,
        "width": float(cR[0] - cL[0]),
        "browL": float(f.ff(LID_TOP_L)[1] - f.ff_mean(BROW_L)[1]),
        "browR": float(f.ff(LID_TOP_R)[1] - f.ff_mean(BROW_R)[1]),
        "earL": float(ear(EYE_TOP_L, EYE_BOT_L, OUTER_L, INNER_L)),
        "earR": float(ear(EYE_TOP_R, EYE_BOT_R, OUTER_R, INNER_R)),
        "ocw": f.ocw,
    }


def _med(samples, key):
    return np.median([m[key] for m in samples], axis=0)


class FacialAnalyzer:

    def __init__(self, auto_advance=True):
        self._lock = threading.Lock()
        self.auto_advance = auto_advance
        self._roll_buf = deque(maxlen=60)
        self.reset_exam()

    def reset_exam(self):
        with self._lock:
            self._stage = "IDLE"
            self._t0 = None
            self._buf = {s: [] for s in DURATION}
            self._rest = None
            self._smile = None
            self._brows = None
            self._eyes = None
            self._roll_buf.clear()
            self._dyn = DynamicsTracker()

    def start_exam(self, t=None):
        with self._lock:
            self._rest = self._smile = self._brows = self._eyes = None
            self._buf = {s: [] for s in DURATION}
            self._stage = "REST"
            self._t0 = time.monotonic() if t is None else t

    def next_stage(self, t=None):
        with self._lock:
            self._advance(time.monotonic() if t is None else t)

    @property
    def stage(self):
        return self._stage

    def _advance(self, t):
        if self._stage in DURATION:
            self._finalize(self._stage)
        i = STAGES.index(self._stage)
        self._stage = STAGES[min(i + 1, len(STAGES) - 1)] if self._stage != "IDLE" else "REST"
        self._t0 = t

    def _finalize(self, stage):
        s = [m for _, m in self._buf[stage]]
        if len(s) < MIN_STAGE_SAMPLES:
            return
        if stage == "REST":
            self._rest = {k: _med(s, k) for k in ("cL", "cR", "width", "browL", "browR")}
            return
        if self._rest is None:
            return
        r = self._rest

        if stage == "SMILE":
            gain = np.array([m["width"] - r["width"] for m in s])
            sel = [m for m, g in zip(s, gain) if g >= np.percentile(gain, 70)]
            cL, cR = _med(sel, "cL"), _med(sel, "cR")
            eL = float(np.hypot(max(0.0, -(cL[1] - r["cL"][1])), max(0.0, -(cL[0] - r["cL"][0]))))
            eR = float(np.hypot(max(0.0, -(cR[1] - r["cR"][1])), max(0.0, (cR[0] - r["cR"][0]))))
            if max(eL, eR) < SMILE_MIN_EXC:
                self._smile = {"assessed": False, "note": "no smile detected"}
            else:
                asym = abs(eL - eR) / max(eL, eR)
                on = smile_onset(self._buf["SMILE"], r, self._t0)
                self._smile = {"assessed": True, "eL": eL, "eR": eR, "asym": asym,
                               "idx": _idx(asym, SMILE_ASYM_NORMAL, SMILE_ASYM_SPAN),
                               "weak": "L" if eL < eR else "R", "onset": on}

        elif stage == "BROWS":
            tot = np.array([m["browL"] + m["browR"] for m in s])
            sel = [m for m, v in zip(s, tot) if v >= np.percentile(tot, 70)]
            rL = float(_med(sel, "browL") - r["browL"])
            rR = float(_med(sel, "browR") - r["browR"])
            if max(rL, rR) < BROW_MIN_EXC:
                self._brows = {"assessed": False, "note": "no brow raise detected"}
            else:
                asym = abs(rL - rR) / max(rL, rR)
                self._brows = {"assessed": True, "asym": asym,
                               "idx": _idx(asym, BROW_ASYM_NORMAL, BROW_ASYM_SPAN),
                               "weak": "L" if rL < rR else "R"}

        elif stage == "EYES":
            pL = float(np.percentile([m["earL"] for m in s], 10))
            pR = float(np.percentile([m["earR"] for m in s], 10))
            if min(pL, pR) >= EYE_ATTEMPT_EAR:
                self._eyes = {"assessed": False, "note": "no eye closure detected"}
            else:
                d = abs(pL - pR)
                self._eyes = {"assessed": True, "asym": d,
                              "idx": _idx(d, EYE_ASYM_NORMAL, EYE_ASYM_SPAN),
                              "weak": "L" if pL > pR else "R"}

    def update(self, lms, w, h, yaw=0.0, pitch=0.0, roll=0.0, t=None):
        t = time.monotonic() if t is None else t
        m = measure(lms, w, h)
        reason = None
        if m["ocw"] < MIN_OCW_PX:
            reason = "face too small / far"
        elif abs(yaw) > YAW_MAX:
            reason = f"head turned ({yaw:+.0f} deg)"
        elif abs(pitch) > PITCH_MAX:
            reason = f"head pitched ({pitch:+.0f} deg)"
        ok = reason is None

        with self._lock:
            self._dyn.update(t, m["earL"], m["earR"], ok)
            if ok:
                self._roll_buf.append(m)
                if self._stage in self._buf:
                    self._buf[self._stage].append((t, m))
            if (self.auto_advance and self._stage in DURATION
                    and self._t0 is not None and t - self._t0 >= DURATION[self._stage]):
                self._advance(t)
            return self._report(ok, reason, t)

    def _report(self, ok, reason, t):
        stage = self._stage
        left = max(0.0, DURATION[stage] - (t - self._t0)) if stage in DURATION and self._t0 else 0.0
        out = {
            "ready": False, "valid": ok, "quality_reason": reason,
            "stage": stage, "stage_prompt": PROMPT[stage], "stage_time_left": left,
            "exam_done": stage == "DONE",
            "droop": 0.0, "brow_rest_asym": 0.0,
            "rest_idx": 0.0, "smile_idx": None, "smile_asym": None,
            "brow_raise_asym": None, "eye_close_asym": None,
            "lower_idx": 0.0, "upper_idx": 0.0, "upper_assessed": False,
            "lower_side": None, "upper_side": None, "weak_side": None,
            "pattern": "INSUFFICIENT", "pattern_note": "collecting data",
            "nihss4": 0, "nihss4_label": "Normal",
            "smile_onset_lag_s": None, "smile_late_side": None,
        }
        out.update(self._dyn.report(lambda s: patient_side(s)))
        ref = self._rest
        if ref is None and len(self._roll_buf) >= MIN_STAGE_SAMPLES:
            ref = {k: _med(list(self._roll_buf), k) for k in ("cL", "cR", "browL", "browR")}
        if ref is None:
            return out
        out["ready"] = True

        droop = float(ref["cR"][1] - ref["cL"][1])
        rest_idx = _idx(abs(droop), REST_DROOP_NORMAL, REST_DROOP_SPAN)
        lower = [(rest_idx, "R" if droop > 0 else "L")]
        out.update(droop=droop, rest_idx=rest_idx)

        if self._smile and self._smile["assessed"]:
            out["smile_idx"], out["smile_asym"] = self._smile["idx"], self._smile["asym"]
            on = self._smile.get("onset")
            if on:
                out["smile_onset_lag_s"] = on["lag_s"]
                out["smile_late_side"] = patient_side(on["late"])
            lower.append((self._smile["idx"], self._smile["weak"]))
        lower_idx, lower_side = max(lower, key=lambda x: x[0])

        dh = float(ref["browL"] - ref["browR"])
        brow_rest_idx = _idx(abs(dh), BROW_REST_NORMAL, BROW_REST_SPAN)
        out["brow_rest_asym"] = abs(dh)
        upper = []
        if self._brows and self._brows["assessed"]:
            out["brow_raise_asym"] = self._brows["asym"]
            upper.append((self._brows["idx"], self._brows["weak"]))
        if self._eyes and self._eyes["assessed"]:
            out["eye_close_asym"] = self._eyes["asym"]
            upper.append((self._eyes["idx"], self._eyes["weak"]))
        assessed = bool(upper)
        if assessed:
            upper_idx, upper_side = max(upper, key=lambda x: x[0])
        else:
            upper_idx, upper_side = brow_rest_idx, ("L" if dh < 0 else "R")

        out.update(lower_idx=lower_idx, upper_idx=upper_idx, upper_assessed=assessed,
                   lower_side=patient_side(lower_side) if lower_idx >= LOWER_MILD else None,
                   upper_side=patient_side(upper_side) if upper_idx >= UPPER_FLAG else None)

        low, up = lower_idx >= LOWER_MILD, upper_idx >= UPPER_FLAG
        if not low and not up:
            pat, note = "NORMAL", "no facial asymmetry detected"
        elif low and not up and assessed:
            pat, note = "CENTRAL", "lower-face weakness with upper-face sparing (central pattern)"
        elif low and not up:
            pat, note = "LOWER_ONLY", "lower-face weakness; run brow/eye tasks to classify central vs peripheral"
        elif low and up:
            same = out["lower_side"] == out["upper_side"]
            pat, note = (("PERIPHERAL", "whole hemiface weak incl. upper face (Bell's-palsy-like)")
                         if same else
                         ("MIXED", "upper and lower weakness on opposite sides - re-check"))
        else:
            pat, note = "UPPER_ONLY", "upper-face asymmetry only (atypical; check ptosis/artifact)"
        out.update(pattern=pat, pattern_note=note,
                   weak_side=out["lower_side"] if low else out["upper_side"])

        if lower_idx < LOWER_MILD:
            s, lab = 0, "Normal"
        elif lower_idx < LOWER_MOD:
            s, lab = 1, "Minor Palsy"
        elif up and assessed:
            s, lab = 3, "Complete Palsy"
        else:
            s, lab = 2, "Partial Palsy"
        out.update(nihss4=s, nihss4_label=lab)
        return out
