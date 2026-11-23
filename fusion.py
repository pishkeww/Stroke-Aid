from collections import deque

import numpy as np

PERIPHERAL_DISCOUNT = 1
GAZE_DEV_DEG = 10.0
GAZE_SMALL_DEG = 5.0
HEAD_DEV_DEG = 15.0
MIN_SAMPLES = 20
MIN_CONF = 0.5
RACE_LVO_CUTOFF = 5

DISCLAIMER = ("Decision support only - not a diagnostic device, not clinically validated. "
              "Final triage and routing remain the clinician's judgement.")

_UNSET = object()


def _opposite(side):
    return {"patient_left": "patient_right", "patient_right": "patient_left"}.get(side)


class LVOFusion:
    def __init__(self, window=90):
        self._gaze = deque(maxlen=window)
        self._yaw = deque(maxlen=window)
        self.manual = {"arm": None, "leg": None, "speech": None}

    def set_manual(self, arm=_UNSET, leg=_UNSET, speech=_UNSET):
        for name, val in (("arm", arm), ("leg", leg), ("speech", speech)):
            if val is _UNSET:
                continue
            if val is not None and val not in (0, 1, 2):
                raise ValueError(f"{name} must be 0, 1, 2 or None")
            self.manual[name] = val

    def reset(self):
        self._gaze.clear()
        self._yaw.clear()

    def update(self, scores, features=None):
        features = features or {}
        fac = scores.get("facial") or features.get("facial") or {}
        conf = scores.get("confidence", 1.0)
        flags = []

        if conf >= MIN_CONF:
            self._gaze.append(scores.get("gaze_signed_deg", 0.0))
            self._yaw.append(scores.get("yaw", 0.0))

        face = None
        if fac.get("ready"):
            li = fac["lower_idx"]
            face = 0 if li < 0.25 else 1 if li < 0.60 else 2
            if fac["pattern"] == "PERIPHERAL" and face > 0:
                face = max(0, face - PERIPHERAL_DISCOUNT)
                flags.append("Peripheral (Bell's-like) facial pattern: face points reduced")
            if fac["pattern"] == "LOWER_ONLY":
                flags.append("Upper-face tasks not done: central vs peripheral not yet classified")
            if fac["pattern"] == "MIXED":
                flags.append("Facial findings inconsistent between sides - re-check")
            if not fac.get("exam_done") and fac["stage"] == "IDLE":
                flags.append("Facial exam not run: face score uses resting asymmetry only")

        gaze_head, dev_side, g_med, y_med = None, None, 0.0, 0.0
        if len(self._gaze) >= MIN_SAMPLES:
            g_med = float(np.median(self._gaze))
            y_med = float(np.median(self._yaw))
            gaze_dev = abs(g_med) >= GAZE_DEV_DEG
            head_dev = abs(y_med) >= HEAD_DEV_DEG
            concordant = abs(g_med) < GAZE_SMALL_DEG or np.sign(g_med) == np.sign(y_med)
            gaze_head = 1 if (gaze_dev or (head_dev and concordant)) else 0
            if gaze_head:
                dev_side = ("patient_left" if (g_med if gaze_dev else y_med) > 0
                            else "patient_right")
                flags.append(f"Sustained gaze/head deviation toward {dev_side} "
                             f"(gaze {g_med:+.0f} deg, yaw {y_med:+.0f} deg)")

        weak = fac.get("weak_side")
        if dev_side and weak:
            ok = weak == _opposite(dev_side)
            lateral = "consistent" if ok else "inconsistent"
            if not ok:
                flags.append("Face weakness is on the SAME side as gaze/head deviation: "
                             "unusual for a hemispheric stroke - check for mimic/brainstem")
        else:
            lateral = "unknown"
        lesion_side = _opposite(weak) if weak and lateral != "inconsistent" else dev_side

        items = {"face": face, "gaze_head": gaze_head, **self.manual}
        entered = {k: v for k, v in items.items() if v is not None}
        total = int(sum(entered.values()))
        pending_max = (2 * (face is None) + 1 * (gaze_head is None) +
                       2 * sum(1 for k in ("arm", "leg", "speech") if self.manual[k] is None))
        complete = len(entered) == 5

        n_good = len(self._gaze)
        if face is None and gaze_head is None and not entered:
            level = "INSUFFICIENT_DATA"
        elif total >= RACE_LVO_CUTOFF:
            level = "HIGH"
        elif total >= 3 or gaze_head == 1:
            level = "ELEVATED"
        else:
            level = "LOW"
        provisional = not complete
        if provisional and level in ("LOW", "ELEVATED") and total + pending_max >= RACE_LVO_CUTOFF:
            flags.append("Enter arm / leg / speech to complete RACE "
                         f"(could still reach >= {RACE_LVO_CUTOFF})")
        if gaze_head == 1 and face == 0 and level == "ELEVATED":
            flags.append("Gaze deviation is a cortical sign: treat as possible LVO even with normal face")
        if n_good < MIN_SAMPLES:
            flags.append("Waiting for stable, well-framed gaze/head data")

        return {
            "race_face": face, "race_gaze_head": gaze_head,
            "race_arm": self.manual["arm"], "race_leg": self.manual["leg"],
            "race_speech": self.manual["speech"],
            "race_total": total, "race_max": 9, "pending_max": pending_max,
            "complete": complete, "provisional": provisional,
            "level": level,
            "recommendation": _recommend(level, provisional),
            "pattern": fac.get("pattern", "INSUFFICIENT"),
            "weak_face_side": weak, "deviation_side": dev_side,
            "lateralisation": lateral, "suspected_lesion_side": lesion_side,
            "flags": flags, "quality": conf, "disclaimer": DISCLAIMER,
        }


def _recommend(level, provisional):
    tail = " (provisional - RACE incomplete)" if provisional else ""
    return {
        "HIGH": "LVO suspected: transport to a thrombectomy-capable stroke centre (CT/CTA), "
                "pre-alert the centre, avoid diversion to a non-CT facility." + tail,
        "ELEVATED": "Possible stroke/LVO: complete RACE (arm, leg, speech). If RACE >= 5 or deficits "
                    "persist, route to a thrombectomy-capable centre; otherwise nearest CT-capable "
                    "stroke unit." + tail,
        "LOW": "Few camera-detected deficits. This does NOT exclude stroke: apply full FAST/BE-FAST; "
               "if any sign is present go to the nearest CT-capable stroke unit." + tail,
        "INSUFFICIENT_DATA": "Not enough usable data: face the camera, improve lighting, "
                             "keep head still and frontal.",
    }[level]
