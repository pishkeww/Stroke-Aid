from collections import deque
import numpy as np
import time


class EMAFilter:
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.value = None

    def update(self, measurement: float) -> float:
        if self.value is None:
            self.value = measurement
        else:
            self.value = self.alpha * measurement + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class KalmanFilter1D:
    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-2):
        self.Q = process_noise
        self.R = measurement_noise

        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1.0
        self.initialized = False

        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q_mat = np.array([[self.Q, 0], [0, self.Q]])
        self.R_mat = np.array([[self.R]])

    def update(self, z: float) -> float:
        if not self.initialized:
            self.x[0] = z
            self.initialized = True
            return z

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q_mat

        y = z - (self.H @ self.x)[0]
        S = (self.H @ self.P @ self.H.T + self.R_mat)[0, 0]
        K = (self.P @ self.H.T) / S
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.x[0])

    def reset(self):
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1.0
        self.initialized = False


class EventDetector:
    def __init__(self, name: str, threshold: float, min_frames: int = 15,
                 reset_ratio: float = 0.7):
        self.name = name
        self.threshold = threshold
        self.min_frames = min_frames
        self.reset_ratio = reset_ratio
        self._count   = 0
        self._active  = False
        self._events  = []

    def update(self, value: float) -> bool:
        if value >= self.threshold:
            self._count += 1
        else:
            if self._active and value < self.threshold * self.reset_ratio:
                self._active = False
            self._count = 0

        if self._count >= self.min_frames and not self._active:
            self._active = True
            self._events.append((time.time(), value))
            return True

        return False

    @property
    def active(self):
        return self._active

    @property
    def events(self):
        return list(self._events)

    def reset(self):
        self._count  = 0
        self._active = False


def score_eye_opening(avg_ear: float):
    if avg_ear > 0.20:
        return 0, "Eyes Open"
    elif avg_ear > 0.10:
        return 1, "Ptosis / Partial"
    else:
        return 2, "Eyes Closed"


def score_gaze(gaze_deg: float):
    if gaze_deg < 5.0:
        return 0, "Normal"
    elif gaze_deg < 15.0:
        return 1, "Partial Gaze Palsy"
    else:
        return 2, "Forced Deviation"


def score_visual_field(blink_asym: float):
    if blink_asym < 0.10:
        return 0, "No Loss Detected"
    elif blink_asym < 0.25:
        return 1, "Possible Hemianopia"
    else:
        return 2, "Likely Hemianopia"


def score_facial_palsy(face_asym: float):
    if face_asym < 0.05:
        return 0, "Normal"
    elif face_asym < 0.15:
        return 1, "Minor Palsy"
    elif face_asym < 0.28:
        return 2, "Partial Palsy"
    else:
        return 3, "Complete Palsy"


class ConfidenceEstimator:

    def estimate(self, features: dict) -> float:
        yaw_pen   = max(0.0, (abs(features["yaw"])   - 20) / 40)
        pitch_pen = max(0.0, (abs(features["pitch"]) - 20) / 40)
        iod_pen   = max(0.0, 1.0 - features["iod"] / 80)
        ear_pen   = max(0.0, 0.15 - features["avg_ear"]) * 4

        penalty = min(1.0, yaw_pen + pitch_pen + iod_pen + ear_pen)
        return round(1.0 - penalty, 2)


class NIHSSScorer:

    def __init__(self):
        self.kf_gaze     = KalmanFilter1D(process_noise=5e-3, measurement_noise=2e-2)
        self.kf_asym     = KalmanFilter1D(process_noise=1e-3, measurement_noise=1e-2)

        self.ema_ear_l   = EMAFilter(alpha=0.20)
        self.ema_ear_r   = EMAFilter(alpha=0.20)
        self.ema_blink   = EMAFilter(alpha=0.10)
        self.ema_droop   = EMAFilter(alpha=0.12)
        self.ema_brow    = EMAFilter(alpha=0.12)

        self.use_baseline = False
        self._gaze_init_buf = []
        self._gaze_baseline = None
        self._BASELINE_FRAMES = 30

        self.evt_gaze  = EventDetector("Sustained Gaze Deviation",
                                        threshold=5.0, min_frames=20)
        self.evt_asym  = EventDetector("Persistent Lower-Face Asymmetry",
                                        threshold=0.25, min_frames=25)
        self.evt_ptosis = EventDetector("Prolonged Ptosis / Eye Closure",
                                         threshold=0.20, min_frames=20)

        self.confidence = ConfidenceEstimator()


    def update(self, features: dict) -> dict:

        signed_raw = features["gaze_signed_deg"]
        if self.use_baseline and self._gaze_baseline is None:
            self._gaze_init_buf.append(features["raw_gaze"])
            if len(self._gaze_init_buf) >= self._BASELINE_FRAMES:
                self._gaze_baseline = float(np.median(self._gaze_init_buf))
        if self.use_baseline and self._gaze_baseline is not None:
            off = features["raw_gaze"] - self._gaze_baseline
            signed_raw = float(np.degrees(np.arcsin(np.clip(off / 0.42, -1.0, 1.0))))
        elif self.use_baseline:
            signed_raw = 0.0

        gaze_signed_sm = self.kf_gaze.update(signed_raw)
        gaze_deg_sm    = abs(gaze_signed_sm)
        gaze_direction = "patient_left" if gaze_signed_sm > 0 else "patient_right"

        ear_l = self.ema_ear_l.update(features["ear_left"])
        ear_r = self.ema_ear_r.update(features["ear_right"])
        avg_ear = (ear_l + ear_r) / 2.0
        blink_asym = self.ema_blink.update(features["blink_asym"])

        facial = features["facial"]
        droop = self.ema_droop.update(features["mouth_droop"])
        brow  = self.ema_brow.update(features["brow_asym"])
        nlf   = features["nlf_asym"]

        face_asym_sm = max(0.0, self.kf_asym.update(facial["lower_idx"]))

        eo_score, eo_label   = score_eye_opening(avg_ear)
        g_score,  g_label    = score_gaze(gaze_deg_sm)
        vf_score, vf_label   = score_visual_field(blink_asym)
        fp_score, fp_label   = facial["nihss4"], facial["nihss4_label"]

        new_events = []
        if self.evt_gaze.update(gaze_deg_sm):
            new_events.append(f"⚠ Sustained gaze deviation {gaze_deg_sm:.1f}°")
        if self.evt_asym.update(face_asym_sm):
            new_events.append(f"⚠ Persistent lower-face asymmetry idx {face_asym_sm:.2f}")
        if self.evt_ptosis.update(1.0 - avg_ear / 0.25):
            new_events.append(f"⚠ Prolonged ptosis / eye closure EAR={avg_ear:.3f}")

        conf = self.confidence.estimate({**features, "avg_ear": avg_ear})

        calibrating = self.use_baseline and self._gaze_baseline is None
        calib_pct   = int(100 * len(self._gaze_init_buf) / self._BASELINE_FRAMES) \
                      if calibrating else 100

        return {
            "gaze_deg":      gaze_deg_sm,
            "gaze_signed_deg": gaze_signed_sm,
            "facial":        facial,
            "ear_left":      ear_l,
            "ear_right":     ear_r,
            "avg_ear":       avg_ear,
            "blink_asym":    blink_asym,
            "face_asym":     face_asym_sm,
            "mouth_droop":   droop,
            "brow_asym":     brow,
            "nlf_asym":      nlf,
            "yaw": features["yaw"], "pitch": features["pitch"], "roll": features["roll"],
            "eye_opening": eo_score, "eye_opening_label": eo_label,
            "gaze":        g_score,  "gaze_label":        g_label,
            "visual_field":vf_score, "visual_field_label":vf_label,
            "facial_palsy":fp_score, "facial_palsy_label":fp_label,
            "partial_nihss": eo_score + g_score + vf_score + fp_score,
            "new_events":  new_events,
            "evt_gaze_active":  self.evt_gaze.active,
            "evt_asym_active":  self.evt_asym.active,
            "evt_ptosis_active":self.evt_ptosis.active,
            "all_events":  self.evt_gaze.events + self.evt_asym.events + self.evt_ptosis.events,
            "confidence":   conf,
            "calibrating":  calibrating,
            "calib_pct":    calib_pct,
            "gaze_direction": gaze_direction,
            "iod": features["iod"],
        }

    def reset_baseline(self):
        self._gaze_init_buf = []
        self._gaze_baseline = None
        self.kf_gaze.reset()
        self.kf_asym.reset()
        for f in [self.ema_ear_l, self.ema_ear_r, self.ema_blink,
                  self.ema_droop, self.ema_brow]:
            f.reset()
        self.evt_gaze.reset()
        self.evt_asym.reset()
        self.evt_ptosis.reset()
