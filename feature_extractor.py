import math
import numpy as np
import cv2

from facial_analysis import FacialAnalyzer

IRIS_L = [469, 470, 471, 472]
IRIS_R = [474, 475, 476, 477]

EYE_L_CORNERS = (33, 133)
EYE_R_CORNERS = (362, 263)

EYE_R_TOP, EYE_R_BOTTOM = [386, 385, 384, 387], [374, 373, 390, 388]
EYE_L_TOP, EYE_L_BOTTOM = [159, 158, 157, 160], [145, 144, 163, 153]

CHEEK_L_HIGH, CHEEK_L_LOW = 116, 207
CHEEK_R_HIGH, CHEEK_R_LOW = 345, 427

MOUTH_L, MOUTH_R = 61, 291
BROW_L, BROW_R = 105, 334
FOREHEAD, NOSE_TIP, CHIN = 10, 1, 152

INNER_L, INNER_R = 133, 362

_EYEBALL_R_OVER_WIDTH = 0.42

_MODEL_LM_IDS = [1, 152, 263, 33, 291, 61]
_MODEL_3D = np.array([
    [ 0.000,  0.000,  0.000],
    [ 0.000,  0.066,  0.013],
    [ 0.045, -0.034,  0.027],
    [-0.045, -0.034,  0.027],
    [ 0.030,  0.030,  0.025],
    [-0.030,  0.030,  0.025],
], dtype=np.float64)


def _lm(lms, idx, w, h):
    p = lms[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float64)


def _lm_mean(lms, indices, w, h):
    return np.mean([_lm(lms, i, w, h) for i in indices], axis=0)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def estimate_head_pose(lms, w, h):
    image_pts = np.array([_lm(lms, i, w, h) for i in _MODEL_LM_IDS], dtype=np.float64)
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(_MODEL_3D, image_pts, cam, np.zeros((4, 1)),
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:
        return 0.0, 0.0, 0.0

    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return -math.degrees(rx), -math.degrees(ry), math.degrees(rz)


def compute_ear(lms, top_ids, bot_ids, c0, c1, w, h):
    top = _lm_mean(lms, top_ids, w, h)
    bot = _lm_mean(lms, bot_ids, w, h)
    return _dist(top, bot) / (_dist(_lm(lms, c0, w, h), _lm(lms, c1, w, h)) + 1e-9)


def _iris_offset(lms, iris_ids, left_corner, right_corner, w, h):
    iris = _lm_mean(lms, iris_ids, w, h)
    a = _lm(lms, left_corner, w, h)
    b = _lm(lms, right_corner, w, h)
    axis = b - a
    return float(np.dot(iris - a, axis) / (np.dot(axis, axis) + 1e-9) - 0.5), iris


class FeatureExtractor:

    def __init__(self):
        self.facial = FacialAnalyzer()

    def extract(self, lms, w, h, t=None):
        iod = max(_dist(_lm(lms, INNER_L, w, h), _lm(lms, INNER_R, w, h)), 1.0)
        pitch, yaw, roll = estimate_head_pose(lms, w, h)

        off_l, iris_l = _iris_offset(lms, IRIS_L, *EYE_L_CORNERS, w, h)
        off_r, iris_r = _iris_offset(lms, IRIS_R, *EYE_R_CORNERS, w, h)
        raw_gaze = (off_l + off_r) / 2.0
        gaze_signed = math.degrees(math.asin(
            max(-1.0, min(1.0, raw_gaze / _EYEBALL_R_OVER_WIDTH))))

        ear_l = compute_ear(lms, EYE_L_TOP, EYE_L_BOTTOM, *EYE_L_CORNERS, w, h)
        ear_r = compute_ear(lms, EYE_R_TOP, EYE_R_BOTTOM, *EYE_R_CORNERS, w, h)
        avg_ear = (ear_l + ear_r) / 2.0
        blink_asym = abs(ear_l - ear_r) / (avg_ear + 1e-9)

        facial = self.facial.update(lms, w, h, yaw=yaw, pitch=pitch, roll=roll, t=t)

        ml, mr = _lm(lms, MOUTH_L, w, h), _lm(lms, MOUTH_R, w, h)
        lb, rb = _lm(lms, BROW_L, w, h), _lm(lms, BROW_R, w, h)
        fore = _lm(lms, FOREHEAD, w, h)

        nlf_l = _dist(_lm(lms, CHEEK_L_HIGH, w, h), _lm(lms, CHEEK_L_LOW, w, h)) / iod
        nlf_r = _dist(_lm(lms, CHEEK_R_HIGH, w, h), _lm(lms, CHEEK_R_LOW, w, h)) / iod
        nlf_asym = abs(nlf_l - nlf_r) / (max(nlf_l, nlf_r) + 1e-9)

        return {
            "pitch": pitch, "yaw": yaw, "roll": roll,
            "raw_gaze": raw_gaze,
            "corrected_gaze": raw_gaze,
            "gaze_signed_deg": gaze_signed,
            "gaze_deg": abs(gaze_signed),
            "gaze_direction": "patient_left" if gaze_signed > 0 else "patient_right",
            "ear_left": ear_r, "ear_right": ear_l,
            "avg_ear": avg_ear, "blink_asym": blink_asym,
            "mouth_droop": abs(facial["droop"]),
            "nlf_asym": nlf_asym,
            "brow_asym": facial["brow_rest_asym"],
            "face_asym": facial["lower_idx"],
            "facial": facial,
            "iod": iod,
            "left_iris_px": iris_r, "right_iris_px": iris_l,
            "mouth_left_px": ml, "mouth_right_px": mr,
            "left_brow_px": lb, "right_brow_px": rb,
            "forehead_px": fore,
            "nose_tip_px": _lm(lms, NOSE_TIP, w, h),
        }
