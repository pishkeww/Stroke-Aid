import math
from types import SimpleNamespace

import numpy as np
import pytest

import facial_analysis as fa
from facial_analysis import FacialAnalyzer, DURATION
from feature_extractor import FeatureExtractor, estimate_head_pose, _MODEL_3D, _MODEL_LM_IDS
from scoring import NIHSSScorer
from fusion import LVOFusion

W, H, OCW_PX = 1280, 720, 300.0
RNG = np.random.default_rng(0)


def make_face(roll=0.0, mL=(0, 0, 0), mR=(0, 0, 0), brow=(0.0, 0.0), ap=(0.09, 0.09),
              gaze=0.0, noise=0.002, center=(640, 360)):
    pts = {}
    pts[33], pts[263] = (-0.5, 0.0), (0.5, 0.0)
    pts[133], pts[362] = (-0.17, 0.0), (0.17, 0.0)
    pts[1], pts[152] = (0.0, 0.378), (0.0, 1.10)
    pts[61] = (-0.333 - mL[1], 0.71 - mL[0] + mL[2])
    pts[291] = (0.333 + mR[1], 0.71 - mR[0] + mR[2])
    for i, u in zip((70, 63, 105, 66, 107), (-0.45, -0.40, -0.335, -0.27, -0.22)):
        pts[i] = (u, -0.17 - brow[0])
    for i, u in zip((300, 293, 334, 296, 336), (0.22, 0.27, 0.335, 0.40, 0.45)):
        pts[i] = (u, -0.17 - brow[1])
    for tops, bots, uc, a in (((159, 158, 157, 160), (145, 144, 163, 153), -0.335, ap[0]),
                              ((386, 385, 384, 387), (374, 373, 390, 388), 0.335, ap[1])):
        for k, i in enumerate(tops):
            pts[i] = (uc + (k - 1.5) * 0.04, -a / 2)
        for k, i in enumerate(bots):
            pts[i] = (uc + (k - 1.5) * 0.04, a / 2)
    pts[10] = (0.0, -0.6)
    for ids, uc in (((469, 470, 471, 472), -0.335), ((474, 475, 476, 477), 0.335)):
        c = uc + gaze * 0.33
        for i, (du, dv) in zip(ids, ((0, -0.03), (0.03, 0), (0, 0.03), (-0.03, 0))):
            pts[i] = (c + du, dv)

    th = math.radians(roll)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    lms = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(478)]
    for i, (u, v) in pts.items():
        p = R @ (np.array([u, v]) + RNG.normal(0, noise, 2)) * OCW_PX + np.array(center)
        lms[i] = SimpleNamespace(x=p[0] / W, y=p[1] / H, z=0.0)
    return lms


def run(analyzer, params, seconds, t, **kw):
    out = None
    for _ in range(int(seconds * 30) + 2):
        t += 1 / 30
        out = analyzer.update(make_face(**params), W, H, t=t, **kw)
    return out, t


def run_exam(rest, smile, brows, eyes):
    a = FacialAnalyzer()
    a.start_exam(t=0.0)
    t = 0.0
    for stage, p in zip(("REST", "SMILE", "BROWS", "EYES"), (rest, smile, brows, eyes)):
        out, t = run(a, p, DURATION[stage], t)
    assert out["stage"] == "DONE"
    return out


def _rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = map(float, (math.cos(rx), math.sin(rx), math.cos(ry),
                                         math.sin(ry), math.cos(rz), math.sin(rz)))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _project(yaw, pitch, roll, dist=0.55):
    R = _rot(-math.radians(pitch), -math.radians(yaw), math.radians(roll))
    X = (R @ _MODEL_3D.T).T + np.array([0.0, 0.0, dist])
    lms = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(478)]
    for i, p in zip(_MODEL_LM_IDS, X):
        lms[i] = SimpleNamespace(x=(W * p[0] / p[2] + W / 2) / W, y=(W * p[1] / p[2] + H / 2) / H, z=0)
    return lms, R


@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (25, 0, 0), (-25, 0, 0), (0, 15, 0),
                                            (0, -15, 0), (0, 0, 20), (-15, 10, -10)])
def test_head_pose_recovers_known_angles(yaw, pitch, roll):
    lms, _ = _project(yaw, pitch, roll)
    p, y, r = estimate_head_pose(lms, W, H)
    assert (y, p, r) == pytest.approx((yaw, pitch, roll), abs=3.0)


def test_positive_yaw_means_nose_toward_image_right():
    lms, _ = _project(25, 0, 0)
    nose_x = lms[1].x
    mid_eyes_x = (lms[33].x + lms[263].x) / 2
    assert nose_x > mid_eyes_x


def test_head_roll_does_not_create_droop():
    a = FacialAnalyzer()
    out, _ = run(a, dict(roll=20), 2.5, 0.0)
    assert abs(out["droop"]) < 0.01 and out["lower_idx"] < 0.1
    f = make_face(roll=20, noise=0)
    naive = abs(f[61].y - f[291].y) * H / OCW_PX
    assert naive > 0.2


def test_resting_droop_detected_without_baseline_and_side_is_correct():
    a = FacialAnalyzer()
    out, _ = run(a, dict(mR=(0, 0, 0.09)), 2.5, 0.0)
    assert out["lower_idx"] > 0.6
    assert out["lower_side"] == "patient_left"


def test_normal_face_scores_normal():
    out = run_exam({}, dict(mL=(0.06, 0.10, 0), mR=(0.06, 0.10, 0)),
                  dict(brow=(0.06, 0.06)), dict(ap=(0.01, 0.01)))
    assert out["pattern"] == "NORMAL" and out["nihss4"] == 0


def test_central_pattern_upper_face_sparing():
    droop = dict(mR=(0, 0, 0.07))
    out = run_exam(droop,
                   dict(mL=(0.06, 0.10, 0), mR=(0.01, 0.02, 0.07)),
                   dict(brow=(0.06, 0.06), mR=(0, 0, 0.07)),
                   dict(ap=(0.01, 0.01), mR=(0, 0, 0.07)))
    assert out["upper_assessed"] and out["upper_idx"] < 0.5
    assert out["pattern"] == "CENTRAL"
    assert out["weak_side"] == "patient_left"
    assert out["nihss4"] >= 1


def test_peripheral_pattern_bells_palsy_like():
    droop = dict(mR=(0, 0, 0.07))
    out = run_exam(droop,
                   dict(mL=(0.06, 0.10, 0), mR=(0.01, 0.02, 0.07)),
                   dict(brow=(0.06, 0.0), mR=(0, 0, 0.07)),
                   dict(ap=(0.01, 0.05), mR=(0, 0, 0.07)))
    assert out["pattern"] == "PERIPHERAL"
    assert out["upper_side"] == out["lower_side"] == "patient_left"
    assert out["nihss4"] == 3


def test_lower_only_until_upper_tasks_done():
    a = FacialAnalyzer()
    out, _ = run(a, dict(mR=(0, 0, 0.07)), 2.5, 0.0)
    assert out["pattern"] == "LOWER_ONLY" and not out["upper_assessed"]


def test_bad_head_pose_frames_are_rejected():
    a = FacialAnalyzer()
    out = a.update(make_face(), W, H, yaw=35.0, pitch=0.0, t=0.0)
    assert not out["valid"] and "turned" in out["quality_reason"]
    assert len(a._roll_buf) == 0


def test_gaze_sign_and_magnitude_and_no_baseline_hiding():
    ex, sc = FeatureExtractor(), NIHSSScorer()
    for _ in range(80):
        f = ex.extract(make_face(gaze=0.10), W, H)
        s = sc.update(f)
    assert f["gaze_signed_deg"] == pytest.approx(math.degrees(math.asin(0.10 / 0.42)), abs=1.0)
    assert s["gaze_direction"] == "patient_left"
    assert s["gaze"] >= 1
    ex2, sc2 = FeatureExtractor(), NIHSSScorer()
    for _ in range(80):
        s2 = sc2.update(ex2.extract(make_face(gaze=-0.10), W, H))
    assert s2["gaze_direction"] == "patient_right" and s2["gaze_signed_deg"] < 0


def test_gaze_invariant_to_head_roll():
    ex = FeatureExtractor()
    a = ex.extract(make_face(gaze=0.08, roll=0, noise=0), W, H)["gaze_signed_deg"]
    b = ex.extract(make_face(gaze=0.08, roll=25, noise=0), W, H)["gaze_signed_deg"]
    assert a == pytest.approx(b, abs=0.5)


def _scores(gaze=0.0, yaw=0.0, conf=0.9, **fac):
    base = {"ready": True, "lower_idx": 0.0, "pattern": "NORMAL", "weak_side": None,
            "stage": "DONE", "exam_done": True, "upper_assessed": True}
    base.update(fac)
    return {"gaze_signed_deg": gaze, "yaw": yaw, "confidence": conf, "facial": base}


def feed(fu, n=40, **kw):
    out = None
    for _ in range(n):
        out = fu.update(_scores(**kw))
    return out


def test_fusion_high_when_race_at_least_5():
    fu = LVOFusion()
    fu.set_manual(arm=2, leg=1, speech=0)
    out = feed(fu, gaze=-18.0, lower_idx=0.8, pattern="CENTRAL", weak_side="patient_left")
    assert out["race_face"] == 2 and out["race_gaze_head"] == 1
    assert out["race_total"] == 6 and out["level"] == "HIGH" and out["complete"]
    assert out["lateralisation"] == "consistent"


def test_fusion_peripheral_pattern_is_discounted():
    fu = LVOFusion()
    out = feed(fu, lower_idx=0.8, pattern="PERIPHERAL", weak_side="patient_left")
    assert out["race_face"] == 1 and any("Peripheral" in f for f in out["flags"])


def test_fusion_flags_same_side_face_and_gaze():
    fu = LVOFusion()
    out = feed(fu, gaze=-18.0, lower_idx=0.8, pattern="CENTRAL", weak_side="patient_right")
    assert out["lateralisation"] == "inconsistent"


def test_fusion_provisional_without_manual_items_and_low_when_normal():
    fu = LVOFusion()
    out = feed(fu)
    assert out["level"] == "LOW" and out["provisional"] and out["race_total"] == 0
    assert any("Enter arm" in f for f in out["flags"])


def test_fusion_ignores_low_quality_frames():
    fu = LVOFusion()
    out = feed(fu, gaze=-25.0, conf=0.2)
    assert out["race_gaze_head"] is None


def test_fusion_gaze_alone_is_elevated_cortical_sign():
    fu = LVOFusion()
    out = feed(fu, gaze=20.0)
    assert out["level"] == "ELEVATED"


def test_fusion_manual_validation():
    with pytest.raises(ValueError):
        LVOFusion().set_manual(arm=3)
