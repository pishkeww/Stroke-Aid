import os

import cv2
import numpy as np
import pytest

import capture
import face_dynamics as fd
import facial_analysis as fa
import selftest
from capture import FrameSource, parse_source, transform
from facial_analysis import FacialAnalyzer
from test_triage import make_face, W, H

FPS = 30.0


def feed(an, ap_fn, seconds, t0=0.0, face_fn=None):
    out = None
    n = int(seconds * FPS)
    for i in range(n):
        t = t0 + i / FPS
        kw = dict(face_fn(t) if face_fn else {})
        kw["ap"] = ap_fn(t)
        out = an.update(make_face(**kw), W, H, t=t)
    return out


def blink_ap(period=2.0, frames=5, lagR=0):
    def f(t):
        k = int(round(t * FPS)) % int(period * FPS)
        L = 0.01 if 0 <= k < frames else 0.09
        R = 0.01 if lagR <= k < frames + lagR else 0.09
        return (L, R)
    return f


def test_blink_rate_and_synchrony():
    an = FacialAnalyzer()
    out = feed(an, blink_ap(), 30.0)
    assert 24 <= out["blink_rate_per_min"] <= 36
    assert out["blink_lag_ms"] < 40
    assert out["ptosis_asym"] < 0.08 and not out["ptosis_flag"]


def test_blink_lag_measured():
    an = FacialAnalyzer()
    out = feed(an, blink_ap(lagR=3), 30.0)
    assert 60 <= out["blink_lag_ms"] <= 140


def test_ptosis_detected_and_sided():
    an = FacialAnalyzer()
    out = feed(an, lambda t: (0.05, 0.09), 12.0)
    assert out["ptosis_flag"] and out["ptosis_asym"] > 0.3
    assert out["ptosis_side"] == fa.patient_side("L")
    out2 = feed(FacialAnalyzer(), lambda t: (0.09, 0.05), 12.0)
    assert out2["ptosis_side"] == fa.patient_side("R")


def test_long_eye_closure_is_not_a_blink():
    an = FacialAnalyzer()
    def ap(t):
        return (0.01, 0.01) if 5.0 <= t < 6.5 else (0.09, 0.09)
    out = feed(an, ap, 20.0)
    assert out["blink_count"] == 0


def test_smile_onset_lag():
    an = FacialAnalyzer()
    t = 0.0
    an.start_exam(t=t)
    for i in range(int(3.0 * FPS) + 2):
        t = i / FPS
        an.update(make_face(), W, H, t=t)
    t0 = t
    out = None
    for i in range(int(5.0 * FPS) + 2):
        tt = t0 + (i + 1) / FPS
        el = tt - t0
        ramp = lambda start: float(np.clip((el - start) / 0.5, 0, 1)) * 0.10
        out = an.update(make_face(mL=(ramp(1.0), ramp(1.0) * 0.5, 0),
                                  mR=(ramp(1.8), ramp(1.8) * 0.5, 0)), W, H, t=tt)
    out = an.update(make_face(), W, H, t=tt + 0.1)
    assert out["smile_onset_lag_s"] is not None
    assert 0.6 <= out["smile_onset_lag_s"] <= 1.0
    assert out["smile_late_side"] == fa.patient_side("R")


def healthy_frames(seed_noise=0.002, seconds=22.0):
    for i in range(int(seconds * FPS)):
        t = i / FPS
        yield t, make_face(noise=seed_noise), W, H


def test_selftest_run_and_report(tmp_path):
    for sub in ("H1", "H2"):
        for rep in (1, 2):
            run, rows = selftest.run_rep(healthy_frames(), sub, "normal", rep)
            assert run["pattern"] in ("NORMAL", "INSUFFICIENT")
            assert float(run["lower_idx"]) < fa.LOWER_MILD
            selftest._append(str(tmp_path / "selftest_runs.csv"), selftest.RUN_COLUMNS, [run])
            selftest._append(str(tmp_path / "selftest_frames.csv"), selftest.FRAME_COLUMNS, rows)
    text = selftest.report(str(tmp_path))
    assert "False-positive rate" in text and "Test-retest" in text and "Resting droop vs head yaw" in text
    assert "lower_idx >= 0.25" in text


def test_selftest_flags_a_threshold_that_is_too_tight(tmp_path, monkeypatch):
    run, rows = selftest.run_rep(healthy_frames(noise := 0.02), "H3", "noisy", 1)
    selftest._append(str(tmp_path / "selftest_runs.csv"), selftest.RUN_COLUMNS, [run])
    assert "headroom" in selftest.report(str(tmp_path))


@pytest.mark.parametrize("text,kind", [(0, "device"), ("2", "device"), ("usb:1", "device"),
                                        ("http://192.168.42.129:8080/video", "url"),
                                        ("rtsp://10.0.0.5/live", "url")])
def test_parse_source_kinds(text, kind):
    assert parse_source(text).kind == kind


def test_parse_source_errors_and_file(tmp_path):
    with pytest.raises(ValueError):
        parse_source("no_such_file.mp4")
    with pytest.raises(ValueError):
        parse_source(0, rotate=45)
    p = tmp_path / "v.avi"
    p.write_bytes(b"x")
    s = parse_source(str(p), rotate=90, flip=True)
    assert s.kind == "file" and not s.live and s.rotate == 90 and s.flip


def test_transform_rotate_and_mirror():
    fr = np.zeros((10, 20, 3), np.uint8)
    fr[0, 0] = (255, 0, 0)
    assert transform(fr, 90).shape[:2] == (20, 10)
    assert transform(fr, 180).shape[:2] == (10, 20)
    m = transform(fr, 0, True)
    assert m[0, -1, 0] == 255 and m[0, 0, 0] == 0


def test_file_source_plays_all_frames_with_rotation(tmp_path):
    p = str(tmp_path / "clip.avi")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 32))
    for i in range(12):
        vw.write(np.full((32, 64, 3), i * 10, np.uint8))
    vw.release()
    src = FrameSource(parse_source(p, rotate=90))
    n = 0
    while True:
        ok, fr = src.read()
        if not ok:
            break
        assert fr.shape[:2] == (64, 32)
        n += 1
    src.release()
    assert n == 12


def test_unopenable_source_raises(monkeypatch):
    monkeypatch.setattr(capture, "open_capture", lambda *a, **k: None)
    with pytest.raises(IOError):
        FrameSource("http://127.0.0.1:9/video", wait=0.2)
