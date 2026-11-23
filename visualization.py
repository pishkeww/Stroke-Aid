import cv2
import numpy as np
import math


C = {
    "bg": ( 0,  0, 0),
    "panel": (20,  22, 32),
    "accent":  (160, 200, 0),
    "teal": (180, 200,   0),
    "warn": ( 0, 165, 255),
    "danger": ( 50,  50, 220),
    "green": ( 60, 200,  80),
    "yellow": ( 30, 200, 220),
    "red": ( 60,  80, 220),
    "white":(255, 255, 255),
    "subtext": (120, 120, 140),
    "event":( 0, 100, 255),
    "kalman":(255, 180, 0),
}

SCORE_COLOR = {0: C["green"], 1: C["yellow"], 2: C["warn"], 3: C["danger"]}
MAX_SCORES  = {"1b": 2, "2": 2, "3": 2, "4": 3}


def _alpha_rect(img, x, y, w, h, color, alpha=0.75):
    roi = img[y:y+h, x:x+w]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
    img[y:y+h, x:x+w] = roi


def _text(img, s, pt, scale=0.40, color=C["white"], thickness=1):
    cv2.putText(img, s, pt, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def _progress_bar(img, x, y, total_w, value, max_val, color):
    cv2.rectangle(img, (x, y), (x + total_w, y + 6), (45, 50, 65), -1)
    fill = int(total_w * min(value / max(max_val, 1e-9), 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + 6), color, -1)


PANEL_W = 360
PAD     = 12


def draw_score_panel(frame, scores, fps, paused):
    h, w = frame.shape[:2]
    px = w - PANEL_W - PAD
    _alpha_rect(frame, px, PAD, PANEL_W, h - 2*PAD, C["panel"], alpha=0.88)

    x0 = px + 14
    y  = PAD + 18

    _text(frame, "Facial Tracker", (x0, y), 0.50, C["accent"], 1)
    y += 16
    cv2.line(frame, (x0, y), (x0 + PANEL_W - 28, y), C["accent"], 1)
    y += 14

    if scores.get("calibrating"):
        pct = scores["calib_pct"]
        _text(frame, f"Calibrating gaze baseline… {pct}%", (x0, y), 0.38, C["warn"])
        bw = PANEL_W - 40
        _progress_bar(frame, x0, y + 8, bw, pct, 100, C["warn"])
        y += 28

    items = [
        ("1b", "Eye Opening",   scores["eye_opening"],  scores["eye_opening_label"]),
        ("2",  "Best Gaze",     scores["gaze"],         scores["gaze_label"]),
        ("3",  "Visual Fields", scores["visual_field"], scores["visual_field_label"]),
        ("4",  "Facial Palsy",  scores["facial_palsy"], scores["facial_palsy_label"]),
    ]
    bar_w = PANEL_W - 70

    for item_id, name, score, label in items:
        mx  = MAX_SCORES[item_id]
        sc  = SCORE_COLOR.get(score, C["white"])

        _text(frame, f"Item {item_id}: {name}", (x0, y), 0.42, C["subtext"])

        bx = x0 + PANEL_W - 46
        cv2.circle(frame, (bx, y - 5), 13, sc, -1)
        _text(frame, str(score), (bx - 5, y), 0.48, (20, 20, 20), 1)

        y += 4
        evt_active = (
            (item_id == "2"  and scores.get("evt_gaze_active"))  or
            (item_id == "4"  and scores.get("evt_asym_active"))  or
            (item_id == "1b" and scores.get("evt_ptosis_active"))
        )
        if evt_active:
            cv2.circle(frame, (x0 + bar_w + 6, y + 3), 5, C["event"], -1)

        _progress_bar(frame, x0, y + 2, bar_w, score, mx, sc)
        y += 14
        _text(frame, label, (x0, y), 0.37, sc)
        y += 20


    cv2.line(frame, (x0, y), (x0 + PANEL_W - 28, y), C["subtext"], 1)
    y += 14
    total = scores["partial_nihss"]
    tc = C["green"] if total <= 1 else C["yellow"] if total <= 3 else C["danger"]
    _text(frame, f"Partial NIHSS: {total} / 9", (x0, y), 0.50, tc, 1)
    y += 20

    _text(frame, "Head Pose:", (x0, y), 0.37, C["subtext"])
    y += 14
    _text(frame,
          f"Yaw {scores['yaw']:+.1f}°  Pitch {scores['pitch']:+.1f}°  Roll {scores['roll']:+.1f}°",
          (x0, y), 0.36, C["subtext"])
    y += 18

    _text(frame, " Raw Metrics ", (x0, y), 0.36, C["subtext"])
    y += 14
    metrics = [
        f"Gaze corr.   : {scores['gaze_deg']:.2f}°  [{scores['gaze_direction']}]",
        f"EAR L / R    : {scores['ear_left']:.3f} / {scores['ear_right']:.3f}",
        f"Blink asym.  : {scores['blink_asym']:.3f}",
        f"Face asym.   : {scores['face_asym']:.4f}",
        f"Mouth droop  : {scores['mouth_droop']:.4f}",
        f"Brow asym.   : {scores['brow_asym']:.4f}",
        f"NLF asym.    : {scores['nlf_asym']:.4f}",
        f"IOD          : {scores['iod']:.1f} px",
    ]
    for m in metrics:
        _text(frame, m, (x0, y), 0.34, C["subtext"])
        y += 13

    y += 4
    conf = scores.get("confidence", 1.0)
    conf_col = C["green"] if conf > 0.7 else C["yellow"] if conf > 0.4 else C["danger"]
    _text(frame, f"Confidence: {int(conf*100)}%", (x0, y), 0.38, conf_col)
    y += 12
    _progress_bar(frame, x0, y, PANEL_W - 40, conf, 1.0, conf_col)
    y += 16

    fac, tri = scores.get("facial"), scores.get("triage")
    if fac:
        y += 4
        _text(frame, f"Face: {fac['pattern']}  low {fac['lower_idx']:.2f}  up {fac['upper_idx']:.2f}"
                     f"{'' if fac['upper_assessed'] else '*'}", (x0, y), 0.36, C["subtext"])
        y += 13
    if tri:
        lvl_col = {"HIGH": C["danger"], "ELEVATED": C["warn"]}.get(tri["level"], C["green"])
        _text(frame, f"RACE {tri['race_total']}/9{'*' if tri['provisional'] else ''}  "
                     f"LVO: {tri['level']}", (x0, y), 0.46, lvl_col)
        y += 16
    rt = scores.get("routing")
    if rt:
        el = rt["windows"]["elapsed_min"]
        _text(frame, f"LKW +{int(el // 60)}h{int(el % 60):02d}m" if el is not None else "LKW NOT SET",
              (x0, y), 0.38, C["subtext"] if el is not None else C["warn"])
        y += 13
        p = rt["primary"]
        _text(frame, (f"-> {p['name']} ({p['eta_min']}m)" if p else "-> NO CT DESTINATION")[:46],
              (x0, y), 0.36, C["accent"])
        y += 13

    if scores.get("new_events"):
        for ev in scores["new_events"][-3:]:
            _text(frame, ev[:46], (x0, y), 0.34, C["event"])
            y += 12

    y = h - PAD - 28
    status = "[ PAUSED ]" if paused else f"FPS: {fps:.1f}"
    _text(frame, status, (x0, y),
          0.40, C["warn"] if paused else C["accent"])
    y += 14
    _text(frame, "Q=Quit R=Reset S=Save P=Pause E=Exam A/L/V=RACE", (x0, y), 0.30, C["subtext"])


def draw_exam_banner(frame, fac):
    right = frame.shape[1] - (PANEL_W + PAD)
    if fac["stage"] not in ("IDLE", "DONE"):
        _alpha_rect(frame, 0, 0, max(right, 1), 34, C["panel"], 0.85)
        _text(frame, f"EXAM [{fac['stage']}] {fac['stage_prompt']}  "
                     f"({fac['stage_time_left']:.0f}s)", (12, 22), 0.62, C["accent"], 1)
    elif not fac["valid"] and fac.get("quality_reason"):
        _text(frame, f"Hold still: {fac['quality_reason']}", (12, 22), 0.5, C["warn"])


def draw_face_overlay(frame, features, scores):
    if features is None:
        return

    li  = features["left_iris_px"].astype(int)
    ri  = features["right_iris_px"].astype(int)
    ml  = features["mouth_left_px"].astype(int)
    mr  = features["mouth_right_px"].astype(int)
    lb  = features["left_brow_px"].astype(int)
    rb  = features["right_brow_px"].astype(int)
    iod = features["iod"]

    r = max(3, int(iod * 0.18))
    for cx, col in [(tuple(li), C["kalman"]), (tuple(ri), (0, 180, 255))]:
        cv2.circle(frame, cx, r, col, 1, cv2.LINE_AA)
        cv2.circle(frame, cx, 2, col, -1)

    gaze_col = SCORE_COLOR.get(scores.get("gaze", 0), C["white"])
    cv2.arrowedLine(frame, tuple(ri), tuple(li), gaze_col, 1, tipLength=0.15,
                    line_type=cv2.LINE_AA)

    palsy_col = SCORE_COLOR.get(scores.get("facial_palsy", 0), C["white"])
    cv2.line(frame, tuple(ml), tuple(mr), palsy_col, 1, cv2.LINE_AA)
    cv2.circle(frame, tuple(ml), 4, palsy_col, -1)
    cv2.circle(frame, tuple(mr), 4, palsy_col, -1)

    for b in [tuple(lb), tuple(rb)]:
        cv2.circle(frame, b, 3, C["accent"], -1)

    nose = features["nose_tip_px"].astype(int)
    fore = features["forehead_px"].astype(int)
    cv2.line(frame, tuple(fore), tuple(nose), (60, 60, 80), 1, cv2.LINE_AA)

    fac = scores.get("facial")
    if fac:
        draw_exam_banner(frame, fac)
        side = fac.get("lower_side")
        if side:
            img_left = (side == "patient_right")
            pt = tuple(features["mouth_left_px" if img_left else "mouth_right_px"].astype(int))
            cv2.circle(frame, pt, 11, C["danger"], 2, cv2.LINE_AA)
            _text(frame, "weak", (pt[0] - 14, pt[1] + 26), 0.38, C["danger"])

    if scores.get("evt_gaze_active") or scores.get("evt_asym_active"):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), C["danger"], 3)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def draw_no_face(frame):
    h, w = frame.shape[:2]
    _text(frame, "No face detected // please centre your face",
          (30, h // 2), 0.65, C["warn"], 1)
