import sys
import time
import threading
import csv
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox, QTextEdit,
    QProgressBar, QComboBox, QSlider, QSplitter, QFileDialog,
    QStatusBar, QFrame, QSizePolicy, QScrollArea, QLineEdit, QCheckBox, QInputDialog,
)
from PyQt5.QtCore    import Qt, QTimer, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui     import QImage, QPixmap, QColor, QPainter, QPen, QFont, QFontDatabase

import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))
from feature_extractor import FeatureExtractor
from scoring           import NIHSSScorer
from fusion            import LVOFusion
from routing           import TriageRouter, load_destinations, fmt_minutes
from visualization     import draw_face_overlay, draw_no_face, draw_exam_banner
import version
from capture import FrameSource, SourceSpec, parse_source, list_cameras


QSS = """
QMainWindow, QWidget {
    background-color: #0d0f16;
    color: #d0d4e0;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #2a2e40;
    border-radius: 6px;
    margin-top: 8px;
    padding: 6px;
    font-size: 11px;
    color: #606480;
}QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}QPushButton {
    background-color: #1a1d2a;
    border: 1px solid #3a3e55;
    border-radius: 4px;
    color: #c0c8e0;
    padding: 5px 14px;
    font-size: 11px;
}
QPushButton:hover  { background-color: #252838; border-color: #00c8a0; }
QPushButton:pressed{ background-color: #00c8a0; color: #0d0f16; }
QPushButton#danger { border-color: #dd3333; }
QPushButton#danger:hover { background-color: #dd3333; color: #ffffff; }
QProgressBar {
    border: 1px solid #2a2e40;
    border-radius: 3px;
    background: #1a1d2a;
    height: 10px;
    text-align: center;
    font-size: 9px;
}
QProgressBar::chunk { border-radius: 3px; }
QTextEdit {
    background-color: #0a0c14;
    border: 1px solid #2a2e40;
    color: #80a0c0;
    font-family: 'Courier New', monospace;
    font-size: 11px;
}
QComboBox {
    background-color: #1a1d2a;
    border: 1px solid #3a3e55;
    border-radius: 4px;
    color: #c0c8e0;
    padding: 3px 8px;
}
QLabel#header {
    font-size: 18px;
    font-weight: bold;
    color: #00c8a0;
    letter-spacing: 3px;
}
QLabel#subheader { color: #505870; font-size: 10px; }
QSplitter::handle { background: #1e2130; }
"""

SCORE_COLORS = {
    0: "#3dc85a",
    1: "#dcc830",
    2: "#ff9020",
    3: "#e03030",
}
CONF_COLORS = {
    "high":   "#3dc85a",
    "medium": "#dcc830",
    "low":    "#e03030",
}


class ProcessingWorker(QThread):
    frame_ready  = pyqtSignal(np.ndarray, dict, dict)
    no_face      = pyqtSignal(np.ndarray)
    stream_ended = pyqtSignal()

    def __init__(self, source=0, parent=None, router=None):
        super().__init__(parent)
        self.router   = router
        self._frame_idx = 0
        self.source   = source
        self._running = True
        self._cap     = None
        self._paused  = False
        self._mutex   = QMutex()

        self._extractor = FeatureExtractor()
        self._scorer    = NIHSSScorer()
        self._fusion    = LVOFusion()

        mp_face = mp.solutions.face_mesh
        self._mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )

        self._last_features = None
        self._last_scores   = {}
        self._face_loss     = 0


    def pause(self):
        with QMutexLocker(self._mutex):
            self._paused = True

    def resume(self):
        with QMutexLocker(self._mutex):
            self._paused = False

    def stop(self):
        self._running = False
        if self._cap is not None:
            self._cap.abort()
        self.wait()

    def reset_baseline(self):
        self._scorer.reset_baseline()
        self._extractor.facial.reset_exam()
        self._fusion.reset()
        self._last_features = None
        self._last_scores   = {}

    def start_or_next_exam(self):
        fa = self._extractor.facial
        t = time.monotonic()
        if fa.stage in ("IDLE", "DONE"):
            fa.start_exam(t=t)
        else:
            fa.next_stage(t=t)
        return fa.stage

    def set_manual(self, **kw):
        self._fusion.set_manual(**kw)


    def run(self):
        try:
            cap = FrameSource(self.source)
        except (IOError, ValueError):
            self.stream_ended.emit()
            return
        self._cap = cap

        t_prev = time.time()
        fps    = 0.0

        while self._running:
            with QMutexLocker(self._mutex):
                paused = self._paused

            if paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                if self._running:
                    self.stream_ended.emit()
                break

            t_now = time.time()
            fps   = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            h, w = frame.shape[:2]
            t_mono = time.monotonic()
            self._frame_idx += 1
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = self._mesh.process(rgb)

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                self._face_loss = 0

                features = self._extractor.extract(lms, w, h, t=t_mono)
                scores   = self._scorer.update(features)
                scores["triage"] = self._fusion.update(scores, features)
                if self.router:
                    scores["routing"] = self.router.recommend(scores["triage"])
                scores["_fps"] = fps

                self._last_features = features
                self._last_scores   = scores

                draw_face_overlay(frame, features, scores)
                self.frame_ready.emit(frame, features, scores)

            else:
                self._face_loss += 1
                if self._face_loss <= 30 and self._last_features:
                    draw_face_overlay(frame, self._last_features, self._last_scores)
                    self.frame_ready.emit(frame, self._last_features, self._last_scores)
                else:
                    draw_no_face(frame)
                    self.no_face.emit(frame)

        cap.release()


class ScoreCard(QGroupBox):
    def __init__(self, item_id, name, max_score):
        super().__init__(f"Item {item_id}: {name}")
        self.max_score = max_score
        self._score    = 0

        self._score_lbl = QLabel("0")
        self._score_lbl.setAlignment(Qt.AlignCenter)
        self._score_lbl.setFont(QFont("Courier New", 28, QFont.Bold))

        self._bar = QProgressBar()
        self._bar.setMaximum(max_score)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)

        self._label_lbl = QLabel("—")
        self._label_lbl.setAlignment(Qt.AlignCenter)
        self._label_lbl.setStyleSheet("font-size: 10px; color: #707090;")

        lay = QVBoxLayout()
        lay.addWidget(self._score_lbl)
        lay.addWidget(self._bar)
        lay.addWidget(self._label_lbl)
        self.setLayout(lay)
        self.setFixedHeight(110)

    def update_score(self, score, label):
        self._score = score
        col = SCORE_COLORS.get(score, "#ffffff")
        self._score_lbl.setText(str(score))
        self._score_lbl.setStyleSheet(f"color: {col}; font-size: 28px; font-weight: bold;")
        self._bar.setValue(score)
        self._bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {col}; }}"
        )
        self._label_lbl.setText(label)
        self._label_lbl.setStyleSheet(f"font-size: 10px; color: {col};")


class TrendPlot(QWidget):
    def __init__(self, title, color="#00c8a0", y_max=1.0, threshold=None):
        super().__init__()
        self.title     = title
        self.color     = QColor(color)
        self.y_max     = y_max
        self.threshold = threshold
        self._buf      = deque(maxlen=300)
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def push(self, v):
        self._buf.append(float(v))
        self.update()

    def paintEvent(self, _):
        if not self._buf:
            return
        p   = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pad  = 28

        p.fillRect(0, 0, w, h, QColor("#0a0c14"))

        if self.threshold is not None:
            ty = int(h - pad - (self.threshold / self.y_max) * (h - 2*pad))
            p.setPen(QPen(QColor("#ff602040"), 1, Qt.DashLine))
            p.drawLine(pad, ty, w - 8, ty)

        buf  = list(self._buf)
        step = max(1, (w - pad) // max(len(buf), 1))
        pen  = QPen(self.color, 1.5)
        p.setPen(pen)
        pts  = []
        for i, v in enumerate(buf):
            x = pad + int(i * (w - pad - 8) / max(len(buf)-1, 1))
            y = int(h - pad - min(v / self.y_max, 1.0) * (h - 2*pad))
            pts.append((x, y))
        for i in range(1, len(pts)):
            p.drawLine(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])

        p.setPen(QPen(QColor("#505870"), 1))
        p.setFont(QFont("Courier New", 8))
        p.drawText(2, h - 4, "0")
        p.drawText(2, pad - 4, f"{self.y_max:.1f}")
        p.drawText(w // 2 - 30, h - 2, self.title)

        if buf:
            p.setPen(QPen(self.color, 1))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(w - 52, 14, f"{buf[-1]:.3f}")


class MainWindow(QMainWindow):

    def __init__(self, router=None, source=0):
        super().__init__()
        self._router = router
        title = "Prehospital Stroke / LVO Triage Aid — decision support only"
        self.setWindowTitle(title + f"  (v{version.VERSION})")
        self.resize(1440, 860)
        self.setStyleSheet(QSS)

        self._worker   = None
        self._paused   = False
        self._source   = 0
        self._log_rows = deque(maxlen=200000)
        self._event_count = 0

        spec = parse_source(source)
        self._spec_value = spec.value
        self._build_ui()
        if spec.kind == "device" and self._src_combo.findData(("device", spec.value)) < 0:
            self._fill_sources([(spec.value, 0, 0), (0, 0, 0)])
        i = self._src_combo.findData((spec.kind, spec.value))
        if i < 0 and spec.kind in ("file", "url"):
            self._src_combo.insertItem(0, spec.label(), (spec.kind, spec.value))
            i = 0
        if i >= 0:
            self._src_combo.setCurrentIndex(i)
        self._rot_combo.setCurrentIndex(max(0, self._rot_combo.findData(spec.rotate)))
        self._mirror_chk.setChecked(spec.flip)
        self._start_worker(source=spec)


    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 6, 8, 6)

        top = QHBoxLayout()
        hdr = QLabel("Face Tracker")
        hdr.setObjectName("header")
        sub = QLabel("v3 · offline · not a diagnostic device")
        sub.setObjectName("subheader")
        top.addWidget(hdr)
        top.addSpacing(18)
        top.addWidget(sub)
        top.addStretch()

        self._src_combo = QComboBox()
        self._fill_sources([(0, 0, 0), (1, 0, 0)])
        self._src_combo.activated.connect(self._on_source_change)
        top.addWidget(QLabel("Source:"))
        top.addWidget(self._src_combo)
        self._rot_combo = QComboBox()
        for d in (0, 90, 180, 270):
            self._rot_combo.addItem(f"Rotate {d}", d)
        self._rot_combo.activated.connect(self._on_transform_change)
        self._mirror_chk = QCheckBox("Mirror")
        self._mirror_chk.toggled.connect(self._on_transform_change)
        top.addWidget(self._rot_combo)
        top.addWidget(self._mirror_chk)
        top.addSpacing(12)

        self._btn_pause = QPushButton("⏸Pause")
        self._btn_reset = QPushButton("↺ Reset Baseline")
        self._btn_save  = QPushButton("Save Frame")
        self._btn_export= QPushButton("Export CSV")

        for btn in [self._btn_pause, self._btn_reset, self._btn_save, self._btn_export]:
            top.addWidget(btn)

        self._btn_pause.clicked.connect(self._toggle_pause)
        self._btn_reset.clicked.connect(self._reset_baseline)
        self._btn_save.clicked.connect(self._save_frame)
        self._btn_export.clicked.connect(self._export_csv)

        root.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        video_box = QGroupBox("Live Feed")
        vlay = QVBoxLayout(video_box)
        self._video_lbl = QLabel()
        self._video_lbl.setAlignment(Qt.AlignCenter)
        self._video_lbl.setMinimumSize(640, 400)
        self._video_lbl.setStyleSheet("background:#050608; border-radius:4px;")
        vlay.addWidget(self._video_lbl)
        splitter.addWidget(video_box)

        right = QWidget()
        rlay  = QVBoxLayout(right)
        rlay.setSpacing(8)

        tri_box = QGroupBox("LVO Triage  (RACE-aligned, 0-9)")
        trlay   = QVBoxLayout(tri_box)
        self._race_lbl = QLabel("RACE 0 / 9")
        self._race_lbl.setFont(QFont("Courier New", 22, QFont.Bold))
        self._race_lbl.setAlignment(Qt.AlignCenter)
        self._level_lbl = QLabel("—")
        self._level_lbl.setAlignment(Qt.AlignCenter)
        self._reco_lbl = QLabel("")
        self._reco_lbl.setWordWrap(True)
        self._reco_lbl.setStyleSheet("font-size: 11px;")
        self._flags_lbl = QLabel("")
        self._flags_lbl.setWordWrap(True)
        self._flags_lbl.setStyleSheet("font-size: 10px; color: #dcc830;")
        trlay.addWidget(self._race_lbl)
        trlay.addWidget(self._level_lbl)
        trlay.addWidget(self._reco_lbl)
        trlay.addWidget(self._flags_lbl)

        manual = QGridLayout()
        self._manual = {}
        for row, (key, title, opts) in enumerate([
            ("arm",    "Arm motor (paramedic)",  ["0 normal", "1 mild drift/weak", "2 severe/no effort"]),
            ("leg",    "Leg motor (paramedic)",  ["0 normal", "1 mild drift/weak", "2 severe/no effort"]),
            ("speech", "Aphasia/agnosia",        ["0 normal", "1 moderate", "2 severe"]),
        ]):
            cb = QComboBox()
            cb.addItem("not entered", None)
            for i, o in enumerate(opts):
                cb.addItem(o, i)
            cb.currentIndexChanged.connect(
                lambda _i, k=key, c=cb: self._on_manual(k, c.currentData()))
            manual.addWidget(QLabel(title), row, 0)
            manual.addWidget(cb, row, 1)
            self._manual[key] = cb
        trlay.addLayout(manual)
        rlay.addWidget(tri_box)

        rt_box = QGroupBox("Last Known Well & Routing")
        rtlay  = QVBoxLayout(rt_box)
        lkw_row = QHBoxLayout()
        self._lkw_edit = QLineEdit()
        self._lkw_edit.setPlaceholderText("LKW  HH:MM  or  minutes ago (e.g. 45)")
        self._btn_lkw = QPushButton("Set LKW")
        self._btn_lkw_unk = QPushButton("Unknown")
        lkw_row.addWidget(self._lkw_edit)
        lkw_row.addWidget(self._btn_lkw)
        lkw_row.addWidget(self._btn_lkw_unk)
        self._btn_lkw.clicked.connect(self._on_set_lkw)
        self._lkw_edit.returnPressed.connect(self._on_set_lkw)
        self._btn_lkw_unk.clicked.connect(self._on_lkw_unknown)
        self._clock_lbl = QLabel("LKW not set")
        self._clock_lbl.setAlignment(Qt.AlignCenter)
        self._clock_lbl.setStyleSheet("font-size: 13px;")
        self._route_lbl = QLabel("No routing (no destinations loaded)" if not self._router else "")
        self._route_lbl.setWordWrap(True)
        self._route_warn = QLabel("")
        self._route_warn.setWordWrap(True)
        self._route_warn.setStyleSheet("font-size: 10px; color: #dcc830;")
        rtlay.addLayout(lkw_row)
        rtlay.addWidget(self._clock_lbl)
        rtlay.addWidget(self._route_lbl)
        rtlay.addWidget(self._route_warn)
        rlay.addWidget(rt_box)

        fac_box = QGroupBox("Facial Exam  (central vs peripheral)")
        flay    = QVBoxLayout(fac_box)
        self._exam_prompt = QLabel("Press 'Start facial exam'")
        self._exam_prompt.setAlignment(Qt.AlignCenter)
        self._exam_prompt.setStyleSheet("font-size: 13px; color: #00c8a0;")
        self._pattern_lbl = QLabel("Pattern: —")
        self._pattern_lbl.setWordWrap(True)
        self._lower_bar = QProgressBar(); self._lower_bar.setMaximum(100); self._lower_bar.setFormat("lower face  %v%")
        self._upper_bar = QProgressBar(); self._upper_bar.setMaximum(100); self._upper_bar.setFormat("upper face  %v%")
        for b in (self._lower_bar, self._upper_bar):
            b.setTextVisible(True)
        self._btn_exam = QPushButton("Start facial exam")
        self._btn_exam.clicked.connect(self._on_exam)
        flay.addWidget(self._exam_prompt)
        flay.addWidget(self._btn_exam)
        flay.addWidget(self._lower_bar)
        flay.addWidget(self._upper_bar)
        flay.addWidget(self._pattern_lbl)
        rlay.addWidget(fac_box)

        score_box = QGroupBox("NIHSS Scores")
        sgrid     = QGridLayout(score_box)
        self._cards = {
            "1b": ScoreCard("1b", "Eye Open (proxy)",   2),
            "2":  ScoreCard("2",  "Best Gaze",     2),
            "3":  ScoreCard("3",  "Vis.Field (proxy)", 2),
            "4":  ScoreCard("4",  "Facial Palsy",  3),
        }
        sgrid.addWidget(self._cards["1b"], 0, 0)
        sgrid.addWidget(self._cards["2"],  0, 1)
        sgrid.addWidget(self._cards["3"],  1, 0)
        sgrid.addWidget(self._cards["4"],  1, 1)
        rlay.addWidget(score_box)

        tot_box = QGroupBox("Partial NIHSS Total  (Items 1b+2+3+4,  max 9)")
        tlay    = QHBoxLayout(tot_box)
        self._total_lbl = QLabel("0 / 9")
        self._total_lbl.setFont(QFont("Courier New", 22, QFont.Bold))
        self._total_lbl.setAlignment(Qt.AlignCenter)
        self._total_bar = QProgressBar()
        self._total_bar.setMaximum(9)
        self._total_bar.setValue(0)
        tlay.addWidget(self._total_lbl)
        tlay.addWidget(self._total_bar)
        rlay.addWidget(tot_box)

        pose_box = QGroupBox("Head Pose (solvePnP)")
        play     = QHBoxLayout(pose_box)
        self._pose_lbl = QLabel("Yaw: 0°   Pitch: 0°   Roll: 0°")
        self._pose_lbl.setAlignment(Qt.AlignCenter)
        play.addWidget(self._pose_lbl)
        rlay.addWidget(pose_box)

        conf_box = QGroupBox("Detection Confidence")
        clay     = QVBoxLayout(conf_box)
        self._conf_bar = QProgressBar()
        self._conf_bar.setMaximum(100)
        self._conf_bar.setValue(100)
        self._conf_bar.setFormat("%v%")
        self._conf_bar.setTextVisible(True)
        clay.addWidget(self._conf_bar)
        rlay.addWidget(conf_box)

        self._fps_lbl = QLabel("FPS: —")
        self._fps_lbl.setAlignment(Qt.AlignRight)
        self._fps_lbl.setStyleSheet("color: #505870; font-size: 10px;")
        rlay.addWidget(self._fps_lbl)

        rlay.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setSizes([820, 600])

        root.addWidget(splitter, stretch=5)

        plot_box = QGroupBox("Trend Plots (last 300 frames)")
        play2    = QHBoxLayout(plot_box)
        self._plot_gaze  = TrendPlot("Gaze Deviation (°)",   "#00c8a0", y_max=30.0, threshold=5.0)
        self._plot_asym  = TrendPlot("Lower-face asymmetry idx", "#ff9020", y_max=1.0, threshold=0.25)
        self._plot_ear   = TrendPlot("Avg EAR",              "#60a0ff", y_max=0.40, threshold=0.15)
        for pl in [self._plot_gaze, self._plot_asym, self._plot_ear]:
            play2.addWidget(pl)
        root.addWidget(plot_box, stretch=2)

        evt_box = QGroupBox("Event Log")
        elay    = QVBoxLayout(evt_box)
        self._event_log = QTextEdit()
        self._event_log.setReadOnly(True)
        self._event_log.setMaximumHeight(90)
        elay.addWidget(self._event_log)
        root.addWidget(evt_box, stretch=1)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status = QLabel("Initialising…")
        sb.addWidget(self._status)

        self._last_frame_bgr = None


    def _start_worker(self, source=0):
        if self._worker and self._worker.isRunning():
            self._worker.stop()

        self._paused = False
        self._btn_pause.setText("⏸  Pause")
        self._worker = ProcessingWorker(source=source, parent=self, router=self._router)
        for key, cb in self._manual.items():
            self._worker.set_manual(**{key: cb.currentData()})
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.no_face.connect(self._on_no_face)
        self._worker.stream_ended.connect(self._on_stream_ended)
        self._worker.start()
        self._status.setText(f"Streaming from {parse_source(source).label()}")


    def _on_frame(self, frame, features, scores):
        self._last_frame_bgr = frame.copy()

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi   = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qi).scaled(
            self._video_lbl.width(), self._video_lbl.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._video_lbl.setPixmap(pix)

        self._cards["1b"].update_score(scores["eye_opening"], scores["eye_opening_label"])
        self._cards["2"].update_score (scores["gaze"],        scores["gaze_label"])
        self._cards["3"].update_score (scores["visual_field"],scores["visual_field_label"])
        self._cards["4"].update_score (scores["facial_palsy"],scores["facial_palsy_label"])

        total = scores["partial_nihss"]
        tcol  = "#3dc85a" if total <= 1 else "#dcc830" if total <= 3 else "#e03030"
        self._total_lbl.setText(f"{total} / 9")
        self._total_lbl.setStyleSheet(f"color:{tcol}; font-size:22px; font-weight:bold;")
        self._total_bar.setValue(total)
        self._total_bar.setStyleSheet(f"QProgressBar::chunk{{background:{tcol};}}")

        self._pose_lbl.setText(
            f"Yaw: {scores['yaw']:+.1f}°    Pitch: {scores['pitch']:+.1f}°    Roll: {scores['roll']:+.1f}°"
        )

        conf = int(scores.get("confidence", 1.0) * 100)
        ccol = "#3dc85a" if conf > 70 else "#dcc830" if conf > 40 else "#e03030"
        self._conf_bar.setValue(conf)
        self._conf_bar.setStyleSheet(f"QProgressBar::chunk{{background:{ccol};}}")

        self._fps_lbl.setText(f"FPS: {scores.get('_fps', 0):.1f}")

        self._plot_gaze.push(scores["gaze_deg"])
        self._plot_asym.push(scores["face_asym"])
        self._plot_ear.push(scores["avg_ear"])

        fac = scores["facial"]
        self._exam_prompt.setText(
            fac["stage_prompt"] if fac["stage"] in ("IDLE", "DONE")
            else f"[{fac['stage']}] {fac['stage_prompt']}  ({fac['stage_time_left']:.0f}s)")
        self._btn_exam.setText("Start facial exam" if fac["stage"] in ("IDLE", "DONE")
                               else "Skip to next step")
        self._lower_bar.setValue(int(fac["lower_idx"] * 100))
        self._upper_bar.setValue(int(fac["upper_idx"] * 100))
        pend = "" if fac["upper_assessed"] else "  (upper face provisional)"
        weak = f"  weak: {fac['weak_side']}" if fac["weak_side"] else ""
        self._pattern_lbl.setText(f"Pattern: {fac['pattern']}{weak}\n{fac['pattern_note']}{pend}")

        tri = scores["triage"]
        lcol = {"HIGH": "#e03030", "ELEVATED": "#ff9020", "LOW": "#3dc85a"}.get(tri["level"], "#d0d4e0")
        star = "*" if tri["provisional"] else ""
        self._race_lbl.setText(f"RACE {tri['race_total']} / 9{star}")
        self._race_lbl.setStyleSheet(f"color:{lcol}; font-size:22px; font-weight:bold;")
        self._level_lbl.setText(f"LVO suspicion: {tri['level']}")
        self._level_lbl.setStyleSheet(f"color:{lcol}; font-size:13px; font-weight:bold;")
        self._reco_lbl.setText(tri["recommendation"])
        self._flags_lbl.setText("\n".join("• " + f for f in tri["flags"][-4:]))

        rt = scores.get("routing")
        if rt:
            w_ = rt["windows"]
            if w_["lkw_known"]:
                ivt = w_["ivt_remaining_min"]
                self._clock_lbl.setText(
                    f"Since LKW {fmt_minutes(w_['elapsed_min'])}   |   thrombolysis window left "
                    f"{fmt_minutes(ivt) if ivt > 0 else 'CLOSED'}")
                self._clock_lbl.setStyleSheet(
                    "font-size:13px; color:%s;" % ("#3dc85a" if ivt > 90 else "#ff9020" if ivt > 0 else "#e03030"))
            else:
                self._clock_lbl.setText("LKW unknown / wake-up" if w_["lkw_unknown_flag"] else "LKW NOT SET")
                self._clock_lbl.setStyleSheet("font-size:13px; color:#e03030;")
            p, a = rt["primary"], rt["alternate"]
            if p:
                txt = f"→ {rt['mode'].replace('_', ' ')}: {p['name']}  (ETA {p['eta_min']} min)"
                if p["phone"]:
                    txt += f"  ☎ {p['phone']}"
                if a:
                    txt += f"\nalt: {a['name']} (ETA {a['eta_min']} min)"
                txt += "\n" + " ".join(rt["rationale"])
            else:
                txt = "No CT destination available. " + " ".join(rt["rationale"])
            self._route_lbl.setText(txt)
            self._route_warn.setText("\n".join("⚠ " + x for x in rt["warnings"][:4]))

        self._log_rows.append(self._flat_row(scores))

        for ev in scores.get("new_events", []):
            ts = datetime.now().strftime("%H:%M:%S")
            self._event_count += 1
            self._event_log.append(f"[{ts}] {ev}")

        if scores.get("calibrating"):
            self._status.setText(f"Calibrating gaze baseline… {scores['calib_pct']}%")
        else:
            self._status.setText("Tracking active")

    @staticmethod
    def _flat_row(s):
        fac, tri = s["facial"], s["triage"]
        return {
            "timestamp": datetime.now().isoformat(),
            "gaze_signed_deg": round(s["gaze_signed_deg"], 2), "avg_ear": round(s["avg_ear"], 4),
            "yaw": round(s["yaw"], 1), "pitch": round(s["pitch"], 1), "roll": round(s["roll"], 1),
            "confidence": s["confidence"],
            "lower_idx": round(fac["lower_idx"], 3), "upper_idx": round(fac["upper_idx"], 3),
            "upper_assessed": fac["upper_assessed"], "facial_pattern": fac["pattern"],
            "weak_side": fac["weak_side"], "exam_stage": fac["stage"],
            "race_face": tri["race_face"], "race_gaze_head": tri["race_gaze_head"],
            "race_arm": tri["race_arm"], "race_leg": tri["race_leg"],
            "race_speech": tri["race_speech"], "race_total": tri["race_total"],
            "lvo_level": tri["level"],
            "lkw_elapsed_min": (None if not s.get("routing") or s["routing"]["windows"]["elapsed_min"] is None
                                else round(s["routing"]["windows"]["elapsed_min"])),
            "route_mode": s.get("routing", {}).get("mode"),
            "route_primary": (s.get("routing") or {}).get("primary", {}) and s["routing"]["primary"]["name"],
        }

    def _on_set_lkw(self):
        if not self._router:
            return
        txt = self._lkw_edit.text().strip()
        try:
            if ":" in txt:
                self._router.clock.set_clock_time(txt)
            else:
                self._router.clock.set_minutes_ago(float(txt))
            self._status.setText("LKW set.")
        except ValueError as e:
            self._status.setText(f"LKW not set: {e}")

    def _on_lkw_unknown(self):
        if self._router:
            self._router.clock.set_unknown()
            self._lkw_edit.clear()

    def _on_manual(self, key, value):
        if self._worker:
            self._worker.set_manual(**{key: value})

    def _on_exam(self):
        if self._worker:
            self._worker.start_or_next_exam()

    def _on_no_face(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi  = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qi).scaled(
            self._video_lbl.width(), self._video_lbl.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._video_lbl.setPixmap(pix)
        self._status.setText("⚠ No face detected")

    def _on_stream_ended(self):
        self._status.setText("Stream ended.")

    def _fill_sources(self, cams):
        self._src_combo.clear()
        for i, w, h in cams:
            self._src_combo.addItem(f"Camera {i}" + (f" ({w}x{h})" if w else ""), ("device", i))
        self._src_combo.addItem("Phone / network stream URL…", ("url", None))
        self._src_combo.addItem("Video file…", ("file", None))
        self._src_combo.addItem("Detect cameras…", ("detect", None))

    def _current_spec(self):
        return parse_source(self._spec_value, int(self._rot_combo.currentData()),
                            self._mirror_chk.isChecked())

    def _on_transform_change(self, *a):
        if self._spec_value is not None:
            self._start_worker(source=self._current_spec())

    def _on_source_change(self, idx):
        kind, val = self._src_combo.itemData(idx)
        if kind == "detect":
            self._status.setText("Scanning cameras...")
            QApplication.processEvents()
            if self._worker and self._worker.isRunning():
                self._worker.stop()
            self._fill_sources(list_cameras())
            self._status.setText("Pick a camera")
            return
        if kind == "url":
            url, ok = QInputDialog.getText(
                self, "Network camera",
                "Stream URL (e.g. http://192.168.42.129:8080/video or rtsp://...):")
            if not ok or not url.strip():
                return
            val = url.strip()
        elif kind == "file":
            val, _ = QFileDialog.getOpenFileName(
                self, "Open video file", "", "Video files (*.mp4 *.avi *.mov *.mkv *.webm)")
            if not val:
                return
        try:
            self._spec_value = val
            self._start_worker(source=self._current_spec())
        except ValueError as e:
            self._status.setText(str(e))

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._worker.pause()
            self._btn_pause.setText("▶  Resume")
        else:
            self._worker.resume()
            self._btn_pause.setText("⏸  Pause")

    def _reset_baseline(self):
        if self._worker:
            self._worker.reset_baseline()
        self._status.setText("Baseline reset — look straight ahead.")

    def _save_frame(self):
        if self._last_frame_bgr is not None:
            fname = f"nihss_frame_{datetime.now().strftime('%H%M%S')}.png"
            cv2.imwrite(fname, self._last_frame_bgr)
            self._status.setText(f"Frame saved: {fname}")

    def _export_csv(self):
        if not self._log_rows:
            self._status.setText("No data to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"nihss_export_{datetime.now().strftime('%H%M%S')}.csv",
            "CSV files (*.csv)"
        )
        if path:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._log_rows[0].keys())
                w.writeheader()
                w.writerows(self._log_rows)
            self._status.setText(f"Exported: {path}")

    def closeEvent(self, ev):
        if self._worker:
            self._worker.stop()
        super().closeEvent(ev)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Prehospital stroke/LVO triage aid (PyQt5)")
    ap.add_argument("--video", type=str,  default=None,
                    help="Path to video file (omit for webcam)")
    ap.add_argument("--cam",   type=int,  default=0,
                    help="Webcam device index (default 0)")
    ap.add_argument("--source", type=str, default=None,
                    help="camera index, usb:N, stream URL (http/rtsp) or video file; overrides --cam/--video")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270),
                    help="rotate frames clockwise (phone held in portrait)")
    ap.add_argument("--mirror", action="store_true",
                    help="flip frames horizontally (mirrored phone/front camera)")
    ap.add_argument("--list-cameras", action="store_true", help="probe attached cameras and exit")
    ap.add_argument("--headless", action="store_true",
                    help="Run without GUI (OpenCV window only, for SSH)")
    ap.add_argument("--destinations", type=str,
                    default=os.path.join(app_dir(), "destinations.json"),
                    help="Destinations JSON (CT / thrombectomy centres)")
    ap.add_argument("--lat", type=float, default=None, help="Ambulance latitude (optional)")
    ap.add_argument("--lon", type=float, default=None, help="Ambulance longitude (optional)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--lkw", type=str, default=None, help="Last known well, HH:MM (24 h)")
    g.add_argument("--lkw-min-ago", type=float, default=None, help="Last known well, minutes ago")
    g.add_argument("--lkw-unknown", action="store_true", help="LKW unknown / wake-up stroke")
    ap.add_argument("--version", action="store_true", help="print version and code fingerprint")
    args = ap.parse_args()

    if args.version:
        print(version.info())
        return

    if args.list_cameras:
        cams = list_cameras()
        for i, w, h in cams:
            print(f"camera {i}: {w}x{h}")
        if not cams:
            print("no cameras found")
        return

    try:
        source = parse_source(args.source if args.source is not None else (args.video or args.cam),
                              args.rotate, args.mirror)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)

    router = None
    try:
        pos = (args.lat, args.lon) if args.lat is not None and args.lon is not None else None
        router = TriageRouter(load_destinations(args.destinations), position=pos)
        if args.lkw:
            router.clock.set_clock_time(args.lkw)
        elif args.lkw_min_ago is not None:
            router.clock.set_minutes_ago(args.lkw_min_ago)
        elif args.lkw_unknown:
            router.clock.set_unknown()
    except (OSError, ValueError, KeyError) as e:
        print(f"[WARN] Routing disabled: {e}")

    if args.headless:
        sys.path.insert(0, os.path.dirname(__file__))
        from tracker import NIHSSTracker
        NIHSSTracker(source=source, router=router).run()
        return

    app = QApplication(sys.argv)
    win = MainWindow(router=router, source=source)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
