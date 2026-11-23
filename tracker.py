import os
import time
import csv
import sys
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from capture           import ThreadedCapture
from feature_extractor import FeatureExtractor
from scoring           import NIHSSScorer
from fusion            import LVOFusion
from visualization     import draw_score_panel, draw_face_overlay, draw_no_face, draw_exam_banner


_FACE_LOSS_MAX_FRAMES = 30


class NIHSSTracker:

    def __init__(self, source=0, router=None):
        self.router  = router
        self.source  = source
        self.paused  = False
        self.fps     = 0.0
        self._frame_count = 0

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
        self._face_loss_cnt = 0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path   = f"nihss_log_{ts}.csv"
        self._csv_file   = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "frame", "confidence",
            "gaze_deg", "gaze_score", "gaze_label",
            "ear_left", "ear_right", "blink_asym", "eye_opening_score",
            "face_asym", "mouth_droop", "brow_asym", "facial_palsy_score",
            "visual_field_score", "partial_nihss",
            "yaw", "pitch", "roll", "iod",
            "gaze_signed_deg", "lower_idx", "upper_idx", "upper_assessed",
            "facial_pattern", "weak_side", "exam_stage",
            "race_face", "race_gaze_head", "race_arm", "race_leg", "race_speech",
            "race_total", "lvo_level",
        ])


    def run(self):
        cap = ThreadedCapture(self.source)
        print(f"\n  NIHSS Tracker started  |  log → {self._csv_path}")
        print("  Controls: Q=Quit  R=Reset  S=Save frame  P=Pause")
        print("            E=Start/next facial exam step")
        print("            A/L/V = cycle RACE arm / leg / speech score (none->0->1->2)\n")

        t_prev   = time.time()
        last_frame = None

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[INFO] Stream ended.")
                    break

                self._frame_count += 1
                t_now     = time.time()
                self.fps  = 0.9 * self.fps + 0.1 / max(t_now - t_prev, 1e-6)
                t_prev    = t_now
                last_frame = frame.copy()

                h, w = frame.shape[:2]
                t_mono = time.monotonic()
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mesh.process(rgb)

                if results.multi_face_landmarks:
                    lms = results.multi_face_landmarks[0].landmark
                    self._face_loss_cnt = 0

                    features = self._extractor.extract(lms, w, h, t=t_mono)
                    scores   = self._scorer.update(features)
                    scores["triage"] = self._fusion.update(scores, features)
                    if self.router:
                        scores["routing"] = self.router.recommend(scores["triage"])

                    self._last_features = features
                    self._last_scores   = scores

                    draw_face_overlay(frame, features, scores)
                    self._log(scores)

                else:
                    self._face_loss_cnt += 1
                    if self._face_loss_cnt <= _FACE_LOSS_MAX_FRAMES:
                        draw_face_overlay(frame, self._last_features, self._last_scores)
                        if self._last_scores:
                            self._last_scores["confidence"] = max(
                                0.0,
                                self._last_scores.get("confidence", 1.0) - 0.05 * self._face_loss_cnt
                            )
                    else:
                        draw_no_face(frame)

                if self._last_scores:
                    draw_score_panel(frame, self._last_scores, self.fps, self.paused)

            else:
                frame = last_frame.copy() if last_frame is not None else np.zeros((720, 1280, 3), np.uint8)
                if self._last_scores:
                    draw_score_panel(frame, self._last_scores, self.fps, self.paused)

            cv2.imshow("NIHSS Facial & Eye Tracker  [Q=Quit]", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                self._scorer.reset_baseline()
                self._extractor.facial.reset_exam()
                self._fusion.reset()
                self._last_features = None
                self._last_scores   = {}
                print("[INFO] Filters, facial exam and gaze/head window reset.")
            elif key == ord('e'):
                fa = self._extractor.facial
                t = time.monotonic()
                if fa.stage in ("IDLE", "DONE"):
                    fa.start_exam(t=t)
                else:
                    fa.next_stage(t=t)
                print(f"[INFO] Facial exam stage: {fa.stage}")
            elif key in (ord('a'), ord('l'), ord('v')):
                name = {ord('a'): "arm", ord('l'): "leg", ord('v'): "speech"}[key]
                cur = self._fusion.manual[name]
                nxt = 0 if cur is None else (cur + 1 if cur < 2 else None)
                self._fusion.set_manual(**{name: nxt})
                print(f"[INFO] RACE {name} = {nxt}")
            elif key == ord('s'):
                fname = f"nihss_frame_{self._frame_count}.png"
                cv2.imwrite(fname, frame)
                print(f"[INFO] Saved: {fname}")
            elif key == ord('p'):
                self.paused = not self.paused
                print(f"[INFO] {'Paused' if self.paused else 'Resumed'}")

        cap.release()
        cv2.destroyAllWindows()
        self._csv_file.close()
        print(f"\n[INFO] Session ended. Log: {self._csv_path}")


    def _log(self, s):
        fac, tri = s["facial"], s["triage"]
        self._csv_writer.writerow([
            datetime.now().isoformat(), self._frame_count,
            s.get("confidence", ""),
            f"{s['gaze_deg']:.3f}", s["gaze"], s["gaze_label"],
            f"{s['ear_left']:.4f}", f"{s['ear_right']:.4f}", f"{s['blink_asym']:.4f}",
            s["eye_opening"],
            f"{s['face_asym']:.4f}", f"{s['mouth_droop']:.4f}", f"{s['brow_asym']:.4f}",
            s["facial_palsy"],
            s["visual_field"], s["partial_nihss"],
            f"{s['yaw']:.2f}", f"{s['pitch']:.2f}", f"{s['roll']:.2f}",
            f"{s['iod']:.1f}",
            f"{s['gaze_signed_deg']:.2f}", f"{fac['lower_idx']:.3f}", f"{fac['upper_idx']:.3f}",
            fac["upper_assessed"], fac["pattern"], fac["weak_side"], fac["stage"],
            tri["race_face"], tri["race_gaze_head"], tri["race_arm"], tri["race_leg"],
            tri["race_speech"], tri["race_total"], tri["level"],
        ])
        self._csv_file.flush()
