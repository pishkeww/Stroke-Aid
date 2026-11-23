import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2

NET_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://")


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    value: object
    rotate: int = 0
    flip: bool = False

    @property
    def live(self):
        return self.kind in ("device", "url")

    def label(self):
        if self.kind == "device":
            s = f"camera {self.value}"
        elif self.kind == "url":
            s = f"stream {self.value}"
        else:
            s = f"file {os.path.basename(str(self.value))}"
        extra = []
        if self.rotate:
            extra.append(f"rot {self.rotate}")
        if self.flip:
            extra.append("mirror")
        return s + (f" ({', '.join(extra)})" if extra else "")


def parse_source(text, rotate=0, flip=False):
    if isinstance(text, SourceSpec):
        return SourceSpec(text.kind, text.value, rotate or text.rotate, flip or text.flip)
    if rotate not in (0, 90, 180, 270):
        raise ValueError("rotate must be 0, 90, 180 or 270")
    if isinstance(text, int):
        return SourceSpec("device", text, rotate, flip)
    s = str(text).strip()
    if s.lower().startswith("usb:"):
        s = s[4:]
    if s.isdigit():
        return SourceSpec("device", int(s), rotate, flip)
    if s.lower().startswith(NET_PREFIXES):
        return SourceSpec("url", s, rotate, flip)
    if os.path.exists(s):
        return SourceSpec("file", s, rotate, flip)
    raise ValueError(f"source not found: {s!r} (use a camera index, a stream URL or a video file)")


def _backends():
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_capture(spec, width=1280, height=720):
    spec = parse_source(spec)
    if spec.kind == "device":
        for be in _backends():
            cap = cv2.VideoCapture(spec.value, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return None
    cap = cv2.VideoCapture(spec.value)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def transform(frame, rotate=0, flip=False):
    if rotate:
        frame = cv2.rotate(frame, _ROT[rotate])
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def list_cameras(max_index=6):
    found = []
    for i in range(max_index):
        cap = open_capture(i, 640, 480)
        if cap is None:
            continue
        ok, fr = cap.read()
        if ok and fr is not None:
            found.append((i, fr.shape[1], fr.shape[0]))
        cap.release()
    return found


class FrameSource:
    RECONNECT_S = 15.0

    def __init__(self, source=0, width=1280, height=720, wait=5.0):
        self.spec = parse_source(source)
        self.width_req, self.height_req = width, height
        self.cap = open_capture(self.spec, width, height)
        if self.cap is None:
            raise IOError(f"Cannot open capture source: {self.spec.label()}")
        self.state = "connected"
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._stop = threading.Event()
        self._thread = None
        if self.spec.live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
            end = time.time() + wait
            while self._frame is None and time.time() < end:
                time.sleep(0.02)

    def _reader(self):
        lost_at = None
        while not self._stop.is_set():
            ret, fr = self.cap.read() if self.cap is not None else (False, None)
            if ret and fr is not None:
                lost_at = None
                self.state = "connected"
                fr = transform(fr, self.spec.rotate, self.spec.flip)
                with self._lock:
                    self._ret, self._frame = True, fr
                time.sleep(0.001)
                continue
            self.state = "reconnecting"
            lost_at = lost_at or time.time()
            if time.time() - lost_at > self.RECONNECT_S:
                with self._lock:
                    self._ret, self._frame = False, None
                self.state = "lost"
                return
            if self.cap is not None:
                self.cap.release()
            time.sleep(0.5)
            self.cap = open_capture(self.spec, self.width_req, self.height_req)

    def read(self):
        if self._thread is None:
            ret, fr = self.cap.read()
            if not ret or fr is None:
                return False, None
            return True, transform(fr, self.spec.rotate, self.spec.flip)
        while not self._stop.is_set():
            with self._lock:
                if self.state == "lost":
                    return False, None
                fr = self._frame
                self._frame = None
            if fr is not None:
                return True, fr
            time.sleep(0.004)
        return False, None

    @property
    def fps(self):
        return (self.cap.get(cv2.CAP_PROP_FPS) if self.cap is not None else 0) or 30.0

    def abort(self):
        self._stop.set()

    def release(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


ThreadedCapture = FrameSource
