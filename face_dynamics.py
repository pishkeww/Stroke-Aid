from collections import deque

import numpy as np

OPEN_WINDOW = 240
MIN_OPEN_SAMPLES = 60
BLINK_CLOSE, BLINK_OPEN = 0.60, 0.80
BLINK_MIN_S, BLINK_MAX_S = 0.04, 0.70
PAIR_WINDOW_S = 0.25
RATE_WINDOW_S = 60.0
RATE_MIN_SPAN_S = 10.0
PTOSIS_NORMAL, PTOSIS_FLAG = 0.08, 0.20


class _EyeBlinks:
    def __init__(self):
        self.opens = deque(maxlen=OPEN_WINDOW)
        self.closed_at = None
        self.min_ratio = 1.0
        self.events = deque(maxlen=200)

    def open_level(self):
        if len(self.opens) < MIN_OPEN_SAMPLES:
            return None
        return float(np.percentile(self.opens, 80))

    def update(self, t, ear):
        lvl = self.open_level()
        if lvl is None or lvl <= 0:
            self.opens.append(ear)
            return
        r = ear / lvl
        if self.closed_at is None:
            if r < BLINK_CLOSE:
                self.closed_at, self.min_ratio = t, r
            else:
                self.opens.append(ear)
        else:
            self.min_ratio = min(self.min_ratio, r)
            if r > BLINK_OPEN:
                d = t - self.closed_at
                if BLINK_MIN_S <= d <= BLINK_MAX_S:
                    self.events.append((self.closed_at, t, self.min_ratio))
                self.closed_at = None
                self.opens.append(ear)

    def reset_closure(self):
        self.closed_at = None


class DynamicsTracker:
    def __init__(self):
        self.L = _EyeBlinks()
        self.R = _EyeBlinks()
        self.t_first = None
        self.t_last = None

    def update(self, t, earL, earR, ok):
        if not ok:
            self.L.reset_closure()
            self.R.reset_closure()
            return
        if self.t_first is None:
            self.t_first = t
        self.t_last = t
        self.L.update(t, earL)
        self.R.update(t, earR)

    def report(self, side_of):
        out = {"ptosis_asym": None, "ptosis_side": None, "ptosis_flag": False,
               "blink_count": 0, "blink_rate_per_min": None,
               "blink_lag_ms": None, "blink_depth_asym": None}
        oL, oR = self.L.open_level(), self.R.open_level()
        if oL is not None and oR is not None:
            asym = abs(oL - oR) / max(oL, oR, 1e-9)
            out["ptosis_asym"] = float(asym)
            if asym >= PTOSIS_NORMAL:
                out["ptosis_side"] = side_of("L" if oL < oR else "R")
            out["ptosis_flag"] = bool(asym >= PTOSIS_FLAG)

        if self.t_last is None:
            return out
        horizon = self.t_last - RATE_WINDOW_S
        ev_l = [e for e in self.L.events if e[0] >= horizon]
        ev_r = [e for e in self.R.events if e[0] >= horizon]
        pairs = []
        used = set()
        for a in ev_l:
            best, bj = None, None
            for j, b in enumerate(ev_r):
                if j in used:
                    continue
                d = abs(a[0] - b[0])
                if d <= PAIR_WINDOW_S and (best is None or d < best):
                    best, bj = d, j
            if bj is not None:
                used.add(bj)
                pairs.append((a, ev_r[bj]))
        out["blink_count"] = len(pairs)
        span = min(RATE_WINDOW_S, self.t_last - self.t_first)
        if span >= RATE_MIN_SPAN_S:
            out["blink_rate_per_min"] = float(len(pairs) * 60.0 / span)
        if pairs:
            out["blink_lag_ms"] = float(np.mean([abs(a[0] - b[0]) for a, b in pairs]) * 1000.0)
            out["blink_depth_asym"] = float(np.mean([abs(a[2] - b[2]) for a, b in pairs]))
        return out


def smile_onset(samples, rest, t0):
    def exc(c, rc, sign):
        return float(np.hypot(max(0.0, -(c[1] - rc[1])), max(0.0, sign * (c[0] - rc[0]))))

    ts = np.array([t for t, _ in samples]) - t0
    eL = np.array([exc(m["cL"], rest["cL"], -1.0) for _, m in samples])
    eR = np.array([exc(m["cR"], rest["cR"], 1.0) for _, m in samples])
    peak = max(np.percentile(eL, 90), np.percentile(eR, 90))
    if peak <= 0:
        return None
    thr = 0.3 * peak

    def first(e):
        k = np.nonzero(e >= thr)[0]
        return float(ts[k[0]]) if len(k) else None

    a, b = first(eL), first(eR)
    if a is None or b is None:
        return None
    return {"lag_s": abs(a - b), "late": "L" if a > b else "R", "tL": a, "tR": b}
