import json
import math
from datetime import datetime, timedelta

IVT_WINDOW_MIN = 4.5 * 60
EVT_WINDOW_MIN = 6 * 60
EVT_EXTENDED_MIN = 24 * 60
MAX_BYPASS_EXTRA_MIN = 30
MAX_BYPASS_EXTRA_MIN_ELEVATED = 15
DETOUR_FACTOR = 1.4
DEFAULT_SPEED_KMH = 40.0


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


class LKWClock:

    def __init__(self):
        self.lkw = None
        self.unknown = False

    def set_minutes_ago(self, minutes, now=None):
        now = now or datetime.now()
        if minutes < 0:
            raise ValueError("minutes must be >= 0")
        self.lkw, self.unknown = now - timedelta(minutes=minutes), False

    def set_clock_time(self, hhmm, now=None):
        now = now or datetime.now()
        try:
            hh, mm = (int(x) for x in hhmm.strip().split(":"))
            t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except (ValueError, AttributeError):
            raise ValueError("LKW must be HH:MM (24 h)")
        if t > now:
            t -= timedelta(days=1)
        self.lkw, self.unknown = t, False

    def set_unknown(self):
        self.lkw, self.unknown = None, True

    def clear(self):
        self.lkw, self.unknown = None, False

    def elapsed_min(self, now=None):
        return None if self.lkw is None else ((now or datetime.now()) - self.lkw).total_seconds() / 60.0


class Destination:
    def __init__(self, d):
        self.name = d["name"]
        self.caps = set(c.lower() for c in d.get("capabilities", []))
        self.lat, self.lon = d.get("lat"), d.get("lon")
        self.eta_override = d.get("eta_min")
        self.open = bool(d.get("available_24x7", True)) and bool(d.get("available_now", True))
        self.phone = d.get("phone", "")
        self.notes = d.get("notes", "")
        self.example = bool(d.get("example", False))

    @property
    def has_ct(self):
        return "ct" in self.caps

    @property
    def can_ivt(self):
        return self.has_ct and "thrombolysis" in self.caps

    @property
    def can_evt(self):
        return self.has_ct and "thrombectomy" in self.caps


def load_destinations(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


class TriageRouter:
    def __init__(self, config, clock=None, position=None):
        self.destinations = [Destination(d) for d in config.get("destinations", [])]
        self.speed = float(config.get("avg_speed_kmh", DEFAULT_SPEED_KMH))
        self.max_bypass = float(config.get("max_bypass_extra_min", MAX_BYPASS_EXTRA_MIN))
        self.max_bypass_elev = float(config.get("max_bypass_extra_min_elevated",
                                                MAX_BYPASS_EXTRA_MIN_ELEVATED))
        self.clock = clock or LKWClock()
        self.position = position

    def set_position(self, lat, lon):
        self.position = (lat, lon)

    def set_eta(self, name, minutes):
        for d in self.destinations:
            if d.name == name:
                d.eta_override = minutes
                return
        raise KeyError(name)

    def eta_min(self, d):
        if d.eta_override is not None:
            return float(d.eta_override), "entered"
        if self.position and d.lat is not None and d.lon is not None:
            km = _haversine_km(*self.position, d.lat, d.lon) * DETOUR_FACTOR
            return km / self.speed * 60.0, "estimated from distance"
        return None, None

    def recommend(self, triage=None, now=None):
        now = now or datetime.now()
        level = (triage or {}).get("level", "INSUFFICIENT_DATA")
        el = self.clock.elapsed_min(now)
        warnings, why = [], []

        windows = {
            "lkw_known": el is not None, "lkw_unknown_flag": self.clock.unknown,
            "elapsed_min": el,
            "ivt_remaining_min": None if el is None else IVT_WINDOW_MIN - el,
            "evt_remaining_min": None if el is None else EVT_WINDOW_MIN - el,
            "evt_ext_remaining_min": None if el is None else EVT_EXTENDED_MIN - el,
        }
        if el is None:
            warnings.append("LKW UNKNOWN: record it (ask witnesses). Time-based eligibility cannot be "
                            "judged; the centre decides on imaging. Still route to a CT centre.")
        elif el > EVT_EXTENDED_MIN:
            warnings.append("More than 24 h since LKW: outside standard treatment windows; still needs CT.")
        elif el > IVT_WINDOW_MIN:
            warnings.append("Beyond the 4.5 h thrombolysis window; thrombectomy may still be possible.")
        if any(d.example for d in self.destinations):
            warnings.append("EXAMPLE destinations in use - replace destinations.json with real centres.")

        cands = []
        for d in self.destinations:
            if not d.open:
                continue
            eta, src = self.eta_min(d)
            if eta is None:
                warnings.append(f"ETA unknown for {d.name}: enter it to include it in ranking.")
                continue
            cands.append((d, eta, src))
        ct = sorted([c for c in cands if c[0].has_ct], key=lambda c: c[1])
        if not ct:
            return self._out("NO_CT", None, None, why + ["No reachable CT-capable destination configured."],
                             warnings + ["Transport to nearest hospital and contact medical control."],
                             windows, level)

        def arrival(c):
            return None if el is None else el + c[1]

        def evt_ok(c):
            a = arrival(c)
            return c[0].can_evt and (a is None or a <= EVT_EXTENDED_MIN)

        nearest_ct = ct[0]
        evt = [c for c in ct if evt_ok(c)]
        nearest_evt = evt[0] if evt else None

        primary, alternate, mode = nearest_ct, nearest_evt, "NEAREST_CT"
        if level == "HIGH":
            if nearest_evt is None:
                why.append("LVO suspected but no thrombectomy-capable centre is available/in window; "
                           "going to nearest CT centre.")
                warnings.append("No thrombectomy centre reachable inside the window.")
            else:
                extra = nearest_evt[1] - nearest_ct[1]
                if nearest_evt is nearest_ct or extra <= self.max_bypass:
                    primary, alternate, mode = nearest_evt, nearest_ct, "DIRECT_EVT"
                    why.append(f"LVO suspected: thrombectomy centre costs {extra:.0f} min extra "
                               f"(limit {self.max_bypass:.0f}).")
                else:
                    primary, alternate, mode = nearest_ct, nearest_evt, "DRIP_AND_SHIP"
                    why.append(f"LVO suspected but thrombectomy centre is {extra:.0f} min further "
                               f"(limit {self.max_bypass:.0f}): nearest CT centre, pre-alert for transfer.")
        elif level == "ELEVATED":
            if nearest_evt and (nearest_evt is nearest_ct or
                                nearest_evt[1] - nearest_ct[1] <= self.max_bypass_elev):
                primary, alternate, mode = nearest_evt, nearest_ct, "DIRECT_EVT"
                why.append("Possible LVO and thrombectomy centre is within the bypass margin.")
            else:
                why.append("Possible stroke/LVO: nearest CT centre; complete RACE to refine.")
        else:
            why.append("Nearest CT-capable centre (does not exclude stroke).")

        a = arrival(primary)
        if a is not None:
            if primary[0].can_ivt and a > IVT_WINDOW_MIN >= (el or 0):
                warnings.append("Expected arrival is AFTER the 4.5 h thrombolysis window.")
            elif el is not None and el <= IVT_WINDOW_MIN and IVT_WINDOW_MIN - el < primary[1] + 30:
                warnings.append("Thrombolysis window closing: allow ~30 min for door-to-needle.")
        if mode == "NEAREST_CT" and level != "HIGH" and not primary[0].can_ivt:
            warnings.append(f"{primary[0].name} has no thrombolysis listed; consider an alternative.")

        return self._out(mode, primary, alternate if alternate is not primary else None,
                         why, warnings, windows, level)

    def _out(self, mode, primary, alternate, why, warnings, windows, level):
        def pack(c):
            if c is None:
                return None
            d, eta, src = c
            el = windows["elapsed_min"]
            return {"name": d.name, "eta_min": round(eta), "eta_source": src,
                    "arrival_elapsed_min": None if el is None else round(el + eta),
                    "capabilities": sorted(d.caps), "phone": d.phone, "notes": d.notes}
        return {"mode": mode, "level": level, "primary": pack(primary), "alternate": pack(alternate),
                "rationale": why, "warnings": warnings, "windows": windows}


def fmt_minutes(m):
    if m is None:
        return "—"
    s = "-" if m < 0 else ""
    m = abs(int(m))
    return f"{s}{m // 60}h{m % 60:02d}m"
