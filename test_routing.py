import os
from datetime import datetime

import pytest

from routing import TriageRouter, LKWClock, load_destinations

NOW = datetime(2026, 11, 23, 14, 0, 0)


def cfg(dests, **kw):
    return {"destinations": dests, **kw}


CLINIC = {"name": "Clinic", "capabilities": [], "eta_min": 10}
DISTRICT = {"name": "District", "capabilities": ["ct", "thrombolysis"], "eta_min": 35}
CSC = {"name": "CSC", "capabilities": ["ct", "cta", "thrombolysis", "thrombectomy"], "eta_min": 90}
CSC_NEAR = dict(CSC, name="CSC-near", eta_min=55)


def router(dests, lkw_min_ago=60, **kw):
    r = TriageRouter(cfg(dests, **kw))
    if lkw_min_ago is not None:
        r.clock.set_minutes_ago(lkw_min_ago, now=NOW)
    return r


def rec(r, level):
    return r.recommend({"level": level}, now=NOW)


def test_never_routes_to_non_ct_clinic():
    out = rec(router([CLINIC, DISTRICT, CSC]), "LOW")
    assert out["primary"]["name"] == "District"


def test_no_ct_destination():
    out = rec(router([CLINIC]), "HIGH")
    assert out["mode"] == "NO_CT" and out["primary"] is None and out["warnings"]


def test_high_lvo_far_thrombectomy_centre_uses_drip_and_ship():
    out = rec(router([DISTRICT, CSC]), "HIGH")
    assert out["mode"] == "DRIP_AND_SHIP"
    assert out["primary"]["name"] == "District" and out["alternate"]["name"] == "CSC"


def test_high_lvo_within_bypass_margin_goes_direct():
    out = rec(router([DISTRICT, CSC_NEAR]), "HIGH")
    assert out["mode"] == "DIRECT_EVT" and out["primary"]["name"] == "CSC-near"


def test_bypass_margin_is_configurable():
    out = rec(router([DISTRICT, CSC], max_bypass_extra_min=60), "HIGH")
    assert out["mode"] == "DIRECT_EVT"


def test_elevated_uses_tighter_margin():
    out = rec(router([DISTRICT, CSC_NEAR]), "ELEVATED")
    assert out["primary"]["name"] == "District"


def test_low_goes_nearest_ct():
    out = rec(router([DISTRICT, CSC_NEAR]), "LOW")
    assert out["primary"]["name"] == "District" and out["mode"] == "NEAREST_CT"


def test_thrombolysis_window_math_and_warning_after_4h30():
    r = router([DISTRICT], lkw_min_ago=250)
    out = rec(r, "LOW")
    assert out["windows"]["ivt_remaining_min"] == pytest.approx(20, abs=0.1)
    assert any("AFTER the 4.5 h" in w for w in out["warnings"])


def test_beyond_window_still_routes_to_evt_up_to_24h():
    out = rec(router([DISTRICT, CSC_NEAR], lkw_min_ago=8 * 60), "HIGH")
    assert out["primary"]["name"] == "CSC-near"
    assert any("Beyond the 4.5 h" in w for w in out["warnings"])


def test_evt_excluded_when_arrival_after_24h():
    out = rec(router([DISTRICT, CSC_NEAR], lkw_min_ago=23 * 60 + 30), "HIGH")
    assert out["primary"]["name"] == "District"
    assert any("No thrombectomy" in w for w in out["warnings"])


def test_unknown_lkw_warns_but_still_routes_to_ct():
    r = router([DISTRICT, CSC_NEAR], lkw_min_ago=None)
    out = rec(r, "HIGH")
    assert out["primary"] and any("LKW UNKNOWN" in w for w in out["warnings"])
    r.clock.set_unknown()
    assert rec(r, "HIGH")["windows"]["lkw_unknown_flag"]


def test_closed_destination_is_skipped():
    d = dict(DISTRICT, available_now=False)
    out = rec(router([d, CSC]), "LOW")
    assert out["primary"]["name"] == "CSC"


def test_unknown_eta_is_flagged_not_silently_ranked():
    d = {"name": "NoETA", "capabilities": ["ct"], "eta_min": None}
    out = rec(router([d, CSC]), "LOW")
    assert out["primary"]["name"] == "CSC" and any("ETA unknown" in w for w in out["warnings"])


def test_eta_estimated_from_gps_when_not_entered():
    d = {"name": "Far", "capabilities": ["ct", "thrombolysis"], "lat": 25.0, "lon": 85.9}
    r = TriageRouter(cfg([d], avg_speed_kmh=40), position=(25.0, 85.0))
    r.clock.set_minutes_ago(30, now=NOW)
    out = r.recommend({"level": "LOW"}, now=NOW)
    assert 170 < out["primary"]["eta_min"] < 210
    assert out["primary"]["eta_source"] == "estimated from distance"


def test_example_destinations_trigger_warning():
    d = dict(DISTRICT, example=True)
    assert any("EXAMPLE" in w for w in rec(router([d]), "LOW")["warnings"])


def test_shipped_destinations_file_loads_and_never_picks_the_no_ct_example():
    here = os.path.dirname(os.path.abspath(__file__))
    r = TriageRouter(load_destinations(os.path.join(here, "destinations.json")))
    r.clock.set_minutes_ago(30, now=NOW)
    out = r.recommend({"level": "HIGH"}, now=NOW)
    assert out["primary"]["name"] != "EXAMPLE Primary clinic (no CT)"
    assert any("EXAMPLE" in w for w in out["warnings"])


def test_lkw_clock_time_in_future_means_yesterday():
    c = LKWClock()
    c.set_clock_time("23:30", now=NOW)
    assert c.elapsed_min(NOW) == pytest.approx(14 * 60 + 30)
    c.set_clock_time("13:15", now=NOW)
    assert c.elapsed_min(NOW) == pytest.approx(45)


@pytest.mark.parametrize("bad", ["25:00", "abc", "12", "", "12:60"])
def test_lkw_clock_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        LKWClock().set_clock_time(bad, now=NOW)


def test_lkw_negative_minutes_rejected():
    with pytest.raises(ValueError):
        LKWClock().set_minutes_ago(-5)
