import numpy as np
import pandas as pd
from _loader import load
from common import track_record

tr = load("clinic_w5_track_record", "track_record")
BT, LIVE = track_record()


def test_bands_catch_the_broken_strategy():
    inside = tr.inside_bands(BT, LIVE)
    assert list(inside.index) == list(LIVE.index) and list(inside.columns) == ["momentum", "pairs", "options"]
    assert inside["options"].iloc[-1] == False and inside[["momentum", "pairs"]].mean().min() > 0.8   # noqa: E712


def test_weekly_reviews_and_capital():
    inside = tr.inside_bands(BT, LIVE)
    dec = tr.weekly_reviews(inside, {})
    assert dec.to_dict("list") == {"momentum": ["scale", "hold", "scale", "scale"],
                                   "pairs": ["scale", "scale", "scale", "scale"],
                                   "options": ["scale", "demote", "demote", "demote"]}
    cap = tr.capital_path(dec, LIVE)
    np.testing.assert_allclose(cap["options"], [0.1, 0.0, 0.0, 0.0])                     # back to paper, and stays
    np.testing.assert_allclose(cap["pairs"], [0.1, 0.2, 0.2, 0.3])
    np.testing.assert_allclose(cap["momentum"], [0.1, 0.1, 0.1, 0.2])
    incident_week = tr.weekly_reviews(inside, {1: {"pairs": 2}})
    assert incident_week.loc[1, "pairs"] == "demote"


def test_dossier_and_graduation():
    d = tr.dossier()
    assert d["go_live"] == (True, []) and d["graduation"] == (True, []) and d["audit_ok"]
    bad = tr.audit_trail(bypass="O007")
    ok, why = tr.gl.graduation(bad, {}, set(), {"momentum": 0.97})
    assert not ok and why == ["orders bypassed risk: O007"]
    assert isinstance(d["capital"], pd.DataFrame) and len(d["decisions"]) == 4
