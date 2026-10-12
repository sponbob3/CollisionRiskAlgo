"""Comparison with normal landings (FRAMEWORK.md section 11.6): matching
weights, the placebo test, the pre-trend check, the regression
confirmation, leave-one-out and the similar-landing choice."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proximity_pipeline import compare as C


def _frame(n, rng, after_factor, days=30, hours=(8, 12, 17),
           mean=8.0, early_factor=1.0, prefix="e"):
    """Paired windows with day-level variation: counts and traffic
    before (and 20-10 min before), after = before rate x after_factor."""
    day = rng.integers(0, days, n)
    hour = rng.choice(hours, n)
    level = np.exp(rng.normal(0, 0.3, days))[day] * (1 + 0.5 * (hour == 17))
    traffic = 0.5 * level
    return pd.DataFrame({
        "event_id": [f"{prefix}{i}" for i in range(n)],
        "day": [f"d{d:02d}" for d in day], "hour": hour,
        "flow": np.where(rng.uniform(size=n) < 0.6, "north", "south"),
        "t0": 1_700_000_000 + np.arange(n) * 3600,
        "early_T1_any": rng.poisson(mean * level * early_factor),
        "pre_T1_any": rng.poisson(mean * level),
        "post_T1_any": rng.poisson(mean * level * after_factor),
        "early_pair_hours": traffic * early_factor,
        "pre_pair_hours": traffic,
        "post_pair_hours": traffic * after_factor,
        "pre_aircraft_s": 600 * 3 * level,
        "excluded": False, "pre_bad": False, "post_bad": False,
        "early_bad": False, "cluster_size": 1,
        "pre_contaminated": False, "callsign": "X",
    })


def test_match_weights_follow_the_go_around_strata():
    g = pd.DataFrame({"hour": [8, 8, 17], "flow": ["n", "n", "s"]})
    c = pd.DataFrame({"hour": [8, 8, 8, 8, 17, 3], "flow": ["n"] * 4
                      + ["s", "n"]})
    keep, w = C.match_weights(g, c)
    assert keep.all()
    assert w.tolist() == [0.5, 0.5, 0.5, 0.5, 1.0, 0.0]
    g2 = pd.DataFrame({"hour": [8, 23]})
    keep, w = C.match_weights(g2, pd.DataFrame({"hour": [8, 8]}))
    assert keep.tolist() == [True, False]        # no landing at 23 h
    assert w.sum() == pytest.approx(1.0)


def test_did_placebo_and_pretrend_with_an_effect():
    rng = np.random.default_rng(1)
    g = _frame(150, rng, after_factor=1.3)
    c = _frame(1500, rng, after_factor=0.9, prefix="c")
    r = C.did(g, c, "T1_any", reps=400, seed=1)
    assert 1.25 < r["vs_normal"] < 1.65
    assert r["vs_normal_lo"] > 1.0
    p = C.placebo(g, c, "T1_any", n_sets=400, seed=2)
    assert p["share_at_least_as_large"] < 0.01
    assert 0.8 < p["placebo_median"] < 1.0
    # pre-trend: no difference before the event (early -> pre)
    pt = C.did(g, c, "T1_any", reps=400, seed=3, frm="early", to="pre")
    assert 0.85 < pt["vs_normal"] < 1.15


def test_no_effect_is_flagged_at_about_the_nominal_rate():
    """Without an effect the 95 % interval excludes 1 in about 5 % of
    datasets (60 simulations, binomial tolerance), and the placebo
    share is not extreme on average."""
    hits, shares = 0, []
    for s in range(60):
        rng = np.random.default_rng(300 + s)
        g = _frame(130, rng, after_factor=0.9, days=60)
        c = _frame(1300, rng, after_factor=0.9, days=60, prefix="c")
        r = C.did(g, c, "T1_any", reps=200, seed=s)
        hits += not (r["vs_normal_lo"] <= 1.0 <= r["vs_normal_hi"])
        shares.append(C.placebo(g, c, "T1_any", n_sets=200, seed=s)
                      ["share_at_least_as_large"])
    assert hits <= 9, hits
    assert 0.3 < np.mean(shares) < 0.7


def test_regression_check_recovers_the_interaction():
    rng = np.random.default_rng(7)
    g = _frame(200, rng, after_factor=1.3)
    c = _frame(2000, rng, after_factor=0.9, prefix="c")
    r = C.regression_check(g, c, "T1_any")
    assert r["ratio_lo"] < 1.3 / 0.9 < r["ratio_hi"]
    assert r["p_value"] < 0.01


def test_leave_one_out_finds_the_outlier_and_robustness_rows():
    rng = np.random.default_rng(9)
    g = _frame(60, rng, after_factor=1.0)
    c = _frame(600, rng, after_factor=1.0, prefix="c")
    g.loc[7, ["pre_T1_any", "post_T1_any"]] = [5, 200]
    loo = C.leave_one_out(g, c, "T1_any")
    assert loo.loc[loo["vs_normal_without"].idxmin(), "event_id"] == "e7"
    C.add_traffic_band(g, c)
    rb = C.robustness(g, c, "T1_any", reps=100, seed=0)
    assert rb.iloc[0]["variation"].startswith("Main analysis")
    row = rb[rb["variation"].str.startswith("Without the most")].iloc[0]
    assert "e7" in row["note"]
    assert row["vs_normal"] < rb.iloc[0]["vs_normal"]


def test_before_checks_and_similar_landings():
    rng = np.random.default_rng(11)
    g = _frame(80, rng, after_factor=1.0)
    c = _frame(800, rng, after_factor=1.0, prefix="c")
    for df in (g, c):
        df["pre_T1_any_expected"] = df["pre_T1_any"].mean()
        df["pre_T1_any_in_control"] = True
    b = C.before_checks(g, c, reps=200, seed=0).set_index("group")
    assert b.loc["go-arounds", "in_normal_range"] == 1.0
    assert np.isfinite(b.iloc[-1]["traffic_ratio"])
    row = g.iloc[0]
    sim = C.similar_landings(row, c, k=5)
    assert len(sim) == 5 and (sim["hour"] == row["hour"]).all()
    gap = np.abs(sim["pre_pair_hours"] - row["pre_pair_hours"]).to_numpy()
    assert gap[0] <= gap.max()
