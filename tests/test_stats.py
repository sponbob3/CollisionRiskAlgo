"""FRAMEWORK.md test 7: a known post-go-around rate increase is detected
(IRR estimate with CI), and with no effect the false-positive rate is
close to alpha across repeated simulations. Also the end-to-end run on a
synthetic dataset with injected close pairs."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from proximity_pipeline import config, stats


def _paired_events(n_events, irr, alpha, seed, mean_pre=3.0):
    """Paired pre/post counts with event-level heterogeneity (a random
    event effect), over-dispersion, and day clusters."""
    rng = np.random.default_rng(seed)
    days = np.repeat(np.arange(n_events // 3 + 1), 3)[:n_events]
    rows = []
    for i in range(n_events):
        eff = np.exp(rng.normal(0, 0.3))
        e_pre = rng.uniform(0.5, 1.5); e_post = rng.uniform(0.5, 1.5)
        mu_pre = mean_pre * eff * e_pre
        mu_post = mean_pre * eff * e_post * irr

        def draw(mu):
            if alpha > 0:
                return rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
            return rng.poisson(mu)

        y_pre, y_post = draw(mu_pre), draw(mu_post)
        rows.append({"event_id": f"e{i}", "day": f"d{days[i]}",
                     "pre_T1_any": y_pre, "post_T1_any": y_post,
                     "pre_pair_hours": e_pre, "post_pair_hours": e_post,
                     "pre_T1_any_expected": mu_pre,
                     "post_T1_any_expected": mu_pre * e_post / e_pre,
                     "pre_T1_any_z": (y_pre - mu_pre) / np.sqrt(mu_pre),
                     "post_T1_any_z": (y_post - mu_pre * e_post / e_pre)
                     / np.sqrt(mu_pre * e_post / e_pre),
                     "excluded": False})
    r = pd.DataFrame(rows)
    r["diff_T1_any_z"] = r["post_T1_any_z"] - r["pre_T1_any_z"]
    return r


def test_known_effect_is_detected():
    r = _paired_events(120, irr=1.5, alpha=0.3, seed=1)
    out = stats.irr_post_pre(r, "T1_any", reps=400, seed=2)
    assert 1.3 < out["irr"] < 1.75, out
    assert out["irr_lo"] > 1.0 and out["irr_hi"] < 2.2
    assert out["wilcoxon_paired_p"] < 0.01
    s = stats.sir_window(r, "T1_any", "post", reps=400, seed=3)
    assert s["sir_post_lo"] > 1.0
    mde_irr, _ = stats.minimum_detectable(r["pre_T1_any"].sum(),
                                          r["post_T1_any_expected"].sum(),
                                          len(r), 0.3)
    assert 1.1 < mde_irr < 1.5


def test_false_positive_rate_is_close_to_alpha():
    """200 null simulations: the day-clustered bootstrap CI excludes 1 in
    about 5 % of them (binomial tolerance)."""
    hits = 0
    n_sim = 200
    for s in range(n_sim):
        r = _paired_events(60, irr=1.0, alpha=0.3, seed=100 + s)
        out = stats.irr_post_pre(r, "T1_any", reps=200, seed=s)
        if out["irr_lo"] > 1.0 or out["irr_hi"] < 1.0:
            hits += 1
    rate = hits / n_sim
    assert rate <= 0.05 + 2.5 * np.sqrt(0.05 * 0.95 / n_sim) + 0.01, rate


def test_holm_and_equilibrium_helpers():
    assert stats.holm([0.01, 0.02, 0.03]) == [0.03, 0.04, 0.04]
    r = _paired_events(80, irr=1.0, alpha=0.0, seed=7)
    r["pre_T1_any_in_control"] = r["pre_T1_any_z"] < 1.645
    eq = stats.equilibrium(r, reps=200)
    assert eq["verdict"] == "PASS", eq
    assert eq["sir_pre_ci"][0] <= 1.0 <= eq["sir_pre_ci"][1]


@pytest.mark.slow
def test_end_to_end_run_detects_injected_increase(effect_dataset, kbna,
                                                  tmp_path, monkeypatch):
    """The whole pipeline on synthetic days where injected close pairs
    occur four times as often in post-go-around windows: the injected
    pairs are found as encounters, the run produces every required
    output, and the post/pre rate ratio of injected-pair encounters
    exceeds 1 (the natural post-go-around traffic is calmer, so the
    check is on the pairs the generator injected)."""
    from proximity_pipeline import pipeline
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(config, "BOOTSTRAP_REPS", 200)
    monkeypatch.setattr(config, "STABILITY_REPS", 2)
    args = argparse.Namespace(
        goaround_run=None, rebuild_baseline=False, baseline_only=False,
        baseline_from=None, window=None, include_ambiguous=False,
        no_sensitivity=True, no_plots=True, no_event_plots=True,
        goaround_report=False, calibrate=False, limit=None, workers=2,
        probability=False)
    run_dir = pipeline.run_all(effect_dataset, args, ["test"])
    for name in ("run_config.txt", "ceiling_check.csv", "data_quality.csv",
                 "window_metrics.parquet", "encounters.parquet",
                 "encounters.csv", "go_around_risk.csv",
                 "results_primary.csv", "equilibrium.csv", "epoch.csv",
                 "change_summary.csv", "go_around_aircraft.csv"):
        assert (run_dir / name).exists(), name
    base = config.OUTPUT_DIR / "KBNA" / "proximity_risk" / "baseline" / "effecttest"
    assert (base / "baseline.json").exists()
    # the probability model is an add-on (--probability), off by default
    assert not (base / "error_model.csv").exists()
    assert not (run_dir / "probability").exists()
    ch = pd.read_csv(run_dir / "change_summary.csv").set_index("measure")
    t1 = ch.loc["T1_any"]
    assert t1["ratio"] == pytest.approx(t1["after_total"] / t1["before_total"])
    assert np.isfinite(t1["normal_ratio"]) and np.isfinite(t1["vs_normal"])
    enc = pd.read_parquet(run_dir / "encounters.parquet")
    inj = enc[enc["callsign_a"].str.startswith("INT")
              & enc["callsign_b"].str.startswith("INT")
              & (enc["tier"] == "T1") & (enc["kind"] == "any")]
    assert len(inj) > 10
    # injected pairs (airspace-wide, not involving the go-around aircraft)
    # attributed to the pre / post windows of the go-arounds by time
    risk = pd.read_csv(run_dir / "go_around_risk.csv")
    w = config.WINDOW_MIN * 60
    t = inj["t_min"].to_numpy()
    n_pre = sum(((t >= t0 - w) & (t < t0)).sum() for t0 in risk["t0"])
    n_post = sum(((t >= t0) & (t < t0 + w)).sum() for t0 in risk["t0"])
    assert n_post > 2 * n_pre, (n_pre, n_post)
    res = pd.read_csv(run_dir / "results_primary.csv")
    prim = res[(res["scope"] == "airspace") & (res["endpoint"] == "T1_any")
               & (res["event_set"] == "all")].iloc[0]
    assert prim["n_events"] == len(risk[~risk["excluded"]])
    assert np.isfinite(prim["irr"]) and np.isfinite(prim["sir_post"])
    # reuse: a second run with the same data reuses the stored baseline;
    # this run also exercises the figures, event plots and PDFs
    args.no_plots = False
    args.no_event_plots = False
    args.goaround_run = 1
    run2 = pipeline.run_all(effect_dataset, args, ["test"])
    txt = (run2 / "run_config.txt").read_text()
    assert "baseline_reused = True" in txt
    for name in ("summary.pdf", "figures/superposed_epoch.png",
                 "figures/forest.png", "figures/equilibrium.png",
                 "figures/daily_levels.png", "figures/encounter_density.png",
                 "figures/change_vs_normal.png",
                 "figures/ceiling_check.png"):
        assert (run2 / name).exists(), name
    assert len(list((run2 / "events").glob("*.png"))) == len(risk)
    assert (base / "baseline_report.pdf").exists()
    assert (base / "validation" / "calibration_exceedance.png").exists()


def _pairs(pre, post, hours, days, col="T1_any"):
    return pd.DataFrame({"event_id": [f"e{i}" for i in range(len(pre))],
                         f"pre_{col}": pre, f"post_{col}": post,
                         "hour": hours, "day": days, "excluded": False})


def test_change_summary_totals_and_normal_landings(monkeypatch):
    """Totals ratio = sum after / sum before; normal landings are
    weighted to the go-arounds' hours; vs-normal = the ratio of the two."""
    monkeypatch.setattr(stats, "CHANGE_MEASURES",
                        [("T1_any", "T1_any", "T1 encounters")])
    ga = _pairs([10, 10, 10, 10], [15, 15, 15, 15], [8, 8, 17, 17],
                ["d1", "d2", "d3", "d4"])
    # normal landings: at 08 h nothing changes, at 17 h counts halve, and
    # a 03 h landing (no go-around at that hour) must get zero weight
    ctl = _pairs([10, 10, 10, 10, 50], [10, 10, 5, 5, 500],
                 [8, 8, 17, 17, 3], ["d1", "d2", "d3", "d4", "d1"])
    out = stats.change_summary(ga, ctl, reps=200, seed=1).iloc[0]
    assert out["ratio"] == pytest.approx(1.5)
    assert out["normal_ratio"] == pytest.approx(30 / 40)
    assert out["vs_normal"] == pytest.approx(1.5 / 0.75)
    assert out["n_controls"] == 4
    assert out["ratio_lo"] <= 1.5 <= out["ratio_hi"]


def test_go_around_aircraft_involvement():
    """The go-around aircraft's encounters vs an average aircraft in the
    same window: here every aircraft is in 2 encounters per 600 s, so
    the ratio is 1; doubling its own encounters doubles the ratio."""
    ewm = pd.DataFrame({"event_id": ["e0", "e0"], "window": ["pre", "post"],
                        "T1_any": [10, 10], "aircraft_s": [6000, 6000],
                        "day": ["d1", "d1"], "t_start": [0, 600],
                        "t_end": [600, 1200]})
    # 10 aircraft x 600 s; 10 encounters -> 20 involvements -> 2 per aircraft
    einv = ewm.copy()
    einv["T1_any"] = [2, 4]
    einv["own_present_s"] = [600, 600]
    risk = pd.DataFrame({"event_id": ["e0"], "excluded": [False]})
    out = stats.ga_aircraft_involvement(ewm, einv, risk, {"T1": (3, 1000)},
                                        reps=50, seed=0).set_index("window")
    assert out.loc["pre", "ratio"] == pytest.approx(1.0)
    assert out.loc["post", "ratio"] == pytest.approx(4 / (600 * 16 / 5400))
    assert out.loc["post", "share_of_window_in_area"] == pytest.approx(1.0)


def test_plain_labels():
    from proximity_pipeline import labels as L
    assert L.metric("T2_predicted") == "T2 close encounters, predicted"
    assert L.sensitivity_row("window_min", 5.0) == "Window 5 min"
    assert L.sensitivity_row("T1_H", "x0.75") == "T1 horizontal size ×0.75"
    assert L.sensitivity_row("ceiling_ft_agl", 2500.0) == "Ceiling 2,500 ft"
    assert "_" not in L.sensitivity_row("event_set", "include_ambiguous")
