"""FRAMEWORK.md test 6: on simulated negative binomial windows with known
covariate effects, the fitted limits achieve nominal exceedance (within
binomial error), and the machinery's fallbacks behave."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from proximity_pipeline import baseline, config


def simulate_windows(n_days: int, alpha: float, seed: int,
                     start="2025-01-06") -> pd.DataFrame:
    """Clock windows with hour / weekend / flow / traffic effects."""
    rng = np.random.default_rng(seed)
    rows = []
    day0 = pd.Timestamp(start, tz="UTC")
    for d in range(n_days):
        day = day0 + pd.Timedelta(days=d)
        weekend = day.dayofweek >= 5
        flow = "north" if rng.random() < 0.6 else "south"
        for k in range(144):
            hour = (k * 10 // 60)
            traffic = 4 + 6 * np.exp(-((hour - 14) / 5.0) ** 2)
            n_ac = max(0, rng.poisson(traffic))
            pair_hours = n_ac * (n_ac - 1) / 2 * rng.uniform(0.02, 0.05)
            log_rate = (np.log(3.0) + 0.4 * np.sin(2 * np.pi * hour / 24)
                        + (0.3 if weekend else 0.0)
                        + (-0.4 if flow == "south" else 0.0)
                        + 0.3 * np.log(max(n_ac, 1)))
            mu = pair_hours * np.exp(log_rate)
            if alpha > 0:
                y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)) \
                    if mu > 0 else 0
            else:
                y = rng.poisson(mu)
            t_start = int(day.value // 10**9) + k * 600
            rows.append({"window_id": f"d{d:03d}_{k:03d}",
                         "day": f"d{d:03d}", "t_start": t_start,
                         "t_end": t_start + 600, "n_aircraft": n_ac,
                         "pair_hours": pair_hours, "local_hour": hour,
                         "daytype": "weekend" if weekend else "weekday",
                         "month": f"{day:%Y-%m}", "flow": flow,
                         "quality": "good", "partial_coverage": False,
                         "T1_any": int(y)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("alpha", [0.5, 0.0])
def test_fitted_limits_achieve_nominal_exceedance(alpha):
    train = simulate_windows(60, alpha, seed=1)
    test = simulate_windows(40, alpha, seed=2, start="2025-04-07")
    model = baseline.fit_count_model(train, "T1_any")
    assert model.kind == ("nb" if alpha > 0 else "poisson"), model.notes
    if alpha > 0:
        assert 0.3 < model.alpha < 0.8
    coef = dict(zip(model.coef_names, model.coef))
    assert abs(coef["daytype[weekend]"] - 0.3) < 0.12
    assert abs(coef["flow[south]"] + 0.4) < 0.12
    assert abs(coef["log_n_aircraft"] - 0.3) < 0.12
    sc = model.score(test[test["pair_hours"] > 0], seed=3)
    n = len(sc)
    r95 = sc["T1_any_exceed95"].mean()
    r99 = sc["T1_any_exceed99"].mean()
    # nominal 5 % / 1 % within binomial error (n ~ 5,000)
    assert abs(r95 - 0.05) < 3 * np.sqrt(0.05 * 0.95 / n) + 0.005, r95
    assert abs(r99 - 0.01) < 3 * np.sqrt(0.01 * 0.99 / n) + 0.003, r99
    # PIT values uniform: randomized quantile residuals ~ N(0, 1)
    z = sc["T1_any_z"].dropna()
    assert stats.kstest(z, "norm").pvalue > 0.01
    assert abs(z.mean()) < 0.06 and abs(z.std() - 1) < 0.06


def test_out_of_sample_validation_passes_on_simulated_data():
    df = simulate_windows(90, 0.5, seed=5)
    oos = baseline.out_of_sample(df, "T1_any", seed=1)
    assert oos["fold_kind"] == "month"
    assert oos["pass95"] and oos["pass99"], (oos["exceed95_rate"],
                                             oos["exceed99_rate"])
    assert oos["pit_ks_p"] > 0.01


def test_phase1_removes_only_extreme_windows_within_budget():
    df = simulate_windows(30, 0.3, seed=7)
    # inject a handful of absurd windows
    idx = df.index[df["pair_hours"] > 0][:5]
    df.loc[idx, "T1_any"] = 200
    model, removed = baseline.phase1(df, "T1_any")
    # the injected windows go (other genuinely extreme windows may too),
    # never more than the budget
    assert set(idx) <= set(removed.index)
    assert len(removed) <= config.PHASE1_MAX_REMOVED_FRAC * len(df)
    assert model.kind in ("nb", "poisson")


def test_sparse_metric_falls_back_to_pooled_rate():
    df = simulate_windows(20, 0.0, seed=3)
    df["T3_any"] = (np.random.default_rng(1).random(len(df)) < 0.002).astype(int)
    m = baseline.fit_count_model(df, "T3_any")
    assert m.kind == "pooled"
    mu = m.mu(df)
    assert np.allclose(mu, m.pooled_rate * df["pair_hours"])


def test_build_and_reload_baseline(tmp_path):
    df = simulate_windows(35, 0.4, seed=9)
    for col in ("T1_observed", "T1_predicted", "T1_any_nonprocedural"):
        df[col] = df["T1_any"]
    for col in ("T2_any", "T2_observed", "T2_predicted", "T3_any",
                "T3_observed", "T3_predicted"):
        df[col] = 0
    df["s_min"] = np.random.default_rng(2).uniform(0.3, 1.3, len(df))
    bl = baseline.build_baseline(df, np.array([]), config.TIERS, quiet=True)
    assert bl.verdict in ("VALID", "VALID WITH WARNINGS"), bl.warnings
    assert bl.count_models["T1_any"].kind == "nb"
    assert bl.count_models["T3_any"].kind == "pooled"
    assert "s_min" in bl.continuous_models
    em = pd.DataFrame({"phase": ["all"], "tau_s": [10.0], "n": [1],
                       "sigma_along_nm": [0.1], "sigma_cross_nm": [0.1],
                       "sigma_z_ft": [50.0]})
    baseline.save_baseline(bl, tmp_path, em, "hash", "sig")
    assert baseline.baseline_is_reusable(tmp_path, "hash", "sig")
    assert not baseline.baseline_is_reusable(tmp_path, "other", "sig")
    bl2 = baseline.load_baseline(tmp_path)
    a = bl.score_counts(df.head(100), seed=1)
    b = bl2.score_counts(df.head(100), seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert (tmp_path / "validation" / "calibration_summary.csv").exists()
