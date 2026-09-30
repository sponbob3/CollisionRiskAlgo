"""FRAMEWORK.md test 5: the analytic conflict probability agrees with
Monte Carlo within 0.01 on a sample of geometries; plus the error-model
and calibration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from proximity_pipeline import config
from proximity_pipeline import probability as pr


def test_conflict_probability_matches_monte_carlo():
    rng = np.random.default_rng(42)
    H, V = config.TIERS["T1"]
    worst = 0.0
    for i in range(25):
        rx, ry = rng.uniform(-5, 5, 2)
        ang = rng.uniform(0, 2 * np.pi)
        sp = rng.uniform(0.01, 0.15)
        vx, vy = sp * np.cos(ang), sp * np.sin(ang)
        if i < 3:
            vx = vy = 0.0                     # stationary relative track
        dz = rng.uniform(-1500, 1500); vz = rng.uniform(-10, 10)
        a = vx * vx + vy * vy
        t_cpa = float(np.clip(-(rx * vx + ry * vy) / a, 0, 120)) if a > 0 else 0.0
        ta, tb = rng.uniform(0, 360, 2)
        sig_a = (rng.uniform(0.05, 1.0), rng.uniform(0.05, 0.5),
                 rng.uniform(30, 400))
        sig_b = (rng.uniform(0.05, 1.0), rng.uniform(0.05, 0.5),
                 rng.uniform(30, 400))
        p = pr.conflict_probability(
            [rx], [ry], [vx], [vy], [dz], [vz], [t_cpa], [ta], [tb],
            [[sig_a[0]], [sig_a[1]], [sig_a[2]]],
            [[sig_b[0]], [sig_b[1]], [sig_b[2]]], H, V)[0]
        m = pr.conflict_probability_mc(rx, ry, vx, vy, dz, vz, t_cpa, ta, tb,
                                       sig_a, sig_b, H, V,
                                       config.PROB_MC_DRAWS, seed=i)
        worst = max(worst, abs(p - m))
        assert abs(p - m) < 0.01, (i, p, m)
    assert worst < 0.01


def test_error_model_interpolates_and_floors():
    table = pd.DataFrame({
        "phase": ["all"] * 3 + ["arrival"] * 3,
        "tau_s": [10.0, 20.0, 30.0] * 2, "n": [100] * 6,
        "sigma_along_nm": [0.1, 0.2, 0.3, 0.05, 0.1, 0.15],
        "sigma_cross_nm": [0.1, 0.2, 0.3, 0.05, 0.1, 0.15],
        "sigma_z_ft": [50, 100, 150, 40, 80, 120],
    })
    em = pr.ErrorModel(table)
    sa, sc, sz = em.sigma(np.array(["all", "arrival", "departure"]),
                          np.array([15.0, 15.0, 0.0]))
    assert abs(sa[0] - 0.15) < 1e-9 and abs(sa[1] - 0.075) < 1e-9
    assert sa[2] == config.ERROR_MODEL_SIGMA0_NM       # floor at tau -> 0
    assert sz[2] == config.ERROR_MODEL_SIGMA0_FT
    # departure falls back to the pooled curve
    sa_d, _, _ = em.sigma(np.array(["departure"]), np.array([20.0]))
    assert abs(sa_d[0] - 0.2) < 1e-9


def test_calibration_metrics():
    rng = np.random.default_rng(0)
    prob = rng.uniform(0, 1, 20000)
    hit = rng.uniform(0, 1, 20000) < prob        # perfectly calibrated
    cal = pr.calibration(prob, hit)
    assert cal["ece"] < 0.03 and cal["bss"] > 0.25   # 1/3 for uniform p
    assert len(cal["table"]) == config.PROB_RELIABILITY_BINS
    bad = pr.calibration(np.full(20000, 0.9), hit)   # over-confident
    assert bad["ece"] > 0.3 and bad["bss"] < 0
