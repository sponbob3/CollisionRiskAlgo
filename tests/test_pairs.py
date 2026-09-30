"""FRAMEWORK.md tests 1-4: zone-entry intervals vs brute force, no missed
passes between samples, synthetic scenarios (tier, class, separation),
pruning safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proximity_pipeline import config, geometry, loading, pairs
from proximity_pipeline import goaround_adapter as ga
from tests.synthetic_tracks import DAY0, day_grid_from, straight_track

T1_H, T1_V = config.TIERS["T1"]
T3_H, T3_V = config.TIERS["T3"]
FT_PER_NM = ga.FT_PER_NM


def _brute_force(rx, ry, vx, vy, dz, vz, H, V, tau_max, dt=0.01):
    t = np.arange(0.0, tau_max + dt / 2, dt)
    inside = ((np.hypot(rx + vx * t, ry + vy * t) < H)
              & (np.abs(dz + vz * t) < V))
    if not inside.any():
        return np.nan, np.nan
    idx = np.flatnonzero(inside)
    return t[idx[0]], t[idx[-1]]


def test_zone_intervals_match_brute_force():
    """Test 1: analytic entry/exit vs 10 ms sampling on random geometries
    (agreement within one sampling step)."""
    rng = np.random.default_rng(1)
    n = 400
    rx = rng.uniform(-6, 6, n); ry = rng.uniform(-6, 6, n)
    speed = rng.uniform(0, 0.15, n)                 # NM/s (up to 540 kt)
    ang = rng.uniform(0, 2 * np.pi, n)
    vx, vy = speed * np.cos(ang), speed * np.sin(ang)
    dz = rng.uniform(-2500, 2500, n); vz = rng.uniform(-40, 40, n)
    vz[:20] = 0.0; vx[20:40] = 0.0; vy[20:40] = 0.0    # degenerate cases
    tau = 120.0
    t_in, t_out = pairs.zone_intervals(rx, ry, vx, vy, dz, vz, T1_H, T1_V,
                                       tau)
    n_inside = 0
    for i in range(n):
        b_in, b_out = _brute_force(rx[i], ry[i], vx[i], vy[i], dz[i], vz[i],
                                   T1_H, T1_V, tau)
        if np.isnan(b_in):
            # brute force can miss a sub-10 ms dip; the analytic one must
            # then be shorter than the sampling step
            assert np.isnan(t_in[i]) or (t_out[i] - t_in[i]) < 0.02, i
        else:
            n_inside += 1
            assert np.isfinite(t_in[i]), i
            assert abs(t_in[i] - b_in) <= 0.011, (i, t_in[i], b_in)
            assert abs(t_out[i] - b_out) <= 0.011, (i, t_out[i], b_out)
    assert n_inside > 50


def test_head_on_pass_between_samples_is_observed_t3(kbna, tmp_path):
    """Test 2: two aircraft at 300 kt each (600 kt closing), closest point
    at t = 30.5 s (between grid samples), lateral offset 300 ft, same
    height. At the samples (t = 30, 31) they are ~590 ft apart, outside
    the 500 ft T3 cylinder; the analytic interval within the step catches
    the 300 ft pass."""
    gs = 300.0
    d0 = 2 * gs / 3600.0 * 30.5          # so they meet at 30.5 s
    off = 300.0 / FT_PER_NM
    a = straight_track("aaaaaa", 3600, -d0 / 2, 0.0, 2000.0, gs, 90.0, 60)
    b = straight_track("bbbbbb", 3600, d0 / 2, off, 2000.0, gs, 270.0, 60)
    dg = day_grid_from([a, b], tmp_path)
    dp = pairs.compute_day(dg)
    enc = pairs.encounters(dp.pairs, config.TIERS, config.T_LOOKAHEAD_S,
                           10.0, 12000.0, 4000.0)
    t3 = enc[(enc.tier == "T3") & (enc.kind == "observed")]
    assert len(t3) == 1
    assert abs(t3["d_h_min_nm"].iat[0] * FT_PER_NM - 300.0) < 40.0
    assert t3["geometry_class"].iat[0] == "head_on"
    # at the integer samples the pair is outside T3: the within-step
    # refinement is what finds it
    ev = pairs.evaluate_tier(dp.pairs, T3_H, T3_V, 120.0)
    assert (ev["s_now"][ev["observed"]] > 1.0).all()


def _rwy(name):
    names, tx, ty, brg = geometry.runway_table()
    k = names.index(name)
    return tx[k], ty[k], brg[k]


def test_scenarios_tier_class_and_separation(kbna, tmp_path):
    """Test 3: head-on, crossing, in-trail, parallel-runway and
    runway-area encounters get the right tier, class and separation."""
    tx, ty, b = _rwy("20L")
    tx2, ty2, b2 = _rwy("20R")
    ub = np.radians(b)
    tracks = []
    # (a) crossing at 90 deg: both reach the crossing point after 100 s
    #     (5 NM at 180 kt), 0.4 NM apart at that moment, so the closest
    #     approach is 0.4 / sqrt(2) = 0.28 NM, 300 ft vertical: T1, T2
    tracks += [
        straight_track("c00001", 1000, -5.0, 3.0, 2500.0, 180.0, 90.0, 200),
        straight_track("c00002", 1000, 0.4, -2.0, 2800.0, 180.0, 0.0, 200),
    ]
    # (b) in trail on final to 20L: 2.5 NM apart, same glide (700 ft): T1
    d1, d2 = 6.0, 8.5
    for name, d in (("t00001", d1), ("t00002", d2)):
        tracks.append(straight_track(
            name, 3000, tx - d * np.sin(ub), ty - d * np.cos(ub),
            d * 318.0, 140.0, b, 120, vrate_fpm=-700.0))
    # (c) parallel runways 20L / 20R (centrelines 0.97 NM apart) side by
    #     side on 5 NM final, 200 ft apart: T1 only
    tracks += [
        straight_track("p00001", 5000, tx - 5 * np.sin(ub), ty - 5 * np.cos(ub),
                       1600.0, 140.0, b, 100, vrate_fpm=-700.0),
        straight_track("p00002", 5000, tx2 - 5 * np.sin(ub),
                       ty2 - 5 * np.cos(ub), 1800.0, 140.0, b2, 100,
                       vrate_fpm=-700.0),
    ]
    # (d) head-on at 4,000 ft, 0.3 NM lateral offset, 50 ft vertical:
    #     T1, T2 and (500 ft x 100 ft) T3? 0.3 NM = 1823 ft > 500 ft: T2
    tracks += [
        straight_track("h00001", 7000, -6.0, -6.0, 4000.0, 200.0, 45.0, 150),
        straight_track("h00002", 7000, 4.0 + 0.3 * np.cos(np.radians(45)),
                       4.0 - 0.3 * np.sin(np.radians(45)), 4050.0, 200.0,
                       225.0, 150),
    ]
    # (e) runway area: one on short final to 20L at 300 ft, one climbing
    #     out of 20L just past the threshold at 200 ft (same runway end,
    #     both within 1 NM of the threshold, below 500 ft)
    tracks += [
        straight_track("r00001", 9000, tx - 0.9 * np.sin(ub),
                       ty - 0.9 * np.cos(ub), 300.0, 130.0, b, 25,
                       vrate_fpm=-600.0),
        straight_track("r00002", 9000, tx + 0.2 * np.sin(ub),
                       ty + 0.2 * np.cos(ub), 150.0, 140.0, b, 25,
                       vrate_fpm=1500.0),
    ]
    dg = day_grid_from(tracks, tmp_path)
    dp = pairs.compute_day(dg)
    enc = pairs.encounters(dp.pairs, config.TIERS, config.T_LOOKAHEAD_S,
                           10.0, 12000.0, 4001.0)
    legs = dg.legs.set_index("leg")["icao24"]
    enc["ac_a"] = legs.loc[enc["a"]].to_numpy()
    enc["ac_b"] = legs.loc[enc["b"]].to_numpy()
    obs = enc[enc.kind == "observed"]

    def get(prefix, tier):
        sel = obs[(obs.ac_a.str.startswith(prefix)) & (obs.tier == tier)]
        assert len(sel) >= 1, (prefix, tier, obs[["ac_a", "tier"]])
        return sel.sort_values("s_min").iloc[0]

    c = get("c", "T1")
    assert c.geometry_class == "crossing"
    assert abs(c.d_h_min_nm - 0.4 / np.sqrt(2)) < 0.03
    assert abs(abs(c.dz_at_min_ft) - 300) < 40
    assert len(obs[(obs.ac_a == "c00001") & (obs.tier == "T2")]) == 1
    assert len(obs[(obs.ac_a == "c00001") & (obs.tier == "T3")]) == 0

    t = get("t", "T1")
    assert t.geometry_class == "in_trail_same_runway", t
    assert abs(t.d_h_min_nm - 2.5) < 0.1
    assert len(obs[(obs.ac_a == "t00001") & (obs.tier == "T2")]) == 0

    p = get("p", "T1")
    assert p.geometry_class == "parallel_runways", p
    assert abs(p.d_h_min_nm - 0.97) < 0.1
    assert len(obs[(obs.ac_a == "p00001") & (obs.tier == "T2")]) == 0

    h = get("h", "T1")
    assert h.geometry_class == "head_on"
    assert abs(h.d_h_min_nm - 0.3) < 0.03 and abs(abs(h.dz_at_min_ft) - 50) < 20
    assert len(obs[(obs.ac_a == "h00001") & (obs.tier == "T2")]) == 1
    assert len(obs[(obs.ac_a == "h00001") & (obs.tier == "T3")]) == 0

    r = get("r", "T1")
    assert r.geometry_class == "runway_area", r
    # predicted encounters exist for the converging pairs
    pred = enc[enc.kind == "predicted"]
    assert set(pred[pred.tier == "T1"]["ac_a"]) >= {"c00001", "h00001"}
    assert (pred["t_entry_s"] > 0).all()


def test_pruning_never_drops_a_flagged_pair(synth_dataset, kbna, monkeypatch):
    """Test 4: with an active pruning radius (max speed set just above
    the fastest synthetic aircraft), the flagged pair-seconds are exactly
    those of the unpruned computation."""
    dg = loading.build_day_grid(sorted(synth_dataset.glob("*.parquet"))[0])
    monkeypatch.setattr(config, "PRUNE_VMAX_KT", 130.0)  # prune actively
    monkeypatch.setattr(config, "T_LOOKAHEAD_SENSITIVITY_S", (60.0,))
    monkeypatch.setattr(config, "T_LOOKAHEAD_S", 60.0)
    pruned = pairs.compute_day(dg, prune=True)
    full = pairs.compute_day(dg, prune=False)
    assert pruned.n_pruned > 0
    key = ["t", "a", "b"]
    a = pruned.pairs[key].sort_values(key).reset_index(drop=True)
    b = full.pairs[key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_prune_radius_formula():
    assert pairs.prune_radius_nm(3.0, 120.0) == pytest.approx(
        3.0 + 2 * 350 / 3600 * 120)
