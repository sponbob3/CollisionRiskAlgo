"""
Stage 2: pairwise geometry, zone-entry intervals, pruning and encounter
episodes (FRAMEWORK.md sections 5 and 6).

The per-day computation (compute_day) runs ONCE in the widest volume,
longest lookahead and loosest T1 thresholds any sweep can ask for, and
stores every flagged pair-second with its relative state (r, v, dz, vz)
in the per-day cache. Everything downstream - tiers, thresholds, radius,
ceiling, lookahead, observed vs predicted - is a filter on those rows and
re-evaluates the same analytic intervals, so sensitivity sweeps never
repeat the pairwise work (section 12).

Zone-entry intervals (section 6.2): for relative position r, relative
velocity v, vertical separation dz and relative vertical rate vz, the
pair is inside the (H, V) zone while |r + v t| < H and |dz + vz t| < V.
The first is a quadratic in t, the second linear; the pair is inside
during the intersection of the two solution intervals, clipped to
[0, tau_max]. tau_max = the grid step for "observed" (catches passes
between two samples) and the lookahead for "predicted".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from . import geometry
from .loading import DayGrid

EPS_V = 1e-9        # relative speed below this counts as zero (NM/s, ft/s)
INF = np.inf


# ------------------------------------------------- analytic intervals ----

def zone_intervals(rx, ry, vx, vy, dz, vz, H: float, V: float,
                   tau_max: float):
    """Entry and exit times (s) of the (H, V) zone within [0, tau_max]
    for arrays of pair states. Returns (t_in, t_out); NaN where the pair
    is never inside during the span. r in NM, v in NM/s, dz in ft,
    vz in ft/s."""
    rx = np.asarray(rx, float); ry = np.asarray(ry, float)
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    dz = np.asarray(dz, float); vz = np.asarray(vz, float)

    # horizontal: a t^2 + b t + c < 0
    a = vx * vx + vy * vy
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry - H * H
    h_in = np.full(a.shape, np.nan)
    h_out = np.full(a.shape, np.nan)
    still = a < EPS_V * EPS_V
    inside_now = c < 0
    h_in[still & inside_now] = -INF
    h_out[still & inside_now] = INF
    moving = ~still
    disc = b * b - 4.0 * a * c
    ok = moving & (disc > 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        h_in[ok] = ((-b - sq) / (2.0 * a))[ok]
        h_out[ok] = ((-b + sq) / (2.0 * a))[ok]

    # vertical: |dz + vz t| < V
    v_in = np.full(a.shape, np.nan)
    v_out = np.full(a.shape, np.nan)
    vstill = np.abs(vz) < EPS_V
    vin_now = np.abs(dz) < V
    v_in[vstill & vin_now] = -INF
    v_out[vstill & vin_now] = INF
    vm = ~vstill
    with np.errstate(divide="ignore", invalid="ignore"):
        t3 = (-V - dz) / vz
        t4 = (V - dz) / vz
    v_in[vm] = np.minimum(t3, t4)[vm]
    v_out[vm] = np.maximum(t3, t4)[vm]

    t_in = np.maximum(np.maximum(h_in, v_in), 0.0)
    t_out = np.minimum(np.minimum(h_out, v_out), tau_max)
    # (-inf / +inf bounds mean "inside for the whole span": valid)
    valid = ~np.isnan(h_in) & ~np.isnan(v_in) & (t_in < t_out)
    t_in[~valid] = np.nan
    t_out[~valid] = np.nan
    return t_in, t_out


def closest_approach(rx, ry, vx, vy, dz, vz, tau_max: float):
    """Time of horizontal closest approach within [0, tau_max], the
    horizontal distance then, and the vertical separation then."""
    rx = np.asarray(rx, float); ry = np.asarray(ry, float)
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    a = vx * vx + vy * vy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(a > EPS_V * EPS_V, -(rx * vx + ry * vy) / a, 0.0)
    t = np.clip(t, 0.0, tau_max)
    d = np.hypot(rx + vx * t, ry + vy * t)
    dzt = np.asarray(dz, float) + np.asarray(vz, float) * t
    return t, d, dzt


def min_distance_in_step(rx, ry, vx, vy, dz, vz, step: float):
    """Minimum horizontal distance within one grid step and the vertical
    separation at that moment (refines the sample-time separation so a
    pass between two samples is measured at its true closest point)."""
    return closest_approach(rx, ry, vx, vy, dz, vz, step)


def prune_radius_nm(H: float, lookahead_s: float) -> float:
    """Pairs further apart than this cannot enter the H zone within the
    lookahead even at the maximum closing speed (section 6.1)."""
    return H + 2.0 * config.PRUNE_VMAX_KT / 3600.0 * lookahead_s


# ------------------------------------------------ widest computation ----

def loosest_t1() -> tuple[float, float]:
    f = max(config.TIER_SENSITIVITY_FACTORS)
    H, V = config.TIERS["T1"]
    return H * f, V * f


def max_lookahead_s() -> float:
    return max(config.T_LOOKAHEAD_S, *config.T_LOOKAHEAD_SENSITIVITY_S)


PAIR_COLUMNS = ["t", "a", "b", "rx", "ry", "vx", "vy", "dz", "vz", "zref",
                "xa", "ya", "ha", "xb", "yb", "hb", "ra", "rb",
                "tra", "trb", "pha", "phb", "ia", "ib"]


@dataclass
class DayPairs:
    pairs: pd.DataFrame        # flagged pair-seconds (PAIR_COLUMNS)
    presence: pd.DataFrame     # airborne aircraft-seconds in the volume
    n_candidates: int          # pair-seconds before pruning
    n_pruned: int              # dropped by the pruning radius


def compute_day(dg: DayGrid, radius_nm: float | None = None,
                ceiling_ft: float | None = None, H: float | None = None,
                V: float | None = None, lookahead_s: float | None = None,
                prune: bool = True) -> DayPairs:
    """All flagged pair-seconds of one day in the computation volume.

    A pair-second is flagged when the pair is inside the (H, V) zone
    during the grid step (observed) or enters it within the lookahead
    (predicted). Defaults: widest radius, computation ceiling, loosest T1
    thresholds and longest lookahead (section 12)."""
    from . import airspace
    radius_nm = radius_nm or airspace.computation_radius_nm()
    ceiling_ft = ceiling_ft or airspace.computation_ceiling_ft_agl()
    if H is None or V is None:
        H, V = loosest_t1()
    lookahead_s = lookahead_s or max_lookahead_s()
    step = config.GRID_STEP_S

    g = dg.grid
    # presence: every airborne aircraft-second below the computation
    # ceiling out to the load radius (so radius containment can be
    # measured); pairs are formed only inside the computation radius
    air = g[g["airborne"] & (g["h"] <= ceiling_ft)]
    phase_all = geometry.flight_phase(air["x"], air["y"], air["track"],
                                      air["h"], air["vrate"])
    presence = pd.DataFrame({
        "t": air["t"].to_numpy(),
        "leg": air["leg"].to_numpy(),
        "r": air["r"].to_numpy(),
        "h": air["h"].to_numpy(),
        "interpolated": air["interpolated"].to_numpy(),
        "interp_long": air["interp_long"].to_numpy(),
        "phase": phase_all,
    })
    in_radius = (air["r"] <= radius_nm).to_numpy()
    cols = ["t", "leg", "x", "y", "z_baro", "z_geo", "h", "r", "vx", "vy",
            "vz", "track"]
    inv = air[cols][in_radius].copy()
    inv["phase"] = phase_all[in_radius]
    inv["interpolated"] = presence["interpolated"].to_numpy()[in_radius]
    inv = inv.sort_values(["t", "leg"]).reset_index(drop=True)

    d_prune = prune_radius_nm(H, lookahead_s)
    out, n_cand, n_pruned = [], 0, 0
    t_min, t_max = int(inv["t"].min()), int(inv["t"].max())
    for c0 in range(t_min, t_max + 1, config.PAIR_CHUNK_S):
        chunk = inv[(inv["t"] >= c0) & (inv["t"] < c0 + config.PAIR_CHUNK_S)]
        if len(chunk) < 2:
            continue
        m = chunk.merge(chunk, on="t", suffixes=("_a", "_b"))
        m = m[m["leg_a"] < m["leg_b"]]
        if m.empty:
            continue
        n_cand += len(m)
        rx = (m["x_b"] - m["x_a"]).to_numpy(float)
        ry = (m["y_b"] - m["y_a"]).to_numpy(float)
        if prune:
            keep = np.hypot(rx, ry) <= d_prune
            n_pruned += int((~keep).sum())
            m = m[keep]
            rx, ry = rx[keep], ry[keep]
        vx = (m["vx_b"] - m["vx_a"]).to_numpy(float)
        vy = (m["vy_b"] - m["vy_a"]).to_numpy(float)
        vz = (m["vz_b"] - m["vz_a"]).to_numpy(float)
        # vertical separation: barometric when both have it (the reference
        # ATC separates on; the shared pressure bias cancels), else
        # geometric for both - never mixed within a pair (section 4.2)
        both_baro = (m["z_baro_a"].notna() & m["z_baro_b"].notna()).to_numpy()
        dz = np.where(both_baro,
                      (m["z_baro_b"] - m["z_baro_a"]).to_numpy(float),
                      (m["z_geo_b"] - m["z_geo_a"]).to_numpy(float))
        t_obs, _ = zone_intervals(rx, ry, vx, vy, dz, vz, H, V, step)
        t_pred, _ = zone_intervals(rx, ry, vx, vy, dz, vz, H, V, lookahead_s)
        flagged = np.isfinite(t_obs) | np.isfinite(t_pred)
        flagged &= np.isfinite(dz)
        if not flagged.any():
            continue
        f = m[flagged]
        out.append(pd.DataFrame({
            "t": f["t"].to_numpy(np.int64),
            "a": f["leg_a"].to_numpy(np.int32),
            "b": f["leg_b"].to_numpy(np.int32),
            "rx": rx[flagged].astype(np.float32),
            "ry": ry[flagged].astype(np.float32),
            "vx": vx[flagged].astype(np.float32),
            "vy": vy[flagged].astype(np.float32),
            "dz": dz[flagged].astype(np.float32),
            "vz": vz[flagged].astype(np.float32),
            "zref": both_baro[flagged].astype(np.int8),
            "xa": f["x_a"].to_numpy(np.float32),
            "ya": f["y_a"].to_numpy(np.float32),
            "ha": f["h_a"].to_numpy(np.float32),
            "xb": f["x_b"].to_numpy(np.float32),
            "yb": f["y_b"].to_numpy(np.float32),
            "hb": f["h_b"].to_numpy(np.float32),
            "ra": f["r_a"].to_numpy(np.float32),
            "rb": f["r_b"].to_numpy(np.float32),
            "tra": f["track_a"].to_numpy(np.float32),
            "trb": f["track_b"].to_numpy(np.float32),
            "pha": f["phase_a"].to_numpy(np.int16),
            "phb": f["phase_b"].to_numpy(np.int16),
            "ia": f["interpolated_a"].to_numpy(bool),
            "ib": f["interpolated_b"].to_numpy(bool),
        }))
    pairs = (pd.concat(out, ignore_index=True) if out
             else pd.DataFrame({c: pd.Series(dtype=float)
                                for c in PAIR_COLUMNS}))
    return DayPairs(pairs=pairs, presence=presence, n_candidates=n_cand,
                    n_pruned=n_pruned)


# ---------------------------------------------------- tier evaluation ----

def evaluate_tier(pairs: pd.DataFrame, H: float, V: float,
                  lookahead_s: float) -> pd.DataFrame:
    """Re-evaluate cached pair-seconds against one (H, V) zone: observed
    (inside during the grid step) and predicted (enters within the
    lookahead, not yet inside) flags, with the separations that describe
    each. Returns a frame aligned with `pairs`."""
    step = config.GRID_STEP_S
    rx, ry = pairs["rx"].to_numpy(float), pairs["ry"].to_numpy(float)
    vx, vy = pairs["vx"].to_numpy(float), pairs["vy"].to_numpy(float)
    dz, vz = pairs["dz"].to_numpy(float), pairs["vz"].to_numpy(float)
    t_in_obs, _ = zone_intervals(rx, ry, vx, vy, dz, vz, H, V, step)
    t_in_pred, _ = zone_intervals(rx, ry, vx, vy, dz, vz, H, V, lookahead_s)
    observed = np.isfinite(t_in_obs)
    predicted = np.isfinite(t_in_pred) & ~observed
    # observed: separation at the closest point within the step
    _, d_step, dz_step = min_distance_in_step(rx, ry, vx, vy, dz, vz, step)
    d_now = np.hypot(rx, ry)
    s_now = np.maximum(d_now / H, np.abs(dz) / V)
    s_step = np.maximum(d_step / H, np.abs(dz_step) / V)
    # predicted: closest approach within the lookahead
    t_cpa, d_cpa, dz_cpa = closest_approach(rx, ry, vx, vy, dz, vz,
                                            lookahead_s)
    s_cpa = np.maximum(d_cpa / H, np.abs(dz_cpa) / V)
    return pd.DataFrame({
        "observed": observed,
        "predicted": predicted,
        "s_now": s_now,
        "d_h_step": d_step, "dz_step": dz_step, "s_step": s_step,
        "t_entry": np.where(predicted, t_in_pred, np.nan),
        "t_cpa": t_cpa, "d_h_cpa": d_cpa, "dz_cpa": dz_cpa, "s_cpa": s_cpa,
    }, index=pairs.index)


def volume_mask(pairs: pd.DataFrame, radius_nm: float,
                ceiling_ft: float) -> np.ndarray:
    """Both aircraft inside the given cylinder."""
    return ((pairs["ra"] <= radius_nm) & (pairs["rb"] <= radius_nm)
            & (pairs["ha"] <= ceiling_ft) & (pairs["hb"] <= ceiling_ft)
            ).to_numpy()


def episodes(pairs: pd.DataFrame, ev: pd.DataFrame, kind: str,
             H: float, V: float, tier: str, nominal_ceiling_ft: float,
             merge_gap_s: float | None = None) -> pd.DataFrame:
    """Encounter episodes (section 6.3) for one tier and kind
    (observed | predicted): consecutive flagged seconds of a pair, with
    gaps up to EPISODE_MERGE_GAP_S merged. One row per episode."""
    merge_gap_s = (config.EPISODE_MERGE_GAP_S if merge_gap_s is None
                   else merge_gap_s)
    flag = ev[kind].to_numpy()
    if not flag.any():
        return pd.DataFrame()
    p = pairs[flag]
    e = ev[flag]
    order = np.lexsort((p["t"].to_numpy(), p["b"].to_numpy(),
                        p["a"].to_numpy()))
    p = p.iloc[order]
    e = e.iloc[order]
    t = p["t"].to_numpy()
    a = p["a"].to_numpy(); b = p["b"].to_numpy()
    new = np.ones(len(p), dtype=bool)
    new[1:] = ((a[1:] != a[:-1]) | (b[1:] != b[:-1])
               | (t[1:] - t[:-1] > merge_gap_s + 1))
    ep = np.cumsum(new) - 1

    if kind == "observed":
        s = e["s_step"].to_numpy()
        d_h = e["d_h_step"].to_numpy()
        dzv = e["dz_step"].to_numpy()
    else:
        s = e["s_cpa"].to_numpy()
        d_h = e["d_h_cpa"].to_numpy()
        dzv = e["dz_cpa"].to_numpy()
    # sample of minimum separation within each episode: minimum s, ties
    # (a stretch where the vertical term dominates s) broken by the
    # horizontal distance so the point is the true closest approach
    key = s + 1e-6 * d_h / H
    df = pd.DataFrame({"ep": ep, "key": key})
    imin = df.groupby("ep")["key"].idxmin().to_numpy()
    first = np.flatnonzero(new)
    last = np.append(first[1:] - 1, len(p) - 1)
    pm = p.iloc[imin]
    em = e.iloc[imin]
    h_max = np.maximum(pm["ha"].to_numpy(), pm["hb"].to_numpy())
    out = pd.DataFrame({
        "a": pm["a"].to_numpy(), "b": pm["b"].to_numpy(),
        "tier": tier, "kind": kind,
        "t_start": t[first], "t_end": t[last],
        "duration_s": t[last] - t[first] + 1,
        "t_min": pm["t"].to_numpy(),
        "d_h_min_nm": d_h[imin], "dz_at_min_ft": dzv[imin],
        "s_min": s[imin],
        "t_entry_s": (em["t_entry"].to_numpy() if kind == "predicted"
                      else np.nan),
        "xa": pm["xa"].to_numpy(), "ya": pm["ya"].to_numpy(),
        "ha": pm["ha"].to_numpy(),
        "xb": pm["xb"].to_numpy(), "yb": pm["yb"].to_numpy(),
        "hb": pm["hb"].to_numpy(),
        "h_max_ft": h_max,
        "inside_ceiling": h_max <= nominal_ceiling_ft,
        "zref": np.where(pm["zref"].to_numpy() == 1, "baro", "geo"),
        "phase_a": geometry.phase_label(pm["pha"]),
        "phase_b": geometry.phase_label(pm["phb"]),
        "geometry_class": geometry.geometry_class(
            pm["pha"], pm["phb"], pm["tra"], pm["trb"],
            pm["xa"], pm["ya"], pm["ha"], pm["xb"], pm["yb"], pm["hb"]),
        "interpolated": (pm["ia"] | pm["ib"]).to_numpy(),
    })
    return out.reset_index(drop=True)


ENCOUNTER_COLUMNS = [
    "a", "b", "tier", "kind", "t_start", "t_end", "duration_s", "t_min",
    "d_h_min_nm", "dz_at_min_ft", "s_min", "t_entry_s", "xa", "ya", "ha",
    "xb", "yb", "hb", "h_max_ft", "inside_ceiling", "zref", "phase_a",
    "phase_b", "geometry_class", "interpolated"]


def empty_encounters() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in
                         ENCOUNTER_COLUMNS})


def encounters(pairs: pd.DataFrame, tiers: dict, lookahead_s: float,
               radius_nm: float, ceiling_ft: float, nominal_ceiling_ft: float,
               merge_gap_s: float | None = None) -> pd.DataFrame:
    """All encounter episodes of a day for every tier, observed and
    predicted, in the extended volume (radius, ceiling_ft) and tagged
    inside/outside the nominal ceiling."""
    if pairs.empty:
        return empty_encounters()
    p = pairs[volume_mask(pairs, radius_nm, ceiling_ft)]
    if p.empty:
        return empty_encounters()
    frames = []
    for tier, (H, V) in tiers.items():
        ev = evaluate_tier(p, H, V, lookahead_s)
        for kind in ("observed", "predicted"):
            frames.append(episodes(p, ev, kind, H, V, tier,
                                   nominal_ceiling_ft, merge_gap_s))
    frames = [f for f in frames if len(f)]
    if not frames:
        return empty_encounters()
    return pd.concat(frames, ignore_index=True)
