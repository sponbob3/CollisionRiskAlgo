"""
Probability model (FRAMEWORK.md section 7): the straight-line prediction
error measured from the data (7.1), the Paielli & Erzberger conflict
probability (7.2), the expected-conflicts metric (7.3) and the
calibration outputs that show whether the model works (7.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config
from . import geometry
from .loading import DayGrid

PHASES = ("arrival", "departure", "other")


def phase_group(code) -> np.ndarray:
    """arrival / departure / other from a phase code."""
    code = np.asarray(code, int)
    out = np.full(len(code), "other", dtype=object)
    out[(code > 0) & (code < geometry.DEPARTURE_BASE + 1)] = "arrival"
    out[code >= geometry.DEPARTURE_BASE + 1] = "departure"
    return out


def taus() -> np.ndarray:
    step = config.ERROR_MODEL_TAU_STEP_S
    top = max(config.T_LOOKAHEAD_S, *config.T_LOOKAHEAD_SENSITIVITY_S)
    return np.arange(step, top + step / 2, step)


# ------------------------------------------------- 7.1 error samples ----

def sample_prediction_errors(dg: DayGrid, presence: pd.DataFrame,
                             seed: int) -> pd.DataFrame:
    """Dead-reckoning error samples for one day: for random airborne
    aircraft-seconds in the volume, predict the position tau seconds
    ahead by straight-line extrapolation and compare with the actual
    position, decomposed into along-track, cross-track (NM) and vertical
    (ft) components in the aircraft's track frame at the prediction
    time. One row per (sample, tau) where the actual position exists."""
    rng = np.random.default_rng(seed)
    g = dg.grid
    if presence.empty:
        return pd.DataFrame(columns=["phase", "tau", "e_along", "e_cross",
                                     "e_z"])
    from . import airspace
    presence = presence[presence["r"] <= airspace.computation_radius_nm()]
    if presence.empty:
        return pd.DataFrame(columns=["phase", "tau", "e_along", "e_cross",
                                     "e_z"])
    n = min(config.ERROR_MODEL_SAMPLES_PER_DAY, len(presence))
    pick = presence.iloc[rng.choice(len(presence), n, replace=False)]
    # index the grid by (leg, t) for the lookups
    key = g["leg"].to_numpy().astype(np.int64) * (1 << 32) + g["t"].to_numpy()
    order = np.argsort(key)
    key_sorted = key[order]
    rows = []
    x = g["x"].to_numpy(float); y = g["y"].to_numpy(float)
    z = g["h"].to_numpy(float)
    vx = g["vx"].to_numpy(float); vy = g["vy"].to_numpy(float)
    vz = g["vz"].to_numpy(float)
    trk = g["track"].to_numpy(float)
    leg0 = pick["leg"].to_numpy().astype(np.int64)
    t0 = pick["t"].to_numpy().astype(np.int64)
    ph = phase_group(pick["phase"])
    k0 = leg0 * (1 << 32) + t0
    i0 = order[np.searchsorted(key_sorted, k0)]
    for tau in taus():
        k1 = k0 + int(tau)
        pos = np.searchsorted(key_sorted, k1)
        ok = pos < len(key_sorted)
        pos = np.minimum(pos, len(key_sorted) - 1)
        ok &= key_sorted[pos] == k1
        i1 = order[pos]
        px = x[i0] + vx[i0] * tau
        py = y[i0] + vy[i0] * tau
        pz = z[i0] + vz[i0] * tau
        ex, ey, ez = x[i1] - px, y[i1] - py, z[i1] - pz
        b = np.radians(trk[i0])
        along = ex * np.sin(b) + ey * np.cos(b)
        cross = ex * np.cos(b) - ey * np.sin(b)
        good = ok & np.isfinite(along) & np.isfinite(ez)
        rows.append(pd.DataFrame({
            "phase": ph[good], "tau": np.float32(tau),
            "e_along": along[good].astype(np.float32),
            "e_cross": cross[good].astype(np.float32),
            "e_z": ez[good].astype(np.float32),
        }))
    return pd.concat(rows, ignore_index=True)


def fit_error_model(samples: pd.DataFrame) -> pd.DataFrame:
    """Robust (MAD-based) sigma_along, sigma_cross, sigma_z per phase and
    tau, with sample counts (error_model.csv)."""
    def mad_sigma(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) < 20:
            return np.nan
        return 1.4826 * np.median(np.abs(v - np.median(v)))

    rows = []
    for (phase, tau), grp in samples.groupby(["phase", "tau"]):
        rows.append({"phase": phase, "tau_s": float(tau), "n": len(grp),
                     "sigma_along_nm": mad_sigma(grp["e_along"]),
                     "sigma_cross_nm": mad_sigma(grp["e_cross"]),
                     "sigma_z_ft": mad_sigma(grp["e_z"])})
    # pooled "all" rows for phases with too few samples
    for tau, grp in samples.groupby("tau"):
        rows.append({"phase": "all", "tau_s": float(tau), "n": len(grp),
                     "sigma_along_nm": mad_sigma(grp["e_along"]),
                     "sigma_cross_nm": mad_sigma(grp["e_cross"]),
                     "sigma_z_ft": mad_sigma(grp["e_z"])})
    return pd.DataFrame(rows).sort_values(["phase", "tau_s"]).reset_index(
        drop=True)


class ErrorModel:
    """Interpolates sigma(tau) per phase from error_model.csv, falling
    back to the pooled model where a phase has too few samples, and to a
    position-noise floor at tau -> 0."""

    def __init__(self, table: pd.DataFrame):
        self.table = table
        self.curves = {}
        for phase, grp in table.groupby("phase"):
            grp = grp.dropna().sort_values("tau_s")
            if len(grp) >= 2:
                self.curves[phase] = grp
        if "all" not in self.curves:
            raise ValueError("error model has no usable rows")

    def _curve(self, phase):
        return self.curves.get(phase, self.curves["all"])

    def sigma(self, phase, tau) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = np.asarray(phase, dtype=object)
        tau = np.asarray(tau, float)
        sa = np.empty(len(tau)); sc = np.empty(len(tau)); sz = np.empty(len(tau))
        for ph in np.unique(phase):
            m = phase == ph
            c = self._curve(ph)
            t = np.concatenate([[0.0], c["tau_s"].to_numpy()])
            for arr, col, floor in ((sa, "sigma_along_nm",
                                     config.ERROR_MODEL_SIGMA0_NM),
                                    (sc, "sigma_cross_nm",
                                     config.ERROR_MODEL_SIGMA0_NM),
                                    (sz, "sigma_z_ft",
                                     config.ERROR_MODEL_SIGMA0_FT)):
                v = np.concatenate([[floor], c[col].to_numpy()])
                arr[m] = np.maximum(np.interp(tau[m], t, v), floor)
        return sa, sc, sz


# ------------------------------------------ 7.2 conflict probability ----

def conflict_probability(rx, ry, vx, vy, dz, vz, t_cpa, track_a, track_b,
                         sig_a, sig_b, H: float, V: float) -> np.ndarray:
    """Paielli & Erzberger conflict probability for pair states.

    rx, ry, vx, vy: relative position (NM) and velocity (NM/s);
    dz, vz: vertical separation (ft) and rate (ft/s); t_cpa: time to the
    predicted horizontal closest approach (s); track_a/b: tracks (deg);
    sig_a, sig_b: (sigma_along, sigma_cross, sigma_z) of each aircraft at
    t_cpa. The relative position error at closest approach is Gaussian
    with covariance = sum of both aircraft's covariances, each rotated
    from its along/cross frame to the plane. The covariance is whitened
    (a linear transform making the error circular), the relative track
    becomes a straight line through the transformed space, and the
    conflict circle becomes an ellipse whose extent perpendicular to that
    line is its support half-width w. The line passes through the ellipse
    when the (unit-variance) miss distance along the perpendicular lies
    within +/- w, which is a difference of two normal CDFs. The vertical
    probability P(|dz_cpa + e_z| < V) multiplies in.
    """
    rx = np.asarray(rx, float); ry = np.asarray(ry, float)
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    dz = np.asarray(dz, float); vz = np.asarray(vz, float)
    t_cpa = np.asarray(t_cpa, float)
    n = len(rx)
    sa_a, sc_a, sz_a = (np.asarray(s, float) for s in sig_a)
    sa_b, sc_b, sz_b = (np.asarray(s, float) for s in sig_b)

    def cov(track, sa, sc):
        b = np.radians(np.asarray(track, float))
        # along-track unit vector (sin b, cos b); cross-track (cos b, -sin b)
        ux, uy = np.sin(b), np.cos(b)
        cx, cy = np.cos(b), -np.sin(b)
        sxx = sa ** 2 * ux * ux + sc ** 2 * cx * cx
        syy = sa ** 2 * uy * uy + sc ** 2 * cy * cy
        sxy = sa ** 2 * ux * uy + sc ** 2 * cx * cy
        return sxx, syy, sxy

    axx, ayy, axy = cov(track_a, sa_a, sc_a)
    bxx, byy, bxy = cov(track_b, sa_b, sc_b)
    sxx, syy, sxy = axx + bxx, ayy + byy, axy + bxy
    # whitening transform L^-1 with Sigma = L L^T (Cholesky)
    l11 = np.sqrt(np.maximum(sxx, 1e-12))
    l21 = sxy / l11
    l22 = np.sqrt(np.maximum(syy - l21 ** 2, 1e-12))
    # relative position at closest approach, then transformed
    px = rx + vx * t_cpa
    py = ry + vy * t_cpa
    tx = px / l11
    ty = (py - l21 * tx) / l22
    tvx = vx / l11
    tvy = (vy - l21 * tvx) / l22
    speed = np.hypot(tvx, tvy)
    still = speed < 1e-12
    # perpendicular to the transformed relative track
    nx = np.where(still, 1.0, -tvy / np.where(still, 1.0, speed))
    ny = np.where(still, 0.0, tvx / np.where(still, 1.0, speed))
    miss = tx * nx + ty * ny
    # support half-width of the transformed circle (the ellipse
    # {q : |L q| <= H}) along n: H |L^-T n|, with
    # L^-T = [[1/l11, -l21/(l11 l22)], [0, 1/l22]]
    def support(nx_, ny_):
        ax_ = nx_ / l11 - l21 * ny_ / (l11 * l22)
        ay_ = ny_ / l22
        return H * np.hypot(ax_, ay_)

    w = support(nx, ny)
    p_h = norm.cdf(w - miss) - norm.cdf(-w - miss)
    # a stationary relative track: the pair stays where it is, so the
    # question is whether the Gaussian relative position lies inside the
    # disc - a two-dimensional integral, done numerically
    if still.any():
        p_h = np.where(still, _disc_probability(px, py, sxx, syy, sxy, H),
                       p_h)
    sz = np.sqrt(sz_a ** 2 + sz_b ** 2)
    dz_cpa = dz + vz * t_cpa
    p_z = norm.cdf((V - dz_cpa) / sz) - norm.cdf((-V - dz_cpa) / sz)
    return np.clip(p_h * p_z, 0.0, 1.0)


def _disc_probability(px, py, sxx, syy, sxy, H: float, n: int = 48):
    """P(|p + e| < H) for e ~ N(0, Sigma) by Gauss-Legendre quadrature
    over the disc in polar coordinates (vectorised over rows)."""
    from scipy.special import roots_legendre
    xr, wr = roots_legendre(n)
    r = H * (xr + 1) / 2; w_r = wr * H / 2
    th = np.pi * (xr + 1); w_t = wr * np.pi
    R, T = np.meshgrid(r, th, indexing="ij")
    W = np.outer(w_r, w_t) * R
    X, Y = R * np.cos(T), R * np.sin(T)
    px = np.asarray(px, float)[:, None, None]; py = np.asarray(py, float)[:, None, None]
    sxx = np.asarray(sxx, float)[:, None, None]
    syy = np.asarray(syy, float)[:, None, None]
    sxy = np.asarray(sxy, float)[:, None, None]
    det = sxx * syy - sxy ** 2
    dx, dy = X[None] - px, Y[None] - py
    q = (syy * dx * dx - 2 * sxy * dx * dy + sxx * dy * dy) / det
    dens = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
    return np.sum(dens * W[None], axis=(1, 2))


def conflict_probability_mc(rx, ry, vx, vy, dz, vz, t_cpa, track_a,
                            track_b, sig_a, sig_b, H, V, draws, seed=0):
    """Monte Carlo reference for one geometry: perturb the relative
    position at closest approach with the same Gaussian error and count
    how often the shifted straight relative track passes within H while
    the vertical separation at closest approach is within V."""
    rng = np.random.default_rng(seed)
    b_a = np.radians(track_a); b_b = np.radians(track_b)
    ea = rng.normal(size=(draws, 2)) * np.array([sig_a[0], sig_a[1]])
    eb = rng.normal(size=(draws, 2)) * np.array([sig_b[0], sig_b[1]])

    def to_plane(e, b):
        return (e[:, 0] * np.sin(b) + e[:, 1] * np.cos(b),
                e[:, 0] * np.cos(b) - e[:, 1] * np.sin(b))

    ax, ay = to_plane(ea, b_a)
    bx, by = to_plane(eb, b_b)
    ex, ey = bx - ax, by - ay
    px = rx + vx * t_cpa + ex
    py = ry + vy * t_cpa + ey
    speed = np.hypot(vx, vy)
    if speed < 1e-12:
        miss = np.hypot(px, py)
    else:
        miss = np.abs(px * (-vy / speed) + py * (vx / speed))
    ez = rng.normal(size=draws) * np.sqrt(sig_a[2] ** 2 + sig_b[2] ** 2)
    hit = (miss < H) & (np.abs(dz + vz * t_cpa + ez) < V)
    return float(hit.mean())


# ------------------------------------------------- 7.4 calibration ----

def calibration(prob: np.ndarray, hit: np.ndarray,
                n_bins: int | None = None) -> dict:
    """Reliability table, Brier score, Brier skill score against a
    constant-rate reference, and expected calibration error."""
    n_bins = n_bins or config.PROB_RELIABILITY_BINS
    prob = np.asarray(prob, float); hit = np.asarray(hit, bool)
    if len(prob) == 0:
        return {"table": pd.DataFrame(), "brier": np.nan, "bss": np.nan,
                "ece": np.nan, "n": 0, "base_rate": np.nan}
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1],
                         "n": 0, "mean_pred": np.nan, "obs_frac": np.nan})
            continue
        mp, of = prob[m].mean(), hit[m].mean()
        ece += m.mean() * abs(mp - of)
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1],
                     "n": int(m.sum()), "mean_pred": mp, "obs_frac": of})
    base = hit.mean()
    brier = float(np.mean((prob - hit) ** 2))
    ref = float(np.mean((base - hit) ** 2))
    bss = 1.0 - brier / ref if ref > 0 else np.nan
    return {"table": pd.DataFrame(rows), "brier": brier, "bss": bss,
            "ece": float(ece), "n": int(len(prob)), "base_rate": float(base)}


# ------------------------------------------- per-day pair probabilities --

def day_pair_probabilities(dc, spec, error_model: "ErrorModel",
                           tier: str) -> pd.DataFrame:
    """Conflict probability of every cached pair-second of one day that
    is not yet inside the tier zone and has its closest approach within
    the lookahead. Columns: t, a, b, prob, hit (the pair actually enters
    the zone, observed, within the lookahead after t)."""
    from . import pairs as pairs_mod
    p = dc.pairs
    if p.empty:
        return pd.DataFrame(columns=["t", "a", "b", "prob", "hit"])
    p = p[pairs_mod.volume_mask(p, spec.radius_nm, spec.ceiling_ft)]
    if p.empty:
        return pd.DataFrame(columns=["t", "a", "b", "prob", "hit"])
    H, V = spec.tiers[tier]
    ev = pairs_mod.evaluate_tier(p, H, V, spec.lookahead_s)
    cand = (~ev["observed"]) & (ev["t_cpa"] > 0) \
        & (ev["t_cpa"] <= spec.lookahead_s)
    q = p[cand.to_numpy()]
    e = ev[cand.to_numpy()]
    if q.empty:
        return pd.DataFrame(columns=["t", "a", "b", "prob", "hit"])
    ph_a = phase_group(q["pha"]); ph_b = phase_group(q["phb"])
    t_cpa = e["t_cpa"].to_numpy(float)
    sig_a = error_model.sigma(ph_a, t_cpa)
    sig_b = error_model.sigma(ph_b, t_cpa)
    prob = conflict_probability(
        q["rx"], q["ry"], q["vx"], q["vy"], q["dz"], q["vz"], t_cpa,
        q["tra"], q["trb"], sig_a, sig_b, H, V)
    # did the pair actually enter the zone (observed) within the
    # lookahead after t? Pairs get compact indices so the combined
    # (pair, t) key fits comfortably in int64.
    obs = p[ev["observed"].to_numpy()]
    pair_all = np.concatenate([
        obs["a"].to_numpy().astype(np.int64) * 100000 + obs["b"].to_numpy(),
        q["a"].to_numpy().astype(np.int64) * 100000 + q["b"].to_numpy()])
    _, inv = np.unique(pair_all, return_inverse=True)
    idx_obs, idx_q = inv[:len(obs)].astype(np.int64), inv[len(obs):].astype(np.int64)
    t_obs = obs["t"].to_numpy().astype(np.int64)
    t_q = q["t"].to_numpy().astype(np.int64)
    comb_obs = np.sort(idx_obs * (1 << 34) + t_obs)
    comb_q = idx_q * (1 << 34) + t_q
    hit = np.zeros(len(q), dtype=bool)
    if len(comb_obs):
        lo = np.searchsorted(comb_obs, comb_q + 1)
        ok = lo < len(comb_obs)
        nxt = comb_obs[np.minimum(lo, len(comb_obs) - 1)]
        hit[ok] = ((nxt[ok] >> 34) == idx_q[ok]) & (
            (nxt[ok] & ((1 << 34) - 1)) - t_q[ok] <= spec.lookahead_s)
    return pd.DataFrame({"t": t_q, "a": q["a"].to_numpy(),
                         "b": q["b"].to_numpy(), "prob": prob, "hit": hit})


def expected_conflicts(prob_tables: dict, intervals: pd.DataFrame,
                       tier: str) -> np.ndarray:
    """Per interval: sum over pairs of the maximum conflict probability
    the pair reached in the interval (section 7.3). prob_tables maps
    day -> the frame from day_pair_probabilities (sorted by t)."""
    out = np.zeros(len(intervals))
    if not prob_tables:
        return out
    frames = [f for f in prob_tables.values() if len(f)]
    if not frames:
        return out
    allp = pd.concat(frames, ignore_index=True).sort_values("t")
    t = allp["t"].to_numpy()
    key = (allp["a"].to_numpy().astype(np.int64) << 20) | allp["b"].to_numpy()
    prob = allp["prob"].to_numpy()
    lo = np.searchsorted(t, intervals["t_start"].to_numpy())
    hi = np.searchsorted(t, intervals["t_end"].to_numpy())
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = pd.Series(prob[a:b]).groupby(key[a:b]).max().sum()
    return out


def calibration_from_tables(prob_tables: dict) -> dict:
    frames = [f for f in prob_tables.values() if len(f)]
    if not frames:
        return calibration(np.array([]), np.array([]))
    allp = pd.concat(frames, ignore_index=True)
    return calibration(allp["prob"].to_numpy(), allp["hit"].to_numpy())
