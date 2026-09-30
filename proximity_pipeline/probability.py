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
    # "line" is a point; use the radial direction instead
    if still.any():
        rad = np.maximum(np.hypot(tx, ty), 1e-12)
        wr = support(tx / rad, ty / rad)
        p_h = np.where(still, norm.cdf(wr - rad) - norm.cdf(-wr - rad), p_h)
    sz = np.sqrt(sz_a ** 2 + sz_b ** 2)
    dz_cpa = dz + vz * t_cpa
    p_z = norm.cdf((V - dz_cpa) / sz) - norm.cdf((-V - dz_cpa) / sz)
    return np.clip(p_h * p_z, 0.0, 1.0)


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
