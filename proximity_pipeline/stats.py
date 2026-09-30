"""
Statistical analysis (FRAMEWORK.md section 11): per-event scoring, the
equilibrium check on pre windows, post-vs-pre rate ratios (conditional
likelihood with event fixed effects, day-clustered bootstrap), post-vs-
baseline standardised incidence ratios (day-block bootstrap), Wilcoxon
confirmations, Holm correction, minimum detectable effects, and the
superposed-epoch time course.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import config
from .baseline import Baseline, wilson

PRIMARY = "T1_any"
SECONDARY_COUNTS = ["T2_observed", "T2_predicted", "T3_observed",
                    "T3_predicted"]
SECONDARY_CONTINUOUS = ["expected_conflicts_T1", "s_min"]


# ------------------------------------------------------ per event -------

def score_event_windows(win: pd.DataFrame, bl: Baseline,
                        seed: int) -> pd.DataFrame:
    """Score pre/post window rows (from interval_metrics) against the
    frozen baseline; returns the rows with expected / O-E / z / p /
    in-control columns for every metric."""
    sc = bl.score_counts(win, seed=seed)
    return pd.concat([win, sc], axis=1)


def event_risk_table(scored: pd.DataFrame, events: pd.DataFrame,
                     metrics: list[str]) -> pd.DataFrame:
    """go_around_risk.csv: one row per event with pre and post values,
    expected, O/E, z, p, in-control flags and post - pre differences."""
    pre = scored[scored["window"] == "pre"].set_index("event_id")
    post = scored[scored["window"] == "post"].set_index("event_id")
    base = events.set_index("event_id").copy()
    cols = {}
    for w, src in (("pre", pre), ("post", post)):
        src = src.reindex(base.index)
        for c in ("n_aircraft", "pair_hours", "quality", "flow",
                  "local_hour", "partial_coverage"):
            cols[f"{w}_{c}"] = src[c]
        for m in metrics:
            if m not in src.columns:
                continue
            cols[f"{w}_{m}"] = src[m]
            for suf in ("expected", "oe", "z", "p_upper", "exceed95",
                        "exceed99", "pct"):
                col = f"{m}_{suf}"
                if col in src.columns:
                    cols[f"{w}_{m}_{suf}"] = src[col]
            if f"{m}_exceed95" in src.columns:
                cols[f"{w}_{m}_in_control"] = ~src[f"{m}_exceed95"].astype(bool)
    for m in metrics:
        if f"pre_{m}" in cols:
            cols[f"diff_{m}"] = cols[f"post_{m}"] - cols[f"pre_{m}"]
        if f"pre_{m}_z" in cols:
            cols[f"diff_{m}_z"] = cols[f"post_{m}_z"] - cols[f"pre_{m}_z"]
    out = pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1)
    out["window_bad"] = ((out["pre_quality"] == "bad")
                         | (out["post_quality"] == "bad"))
    out["excluded"] = (out["window_bad"]
                       | out["pre_partial_coverage"].fillna(True).astype(bool)
                       | out["post_partial_coverage"].fillna(True).astype(bool))
    return out.reset_index()


# ---------------------------------------------------- equilibrium -------

def day_block_bootstrap_ratio(num: np.ndarray, den: np.ndarray,
                              days: np.ndarray, reps: int,
                              seed: int) -> tuple[float, float, float]:
    """Ratio of sums with a bootstrap over day blocks."""
    ok = np.isfinite(num) & np.isfinite(den)
    num, den, days = num[ok], den[ok], days[ok]
    if den.sum() <= 0:
        return np.nan, np.nan, np.nan
    point = num.sum() / den.sum()
    uniq = np.unique(days)
    if len(uniq) < 2:
        return point, np.nan, np.nan
    by_day_n = np.array([num[days == d].sum() for d in uniq])
    by_day_d = np.array([den[days == d].sum() for d in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    boot = by_day_n[idx].sum(axis=1) / np.maximum(by_day_d[idx].sum(axis=1),
                                                  1e-12)
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def equilibrium(risk: pd.DataFrame, metric: str = PRIMARY,
                reps: int | None = None) -> dict:
    """Section 11.2: are pre windows within the baseline?"""
    reps = reps or config.BOOTSTRAP_REPS
    r = risk[~risk["excluded"]]
    z = r[f"pre_{metric}_z"].to_numpy(float)
    ok = np.isfinite(z)
    n = int(ok.sum())
    in_ctl = r[f"pre_{metric}_in_control"].to_numpy(bool)[ok]
    k = int(in_ctl.sum())
    lo, hi = wilson(k, n)
    sir, sir_lo, sir_hi = day_block_bootstrap_ratio(
        r[f"pre_{metric}"].to_numpy(float),
        r[f"pre_{metric}_expected"].to_numpy(float),
        r["day"].to_numpy(), reps, config.RANDOM_SEED)
    ks = stats.kstest(z[ok], "norm") if n >= 5 else None
    ks_p = float(ks.pvalue) if ks else np.nan
    sir_ok = (np.isnan(sir_lo) or sir_lo <= 1.0 <= sir_hi)
    ks_ok = (np.isnan(ks_p) or ks_p >= 0.05)
    share_ok = (np.isnan(hi) or hi >= config.LIMIT_WARNING)
    verdict = "PASS" if (n > 0 and sir_ok and ks_ok and share_ok) else "FAIL"
    if n == 0:
        verdict = "NOT ASSESSED"
    return {"metric": metric, "n_events": n, "in_control": k,
            "in_control_share": k / n if n else np.nan,
            "in_control_ci": [lo, hi], "sir_pre": sir,
            "sir_pre_ci": [sir_lo, sir_hi], "ks_p": ks_p,
            "z_mean": float(np.nanmean(z)) if n else np.nan,
            "z_sd": float(np.nanstd(z)) if n else np.nan,
            "verdict": verdict}


# ----------------------------------------------------------- IRR --------

def _cond_loglik(theta, pre, post, e_pre, e_post):
    n = pre + post
    q = e_pre + np.exp(theta) * e_post
    return float(np.sum(post * theta - n * np.log(q)))


def irr_mle(pre, post, e_pre, e_post) -> float:
    """Rate ratio post/pre from the conditional likelihood of a Poisson
    model with event fixed effects and exposure offsets: given each
    event's total, its post count is binomial with odds IRR e_post /
    e_pre, so the event effects cancel."""
    ok = (np.isfinite(pre) & np.isfinite(post) & (e_pre > 0) & (e_post > 0)
          & ((pre + post) > 0))
    if ok.sum() == 0:
        return np.nan
    if post[ok].sum() == 0:
        return 0.0
    if pre[ok].sum() == 0:
        return np.inf
    res = optimize.minimize_scalar(
        lambda th: -_cond_loglik(th, pre[ok], post[ok], e_pre[ok], e_post[ok]),
        bounds=(-8, 8), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(res.x))


def irr_post_pre(risk: pd.DataFrame, metric: str, reps: int,
                 seed: int) -> dict:
    """IRR with a day-clustered bootstrap CI, plus the Wilcoxon
    signed-rank test on the post - pre z-scores (section 11.3)."""
    r = risk[~risk["excluded"]]
    pre = r[f"pre_{metric}"].to_numpy(float)
    post = r[f"post_{metric}"].to_numpy(float)
    e_pre = r["pre_pair_hours"].to_numpy(float)
    e_post = r["post_pair_hours"].to_numpy(float)
    days = r["day"].to_numpy()
    point = irr_mle(pre, post, e_pre, e_post)
    uniq = np.unique(days)
    lo = hi = np.nan
    if len(uniq) >= 2 and np.isfinite(point):
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(days == d) for d in uniq]
        boot = []
        for _ in range(reps):
            pick = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in pick])
            boot.append(irr_mle(pre[idx], post[idx], e_pre[idx], e_post[idx]))
        boot = np.array([b for b in boot if np.isfinite(b)])
        if len(boot):
            lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    zd = r[f"diff_{metric}_z"].to_numpy(float) if f"diff_{metric}_z" in r else \
        np.array([])
    zd = zd[np.isfinite(zd)]
    w_p = np.nan
    if len(zd) >= 6 and np.any(zd != 0):
        w_p = float(stats.wilcoxon(zd).pvalue)
    return {"n_events": int(len(r)), "sum_pre": float(np.nansum(pre)),
            "sum_post": float(np.nansum(post)),
            "exposure_pre_h": float(np.nansum(e_pre)),
            "exposure_post_h": float(np.nansum(e_post)),
            "irr": point, "irr_lo": lo, "irr_hi": hi,
            "wilcoxon_paired_p": w_p}


def sir_window(risk: pd.DataFrame, metric: str, window: str, reps: int,
               seed: int) -> dict:
    """SIR = sum observed / sum expected for pre or post windows with a
    day-block bootstrap CI, and the one-sample Wilcoxon test of the
    window z-scores against 0."""
    r = risk[~risk["excluded"]]
    obs = r[f"{window}_{metric}"].to_numpy(float)
    exp_ = r[f"{window}_{metric}_expected"].to_numpy(float)
    sir, lo, hi = day_block_bootstrap_ratio(obs, exp_, r["day"].to_numpy(),
                                            reps, seed)
    z = r[f"{window}_{metric}_z"].to_numpy(float)
    z = z[np.isfinite(z)]
    w_p = np.nan
    if len(z) >= 6 and np.any(z != 0):
        w_p = float(stats.wilcoxon(z).pvalue)
    return {f"sir_{window}": sir, f"sir_{window}_lo": lo,
            f"sir_{window}_hi": hi, f"wilcoxon_{window}_p": w_p,
            f"sum_expected_{window}": float(np.nansum(exp_))}


def continuous_endpoint(risk: pd.DataFrame, metric: str) -> dict:
    """Post vs pre (paired Wilcoxon on the values) and post vs baseline
    (one-sample Wilcoxon on the post z-scores) for a continuous metric."""
    r = risk[~risk["excluded"]]
    out = {"n_events": int(len(r))}
    if f"pre_{metric}" not in r.columns:
        return out
    pre = r[f"pre_{metric}"].to_numpy(float)
    post = r[f"post_{metric}"].to_numpy(float)
    ok = np.isfinite(pre) & np.isfinite(post)
    out["median_pre"] = float(np.nanmedian(pre)) if ok.any() else np.nan
    out["median_post"] = float(np.nanmedian(post)) if ok.any() else np.nan
    d = (post - pre)[ok]
    out["wilcoxon_paired_p"] = (float(stats.wilcoxon(d).pvalue)
                                if len(d) >= 6 and np.any(d != 0) else np.nan)
    for w in ("pre", "post"):
        z = r.get(f"{w}_{metric}_z", pd.Series(dtype=float)).to_numpy(float)
        z = z[np.isfinite(z)]
        out[f"wilcoxon_{w}_p"] = (float(stats.wilcoxon(z).pvalue)
                                  if len(z) >= 6 and np.any(z != 0) else np.nan)
        out[f"z_mean_{w}"] = float(z.mean()) if len(z) else np.nan
    return out


def holm(pvals: list[float]) -> list[float]:
    p = np.array(pvals, float)
    adj = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return adj.tolist()
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


def minimum_detectable(sum_pre: float, sum_expected_post: float,
                       n_events: int, alpha_disp: float) -> tuple[float, float]:
    """Minimum detectable IRR (post/pre) and SIR (post/baseline) at the
    configured power and alpha, from the normal approximation of the
    log ratio with a per-event over-dispersion allowance."""
    if n_events <= 0 or not np.isfinite(sum_pre) or sum_pre <= 0 \
            or not np.isfinite(sum_expected_post) or sum_expected_post <= 0:
        return np.nan, np.nan
    za = stats.norm.ppf(1 - config.POWER_ALPHA / 2)
    zb = stats.norm.ppf(config.POWER_TARGET)
    extra = alpha_disp / n_events if np.isfinite(alpha_disp) else 0.0
    se_irr = np.sqrt(2.0 / max(sum_pre, 1e-9) + 2 * extra)
    se_sir = np.sqrt(1.0 / max(sum_expected_post, 1e-9) + extra)
    return float(np.exp((za + zb) * se_irr)), float(np.exp((za + zb) * se_sir))


def primary_results(risk: pd.DataFrame, bl: Baseline, scope: str,
                    event_set: str, reps: int) -> pd.DataFrame:
    """results_primary.csv rows for one scope and event set."""
    rows = []
    seed = config.RANDOM_SEED
    counts = [PRIMARY] + [m for m in SECONDARY_COUNTS
                          if f"pre_{m}" in risk.columns]
    for m in counts:
        row = {"scope": scope, "event_set": event_set, "endpoint": m,
               "type": "count", "primary": m == PRIMARY}
        row.update(irr_post_pre(risk, m, reps, seed))
        row.update(sir_window(risk, m, "pre", reps, seed))
        row.update(sir_window(risk, m, "post", reps, seed))
        model = bl.count_models.get(m)
        alpha = model.alpha if model else 0.0
        row["baseline_model"] = model.kind if model else ""
        row["mde_irr"], row["mde_sir"] = minimum_detectable(
            row["sum_pre"], row["sum_expected_post"], row["n_events"], alpha)
        rows.append(row)
    for m in SECONDARY_CONTINUOUS:
        if f"pre_{m}" not in risk.columns:
            continue
        row = {"scope": scope, "event_set": event_set, "endpoint": m,
               "type": "continuous", "primary": False}
        row.update(continuous_endpoint(risk, m))
        rows.append(row)
    df = pd.DataFrame(rows)
    # Holm across secondary endpoints (paired test and post-vs-baseline
    # test separately)
    sec = ~df["primary"].astype(bool)
    for col in ("wilcoxon_paired_p", "wilcoxon_post_p"):
        if col in df.columns:
            adj = np.full(len(df), np.nan)
            adj[sec.to_numpy()] = holm(df.loc[sec, col].tolist())
            df[f"{col}_holm"] = adj
    return df


# ------------------------------------------------- superposed epoch -----

def superposed_epoch(events: pd.DataFrame, exposure, enc: pd.DataFrame,
                     clock: pd.DataFrame, bl: Baseline, metric: str,
                     reps: int, seed: int) -> pd.DataFrame:
    """O/E of the encounter rate in 1-min bins around t0, pooled across
    events (ratio of sums), with a day-block bootstrap band. E for a bin
    is the baseline rate per pair-hour of the clock window containing
    the bin, times the bin's pair-hours (section 11.5)."""
    from .windows import attribute_encounters
    model = bl.count_models.get(metric)
    if model is None or events.empty:
        return pd.DataFrame(columns=["minute", "observed", "expected", "oe",
                                     "oe_lo", "oe_hi", "n_events"])
    half = int(config.EPOCH_RANGE_MIN)
    step = int(config.EPOCH_BIN_MIN * 60)
    mins = np.arange(-half, half, config.EPOCH_BIN_MIN)
    clock = clock.copy()
    clock["rate"] = model.mu(clock) / clock["pair_hours"].replace(0, np.nan)
    ct = clock["t_start"].to_numpy()
    order = np.argsort(ct)
    ct_sorted = ct[order]
    rate_sorted = clock["rate"].to_numpy()[order]
    cw_len = int((clock["t_end"] - clock["t_start"]).iloc[0])
    rows = []
    for _, e in events.iterrows():
        t0 = int(e["t0"])
        for mnt in mins:
            a = t0 + int(mnt * 60); b = a + step
            m = exposure.interval(a, b)
            centre = (a + b) // 2
            i = np.searchsorted(ct_sorted, centre, side="right") - 1
            rate = (rate_sorted[i] if i >= 0 and ct_sorted[i] + cw_len > centre
                    else np.nan)
            rows.append({"event_id": e["event_id"], "day": e["day"],
                         "minute": mnt, "t_start": a, "t_end": b,
                         "pair_hours": m["pair_s"] / 3600.0,
                         "rate": rate})
    bins = pd.DataFrame(rows)
    if bins.empty:
        return pd.DataFrame(columns=["minute", "observed", "expected", "oe",
                                     "oe_lo", "oe_hi", "n_events"])
    tier = metric.split("_")[0]
    counts = attribute_encounters(enc, bins, {tier: config.TIERS[tier]})
    bins["observed"] = counts[metric].to_numpy()
    bins["expected"] = bins["rate"].fillna(0.0) * bins["pair_hours"]
    days = bins["day"].unique()
    rng = np.random.default_rng(seed)
    out = []
    piv_o = bins.pivot_table(index="day", columns="minute", values="observed",
                             aggfunc="sum").reindex(days).fillna(0)
    piv_e = bins.pivot_table(index="day", columns="minute", values="expected",
                             aggfunc="sum").reindex(days).fillna(0)
    O = piv_o.to_numpy(); E = piv_e.to_numpy()
    idx = rng.integers(0, len(days), size=(reps, len(days)))
    boot = O[idx].sum(axis=1) / np.maximum(E[idx].sum(axis=1), 1e-12)
    for j, mnt in enumerate(piv_o.columns):
        o, ex = O[:, j].sum(), E[:, j].sum()
        out.append({"minute": mnt, "observed": o, "expected": ex,
                    "oe": o / ex if ex > 0 else np.nan,
                    "oe_lo": float(np.percentile(boot[:, j], 2.5)),
                    "oe_hi": float(np.percentile(boot[:, j], 97.5)),
                    "n_events": int(len(events))})
    return pd.DataFrame(out)
