"""
Comparison with normal landings (FRAMEWORK.md section 11.6): the change
from the window before to the window after each go-around, next to the
same change around normal full-stop landings matched on the hour of day
(main analysis), with the checks that make that comparison credible:

- the before-windows of both groups scored against the baseline
  (was the airspace normal beforehand, for go-arounds and for landings);
- before-window traffic of both groups (are go-arounds in busier times);
- a pre-trend check (the change from 20-10 min before to the last
  10 min before: were the two groups already moving apart);
- a placebo test (random sets of hour-matched landings standing in for
  the go-arounds);
- a regression confirmation (Poisson model of window counts with a
  go-around x after interaction, hour and runway-flow effects, standard
  errors clustered by day);
- a robustness set (finer matching, leave-one-out, isolated go-arounds,
  other window lengths, ...).

Every interval resamples whole days, so events on the same day are not
treated as independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config
from .baseline import wilson

STRATA_MAIN = ("hour",)
STRATA_FLOW = ("hour", "flow")
STRATA_TRAFFIC = ("hour", "pre_traffic_band")

# (key, column in the paired frame, label)
CHANGE_MEASURES = [
    ("aircraft_time", "aircraft_s", "Aircraft time in the area"),
    ("pair_time", "pair_hours", "Traffic (aircraft-pair time)"),
    ("T1_any", "T1_any", "T1 encounters"),
    ("T2_any", "T2_any", "T2 close encounters"),
    ("T3_any", "T3_any", "T3 near-collision encounters"),
]
# Encounter measures with fewer than this many encounters in the
# go-around windows (before + after) are reported as too few to tell.
MIN_ENCOUNTERS_FOR_RATIO = 10


# ------------------------------------------------------- paired frame ---

def paired_windows(win: pd.DataFrame, events: pd.DataFrame,
                   cols: list[str],
                   windows: tuple[str, ...] = ("pre", "post")) -> pd.DataFrame:
    """One row per event with <window>_<col> for every window present in
    `win` (early | pre | post), the baseline scores of the before-window
    when they are there, day, local hour of t0, runway flow of the
    before-window, and exclusion flags (bad-quality or partially covered
    windows; `excluded` covers before and after, `early_bad` the early
    window used only by the pre-trend check)."""
    base = events.set_index("event_id")
    out = pd.DataFrame(index=base.index)
    extra = [c for c in win.columns if c.startswith("T1_any_")
             and c.split("T1_any_")[1] in ("expected", "z", "zmid",
                                           "in_control", "exceed95")]
    for w in windows:
        sub = (win[win["window"] == w] if "window" in win.columns
               else win.iloc[:0])
        if sub.empty and len(base):
            continue                  # window not computed (e.g. early)
        src = sub.set_index("event_id").reindex(base.index)
        for c in cols + (extra if w == "pre" else []):
            out[f"{w}_{c}"] = (src[c].to_numpy(float) if c in src.columns
                               else np.full(len(src), np.nan))
        q = src["quality"] if "quality" in src.columns else pd.Series(
            "bad", index=src.index)
        pc = (src["partial_coverage"] if "partial_coverage" in src.columns
              else pd.Series(True, index=src.index))
        out[f"{w}_bad"] = ((q == "bad") | pc.fillna(True).astype(bool)
                           ).to_numpy()
        if w == "pre":
            out["flow"] = (src["flow"].astype(str).to_numpy()
                           if "flow" in src.columns else "none")
            out["day"] = (src["day"].to_numpy() if "day" in src.columns
                          else None)
        if w == "post" and "day" not in out:
            out["day"] = src["day"].to_numpy() if "day" in src.columns else None
    out["excluded"] = (out.get("pre_bad", True) | out.get("post_bad", True))
    if "early_bad" not in out:
        out["early_bad"] = True
    t0 = pd.to_datetime(base["t0"].to_numpy(), unit="s", utc=True)
    out["hour"] = t0.tz_convert(_local_tz()).hour
    out["t0"] = base["t0"].to_numpy()
    for c in ("callsign", "runway", "icao24", "outcome", "cluster_size",
              "pre_contaminated"):
        if c in base.columns:
            out[c] = base[c].to_numpy()
    return out.reset_index()


def _local_tz() -> str:
    from . import goaround_adapter as ga
    return ga.config.LOCAL_TZ


def add_traffic_band(g: pd.DataFrame, c: pd.DataFrame) -> None:
    """Before-window traffic band (low / mid / high: terciles of the
    landings' before-window traffic), for matching on traffic."""
    ref = c.loc[~c["excluded"], "pre_pair_hours"].dropna()
    if len(ref) < 3:
        g["pre_traffic_band"] = "all"; c["pre_traffic_band"] = "all"
        return
    cuts = np.percentile(ref, [100 / 3, 200 / 3])
    for df in (g, c):
        v = df["pre_pair_hours"].to_numpy(float)
        df["pre_traffic_band"] = np.where(v <= cuts[0], "low",
                                          np.where(v <= cuts[1], "mid",
                                                   "high"))


# ----------------------------------------------------------- matching ---

def _keys(df: pd.DataFrame, keys) -> np.ndarray:
    parts = [df[k].astype(str).to_numpy() for k in keys]
    out = parts[0].astype(object)
    for p in parts[1:]:
        out = out + "|" + p
    return out


def match_weights(g: pd.DataFrame, c: pd.DataFrame,
                  keys=STRATA_MAIN) -> tuple[np.ndarray, np.ndarray]:
    """Go-arounds kept (their stratum has landings) and the landing
    weights that give the landings the go-arounds' mix of strata: a
    landing in a stratum with n_g go-arounds and n_c landings weighs
    n_g / n_c, so the weights in each stratum add up to the number of
    go-arounds there."""
    kg, kc = _keys(g, keys), _keys(c, keys)
    nc = pd.Series(kc).value_counts()
    keep = np.isin(kg, nc.index.to_numpy())
    ng = pd.Series(kg[keep]).value_counts()
    w = np.array([ng.get(k, 0) / nc[k] for k in kc], float)
    return keep, w


# ------------------------------------------------------------ ratios ----

def _ratio_ci(per_day: dict, n_days: int, reps: int, seed: int, fn
              ) -> tuple[float, float, float]:
    """Point value and day-block bootstrap 95 % CI of fn(sums), where
    sums are the per-day arrays summed over a resample of days."""
    sums = {k: v.sum() for k, v in per_day.items()}
    point = float(fn(sums))
    if n_days < 2 or not np.isfinite(point):
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_days, size=(reps, n_days))
    boot = fn({k: v[idx].sum(axis=1) for k, v in per_day.items()})
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return point, np.nan, np.nan
    return (point, float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(b) > 0, np.asarray(a) / np.asarray(b),
                        np.nan)


def _usable(df: pd.DataFrame, frm: str, to: str) -> np.ndarray:
    ok = ~df[f"{frm}_bad"].astype(bool) & ~df[f"{to}_bad"].astype(bool)
    return ok.to_numpy()


def did(g: pd.DataFrame, c: pd.DataFrame, col: str, reps: int, seed: int,
        keys=STRATA_MAIN, frm: str = "pre", to: str = "post") -> dict:
    """Change from window `frm` to window `to` (sum to / sum from) around
    go-arounds, the same around landings weighted to the go-arounds'
    strata, and the ratio of the two changes ("go-around vs normal").
    Point values with day-block bootstrap 95 % CIs (the same day
    resamples for all three)."""
    if f"{frm}_{col}" not in g.columns or f"{to}_{col}" not in g.columns:
        return {}
    g = g[_usable(g, frm, to)]
    c = c[_usable(c, frm, to)]
    keep, w = match_weights(g, c, keys)
    n_unmatched = int((~keep).sum())
    g = g[keep]
    c = c.assign(w=w)
    c = c[c["w"] > 0]
    days = np.array(sorted(set(g["day"].dropna()) | set(c["day"].dropna())))

    def per_day(df, which, weight=None):
        v = df[f"{which}_{col}"].fillna(0.0).to_numpy(float)
        if weight is not None:
            v = v * df[weight].to_numpy(float)
        s = pd.Series(v).groupby(df["day"].to_numpy()).sum()
        return s.reindex(days, fill_value=0.0).to_numpy()

    pdays = {"gp": per_day(g, frm), "ga": per_day(g, to),
             "cp": per_day(c, frm, "w"), "ca": per_day(c, to, "w")}
    r_g = _ratio_ci(pdays, len(days), reps, seed,
                    lambda s: _div(s["ga"], s["gp"]))
    r_c = _ratio_ci(pdays, len(days), reps, seed,
                    lambda s: _div(s["ca"], s["cp"]))
    r_d = _ratio_ci(pdays, len(days), reps, seed,
                    lambda s: _div(_div(s["ga"], s["gp"]),
                                   _div(s["ca"], s["cp"])))
    d = (g[f"{to}_{col}"] - g[f"{frm}_{col}"]).dropna().to_numpy()
    p = (float(sps.wilcoxon(d).pvalue)
         if len(d) >= 6 and np.any(d != 0) else np.nan)
    wsum = float(c["w"].sum())
    return {
        "n_events": int(len(g)), "n_unmatched": n_unmatched,
        "n_controls": int(len(c)),
        "before_total": float(g[f"{frm}_{col}"].sum()),
        "after_total": float(g[f"{to}_{col}"].sum()),
        "ga_mean_before": float(g[f"{frm}_{col}"].mean()) if len(g) else np.nan,
        "ga_mean_after": float(g[f"{to}_{col}"].mean()) if len(g) else np.nan,
        "normal_mean_before": (float((c[f"{frm}_{col}"] * c["w"]).sum() / wsum)
                               if wsum > 0 else np.nan),
        "normal_mean_after": (float((c[f"{to}_{col}"] * c["w"]).sum() / wsum)
                              if wsum > 0 else np.nan),
        "ratio": r_g[0], "ratio_lo": r_g[1], "ratio_hi": r_g[2],
        "paired_p": p,
        "normal_ratio": r_c[0], "normal_ratio_lo": r_c[1],
        "normal_ratio_hi": r_c[2],
        "vs_normal": r_d[0], "vs_normal_lo": r_d[1], "vs_normal_hi": r_d[2],
    }


def change_summary(g: pd.DataFrame, c: pd.DataFrame, reps: int, seed: int,
                   keys=STRATA_MAIN, frm: str = "pre",
                   to: str = "post") -> pd.DataFrame:
    """One row per measure in CHANGE_MEASURES (change_summary.csv).
    Encounter measures with too few encounters are flagged
    (too_few = True); their ratios stay in the CSV for completeness."""
    rows = []
    for key, col, label in CHANGE_MEASURES:
        r = did(g, c, col, reps, seed, keys, frm, to)
        if not r:
            continue
        r = {"measure": key, "label": label, **r}
        r["too_few"] = bool(key.endswith("_any") and
                            r["before_total"] + r["after_total"]
                            < MIN_ENCOUNTERS_FOR_RATIO)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ placebo ---

def placebo(g: pd.DataFrame, c: pd.DataFrame, col: str, n_sets: int,
            seed: int, keys=STRATA_MAIN) -> dict:
    """Placebo test: draw, for every go-around, a random normal landing
    from the same stratum (same hour of day), n_sets times, and compute
    each set's after / before ratio exactly as for the go-arounds. If a
    go-around changed nothing, its ratio would look like one of these.
    Returns the go-around ratio, the placebo ratios, and the share of
    placebo sets with a ratio at least as large (one-sided p-value with
    the usual +1 correction)."""
    g = g[_usable(g, "pre", "post")]
    c = c[_usable(c, "pre", "post")]
    keep, _ = match_weights(g, c, keys)
    g = g[keep]
    if g.empty:
        return {}
    kg, kc = _keys(g, keys), _keys(c, keys)
    pre_c = c[f"pre_{col}"].fillna(0).to_numpy(float)
    post_c = c[f"post_{col}"].fillna(0).to_numpy(float)
    rng = np.random.default_rng(seed)
    s_pre = np.zeros(n_sets); s_post = np.zeros(n_sets)
    for k, m in pd.Series(kg).value_counts().items():
        pool = np.flatnonzero(kc == k)
        pick = pool[rng.integers(0, len(pool), size=(n_sets, m))]
        s_pre += pre_c[pick].sum(axis=1)
        s_post += post_c[pick].sum(axis=1)
    ratios = _div(s_post, s_pre)
    ratios = ratios[np.isfinite(ratios)]
    ga_ratio = float(_div(g[f"post_{col}"].sum(), g[f"pre_{col}"].sum()))
    n_ge = int((ratios >= ga_ratio).sum())
    return {"measure": col, "n_events": int(len(g)), "n_sets": int(len(ratios)),
            "ga_ratio": ga_ratio,
            "placebo_median": float(np.median(ratios)) if len(ratios) else np.nan,
            "placebo_lo": float(np.percentile(ratios, 2.5)) if len(ratios) else np.nan,
            "placebo_hi": float(np.percentile(ratios, 97.5)) if len(ratios) else np.nan,
            "share_at_least_as_large": n_ge / len(ratios) if len(ratios) else np.nan,
            "p_value": (n_ge + 1) / (len(ratios) + 1) if len(ratios) else np.nan,
            "ratios": ratios}


# ---------------------------------------------------- before-window -----

def before_checks(g: pd.DataFrame, c: pd.DataFrame, reps: int, seed: int,
                  keys=STRATA_MAIN) -> pd.DataFrame:
    """Was the airspace normal before, for both groups? Rows for the
    go-arounds and for the (hour-matched, weighted) landings: share of
    before-windows inside the baseline's normal range, observed ÷
    expected T1 encounters in the before-windows, and average
    before-window traffic; plus a row comparing the go-arounds'
    before-window traffic with the landings'."""
    g = g[~g["pre_bad"].astype(bool)]
    c = c[~c["pre_bad"].astype(bool)]
    keep, w = match_weights(g, c, keys)
    g = g[keep]
    c = c.assign(w=w)
    c = c[c["w"] > 0]
    days = np.array(sorted(set(g["day"].dropna()) | set(c["day"].dropna())))
    rows = []
    for name, df, wcol in (("go-arounds", g, None), ("normal landings", c, "w")):
        wt = (df[wcol].to_numpy(float) if wcol else np.ones(len(df)))
        row = {"group": name, "n": int(len(df))}
        if "pre_T1_any_in_control" in df.columns:
            zc = df["pre_T1_any_expected"].to_numpy(float) > 0
            inside = df["pre_T1_any_in_control"].astype(bool).to_numpy()
            share = (float((wt[zc] * inside[zc]).sum() / wt[zc].sum())
                     if wt[zc].sum() > 0 else np.nan)
            n_eff = int(zc.sum())
            lo, hi = wilson(int(round(share * n_eff)), n_eff) if n_eff else (np.nan, np.nan)
            row.update({"in_normal_range": share, "in_normal_range_lo": lo,
                        "in_normal_range_hi": hi})

            def per_day(col):
                v = df[col].fillna(0).to_numpy(float) * wt
                return (pd.Series(v).groupby(df["day"].to_numpy()).sum()
                        .reindex(days, fill_value=0.0).to_numpy())
            oe = _ratio_ci({"o": per_day("pre_T1_any"),
                            "e": per_day("pre_T1_any_expected")},
                           len(days), reps, seed,
                           lambda s: _div(s["o"], s["e"]))
            row.update({"oe_before": oe[0], "oe_before_lo": oe[1],
                        "oe_before_hi": oe[2]})
        tot = wt.sum()
        row["traffic_before_h"] = (float((df["pre_pair_hours"].fillna(0)
                                          * wt).sum() / tot) if tot else np.nan)
        row["aircraft_before"] = (float((df["pre_aircraft_s"].fillna(0)
                                         * wt).sum() / tot
                                        / (config.WINDOW_MIN * 60))
                                  if tot and "pre_aircraft_s" in df else np.nan)
        rows.append(row)

    def per_day_mean(df, wcol):
        wt = df[wcol].to_numpy(float) if wcol else np.ones(len(df))
        v = df["pre_pair_hours"].fillna(0).to_numpy(float) * wt
        s = (pd.Series(v).groupby(df["day"].to_numpy()).sum()
             .reindex(days, fill_value=0.0).to_numpy())
        n = (pd.Series(wt).groupby(df["day"].to_numpy()).sum()
             .reindex(days, fill_value=0.0).to_numpy())
        return s, n
    gs, gn = per_day_mean(g, None)
    cs, cn = per_day_mean(c, "w")
    r = _ratio_ci({"gs": gs, "gn": gn, "cs": cs, "cn": cn}, len(days), reps,
                  seed, lambda s: _div(_div(s["gs"], s["gn"]),
                                       _div(s["cs"], s["cn"])))
    rows.append({"group": "go-arounds ÷ landings (before-window traffic)",
                 "n": int(len(g)), "traffic_ratio": r[0],
                 "traffic_ratio_lo": r[1], "traffic_ratio_hi": r[2]})
    return pd.DataFrame(rows)


# -------------------------------------------------------- regression ----

def _poisson_cluster(y, X, w, groups, max_iter=100):
    """Weighted Poisson regression (IRLS) with day-clustered sandwich
    standard errors. Returns coefficients and their covariance."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max((w * y).sum() / max(w.sum(), 1e-12), 1e-6))
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        W = w * mu
        A = (X * W[:, None]).T @ X
        b = (X * W[:, None]).T @ z
        new = np.linalg.lstsq(A + 1e-9 * np.eye(len(beta)), b, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    mu = np.exp(np.clip(X @ beta, -30, 30))
    bread = np.linalg.pinv((X * (w * mu)[:, None]).T @ X)
    sc = X * (w * (y - mu))[:, None]
    S = pd.DataFrame(sc).groupby(np.asarray(groups)).sum().to_numpy()
    G = S.shape[0]
    meat = S.T @ S * (G / max(G - 1, 1))
    return beta, bread @ meat @ bread


def regression_check(g: pd.DataFrame, c: pd.DataFrame, col: str,
                     keys=STRATA_MAIN) -> dict:
    """Confirmation of the go-around vs normal comparison by regression:
    window counts ~ after + go-around + after x go-around + hour of day
    + runway flow (Poisson, landings weighted to the go-arounds' hours,
    standard errors clustered by day). exp(after x go-around) is the
    go-around vs normal ratio adjusted for hour and flow."""
    g = g[_usable(g, "pre", "post")]
    c = c[_usable(c, "pre", "post")]
    keep, w = match_weights(g, c, keys)
    g = g[keep].assign(w=1.0, is_ga=1.0)
    c = c.assign(w=w, is_ga=0.0)
    c = c[c["w"] > 0]
    both = pd.concat([g, c], ignore_index=True)
    if len(g) < 5 or len(c) < 5:
        return {}
    long = pd.concat([
        both.assign(y=both[f"pre_{col}"], post=0.0),
        both.assign(y=both[f"post_{col}"], post=1.0)], ignore_index=True)
    long = long[np.isfinite(long["y"].to_numpy(float))]
    X = pd.DataFrame({"const": 1.0, "post": long["post"],
                      "ga": long["is_ga"],
                      "post_x_ga": long["post"] * long["is_ga"]})
    for f in ("hour", "flow"):
        d = pd.get_dummies(long[f].astype(str), prefix=f, drop_first=True,
                           dtype=float)
        X = pd.concat([X, d], axis=1)
    Xv = X.to_numpy(float)
    y = long["y"].to_numpy(float)
    beta, V = _poisson_cluster(y, Xv, long["w"].to_numpy(float),
                               long["day"].astype(str).to_numpy())
    j = list(X.columns).index("post_x_ga")
    se = float(np.sqrt(max(V[j, j], 0)))
    b = float(beta[j])
    zval = b / se if se > 0 else np.nan
    return {"measure": col, "n_events": int(len(g)),
            "n_controls": int(len(c)),
            "ratio": float(np.exp(b)),
            "ratio_lo": float(np.exp(b - 1.96 * se)),
            "ratio_hi": float(np.exp(b + 1.96 * se)),
            "p_value": (float(2 * sps.norm.sf(abs(zval)))
                        if np.isfinite(zval) else np.nan),
            "adjusted_for": "hour of day, runway flow; clustered by day"}


# --------------------------------------------------------- robustness ---

def leave_one_out(g: pd.DataFrame, c: pd.DataFrame, col: str,
                  keys=STRATA_MAIN) -> pd.DataFrame:
    """Go-around vs normal ratio (point value) with each go-around left
    out in turn: how much does any single event move the result?"""
    g = g[_usable(g, "pre", "post")]
    c = c[_usable(c, "pre", "post")]
    rows = []
    for i in range(len(g)):
        gi = g.drop(index=g.index[i])
        keep, w = match_weights(gi, c, keys)
        gi = gi[keep]
        cp = (c[f"pre_{col}"] * w).sum(); ca = (c[f"post_{col}"] * w).sum()
        gp, ga_ = gi[f"pre_{col}"].sum(), gi[f"post_{col}"].sum()
        rows.append({"event_id": g["event_id"].iloc[i],
                     "callsign": g.get("callsign", pd.Series([""] * len(g))).iloc[i],
                     "before": float(g[f"pre_{col}"].iloc[i]),
                     "after": float(g[f"post_{col}"].iloc[i]),
                     "vs_normal_without": float(_div(_div(ga_, gp),
                                                     _div(ca, cp)))})
    return pd.DataFrame(rows)


def robustness(g: pd.DataFrame, c: pd.DataFrame, col: str, reps: int,
               seed: int, extra: list | None = None) -> pd.DataFrame:
    """The go-around vs normal ratio for `col` under every variation.
    `extra` adds (label, g, c) triples computed elsewhere (other window
    lengths)."""
    rows = []

    def add(label, gg, cc, keys=STRATA_MAIN, note=""):
        r = did(gg, cc, col, reps, seed, keys)
        if r:
            rows.append({"variation": label, "note": note, **r})

    add("Main analysis (landings matched on hour of day)", g, c)
    add("Landings matched on hour and runway flow", g, c, STRATA_FLOW)
    if "pre_traffic_band" in g.columns:
        add("Landings matched on hour and before-window traffic", g, c,
            STRATA_TRAFFIC)
    loo = leave_one_out(g, c, col)
    if len(loo):
        top = loo.loc[loo["vs_normal_without"].idxmin()]
        note = (f"{top['callsign']} ({top['event_id']}, "
                f"{top['before']:.0f} → {top['after']:.0f})")
        add("Without the most influential go-around",
            g[g["event_id"] != top["event_id"]], c, note=note)
    if "cluster_size" in g.columns:
        add("Isolated go-arounds only (no other within the window)",
            g[g["cluster_size"] == 1], c)
    if "pre_contaminated" in g.columns:
        add("No other go-around in the before-window",
            g[~g["pre_contaminated"].astype(bool)], c)
    if g["day"].notna().any():
        busiest = g["day"].value_counts().idxmax()
        n_b = int((g["day"] == busiest).sum())
        add("Without the day with the most go-arounds",
            g[g["day"] != busiest], c[c["day"] != busiest],
            note=f"{busiest}, {n_b} go-arounds")
    if "pre_T1_any_in_control" in g.columns and \
            "pre_T1_any_in_control" in c.columns:
        add("Only windows with a normal before-window (both groups)",
            g[g["pre_T1_any_in_control"].fillna(False).astype(bool)],
            c[c["pre_T1_any_in_control"].fillna(False).astype(bool)])
    if col == "T1_any" and "pre_T1_observed" in g.columns:
        r = did(g, c, "T1_observed", reps, seed)
        if r:
            rows.append({"variation": "Observed encounters only "
                                      "(no 120-second projection)",
                         "note": "", **r})
    for label, gg, cc in (extra or []):
        add(label, gg, cc)
    out = pd.DataFrame(rows)
    out.insert(0, "measure", col)
    return out


# ------------------------------------------------ minute by minute -----

def raw_epoch(g: pd.DataFrame, c: pd.DataFrame, exposure, enc: pd.DataFrame,
              reps: int, seed: int, keys=STRATA_MAIN) -> pd.DataFrame:
    """Minute-by-minute raw values around go-arounds and around the
    hour-matched landings (no baseline): average number of aircraft in
    the area and T1 encounters per event, per minute, with day-block
    bootstrap 95 % bands."""
    half = int(config.EPOCH_RANGE_MIN)
    step = int(config.EPOCH_BIN_MIN * 60)
    mins = np.arange(-half, half, config.EPOCH_BIN_MIN)
    gg = g[~g["pre_bad"].astype(bool) & ~g["post_bad"].astype(bool)]
    cc = c[~c["pre_bad"].astype(bool) & ~c["post_bad"].astype(bool)]
    keep, w = match_weights(gg, cc, keys)
    gg = gg[keep]
    cc = cc.assign(w=w)
    cc = cc[cc["w"] > 0]
    e = enc[(enc["tier"] == "T1") & (enc["kind"] == "any")
            & enc["inside_ceiling"]]
    te = np.sort(e["t_min"].to_numpy())
    days = np.array(sorted(set(gg["day"].dropna()) | set(cc["day"].dropna())))
    out = []
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, max(len(days), 1), size=(reps, len(days)))
    for group, df, wt in (("go-around", gg, np.ones(len(gg))),
                          ("normal landing", cc, cc["w"].to_numpy(float))):
        if df.empty:
            continue
        t0 = df["t0"].to_numpy(np.int64)
        starts = (t0[:, None] + (mins * 60).astype(np.int64)[None, :])
        ends = starts + step
        ac_s, _ = exposure.sums(starts.ravel(), ends.ravel())
        ac = ac_s.reshape(starts.shape) / step
        n_enc = (np.searchsorted(te, ends.ravel())
                 - np.searchsorted(te, starts.ravel())).reshape(starts.shape)
        dayv = df["day"].to_numpy()
        W = (pd.Series(wt).groupby(dayv).sum().reindex(days, fill_value=0.0)
             .to_numpy())
        A = (pd.DataFrame(ac * wt[:, None]).groupby(dayv).sum()
             .reindex(days, fill_value=0.0).to_numpy())
        N = (pd.DataFrame(n_enc * wt[:, None]).groupby(dayv).sum()
             .reindex(days, fill_value=0.0).to_numpy())
        bw = W[idx].sum(axis=1)[:, None]
        bA = A[idx].sum(axis=1) / np.maximum(bw, 1e-12)
        bN = N[idx].sum(axis=1) / np.maximum(bw, 1e-12)
        tot = W.sum()
        for j, m in enumerate(mins):
            out.append({"group": group, "minute": float(m),
                        "aircraft": A[:, j].sum() / tot,
                        "aircraft_lo": float(np.percentile(bA[:, j], 2.5)),
                        "aircraft_hi": float(np.percentile(bA[:, j], 97.5)),
                        "t1_per_event": N[:, j].sum() / tot,
                        "t1_lo": float(np.percentile(bN[:, j], 2.5)),
                        "t1_hi": float(np.percentile(bN[:, j], 97.5)),
                        "n": int(len(df))})
    return pd.DataFrame(out)


# ------------------------------------------- similar landings (plots) ---

def similar_landings(row: pd.Series, c: pd.DataFrame, k: int = 10
                     ) -> pd.DataFrame:
    """The k normal landings most similar to one go-around: same hour of
    day, same runway flow when enough are available, and the closest
    before-window traffic. The first row is the most similar landing."""
    cc = c[~c["excluded"].astype(bool)]
    same_h = cc[cc["hour"] == row["hour"]]
    if same_h.empty:
        same_h = cc
    same_f = same_h[same_h["flow"] == row.get("flow")]
    pool = same_f if len(same_f) >= k else same_h
    d = np.abs(np.log1p(pool["pre_pair_hours"].to_numpy(float))
               - np.log1p(float(row["pre_pair_hours"])))
    return pool.iloc[np.argsort(d, kind="stable")[:k]]
