"""
Baseline and noise framework (FRAMEWORK.md section 9).

For each count metric a negative binomial regression with exposure offset

    log E[count] = log(pair_hours) + hour + daytype + month + flow
                   + beta * log(n_aircraft)

gives, for any window, the expected count and its full predictive
distribution given that window's conditions (a risk-adjusted control
chart). Poisson is used when over-dispersion is not significant; sparse
metrics fall back to a pooled exposure-adjusted Poisson rate. Continuous
metrics use empirical conditional quantiles within strata. Phase I trims
extreme windows and refits; Phase II scores windows with the frozen
model (mid-p upper-tail p-value, randomized quantile residual, control
limits). Validation: out-of-sample calibration, PIT, stability curve,
data-sufficiency table, day-block bootstrap intervals, and a verdict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import config

COUNT_KINDS = ("any", "observed", "predicted")


def count_metrics(tiers) -> list[str]:
    cols = [f"{t}_{k}" for t in tiers for k in COUNT_KINDS]
    cols.append("T1_any_nonprocedural")
    return cols


def continuous_metrics(tiers) -> list[str]:
    return ["s_min"] + [f"expected_conflicts_{t}" for t in tiers]


PRIMARY_METRIC = "T1_any"
# bumped when the baseline's inputs change meaning (2: UTC month covariate,
# no probability-model metrics), so stored baselines are rebuilt
BASELINE_VERSION = 2
S_MIN_CAP = 1.25     # s_min is undefined without a close pair: capped here


# ---------------------------------------------------------- design ----

@dataclass
class Design:
    """Treatment-coded design: intercept, factor dummies, log(n_aircraft).
    For each factor the reference level is the most frequent one; levels
    with too few windows or no events are merged into it (recorded in
    `merged`), because they cannot be estimated on their own."""
    hour_block_h: int
    levels: dict = field(default_factory=dict)   # factor -> [ref, others]
    factors: tuple = ("hour_block", "daytype", "month", "flow")
    use_log_n: bool = True
    merged: dict = field(default_factory=dict)   # factor -> merged levels

    def factor_values(self, df: pd.DataFrame) -> dict:
        vals = {
            "hour_block": (df["local_hour"] // self.hour_block_h).astype(int)
            .astype(str).to_numpy(),
            "daytype": df["daytype"].astype(str).to_numpy(),
            "month": df["month"].astype(str).to_numpy(),
            "flow": df["flow"].astype(str).to_numpy(),
        }
        return {f: vals[f] for f in self.factors}

    def fit_levels(self, df: pd.DataFrame, y: np.ndarray | None = None) -> None:
        self.levels, self.merged = {}, {}
        for f, v in self.factor_values(df).items():
            uniq, counts = np.unique(v, return_counts=True)
            ref = str(uniq[np.argmax(counts)])
            keep, merged = [ref], []
            for lev, cnt in zip(uniq, counts):
                lev = str(lev)
                if lev == ref:
                    continue
                events = (y[v == lev].sum() if y is not None else 1)
                if cnt < config.BASELINE_MIN_LEVEL_WINDOWS or events <= 0:
                    merged.append(lev)
                else:
                    keep.append(lev)
            self.levels[f] = keep
            if merged:
                self.merged[f] = merged

    def matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str], int]:
        """Design matrix. Levels not in the design (merged, or unseen at
        scoring time) map to the reference level; the unseen count is
        returned."""
        n = len(df)
        cols = [np.ones(n)]
        names = ["intercept"]
        unknown = 0
        for f, vals in self.factor_values(df).items():
            levels = self.levels[f]
            known = np.isin(vals, levels + self.merged.get(f, []))
            unknown += int((~known).sum())
            for lev in levels[1:]:
                cols.append((vals == lev).astype(float))
                names.append(f"{f}[{lev}]")
        if self.use_log_n:
            cols.append(np.log(np.maximum(df["n_aircraft"].to_numpy(float),
                                          1.0)))
            names.append("log_n_aircraft")
        return np.column_stack(cols), names, unknown


# ------------------------------------------------ count distributions ----

def _nb_params(mu, alpha):
    n = 1.0 / alpha
    p = n / (n + mu)
    return n, p


def count_cdf(y, mu, alpha):
    y = np.asarray(y, float); mu = np.maximum(np.asarray(mu, float), 1e-12)
    if alpha <= 0:
        return stats.poisson.cdf(y, mu)
    n, p = _nb_params(mu, alpha)
    return stats.nbinom.cdf(y, n, p)


def count_pmf(y, mu, alpha):
    y = np.asarray(y, float); mu = np.maximum(np.asarray(mu, float), 1e-12)
    if alpha <= 0:
        return stats.poisson.pmf(y, mu)
    n, p = _nb_params(mu, alpha)
    return stats.nbinom.pmf(y, n, p)


def count_quantile(q, mu, alpha):
    mu = np.maximum(np.asarray(mu, float), 1e-12)
    if alpha <= 0:
        return stats.poisson.ppf(q, mu)
    n, p = _nb_params(mu, alpha)
    return stats.nbinom.ppf(q, n, p)


def nb_loglik(y, mu, alpha):
    if alpha <= 0:
        return float(np.sum(stats.poisson.logpmf(y, mu)))
    n, p = _nb_params(mu, alpha)
    return float(np.sum(stats.nbinom.logpmf(y, n, p)))


# ------------------------------------------------------ count model -----

@dataclass
class CountModel:
    metric: str
    kind: str                     # nb | poisson | pooled
    design: Design | None
    coef: np.ndarray | None
    coef_names: list[str]
    alpha: float                  # NB dispersion (0 for Poisson)
    n_windows: int
    n_events: int
    pooled_rate: float            # events per pair-hour
    dispersion_p: float
    notes: list = field(default_factory=list)

    def mu(self, df: pd.DataFrame) -> np.ndarray:
        """Expected count for each window given its conditions."""
        exposure = np.maximum(df["pair_hours"].to_numpy(float), 0.0)
        if self.kind == "pooled" or self.design is None:
            return self.pooled_rate * exposure
        X, _, unknown = self.design.matrix(df)
        if unknown:
            self.notes.append(f"{unknown} unknown covariate levels mapped "
                              "to the reference level at scoring")
        eta = X @ self.coef
        return np.exp(np.clip(eta, -30, 30)) * exposure

    def score(self, df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        """Phase II scoring (section 9.4)."""
        y = df[self.metric].to_numpy(float)
        mu = self.mu(df)
        cdf_y = count_cdf(y, mu, self.alpha)
        cdf_ym1 = np.where(y > 0, count_cdf(y - 1, mu, self.alpha), 0.0)
        pmf_y = np.clip(cdf_y - cdf_ym1, 0.0, 1.0)
        p_upper_mid = 1.0 - cdf_ym1 - 0.5 * pmf_y
        # randomized quantile residual: u ~ U(F(y-1), F(y)), seeded per
        # window so the score is reproducible
        rng = np.random.default_rng(seed)
        u = cdf_ym1 + rng.uniform(size=len(y)) * pmf_y
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = stats.norm.ppf(u)
        ok = mu > 0
        z = np.where(ok, z, np.nan)
        return pd.DataFrame({
            f"{self.metric}_expected": mu,
            f"{self.metric}_oe": np.where(mu > 0, y / np.maximum(mu, 1e-12),
                                          np.nan),
            f"{self.metric}_p_upper": np.where(ok, p_upper_mid, np.nan),
            f"{self.metric}_z": z,
            f"{self.metric}_pit": np.where(ok, u, np.nan),
            f"{self.metric}_exceed95": ok & (p_upper_mid
                                            < 1 - config.LIMIT_WARNING),
            f"{self.metric}_exceed99": ok & (p_upper_mid
                                            < 1 - config.LIMIT_ACTION),
            f"{self.metric}_limit95": count_quantile(config.LIMIT_WARNING,
                                                     mu, self.alpha),
            f"{self.metric}_limit99": count_quantile(config.LIMIT_ACTION,
                                                     mu, self.alpha),
        }, index=df.index)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric, "kind": self.kind,
            "design": None if self.design is None else {
                "hour_block_h": self.design.hour_block_h,
                "levels": self.design.levels,
                "factors": list(self.design.factors),
                "use_log_n": self.design.use_log_n,
                "merged": self.design.merged},
            "coef": None if self.coef is None else self.coef.tolist(),
            "coef_names": self.coef_names, "alpha": self.alpha,
            "n_windows": self.n_windows, "n_events": self.n_events,
            "pooled_rate": self.pooled_rate,
            "dispersion_p": self.dispersion_p, "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CountModel":
        design = None
        if d["design"]:
            design = Design(d["design"]["hour_block_h"],
                            d["design"]["levels"],
                            tuple(d["design"]["factors"]),
                            d["design"]["use_log_n"],
                            d["design"].get("merged", {}))
        return cls(d["metric"], d["kind"], design,
                   None if d["coef"] is None else np.array(d["coef"]),
                   d["coef_names"], d["alpha"], d["n_windows"],
                   d["n_events"], d["pooled_rate"], d["dispersion_p"],
                   list(d.get("notes", [])))


def _glm_fit(y, X, offset, alpha):
    import warnings
    import statsmodels.api as sm
    fam = (sm.families.Poisson() if alpha <= 0
           else sm.families.NegativeBinomial(alpha=alpha))
    model = sm.GLM(y, X, family=fam, offset=offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(maxiter=200, tol=1e-8)
    return res


def fit_count_model(df: pd.DataFrame, metric: str,
                    hour_block_h: int | None = None,
                    factors: tuple | None = None) -> CountModel:
    """Fit the conditional model for one count metric on baseline
    windows (exposure > 0). Negative binomial with the dispersion
    profiled by maximum likelihood; Poisson when the boundary
    likelihood-ratio test is not significant; pooled rate when the
    metric has too few events (section 9.2)."""
    d = df[df["pair_hours"] > 0]
    y = d[metric].to_numpy(float)
    exposure = d["pair_hours"].to_numpy(float)
    n_events = int(y.sum())
    pooled = n_events / exposure.sum() if exposure.sum() > 0 else 0.0
    notes = []
    if n_events < config.GLM_MIN_EVENTS or len(d) < 20:
        notes.append(f"pooled Poisson rate: {n_events} events < "
                     f"GLM_MIN_EVENTS={config.GLM_MIN_EVENTS}")
        return CountModel(metric, "pooled", None, None, [], 0.0, len(d),
                          n_events, pooled, np.nan, notes)

    hb = hour_block_h or config.BASELINE_HOUR_BLOCK_H
    factors = factors or ("hour_block", "daytype", "month", "flow")
    design = Design(hb, factors=factors)
    design.fit_levels(d, y)
    # coarsen hour blocks when some hour level has no events at all
    hv = design.factor_values(d)["hour_block"]
    empty = [lev for lev in np.unique(hv) if y[hv == lev].sum() == 0]
    if empty and hb < config.BASELINE_HOUR_BLOCK_FALLBACK_H:
        notes.append(f"hour blocks coarsened to "
                     f"{config.BASELINE_HOUR_BLOCK_FALLBACK_H} h: no events "
                     f"in hour levels {empty}")
        design = Design(config.BASELINE_HOUR_BLOCK_FALLBACK_H,
                        factors=factors)
        design.fit_levels(d, y)
    for f, merged in design.merged.items():
        notes.append(f"{f}: levels {merged} merged into the reference "
                     f"level '{design.levels[f][0]}' (too few windows or "
                     "no events)")
    # drop single-level factors (nothing to estimate)
    keep = tuple(f for f in factors if len(design.levels[f]) > 1)
    if keep != factors:
        notes.append(f"factors without variation dropped: "
                     f"{[f for f in factors if f not in keep]}")
        design = Design(design.hour_block_h, factors=keep)
        design.fit_levels(d, y)
    X, names, _ = design.matrix(d)
    # aliasing (e.g. every weekend window is also a south-flow window on
    # a short dataset) makes the design rank-deficient and the fitted
    # coefficients arbitrary; drop factors, least important first, until
    # the design has full rank
    for drop in ("month", "flow", "daytype", "hour_block"):
        if np.linalg.matrix_rank(X) == X.shape[1]:
            break
        if drop in design.factors:
            notes.append(f"design rank-deficient (aliased factors): "
                         f"'{drop}' dropped")
            design = Design(design.hour_block_h,
                            factors=tuple(f for f in design.factors
                                          if f != drop))
            design.fit_levels(d, y)
            X, names, _ = design.matrix(d)
    offset = np.log(exposure)

    try:
        pois = _glm_fit(y, X, offset, 0.0)
        mu_p = pois.fittedvalues
        ll_p = nb_loglik(y, mu_p, 0.0)

        def neg_ll(log_alpha):
            a = float(np.exp(log_alpha))
            try:
                res = _glm_fit(y, X, offset, a)
            except Exception:
                return 1e12
            return -nb_loglik(y, res.fittedvalues, a)

        opt = optimize.minimize_scalar(neg_ll, bounds=(np.log(1e-4),
                                                       np.log(50.0)),
                                       method="bounded",
                                       options={"xatol": 1e-3})
        alpha = float(np.exp(opt.x))
        ll_nb = -float(opt.fun)
        lr = max(0.0, 2.0 * (ll_nb - ll_p))
        p_disp = 0.5 * stats.chi2.sf(lr, 1)        # boundary test
        if p_disp < config.BASELINE_DISPERSION_ALPHA:
            res = _glm_fit(y, X, offset, alpha)
            kind = "nb"
        else:
            res, alpha, kind = pois, 0.0, "poisson"
            notes.append(f"Poisson: dispersion not significant "
                         f"(p={p_disp:.3f})")
        coef = np.asarray(res.params, float)
        if not np.all(np.isfinite(coef)):
            raise ValueError("non-finite coefficients")
        if np.abs(coef[1:]).max(initial=0.0) > 8.0:
            raise ValueError("implausible coefficient magnitude "
                             f"({np.abs(coef[1:]).max():.1f})")
    except Exception as e:  # fall back to a simpler design, then pooled
        simpler = {"month": ("hour_block", "daytype", "flow"),
                   "flow": ("hour_block", "daytype"),
                   "daytype": ("hour_block",)}
        for drop, fac in simpler.items():
            if drop in factors:
                m = fit_count_model(df, metric, hour_block_h, fac)
                m.notes.insert(0, f"design simplified (dropped {drop}): {e}")
                return m
        notes.append(f"regression failed ({e}); pooled rate used")
        return CountModel(metric, "pooled", None, None, [], 0.0, len(d),
                          n_events, pooled, np.nan, notes)
    return CountModel(metric, kind, design, coef, names, alpha, len(d),
                      n_events, pooled, float(p_disp), notes)


def phase1(df: pd.DataFrame, metric: str) -> tuple[CountModel, pd.DataFrame]:
    """Phase I (section 9.3): fit, remove windows beyond the 99.9 %
    predictive limit, refit; at most PHASE1_MAX_ITER iterations and
    PHASE1_MAX_REMOVED_FRAC of the windows."""
    work = df.copy()
    removed = []
    budget = int(np.floor(config.PHASE1_MAX_REMOVED_FRAC * len(work)))
    model = fit_count_model(work, metric)
    for it in range(config.PHASE1_MAX_ITER):
        if budget <= 0:
            break
        sub = work[work["pair_hours"] > 0]
        mu = model.mu(sub)
        p_upper = 1.0 - count_cdf(sub[metric].to_numpy(float) - 1, mu,
                                  model.alpha)
        # beyond the limit: P(Y >= y) < 1 - q
        beyond = np.flatnonzero(p_upper < 1.0 - config.PHASE1_QUANTILE)
        if len(beyond) == 0:
            break
        # most extreme first, within the budget
        beyond = beyond[np.argsort(p_upper[beyond])][:budget]
        idx = sub.index[beyond]
        rem = work.loc[idx].copy()
        rem["phase1_iteration"] = it + 1
        rem["expected"] = mu[beyond]
        rem["p_upper"] = p_upper[beyond]
        rem["metric"] = metric
        removed.append(rem)
        budget -= len(idx)
        work = work.drop(idx)
        model = fit_count_model(work, metric)
    rem_df = (pd.concat(removed) if removed
              else pd.DataFrame(columns=list(df.columns)
                                + ["phase1_iteration", "expected",
                                   "p_upper", "metric"]))
    return model, rem_df


# ------------------------------------------------ continuous metrics ----

def strata_keys(df: pd.DataFrame, terciles: np.ndarray) -> np.ndarray:
    hb = (df["local_hour"] // config.STRATA_HOUR_BLOCK_H).astype(int)
    tt = np.digitize(df["pair_hours"].to_numpy(float), terciles)
    return np.array([f"h{h}|t{t}|{f}" for h, t, f in
                     zip(hb, tt, df["flow"].astype(str))], dtype=object)


@dataclass
class ContinuousModel:
    metric: str
    higher_is_worse: bool
    terciles: list
    strata: dict          # key -> sorted values (list)
    pooled: list          # all values, sorted

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        v = df[self.metric].to_numpy(float)
        if self.metric == "s_min":
            v = np.where(np.isnan(v), S_MIN_CAP, np.minimum(v, S_MIN_CAP))
        keys = strata_keys(df, np.array(self.terciles))
        frac = np.full(len(v), np.nan)
        n_used = np.zeros(len(v), dtype=int)
        pooled = np.array(self.pooled)
        for i, (val, k) in enumerate(zip(v, keys)):
            if np.isnan(val):
                continue
            ref = np.array(self.strata.get(k, []))
            if len(ref) < config.SUFFICIENCY_MIN_WINDOWS_PER_STRATUM:
                ref = pooled
            if len(ref) == 0:
                continue
            n_used[i] = len(ref)
            # mid-rank empirical CDF
            lo = np.searchsorted(ref, val, side="left")
            hi = np.searchsorted(ref, val, side="right")
            frac[i] = (lo + 0.5 * (hi - lo) + 0.5) / (len(ref) + 1)
        worse = frac if self.higher_is_worse else 1.0 - frac
        worse = np.clip(worse, 1e-6, 1 - 1e-6)
        return pd.DataFrame({
            f"{self.metric}_pct": worse,
            f"{self.metric}_z": stats.norm.ppf(worse),
            f"{self.metric}_exceed95": worse > config.LIMIT_WARNING,
            f"{self.metric}_exceed99": worse > config.LIMIT_ACTION,
            f"{self.metric}_ref_n": n_used,
        }, index=df.index)

    def to_dict(self):
        return {"metric": self.metric, "higher_is_worse": self.higher_is_worse,
                "terciles": self.terciles, "strata": self.strata,
                "pooled": self.pooled}

    @classmethod
    def from_dict(cls, d):
        return cls(d["metric"], d["higher_is_worse"], d["terciles"],
                   d["strata"], d["pooled"])


def fit_continuous(df: pd.DataFrame, metric: str,
                   terciles: np.ndarray) -> ContinuousModel:
    d = df[df["pair_hours"] > 0]
    v = d[metric].to_numpy(float)
    if metric == "s_min":
        v = np.where(np.isnan(v), S_MIN_CAP, np.minimum(v, S_MIN_CAP))
    keys = strata_keys(d, terciles)
    ok = np.isfinite(v)
    strata = {}
    for k in np.unique(keys[ok]):
        strata[str(k)] = np.sort(v[ok & (keys == k)]).tolist()
    return ContinuousModel(metric, metric != "s_min", terciles.tolist(),
                           strata, np.sort(v[ok]).tolist())


# ------------------------------------------------------ cross-check ----

def crosscheck(model: CountModel, df: pd.DataFrame,
               terciles: np.ndarray) -> pd.DataFrame:
    """Model-based vs non-parametric 95 % limits per stratum: the mean
    model limit against the empirical 95th percentile of the counts."""
    d = df[df["pair_hours"] > 0]
    keys = strata_keys(d, terciles)
    lim = count_quantile(config.LIMIT_WARNING, model.mu(d), model.alpha)
    y = d[model.metric].to_numpy(float)
    rows = []
    for k in np.unique(keys):
        m = keys == k
        if m.sum() < config.SUFFICIENCY_MIN_WINDOWS_PER_STRATUM:
            continue
        emp = float(np.percentile(y[m], 100 * config.LIMIT_WARNING))
        mod = float(np.mean(lim[m]))
        diff = abs(emp - mod)
        agree = diff <= max(1.0, config.CROSSCHECK_MAX_REL_DIFF
                            * max(emp, mod))
        rows.append({"metric": model.metric, "stratum": k,
                     "n_windows": int(m.sum()), "empirical_limit95": emp,
                     "model_limit95_mean": mod, "agree": agree})
    return pd.DataFrame(rows, columns=["metric", "stratum", "n_windows",
                                       "empirical_limit95",
                                       "model_limit95_mean", "agree"])


# ------------------------------------------------------- validation ----

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def fold_labels(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Leave-one-month-out folds, or weeks if the dataset is shorter than
    3 months, or days if shorter than 2 weeks."""
    t = pd.to_datetime(df["t_start"], unit="s", utc=True)
    months = df["month"].astype(str).to_numpy()
    if len(np.unique(months)) >= config.VALIDATION_MIN_MONTHS_FOR_LOMO:
        return months, "month"
    weeks = t.dt.strftime("%G-W%V").to_numpy()
    if len(np.unique(weeks)) >= 2:
        return weeks, "week"
    return df["day"].astype(str).to_numpy(), "day"


def out_of_sample(df: pd.DataFrame, metric: str, seed: int) -> dict:
    """Refit without each fold, score the held-out windows; exceedance
    rates against the nominal 5 % / 1 % with binomial (Wilson) CIs, and
    the PIT values for the histogram / QQ-plot."""
    d = df[df["pair_hours"] > 0]
    folds, fold_kind = fold_labels(d)
    z_all, pit_all, ex95, ex99, held = [], [], [], [], []
    for f in np.unique(folds):
        train = d[folds != f]
        test = d[folds == f]
        if len(train) < 20 or test.empty:
            continue
        model = fit_count_model(train, metric)
        sc = model.score(test, seed=seed)
        z_all.append(sc[f"{metric}_z"].to_numpy())
        pit_all.append(sc[f"{metric}_pit"].to_numpy())
        ex95.append(sc[f"{metric}_exceed95"].to_numpy())
        ex99.append(sc[f"{metric}_exceed99"].to_numpy())
        held.append(test.index.to_numpy())
    if not z_all:
        return {"fold_kind": fold_kind, "n_folds": 0, "n": 0}
    z = np.concatenate(z_all); pit = np.concatenate(pit_all)
    e95 = np.concatenate(ex95); e99 = np.concatenate(ex99)
    ok = np.isfinite(z)
    n = int(ok.sum())
    k95, k99 = int(e95[ok].sum()), int(e99[ok].sum())
    r95, r99 = (k95 / n if n else np.nan), (k99 / n if n else np.nan)
    lo95, hi95 = wilson(k95, n); lo99, hi99 = wilson(k99, n)
    ks = stats.kstest(pit[ok], "uniform") if n > 5 else None
    return {
        "fold_kind": fold_kind, "n_folds": len(z_all), "n": n,
        "exceed95_rate": r95, "exceed95_ci": [lo95, hi95],
        "exceed99_rate": r99, "exceed99_ci": [lo99, hi99],
        "pass95": bool(config.VALIDATION_PASS_95[0] <= r95
                       <= config.VALIDATION_PASS_95[1]) if n else False,
        "pass99": bool(config.VALIDATION_PASS_99[0] <= r99
                       <= config.VALIDATION_PASS_99[1]) if n else False,
        "pit_ks_p": float(ks.pvalue) if ks else np.nan,
        "z": z[ok].tolist(), "pit": pit[ok].tolist(),
        "held_index": np.concatenate(held)[ok].tolist(),
    }


def stability_curve(df: pd.DataFrame, metric: str, seed: int) -> dict:
    """Refit on random subsets of days (bootstrap by day) at growing
    fractions; report the pooled rate and the mean 95 % limit, and the
    number of days at which they stabilise within +/- 5 % of the
    full-data value."""
    d = df[df["pair_hours"] > 0]
    days = np.array(sorted(d["day"].unique()))
    rng = np.random.default_rng(seed)
    full = fit_count_model(d, metric)
    full_rate = full.pooled_rate
    full_lim = float(np.mean(count_quantile(config.LIMIT_WARNING,
                                            full.mu(d), full.alpha)))
    rows = []
    for frac in config.STABILITY_FRACTIONS:
        k = max(2, int(round(frac * len(days))))
        if k > len(days):
            continue
        rates, lims = [], []
        for r in range(config.STABILITY_REPS if frac < 1 else 1):
            pick = (days if frac >= 1
                    else rng.choice(days, k, replace=False))
            sub = d[d["day"].isin(pick)]
            if sub[metric].sum() == 0:
                continue
            m = fit_count_model(sub, metric)
            rates.append(m.pooled_rate)
            lims.append(float(np.mean(count_quantile(
                config.LIMIT_WARNING, m.mu(d), m.alpha))))
        if rates:
            rows.append({"fraction": frac, "n_days": k,
                         "rate_mean": float(np.mean(rates)),
                         "rate_lo": float(np.percentile(rates, 2.5)),
                         "rate_hi": float(np.percentile(rates, 97.5)),
                         "limit95_mean": float(np.mean(lims)),
                         "limit95_lo": float(np.percentile(lims, 2.5)),
                         "limit95_hi": float(np.percentile(lims, 97.5))})
    table = pd.DataFrame(rows)
    stable_days = None
    if len(table):
        tol = config.STABILITY_PLATEAU_TOL
        within = ((np.abs(table["rate_mean"] - full_rate)
                   <= tol * max(full_rate, 1e-12))
                  & (np.abs(table["limit95_mean"] - full_lim)
                     <= tol * max(full_lim, 1e-12))).to_numpy()
        # first fraction from which every larger fraction is within tol
        for i in range(len(within)):
            if within[i:].all():
                stable_days = int(table["n_days"].iat[i])
                break
    return {"table": table, "full_rate": full_rate, "full_limit95": full_lim,
            "stable_from_days": stable_days, "n_days": int(len(days))}


def sufficiency_table(df: pd.DataFrame, metric: str,
                      terciles: np.ndarray) -> pd.DataFrame:
    d = df[df["pair_hours"] > 0]
    keys = strata_keys(d, terciles)
    rows = []
    for k in np.unique(keys):
        m = keys == k
        rows.append({"stratum": k, "n_windows": int(m.sum()),
                     "n_events": int(d[metric].to_numpy()[m].sum()),
                     "pair_hours": float(d["pair_hours"].to_numpy()[m].sum()),
                     "flagged": bool(m.sum()
                                     < config.SUFFICIENCY_MIN_WINDOWS_PER_STRATUM)})
    return pd.DataFrame(rows)


def day_block_bootstrap_rate(df: pd.DataFrame, metric: str, reps: int,
                             seed: int) -> tuple[float, float, float]:
    """Pooled rate (events per pair-hour) with a day-block bootstrap CI."""
    d = df[df["pair_hours"] > 0]
    by_day = d.groupby("day").agg(y=(metric, "sum"), e=("pair_hours", "sum"))
    y = by_day["y"].to_numpy(float); e = by_day["e"].to_numpy(float)
    rate = y.sum() / e.sum() if e.sum() > 0 else np.nan
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 2:
        return rate, np.nan, np.nan
    idx = rng.integers(0, n, size=(reps, n))
    boot = y[idx].sum(axis=1) / np.maximum(e[idx].sum(axis=1), 1e-12)
    return rate, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ----------------------------------------------------------- baseline ----

BASELINE_PARAMS = (
    "WINDOW_MIN", "BASELINE_EXCLUSION_MIN", "GLM_MIN_EVENTS",
    "BASELINE_HOUR_BLOCK_H", "BASELINE_HOUR_BLOCK_FALLBACK_H",
    "BASELINE_DISPERSION_ALPHA", "PHASE1_QUANTILE", "PHASE1_MAX_ITER",
    "PHASE1_MAX_REMOVED_FRAC", "STRATA_HOUR_BLOCK_H", "TIERS",
    "T_LOOKAHEAD_S", "VOLUME_RADIUS_NM", "EPISODE_MERGE_GAP_S",
    "QUALITY_DEGRADED_INTERP_FRAC", "QUALITY_BAD_INTERP_FRAC",
    "QUALITY_INTERP_GAP_S",
    "QUALITY_DEGRADED_GAPS_PER_AC", "QUALITY_BAD_GAPS_PER_AC",
    "QUALITY_OUTAGE_MIN_S", "FLOW_LOOKAROUND_MIN", "FLOW_DOMINANCE_SHARE",
    "FLOWS", "INNER_RADIUS_NM",
)


@dataclass
class Baseline:
    scope: str                       # airspace | ga_involved
    count_models: dict               # metric -> CountModel
    continuous_models: dict          # metric -> ContinuousModel
    terciles: list
    exclusion_log: dict
    validation: dict                 # per metric summary (+ verdict)
    verdict: str
    warnings: list
    info: dict                       # dates, days, windows, hash
    phase1_removed: pd.DataFrame
    windows: pd.DataFrame            # scored baseline windows

    def score_counts(self, df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        parts = [m.score(df, seed=seed) for m in self.count_models.values()]
        parts += [m.score(df) for m in self.continuous_models.values()]
        return pd.concat(parts, axis=1)

    def to_json(self) -> dict:
        return {
            "scope": self.scope,
            "count_models": {k: m.to_dict() for k, m in
                             self.count_models.items()},
            "continuous_models": {k: m.to_dict() for k, m in
                                  self.continuous_models.items()},
            "terciles": self.terciles,
            "exclusion_log": self.exclusion_log,
            "validation": {k: {kk: vv for kk, vv in v.items()
                               if kk not in ("z", "pit", "held_index",
                                             "table")}
                           for k, v in self.validation.items()},
            "verdict": self.verdict, "warnings": self.warnings,
            "info": self.info,
        }


def baseline_param_hash(extra: dict | None = None) -> str:
    from .pipeline import param_hash, day_param_hash
    d = {"day": day_param_hash(), "base": param_hash(BASELINE_PARAMS),
         "version": BASELINE_VERSION}
    if extra:
        d.update(extra)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()
                          ).hexdigest()[:16]


def exclude_windows(clock: pd.DataFrame, event_t0: np.ndarray,
                    exclusion_min: float) -> tuple[pd.DataFrame, dict]:
    """Section 9.1: drop windows overlapping +/- exclusion_min of any
    go-around (or ambiguous), bad-quality windows; log every reason."""
    ex = int(exclusion_min * 60)
    t_ev = np.sort(np.asarray(event_t0, np.int64))
    near = np.zeros(len(clock), dtype=bool)
    if len(t_ev):
        lo = np.searchsorted(t_ev, clock["t_start"].to_numpy() - ex)
        hi = np.searchsorted(t_ev, clock["t_end"].to_numpy() + ex,
                             side="right")
        near = (hi - lo) > 0
    bad = (clock["quality"] == "bad").to_numpy()
    partial = clock["partial_coverage"].to_numpy(bool)
    zero = (clock["pair_hours"] <= 0).to_numpy()
    keep = ~near & ~bad & ~partial
    log = {
        "n_windows": int(len(clock)),
        "excluded_near_go_around": int(near.sum()),
        "excluded_bad_quality": int((bad & ~near).sum()),
        "excluded_partial_coverage": int((partial & ~near & ~bad).sum()),
        "kept": int(keep.sum()),
        "kept_zero_exposure": int((keep & zero).sum()),
        "kept_used_in_rate_models": int((keep & ~zero).sum()),
        "quality_counts": clock["quality"].value_counts().to_dict(),
    }
    return clock[keep].copy(), log


def build_baseline(clock: pd.DataFrame, event_t0: np.ndarray, tiers,
                   scope: str = "airspace", validate: bool = True,
                   exclusion_min: float | None = None,
                   quiet: bool = False) -> Baseline:
    """Phase I baseline for every metric plus validation (section 9)."""
    exclusion_min = (config.BASELINE_EXCLUSION_MIN if exclusion_min is None
                     else exclusion_min)
    base, log = exclude_windows(clock, event_t0, exclusion_min)
    if not quiet:
        print(f"baseline [{scope}]: {log['n_windows']} windows, "
              f"{log['excluded_near_go_around']} near go-arounds, "
              f"{log['excluded_bad_quality']} bad quality, "
              f"{log['excluded_partial_coverage']} partial -> "
              f"{log['kept']} kept ({log['kept_used_in_rate_models']} with "
              f"exposure)")
    with_exp = base[base["pair_hours"] > 0]
    terciles = (np.percentile(with_exp["pair_hours"], [100 / 3, 200 / 3])
                if len(with_exp) else np.array([0.0, 0.0]))
    warnings: list[str] = []
    n_days = int(base["day"].nunique())
    if n_days < config.MIN_BASELINE_DAYS:
        warnings.append(f"only {n_days} days of baseline data "
                        f"(minimum {config.MIN_BASELINE_DAYS}, recommended "
                        f"{config.RECOMMENDED_BASELINE_DAYS})")

    count_models, removed, validation = {}, [], {}
    seed = config.RANDOM_SEED
    for metric in count_metrics(tiers):
        if metric not in base.columns:
            continue
        model, rem = phase1(base, metric)
        count_models[metric] = model
        if len(rem):
            removed.append(rem)
        v = {"kind": model.kind, "n_events": model.n_events,
             "notes": model.notes}
        if model.kind != "pooled":
            xc = crosscheck(model, base, terciles)
            v["crosscheck_n"] = int(len(xc))
            v["crosscheck_disagree"] = int((~xc["agree"]).sum()) if len(xc) else 0
            v["crosscheck_table"] = xc
            if len(xc) and v["crosscheck_disagree"] > 0.2 * len(xc):
                warnings.append(f"{metric}: model and non-parametric 95% "
                                f"limits disagree in {v['crosscheck_disagree']}"
                                f"/{len(xc)} strata")
        rate, lo, hi = day_block_bootstrap_rate(base, metric,
                                                config.BOOTSTRAP_REPS, seed)
        v["rate_per_pair_hour"] = rate
        v["rate_ci"] = [lo, hi]
        if validate and model.kind != "pooled":
            oos = out_of_sample(base, metric, seed)
            v.update({k: val for k, val in oos.items()})
            if metric == PRIMARY_METRIC and oos.get("n", 0):
                if not oos["pass95"]:
                    warnings.append(
                        f"{metric}: out-of-sample 95% exceedance "
                        f"{100 * oos['exceed95_rate']:.1f}% outside "
                        f"[{100 * config.VALIDATION_PASS_95[0]:.0f}%, "
                        f"{100 * config.VALIDATION_PASS_95[1]:.0f}%]")
                if not oos["pass99"]:
                    warnings.append(
                        f"{metric}: out-of-sample 99% exceedance "
                        f"{100 * oos['exceed99_rate']:.2f}% outside "
                        f"[{100 * config.VALIDATION_PASS_99[0]:.1f}%, "
                        f"{100 * config.VALIDATION_PASS_99[1]:.0f}%]")
            if metric == PRIMARY_METRIC:
                v["stability"] = stability_curve(base, metric, seed)
                if v["stability"]["stable_from_days"] is None:
                    warnings.append(f"{metric}: baseline estimates have not "
                                    f"stabilised within +/-"
                                    f"{100 * config.STABILITY_PLATEAU_TOL:.0f}%"
                                    f" over the available days")
        if metric == PRIMARY_METRIC:
            suff = sufficiency_table(base, metric, terciles)
            v["sufficiency_table"] = suff
            n_flag = int(suff["flagged"].sum()) if len(suff) else 0
            if n_flag:
                warnings.append(f"{n_flag}/{len(suff)} strata have fewer "
                                f"than {config.SUFFICIENCY_MIN_WINDOWS_PER_STRATUM}"
                                f" windows")
        validation[metric] = v
        if not quiet:
            print(f"  {metric}: {model.kind} (events={model.n_events}, "
                  f"alpha={model.alpha:.3f})"
                  + (f"  oos 95%={100 * v['exceed95_rate']:.1f}% "
                     f"99%={100 * v['exceed99_rate']:.2f}% "
                     f"[{v['fold_kind']}]" if "exceed95_rate" in v else ""))

    continuous_models = {}
    for metric in continuous_metrics(tiers):
        if metric in base.columns and base[metric].notna().any():
            continuous_models[metric] = fit_continuous(base, metric, terciles)

    # verdict
    prim = validation.get(PRIMARY_METRIC, {})
    if not count_models or PRIMARY_METRIC not in count_models:
        verdict = "NOT VALID"
        warnings.append("primary metric could not be modelled")
    elif prim.get("n", 0) and not prim.get("pass95") and not prim.get("pass99"):
        verdict = "NOT VALID"
    elif warnings:
        verdict = "VALID WITH WARNINGS"
    else:
        verdict = "VALID"

    scored = base.copy()
    bl = Baseline(scope, count_models, continuous_models, terciles.tolist(),
                  log, validation, verdict, warnings,
                  {"n_days": n_days, "n_windows": int(len(base)),
                   "date_first": str(base["day"].min()) if len(base) else "",
                   "date_last": str(base["day"].max()) if len(base) else "",
                   "exclusion_min": exclusion_min},
                  pd.concat(removed) if removed else pd.DataFrame(),
                  scored)
    sc = bl.score_counts(scored, seed=seed)
    bl.windows = pd.concat([scored, sc], axis=1)
    return bl


# ----------------------------------------------------------- storage ----

def save_baseline(bl: Baseline, folder: Path, error_model: pd.DataFrame,
                  param_hash: str, dataset_signature: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    bl.info["param_hash"] = param_hash
    bl.info["dataset_signature"] = dataset_signature
    (folder / "baseline.json").write_text(
        json.dumps(bl.to_json(), indent=1, default=_json_default))
    bl.windows.to_parquet(folder / "baseline_windows.parquet", index=False)
    if error_model is not None and len(error_model):
        # only with the optional probability add-on (--probability)
        error_model.to_csv(folder / "error_model.csv", index=False)
    bl.phase1_removed.to_csv(folder / "phase1_removed.csv", index=False)
    val = folder / "validation"
    val.mkdir(exist_ok=True)
    for metric, v in bl.validation.items():
        if "crosscheck_table" in v and len(v["crosscheck_table"]):
            v["crosscheck_table"].to_csv(val / f"crosscheck_{metric}.csv",
                                         index=False)
        if "sufficiency_table" in v:
            v["sufficiency_table"].to_csv(val / f"sufficiency_{metric}.csv",
                                          index=False)
        if "stability" in v and len(v["stability"]["table"]):
            v["stability"]["table"].to_csv(val / f"stability_{metric}.csv",
                                           index=False)
        if "z" in v:
            pd.DataFrame({"z": v["z"], "pit": v["pit"]}).to_csv(
                val / f"held_out_scores_{metric}.csv", index=False)
    rows = []
    for metric, v in bl.validation.items():
        rows.append({"metric": metric, "model": v.get("kind"),
                     "n_events": v.get("n_events"),
                     "rate_per_pair_hour": v.get("rate_per_pair_hour"),
                     "rate_ci_lo": (v.get("rate_ci") or [np.nan])[0],
                     "rate_ci_hi": (v.get("rate_ci") or [np.nan, np.nan])[1],
                     "fold_kind": v.get("fold_kind"),
                     "n_held_out": v.get("n"),
                     "exceed95_rate": v.get("exceed95_rate"),
                     "exceed95_ci_lo": (v.get("exceed95_ci") or [np.nan])[0],
                     "exceed95_ci_hi": (v.get("exceed95_ci") or [np.nan, np.nan])[1],
                     "exceed99_rate": v.get("exceed99_rate"),
                     "exceed99_ci_lo": (v.get("exceed99_ci") or [np.nan])[0],
                     "exceed99_ci_hi": (v.get("exceed99_ci") or [np.nan, np.nan])[1],
                     "pass95": v.get("pass95"), "pass99": v.get("pass99"),
                     "pit_ks_p": v.get("pit_ks_p"),
                     "crosscheck_disagree": v.get("crosscheck_disagree"),
                     "crosscheck_n": v.get("crosscheck_n")})
    pd.DataFrame(rows).to_csv(val / "calibration_summary.csv", index=False)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return None
    return str(o)


def load_baseline(folder: Path) -> Baseline:
    d = json.loads((folder / "baseline.json").read_text())
    windows = pd.read_parquet(folder / "baseline_windows.parquet")
    removed_path = folder / "phase1_removed.csv"
    removed = (pd.read_csv(removed_path) if removed_path.stat().st_size > 1
               else pd.DataFrame())
    validation = d["validation"]
    val = folder / "validation"
    for metric, v in validation.items():
        p = val / f"held_out_scores_{metric}.csv"
        if p.exists():
            t = pd.read_csv(p)
            v["z"] = t["z"].tolist(); v["pit"] = t["pit"].tolist()
        p = val / f"stability_{metric}.csv"
        if p.exists():
            v.setdefault("stability", {})["table"] = pd.read_csv(p)
        p = val / f"sufficiency_{metric}.csv"
        if p.exists():
            v["sufficiency_table"] = pd.read_csv(p)
        p = val / f"crosscheck_{metric}.csv"
        if p.exists():
            v["crosscheck_table"] = pd.read_csv(p)
    return Baseline(
        d["scope"],
        {k: CountModel.from_dict(m) for k, m in d["count_models"].items()},
        {k: ContinuousModel.from_dict(m)
         for k, m in d["continuous_models"].items()},
        d["terciles"], d["exclusion_log"], validation, d["verdict"],
        d["warnings"], d["info"], removed, windows)


def baseline_is_reusable(folder: Path, param_hash: str,
                         dataset_signature: str) -> bool:
    p = folder / "baseline.json"
    if not p.exists():
        return False
    try:
        info = json.loads(p.read_text())["info"]
    except Exception:
        return False
    return (info.get("param_hash") == param_hash
            and info.get("dataset_signature") == dataset_signature)
