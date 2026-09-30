"""
All figures (FRAMEWORK.md section 16), in the go-around pipeline's style:
light background, minimal chart junk, one question per figure, at most
four panels, labelled axes with units, and fixed colours for pre, post,
baseline and the three tiers across every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config
from . import goaround_adapter as ga
from .baseline import PRIMARY_METRIC, count_pmf

# fixed colours (validated categorical slots): pre, post, baseline, tiers
PRE, POST, BASE = "#2a78d6", "#eb6834", "#7a7975"
TIER = {"T1": "#1baf7a", "T2": "#eda100", "T3": "#e34948"}
INK, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
SEQ = "Blues"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": RULE, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
DPI = config.FIGURE_DPI


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=11)


# ------------------------------------------------ baseline validation ----

def baseline_figures(bl, calib: dict, em_table: pd.DataFrame,
                     outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    v = bl.validation.get(PRIMARY_METRIC, {})

    # 1. out-of-sample exceedance vs nominal
    rows = [(m, bl.validation[m]) for m in bl.validation
            if "exceed95_rate" in bl.validation[m]
            and bl.validation[m].get("n", 0)]
    if rows:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        y = np.arange(len(rows))
        for i, (m, r) in enumerate(rows):
            for off, key, nominal, col in ((-0.18, "95", 0.05, PRE),
                                           (0.18, "99", 0.01, POST)):
                rate = r[f"exceed{key}_rate"]; lo, hi = r[f"exceed{key}_ci"]
                ax.errorbar(100 * rate, i + off, xerr=[[100 * (rate - lo)],
                                                         [100 * (hi - rate)]],
                            fmt="o", color=col, ms=6, capsize=3, lw=1.2)
        ax.axvline(5, color=PRE, lw=1, ls="--")
        ax.axvline(1, color=POST, lw=1, ls="--")
        ax.axvspan(100 * config.VALIDATION_PASS_95[0],
                   100 * config.VALIDATION_PASS_95[1], color=PRE, alpha=0.08)
        ax.axvspan(100 * config.VALIDATION_PASS_99[0],
                   100 * config.VALIDATION_PASS_99[1], color=POST, alpha=0.08)
        ax.set_yticks(y, [m for m, _ in rows]); ax.invert_yaxis()
        ax.set_xlabel("held-out windows beyond the limit (%)")
        ax.plot([], [], "o", color=PRE, label="95 % limit (nominal 5 %)")
        ax.plot([], [], "o", color=POST, label="99 % limit (nominal 1 %)")
        ax.legend(fontsize=8, loc="lower right")
        _title(ax, f"Out-of-sample calibration (leave-one-{v.get('fold_kind', '?')}"
                   "-out): shaded = pass range")
        _save(fig, outdir / "calibration_exceedance.png")

    # 2. PIT histogram and z QQ-plot of held-out windows
    if v.get("z"):
        z = np.asarray(v["z"]); pit = np.asarray(v["pit"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        ax1.hist(pit, bins=10, range=(0, 1), color=PRE, edgecolor=SURFACE,
                 density=True)
        ax1.axhline(1, color=BASE, lw=1.2, ls="--")
        ax1.set_xlabel("probability integral transform of held-out windows")
        ax1.set_ylabel("density")
        _title(ax1, "PIT histogram (uniform if the baseline is calibrated)")
        (osm, osr), _ = sps.probplot(z, dist="norm")
        ax2.plot(osm, osr, ".", color=PRE, ms=4)
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        ax2.plot(lim, lim, color=BASE, lw=1.2, ls="--")
        ax2.set_xlabel("theoretical N(0, 1) quantile")
        ax2.set_ylabel("held-out z-score")
        _title(ax2, f"QQ-plot of z-scores (KS p = {v.get('pit_ks_p', np.nan):.2f})")
        _save(fig, outdir / "calibration_pit_qq.png")

    # 3. stability curve
    st = v.get("stability", {})
    tab = st.get("table")
    if tab is not None and len(tab):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, col, lab in ((ax1, "rate", "baseline rate (encounters per "
                                            "pair-hour)"),
                             (ax2, "limit95", "mean 95 % limit (encounters "
                                              "per window)")):
            ax.fill_between(tab["n_days"], tab[f"{col}_lo"], tab[f"{col}_hi"],
                            color=PRE, alpha=0.15, lw=0)
            ax.plot(tab["n_days"], tab[f"{col}_mean"], "o-", color=PRE, ms=5)
            full = st.get("full_rate" if col == "rate" else "full_limit95")
            if full is not None:
                ax.axhspan(full * (1 - config.STABILITY_PLATEAU_TOL),
                           full * (1 + config.STABILITY_PLATEAU_TOL),
                           color=BASE, alpha=0.15, lw=0)
            ax.set_xlabel("days used (random subsets)")
            ax.set_ylabel(lab)
            ax.set_ylim(bottom=0)
        stable = st.get("stable_from_days")
        _title(ax1, "Stability: estimate vs days of data (band = "
                    f"{100 * config.STABILITY_PLATEAU_TOL:.0f} % of full value)")
        _title(ax2, "Stabilises from "
                    f"{stable if stable is not None else 'not reached'} days")
        _save(fig, outdir / "stability_curve.png")

    # 4. reliability diagram (T1) and error growth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    c = calib.get("T1", {})
    tab = c.get("table")
    if tab is not None and len(tab):
        t = tab[tab["n"] > 0]
        ax1.plot([0, 1], [0, 1], color=BASE, lw=1.2, ls="--")
        ax1.plot(t["mean_pred"], t["obs_frac"], "o-", color=TIER["T1"], ms=6)
        for _, r in t.iterrows():
            ax1.annotate(f"n={int(r['n']):,}", (r["mean_pred"], r["obs_frac"]),
                         xytext=(4, -10), textcoords="offset points",
                         fontsize=7, color=MUTED)
        ax1.set_xlabel("predicted conflict probability (T1)")
        ax1.set_ylabel("observed fraction entering T1 within lookahead")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        label = ("poorly calibrated" if c.get("ece", 1) > config.PROB_MAX_ECE
                 else "calibrated")
        _title(ax1, f"Reliability: ECE {c.get('ece', np.nan):.3f}, Brier skill "
                    f"{c.get('bss', np.nan):.2f} ({label})")
    if len(em_table):
        allrows = em_table[em_table["phase"] == "all"]
        ax2.plot(allrows["tau_s"], allrows["sigma_along_nm"] * ga.FT_PER_NM,
                 "o-", color=PRE, ms=4, label="along-track")
        ax2.plot(allrows["tau_s"], allrows["sigma_cross_nm"] * ga.FT_PER_NM,
                 "s-", color=POST, ms=4, label="cross-track")
        ax2.plot(allrows["tau_s"], allrows["sigma_z_ft"], "^-", color=BASE,
                 ms=4, label="vertical")
        ax2.set_xlabel("prediction horizon tau (s)")
        ax2.set_ylabel("robust error sigma (ft)")
        ax2.set_ylim(bottom=0)
        ax2.legend(fontsize=8)
        _title(ax2, "Straight-line prediction error growth (all phases)")
    _save(fig, outdir / "probability_model.png")


# --------------------------------------------------- study figures -------

def fig_epoch(epoch: pd.DataFrame, out: Path, n_events: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    if len(epoch):
        ok = epoch["expected"] > 0
        ax.fill_between(epoch["minute"][ok], epoch["oe_lo"][ok],
                        epoch["oe_hi"][ok], color=POST, alpha=0.15, lw=0,
                        label="95 % day-block bootstrap band")
        ax.plot(epoch["minute"][ok], epoch["oe"][ok], color=POST, lw=2,
                label="observed / expected (all events pooled)")
    ax.axhline(1, color=BASE, lw=1.2, ls="--", label="baseline (O/E = 1)")
    ax.axvline(0, color=INK, lw=1)
    ax.axvspan(-config.WINDOW_MIN, 0, color=PRE, alpha=0.06, lw=0)
    ax.axvspan(0, config.WINDOW_MIN, color=POST, alpha=0.06, lw=0)
    ax.set_xlabel("minutes from go-around climb start")
    ax.set_ylabel("T1 encounter rate, observed / expected")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper left")
    _title(ax, f"Superposed epoch of T1 encounters around {n_events} "
               "go-arounds (shaded: pre | post windows)")
    _save(fig, out)


def _log_axis(ax, values):
    from matplotlib.ticker import FixedLocator, FuncFormatter
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], float)
    lo = max(0.05, np.min(v) / 1.3) if len(v) else 0.2
    hi = min(50.0, np.max(v) * 1.3) if len(v) else 5.0
    ax.set_xscale("log")
    ax.set_xlim(min(lo, 0.8), max(hi, 1.25))
    ticks = [t for t in (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 25, 50)
             if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([]))


def fig_forest(results: pd.DataFrame, out: Path) -> None:
    r = results[(results["event_set"] == "all") & (results["type"] == "count")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    labels, y, ticks = [], 0, []
    vals0, vals1 = [], []
    for scope, name in (("airspace", "airspace-wide"),
                        ("ga_involved", "go-around-involved")):
        sub = r[r["scope"] == scope]
        for _, row in sub.iterrows():
            ticks.append(y)
            labels.append(f"{name}: {row['endpoint']}")
            if np.isfinite(row["irr"]) and row["irr"] > 0:
                axes[0].errorbar(
                    row["irr"], y,
                    xerr=[[max(row["irr"] - row["irr_lo"], 0)],
                          [max(row["irr_hi"] - row["irr"], 0)]],
                    fmt="o", color=POST, ms=6, capsize=3)
                vals0 += [row["irr"], row["irr_lo"], row["irr_hi"]]
            else:
                axes[0].annotate("no events" if row["sum_pre"] + row["sum_post"]
                                 == 0 else "not estimable", (1.0, y),
                                 ha="center", va="center", fontsize=8,
                                 color=MUTED)
            for off, w, col in ((-0.18, "pre", PRE), (0.18, "post", POST)):
                v, lo, hi = row[f"sir_{w}"], row[f"sir_{w}_lo"], row[f"sir_{w}_hi"]
                if np.isfinite(v) and v > 0:
                    axes[1].errorbar(v, y + off,
                                     xerr=[[max(v - lo, 0)], [max(hi - v, 0)]],
                                     fmt="o", color=col, ms=5, capsize=3)
                    vals1 += [v, lo, hi]
                elif w == "post":
                    axes[1].annotate("no events", (1.0, y), ha="center",
                                     va="center", fontsize=8, color=MUTED)
            y += 1
        y += 0.6
    _log_axis(axes[0], vals0); _log_axis(axes[1], vals1)
    for ax in axes:
        ax.axvline(1, color=BASE, lw=1.2, ls="--")
    axes[0].invert_yaxis()
    axes[0].set_yticks(ticks, labels, fontsize=8)
    axes[0].set_xlabel("rate ratio post / pre (log scale)")
    axes[1].set_xlabel("standardised incidence ratio vs baseline (log scale)")
    axes[1].plot([], [], "o", color=PRE, label="pre window")
    axes[1].plot([], [], "o", color=POST, label="post window")
    axes[1].legend(fontsize=8, loc="lower right")
    _title(axes[0], "Post vs pre (IRR, day-clustered bootstrap 95 % CI)")
    _title(axes[1], "Pre and post vs baseline (SIR, day-block bootstrap CI)")
    _save(fig, out)


def fig_equilibrium(risk: pd.DataFrame, eq: dict, out: Path) -> None:
    z = risk.loc[~risk["excluded"], f"pre_{PRIMARY_METRIC}_z"].dropna()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if len(z):
        ax.hist(z, bins=np.arange(-4, 4.25, 0.5), density=True, color=PRE,
                edgecolor=SURFACE, label="pre-window z-scores")
    x = np.linspace(-4, 4, 200)
    ax.plot(x, sps.norm.pdf(x), color=BASE, lw=1.5, label="N(0, 1)")
    ax.axvline(sps.norm.ppf(config.LIMIT_WARNING), color=POST, lw=1, ls="--",
               label="95 % limit")
    ax.set_xlabel("z-score of the pre window under the baseline")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ci = eq.get("in_control_ci", [np.nan, np.nan])
    _title(ax, f"Equilibrium {eq.get('verdict', '')}: "
               f"{100 * eq.get('in_control_share', np.nan):.0f} % of pre "
               f"windows in control [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}] %, "
               f"SIR_pre {eq.get('sir_pre', np.nan):.2f}")
    _save(fig, out)


def fig_control_chart(bl, wm: pd.DataFrame, risk: pd.DataFrame,
                      out: Path) -> None:
    sc = bl.score_counts(wm, seed=config.RANDOM_SEED)
    z = sc[f"{PRIMARY_METRIC}_z"].to_numpy()
    t = pd.to_datetime(wm["t_start"], unit="s", utc=True)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ok = np.isfinite(z)
    ax.plot(t[ok], z[ok], ".", color=BASE, ms=3, alpha=0.6,
            label="clock windows")
    for q, lab, col in ((config.LIMIT_WARNING, "95 % (warning)", PRE),
                        (config.LIMIT_ACTION, "99 % (action)", POST)):
        ax.axhline(sps.norm.ppf(q), color=col, lw=1, ls="--", label=lab)
    r = risk[~risk["excluded"]]
    for w, col, m in (("pre", PRE, "v"), ("post", POST, "^")):
        zz = r[f"{w}_{PRIMARY_METRIC}_z"]
        tt = pd.to_datetime(r["t0"], unit="s", utc=True)
        ax.plot(tt, zz, m, color=col, ms=6, label=f"go-around {w} window")
    ax.set_ylabel("z-score of T1 encounters vs baseline")
    ax.set_xlabel("date (UTC)")
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    _title(ax, "Risk-adjusted control chart: every 10-min window over the "
               "dataset")
    _save(fig, out)


def fig_density(enc: pd.DataFrame, risk: pd.DataFrame, out: Path) -> None:
    """Where encounters happen: baseline vs post-go-around density."""
    e = enc[(enc["tier"] == "T1") & (enc["kind"] == "any")
            & enc["inside_ceiling"]]
    w = config.WINDOW_MIN * 60
    t = e["t_min"].to_numpy()
    post = np.zeros(len(e), dtype=bool)
    near = np.zeros(len(e), dtype=bool)
    ex = config.BASELINE_EXCLUSION_MIN * 60
    for t0 in risk["t0"]:
        post |= (t >= t0) & (t < t0 + w)
        near |= (t >= t0 - ex) & (t < t0 + ex)
    x = (e["xa"] + e["xb"]) / 2; y = (e["ya"] + e["yb"]) / 2
    R = config.VOLUME_RADIUS_NM
    bins = np.linspace(-R, R, 21)
    hb, _, _ = np.histogram2d(x[~near], y[~near], bins=[bins, bins])
    hp, _, _ = np.histogram2d(x[post], y[post], bins=[bins, bins])
    hb = hb / max(hb.sum(), 1); hp = hp / max(hp.sum(), 1)
    vmax = max(hb.max(), hp.max(), 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, h, n, title in ((axes[0], hb, int((~near).sum()),
                             "baseline windows"),
                            (axes[1], hp, int(post.sum()),
                             "post-go-around windows")):
        im = ax.imshow(h.T, origin="lower", extent=[-R, R, -R, R], cmap=SEQ,
                       vmin=0, vmax=vmax)
        _draw_runways_xy(ax)
        for rr in (5, 10):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, color=MUTED,
                                    lw=0.8, ls=":"))
        ax.set_aspect("equal"); ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        ax.set_xlabel("east of airport (NM)"); ax.set_ylabel("north (NM)")
        ax.grid(False)
        _title(ax, f"T1 encounters, {title} (n={n:,})")
    fig.colorbar(im, ax=axes, label="share of encounters per cell",
                 shrink=0.8)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _draw_runways_xy(ax):
    from .geometry import runway_table
    from .loading import project
    names, tx, ty, brg = runway_table()
    drawn = set()
    for i, n in enumerate(names):
        for j, m in enumerate(names):
            if j <= i or (i, j) in drawn:
                continue
            if abs((brg[i] - brg[j] + 360) % 360 - 180) < 15 and \
                    np.hypot(tx[i] - tx[j], ty[i] - ty[j]) < 4:
                ax.plot([tx[i], tx[j]], [ty[i], ty[j]], lw=3, color="0.35",
                        solid_capstyle="butt", zorder=3)
                drawn.add((i, j))
        ax.annotate(n, (tx[i], ty[i]), fontsize=6, color="0.35", zorder=4)


def fig_sensitivity(sens: pd.DataFrame, out: Path) -> None:
    s = sens[(sens["scope"] == "airspace") & sens["irr"].notna()].copy()
    if s.empty:
        return
    s["label"] = s["parameter"].astype(str) + " = " + s["value"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.28 * len(s) + 1.8),
                             sharey=True)
    y = np.arange(len(s))
    for ax, col, lab in ((axes[0], "irr", "IRR post / pre"),
                         (axes[1], "sir_post", "SIR post vs baseline")):
        for i, (_, r) in enumerate(s.iterrows()):
            colr = INK if r["is_base"] else POST
            ax.errorbar(r[col], i, xerr=[[max(r[col] - r[f"{col}_lo"], 0)],
                                        [max(r[f"{col}_hi"] - r[col], 0)]],
                        fmt="o", color=colr, ms=5, capsize=3)
        ax.axvline(1, color=BASE, lw=1.2, ls="--")
        base = s[s["is_base"]]
        if len(base):
            ax.axvline(base[col].iat[0], color=INK, lw=0.8, ls=":")
        _log_axis(ax, pd.concat([s[col], s[f"{col}_lo"], s[f"{col}_hi"]]))
        ax.set_xlabel(f"{lab} (log scale)")
    axes[0].set_yticks(y, s["label"], fontsize=8); axes[0].invert_yaxis()
    _title(axes[0], "Sensitivity of the primary result (airspace-wide T1)")
    _title(axes[1], "black = base setting; dotted = base value")
    _save(fig, out)


def fig_ceiling(heights: pd.DataFrame, chk: dict, out: Path) -> None:
    h = heights["max_height_post_ft"].dropna()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if len(h):
        ax.hist(h, bins=20, color=PRE, edgecolor=SURFACE,
                label="go-around aircraft: max height in post window")
    ax.axvline(chk["charted_ceiling_ft_agl"], color=BASE, lw=1.5, ls="--",
               label=f"charted ceiling ({chk['charted_ceiling_ft_agl']:.0f} ft)")
    if chk["override"]:
        ax.axvline(chk["effective_ceiling_ft_agl"], color=POST, lw=1.5,
                   label=f"effective ceiling "
                         f"({chk['effective_ceiling_ft_agl']:.0f} ft)")
    if np.isfinite(chk["p95_ga_max_height_ft_agl"]):
        ax.axvline(chk["p95_ga_max_height_ft_agl"], color=INK, lw=1, ls=":",
                   label="95th percentile")
    ax.set_xlabel("height above field (ft)")
    ax.set_ylabel("go-around events")
    ax.legend(fontsize=8)
    _title(ax, "Ceiling check: does the volume contain the go-around "
               "climb-outs?")
    _save(fig, out)


def study_figures(res: dict, chk: dict, ev: pd.DataFrame, figdir: Path,
                  sens: pd.DataFrame | None = None) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    risk = res["risk"]
    fig_epoch(res["epoch"], figdir / "superposed_epoch.png", len(ev))
    fig_forest(res["results"], figdir / "forest.png")
    eq = res["equilibrium"]
    eq_air = eq[eq["scope"] == "airspace"].iloc[0].to_dict() if len(eq) else {}
    fig_equilibrium(risk, eq_air, figdir / "equilibrium.png")
    fig_control_chart(res["baseline"], res["window_metrics"], risk,
                      figdir / "control_chart.png")
    fig_density(res["encounters"], risk, figdir / "encounter_density.png")
    if sens is not None and len(sens):
        fig_sensitivity(sens, figdir / "sensitivity.png")
    heights = risk[["event_id", "max_height_post_ft"]] \
        if "max_height_post_ft" in risk.columns else res.get("heights")
    if heights is not None:
        fig_ceiling(heights, chk, figdir / "ceiling_check.png")


# ------------------------------------------------------ per event -------

def event_plots(res: dict, ev: pd.DataFrame, days: dict, outdir: Path) -> None:
    """One PNG per go-around: maps pre | post, timeline, and the pre and
    post values against the baseline predictive distribution."""
    from .loading import build_day_grid
    outdir.mkdir(parents=True, exist_ok=True)
    risk = res["risk"].set_index("event_id")
    enc = res["encounters"]
    bl = res["baseline"]
    spec = res["spec"]
    w = int(config.WINDOW_MIN * 60)
    if ev.empty:
        return
    # group events by the day file holding t0 so each day is loaded once
    ev = ev.copy()
    ev["day"] = risk.reindex(ev["event_id"])["day"].to_numpy()
    files = {f.stem: f for f in res.get("files", [])}
    for day, group in ev.groupby("day"):
        raw = files.get(day)
        if raw is None:
            continue
        dg = build_day_grid(raw)
        g = dg.grid
        legs = dg.legs.set_index("leg")
        for _, e in group.iterrows():
            try:
                _plot_event(e, risk.loc[e["event_id"]], g, legs, enc, bl,
                            spec, w, outdir)
            except Exception as err:   # report, never absorb
                print(f"  event plot failed {e['event_id']}: {err}", flush=True)


def _plot_event(e, r, g, legs, enc, bl, spec, w, outdir):
    t0 = int(e["t0"])
    R = spec.radius_nm
    fig = plt.figure(figsize=(11, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.9, 0.9])
    ga_leg = legs[legs["icao24"] == e["icao24"]].index
    # ---- row 1: maps pre | post
    for col, (a, b, name, colr) in enumerate(((t0 - w, t0, "pre", PRE),
                                              (t0, t0 + w, "post", POST))):
        ax = fig.add_subplot(gs[0, col])
        win = g[(g["t"] >= a) & (g["t"] < b) & g["airborne"]]
        for leg, tr in win.groupby("leg"):
            if leg in ga_leg:
                ax.plot(tr["x"], tr["y"], color=colr, lw=2, zorder=5)
            else:
                ax.plot(tr["x"], tr["y"], color="0.7", lw=0.7, zorder=2)
        ee = enc[(enc["t_min"] >= a) & (enc["t_min"] < b)
                 & (enc["kind"] == "any") & enc["inside_ceiling"]]
        for tier in ("T1", "T2", "T3"):
            sub = ee[ee["tier"] == tier]
            for _, x in sub.iterrows():
                ax.plot([x["xa"], x["xb"]], [x["ya"], x["yb"]], color=TIER[tier],
                        lw=2.2, zorder=6)
                ax.plot([x["xa"], x["xb"]], [x["ya"], x["yb"]], "o",
                        color=TIER[tier], ms=4, zorder=7)
        _draw_runways_xy(ax)
        for rr in (5, 10):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, color=MUTED,
                                    lw=0.8, ls=":"))
        ax.set_aspect("equal"); ax.set_xlim(-R - 1, R + 1); ax.set_ylim(-R - 1, R + 1)
        ax.set_xlabel("east of airport (NM)"); ax.set_ylabel("north (NM)")
        ax.grid(False)
        n_ac = win["leg"].nunique()
        _title(ax, f"{name} window ({config.WINDOW_MIN:g} min): {n_ac} aircraft, "
                   f"{len(ee)} encounters")
        for tier in ("T1", "T2", "T3"):
            ax.plot([], [], color=TIER[tier], lw=2, label=f"{tier} encounter")
        ax.plot([], [], color=colr, lw=2, label="go-around aircraft")
        ax.legend(fontsize=7, loc="upper right")
    # ---- row 2: timeline
    ax = fig.add_subplot(gs[1, :])
    ts = np.arange(t0 - w, t0 + w)
    win = g[(g["t"] >= t0 - w) & (g["t"] < t0 + w) & g["airborne"]
            & (g["r"] <= R) & (g["h"] <= spec.ceiling_ft)]
    n = win.groupby("t").size().reindex(ts, fill_value=0)
    ax.step((ts - t0) / 60.0, n.to_numpy(), color=BASE, lw=1.2, where="post",
            label="aircraft in volume")
    ax.set_ylabel("aircraft in volume")
    ax2 = ax.twinx()
    ee = enc[(enc["t_end"] >= t0 - w) & (enc["t_start"] < t0 + w)
             & (enc["kind"] == "any") & enc["inside_ceiling"]]
    for i, (_, x) in enumerate(ee.sort_values("t_start").iterrows()):
        ax2.plot([(x["t_start"] - t0) / 60, (x["t_end"] - t0) / 60],
                 [x["s_min"], x["s_min"]], color=TIER[x["tier"]], lw=3,
                 solid_capstyle="butt")
    for tier, top in (("T1", 1.0), ("T2", 0.5 / 3), ("T3", 500 / 6076.12 / 3)):
        ax2.axhspan(0, top, color=TIER[tier], alpha=0.08, lw=0)
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel("min normalised separation s (bars: encounters)")
    ax2.grid(False)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("minutes from climb start")
    ax.set_xlim(-config.WINDOW_MIN, config.WINDOW_MIN)
    ax.legend(fontsize=7, loc="upper left")
    _title(ax, "Timeline: traffic and encounter episodes (shaded bands: tier "
               "zones, s < 1 = inside T1)")
    # ---- row 3: against the baseline
    ax = fig.add_subplot(gs[2, 0])
    model = bl.count_models.get(PRIMARY_METRIC)
    rows = []
    for name, colr in (("pre", PRE), ("post", POST)):
        mu = r.get(f"{name}_{PRIMARY_METRIC}_expected", np.nan)
        y = r.get(f"{name}_{PRIMARY_METRIC}", np.nan)
        if model is not None and np.isfinite(mu) and mu > 0:
            k = np.arange(0, int(max(mu * 3, y if np.isfinite(y) else 0, 5)) + 3)
            pmf = count_pmf(k, mu, model.alpha)
            ax.plot(k, pmf, drawstyle="steps-mid", color=colr, lw=1.5,
                    label=f"{name}: expected {mu:.1f}")
            if np.isfinite(y):
                ax.plot([y], [count_pmf(np.array([y]), mu, model.alpha)[0]],
                        "o", color=colr, ms=9, zorder=5)
            ax.axvline(r.get(f"{name}_{PRIMARY_METRIC}_limit95", np.nan)
                       if f"{name}_{PRIMARY_METRIC}_limit95" in r else np.nan,
                       color=colr, lw=0.8, ls="--")
        rows.append([name, f"{y:.0f}" if np.isfinite(y) else "-",
                     f"{mu:.2f}" if np.isfinite(mu) else "-",
                     f"{r.get(f'{name}_{PRIMARY_METRIC}_oe', np.nan):.2f}",
                     f"{r.get(f'{name}_{PRIMARY_METRIC}_z', np.nan):.2f}",
                     "yes" if r.get(f"{name}_{PRIMARY_METRIC}_in_control", False)
                     else "no"])
    ax.set_xlabel("T1 encounters in the window")
    ax.set_ylabel("baseline probability")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    _title(ax, "Observed (dots) vs baseline predictive distribution")
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["window", "observed", "expected",
                                            "O/E", "z", "in control"],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.6)
    _title(ax, "Primary metric against the baseline")
    fig.suptitle(f"GO-AROUND {e['callsign']} ({e['icao24']})  rwy {e['runway']}"
                 f"  {e['t0_utc']:%Y-%m-%d %H:%M:%S}Z   pre z "
                 f"{r.get(f'pre_{PRIMARY_METRIC}_z', np.nan):.2f}, post z "
                 f"{r.get(f'post_{PRIMARY_METRIC}_z', np.nan):.2f}",
                 fontsize=11)
    name = f"{e['t0_utc']:%Y%m%d_%H%M%S}Z_{e['callsign']}_{e['runway']}.png"
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=110)
    plt.close(fig)
