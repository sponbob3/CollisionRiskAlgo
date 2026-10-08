"""
All figures (FRAMEWORK.md section 16), in the go-around pipeline's style:
light background, minimal chart junk, one question per figure, plain
labels (proximity_pipeline/labels.py), legends outside the data, and
fixed colours for before, after, baseline/normal and the three tiers
across every figure. Every figure uses constrained layout so that no
text overlaps; captions explaining how to read each figure live in the
PDF reports, directly under the figure.
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
from . import labels as L
from .baseline import PRIMARY_METRIC, count_pmf

# fixed colours (validated categorical slots): before, after, baseline,
# tiers
PRE, POST, BASE = "#2a78d6", "#eb6834", "#7a7975"
TIER = {"T1": "#1baf7a", "T2": "#eda100", "T3": "#e34948"}
INK, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
BAND = "#e9e8e4"
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


def _fig(*args, **kw):
    kw.setdefault("layout", "constrained")
    return plt.subplots(*args, **kw)


def _save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=11)


def _log_axis(ax, values):
    from matplotlib.ticker import FixedLocator, FuncFormatter
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], float)
    lo = max(0.05, np.min(v) / 1.3) if len(v) else 0.2
    hi = min(50.0, np.max(v) * 1.3) if len(v) else 5.0
    ax.set_xscale("log")
    ax.set_xlim(min(lo, 0.8), max(hi, 1.25))
    ticks = [t for t in (0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 10, 25, 50)
             if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([]))


def _ci_dot(ax, x, lo, hi, y, color, ms=6):
    if not np.isfinite(x):
        return False
    xerr = None
    if np.isfinite(lo) and np.isfinite(hi):
        xerr = [[max(x - lo, 0)], [max(hi - x, 0)]]
    ax.errorbar(x, y, xerr=xerr, fmt="o", color=color, ms=ms, capsize=3,
                lw=1.4)
    return True


# ------------------------------------------------ baseline validation ----

def baseline_figures(bl, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    v = bl.validation.get(PRIMARY_METRIC, {})

    # 1. held-out windows beyond the normal-range limits vs the target
    rows = [(m, bl.validation[m]) for m in bl.validation
            if "exceed95_rate" in bl.validation[m]
            and bl.validation[m].get("n", 0)]
    if rows:
        fig, ax = _fig(figsize=(9, 0.42 * len(rows) + 1.9))
        y = np.arange(len(rows))
        ax.axvspan(100 * config.VALIDATION_PASS_95[0],
                   100 * config.VALIDATION_PASS_95[1], color=PRE, alpha=0.10,
                   lw=0)
        ax.axvspan(100 * config.VALIDATION_PASS_99[0],
                   100 * config.VALIDATION_PASS_99[1], color=POST, alpha=0.10,
                   lw=0)
        for i, (m, r) in enumerate(rows):
            for off, key, col in ((-0.17, "95", PRE), (0.17, "99", POST)):
                rate = r[f"exceed{key}_rate"]; lo, hi = r[f"exceed{key}_ci"]
                _ci_dot(ax, 100 * rate, 100 * lo, 100 * hi, i + off, col)
        ax.set_yticks(y, [L.metric(m) for m, _ in rows])
        ax.invert_yaxis()
        ax.set_xlabel("held-out windows outside the limit (%)")
        ax.plot([], [], "o", color=PRE,
                label="outside the 95 % limit (target about 5 %)")
        ax.plot([], [], "o", color=POST,
                label="outside the 99 % limit (target about 1 %)")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                  ncol=2)
        _title(ax, "Does the baseline predict months it has not seen?")
        _save(fig, outdir / "calibration_exceedance.png")

    # 2. held-out windows: PIT histogram and z-score quantiles
    if v.get("z"):
        z = np.asarray(v["z"]); pit = np.asarray(v["pit"])
        fig, (ax1, ax2) = _fig(1, 2, figsize=(10, 3.8))
        ax1.hist(pit, bins=10, range=(0, 1), color=PRE, edgecolor=SURFACE,
                 density=True)
        ax1.axhline(1, color=BASE, lw=1.2, ls="--")
        ax1.set_xlabel("position of each held-out window in its predicted "
                       "range (0 to 1)")
        ax1.set_ylabel("relative frequency")
        _title(ax1, "Flat at 1 if the baseline is right")
        (osm, osr), _ = sps.probplot(z, dist="norm")
        ax2.plot(osm, osr, ".", color=PRE, ms=4)
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        ax2.plot(lim, lim, color=BASE, lw=1.2, ls="--")
        ax2.set_xlabel("z expected if the baseline is right")
        ax2.set_ylabel("z of held-out windows")
        _title(ax2, "On the dashed line if the baseline is right")
        _save(fig, outdir / "calibration_pit_qq.png")

    # 3. stability: estimate vs days of data used
    st = v.get("stability", {})
    tab = st.get("table")
    if tab is not None and len(tab):
        fig, (ax1, ax2) = _fig(1, 2, figsize=(10, 3.8))
        for ax, col, lab in ((ax1, "rate", "T1 encounters per pair-hour"),
                             (ax2, "limit95", "upper normal limit per window")):
            full = st.get("full_rate" if col == "rate" else "full_limit95")
            if full is not None:
                ax.axhspan(full * (1 - config.STABILITY_PLATEAU_TOL),
                           full * (1 + config.STABILITY_PLATEAU_TOL),
                           color=BAND, lw=0)
            ax.fill_between(tab["n_days"], tab[f"{col}_lo"], tab[f"{col}_hi"],
                            color=PRE, alpha=0.15, lw=0)
            ax.plot(tab["n_days"], tab[f"{col}_mean"], "o-", color=PRE, ms=5)
            ax.set_xlabel("days of data used")
            ax.set_ylabel(lab)
            ax.set_ylim(bottom=0)
        stable = st.get("stable_from_days")
        _title(ax1, "Baseline rate")
        _title(ax2, "Normal-range limit "
                    f"(steady from {stable if stable is not None else '-'} days)")
        _save(fig, outdir / "stability_curve.png")


def probability_figure(calib: dict, em_table: pd.DataFrame,
                       outdir: Path) -> None:
    """Optional add-on (--probability): reliability of the conflict
    probability model and its prediction-error growth."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = _fig(1, 2, figsize=(10, 3.9))
    c = calib.get("T1", {})
    tab = c.get("table")
    if tab is not None and len(tab):
        t = tab[tab["n"] > 0]
        ax1.plot([0, 1], [0, 1], color=BASE, lw=1.2, ls="--")
        ax1.plot(t["mean_pred"], t["obs_frac"], "o-", color=TIER["T1"], ms=6)
        ax1.set_xlabel("predicted chance of entering T1")
        ax1.set_ylabel("share that actually entered T1")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        _title(ax1, f"Reliability (on the dashed line if right; "
                    f"error {c.get('ece', np.nan):.3f})")
        t.to_csv(outdir / "reliability_T1.csv", index=False)
    if len(em_table):
        allrows = em_table[em_table["phase"] == "all"]
        for col, mk, colr, lab in (("sigma_along_nm", "o-", PRE, "along track"),
                                   ("sigma_cross_nm", "s-", POST, "across track")):
            ax2.plot(allrows["tau_s"], allrows[col] * ga.FT_PER_NM, mk,
                     color=colr, ms=4, label=lab)
        ax2.plot(allrows["tau_s"], allrows["sigma_z_ft"], "^-", color=BASE,
                 ms=4, label="vertical")
        ax2.set_xlabel("seconds ahead")
        ax2.set_ylabel("typical prediction error (ft)")
        ax2.set_ylim(bottom=0)
        ax2.legend(fontsize=8, loc="upper left")
        _title(ax2, "Straight-line prediction error")
        em_table.to_csv(outdir / "error_model.csv", index=False)
    _save(fig, outdir / "probability_model.png")
    rows = [{"tier": t, **{k: v for k, v in c.items() if k != "table"}}
            for t, c in calib.items()]
    pd.DataFrame(rows).to_csv(outdir / "calibration_summary.csv", index=False)


# --------------------------------------------------- study figures -------

def fig_change(change: pd.DataFrame, out: Path) -> None:
    """What changes after a go-around, next to the same change after a
    normal landing at the same hours of day."""
    keys = [k for k in ("aircraft_time", "pair_time", "T1_any")
            if k in set(change["measure"])]
    if not keys:
        return
    ch = change.set_index("measure").loc[keys]
    fig, ax = _fig(figsize=(9.5, 3.6))
    y = np.arange(len(keys))
    vals = []
    for off, pref, col, lab in ((-0.17, "", POST, "after a go-around"),
                                (0.17, "normal_", BASE,
                                 "after a normal landing (same hours)")):
        for i, (_, r) in enumerate(ch.iterrows()):
            x = 100 * (r[f"{pref}ratio"] - 1)
            lo = 100 * (r[f"{pref}ratio_lo"] - 1)
            hi = 100 * (r[f"{pref}ratio_hi"] - 1)
            if _ci_dot(ax, x, lo, hi, i + off, col, ms=7):
                vals += [x, lo, hi]
                ax.annotate(f"{x:+.0f}%", (hi if np.isfinite(hi) else x,
                                           i + off),
                            xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK)
        ax.plot([], [], "o", color=col, label=lab)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [ch.loc[k, "label"] for k in keys])
    ax.invert_yaxis()
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if len(v):
        pad = max(5.0, 0.25 * (v.max() - v.min()))
        ax.set_xlim(min(v.min(), 0) - pad, max(v.max(), 0) + 2 * pad)
    ax.set_xlabel("change from the 10 minutes before to the 10 minutes "
                  "after (%)")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2)
    _title(ax, "What changes after a go-around, compared with a normal "
               "landing")
    _save(fig, out)


def fig_epoch(epoch: pd.DataFrame, out: Path, count_text: str) -> None:
    fig, ax = _fig(figsize=(9.5, 3.9))
    ax.axvspan(-config.WINDOW_MIN, 0, color=PRE, alpha=0.06, lw=0)
    ax.axvspan(0, config.WINDOW_MIN, color=POST, alpha=0.06, lw=0)
    if len(epoch):
        ok = epoch["expected"] > 0
        ax.fill_between(epoch["minute"][ok], epoch["oe_lo"][ok],
                        epoch["oe_hi"][ok], color=POST, alpha=0.15, lw=0,
                        label="95 % confidence band")
        ax.plot(epoch["minute"][ok], epoch["oe"][ok], color=POST, lw=2,
                label="observed ÷ expected, all events together")
    ax.axhline(1, color=BASE, lw=1.2, ls="--", label="normal traffic (= 1)")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("minutes from the go-around (climb start)")
    ax.set_ylabel("T1 encounters, observed ÷ expected")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=3)
    _title(ax, f"T1 encounters minute by minute around {count_text}")
    _save(fig, out)


def fig_forest(results: pd.DataFrame, out: Path) -> None:
    typ_sets = [f"{t}_only" for t in config.EVENT_TYPE_BREAKDOWN]
    r = results[(results["type"] == "count") & (results["scope"] == "airspace")
                & ((results["event_set"] == "all")
                   | (results["event_set"].isin(typ_sets)
                      & (results["endpoint"] == PRIMARY_METRIC)))]
    if r.empty:
        return
    fig, axes = _fig(1, 2, figsize=(11, 0.45 * len(r) + 2.0), sharey=True)
    labels, vals0, vals1 = [], [], []
    for y, (_, row) in enumerate(r.iterrows()):
        lab = L.metric(row["endpoint"])
        if row["event_set"] != "all":
            lab += f" ({L.event_set(row['event_set'])})"
        labels.append(lab)
        if _ci_dot(axes[0], row["irr"], row["irr_lo"], row["irr_hi"], y, POST) \
                and row["irr"] > 0:
            vals0 += [row["irr"], row["irr_lo"], row["irr_hi"]]
        else:
            axes[0].annotate("too few to compare", (1.0, y), ha="center",
                             va="center", fontsize=8, color=MUTED)
        for off, w, col in ((-0.17, "pre", PRE), (0.17, "post", POST)):
            v = row[f"sir_{w}"]
            if np.isfinite(v) and v > 0:
                _ci_dot(axes[1], v, row[f"sir_{w}_lo"], row[f"sir_{w}_hi"],
                        y + off, col, ms=5)
                vals1 += [v, row[f"sir_{w}_lo"], row[f"sir_{w}_hi"]]
            elif w == "post":
                axes[1].annotate("too few to compare", (1.0, y), ha="center",
                                 va="center", fontsize=8, color=MUTED)
    _log_axis(axes[0], vals0); _log_axis(axes[1], vals1)
    for ax in axes:
        ax.axvline(1, color=BASE, lw=1.2, ls="--")
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("after ÷ before, per unit of traffic (log scale)")
    axes[1].set_xlabel("observed ÷ expected from normal traffic (log scale)")
    axes[1].plot([], [], "o", color=PRE, label="before the go-around")
    axes[1].plot([], [], "o", color=POST, label="after the go-around")
    axes[1].legend(fontsize=8.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.12), ncol=2)
    _title(axes[0], "After vs before")
    _title(axes[1], "Compared with normal traffic")
    _save(fig, out)


def fig_equilibrium(risk: pd.DataFrame, eq: dict, out: Path) -> None:
    z = risk.loc[~risk["excluded"], f"pre_{PRIMARY_METRIC}_z"].dropna()
    fig, ax = _fig(figsize=(9.5, 3.0))
    lim = sps.norm.ppf(0.5 + config.LIMIT_WARNING / 2)
    ax.axvspan(-lim, lim, color=BAND, lw=0, label="normal range (95 %)")
    if len(z):
        rng = np.random.default_rng(0)
        ax.plot(z, rng.uniform(-0.3, 0.3, len(z)), "o", color=PRE, ms=5,
                alpha=0.75, label="one go-around's before-window")
        inside = int((z.abs() <= lim).sum())
        ax.text(0.01, 0.96, f"{inside} of {len(z)} before-windows "
                f"({100 * inside / len(z):.0f} %) are in the normal range "
                f"(about {100 * config.LIMIT_WARNING:.0f} % expected)",
                transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_ylim(-0.6, 1.1)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(-4, (z.min() - 0.5) if len(z) else -4),
                max(4, (z.max() + 0.5) if len(z) else 4))
    ax.set_xlabel("how unusual the T1 encounter count was (z): 0 = typical, "
                  "beyond ±2 = unusual")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=2)
    _title(ax, "Was the airspace normal before each go-around?")
    _save(fig, out)


def fig_daily(bl, wm: pd.DataFrame, risk: pd.DataFrame, out: Path) -> None:
    """Day by day: T1 encounters observed vs what the baseline expects."""
    model = bl.count_models.get(PRIMARY_METRIC)
    if model is None or wm.empty:
        return
    w = wm[(wm["quality"] != "bad") & ~wm["partial_coverage"].astype(bool)]
    mu = model.mu(w)
    d = pd.DataFrame({"day": pd.to_datetime(w["t_start"], unit="s", utc=True)
                      .dt.floor("D"), "obs": w[PRIMARY_METRIC].to_numpy(float),
                      "exp": mu})
    daily = d.groupby("day").sum()
    daily = daily[daily["exp"] > 0]
    daily["oe"] = daily["obs"] / daily["exp"]
    fig, ax = _fig(figsize=(10, 3.6))
    if len(daily) >= 5:
        lo, hi = np.percentile(daily["oe"], [2.5, 97.5])
        ax.axhspan(lo, hi, color=BAND, lw=0, label="range of 95 % of days")
    ax.axhline(1, color=BASE, lw=1.2, ls="--",
               label="exactly as expected (= 1)")
    ax.plot(daily.index, daily["oe"], "o-", color=PRE, ms=4, lw=1,
            label="one day")
    ev_days = pd.to_datetime(risk["t0"], unit="s", utc=True).dt.floor("D")
    counts = ev_days.value_counts()
    top = max(1.6, float(daily["oe"].max()) * 1.08) if len(daily) else 1.6
    if len(counts):
        ax.plot(counts.index, np.full(len(counts), top), "|", color=POST,
                ms=12, mew=2, label="day with go-arounds")
    ax.set_ylim(0, top * 1.06)
    ax.set_ylabel("T1 encounters, observed ÷ expected")
    ax.set_xlabel("date (UTC)")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4)
    _title(ax, "How well the baseline describes each day")
    _save(fig, out)


def fig_density(enc: pd.DataFrame, risk: pd.DataFrame, out: Path) -> None:
    """Where encounters happen: normal windows vs after events."""
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
    fig, axes = _fig(1, 2, figsize=(11, 5.4))
    for ax, h, n, title in ((axes[0], hb, int((~near).sum()),
                             "Normal traffic"),
                            (axes[1], hp, int(post.sum()),
                             "After go-arounds")):
        im = ax.imshow(h.T, origin="lower", extent=[-R, R, -R, R], cmap=SEQ,
                       vmin=0, vmax=vmax)
        _draw_runways_xy(ax)
        for rr in (5, 10):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, color=MUTED,
                                    lw=0.8, ls=":"))
        ax.set_aspect("equal"); ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        ax.set_xlabel("east of airport (NM)"); ax.set_ylabel("north (NM)")
        ax.grid(False)
        _title(ax, f"{title} ({n:,} T1 encounters)")
    fig.colorbar(im, ax=axes, label="share of encounters in each cell",
                 shrink=0.8)
    _save(fig, out)


def fig_sensitivity(sens: pd.DataFrame, out: Path) -> None:
    s = sens[(sens["scope"] == "airspace") & sens["irr"].notna()].copy()
    if s.empty:
        return
    s["label"] = [L.sensitivity_row(p, v) for p, v in
                  zip(s["parameter"], s["value"])]
    panels = [(c, lab) for c, lab in
              (("total_ratio", "Total T1 encounters\nafter ÷ before"),
               ("irr", "T1 per unit of traffic\nafter ÷ before"),
               ("sir_post", "After the go-around\nvs normal traffic"))
              if c in s.columns and s[c].notna().any()]
    fig, axes = _fig(1, len(panels), figsize=(4.1 * len(panels) + 2.8,
                                               0.3 * len(s) + 1.8),
                     sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(s))
    for ax, (col, lab) in zip(axes, panels):
        vals = []
        for i, (_, r) in enumerate(s.iterrows()):
            colr = INK if r["is_base"] else POST
            if _ci_dot(ax, r[col], r.get(f"{col}_lo", np.nan),
                       r.get(f"{col}_hi", np.nan), i, colr, ms=5):
                vals += [r[col], r.get(f"{col}_lo", np.nan),
                         r.get(f"{col}_hi", np.nan)]
        ax.axvline(1, color=BASE, lw=1.2, ls="--")
        _log_axis(ax, vals)
        ax.set_xlabel("ratio (log scale)")
        _title(ax, lab)
    axes[0].set_yticks(y, s["label"], fontsize=8.5)
    axes[0].invert_yaxis()
    _save(fig, out)


def fig_ceiling(heights: pd.DataFrame, chk: dict, out: Path) -> None:
    h = heights["max_height_post_ft"].dropna()
    fig, ax = _fig(figsize=(9.5, 3.6))
    if len(h):
        ax.hist(h, bins=20, color=PRE, edgecolor=SURFACE,
                label="highest point of each go-around aircraft "
                      "(10 min after)")
    ax.axvline(chk["charted_ceiling_ft_agl"], color=BASE, lw=1.5, ls="--",
               label=f"starting ceiling ({chk['charted_ceiling_ft_agl']:,.0f} ft)")
    if np.isfinite(chk["p95_ga_max_height_ft_agl"]):
        ax.axvline(chk["p95_ga_max_height_ft_agl"], color=INK, lw=1, ls=":",
                   label="95 % of aircraft stay below")
    if chk["override"]:
        ax.axvline(chk["effective_ceiling_ft_agl"], color=POST, lw=1.8,
                   label=f"ceiling used ({chk['effective_ceiling_ft_agl']:,.0f}"
                         " ft)")
    ax.set_xlabel("height above the airfield (ft)")
    ax.set_ylabel("number of go-arounds")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2)
    _title(ax, "Does the analysis area reach high enough?")
    _save(fig, out)


def study_figures(res: dict, chk: dict, ev: pd.DataFrame, figdir: Path,
                  sens: pd.DataFrame | None = None) -> None:
    from .events import event_count_text
    figdir.mkdir(parents=True, exist_ok=True)
    risk = res["risk"]
    fig_change(res["change"], figdir / "change_vs_normal.png")
    fig_epoch(res["epoch"], figdir / "superposed_epoch.png",
              event_count_text(ev))
    fig_forest(res["results"], figdir / "forest.png")
    eq = res["equilibrium"]
    eq_air = eq.iloc[0].to_dict() if len(eq) else {}
    fig_equilibrium(risk, eq_air, figdir / "equilibrium.png")
    fig_daily(res["baseline"], res["window_metrics"], risk,
              figdir / "daily_levels.png")
    fig_density(res["encounters"], risk, figdir / "encounter_density.png")
    if sens is not None and len(sens):
        fig_sensitivity(sens, figdir / "sensitivity.png")
    heights = res.get("heights")
    if heights is not None:
        fig_ceiling(heights, chk, figdir / "ceiling_check.png")


# ---------------------------------------------------------- runways -----

def _runway_segments():
    """Each physical runway once, threshold to threshold: every end is
    paired with the reciprocal end that lies on its own extended
    centreline (smallest cross-track offset), so close parallel runways
    (e.g. four parallels) are not cross-connected."""
    from .geometry import runway_table
    names, tx, ty, brg = runway_table()
    used, segs = set(), []
    for i in range(len(names)):
        if i in used:
            continue
        b = np.radians(brg[i])
        ux, uy = np.sin(b), np.cos(b)
        best, best_x = None, 0.15          # NM: max offset of a true pair
        for j in range(len(names)):
            if j == i or j in used:
                continue
            if abs((brg[i] - brg[j] + 360) % 360 - 180) >= 15:
                continue
            dx, dy = tx[j] - tx[i], ty[j] - ty[i]
            along = dx * ux + dy * uy
            cross = abs(dx * uy - dy * ux)
            if 0 < along < 4 and cross < best_x:
                best, best_x = j, cross
        used.add(i)
        if best is not None:
            used.add(best)
            segs.append((i, best))
    return names, tx, ty, segs


def _draw_runways_xy(ax):
    names, tx, ty, segs = _runway_segments()
    for i, j in segs:
        ax.plot([tx[i], tx[j]], [ty[i], ty[j]], lw=3, color="0.35",
                solid_capstyle="butt", zorder=3)
    for i, n in enumerate(names):
        ax.annotate(n, (tx[i], ty[i]), fontsize=6, color="0.35", zorder=4)


# ------------------------------------------------------ per event -------

def event_plots(res: dict, ev: pd.DataFrame, days: dict, outdir: Path) -> None:
    """One PNG per event (go-around or touch-and-go): maps before | after,
    traffic and encounter timelines, and the before and after counts
    against the baseline."""
    from .loading import build_day_grid
    outdir.mkdir(parents=True, exist_ok=True)
    risk = res["risk"].set_index("event_id")
    enc = res["encounters"]
    bl = res["baseline"]
    spec = res["spec"]
    w = int(config.WINDOW_MIN * 60)
    if ev.empty:
        return
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
    from matplotlib.lines import Line2D
    from .events import event_label
    label = event_label(e["outcome"])
    t0 = int(e["t0"])
    R = spec.radius_nm
    fig = plt.figure(figsize=(11, 14), layout="constrained")
    gs = fig.add_gridspec(5, 2, height_ratios=[1.3, 0.1, 0.38, 0.62, 0.95])
    ga_leg = legs[legs["icao24"] == e["icao24"]].index

    # ---- row 1: maps before | after
    for col, (a, b, name, colr) in enumerate(((t0 - w, t0, "Before", PRE),
                                              (t0, t0 + w, "After", POST))):
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
            for _, x in ee[ee["tier"] == tier].iterrows():
                ax.plot([x["xa"], x["xb"]], [x["ya"], x["yb"]],
                        color=TIER[tier], lw=2.2, zorder=6)
                ax.plot([x["xa"], x["xb"]], [x["ya"], x["yb"]], "o",
                        color=TIER[tier], ms=4, zorder=7)
        _draw_runways_xy(ax)
        for rr in (5, 10):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, color=MUTED,
                                    lw=0.8, ls=":"))
        ax.set_aspect("equal")
        ax.set_xlim(-R - 1, R + 1); ax.set_ylim(-R - 1, R + 1)
        ax.set_xlabel("east of airport (NM)"); ax.set_ylabel("north (NM)")
        ax.grid(False)
        in_vol = win[(win["r"] <= R) & (win["h"] <= spec.ceiling_ft)]
        n_ac = legs.loc[legs.index.intersection(in_vol["leg"].unique()),
                        "icao24"].nunique()
        counts = ee["tier"].value_counts()
        enc_txt = ", ".join(f"{int(counts.get(t, 0))} {t}"
                            for t in ("T1", "T2", "T3"))
        _title(ax, f"{name} ({config.WINDOW_MIN:g} min): {n_ac} aircraft, "
                   f"{enc_txt}")
    handles = [Line2D([], [], color=PRE, lw=2,
                      label=f"{label} aircraft, before"),
               Line2D([], [], color=POST, lw=2,
                      label=f"{label} aircraft, after"),
               Line2D([], [], color="0.7", lw=1, label="other aircraft")]
    handles += [Line2D([], [], color=TIER[t], lw=2, marker="o", ms=4,
                       label=f"{t} encounter (closest point)")
                for t in ("T1", "T2", "T3")]
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center", ncol=3, fontsize=8.5)

    # ---- row 2: traffic in the area
    ts = np.arange(t0 - w, t0 + w)
    ax_n = fig.add_subplot(gs[2, :])
    win = g[(g["t"] >= t0 - w) & (g["t"] < t0 + w) & g["airborne"]
            & (g["r"] <= R) & (g["h"] <= spec.ceiling_ft)]
    n = win.groupby("t").size().reindex(ts, fill_value=0)
    ax_n.step((ts - t0) / 60.0, n.to_numpy(), color=BASE, lw=1.2, where="post")
    ax_n.axvline(0, color=INK, lw=1)
    ax_n.set_xlim(-config.WINDOW_MIN, config.WINDOW_MIN)
    ax_n.set_ylim(bottom=0)
    ax_n.set_ylabel("aircraft\nin area")
    ax_n.tick_params(labelbottom=False)
    _title(ax_n, "Traffic inside the area, second by second")

    # ---- row 3: encounters as bars
    ax_e = fig.add_subplot(gs[3, :], sharex=ax_n)
    ee = enc[(enc["t_end"] >= t0 - w) & (enc["t_start"] < t0 + w)
             & (enc["kind"] == "any") & enc["inside_ceiling"]]
    for _, x in ee.sort_values("t_start").iterrows():
        ax_e.plot([(x["t_start"] - t0) / 60, (x["t_end"] - t0) / 60 + 1 / 60],
                  [x["s_min"], x["s_min"]], color=TIER[x["tier"]], lw=4,
                  solid_capstyle="butt")
    ax_e.axvline(0, color=INK, lw=1)
    ax_e.set_ylim(0, 1.05)
    ax_e.set_ylabel("closest approach\n(1 = edge of T1)")
    ax_e.set_xlabel(f"minutes from the {label} (climb start)")
    ax_e.set_xlim(-config.WINDOW_MIN, config.WINDOW_MIN)
    key = [Line2D([], [], color=TIER[t], lw=4,
                  label=f"{t} encounter" + (" (bar length = how long it "
                                            "lasted)" if t == "T1" else ""))
           for t in ("T1", "T2", "T3")]
    ax_e.legend(handles=key, loc="upper left", bbox_to_anchor=(0, -0.32),
                ncol=3, fontsize=8)
    _title(ax_e, "Encounters over time (lower bar = closer)")

    # ---- row 4: against the baseline
    ax = fig.add_subplot(gs[4, 0])
    model = bl.count_models.get(PRIMARY_METRIC)
    rows = []
    for name, wname, colr in (("pre", "before", PRE), ("post", "after", POST)):
        mu = r.get(f"{name}_{PRIMARY_METRIC}_expected", np.nan)
        y = r.get(f"{name}_{PRIMARY_METRIC}", np.nan)
        if model is not None and np.isfinite(mu) and mu > 0:
            k = np.arange(0, int(max(mu * 3, y if np.isfinite(y) else 0, 5)) + 3)
            pmf = count_pmf(k, mu, model.alpha)
            ax.plot(k, pmf, drawstyle="steps-mid", color=colr, lw=1.5,
                    label=f"normal range, {wname} (expected {mu:.1f})")
            if np.isfinite(y):
                ax.plot([y], [count_pmf(np.array([y]), mu, model.alpha)[0]],
                        "o", color=colr, ms=9, zorder=5)
        oe = r.get(f"{name}_{PRIMARY_METRIC}_oe", np.nan)
        z = r.get(f"{name}_{PRIMARY_METRIC}_z", np.nan)
        rows.append([wname, f"{y:.0f}" if np.isfinite(y) else "-",
                     f"{mu:.1f}" if np.isfinite(mu) else "-",
                     f"{oe:.2f}" if np.isfinite(oe) else "-",
                     f"{z:.2f}" if np.isfinite(z) else "-",
                     "yes" if r.get(f"{name}_{PRIMARY_METRIC}_in_control",
                                    False) else "no"])
    ax.set_xlabel("T1 encounters in the 10-minute window")
    ax.set_ylabel("chance under normal traffic")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "no aircraft pairs in the area in either window:\n"
                "nothing to compare",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=MUTED)
    _title(ax, "T1 encounters vs normal traffic (dot = what happened)")
    ax = fig.add_subplot(gs[4, 1])
    ax.axis("off")
    tb = ax.table(cellText=rows,
                  colLabels=["window", "T1\nencounters", "expected\nfor traffic",
                             "observed\n÷ expected", "how unusual\n(z)",
                             "normal\nrange"],
                  loc="center", cellLoc="center",
                  colWidths=[0.14, 0.15, 0.17, 0.17, 0.2, 0.14])
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 2.4)
    _title(ax, "Before and after vs normal traffic")
    fig.suptitle(f"{label.upper()}  {e['callsign']} ({e['icao24']})  ·  "
                 f"runway {e['runway']}  ·  {e['t0_utc']:%Y-%m-%d %H:%M:%S} UTC",
                 fontsize=12)
    name = (f"{e['outcome']}_{e['t0_utc']:%Y%m%d_%H%M%S}Z_{e['callsign']}_"
            f"{e['runway']}.png")
    fig.savefig(outdir / name, dpi=110)
    plt.close(fig)
