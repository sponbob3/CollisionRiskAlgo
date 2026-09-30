"""
Summary report: statistics tables (markdown) + figures from the classified
approach table produced by pipeline.py.

Usage:
    python -m goaround_pipeline.report
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# dataviz reference palette (light mode), categorical slots 1-3
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

GA_OUTCOMES = ["go_around", "ga_ambiguous"]


def load() -> pd.DataFrame:
    df = pd.read_csv(config.OUTPUT_DIR / "all_approaches.csv")
    df["t_low_utc"] = pd.to_datetime(df["t_low_utc"], utc=True)
    # t_local carries mixed EST/EDT offsets; re-derive from UTC
    df["t_local"] = df["t_low_utc"].dt.tz_convert(config.LOCAL_TZ)
    df["month"] = df["t_local"].dt.to_period("M").astype(str)
    df["hour_local"] = df["t_local"].dt.hour
    df["is_ga"] = df["outcome"].isin(GA_OUTCOMES)
    # denominator: independent approach attempts (continued fragments folded)
    df["is_attempt"] = ~df["outcome"].isin(["continued_approach"])
    return df


def fig_monthly(df, out):
    att = df[df.is_attempt]
    g = att.groupby("month").agg(
        approaches=("is_ga", "size"), ga=("is_ga", "sum"))
    g["rate"] = 1000 * g["ga"] / g["approaches"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(g))
    ax1.bar(x, g["ga"], width=0.68, color=C1)
    ax1.set_xticks(x, [m[2:] for m in g.index], rotation=45, ha="right")
    ax1.set_ylabel("go-around events")
    ax1.set_title("Go-arounds per month", loc="left", fontsize=11)
    ax2.plot(x, g["rate"], color=C2, lw=2, marker="o", ms=5)
    ax2.set_xticks(x, [m[2:] for m in g.index], rotation=45, ha="right")
    ax2.set_ylabel("go-arounds per 1,000 approaches")
    ax2.set_ylim(bottom=0)
    ax2.set_title("Go-around rate", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return g


def fig_runway(df, out):
    att = df[df.is_attempt & (df.runway != "")]
    g = (att.groupby("runway").agg(approaches=("is_ga", "size"),
                                   ga=("is_ga", "sum"))
         .sort_values("approaches", ascending=False))
    g["rate"] = 1000 * g["ga"] / g["approaches"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    y = np.arange(len(g))
    ax1.barh(y, g["approaches"], height=0.68, color=C1)
    ax1.set_yticks(y, g.index); ax1.invert_yaxis()
    ax1.set_xlabel("approach attempts")
    ax1.set_title("Approaches by runway", loc="left", fontsize=11)
    ax2.barh(y, g["ga"], height=0.68, color=C2)
    ax2.set_yticks(y, g.index); ax2.invert_yaxis()
    ax2.set_xlabel("go-around events")
    ax2.set_title("Go-arounds by runway", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return g


def fig_hour(df, out):
    att = df[df.is_attempt]
    g = att.groupby("hour_local").agg(
        approaches=("is_ga", "size"), ga=("is_ga", "sum"))
    g = g.reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(g.index, g["approaches"], width=0.7, color=C1,
           label="all approaches")
    ax.bar(g.index, g["ga"] * 20, width=0.7, color=C2,
           label="go-arounds (x20)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("local hour"); ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Activity by local hour", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_profiles(df, out):
    ga = df[df.outcome == "go_around"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    bins = np.arange(0, 1100, 50)
    ax1.hist(ga["init_agl_ft"].clip(0, 1050), bins=bins, color=C1,
             edgecolor="#fcfcfb", linewidth=1)
    ax1.set_xlabel("altitude at climb initiation (ft AGL)")
    ax1.set_ylabel("go-around events")
    ax1.set_title("Go-around initiation altitude", loc="left", fontsize=11)
    ax2.hist(ga["peak_climb_fpm"].clip(0, 3000), bins=np.arange(0, 3100, 150),
             color=C1, edgecolor="#fcfcfb", linewidth=1)
    ax2.set_xlabel("peak climb rate (fpm)")
    ax2.set_ylabel("go-around events")
    ax2.set_title("Go-around climb performance", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_runway_rate(df, out):
    att = df[df.is_attempt & (df.runway != "") & df.runway.notna()]
    g = (att.groupby("runway").agg(approaches=("is_ga", "size"),
                                   ga=("is_ga", "sum")))
    g = g[g["approaches"] >= 200]
    g["rate"] = 1000 * g["ga"] / g["approaches"]
    g = g.sort_values("rate", ascending=True)
    overall = 1000 * att["is_ga"].sum() / len(att)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(g))
    ax.barh(y, g["rate"], height=0.62, color=C1)
    for yi, (rw, row) in zip(y, g.iterrows()):
        ax.annotate(f"{row['rate']:.1f}  (n={int(row['ga'])})",
                    (row["rate"], yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=MUTED)
    ax.axvline(overall, color=C2, lw=1.4, ls="--")
    ax.annotate(f"airport overall {overall:.1f}", (overall, len(g) - 0.4),
                xytext=(5, 0), textcoords="offset points", fontsize=8.5,
                color=C2)
    ax.set_yticks(y, g.index)
    ax.set_xlabel("go-arounds per 1,000 approach attempts")
    ax.set_title("Go-around rate by runway", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_map(df, out):
    """Where go-arounds begin: low points over the runway layout,
    colored by initiation altitude (sequential, one hue)."""
    from .viz import _draw_runways
    from .approaches import NM_PER_DEG_LAT
    ga = df[(df.outcome == "go_around")
            & df.lat_low.notna() & df.lon_low.notna()]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    _draw_runways(ax)
    pts = ax.scatter(ga["lon_low"], ga["lat_low"],
                     c=ga["min_agl_ft"].clip(0, 800), cmap="Blues_r",
                     s=14, alpha=0.75, edgecolors="none", vmin=0, vmax=800)
    fig.colorbar(pts, ax=ax, label="minimum AGL at low point (ft)",
                 shrink=0.8)
    ax.set_aspect(1.0 / np.cos(np.radians(config.AIRPORT_LATLON[0])))
    lat0, lon0 = config.AIRPORT_LATLON
    span = 4.0 / NM_PER_DEG_LAT  # ~4 NM half-window
    ax.set_xlim(lon0 - span / np.cos(np.radians(lat0)),
                lon0 + span / np.cos(np.radians(lat0)))
    ax.set_ylim(lat0 - span, lat0 + span)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Go-around initiation points at "
                 f"{config.AIRPORT_ICAO} (n={len(ga)})", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_plateau(df, out):
    """The methods figure: plateau-duration separation of the classes."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bins = np.arange(0, 90, 3)
    specs = [
        ("go_around", C2, "go-around"),
        ("low_approach", C3, "low approach"),
        ("touch_and_go", C1, "touch-and-go"),
    ]
    for outcome, color, label in specs:
        vals = df[df.outcome == outcome]["level_low_duration_s"].clip(0, 87)
        ax.hist(vals, bins=bins, histtype="step", lw=2, color=color,
                label=label, density=True)
    ax.axvspan(config.LEVEL_GA_MAX_S, config.LEVEL_LOWAPP_MIN_S,
               color="#e8e7e2", alpha=0.6, zorder=0)
    ax.set_xlabel("plateau duration at profile low point (s)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Class separation by low-point plateau duration "
                 "(shaded = ambiguous band)", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main(out_dir=None) -> None:
    if out_dir is not None:
        from pathlib import Path
        config.OUTPUT_DIR = Path(out_dir)
    df = load()
    figdir = config.OUTPUT_DIR / "figures"
    figdir.mkdir(exist_ok=True)

    monthly = fig_monthly(df, figdir / "monthly_go_arounds.png")
    runway = fig_runway(df, figdir / "runway_distribution.png")
    fig_hour(df, figdir / "hourly_activity.png")
    fig_profiles(df, figdir / "go_around_profiles.png")
    fig_plateau(df, figdir / "plateau_separation.png")
    fig_runway_rate(df, figdir / "runway_rate.png")
    if "lat_low" in df.columns:
        fig_map(df, figdir / "go_around_map.png")

    att = df[df.is_attempt]
    ga = df[df.outcome == "go_around"]
    amb = df[df.outcome == "ga_ambiguous"]
    n_att = len(att)
    lines = [
        "# KDAB go-around detection - summary report",
        "",
        f"Dataset: {df['t_local'].dt.date.min()} to "
        f"{df['t_local'].dt.date.max()}, approaches at {config.AIRPORT_ICAO}.",
        "",
        "## Headline numbers",
        "",
        f"- **{len(ga)} go-around events** "
        f"(plus {len(amb)} ambiguous candidates, kept flagged)",
        f"- {n_att:,} independent approach attempts "
        f"({len(df):,} aligned segments incl. continued fragments)",
        f"- go-around rate: **{1000 * len(ga) / n_att:.1f} per 1,000 "
        "approaches** (strict definition, ambiguous excluded)",
        "",
        "## Outcome breakdown",
        "",
        df["outcome"].value_counts().to_markdown(),
        "",
        "## Monthly",
        "",
        monthly.round(2).to_markdown(),
        "",
        "## By runway",
        "",
        runway.round(2).to_markdown(),
        "",
        "## Go-around event characteristics",
        "",
        ga[["min_agl_ft", "init_agl_ft", "peak_climb_fpm",
            "alt_regain_ft", "level_low_duration_s"]]
        .describe().round(1).to_markdown(),
        "",
        "## What followed each go-around",
        "",
        ga["next_action"].value_counts().to_markdown(),
        "",
    ]
    (config.OUTPUT_DIR / "summary_report.md").write_text("\n".join(lines))
    print(f"Wrote {config.OUTPUT_DIR / 'summary_report.md'} and "
          f"{figdir}/*.png")


if __name__ == "__main__":
    main()
