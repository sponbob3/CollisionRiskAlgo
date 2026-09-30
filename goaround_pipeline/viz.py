"""
Per-event trajectory plots: map panel + altitude/vertical-rate panel.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .approaches import NM_PER_DEG_LAT
from .classify import ClassifiedApproach

def _runway_pairs():
    """Pair reciprocal runway ends (bearings ~180 deg apart, thresholds
    within a few NM) so each physical runway is drawn once, threshold to
    threshold - works for any airport profile."""
    items = list(config.RUNWAYS.items())
    pairs, used = [], set()
    for i, (na, (lata, lona, ba)) in enumerate(items):
        if na in used:
            continue
        for nb, (latb, lonb, bb) in items[i + 1:]:
            if nb in used:
                continue
            recip = abs((ba - bb + 360.0) % 360.0 - 180.0) < 15.0
            coslat = np.cos(np.radians(lata))
            dist = np.hypot((latb - lata) * NM_PER_DEG_LAT,
                            (lonb - lona) * NM_PER_DEG_LAT * coslat)
            if recip and dist < 4.0:
                pairs.append((na, (lata, lona), nb, (latb, lonb)))
                used.update((na, nb))
                break
        else:
            used.add(na)
            # no partner in the database: draw a 1 NM stub along the bearing
            b = np.radians(ba)
            coslat = np.cos(np.radians(lata))
            end = (lata + np.cos(b) / NM_PER_DEG_LAT,
                   lona + np.sin(b) / (NM_PER_DEG_LAT * coslat))
            pairs.append((na, (lata, lona), "", end))
    return pairs


def _draw_runways(ax):
    for na, (lata, lona), nb, (latb, lonb) in _runway_pairs():
        ax.plot([lona, lonb], [lata, latb],
                lw=4, color="0.35", solid_capstyle="butt", zorder=1)
        ax.annotate(na, (lona, lata), fontsize=7, color="0.35",
                    ha="right", va="top")


def plot_event(r: ClassifiedApproach, out_path, window_s: float = 240.0):
    app = r.approach
    leg = app.data
    t = leg["timestamp"]
    sec = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    s0 = sec[app.idx_start] - window_s / 2
    s1 = sec[app.idx_end] + window_s
    m = (sec >= s0) & (sec <= s1)
    w = leg[m]
    wsec = sec[m]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.5), width_ratios=[1, 1.4]
    )
    fig.suptitle(
        f"{r.outcome.upper()}  {app.callsign} ({app.icao24})  rwy {app.runway}"
        f"  {r.t_low:%Y-%m-%d %H:%M:%S}Z   min AGL {r.min_agl_ft:.0f} ft",
        fontsize=10,
    )

    # ---- map panel ----
    _draw_runways(ax1)
    pts = ax1.scatter(w["longitude"], w["latitude"], c=w["agl"],
                      cmap="viridis", s=4, vmin=0, vmax=1200, zorder=2)
    app_m = (wsec >= sec[app.idx_start]) & (wsec <= sec[app.idx_end])
    ax1.plot(w["longitude"][app_m], w["latitude"][app_m],
             color="crimson", lw=0.8, alpha=0.6, zorder=3)
    ilow = leg.index[
        (t == r.t_low)
    ]
    if len(ilow):
        ax1.scatter(leg.loc[ilow, "longitude"], leg.loc[ilow, "latitude"],
                    marker="v", color="crimson", s=60, zorder=4,
                    label="low point")
    fig.colorbar(pts, ax=ax1, label="AGL (ft)", shrink=0.85)
    ax1.set_aspect(1.0 / np.cos(np.radians(config.AIRPORT_LATLON[0])))
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    ax1.legend(loc="upper left", fontsize=7)

    # ---- profile panel ----
    rel = wsec - sec[app.idx_start]
    ax2.plot(rel, w["agl"], color="tab:blue", lw=1.2, label="AGL (ft)")
    ax2.axvspan(0, sec[app.idx_end] - sec[app.idx_start],
                color="tab:blue", alpha=0.08, label="aligned segment")
    tlow_rel = (r.t_low - t.iloc[app.idx_start]).total_seconds()
    ax2.axvline(tlow_rel, color="crimson", lw=0.8, ls="--", label="low point")
    if r.climb_start is not None:
        cs_rel = (r.climb_start - t.iloc[app.idx_start]).total_seconds()
        ax2.axvline(cs_rel, color="tab:green", lw=0.8, ls="--",
                    label="climb start")
    ax2.set_xlabel("seconds from approach start")
    ax2.set_ylabel("AGL (ft)", color="tab:blue")
    ax2.set_ylim(bottom=-50)
    ax3 = ax2.twinx()
    ax3.plot(rel, w["vrate"], color="tab:orange", lw=0.8, alpha=0.7,
             label="vertical rate")
    ax3.axhline(0, color="tab:orange", lw=0.5, ls=":", alpha=0.5)
    ax3.set_ylabel("vertical rate (fpm)", color="tab:orange")
    gs_ax = ax2.twinx()
    gs_ax.spines.right.set_position(("axes", 1.12))
    gs_ax.plot(rel, w["groundspeed"], color="0.5", lw=0.8, alpha=0.6)
    gs_ax.set_ylabel("groundspeed (kt)", color="0.5")
    ax2.legend(loc="upper center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
