"""
Go-around events, their pre/post windows, clusters, matched landing
controls, and the tagging of encounters that involve the event aircraft
(FRAMEWORK.md sections 9.7 and 10).

Events come from the go-around stage's approach table. The anchor t0 is
the climb start (fallback: the profile low point). Matched controls are
normal full-stop arrivals anchored at their low point, outside the
exclusion zone around any go-around.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .loading import epoch_s
from .pipeline import DayCache

EVENT_COLUMNS = ["event_id", "kind", "icao24", "callsign", "runway",
                 "outcome", "anchor", "t0", "t0_utc", "t_pre_start",
                 "t_post_end", "window_min"]


def _anchor(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cs = df["climb_start_utc"] if "climb_start_utc" in df else None
    if cs is None or cs.isna().all():
        return epoch_s(df["t_low_utc"]), np.full(len(df), "t_low")
    use_cs = cs.notna().to_numpy()
    t = np.where(use_cs, epoch_s(cs.fillna(df["t_low_utc"])),
                 epoch_s(df["t_low_utc"]))
    return t.astype(np.int64), np.where(use_cs, "climb_start", "t_low")


def event_table(approaches: pd.DataFrame, outcomes: tuple[str, ...],
                window_min: float) -> pd.DataFrame:
    """One row per go-around event with its pre/post window bounds,
    cluster membership and pre-window contamination flag."""
    ev = approaches[approaches["outcome"].isin(outcomes)].copy()
    ev = ev.sort_values("t_low_utc").reset_index(drop=True)
    w = int(window_min * 60)
    if ev.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS + [
            "cluster_id", "cluster_size", "cluster_first",
            "pre_contaminated"])
    t0, anchor = _anchor(ev)
    out = pd.DataFrame({
        "event_id": [f"ga_{i:04d}" for i in range(len(ev))],
        "kind": "go_around",
        "icao24": ev["icao24"].astype(str).to_numpy(),
        "callsign": ev["callsign"].astype(str).to_numpy(),
        "runway": ev["runway"].astype(str).to_numpy(),
        "outcome": ev["outcome"].to_numpy(),
        "anchor": anchor,
        "t0": t0,
        "t0_utc": pd.to_datetime(t0, unit="s", utc=True),
        "t_pre_start": t0 - w,
        "t_post_end": t0 + w,
        "window_min": window_min,
        "min_agl_ft": ev["min_agl_ft"].to_numpy(),
    })
    out = out.sort_values("t0").reset_index(drop=True)
    # clusters: go-arounds within W of each other (section 10)
    t = out["t0"].to_numpy()
    new = np.ones(len(t), dtype=bool)
    new[1:] = (t[1:] - t[:-1]) > w
    cid = np.cumsum(new) - 1
    out["cluster_id"] = cid
    sizes = np.bincount(cid)
    out["cluster_size"] = sizes[cid]
    out["cluster_first"] = new
    # contaminated pre window: another event (any outcome in the
    # ambiguous set too) within W before t0
    all_ga = approaches[approaches["outcome"].isin(
        config.EVENT_OUTCOMES_AMBIGUOUS)]
    t_all = np.sort(_anchor(all_ga)[0]) if len(all_ga) else np.array([])
    contaminated = np.zeros(len(t), dtype=bool)
    for i, ti in enumerate(t):
        lo = np.searchsorted(t_all, ti - w)
        hi = np.searchsorted(t_all, ti, side="left")
        contaminated[i] = (hi - lo) > 0
    out["pre_contaminated"] = contaminated
    return out


def control_table(approaches: pd.DataFrame, window_min: float,
                  exclusion_min: float, max_n: int | None = None,
                  seed: int = 0) -> pd.DataFrame:
    """Matched landing controls (section 9.7): full-stop arrivals
    anchored at their low point, excluding any within +/- exclusion_min
    of a go-around or ambiguous go-around."""
    ctl = approaches[approaches["outcome"].isin(config.CONTROL_OUTCOMES)
                     & approaches["t_low_utc"].notna()].copy()
    if ctl.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    t_low = epoch_s(ctl["t_low_utc"])
    ga = approaches[approaches["outcome"].isin(config.EVENT_OUTCOMES_AMBIGUOUS)]
    t_ga = np.sort(_anchor(ga)[0]) if len(ga) else np.array([])
    ex = int(exclusion_min * 60)
    keep = np.ones(len(ctl), dtype=bool)
    if len(t_ga):
        lo = np.searchsorted(t_ga, t_low - ex)
        hi = np.searchsorted(t_ga, t_low + ex, side="right")
        keep = (hi - lo) == 0
    ctl = ctl[keep]
    t_low = t_low[keep]
    if max_n and len(ctl) > max_n:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(len(ctl), max_n, replace=False))
        ctl = ctl.iloc[pick]
        t_low = t_low[pick]
    w = int(window_min * 60)
    out = pd.DataFrame({
        "event_id": [f"ctl_{i:05d}" for i in range(len(ctl))],
        "kind": "control",
        "icao24": ctl["icao24"].astype(str).to_numpy(),
        "callsign": ctl["callsign"].astype(str).to_numpy(),
        "runway": ctl["runway"].astype(str).to_numpy(),
        "outcome": ctl["outcome"].to_numpy(),
        "anchor": "t_low",
        "t0": t_low,
        "t0_utc": pd.to_datetime(t_low, unit="s", utc=True),
        "t_pre_start": t_low - w,
        "t_post_end": t_low + w,
        "window_min": window_min,
    })
    return out.sort_values("t0").reset_index(drop=True)


def event_windows(events: pd.DataFrame) -> pd.DataFrame:
    """Two interval rows (pre, post) per event, for interval_metrics."""
    rows = []
    for _, e in events.iterrows():
        rows.append({"event_id": e["event_id"], "window": "pre",
                     "t_start": int(e["t_pre_start"]), "t_end": int(e["t0"])})
        rows.append({"event_id": e["event_id"], "window": "post",
                     "t_start": int(e["t0"]), "t_end": int(e["t_post_end"])})
    return pd.DataFrame(rows, columns=["event_id", "window", "t_start",
                                       "t_end"])


def tag_encounters(enc: pd.DataFrame, events: pd.DataFrame,
                   prefix: str = "ga") -> pd.DataFrame:
    """Mark encounters in which one aircraft is an event aircraft, within
    that event's pre/post span: adds <prefix>_event_id and
    <prefix>_window (pre | post)."""
    enc = enc.copy()
    enc[f"{prefix}_event_id"] = None
    enc[f"{prefix}_window"] = None
    if enc.empty or events.empty:
        return enc
    t = enc["t_min"].to_numpy()
    ia = enc["icao24_a"].to_numpy(); ib = enc["icao24_b"].to_numpy()
    eid = np.full(len(enc), None, dtype=object)
    win = np.full(len(enc), None, dtype=object)
    by_icao: dict[str, list[int]] = {}
    for i, ac in enumerate(events["icao24"].to_numpy()):
        by_icao.setdefault(ac, []).append(i)
    t0 = events["t0"].to_numpy(); ts = events["t_pre_start"].to_numpy()
    te = events["t_post_end"].to_numpy()
    ids = events["event_id"].to_numpy()
    for j in range(len(enc)):
        for ac in (ia[j], ib[j]):
            for i in by_icao.get(ac, ()):
                if ts[i] <= t[j] < te[i]:
                    eid[j] = ids[i]
                    win[j] = "pre" if t[j] < t0[i] else "post"
                    break
            if eid[j] is not None:
                break
    enc[f"{prefix}_event_id"] = eid
    enc[f"{prefix}_window"] = win
    return enc


def event_aircraft_heights(events: pd.DataFrame, days: dict[str, DayCache],
                           radius_nm: float) -> pd.DataFrame:
    """Per event: max height above field of the event aircraft during
    its post window while within the radius, and the share of its
    post-window presence spent beyond the radius (section 3.3)."""
    rows = []
    for _, e in events.iterrows():
        h_max, present, beyond = np.nan, 0, 0
        for dc in days.values():
            t0 = int(dc.meta["t0"])
            if t0 > e["t_post_end"] or t0 + 86400 <= e["t0"]:
                continue
            legs = dc.legs[dc.legs["icao24"] == e["icao24"]]["leg"]
            if legs.empty:
                continue
            p = dc.presence
            m = (p["leg"].isin(legs) & (p["t"] >= e["t0"])
                 & (p["t"] < e["t_post_end"]))
            if not m.any():
                continue
            sub = p[m]
            inside = sub["r"] <= radius_nm
            if inside.any():
                h_max = np.nanmax([h_max, sub.loc[inside, "h"].max()])
            present += len(sub)
            beyond += int((~inside).sum())
        rows.append({"event_id": e["event_id"],
                     "max_height_post_ft": h_max,
                     "post_present_s": present,
                     "post_beyond_radius_s": beyond})
    return pd.DataFrame(rows, columns=["event_id", "max_height_post_ft",
                                       "post_present_s",
                                       "post_beyond_radius_s"])
