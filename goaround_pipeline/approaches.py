"""
Stage 2: detect individual approach attempts within a flight leg.

An approach attempt is a contiguous interval where the aircraft is aligned
with a runway's extended centerline, close to it laterally, tracking the
runway heading, and descending through the approach altitude gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

NM_PER_DEG_LAT = 60.0
FT_PER_NM = 6076.12


def _runway_frame(lat, lon, rwy):
    """Along-track / cross-track distances (NM) of points relative to a
    runway threshold, measured along the approach course. Along-track is
    positive on final (before the threshold), negative past it."""
    th_lat, th_lon, bearing = rwy
    coslat = np.cos(np.radians(th_lat))
    dx = (lon - th_lon) * NM_PER_DEG_LAT * coslat  # east, NM
    dy = (lat - th_lat) * NM_PER_DEG_LAT           # north, NM
    b = np.radians(bearing)
    # unit vector pointing along the runway heading (direction of landing)
    ux, uy = np.sin(b), np.cos(b)
    along = -(dx * ux + dy * uy)      # + = on final, approaching threshold
    cross = dx * uy - dy * ux
    return along, cross


def _track_delta(track, bearing):
    d = (track - bearing + 180.0) % 360.0 - 180.0
    return np.abs(d)


@dataclass
class Approach:
    leg_id: str
    icao24: str
    callsign: str
    runway: str
    start: pd.Timestamp
    end: pd.Timestamp
    idx_start: int
    idx_end: int          # inclusive, index into the leg dataframe
    data: pd.DataFrame = field(repr=False)


def truncated_final(leg: pd.DataFrame,
                    existing: list[Approach]) -> Approach | None:
    """Catch arrivals whose final approach never met the alignment criteria
    because ADS-B coverage was lost during the base turn / short final:
    the leg simply ends low, slow, and close to the field. Counted so the
    approach denominator stays honest; classified downstream (normally as
    a coverage-truncated full stop). Only meaningful for arrivals datasets,
    where a leg ending low near the field implies a landing."""
    if not config.ASSUME_ARRIVALS_DATASET:
        return None
    t = leg["timestamp"]
    if existing:
        last_end = max(a.end for a in existing)
        if (t.iloc[-1] - last_end).total_seconds() < 120.0:
            return None  # the leg's ending is already covered
    agl = leg["agl"].to_numpy()
    if not np.isfinite(agl[-1]) or agl[-1] > config.TRUNCATED_FINAL_MAX_AGL_FT:
        return None
    a_lat, a_lon = config.AIRPORT_LATLON
    dlat = (leg["latitude"].iat[-1] - a_lat) * NM_PER_DEG_LAT
    dlon = ((leg["longitude"].iat[-1] - a_lon)
            * NM_PER_DEG_LAT * np.cos(np.radians(a_lat)))
    if np.hypot(dlat, dlon) > config.TRUNCATED_FINAL_MAX_DIST_NM:
        return None
    vr = leg["vrate"].to_numpy()
    tail = vr[-60:]
    if np.isfinite(tail).any() and np.nanmedian(tail) > 100.0:
        return None  # climbing away, not an arrival
    # segment = last 120 s of the leg
    sec = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    i0 = int(np.searchsorted(sec, sec[-1] - 120.0))
    # best runway guess from track agreement over the tail
    track_tail = leg["track"].to_numpy()[-30:]
    best, best_td = "", 45.0
    for name, rwy in config.RUNWAYS.items():
        td = float(np.nanmedian(_track_delta(track_tail, rwy[2])))
        if td < best_td:
            best, best_td = name, td
    return Approach(
        leg_id=leg.attrs["leg_id"],
        icao24=leg.attrs["icao24"],
        callsign=leg.attrs["callsign"],
        runway=best,
        start=t.iloc[i0],
        end=t.iloc[-1],
        idx_start=i0,
        idx_end=len(leg) - 1,
        data=leg,
    )


def detect_approaches(leg: pd.DataFrame) -> list[Approach]:
    """Find approach attempts in one flight leg, tagged with runway."""
    n = len(leg)
    lat = leg["latitude"].to_numpy()
    lon = leg["longitude"].to_numpy()
    track = leg["track"].to_numpy()
    agl = leg["agl"].to_numpy()
    t = leg["timestamp"]

    # score alignment against every runway end; pick best per point
    best_mask = np.zeros(n, dtype=bool)
    best_rwy = np.full(n, "", dtype=object)
    best_score = np.full(n, np.inf)
    for name, rwy in config.RUNWAYS.items():
        along, cross = _runway_frame(lat, lon, rwy)
        tdelta = _track_delta(track, rwy[2])
        mask = (
            (along > -0.3)
            & (along < config.APPROACH_MAX_DIST_NM)
            & (np.abs(cross) < config.APPROACH_MAX_XTRACK_NM)
            & (tdelta < config.APPROACH_MAX_TRACK_DELTA_DEG)
            & (agl < config.APPROACH_MAX_AGL_FT)
        )
        # score: prefer the runway whose course matches the track best,
        # with cross-track as tiebreaker
        score = tdelta + 20.0 * np.abs(cross)
        better = mask & (score < best_score)
        best_score[better] = score[better]
        best_rwy[better] = name
        best_mask |= mask

    if not best_mask.any():
        return []

    # group contiguous aligned points by time only; runway assignment per
    # group is a vote, weighted toward low-altitude points, so parallel
    # runways (07L/07R) cannot fragment one approach into pieces
    approaches: list[Approach] = []
    idx = np.flatnonzero(best_mask)
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    breaks = np.flatnonzero(
        np.diff(seconds[idx]) > config.APPROACH_MERGE_GAP_S
    )
    groups = np.split(idx, breaks + 1)

    for g in groups:
        if len(g) < 2:
            continue
        i0, i1 = int(g[0]), int(g[-1])
        dur = seconds[i1] - seconds[i0]
        if dur < config.APPROACH_MIN_DURATION_S:
            continue
        seg_agl = agl[i0 : i1 + 1]
        if np.nanmin(seg_agl) > config.APPROACH_MIN_REACHED_AGL_FT:
            continue
        # require an actual descent within the segment
        if np.nanmax(seg_agl) - np.nanmin(seg_agl) < 100.0:
            continue
        # weighted runway vote: points below 500 ft AGL count 5x, since the
        # low part of the approach identifies the runway unambiguously
        weights = np.where(agl[g] < 500.0, 5.0, 1.0)
        votes: dict[str, float] = {}
        for r, w in zip(best_rwy[g], weights):
            votes[r] = votes.get(r, 0.0) + w
        runway = max(votes, key=votes.get)
        approaches.append(
            Approach(
                leg_id=leg.attrs["leg_id"],
                icao24=leg.attrs["icao24"],
                callsign=leg.attrs["callsign"],
                runway=runway,
                start=t.iloc[i0],
                end=t.iloc[i1],
                idx_start=i0,
                idx_end=i1,
                data=leg,
            )
        )
    return approaches
