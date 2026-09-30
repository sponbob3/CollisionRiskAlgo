"""
Flight phase and encounter geometry classes (FRAMEWORK.md section 6.3).

Both reuse the go-around pipeline's runway frame: along-track distance
(positive on final, before the threshold; negative past it) and cross-track
distance relative to each runway threshold, plus the track's deviation
from the runway heading. Here the frame is built in the local plane (NM)
rather than from lat/lon, which is the same construction with the same
sign convention.

Phase codes (int16, per aircraft-second):
    0                 other
    1 + k             arrival on final to runway end k
    DEPARTURE_BASE + 1 + k   departure from runway end k
where k indexes RUNWAY_NAMES (the profile's runway order).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import goaround_adapter as ga

DEPARTURE_BASE = 100
CLASSES = ("in_trail_same_runway", "parallel_runways", "crossing",
           "head_on", "runway_area", "other")
_rwy_cache: dict = {}


def runway_names() -> list[str]:
    return list(ga.config.RUNWAYS)


def runway_table():
    """(names, threshold x, threshold y, bearing) in the local plane."""
    key = (ga.config.AIRPORT_ICAO, tuple(ga.config.RUNWAYS.items()))
    if key not in _rwy_cache:
        from .loading import project
        names = runway_names()
        lat = np.array([ga.config.RUNWAYS[n][0] for n in names])
        lon = np.array([ga.config.RUNWAYS[n][1] for n in names])
        brg = np.array([ga.config.RUNWAYS[n][2] for n in names])
        x, y = project(lat, lon)
        _rwy_cache.clear()
        _rwy_cache[key] = (names, x, y, brg)
    return _rwy_cache[key]


def runway_frame_xy(x, y, k: int):
    """Along-track / cross-track (NM) relative to runway end k."""
    _, tx, ty, brg = runway_table()
    b = np.radians(brg[k])
    ux, uy = np.sin(b), np.cos(b)
    dx, dy = x - tx[k], y - ty[k]
    along = -(dx * ux + dy * uy)
    cross = dx * uy - dy * ux
    return along, cross


def flight_phase(x, y, track, h, vrate) -> np.ndarray:
    """Phase code per aircraft-second (vectorised over samples).

    Arrival on final to k: on the approach side of threshold k, within the
    cross-track and track tolerance of its course, below the height gate,
    not climbing. Departure from k: aligned with the runway heading over or
    beyond the threshold and climbing. The best-matching runway end wins
    (smallest track deviation, cross-track as tie-breaker), as in the
    go-around pipeline's approach detector."""
    names, _, _, brg = runway_table()
    x = np.asarray(x, float); y = np.asarray(y, float)
    track = np.asarray(track, float); h = np.asarray(h, float)
    vrate = np.asarray(vrate, float)
    n = len(x)
    code = np.zeros(n, dtype=np.int16)
    best = np.full(n, np.inf)
    for k in range(len(names)):
        along, cross = runway_frame_xy(x, y, k)
        tdelta = ga.track_delta(track, brg[k])
        aligned = ((np.abs(cross) < config.PHASE_MAX_XTRACK_NM)
                   & (tdelta < config.PHASE_MAX_TRACK_DELTA_DEG))
        arr = (aligned & (along > -0.3)
               & (along < config.PHASE_ARRIVAL_MAX_DIST_NM)
               & (h < config.PHASE_ARRIVAL_MAX_HEIGHT_FT)
               & ~(vrate >= config.PHASE_DEPARTURE_MIN_VRATE_FPM))
        dep = (aligned & (along < 0.5)
               & (along > -config.PHASE_DEPARTURE_MAX_DIST_NM)
               & (vrate >= config.PHASE_DEPARTURE_MIN_VRATE_FPM))
        score = tdelta + 20.0 * np.abs(cross)
        better = (arr | dep) & (score < best)
        best[better] = score[better]
        code[better & arr] = 1 + k
        code[better & dep] = DEPARTURE_BASE + 1 + k
    return code


def phase_runway_index(code) -> np.ndarray:
    """Runway end index of a phase code (-1 for other)."""
    code = np.asarray(code, dtype=int)
    k = np.where(code >= DEPARTURE_BASE + 1, code - DEPARTURE_BASE - 1,
                 code - 1)
    return np.where(code == 0, -1, k)


def phase_label(code) -> np.ndarray:
    names = runway_names()
    code = np.asarray(code, dtype=int)
    out = np.full(len(code), "other", dtype=object)
    k = phase_runway_index(code)
    arr = (code > 0) & (code < DEPARTURE_BASE + 1)
    dep = code >= DEPARTURE_BASE + 1
    out[arr] = [f"arrival_{names[i]}" for i in k[arr]]
    out[dep] = [f"departure_{names[i]}" for i in k[dep]]
    return out


def dist_to_nearest_threshold(x, y) -> np.ndarray:
    _, tx, ty, _ = runway_table()
    x = np.asarray(x, float)[:, None]; y = np.asarray(y, float)[:, None]
    return np.min(np.hypot(x - tx[None, :], y - ty[None, :]), axis=1)


def geometry_class(phase_a, phase_b, track_a, track_b, xa, ya, ha,
                   xb, yb, hb) -> np.ndarray:
    """Geometry class of each encounter at its closest point."""
    _, _, _, brg = runway_table()
    phase_a = np.asarray(phase_a, int); phase_b = np.asarray(phase_b, int)
    dpsi = np.abs(ga.track_delta(np.asarray(track_a, float),
                                 np.asarray(track_b, float)))
    ka, kb = phase_runway_index(phase_a), phase_runway_index(phase_b)
    both_aligned = (ka >= 0) & (kb >= 0)
    same_end = both_aligned & (ka == kb)
    hd = np.abs(ga.track_delta(brg[np.maximum(ka, 0)],
                               brg[np.maximum(kb, 0)]))
    parallel = (both_aligned & (ka != kb)
                & (hd <= config.GEOM_PARALLEL_HEADING_DEG))
    near_a = (dist_to_nearest_threshold(xa, ya)
              <= config.GEOM_RUNWAY_AREA_MAX_DIST_NM)
    near_b = (dist_to_nearest_threshold(xb, yb)
              <= config.GEOM_RUNWAY_AREA_MAX_DIST_NM)
    low = ((np.asarray(ha, float) < config.GEOM_RUNWAY_AREA_MAX_HEIGHT_FT)
           & (np.asarray(hb, float) < config.GEOM_RUNWAY_AREA_MAX_HEIGHT_FT))
    runway_area = near_a & near_b & low

    out = np.full(len(dpsi), "other", dtype=object)
    out[dpsi > config.GEOM_CROSSING_MAX_DPSI_DEG] = "head_on"
    out[(dpsi >= config.GEOM_IN_TRAIL_MAX_DPSI_DEG)
        & (dpsi <= config.GEOM_CROSSING_MAX_DPSI_DEG)] = "crossing"
    out[parallel] = "parallel_runways"
    out[same_end & (dpsi < config.GEOM_IN_TRAIL_MAX_DPSI_DEG)] = \
        "in_trail_same_runway"
    out[runway_area] = "runway_area"
    return out
