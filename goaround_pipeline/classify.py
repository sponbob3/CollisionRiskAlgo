"""
Stage 3: classify the outcome of each detected approach.

Outcomes:
  full_stop     - touchdown, no subsequent climb-out on this leg
  touch_and_go  - touchdown followed by climb-out
  go_around     - no touchdown; abrupt descent-to-climb reversal (V profile)
  low_approach  - no touchdown; extended level segment at low altitude (U)
  ga_ambiguous  - no touchdown + climb, but level-segment duration between
                  the go-around and low-approach cutoffs
  unresolved    - approach segment ends without touchdown or climb evidence
                  (usually coverage loss)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .approaches import Approach, _runway_frame


@dataclass
class ClassifiedApproach:
    approach: Approach
    outcome: str
    coverage_truncated: bool
    # metrics (NaN where not applicable)
    min_agl_ft: float
    t_low: pd.Timestamp
    lat_low: float
    lon_low: float
    touchdown: bool
    level_low_duration_s: float
    climb_start: pd.Timestamp | None
    init_agl_ft: float
    init_dist_thr_nm: float
    peak_climb_fpm: float
    alt_regain_ft: float
    # filled by chain post-processing in pipeline.py
    next_action: str = "leg_end"
    next_gap_s: float = float("nan")


def _plateau_duration(agl: np.ndarray, sec: np.ndarray,
                      ilow: int, min_agl: float) -> float:
    """Duration (s) of the contiguous stretch of samples within
    PLATEAU_BAND_FT of the profile minimum, containing the low point.
    Short sample dropouts inside the band are tolerated."""
    in_band = agl <= min_agl + config.PLATEAU_BAND_FT
    idx = np.flatnonzero(in_band)
    if len(idx) == 0:
        return 0.0
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 5) + 1)
    for run in runs:
        if run[0] <= ilow <= run[-1]:
            return float(sec[run[-1]] - sec[run[0]])
    return 0.0


def _touchdown_evidence(seg: pd.DataFrame, agl: np.ndarray, sec: np.ndarray,
                        min_agl: float, plateau_s: float) -> bool:
    low = agl < 300.0
    if not low.any():
        return False
    if bool(seg["onground"].to_numpy()[low].any()):
        return True
    if min_agl <= config.TOUCHDOWN_AGL_FT:
        return True
    # sustained plateau at the bottom, low enough to be a landing roll
    # (GNSS bias means "on the runway" can indicate up to ~100 ft AGL)
    if (min_agl <= config.TOUCHDOWN_PLATEAU_MAX_AGL_FT
            and plateau_s >= config.TOUCHDOWN_PLATEAU_MIN_S):
        return True
    gs = seg["groundspeed"].to_numpy()
    very_low = agl < 200.0
    if very_low.any() and np.isfinite(gs[very_low]).any():
        if np.nanmin(gs[very_low]) <= config.TOUCHDOWN_GS_KT:
            return True
    # coverage-floor gap: consecutive points straddling a long gap while low
    gaps = np.diff(sec)
    if len(gaps):
        gap_low = (gaps > config.TOUCHDOWN_GAP_S) & (
            agl[:-1] < config.TOUCHDOWN_GAP_AGL_FT
        )
        if gap_low.any():
            return True
    return False


def classify_approach(app: Approach) -> ClassifiedApproach:
    leg = app.data
    t = leg["timestamp"]
    sec_all = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    agl_all = leg["agl"].to_numpy()
    vrate_all = leg["vrate"].to_numpy()

    # analysis window: approach segment plus 180 s of aftermath
    i0 = app.idx_start
    end_sec = sec_all[app.idx_end] + 180.0
    i1 = int(np.searchsorted(sec_all, end_sec, side="right")) - 1
    sl = slice(i0, i1 + 1)
    agl = agl_all[sl]
    vrate = vrate_all[sl]
    sec = sec_all[sl]
    seg = leg.iloc[sl]

    # low point: minimum AGL within the aligned segment itself
    seg_end_rel = app.idx_end - i0
    ilow_rel = int(np.nanargmin(agl[: seg_end_rel + 1]))
    min_agl = float(agl[ilow_rel])
    t_low = t.iloc[i0 + ilow_rel]

    # threshold-relative geometry for metrics (runway can be unknown for
    # coverage-truncated finals)
    if app.runway in config.RUNWAYS:
        rwy = config.RUNWAYS[app.runway]
        along, _ = _runway_frame(
            seg["latitude"].to_numpy(), seg["longitude"].to_numpy(), rwy
        )
    else:
        along = np.full(len(seg), np.nan)

    plateau_s = _plateau_duration(agl, sec, ilow_rel, min_agl)
    touchdown = _touchdown_evidence(
        seg.iloc[: seg_end_rel + 1], agl[: seg_end_rel + 1],
        sec[: seg_end_rel + 1], min_agl, plateau_s,
    )

    # ---- climb detection after the low point --------------------------
    post = slice(ilow_rel, len(agl))
    post_sec = sec[post]
    post_vrate = vrate[post]
    post_agl = agl[post]
    climbing = post_vrate >= config.GA_CLIMB_VRATE_FPM
    climb_start_rel = None
    if climbing.any():
        # first run of sustained climb within 120 s of the low point
        idx = np.flatnonzero(climbing)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1)
        for run in runs:
            if post_sec[run[0]] - post_sec[0] > 120.0:
                break
            if post_sec[run[-1]] - post_sec[run[0]] >= config.GA_CLIMB_DURATION_S:
                climb_start_rel = int(run[0])
                break
    alt_regain = float(np.nanmax(post_agl) - min_agl) if len(post_agl) else 0.0
    sustained_climb = (
        climb_start_rel is not None
        and alt_regain >= config.GA_ALT_REGAIN_FT
    )

    # Shape metric for the climb-away split: time from the last descent into
    # the low band (min AGL + PLATEAU_BAND_FT) until sustained climb start.
    # Anchoring on climb start makes the metric immune to slow altitude
    # drift across the band edge during a level segment; the band-run
    # plateau_s (used for touchdown evidence) remains the fallback when
    # there is no climb.
    level_dur = plateau_s
    if climb_start_rel is not None:
        band_top = min_agl + config.PLATEAU_BAND_FT
        above = np.flatnonzero(agl[: ilow_rel + 1] > band_top)
        entry_rel = int(above[-1]) + 1 if len(above) else 0
        cs_abs_rel = ilow_rel + climb_start_rel
        if cs_abs_rel > entry_rel:
            level_dur = float(sec[cs_abs_rel] - sec[entry_rel])

    # ---- decision tree -------------------------------------------------
    # data ending low + descending at the end of the leg = landing that
    # finished below the ADS-B coverage floor (arrivals dataset: legs end
    # at the final landing)
    tail_gap_s = float(sec_all[-1] - sec_all[app.idx_end])
    coverage_truncated = (
        config.ASSUME_ARRIVALS_DATASET
        and not touchdown
        and tail_gap_s <= config.END_TRUNCATED_MAX_GAP_S
        and np.isfinite(agl_all[-1])
        and agl_all[-1] <= config.END_TRUNCATED_MAX_AGL_FT
        and (np.isnan(vrate_all[-1]) or vrate_all[-1] < 100.0)
    )
    observed_descent = (
        float(np.nanmax(agl[: ilow_rel + 1])) - min_agl
        if ilow_rel >= 1 else 0.0
    )
    if touchdown:
        # airborne climb after touchdown? -> touch and go
        outcome = "touch_and_go" if sustained_climb else "full_stop"
    elif coverage_truncated and not sustained_climb:
        outcome = "full_stop"
    elif (sustained_climb
          and observed_descent < config.GA_MIN_OBSERVED_DESCENT_FT):
        # climb-out with no observed approach descent: the final happened
        # below the coverage floor - could be a go-around OR a touch-and-go
        outcome = "unresolved"
    elif sustained_climb and min_agl <= config.GA_MAX_LOW_AGL_FT:
        if level_dur <= config.LEVEL_GA_MAX_S:
            outcome = "go_around"
        elif level_dur >= config.LEVEL_LOWAPP_MIN_S:
            outcome = "low_approach"
        else:
            outcome = "ga_ambiguous"
    elif sustained_climb:
        outcome = "high_breakoff"
    else:
        outcome = "unresolved"

    if climb_start_rel is not None:
        cs_abs = ilow_rel + climb_start_rel
        climb_start = t.iloc[i0 + cs_abs]
        init_agl = float(agl[cs_abs])
        init_dist = float(along[cs_abs])
        peak_climb = float(
            np.nanmax(post_vrate[climb_start_rel : climb_start_rel + 90])
        )
    else:
        climb_start, init_agl, init_dist, peak_climb = None, np.nan, np.nan, np.nan

    return ClassifiedApproach(
        approach=app,
        outcome=outcome,
        coverage_truncated=coverage_truncated,
        min_agl_ft=min_agl,
        t_low=t_low,
        lat_low=float(seg["latitude"].iloc[ilow_rel]),
        lon_low=float(seg["longitude"].iloc[ilow_rel]),
        touchdown=touchdown,
        level_low_duration_s=level_dur,
        climb_start=climb_start,
        init_agl_ft=init_agl,
        init_dist_thr_nm=init_dist,
        peak_climb_fpm=peak_climb,
        alt_regain_ft=alt_regain,
    )
