"""
Sensitivity analysis (FRAMEWORK.md section 12): one-at-a-time variations
of window length, radius, ceiling, lookahead, tier thresholds, event set,
geometry classes and baseline exclusion, each re-running the analysis
(baseline refit without validation, fewer bootstrap replicates) and
reporting IRR and SIR with intervals for the primary endpoint in both
scopes. The pairwise computation is never repeated: every variation is a
filter on the per-day caches.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from . import events as E
from . import windows as W
from .pipeline import analyse

COLUMNS = ["parameter", "value", "is_base", "scope", "endpoint",
           "event_set", "n_events", "irr", "irr_lo", "irr_hi", "sir_pre",
           "sir_pre_lo", "sir_pre_hi", "sir_post", "sir_post_lo",
           "sir_post_hi", "baseline_model", "elapsed_s"]


def _rows(param, value, res, endpoint="T1_any", event_set="all",
          is_base=False, elapsed=np.nan) -> list[dict]:
    rows = []
    r = res["results"]
    for scope in ("airspace", "ga_involved"):
        sel = r[(r["scope"] == scope) & (r["endpoint"] == endpoint)
                & (r["event_set"] == event_set)]
        if sel.empty:
            continue
        x = sel.iloc[0]
        rows.append({"parameter": param, "value": value, "is_base": is_base,
                     "scope": scope, "endpoint": endpoint,
                     "event_set": event_set, "n_events": x["n_events"],
                     "irr": x["irr"], "irr_lo": x["irr_lo"],
                     "irr_hi": x["irr_hi"], "sir_pre": x["sir_pre"],
                     "sir_pre_lo": x["sir_pre_lo"],
                     "sir_pre_hi": x["sir_pre_hi"],
                     "sir_post": x["sir_post"],
                     "sir_post_lo": x["sir_post_lo"],
                     "sir_post_hi": x["sir_post_hi"],
                     "baseline_model": x["baseline_model"],
                     "elapsed_s": elapsed})
    return rows


def sweep(days: dict, app: pd.DataFrame, error_model, ev: pd.DataFrame,
          ctl: pd.DataFrame, window_min: float, spec: W.Spec,
          effective_ceiling: float, charted_ceiling: float,
          outcomes: tuple, base_res: dict) -> pd.DataFrame:
    reps = config.SENSITIVITY_BOOTSTRAP_REPS
    excl = config.BASELINE_EXCLUSION_MIN
    rows: list[dict] = []
    rows += _rows("base", "base", base_res, is_base=True)
    # variations already contained in the base results
    rows += _rows("event_set", "in_control_pre_only", base_res,
                  event_set="in_control_pre")
    rows += _rows("geometry", "excluding_procedural_T1", base_res,
                  endpoint="T1_any_nonprocedural")
    for typ in config.EVENT_TYPE_BREAKDOWN:
        rows += _rows("event_set", f"{typ}_only", base_res,
                      event_set=f"{typ}_only")

    variations = []
    for w in config.WINDOW_SENSITIVITY_MIN:
        variations.append(("window_min", w, dict(window_min=w)))
    variations.append(("radius_nm", config.VOLUME_RADIUS_SENSITIVITY_NM,
                       dict(spec=W.Spec(config.VOLUME_RADIUS_SENSITIVITY_NM,
                                        spec.ceiling_ft, spec.lookahead_s,
                                        spec.tiers, spec.merge_gap_s))))
    ceilings = sorted(set(config.CEILING_SENSITIVITY_FT_AGL)
                      - {effective_ceiling})
    for c in ceilings:
        variations.append(("ceiling_ft_agl", c,
                           dict(spec=W.Spec(spec.radius_nm, c, spec.lookahead_s,
                                            spec.tiers, spec.merge_gap_s))))
    for la in config.T_LOOKAHEAD_SENSITIVITY_S:
        variations.append(("lookahead_s", la,
                           dict(spec=W.Spec(spec.radius_nm, spec.ceiling_ft,
                                            la, spec.tiers, spec.merge_gap_s))))
    for tier in config.TIER_ORDER:
        for which in ("H", "V"):
            for f in config.TIER_SENSITIVITY_FACTORS:
                tiers = W.scale_tiers(spec.tiers, tier, which, f)
                variations.append((f"{tier}_{which}", f"x{f:g}",
                                   dict(spec=W.Spec(spec.radius_nm,
                                                    spec.ceiling_ft,
                                                    spec.lookahead_s, tiers,
                                                    spec.merge_gap_s))))
    variations.append(("event_set", "include_ambiguous",
                       dict(outcomes=config.EVENT_OUTCOMES_AMBIGUOUS)))
    variations.append(("event_set", "no_clusters", dict(no_clusters=True)))
    for x in config.BASELINE_EXCLUSION_SENSITIVITY_MIN:
        variations.append(("baseline_exclusion_min", x, dict(exclusion=x)))

    print(f"sensitivity sweep: {len(variations)} variations")
    for i, (param, value, kw) in enumerate(variations):
        t = time.time()
        w = kw.get("window_min", window_min)
        sp = kw.get("spec", spec)
        oc = kw.get("outcomes", outcomes)
        ex = kw.get("exclusion", excl)
        e = E.event_table(app, oc, w)
        if kw.get("no_clusters"):
            e = e[e["cluster_size"] == 1].reset_index(drop=True)
        c = E.control_table(app, w, ex, config.CONTROL_MAX_EVENTS,
                            config.RANDOM_SEED)
        try:
            res = analyse(days, app, sp, error_model, e, c, w, ex, None,
                          rebuild=True, validate=False, reps=reps, quiet=True)
            rows += _rows(param, value, res, elapsed=round(time.time() - t, 1))
            r = [x for x in rows[-2:] if x["scope"] == "airspace"]
            msg = (f"IRR {r[0]['irr']:.2f} [{r[0]['irr_lo']:.2f}, "
                   f"{r[0]['irr_hi']:.2f}]  SIR_post {r[0]['sir_post']:.2f}"
                   if r else "no result")
        except Exception as err:   # report, never absorb
            rows.append({"parameter": param, "value": value, "is_base": False,
                         "scope": "airspace", "endpoint": "T1_any",
                         "event_set": "all", "n_events": np.nan,
                         "baseline_model": f"FAILED: {err}"})
            msg = f"FAILED: {err}"
        print(f"  [{i + 1}/{len(variations)}] {param}={value}: {msg}  "
              f"({time.time() - t:.0f} s)", flush=True)
    return pd.DataFrame(rows, columns=COLUMNS)
