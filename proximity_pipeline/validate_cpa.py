"""
Cross-check of this pipeline's closest-point-of-approach distances against
the traffic library's independent implementation
(Traffic.closest_point_of_approach), on a sample of days. Agreement
between two independent implementations is part of the robustness
evidence (FRAMEWORK.md sections 4.6 and 18, test 9); analogous to the
go-around pipeline's validate_traffic.py.

Usage:
    python -m proximity_pipeline.validate_cpa KBNA_2025 20250108 20250115

For every pair of flight legs that came within the T1 zone (horizontal
threshold while within the vertical threshold) on a day, the minimum
horizontal distance at the raw sample times is compared: ours (from the
per-day cache, sample-time distances) against traffic's. Pairs found by
only one side are listed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import airspace, config, loading, pipeline
from . import goaround_adapter as ga

warnings.filterwarnings("ignore")

TOLERANCE_NM = 0.05      # agreement tolerance on the minimum distance
MATCH_H_NM, MATCH_V_FT = config.TIERS["T1"]


def traffic_cpa(path: Path) -> pd.DataFrame:
    """Per-pair minimum lateral distance (NM) from traffic, over pairs
    that come within the T1 horizontal threshold."""
    from traffic.core import Traffic

    raw = ga.load_day(path)
    legs, _ = loading.clean_day(raw)
    radius = airspace.computation_radius_nm()
    ceiling = airspace.computation_ceiling_ft_agl()
    frames = []
    for leg in legs:
        d = leg[["timestamp", "icao24", "callsign", "latitude", "longitude",
                 "altitude", "geoaltitude", "groundspeed", "track",
                 "vertical_rate", "onground", "h"]].copy()
        # same population as the pipeline: airborne samples inside the
        # computation volume
        x, y = loading.project(d["latitude"].to_numpy(),
                               d["longitude"].to_numpy())
        keep = ((~d["onground"]) & (d["groundspeed"] >= config.AIRBORNE_MIN_GS_KT)
                & (np.hypot(x, y) <= radius) & (d["h"] <= ceiling))
        d = d[keep.to_numpy()]
        if len(d) < 2:
            continue
        d["flight_id"] = leg.attrs["leg_id"]
        frames.append(d.drop(columns=["h"]))
    if not frames:
        return pd.DataFrame(columns=["leg_a", "leg_b", "d_min_traffic_nm"])
    t = Traffic(pd.concat(frames, ignore_index=True))
    cpa = t.closest_point_of_approach(
        lateral_separation=MATCH_H_NM * 1852.0,
        vertical_separation=MATCH_V_FT,
        projection=loading.projection(), round_t="d", max_workers=1)
    if cpa is None or cpa.data is None or cpa.data.empty:
        return pd.DataFrame(columns=["leg_a", "leg_b", "d_min_traffic_nm"])
    # traffic returns every timestamp of a candidate pair; keep the
    # samples inside the vertical threshold before taking the minimum
    d = cpa.data
    d = d[d["vertical"].abs() < MATCH_V_FT]
    a = np.minimum(d["flight_id_x"], d["flight_id_y"])
    b = np.maximum(d["flight_id_x"], d["flight_id_y"])
    out = (pd.DataFrame({"leg_a": a, "leg_b": b, "d": d["lateral"]})
           .groupby(["leg_a", "leg_b"])["d"].min().reset_index()
           .rename(columns={"d": "d_min_traffic_nm"}))
    return out[out["d_min_traffic_nm"] < MATCH_H_NM]


def our_cpa(path: Path, cache_dir: Path) -> pd.DataFrame:
    """Per-pair minimum sample-time horizontal distance (NM) from the
    per-day cache (which holds every pair-second within the loosest T1
    zone), restricted to non-interpolated samples so both sides look at
    the same fixes."""
    pipeline.build_day_caches([path], cache_dir, workers=1, quiet=True)
    dc = pipeline.DayCache(cache_dir, path.stem)
    p = dc.pairs
    if p.empty:
        return pd.DataFrame(columns=["leg_a", "leg_b", "d_min_ours_nm"])
    p = p[~p["ia"] & ~p["ib"] & (p["dz"].abs() < MATCH_V_FT)]
    legs = dc.legs.set_index("leg")["leg_id"]
    d = np.hypot(p["rx"].to_numpy(float), p["ry"].to_numpy(float))
    out = (pd.DataFrame({"leg_a": legs.loc[p["a"]].to_numpy(),
                         "leg_b": legs.loc[p["b"]].to_numpy(), "d": d})
           .groupby(["leg_a", "leg_b"])["d"].min().reset_index()
           .rename(columns={"d": "d_min_ours_nm"}))
    return out[out["d_min_ours_nm"] < MATCH_H_NM]


def compare_day(path: Path, cache_dir: Path) -> dict:
    ours = our_cpa(path, cache_dir)
    theirs = traffic_cpa(path)
    m = ours.merge(theirs, on=["leg_a", "leg_b"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both["d_min_ours_nm"] - both["d_min_traffic_nm"]).abs()
    return {
        "day": path.stem, "ours": int(len(ours)), "traffic": int(len(theirs)),
        "both": int(len(both)),
        "agree": int((diff <= TOLERANCE_NM).sum()),
        "max_abs_diff_nm": float(diff.max()) if len(diff) else 0.0,
        "median_abs_diff_nm": float(diff.median()) if len(diff) else 0.0,
        "only_ours": m[m["_merge"] == "left_only"][["leg_a", "leg_b",
                                                    "d_min_ours_nm"]],
        "only_traffic": m[m["_merge"] == "right_only"][["leg_a", "leg_b",
                                                        "d_min_traffic_nm"]],
    }


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        sys.exit("usage: python -m proximity_pipeline.validate_cpa "
                 "<ICAO_label> <day> [<day> ...]")
    name, days = argv[0], argv[1:]
    icao, label = name.split("_", 1)
    airspace.load_airspace(icao.upper())
    dataset = config.DATASETS_DIR / name
    cache_dir = pipeline.cache_root(icao.upper(), label)
    tot = {"ours": 0, "traffic": 0, "both": 0, "agree": 0}
    for day in days:
        matches = [f for f in ga.data_files(dataset) if day in f.stem]
        if not matches:
            print(f"{day}: no daily file matching *{day}* in {dataset}")
            continue
        c = compare_day(matches[0], cache_dir)
        for k in tot:
            tot[k] += c[k]
        print(f"{c['day']}: ours={c['ours']} traffic={c['traffic']} "
              f"both={c['both']} agree(<={TOLERANCE_NM} NM)={c['agree']} "
              f"max|diff|={c['max_abs_diff_nm']:.3f} NM "
              f"median|diff|={c['median_abs_diff_nm']:.3f} NM", flush=True)
        if len(c["only_ours"]):
            print(f"   only ours ({len(c['only_ours'])}):\n"
                  f"{c['only_ours'].to_string(index=False)}")
        if len(c["only_traffic"]):
            print(f"   only traffic ({len(c['only_traffic'])}):\n"
                  f"{c['only_traffic'].to_string(index=False)}")
    print(f"\nTOTAL ours={tot['ours']} traffic={tot['traffic']} "
          f"both={tot['both']} agree={tot['agree']}")


if __name__ == "__main__":
    main(sys.argv[1:])
