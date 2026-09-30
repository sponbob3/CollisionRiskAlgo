"""
Cross-validation of this pipeline against the traffic library's built-in
go-around detector (Flight.go_around, the method of Olive et al.), run on a
sample of days. Agreement between two independent implementations is part of
the robustness evidence.

Usage:
    python -m goaround_pipeline.validate_traffic 20250108 20250115 ...
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

from . import config
from .loading import load_day, split_into_legs
from .pipeline import process_day

warnings.filterwarnings("ignore")


def traffic_go_arounds(path: Path) -> list[dict]:
    """Run traffic's Flight.go_around over each flight leg of one day."""
    from traffic.core import Flight

    df = load_day(path)
    events = []
    for leg in split_into_legs(df):
        f = Flight(
            leg.rename(columns={"agl": "agl_"})  # keep traffic's own names
        )
        try:
            gas = f.go_around(config.AIRPORT_ICAO)
            if gas is None:
                continue
            for g in gas:
                events.append({
                    "leg_id": leg.attrs["leg_id"],
                    "start": g.start,
                    "stop": g.stop,
                })
        except Exception as e:
            events.append({"leg_id": leg.attrs["leg_id"], "error": str(e)})
    return events


def compare_day(path: Path) -> dict:
    ours = [
        r for r in process_day(path)
        if r.outcome in ("go_around", "ga_ambiguous")
    ]
    theirs = [e for e in traffic_go_arounds(path) if "error" not in e]

    matched_ours, matched_theirs = set(), set()
    for i, r in enumerate(ours):
        for j, e in enumerate(theirs):
            if j in matched_theirs or e["leg_id"] != r.approach.leg_id:
                continue
            # a match = our low point falls inside (or within 90 s of)
            # traffic's go-around interval
            pad = pd.Timedelta(seconds=90)
            if e["start"] - pad <= r.t_low <= e["stop"] + pad:
                matched_ours.add(i)
                matched_theirs.add(j)
                break
    return {
        "day": path.stem,
        "ours": len(ours),
        "traffic": len(theirs),
        "both": len(matched_ours),
        "only_ours": [
            f"{r.approach.leg_id}@{r.t_low:%H:%M:%S}({r.outcome},"
            f"{r.min_agl_ft:.0f}ft)"
            for i, r in enumerate(ours) if i not in matched_ours
        ],
        "only_traffic": [
            f"{e['leg_id']}@{e['start']:%H:%M:%S}"
            for j, e in enumerate(theirs) if j not in matched_theirs
        ],
    }


def main(days: list[str]) -> None:
    tot_ours = tot_theirs = tot_both = 0
    for day in days:
        matches = sorted(config.DATA_DIR.glob(f"*{day}*.parquet"))
        if not matches:
            print(f"{day}: no file matching *{day}*.parquet in "
                  f"{config.DATA_DIR}")
            continue
        path = matches[0]
        c = compare_day(path)
        tot_ours += c["ours"]
        tot_theirs += c["traffic"]
        tot_both += c["both"]
        print(f"{day}: ours={c['ours']} traffic={c['traffic']} "
              f"both={c['both']}", flush=True)
        if c["only_ours"]:
            print(f"   only ours:    {c['only_ours']}")
        if c["only_traffic"]:
            print(f"   only traffic: {c['only_traffic']}")
    print(f"\nTOTAL ours={tot_ours} traffic={tot_theirs} both={tot_both}")


if __name__ == "__main__":
    main(sys.argv[1:])
