"""
Threshold sensitivity analysis: perturb each key classifier parameter
one-at-a-time and re-run detection+classification on a systematic sample of
days (every Nth file). If the go-around count is stable across reasonable
threshold choices, the method is robust to its parameters.

Usage:
    python -m goaround_pipeline.sensitivity [every_nth]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import config
from .pipeline import process_day

# parameter -> list of values (first = baseline used by the pipeline)
GRID = {
    "LEVEL_GA_MAX_S": [20.0, 15.0, 25.0],
    "LEVEL_LOWAPP_MIN_S": [30.0, 25.0, 35.0],
    "GA_ALT_REGAIN_FT": [250.0, 200.0, 300.0],
    "GA_CLIMB_VRATE_FPM": [300.0, 200.0, 400.0],
    "TOUCHDOWN_PLATEAU_MIN_S": [20.0, 15.0, 25.0],
    "PLATEAU_BAND_FT": [60.0, 50.0, 75.0],
    "GA_MAX_LOW_AGL_FT": [1000.0, 800.0, 1200.0],
}

COUNTED = ("go_around", "ga_ambiguous")


def run_config(files: list[Path]) -> dict:
    counts = {"go_around": 0, "ga_ambiguous": 0, "low_approach": 0,
              "touch_and_go": 0, "approaches": 0}
    for f in files:
        for r in process_day(f):
            counts["approaches"] += 1
            if r.outcome in counts:
                counts[r.outcome] += 1
    return counts


def main(every_nth: int = 14, limit: int | None = None) -> None:
    files = sorted(config.DATA_DIR.glob("*.parquet"))[:limit][::every_nth]
    print(f"sampling {len(files)} days (every {every_nth}th"
          f"{'' if limit is None else f' of first {limit}'})", flush=True)

    baseline = {p: getattr(config, p) for p in GRID}
    rows = []
    for param, values in GRID.items():
        for v in values:
            if v != baseline[param] or param == next(iter(GRID)):
                pass  # always run baseline once via the first param loop
            setattr(config, param, v)
            c = run_config(files)
            setattr(config, param, baseline[param])
            rows.append({"param": param, "value": v,
                         "baseline": v == baseline[param], **c,
                         "ga_total": c["go_around"] + c["ga_ambiguous"]})
            print(f"{param}={v}: GA={c['go_around']} "
                  f"amb={c['ga_ambiguous']} low={c['low_approach']} "
                  f"tg={c['touch_and_go']}", flush=True)

    out = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out.to_csv(config.OUTPUT_DIR / "sensitivity_analysis.csv", index=False)
    print(f"\nWrote {config.OUTPUT_DIR / 'sensitivity_analysis.csv'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14,
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
