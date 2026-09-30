"""
The only bridge to the vendored go-around pipeline (FRAMEWORK.md
section 13). goaround_pipeline/ is not edited; everything this package
needs from it is imported and re-exported here, and run_goaround()
reproduces run_analysis.py from that repository step by step so that
output/<ICAO>/goaround/run_NN/ is identical to a native run.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd

from goaround_pipeline import config  # the go-around config module
from goaround_pipeline import pipeline, profiles
from goaround_pipeline.approaches import (NM_PER_DEG_LAT, FT_PER_NM,
                                          _runway_frame, _track_delta)
from goaround_pipeline.loading import RAW_COLUMNS, load_day, split_into_legs
from goaround_pipeline.pipeline import data_files

__all__ = [
    "config", "load_profile", "create_profile", "load_day",
    "split_into_legs", "data_files", "runway_frame", "track_delta",
    "NM_PER_DEG_LAT", "FT_PER_NM", "RAW_COLUMNS", "run_goaround",
    "next_run_number", "read_events", "read_approaches", "draw_runways",
]

# runway-frame geometry (along-track / cross-track in NM relative to a
# threshold, positive along-track = on final) reused for flight phases
# and geometry classes
runway_frame = _runway_frame
track_delta = _track_delta


def load_profile(icao: str) -> dict:
    return profiles.load_profile(icao)


def create_profile(icao: str) -> Path:
    return profiles.create_profile(icao)


def draw_runways(ax) -> None:
    from goaround_pipeline.viz import _draw_runways
    _draw_runways(ax)


# ------------------------------------------------------------------ runs --

def next_run_number(base: Path) -> int:
    """Next NN for run_NN folders under base (same rule as run_analysis.py,
    shared by the goaround/ and proximity_risk/ trees so both stages of one
    run carry the same number)."""
    base.mkdir(parents=True, exist_ok=True)
    nums = [
        int(p.name.split("_")[1])
        for p in base.glob("run_*") if p.name.split("_")[1].isdigit()
    ]
    return max(nums) + 1 if nums else 1


def _dump_run_config(run_dir: Path, dataset: Path, n_files: int,
                     argv: list[str]) -> None:
    """run_analysis.py's provenance record, unchanged in content."""
    lines = [
        f"run_time: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"dataset: {dataset.name}  ({n_files} daily files)",
        f"command: {' '.join(argv)}",
        "",
        "[resolved configuration]",
    ]
    for key in sorted(dir(config)):
        if key.isupper():
            lines.append(f"{key} = {getattr(config, key)!r}")
    (run_dir / "run_config.txt").write_text("\n".join(lines) + "\n")


def run_goaround(dataset: Path, run_dir: Path, plots: bool = True,
                 limit: int | None = None, calibrate: bool = False,
                 report: bool = False, argv: list[str] | None = None,
                 quiet: bool = False) -> pd.DataFrame:
    """Run go-around detection exactly as run_analysis.py does: profile
    already loaded, OUTPUT_DIR set, run_config.txt dumped, pipeline run,
    summaries written, optional calibration histogram and report."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = run_dir
    n_files = len(data_files(dataset))
    if not quiet:
        print(f"go-around stage: dataset {dataset.name}  airport "
              f"{config.AIRPORT_ICAO}  preset {config.PRESET}  terrain "
              f"{config.TERRAIN_MODE}  -> {run_dir}")
    _dump_run_config(run_dir, dataset, n_files, argv or sys.argv)
    df = pipeline.run(dataset, run_dir, plots=plots, limit=limit)
    summary = pipeline.write_summaries(df, run_dir)
    if not quiet:
        print()
        print(summary)
    if calibrate:
        _calibrate(df, run_dir)
    if report:
        from goaround_pipeline import report as ga_report
        from goaround_pipeline import report_pdf as ga_report_pdf
        ga_report.main(run_dir)
        ga_report_pdf.main(run_dir)
    return df


def _calibrate(df: pd.DataFrame, run_dir: Path) -> None:
    """run_analysis.py's --calibrate output (plateau-duration histogram),
    reproduced verbatim."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    classes = ["go_around", "ga_ambiguous", "low_approach", "touch_and_go"]
    sub = df[df["outcome"].isin(classes)]
    sub[["outcome", "level_low_duration_s"]].to_csv(
        run_dir / "calibration_plateau.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bins = np.arange(0, 90, 3)
    for outcome, color in zip(classes,
                              ["#eb6834", "#eda100", "#1baf7a", "#2a78d6"]):
        vals = sub[sub["outcome"] == outcome]["level_low_duration_s"]
        if len(vals):
            ax.hist(vals.clip(0, 87), bins=bins, histtype="step", lw=2,
                    color=color, label=f"{outcome} (n={len(vals)})",
                    density=True)
    ax.axvspan(config.LEVEL_GA_MAX_S, config.LEVEL_LOWAPP_MIN_S,
               color="0.9", zorder=0)
    ax.set_xlabel("plateau duration at profile low point (s)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Cutoff calibration - the shaded band should fall in an "
                 "empty gap", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "calibration_plateau.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- tables --

def read_approaches(run_dir: Path) -> pd.DataFrame:
    """all_approaches.csv of a go-around run with UTC timestamps parsed."""
    path = Path(run_dir) / "all_approaches.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the go-around "
                                "stage first or pass --goaround-run N")
    df = pd.read_csv(path, dtype={"callsign": str, "runway": str,
                                  "icao24": str})
    for col in ("approach_start_utc", "approach_end_utc", "t_low_utc",
                "climb_start_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
    df["callsign"] = df["callsign"].fillna("").astype(str)
    df["runway"] = df["runway"].fillna("").astype(str)
    return df


def read_events(run_dir: Path, outcomes: tuple[str, ...]) -> pd.DataFrame:
    """Go-around events (selected outcomes) from a go-around run."""
    df = read_approaches(run_dir)
    return df[df["outcome"].isin(outcomes)].reset_index(drop=True)
