"""
The only bridge to the vendored go-around pipeline (FRAMEWORK.md
section 13). goaround_pipeline/ is not edited; everything this package
needs from it is imported and re-exported here, and run_goaround()
reproduces run_analysis.py from that repository step by step so that
output/<ICAO>/goaround/run_NN/ is identical to a native run.

Three hooks are provided here instead of the vendored functions (the
go-around stage gets them for the duration of run_goaround(), without
editing the vendored files); the first two are used by BOTH stages:

- data_files(): finds daily files in the dataset folder AND its
  subfolders (e.g. one subfolder per month).
- load_day(): the vendored loader plus the position-freshness filter.
  OpenSky state vectors repeat an aircraft's last known position for up
  to 300 s after its last position report, so a large share of rows can
  be frozen copies of an old position. Only rows that carry a NEW
  position report are kept, and they are re-timed to the time of that
  report. Applied when the file has a position-time column
  (POSITION_TIME_COLUMNS); files without one load exactly as before.
- classify_approach(): the vendored classifier plus the hidden-low-point
  rule. A climb-away (no touchdown) classified low_approach or
  ga_ambiguous only because its low point fell inside a data gap (ADS-B
  coverage ending above the runway) is a go-around: when the level time
  that was actually observed - the measured plateau minus the hidden gap
  time - is within the go-around cutoff. Reclassified approaches are
  listed with `reclassified_from` and `low_point_hidden_s` in
  all_approaches.csv and go_around_events.csv.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from goaround_pipeline import config  # the go-around config module
from goaround_pipeline import pipeline, profiles
from goaround_pipeline.approaches import (NM_PER_DEG_LAT, FT_PER_NM,
                                          _runway_frame, _track_delta)
from goaround_pipeline.loading import RAW_COLUMNS, split_into_legs
from goaround_pipeline import loading as _ga_loading
from goaround_pipeline import classify as _ga_classify
from goaround_pipeline.pipeline import results_to_frame as \
    _ga_pipeline_results_to_frame

from . import config as px_config

__all__ = [
    "config", "load_profile", "create_profile", "load_day",
    "split_into_legs", "data_files", "runway_frame", "track_delta",
    "NM_PER_DEG_LAT", "FT_PER_NM", "RAW_COLUMNS", "run_goaround",
    "next_run_number", "read_events", "read_approaches", "draw_runways",
    "has_data_files", "freshness_filter",
]

# ------------------------------------------------------- input files --

def data_files(data_dir: Path) -> list[Path]:
    """Daily files in a dataset folder and any of its subfolders (e.g.
    datasets/KMCO_2025Q1/2025-01/...). When the same day exists as both
    parquet and csv, the parquet wins (the vendored rule). Hidden files
    (names starting with '.', such as macOS '._' copies) are ignored. Two
    different files for the same day name are an error rather than a
    silent pick."""
    data_dir = Path(data_dir)
    by_stem: dict[str, Path] = {}
    for f in sorted(data_dir.rglob("*.csv")) + sorted(
            data_dir.rglob("*.parquet")):
        if any(part.startswith(".") for part in f.relative_to(data_dir).parts):
            continue
        prev = by_stem.get(f.stem)
        if (prev is not None and prev.suffix == f.suffix
                and prev != f):
            raise ValueError(
                f"two daily files named {f.stem}{f.suffix} in {data_dir}: "
                f"{prev.relative_to(data_dir)} and "
                f"{f.relative_to(data_dir)}; keep only one")
        by_stem[f.stem] = f   # parquet sorted second -> overrides csv
    return [by_stem[k] for k in sorted(by_stem)]


def has_data_files(data_dir: Path) -> bool:
    return bool(data_files(data_dir)) if Path(data_dir).is_dir() else False


# ------------------------------------------------- position freshness --

def _read_position_time(path: Path) -> tuple[pd.DataFrame | None, str]:
    """(icao24, timestamp, position time in epoch seconds) of a raw file,
    or (None, "") when it has no position-time column."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        cols = list(pd.read_csv(path, nrows=0).columns)
    else:
        import pyarrow.parquet as pq
        cols = pq.read_schema(path).names
    name = next((c for c in px_config.POSITION_TIME_COLUMNS if c in cols), "")
    if not name:
        return None, ""
    use = ["icao24", "timestamp", name]
    side = (pd.read_csv(path, usecols=use) if path.suffix.lower() == ".csv"
            else pd.read_parquet(path, columns=use))
    side["timestamp"] = pd.to_datetime(
        side["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    pt = side[name]
    if pd.api.types.is_datetime64_any_dtype(pt):
        pt = pd.to_datetime(pt, utc=True).astype("int64") / 1e9
    side["position_time"] = pd.to_numeric(pt, errors="coerce").astype(float)
    side = side.drop(columns=[name]).drop_duplicates(["icao24", "timestamp"])
    return side, name


def freshness_filter(df: pd.DataFrame, side: pd.DataFrame) -> tuple[
        pd.DataFrame, dict]:
    """Keep only rows that carry a new position report, re-timed to the
    report time. `df` is the vendored loader's output (sorted by icao24,
    timestamp); `side` holds each row's position time."""
    n_in = len(df)
    df = df.merge(side, on=["icao24", "timestamp"], how="left")
    ts = df["timestamp"].astype("int64").to_numpy() / 1e9
    pt = df["position_time"].to_numpy()
    prev = df.groupby("icao24", sort=False)["position_time"].shift()
    new_report = (df["position_time"] != prev).to_numpy()
    age = ts - pt
    known = np.isfinite(pt)
    keep = ~known | (new_report & (age <= px_config.MAX_POSITION_AGE_S)
                     & (age >= -px_config.MAX_POSITION_AGE_S))
    out = df[keep].copy()
    k = out["position_time"].notna()
    out.loc[k, "timestamp"] = pd.to_datetime(
        np.round(out.loc[k, "position_time"].to_numpy()), unit="s",
        utc=True).astype("datetime64[ns, UTC]")
    out = (out.drop(columns=["position_time"])
           .sort_values(["icao24", "timestamp"], kind="stable")
           .drop_duplicates(["icao24", "timestamp"], keep="first")
           .reset_index(drop=True))
    stats = {"rows_in": n_in, "rows_kept": int(len(out)),
             "stale_share": 1.0 - len(out) / n_in if n_in else 0.0}
    return out, stats


def load_day(path, return_stats: bool = False):
    """The vendored loader plus the position-freshness filter (see the
    module docstring). With return_stats, also returns a dict with the
    rows in / kept and the stale share."""
    df = _ga_loading.load_day(path)
    stats = {"rows_in": len(df), "rows_kept": len(df), "stale_share": 0.0,
             "position_time_column": ""}
    if px_config.POSITION_FRESHNESS_FILTER and len(df):
        side, name = _read_position_time(Path(path))
        if side is not None:
            df, st = freshness_filter(df, side)
            stats.update(st)
            stats["position_time_column"] = name
    return (df, stats) if return_stats else df


# ------------------------------------------------- hidden low point ----

CLIMB_AWAY_RECLASSIFIABLE = ("low_approach", "ga_ambiguous")


def hidden_low_point(r) -> tuple[float, float]:
    """(hidden s, observed level s) of a climb-away's plateau: the
    plateau runs from the last descent into the low band until the climb
    start (level_low_duration_s); report gaps longer than
    GA_HIDDEN_GAP_MIN_S inside it are hidden time."""
    if r.climb_start is None or not np.isfinite(r.level_low_duration_s):
        return 0.0, float(r.level_low_duration_s)
    t = (r.approach.data["timestamp"].astype("int64").to_numpy() / 1e9)
    cs = pd.Timestamp(r.climb_start).value / 1e9
    entry = cs - float(r.level_low_duration_s)
    lo = max(int(np.searchsorted(t, entry, side="left")) - 1, 0)
    hi = int(np.searchsorted(t, cs, side="right"))
    span = np.clip(t[lo:hi], entry, cs)
    d = np.diff(span)
    hidden = float(np.sum(d[d > px_config.GA_HIDDEN_GAP_MIN_S]))
    return hidden, float(r.level_low_duration_s) - hidden


def classify_approach(app):
    """The vendored classifier plus the hidden-low-point rule (see the
    module docstring)."""
    r = _ga_classify.classify_approach(app)
    r.reclassified_from = ""
    r.low_point_hidden_s = 0.0
    if r.outcome in CLIMB_AWAY_RECLASSIFIABLE:
        hidden, observed = hidden_low_point(r)
        r.low_point_hidden_s = round(hidden, 1)
        if (px_config.GA_HIDDEN_LOW_POINT_AS_GO_AROUND
                and hidden > 0 and observed <= config.LEVEL_GA_MAX_S):
            r.reclassified_from = r.outcome
            r.outcome = "go_around"
    return r


def results_to_frame(results) -> pd.DataFrame:
    """The vendored approach table plus the reclassification columns."""
    df = _ga_pipeline_results_to_frame(results)
    if len(df):
        df["reclassified_from"] = [getattr(r, "reclassified_from", "")
                                   for r in results]
        df["low_point_hidden_s"] = [getattr(r, "low_point_hidden_s", 0.0)
                                    for r in results]
    return df


class _InputHandling:
    """Context manager: the vendored pipeline uses this module's
    data_files(), load_day(), classify_approach() and results_to_frame()
    for the duration of a go-around run."""

    def __enter__(self):
        self._saved = (pipeline.data_files, pipeline.load_day,
                       pipeline.classify_approach, pipeline.results_to_frame)
        pipeline.data_files = data_files
        pipeline.load_day = load_day
        pipeline.classify_approach = classify_approach
        pipeline.results_to_frame = results_to_frame
        return self

    def __exit__(self, *exc):
        (pipeline.data_files, pipeline.load_day,
         pipeline.classify_approach, pipeline.results_to_frame) = self._saved
        return False


# runway-frame geometry (along-track / cross-track in NM relative to a
# threshold, positive along-track = on final) reused for flight phases
# and geometry classes
runway_frame = _runway_frame
track_delta = _track_delta


_CONFIG_DEFAULTS: dict = {}


def config_defaults(name: str):
    """A go-around parameter's value as shipped (config.py plus the
    air_carrier / training_ga presets are applied later by profiles), read
    from the module source so that a loaded profile does not change it."""
    if not _CONFIG_DEFAULTS:
        import ast
        src = Path(config.__file__).read_text()
        for node in ast.parse(src).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                try:
                    _CONFIG_DEFAULTS[node.targets[0].id] = ast.literal_eval(
                        node.value)
                except ValueError:
                    pass
    return _CONFIG_DEFAULTS[name]


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
    with _InputHandling():
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
