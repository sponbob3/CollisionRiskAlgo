"""
Interactive start (python run_proximity.py with no arguments, or the
double-clickable "Run Proximity Analysis.command" on a Mac): asks which
airport/dataset to analyse, shows what was found (data, airport profile,
pairwise cache, go-around runs, stored baseline), asks what to produce,
prints the equivalent one-line command, and starts the run. The command
line options keep working unchanged; this only builds them.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

LINE = "-" * 64


def _ask(question: str, default: str | None = None) -> str:
    hint = f" [{default}]" if default not in (None, "") else ""
    while True:
        try:
            ans = input(f"{question}{hint}: ").strip()
        except EOFError:
            ans = ""
        if ans:
            return ans
        if default is not None:
            return default


def _yes(question: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        try:
            ans = input(f"{question} [{d}]: ").strip().lower()
        except EOFError:
            ans = ""
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  please answer y or n")


def _choose(question: str, options: list[str], default: int = 1) -> int:
    """Numbered menu; returns the 0-based index."""
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        ans = _ask(question, str(default))
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return int(ans) - 1
        print(f"  please type a number from 1 to {len(options)}")


def _size(path: Path) -> str:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    gb = total / 1e9
    return f"{gb:.1f} GB" if gb >= 1 else f"{total / 1e6:.0f} MB"


def _dates(files: list[Path]) -> tuple[str, str]:
    ds = []
    for f in files:
        m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", f.stem)
        if m:
            ds.append(f"{m[1]}-{m[2]}-{m[3]}")
    return (min(ds), max(ds)) if ds else ("?", "?")


def _goaround_runs(ga_base: Path, dataset_name: str) -> list[dict]:
    """Go-around runs of this airport with their dataset and counts."""
    import pandas as pd
    runs = []
    for d in sorted(ga_base.glob("run_*")):
        if not (d / "all_approaches.csv").exists():
            continue
        name, n_files = "?", "?"
        cfg = d / "run_config.txt"
        if cfg.exists():
            m = re.search(r"^dataset:\s*(\S+)\s*\((\d+) daily",
                          cfg.read_text(errors="ignore"), re.M)
            if m:
                name, n_files = m[1], m[2]
        try:
            out = pd.read_csv(d / "all_approaches.csv",
                              usecols=["outcome"])["outcome"].value_counts()
        except Exception:
            out = pd.Series(dtype=int)
        runs.append({"dir": d, "nn": int(d.name.split("_")[1]),
                     "dataset": name, "days": n_files,
                     "go_arounds": int(out.get("go_around", 0)),
                     "touch_and_goes": int(out.get("touch_and_go", 0)),
                     "same_dataset": name == dataset_name})
    return runs


def run() -> None:
    try:
        _run()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if sys.stdin.isatty():
            try:
                input("\npress Enter to close ")
            except (EOFError, KeyboardInterrupt):
                pass


def _run() -> None:
    import run_proximity as rp
    print(LINE)
    print(" Proximity risk after go-arounds")
    print(LINE)

    # 1. dataset / airport
    ds = rp.list_datasets()
    if not ds:
        print(f"no datasets found in {rp.DATASETS_DIR}.\nCreate "
              "datasets/<ICAO>_<label>/ (e.g. datasets/kmco_2025q1/) and put "
              "the daily .parquet or .csv files in it (subfolders are fine).")
        return
    print("\nWhich airport / dataset?")
    labels = []
    for d in ds:
        n = sum(1 for p in d.rglob("*") if p.suffix in (".parquet", ".csv")
                and not p.name.startswith("."))
        labels.append(f"{d.name}  ({d.name.split('_')[0].upper()}, {n} files)")
    dataset = ds[_choose("number", labels)]
    if "_" not in dataset.name:
        print(f"\nthe folder name must be <ICAO>_<label> (e.g. "
              f"{dataset.name}_2025q1); rename datasets/{dataset.name} and "
              "start again.")
        return
    icao, label = dataset.name.split("_", 1)
    icao = icao.upper()

    from proximity_pipeline import config
    from proximity_pipeline import goaround_adapter as ga
    files = ga.data_files(dataset)
    first, last = _dates(files)
    subs = sorted({f.parent.name for f in files if f.parent != dataset})
    print(f"\ndata: {len(files)} daily files, {first} to {last}"
          + (f", in {len(subs)} subfolders" if subs else "")
          + f", {_size(dataset)}")

    # 2. airport profile
    from proximity_pipeline import airspace, profile_setup
    prof_path = profile_setup.profile_path(icao)
    if prof_path.exists():
        print(f"airport profile: {prof_path.relative_to(config.ROOT)} "
              "(found)")
    else:
        print("airport profile: not found - creating it now from the "
              "OurAirports database (ourairports.com) and the data")
    profile_setup.ensure_profile(icao, dataset)
    prof = airspace.load_airspace(icao)
    print(f"  {prof.get('name', icao)}, timezone {ga.config.LOCAL_TZ}, "
          f"{len(ga.config.RUNWAYS)} runway ends, elevation "
          f"{ga.config.FIELD_ELEVATION_FT:,.0f} ft")

    # 3. pairwise cache
    from proximity_pipeline import pipeline
    cache_dir = pipeline.cache_root(icao, label)
    n_cached = (sum(pipeline.cache_is_current(cache_dir, f) for f in files)
                if cache_dir.exists() else 0)
    rebuild_cache = False
    if n_cached:
        print(f"\npairwise cache: {n_cached} of {len(files)} days already "
              f"computed ({_size(cache_dir)}, "
              f"{cache_dir.relative_to(config.ROOT)})")
        if not _yes("use the existing cache (recommended; only missing days "
                    "are computed)", True):
            print("  rebuilding recomputes every day from the raw data "
                  "(this can take hours)")
            rebuild_cache = _ask("  type 'rebuild' to delete the cache and "
                                 "recompute, or press Enter to keep it",
                                 "") == "rebuild"
    else:
        print(f"\npairwise cache: none yet - all {len(files)} days will be "
              "computed (the slowest step; later runs reuse it)")

    # 4. go-around detection
    ga_base = config.OUTPUT_DIR / icao / "goaround"
    runs = _goaround_runs(ga_base, dataset.name)
    same = [r for r in runs if r["same_dataset"]]
    goaround_run = None
    print("\ngo-around detection:")
    if same:
        options = [f"reuse run_{r['nn']:02d}: {r['go_arounds']} go-arounds, "
                   f"{r['touch_and_goes']} touch-and-goes, {r['days']} days"
                   for r in same[::-1]]
        options.append("run the detection again (new run folder)")
        k = _choose("number", options)
        if k < len(same):
            goaround_run = same[::-1][k]["nn"]
    else:
        if runs:
            print(f"  {len(runs)} earlier run(s) found, none for "
                  f"{dataset.name}")
        print("  the detection runs first (a few minutes per month of data)")

    # 5. baseline
    bdir = config.OUTPUT_DIR / icao / "proximity_risk" / "baseline" / label
    rebuild_baseline = False
    if (bdir / "baseline.json").exists():
        import json
        try:
            info = json.loads((bdir / "baseline.json").read_text())
            verdict = info.get("verdict", "?")
        except Exception:
            verdict = "?"
        print(f"\nstored baseline found ({verdict}); it is reused when the "
              "data and settings match, and rebuilt automatically when not")
        rebuild_baseline = _yes("force a rebuild anyway", False)
    else:
        print("\nbaseline: built in this run and stored for later runs")

    # 6. what to produce
    print("\nwhat should the run produce?")
    plots = _yes("figures, summary.pdf and the baseline report", True)
    event_plots = plots and _yes("one picture per go-around (slow: it "
                                 "re-reads each day with go-arounds)", True)
    sensitivity = _yes("sensitivity analysis (the whole analysis rerun with "
                       "each setting changed; can take a few hours)", False)
    probability = _yes("optional conflict-probability model", False)
    window = None
    workers = max(1, (os.cpu_count() or 2) - 1)
    include_ambiguous = False
    if _yes("change advanced settings (window length, workers, ambiguous "
            "go-arounds)", False):
        w = _ask("  window length in minutes", f"{config.WINDOW_MIN:g}")
        window = float(w) if w != f"{config.WINDOW_MIN:g}" else None
        workers = int(_ask("  parallel workers", str(workers)))
        include_ambiguous = _yes("  include ambiguous go-arounds as events",
                                 False)

    # 7. summary and start
    cmd = [f"python run_proximity.py {dataset.name}"]
    if goaround_run:
        cmd.append(f"--goaround-run {goaround_run}")
    if rebuild_baseline:
        cmd.append("--rebuild-baseline")
    if not plots:
        cmd.append("--no-plots")
    elif not event_plots:
        cmd.append("--no-event-plots")
    if not sensitivity:
        cmd.append("--no-sensitivity")
    if probability:
        cmd.append("--probability")
    if window:
        cmd.append(f"--window {window:g}")
    if include_ambiguous:
        cmd.append("--include-ambiguous")
    print(f"\n{LINE}\nready to run. Same as typing:\n  {' '.join(cmd)}\n{LINE}")
    if not _yes("start", True):
        print("not started.")
        return
    if rebuild_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"deleted {cache_dir.relative_to(config.ROOT)}")
    args = argparse.Namespace(
        dataset=dataset.name, goaround_run=goaround_run,
        rebuild_baseline=rebuild_baseline, baseline_only=False,
        baseline_from=None, window=window,
        include_ambiguous=include_ambiguous,
        no_sensitivity=not sensitivity, no_plots=not plots,
        no_event_plots=not event_plots, goaround_report=False,
        calibrate=False, probability=probability, limit=None,
        workers=workers)
    run_dir = pipeline.run_all(dataset, args, argv=["run_proximity.py"]
                               + " ".join(cmd[1:]).split())
    if sys.platform == "darwin" and plots and \
            _yes("\nopen the results folder in Finder", True):
        subprocess.run(["open", str(run_dir)], check=False)
