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


def _step(n: int, total: int, title: str) -> None:
    print(f"\n{LINE}\nSTEP {n} of {total}  -  {title}\n{LINE}")


def _choose(question: str, options: list[str], default: int = 1) -> int:
    """A question with numbered answers; returns the 0-based index.
    Pressing Enter alone picks `default` (always the recommended answer)."""
    print(f"\n{question}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        try:
            ans = input(f"> [{default}] ").strip()
        except EOFError:
            ans = ""
        if not ans:
            return default - 1
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return int(ans) - 1
        print(f"  type a number from 1 to {len(options)}")


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


def _run_matches(info: dict, n_files: int, first: str, last: str) -> bool:
    """A go-around run belongs to this data when it was made from the same
    number of days and its approaches fall inside the data's dates (the
    dataset folder's name is not used: it may have been renamed)."""
    if info["days"] is not None and info["days"] != n_files:
        return False
    if info["first"] and first != "?":
        lo = (pd_ts(first) - _day()).strftime("%Y-%m-%d")
        hi = (pd_ts(last) + _day() * 2).strftime("%Y-%m-%d")
        return lo <= info["first"] and info["last"] <= hi
    return info["days"] == n_files


def pd_ts(s: str):
    import pandas as pd
    return pd.Timestamp(s)


def _day():
    import pandas as pd
    return pd.Timedelta(days=1)


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


TOTAL_STEPS = 8


def _run() -> None:
    import run_proximity as rp
    print(LINE)
    print(" PROXIMITY RISK AFTER GO-AROUNDS")
    print(" Answer with the option number; Enter alone = option 1.")
    print(LINE)

    # ---- 1. dataset / airport
    _step(1, TOTAL_STEPS, "Dataset")
    ds = rp.list_datasets()
    if not ds:
        print(f"no datasets in {rp.DATASETS_DIR}: create datasets/<ICAO>_"
              "<label>/ (e.g. datasets/KMCO_2025Q1/) with the daily .parquet "
              "or .csv files (subfolders are fine)")
        return
    from proximity_pipeline import pipeline
    labels = []
    for d in ds:
        n = sum(1 for p in d.rglob("*") if p.suffix in (".parquet", ".csv")
                and not p.name.startswith("."))
        try:
            code = pipeline.dataset_identity(d)[0]
        except ValueError:
            code = "ICAO unknown"
        labels.append(f"{d.name}  ({code}, {n} daily files)")
    dataset = ds[_choose("Which dataset?", labels)]
    try:
        icao, label = pipeline.dataset_identity(dataset)
    except ValueError:
        code = input("Airport ICAO code for this dataset (e.g. KMCO): "
                     ).strip()
        icao, label = pipeline.dataset_identity(dataset, code)

    from proximity_pipeline import config
    from proximity_pipeline import goaround_adapter as ga
    files = ga.data_files(dataset)
    first, last = _dates(files)
    subs = sorted({f.parent.name for f in files if f.parent != dataset})
    print(f"{dataset.name}: {len(files)} daily files, {first} to {last}"
          + (f", {len(subs)} subfolders" if subs else "")
          + f", {_size(dataset)}")

    # ---- 2. airport profile (nothing to decide)
    _step(2, TOTAL_STEPS, "Airport profile")
    from proximity_pipeline import airspace, profile_setup
    prof_path = profile_setup.profile_path(icao)
    if prof_path.exists():
        print(f"profile found: {prof_path.relative_to(config.ROOT)}")
    else:
        print("no profile yet: creating it from OurAirports "
              "(ourairports.com) and the data")
    profile_setup.ensure_profile(icao, dataset)
    prof = airspace.load_airspace(icao)
    print(f"{prof.get('name', icao)} ({icao}): {len(ga.config.RUNWAYS)} runway "
          f"ends, elevation {ga.config.FIELD_ELEVATION_FT:,.0f} ft, "
          f"{ga.config.LOCAL_TZ}")

    # ---- 3. saved per-day computations (pairwise cache)
    _step(3, TOTAL_STEPS, "Pairwise cache")
    cache_dir = pipeline.cache_root(icao)
    pipeline.adopt_old_caches(files, cache_dir)
    n_cached = (sum(pipeline.cache_is_current(cache_dir, f) for f in files)
                if cache_dir.exists() else 0)
    rebuild_cache = False
    if n_cached:
        rest = len(files) - n_cached
        print(f"existing cache: {n_cached} of {len(files)} days "
              f"({_size(cache_dir)}, {cache_dir.relative_to(config.ROOT)})"
              + (f"; {rest} days still to compute" if rest else ""))
        k = _choose("Use the existing cache?", [
            "use existing cache (only missing days are computed)",
            f"delete the cache and recompute all {len(files)} days (hours)"])
        if k == 1:
            rebuild_cache = _choose("Delete the cache for this airport?", [
                "no, keep and use it",
                "yes, delete and recompute"]) == 1
    else:
        print(f"no cache yet: all {len(files)} days are computed in this run "
              "(slowest step; later runs reuse it)")

    # ---- 4. go-around detection
    _step(4, TOTAL_STEPS, "Go-around detection")
    ga_base = config.OUTPUT_DIR / icao / "goaround"
    runs = [pipeline.goaround_run_info(d)
            for d in sorted(ga_base.glob("run_*"), reverse=True)
            if (d / "all_approaches.csv").exists()]
    for r in runs:
        r["match"] = _run_matches(r, len(files), first, last)
    goaround_run = None
    if runs:
        runs.sort(key=lambda r: (not r["match"], -r["nn"]))
        n_match = sum(r["match"] for r in runs)
        print(f"existing go-around runs for {icao}: {len(runs)} "
              f"({n_match} from this dataset's days and dates)")
        options = []
        for r in runs:
            what = (f"{r['go_arounds']} go-arounds, {r['touch_and_goes']} "
                    f"touch-and-goes, {r['days'] or '?'} days"
                    + (f", {r['first']} to {r['last']}" if r["first"] else ""))
            options.append(f"reuse goaround/run_{r['nn']:02d} ({what})"
                           + ("" if r["match"] else
                              "  [made from DIFFERENT data]"))
        redo = "run go-around detection again (new run folder)"
        if runs[0]["match"]:
            options.append(redo)
        else:
            options.insert(0, redo)
        k = _choose("Which go-around run?", options)
        chosen = (k if runs[0]["match"] else k - 1)
        if 0 <= chosen < len(runs):
            goaround_run = runs[chosen]["nn"]
    else:
        print("no existing go-around runs: detection runs first (a few "
              "minutes per month of data)")

    # ---- 5. baseline
    _step(5, TOTAL_STEPS, "Baseline")
    import json
    broot = config.OUTPUT_DIR / icao / "proximity_risk" / "baseline"
    sig = pipeline.dataset_signature(files)
    same = []
    for bj in (sorted(broot.glob("*/baseline.json")) if broot.exists()
               else []):
        try:
            info = json.loads(bj.read_text())
        except Exception:
            continue
        if info.get("info", {}).get("dataset_signature") == sig:
            same.append((bj.parent, info.get("verdict", "")))
    rebuild_baseline = False
    if same:
        folder, verdict = same[0]
        print(f"existing baseline: baseline/{folder.name}, built from these "
              f"{len(files)} days ({verdict or 'verdict unknown'})")
        k = _choose("Use the existing baseline?", [
            "use existing baseline (rebuilt automatically if settings "
            "changed)",
            "rebuild the baseline (a few minutes)"])
        rebuild_baseline = k == 1
    else:
        print("no existing baseline for these days: built and stored in this "
              "run (a few minutes)")

    # ---- 6. outputs
    _step(6, TOTAL_STEPS, "Outputs")
    plots = _choose("Figures, summary.pdf and baseline report?", [
        "yes", "no (CSV tables only)"]) == 0
    event_plots = False
    if plots:
        event_plots = _choose("Per-event plots (one PNG per go-around; "
                              "~30 min for 3 months)?",
                              ["yes", "no"]) == 0
    sensitivity = _choose("Sensitivity analysis (full rerun per setting; "
                          "a few hours)?", ["no", "yes"]) == 1
    probability = _choose("Conflict-probability add-on (--probability)?",
                          ["no", "yes"]) == 1

    # ---- 7. settings
    _step(7, TOTAL_STEPS, "Settings")
    window = None
    workers = max(1, (os.cpu_count() or 2) - 1)
    include_ambiguous = False
    k = _choose(f"Settings: {config.WINDOW_MIN:g}-min windows, {workers} "
                "workers, ambiguous go-arounds excluded.",
                ["keep defaults", "change"])
    if k == 1:
        w = input(f"Window length in minutes (just Enter = "
                  f"{config.WINDOW_MIN:g}): ").strip()
        if w:
            window = float(w)
        n = input(f"Parallel workers (just Enter = {workers}): ").strip()
        if n.isdigit() and int(n) > 0:
            workers = int(n)
        include_ambiguous = _choose(
            "Include ambiguous go-arounds as events?", ["no", "yes"]) == 1

    # ---- 8. summary and start
    _step(8, TOTAL_STEPS, "Check and start")
    q = f'"{dataset.name}"' if " " in dataset.name else dataset.name
    cmd = [f"python run_proximity.py {q}"]
    if dataset.name.upper().split("_")[0][:4] != icao:
        cmd.append(f"--airport {icao}")
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
    yes_no = {True: "yes", False: "no"}
    print(f"Data:                 {dataset.name} ({len(files)} days, {icao})")
    print("Pairwise cache:       "
          + ("delete and recompute" if rebuild_cache else
             f"use existing ({n_cached} of {len(files)} days)"))
    print("Go-around run:        "
          + (f"reuse goaround/run_{goaround_run:02d}" if goaround_run
             else "new detection"))
    print("Baseline:             "
          + ("rebuild" if rebuild_baseline else
             "use existing" if same else "build new"))
    print(f"Figures and PDFs:     {yes_no[plots]}")
    print(f"Per-event plots:      {yes_no[event_plots]}")
    print(f"Sensitivity analysis: {yes_no[sensitivity]}")
    print(f"Probability add-on:   {yes_no[probability]}")
    print(f"Window:               {window or config.WINDOW_MIN:g} minutes")
    print(f"Command:              {' '.join(cmd)}")
    if _choose("Start?", ["start", "cancel"]) == 1:
        print("cancelled; nothing changed")
        return
    if rebuild_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"deleted {cache_dir.relative_to(config.ROOT)}")
    args = argparse.Namespace(
        dataset=dataset.name, airport=icao, goaround_run=goaround_run,
        rebuild_baseline=rebuild_baseline, baseline_only=False,
        baseline_from=None, window=window,
        include_ambiguous=include_ambiguous,
        no_sensitivity=not sensitivity, no_plots=not plots,
        no_event_plots=not event_plots, goaround_report=False,
        calibrate=False, probability=probability, limit=None,
        workers=workers)
    run_dir = pipeline.run_all(dataset, args, argv=["run_proximity.py"]
                               + " ".join(cmd[1:]).split())
    print(f"\n{LINE}\ndone: {run_dir.relative_to(config.ROOT)}/\n{LINE}")
    if sys.platform == "darwin" and plots and _choose(
            "Open the run folder in Finder?", ["yes", "no"]) == 0:
        subprocess.run(["open", str(run_dir)], check=False)
