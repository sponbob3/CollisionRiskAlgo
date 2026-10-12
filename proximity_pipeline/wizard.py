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
            ans = input(f"Type a number and press Enter (just Enter = "
                        f"{default}): ").strip()
        except EOFError:
            ans = ""
        if not ans:
            return default - 1
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return int(ans) - 1
        print(f"  Please type a number from 1 to {len(options)}.")


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
    print(" Answer each question by typing a number and pressing Enter.")
    print(" Pressing Enter alone always picks the recommended answer.")
    print(LINE)

    # ---- 1. dataset / airport
    _step(1, TOTAL_STEPS, "Data")
    ds = rp.list_datasets()
    if not ds:
        print(f"No data found in {rp.DATASETS_DIR}.\nMake a folder in "
              "datasets/ whose name starts with the airport code (e.g. "
              "datasets/KMCO_2025Q1/) and put the daily .parquet or .csv "
              "files in it (subfolders are fine).")
        return
    from proximity_pipeline import pipeline
    labels = []
    for d in ds:
        n = sum(1 for p in d.rglob("*") if p.suffix in (".parquet", ".csv")
                and not p.name.startswith("."))
        try:
            code = pipeline.dataset_identity(d)[0]
        except ValueError:
            code = "airport unknown"
        labels.append(f"{d.name}  ({code}, {n} daily files)")
    if len(ds) == 1:
        dataset = ds[0]
        print(f"Found one dataset: {labels[0]}")
    else:
        dataset = ds[_choose("Which data do you want to analyse?", labels)]
    try:
        icao, label = pipeline.dataset_identity(dataset)
    except ValueError:
        code = input("Which airport is this? Type its ICAO code (e.g. KMCO) "
                     "and press Enter: ").strip()
        icao, label = pipeline.dataset_identity(dataset, code)

    from proximity_pipeline import config
    from proximity_pipeline import goaround_adapter as ga
    files = ga.data_files(dataset)
    first, last = _dates(files)
    subs = sorted({f.parent.name for f in files if f.parent != dataset})
    print(f"Using {dataset.name}: {len(files)} days of data, {first} to "
          f"{last}" + (f", in {len(subs)} subfolders" if subs else "")
          + f", {_size(dataset)}.")

    # ---- 2. airport profile (nothing to decide)
    _step(2, TOTAL_STEPS, "Airport")
    from proximity_pipeline import airspace, profile_setup
    prof_path = profile_setup.profile_path(icao)
    if prof_path.exists():
        print(f"Airport settings found: {prof_path.relative_to(config.ROOT)}")
    else:
        print("No airport settings yet: creating them now from the "
              "OurAirports database (ourairports.com) and the data.")
    profile_setup.ensure_profile(icao, dataset)
    prof = airspace.load_airspace(icao)
    print(f"{prof.get('name', icao)} ({icao}): {len(ga.config.RUNWAYS)} runway "
          f"ends, elevation {ga.config.FIELD_ELEVATION_FT:,.0f} ft, time zone "
          f"{ga.config.LOCAL_TZ}. Nothing to decide here.")

    # ---- 3. saved per-day computations (pairwise cache)
    _step(3, TOTAL_STEPS, "Saved computations (the slow part)")
    cache_dir = pipeline.cache_root(icao)
    pipeline.adopt_old_caches(files, cache_dir)
    n_cached = (sum(pipeline.cache_is_current(cache_dir, f) for f in files)
                if cache_dir.exists() else 0)
    rebuild_cache = False
    print("Every day of data is first turned into aircraft pairs and "
          "encounters. This is the slowest step, so the result is saved and "
          "reused by later runs.")
    if n_cached:
        rest = len(files) - n_cached
        print(f"Found: {n_cached} of {len(files)} days already computed and "
              f"saved ({_size(cache_dir)})."
              + (f" The other {rest} will be computed in this run." if rest
                 else ""))
        k = _choose("Do you want to use the saved computations?", [
            "Yes, use them (recommended: fast)",
            f"No, delete them and compute all {len(files)} days again from "
            "the raw data (takes hours)"])
        if k == 1:
            sure = _choose("Are you sure? This deletes the saved computations "
                           "for this airport.", [
                               "No, keep them and use them",
                               "Yes, delete them and compute everything "
                               "again"])
            rebuild_cache = sure == 1
    else:
        print(f"Found: nothing saved yet. All {len(files)} days will be "
              "computed in this run (this can take a few hours the first "
              "time). Nothing to decide here.")

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
        print(f"Found: {len(runs)} earlier go-around detection result(s) "
              f"for {icao}"
              + (f", {n_match} made from this same data." if n_match
                 else ", none made from this data."))
        options = []
        for r in runs:
            what = (f"{r['go_arounds']} go-arounds, {r['touch_and_goes']} "
                    f"touch-and-goes, {r['days'] or '?'} days"
                    + (f", {r['first']} to {r['last']}" if r["first"] else ""))
            if r["match"]:
                options.append(f"Use the results in goaround/run_{r['nn']:02d}"
                               f" ({what}; same data)"
                               + (" (recommended)" if r is runs[0] else ""))
            else:
                options.append(f"Use the results in goaround/run_{r['nn']:02d}"
                               f" ({what}; made from DIFFERENT data)")
        redo = ("Detect the go-arounds again (a few minutes per month of "
                "data)" + ("" if runs[0]["match"] else " (recommended)"))
        if runs[0]["match"]:
            options.append(redo)
        else:
            options.insert(0, redo)
        k = _choose("Which go-around results should this run use?", options)
        chosen = (k if runs[0]["match"] else k - 1)
        if 0 <= chosen < len(runs):
            goaround_run = runs[chosen]["nn"]
    else:
        print("Found: no earlier results. The go-arounds will be detected "
              "first (a few minutes per month of data). Nothing to decide "
              "here.")

    # ---- 5. baseline
    _step(5, TOTAL_STEPS, "Baseline (what normal traffic looks like)")
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
        print(f"Found: a saved baseline built from these same {len(files)} "
              f"days (baseline/{folder.name}, rated {verdict or 'unknown'}).")
        k = _choose("Do you want to use the saved baseline?", [
            "Yes, use it (recommended: it is rebuilt automatically anyway "
            "if any setting changed)",
            "No, build a new baseline (adds a few minutes)"])
        rebuild_baseline = k == 1
    else:
        print("Found: no saved baseline for these days. A new one will be "
              "built and saved in this run (a few minutes). Nothing to "
              "decide here.")

    # ---- 6. outputs
    _step(6, TOTAL_STEPS, "What to produce")
    plots = _choose("Do you want the figures and PDF reports?", [
        "Yes: figures, summary.pdf and the baseline report (recommended)",
        "No: only the result tables (CSV files)"]) == 0
    event_plots = False
    if plots:
        event_plots = _choose("Do you want a picture of every single "
                              "go-around?", [
                                  "Yes: one picture per go-around (adds "
                                  "about half an hour for 3 months of data)",
                                  "No: skip these pictures"]) == 0
    sensitivity = _choose("Do you want the sensitivity analysis? (the whole "
                          "analysis is rerun with each setting changed, to "
                          "show the result does not depend on the settings)",
                          ["No: skip it (faster)",
                           "Yes: include it (adds a few hours)"]) == 1
    probability = _choose("Do you want the optional conflict-probability "
                          "model? (an add-on, not part of the main results)",
                          ["No (recommended)", "Yes"]) == 1

    # ---- 7. settings
    _step(7, TOTAL_STEPS, "Settings")
    window = None
    workers = max(1, (os.cpu_count() or 2) - 1)
    include_ambiguous = False
    k = _choose("Do you want to keep the standard settings? "
                f"({config.WINDOW_MIN:g}-minute before/after windows, "
                f"{workers} parallel workers, ambiguous go-arounds not "
                "counted)", ["Yes, keep them (recommended)",
                             "No, let me change them"])
    if k == 1:
        w = input(f"Window length in minutes (just Enter = "
                  f"{config.WINDOW_MIN:g}): ").strip()
        if w:
            window = float(w)
        n = input(f"Parallel workers (just Enter = {workers}): ").strip()
        if n.isdigit() and int(n) > 0:
            workers = int(n)
        include_ambiguous = _choose(
            "Count ambiguous go-arounds as events too?",
            ["No (recommended)", "Yes"]) == 1

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
    print("Saved computations:   "
          + ("delete and compute again" if rebuild_cache else
             f"use ({n_cached} of {len(files)} days saved)"))
    print("Go-around results:    "
          + (f"use goaround/run_{goaround_run:02d}" if goaround_run
             else "detect again"))
    print("Baseline:             "
          + ("build a new one" if rebuild_baseline or not same
             else "use the saved one"))
    print(f"Figures and PDFs:     {yes_no[plots]}")
    print(f"Go-around pictures:   {yes_no[event_plots]}")
    print(f"Sensitivity analysis: {yes_no[sensitivity]}")
    print(f"Probability add-on:   {yes_no[probability]}")
    print(f"Window:               {window or config.WINDOW_MIN:g} minutes")
    print(f"\n(The same run can be started by typing: {' '.join(cmd)})")
    if _choose("Start the analysis now?", ["Start now",
                                           "Cancel (nothing is changed)"]) == 1:
        print("Cancelled. Nothing was changed.")
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
    print(f"\n{LINE}\nDONE. Results: {run_dir.relative_to(config.ROOT)}/  "
          f"(start with summary.pdf)\n{LINE}")
    if sys.platform == "darwin" and plots and _choose(
            "Open the results folder in Finder?", ["Yes", "No"]) == 0:
        subprocess.run(["open", str(run_dir)], check=False)
