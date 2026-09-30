#!/usr/bin/env python
"""
Single entry point for the proximity-risk pipeline.

Usage:
    python run_proximity.py KBNA_2025                 # full run
    python run_proximity.py KBNA_2025 --limit 5       # first 5 days (test)
    python run_proximity.py KBNA_2025 --goaround-run 3   # reuse detection
    python run_proximity.py KBNA_2025 --rebuild-baseline
    python run_proximity.py KBNA_2025 --baseline-only
    python run_proximity.py new-airport KBNA          # profile + airspace
    python run_proximity.py check-data KBNA_2025      # data audit only

Datasets live in datasets/<ICAO>_<label>/ (daily .parquet or .csv files);
the ICAO prefix of the folder name selects airports/<ICAO>.yaml. Every run
writes fresh numbered folders output/<ICAO>/goaround/run_NN/ and
output/<ICAO>/proximity_risk/run_NN/ (same NN). Method: FRAMEWORK.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "output"


def list_datasets() -> list[Path]:
    if not DATASETS_DIR.exists():
        return []
    return sorted(
        d for d in DATASETS_DIR.iterdir()
        if d.is_dir() and (any(d.glob("*.parquet")) or any(d.glob("*.csv")))
    )


def resolve_dataset(name: str | None) -> Path:
    ds = list_datasets()
    if name:
        path = DATASETS_DIR / name
        if not path.is_dir():
            options = ", ".join(d.name for d in ds) or "(none)"
            sys.exit(f"error: no dataset folder '{name}' in datasets/. "
                     f"Available: {options}")
        return path
    if len(ds) == 1:
        return ds[0]
    if not ds:
        sys.exit("error: no datasets found. Create datasets/<ICAO>_<label>/ "
                 "and put daily .parquet/.csv files in it.")
    names = "\n  ".join(d.name for d in ds)
    sys.exit(f"multiple datasets found - pick one:\n  {names}\n"
             f"usage: python run_proximity.py <dataset_name>")


def cmd_new_airport(icao: str) -> None:
    from proximity_pipeline import airspace

    path = airspace.create_profile(icao)
    print(f"wrote {path.relative_to(ROOT)}\n"
          "Review it before running: set timezone, preset "
          "(training_ga/air_carrier), terrain (flat/dem), "
          "assume_arrivals_dataset, and VERIFY airspace.ceiling_ft_msl "
          "against the sectional chart (pre-filled as field elevation + "
          "4,000 ft).")


def cmd_check_data(name: str | None, limit: int | None) -> None:
    from proximity_pipeline import airspace, loading

    dataset = resolve_dataset(name)
    icao = dataset.name.split("_")[0].upper()
    airspace.load_airspace(icao)
    print(f"check-data: {dataset.name}  airport {icao}")
    table = loading.audit_dataset(dataset, limit=limit)
    out = OUTPUT_DIR / icao / "proximity_risk"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"check_data_{dataset.name}.csv"
    table.to_csv(path, index=False)
    print(f"\nper-day table: {path.relative_to(ROOT)}")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "new-airport":
        cmd_new_airport(sys.argv[2])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "check-data":
        ap = argparse.ArgumentParser(prog="run_proximity.py check-data")
        ap.add_argument("dataset", nargs="?")
        ap.add_argument("--limit", type=int, metavar="N")
        a = ap.parse_args(sys.argv[2:])
        cmd_check_data(a.dataset, a.limit)
        return

    ap = argparse.ArgumentParser(
        description="Run the proximity-risk analysis on a dataset folder "
                    "(go-around detection, baseline, encounters, "
                    "statistics, figures, summary.pdf).")
    ap.add_argument("dataset", nargs="?",
                    help="folder name under datasets/ (e.g. KBNA_2025)")
    ap.add_argument("--goaround-run", type=int, metavar="N",
                    help="reuse output/<ICAO>/goaround/run_N instead of "
                         "re-running go-around detection")
    ap.add_argument("--rebuild-baseline", action="store_true",
                    help="force a baseline rebuild even if a matching "
                         "stored baseline exists")
    ap.add_argument("--baseline-only", action="store_true",
                    help="build/validate the baseline and stop")
    ap.add_argument("--baseline-from", metavar="ICAO_label",
                    help="score events against another dataset's stored "
                         "baseline for the same airport")
    ap.add_argument("--window", type=float, default=None, metavar="MIN",
                    help="pre/post window length in minutes (default 10)")
    ap.add_argument("--include-ambiguous", action="store_true",
                    help="primary event set also includes ga_ambiguous")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the sensitivity sweep")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip all figures and per-event plots")
    ap.add_argument("--no-event-plots", action="store_true",
                    help="skip per-event plots only")
    ap.add_argument("--goaround-report", action="store_true",
                    help="go-around stage also writes its figures, "
                         "narrative report and PDF")
    ap.add_argument("--calibrate", action="store_true",
                    help="go-around stage also writes the plateau-duration "
                         "calibration histogram")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="process only the first N daily files (testing)")
    ap.add_argument("--workers", type=int, metavar="N",
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel day processing (default: CPU count - 1)")
    args = ap.parse_args()

    dataset = resolve_dataset(args.dataset)
    from proximity_pipeline import pipeline
    pipeline.run_all(dataset, args, argv=sys.argv)


if __name__ == "__main__":
    main()
