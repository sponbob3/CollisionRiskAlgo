#!/usr/bin/env python
"""
Generate TERMINAL_COMMANDS.pdf at the repository root: every command and
option of the proximity-risk pipeline with examples, from the single
COMMANDS table below (keep it in sync with run_proximity.py).

Usage:
    python tools/make_terminal_commands_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "TERMINAL_COMMANDS.pdf"

# (section, [(command, what it does)]) - the single source of truth
COMMANDS = [
    ("1. Set-up (once per machine)", [
        ("git clone https://github.com/sponbob3/CollisionRiskAlgo.git\n"
         "cd CollisionRiskAlgo",
         "Get the repository."),
        ("python -m venv .venv\n"
         "source .venv/bin/activate          # Windows: .venv\\Scripts\\activate\n"
         "pip install -r requirements.txt",
         "Create the virtual environment and install every dependency "
         "(pandas < 2.3 is pinned for the traffic library)."),
        ("git pull",
         "Update to the latest version later (run inside the repository)."),
    ]),
    ("2. Data", [
        ("mkdir -p datasets/KBNA_2025\n"
         "cp /path/to/daily/*.parquet datasets/KBNA_2025/",
         "One file per day (.parquet preferred, .csv accepted) of OpenSky-"
         "style state vectors for ALL traffic around the airport, columns: "
         "timestamp, icao24, callsign, latitude, longitude, altitude, "
         "geoaltitude, vertical_rate, groundspeed, track, onground. The "
         "folder name is <ICAO>_<label>; the prefix selects "
         "airports/<ICAO>.yaml. datasets/ is gitignored."),
        ("python run_proximity.py check-data KBNA_2025\n"
         "python run_proximity.py check-data KBNA_2025 --limit 5",
         "Data audit only: days found, columns, per-day message and "
         "aircraft counts, gaps and receiver outages, altitude references, "
         "and whether the volume meets the baseline minimum (28 days, "
         "90 recommended). Writes output/KBNA/proximity_risk/"
         "check_data_KBNA_2025.csv."),
    ]),
    ("3. New airport", [
        ("python run_proximity.py new-airport KBNA",
         "Generate airports/KBNA.yaml from the OurAirports database "
         "(runways, elevation) with the airspace block pre-filled as field "
         "elevation + 4,000 ft. Then EDIT: timezone, preset "
         "(air_carrier / training_ga), terrain, assume_arrivals_dataset, "
         "airspace.ceiling_ft_msl (verify against the sectional chart), "
         "and optionally a flows: block (which runway ends operate "
         "together)."),
    ]),
    ("4. Running the analysis", [
        ("python run_proximity.py KBNA_2025",
         "The complete run: go-around detection -> output/KBNA/goaround/"
         "run_NN/, then baseline (built or reused), encounters, window "
         "metrics, every event, sensitivity sweep, tables, figures, "
         "per-event plots and summary.pdf -> output/KBNA/proximity_risk/"
         "run_NN/ (same NN)."),
        ("python run_proximity.py KBNA_2025 --limit 5 --no-sensitivity "
         "--no-plots",
         "Quick test on the first 5 days without the sweep or figures."),
        ("python run_proximity.py KBNA_2025 --workers 4",
         "Parallel day processing (default: CPU count - 1)."),
        ("python run_proximity.py KBNA_2025 --goaround-report --calibrate",
         "Also write the go-around stage's figures, narrative report and "
         "PDF, and its plateau-duration calibration histogram."),
        ("python run_proximity.py KBNA_2025 --window 5\n"
         "python run_proximity.py KBNA_2025 --window 15",
         "Pre/post window length in minutes (default 10; the sweep runs "
         "5 and 15 anyway)."),
        ("python run_proximity.py KBNA_2025 --include-ambiguous",
         "Primary event set also includes ga_ambiguous go-arounds."),
        ("python run_proximity.py KBNA_2025 --no-event-plots",
         "Skip the one-PNG-per-go-around plots only (figures and "
         "summary.pdf are still produced)."),
    ]),
    ("5. Reusing runs", [
        ("python run_proximity.py KBNA_2025 --goaround-run 3",
         "Reuse output/KBNA/goaround/run_03 instead of re-running "
         "go-around detection (the proximity run still gets a new NN)."),
        ("ls output/KBNA/proximity_risk/cache/2025/",
         "Per-day pairwise caches (parameter-hashed). They are reused "
         "automatically by every run and sweep; delete the folder to "
         "force recomputation."),
    ]),
    ("6. Baselines", [
        ("python run_proximity.py KBNA_2025 --baseline-only",
         "Build (or reuse) and validate the baseline, write baseline_"
         "report.pdf, and stop. Stored in output/KBNA/proximity_risk/"
         "baseline/2025/ (baseline.json, baseline_windows.parquet, "
         "error_model.csv, phase1_removed.csv, validation/)."),
        ("python run_proximity.py KBNA_2025 --rebuild-baseline",
         "Force a rebuild even if the stored baseline matches the data "
         "and parameters (it is rebuilt automatically when either "
         "changes)."),
        ("python run_proximity.py KBNA_spring26 --baseline-from KBNA_2025",
         "Score another dataset's events against the stored KBNA_2025 "
         "baseline (same airport)."),
    ]),
    ("7. Checks and tests", [
        ("python -m pytest",
         "The whole test suite (analytic intervals vs brute force, missed "
         "passes, scenarios, pruning, probability vs Monte Carlo, "
         "baseline calibration, injected effect, go-around parity, "
         "traffic cross-check). About two minutes."),
        ("python -m pytest -q -m 'not slow'",
         "Skip the end-to-end synthetic runs."),
        ("python -m proximity_pipeline.validate_cpa KBNA_2025 20250108 "
         "20250115",
         "Cross-check closest-approach distances against the traffic "
         "library on the named days."),
        ("python tools/make_synthetic_dataset.py KBNA_synth --days 6\n"
         "python run_proximity.py KBNA_synth --limit 3",
         "Synthetic traffic for a smoke test without real data "
         "(--intruder-rate and --post-ga-irr inject a known effect)."),
        ("python tools/make_terminal_commands_pdf.py",
         "Regenerate this PDF after changing the CLI."),
    ]),
    ("8. Where things land", [
        ("output/KBNA/goaround/run_NN/",
         "Exact go-around pipeline outputs (all_approaches.csv, "
         "go_around_events.csv, summary.pdf, plots/, run_config.txt)."),
        ("output/KBNA/proximity_risk/run_NN/",
         "run_config.txt, ceiling_check.csv, data_quality.csv, "
         "window_metrics.parquet, encounters.parquet/.csv, "
         "go_around_risk.csv (+ _involved), results_primary.csv, "
         "equilibrium.csv, epoch.csv, sensitivity.csv, figures/, events/, "
         "summary.pdf."),
    ]),
]


def main() -> None:
    st = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=st["Title"], fontSize=16, alignment=0,
                        spaceAfter=2)
    sub = ParagraphStyle("sub", parent=st["Normal"], fontSize=9.5,
                         textColor=colors.HexColor("#52514e"), spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=st["Heading2"], fontSize=12,
                        spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("body", parent=st["Normal"], fontSize=9,
                          leading=12)
    code = ParagraphStyle("code", parent=st["Code"], fontSize=8.2,
                          leading=10.5, backColor=colors.HexColor("#f3f2ee"),
                          borderPadding=4, leftIndent=0)
    story = [
        Paragraph("Proximity risk after go-arounds — terminal commands", h1),
        Paragraph("Every command and option of run_proximity.py with "
                  "examples. Run everything from the repository root with "
                  "the virtual environment active. Method: FRAMEWORK.md.",
                  sub),
    ]
    for section, items in COMMANDS:
        story.append(Paragraph(section, h2))
        rows = []
        for cmd, what in items:
            rows.append([Preformatted(cmd, code), Paragraph(what, body)])
        t = Table(rows, colWidths=[3.35 * inch, 3.55 * inch], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d8d7d2")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Options summary: --goaround-run N, --rebuild-baseline, "
        "--baseline-only, --baseline-from ICAO_label, --window MIN, "
        "--include-ambiguous, --no-sensitivity, --no-plots, "
        "--no-event-plots, --goaround-report, --calibrate, --limit N, "
        "--workers N. Sub-commands: new-airport ICAO, check-data "
        "ICAO_label [--limit N].", body))
    SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.7 * inch,
                      rightMargin=0.7 * inch, topMargin=0.7 * inch,
                      bottomMargin=0.7 * inch,
                      title="Proximity risk - terminal commands").build(story)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
