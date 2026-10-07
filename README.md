# Proximity Risk After Go-Arounds

Does proximity risk in the terminal airspace around an airport increase
after a go-around, relative to the same airspace immediately before the
go-around and to the airport's own baseline behaviour? This repository
answers that question from raw ADS-B data with a rule-based, explainable
pipeline: the go-around detector of
[GoAroundAlgo](https://github.com/sponbob3/GoAroundAlgo) (vendored
unchanged), an analytic pairwise-encounter computation, a
risk-adjusted regression baseline, and pre/post/baseline statistics.
The complete method and build specification is **FRAMEWORK.md**.

Pure Python (pandas / NumPy / SciPy / statsmodels, no machine learning):
every threshold is a named, documented parameter in
`proximity_pipeline/config.py`; nothing airport-specific is in code.

## Quick start

Python 3.12 (3.11 also works). Python 3.13+ is not yet supported by all
of the traffic library's dependencies, which then have to be compiled.

```bash
python3.12 --version               # install Python 3.12 from python.org
                                   # (or: brew install python@3.12)
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# every new terminal: source .venv/bin/activate  (prompt shows "(.venv)")

# put daily .parquet/.csv files of ALL traffic around the airport in
# datasets/<ICAO>_<label>/ (directly or in subfolders, e.g. one per month)
#   e.g. datasets/KMCO_2025Q1/2025-01/KMCO_20250101.parquet
python run_proximity.py check-data KMCO_2025Q1    # audit the data first
python run_proximity.py KMCO_2025Q1               # the whole analysis
```

The airport profile is created automatically on the first run (see
below); nothing has to be edited by hand. Events are go-arounds and
touch-and-goes: both put an aircraft back into the airspace on a
climb-out, so both are analysed; each keeps its own name in every table
and plot, and results are also given per type.

One run writes two fresh numbered folders with the same number:

| folder | content |
|---|---|
| `output/KBNA/goaround/run_NN/` | the go-around pipeline's outputs (`all_approaches.csv`, `go_around_events.csv`, `summary.pdf`, per-event plots, `run_config.txt`); climb-aways whose low point was hidden below the ADS-B coverage floor are go-arounds (`reclassified_from`, `low_point_hidden_s` columns; FRAMEWORK.md section 13) |
| `output/KBNA/proximity_risk/run_NN/` | `summary.pdf` (verdicts on page 1), `results_primary.csv` (IRR / SIR per endpoint and scope), `go_around_risk.csv` (one row per go-around or touch-and-go), `equilibrium.csv`, `epoch.csv`, `sensitivity.csv`, `encounters.csv`, `window_metrics.parquet`, `ceiling_check.csv`, `data_quality.csv`, `figures/`, `events/` (one PNG per go-around or touch-and-go), `run_config.txt` |
| `output/KBNA/proximity_risk/baseline/<label>/` | the stored, validated baseline (`baseline.json`, `baseline_windows.parquet`, `error_model.csv`, `phase1_removed.csv`, `validation/`, `baseline_report.pdf`), reused automatically while the data and parameters match |

Every command and option, with examples, is in **TERMINAL_COMMANDS.pdf**
(regenerate with `python tools/make_terminal_commands_pdf.py`).

## Method (one paragraph)

All traffic is placed on a 1-second grid in a local plane around the
airport (gaps up to 10 s interpolated and flagged). Inside a cylinder of
10 NM radius up to the charted controlled-airspace ceiling (checked
against the data and raised if the go-around climb-outs spill over it),
every pair of airborne aircraft is evaluated each second with the analytic
zone-entry interval (a quadratic in time horizontally, linear vertically)
against three nested tiers: T1 3 NM / 1,000 ft, T2 0.5 NM / 500 ft, T3
500 ft / 100 ft, both as observed (inside during the step, so a pass
between two samples is never missed) and as predicted (straight-line
projection enters within 120 s). Consecutive flagged seconds form one
encounter with a geometry class (in-trail, parallel runways, crossing,
head-on, runway area) and flight phases. Encounter counts per 10-minute
window are modelled by a negative binomial regression on hour, day type,
month, runway flow and traffic with pair-hours as exposure (Phase I
trimming, out-of-sample calibration, stability curve); the frozen model
scores the pre and post windows of every go-around. Post vs pre gives a
rate ratio from a conditional likelihood with event fixed effects and a
day-clustered bootstrap CI; post vs baseline gives a standardised
incidence ratio with a day-block bootstrap CI; Wilcoxon tests confirm
both. The same is done for encounters involving the go-around aircraft
against matched full-stop landings. A Paielli & Erzberger conflict
probability with data-estimated errors gives expected conflicts and its
own calibration report. A one-at-a-time sensitivity sweep re-runs the
analysis over window, radius, ceiling, lookahead, tier thresholds,
event set, geometry and baseline exclusion without repeating the
pairwise computation.

## Airport profiles

`airports/<ICAO>.yaml` is the go-around pipeline's profile plus an
`airspace:` block (charted ceiling, radius, ceiling mode) and an optional
`flows:` block. When it does not exist, the first run creates it
automatically (`proximity_pipeline/profile_setup.py`, FRAMEWORK.md
section 14): runways and elevation from the OurAirports database
(downloaded once), timezone from the coordinates, preset
(air_carrier / training_ga) from final-approach speeds in the data,
`assume_arrivals_dataset` from whether landings are seen on the ground,
the ADS-B coverage floor near the field (raises the go-around pipeline's
landing-truncation heights when coverage ends high), and the ceiling at
field elevation + 4,000 ft (checked against the data every run). Every
detected value is printed and written into the file with its evidence;
the file is the one place to change any of them and is never overwritten.
`python run_proximity.py new-airport KMCO KMCO_2025Q1` creates it without
running the analysis.

OpenSky state vectors repeat an aircraft's last position for up to 300 s
after its last position report; when the files carry the `last_position`
column, those frozen rows are dropped and positions are re-timed to their
report time before either stage runs (FRAMEWORK.md section 4.2).

## Repository layout

```
run_proximity.py              <- the only file you run
FRAMEWORK.md                  <- method and build specification
TERMINAL_COMMANDS.pdf         <- every command with examples
airports/<ICAO>.yaml          <- airport profiles (+ airspace, flows)
datasets/<ICAO>_<label>/      <- your data (gitignored)
output/                       <- runs, baselines, caches (gitignored)
goaround_pipeline/            <- vendored unchanged (VENDORED_FROM.txt)
proximity_pipeline/
  config.py                   every threshold, documented
  airspace.py                 volume, ceiling check, profile extensions
  loading.py                  cleaning, 1 s grid, projection, data audit
  pairs.py                    zone-entry intervals, pruning, encounters
  geometry.py                 flight phases, geometry classes
  probability.py              error model, conflict probability, calibration
  windows.py                  window metrics, exposure, quality, flows
  baseline.py                 regression baseline, Phase I/II, validation
  events.py                   event windows, clusters, matched controls
  stats.py                    IRR, SIR, equilibrium, power, epoch
  sensitivity.py              one-at-a-time sweep
  pipeline.py                 per-day cache and run orchestration
  goaround_adapter.py         the only bridge to goaround_pipeline
  viz.py / report_pdf.py      figures, summary.pdf, baseline_report.pdf
  validate_cpa.py             cross-check vs the traffic library
tools/make_terminal_commands_pdf.py
tools/make_synthetic_dataset.py   <- synthetic traffic for tests/smoke runs
tests/                        pytest suite (FRAMEWORK.md section 18)
```

## Tests

```bash
python -m pytest                 # ~2 min, includes end-to-end synthetic runs
python -m pytest -m "not slow"   # unit tests only
```

The suite checks the analytic intervals against brute force, a head-on
pass between two samples, scenario tiers and classes, pruning safety,
the conflict probability against Monte Carlo, baseline calibration on
simulated windows, detection of an injected effect and the false-positive
rate, go-around output parity with GoAroundAlgo, and closest-approach
agreement with the traffic library.

## Data policy and limitations

Raw ADS-B data is never committed (OpenSky terms; several GB). Proximity
risk is a surrogate for safety, not a collision probability; the
surveillance floor, straight-line prediction, regulatory tier thresholds,
invisible go-around intent and the observational design are the
limitations to state in any publication (FRAMEWORK.md section 20).
