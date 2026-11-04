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

## Getting started

You only type a few setup commands once. After that you start the program
and it asks for everything else.

| You use | Set up with (step 3) | Start with (step 5) |
|---|---|---|
| **macOS** | the macOS / Linux commands | double-click **`Run Proximity Analysis.command`** |
| **Windows** | the Windows commands | double-click **`Run Proximity Analysis.bat`** |
| **Linux** | the macOS / Linux commands | `python run_proximity.py` in a terminal |

### 1. Install Python 3.12

Use **Python 3.12** (3.11 also works). Newer versions (3.13, 3.14) are not
supported yet by some dependencies of the `traffic` library, and macOS's
built-in `python3` is too old.

- **macOS:** download the *macOS 64-bit universal2 installer* for Python
  3.12 from <https://www.python.org/downloads/macos/> and run it.
  (Homebrew's `python@3.12` does not run on older, unsupported macOS
  versions; the python.org installer does.)
- **Windows:** download Python 3.12 from
  <https://www.python.org/downloads/windows/> and tick
  **"Add python.exe to PATH"** in the installer.
- **Linux:** install `python3.12` and `python3.12-venv` with your package
  manager.

Check it in a new terminal window:

```bash
python3.12 --version        # macOS / Linux  -> Python 3.12.x
py -3.12 --version          # Windows        -> Python 3.12.x
```

### 2. Get the code

```bash
git clone https://github.com/sponbob3/CollisionRiskAlgo.git
cd CollisionRiskAlgo
```

(or download the ZIP from GitHub and open a terminal in the unzipped folder).

### 3. Create the Python environment (once)

All commands below are run inside the `CollisionRiskAlgo` folder.

**macOS and Linux users** (Terminal):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows users** (Command Prompt or PowerShell):

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The prompt now starts with `(.venv)`. In every **new** terminal window,
activate the environment again (`source .venv/bin/activate`, or
`.venv\Scripts\activate` on Windows) before running anything. Use
`python -m pip`, not bare `pip`: it always installs into the active
environment. The double-click launchers in step 5 (macOS and Windows)
use the environment by themselves, so with them you never have to
activate it.

### 4. Add your data

Put one file per day (`.parquet` preferred, `.csv` accepted) of OpenSky
ADS-B state vectors for **all traffic around the airport** into a folder
under `datasets/`, directly or in subfolders (e.g. one per month):

```
datasets/
  KMCO_2025Q1/
    2025-01/KMCO_20250101.parquet
    2025-01/KMCO_20250102.parquet
    ...
```

The folder name only has to start with the airport's ICAO code (`KMCO`,
`kmco_2025q1` and `KMCO-2025Q1` all work); capitalisation does not
matter, and renaming it later loses nothing. Required columns:
`timestamp, icao24, callsign, latitude, longitude, altitude,
geoaltitude, vertical_rate, groundspeed, track, onground`
(`last_position` is used when present). `datasets/` is never committed.

### 5. Run it

- **macOS users:** double-click **`Run Proximity Analysis.command`** in
  Finder. A Terminal window opens and the questions start. The first
  time, macOS may block it: right-click it, choose **Open**, then
  **Open**. If it says it cannot be executed (e.g. after downloading a
  ZIP), run `chmod +x "Run Proximity Analysis.command"` once in the
  repository folder.
- **Windows users:** double-click **`Run Proximity Analysis.bat`** in File
  Explorer. A console window opens and the questions start. If Windows
  shows "Windows protected your PC", click **More info**, then **Run
  anyway**.
- **Linux users (or anyone who prefers a terminal):** with the
  environment active, run

  ```bash
  python run_proximity.py
  ```

Both launchers are only shortcuts for `python run_proximity.py`; they
need the environment from step 3 to exist.

**The program asks for everything.** It lists the datasets it finds and
asks which one to analyse, then shows what it found and asks what to do:

1. dataset (always asked);
2. airport profile (created automatically from the OurAirports database
   and the data the first time; needs internet once);
3. pairwise cache: reuse the per-day computations or recompute them;
4. go-around detection: reuse an earlier run made from the same data, or
   detect again;
5. baseline: reuse the stored one or rebuild it;
6. outputs: figures and PDFs, one plot per go-around, sensitivity
   analysis, the optional probability add-on;
7. settings: keep the defaults or change window length, workers and
   ambiguous go-arounds;
8. a summary of every choice and the equivalent command, then start or
   cancel.

Answer with the option number; Enter alone picks option 1. The first run
on new data computes the per-day cache (the slowest step: up to a few
hours for three months of data); later runs reuse it and take minutes.
The sensitivity analysis adds a few hours.

### 6. Results

Open `output/<ICAO>/proximity_risk/run_NN/summary.pdf` first: key results
in plain sentences, the checks behind them and a glossary on page 1, and
every figure captioned. One run writes two numbered folders with the same
number:

| folder | content |
|---|---|
| `output/KMCO/goaround/run_NN/` | the go-around pipeline's outputs (`all_approaches.csv`, `go_around_events.csv`, `summary.pdf`, per-event plots, `run_config.txt`); climb-aways whose low point was hidden below the ADS-B coverage floor are go-arounds (`reclassified_from`, `low_point_hidden_s` columns; FRAMEWORK.md section 13) |
| `output/KMCO/proximity_risk/run_NN/` | `summary.pdf`, `change_summary.csv` (the main result: the before → after change around go-arounds vs the same change around normal landings at the same hours), `landing_comparison/` (the checks behind it: before-window checks of both groups against the baseline, pre-trend, placebo, regression, robustness, leave-one-out, minute by minute), `landing_windows.csv`, `go_around_aircraft.csv` (the event aircraft vs other aircraft), `results_primary.csv` (rates per unit of traffic: IRR / SIR per endpoint), `go_around_risk.csv` (one row per go-around or touch-and-go), `equilibrium.csv`, `epoch.csv`, `sensitivity.csv`, `encounters.csv`, `window_metrics.parquet`, `ceiling_check.csv`, `data_quality.csv`, `figures/`, `events/` (one PNG per go-around or touch-and-go), `run_config.txt` |
| `output/KMCO/proximity_risk/baseline/<label>/` | the stored, validated baseline (`baseline.json`, `baseline_windows.parquet`, `phase1_removed.csv`, `validation/`, `baseline_report.pdf`), reused automatically while the data and settings match |
| `output/KMCO/proximity_risk/cache/` | per-day pairwise computations, reused by every later run |

Events are go-arounds and touch-and-goes: both put an aircraft back into
the airspace on a climb-out, so both are analysed; each keeps its own
name in every table and plot.

### Command-line options (optional)

Everything the questions do can also be typed directly; the program
prints the matching command before it starts.

```bash
python run_proximity.py KMCO_2025Q1                   # whole analysis, no questions
python run_proximity.py KMCO_2025Q1 --goaround-run 1  # reuse goaround/run_01
python run_proximity.py check-data KMCO_2025Q1        # data audit only
python run_proximity.py new-airport KMCO KMCO_2025Q1  # airport profile only
```

| option | effect |
|---|---|
| `--goaround-run N` | reuse `output/<ICAO>/goaround/run_N` instead of detecting again |
| `--rebuild-baseline` | rebuild the baseline even if a matching one is stored |
| `--baseline-only` | build and validate the baseline, then stop |
| `--baseline-from DATASET` | score against another dataset's stored baseline (same airport) |
| `--no-sensitivity` | skip the sensitivity analysis |
| `--no-plots` / `--no-event-plots` | skip all figures / only the per-event plots |
| `--probability` | also run the optional conflict-probability model |
| `--window MIN` | before/after window length (default 10) |
| `--include-ambiguous` | count ambiguous go-arounds as events |
| `--airport ICAO` | airport code, when the folder name does not start with it |
| `--workers N` | parallel day processing (default: CPU count − 1) |
| `--limit N` | first N days only (quick test) |
| `--goaround-report` / `--calibrate` | extra outputs from the go-around stage |

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
both. Because a go-around may itself add traffic, the totals and the
traffic change are reported too, next to the same before-to-after change
around normal landings at the same hours of day, and the go-around
aircraft's own encounters are compared with those of the other aircraft
present. A Paielli & Erzberger conflict-probability model is available
as an optional add-on (`--probability`). A one-at-a-time sensitivity sweep re-runs the
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
run_proximity.py              <- the only file you run (no arguments: asks questions)
Run Proximity Analysis.command <- macOS: double-click in Finder to start
Run Proximity Analysis.bat    <- Windows: double-click in File Explorer to start
FRAMEWORK.md                  <- method and build specification
airports/<ICAO>.yaml          <- airport profiles (+ airspace, flows)
datasets/<folder>/            <- your data (gitignored; name it e.g. KMCO_2025Q1)
output/                       <- runs, baselines, caches (gitignored)
goaround_pipeline/            <- vendored unchanged (VENDORED_FROM.txt)
proximity_pipeline/
  config.py                   every threshold, documented
  airspace.py                 volume, ceiling check, profile extensions
  loading.py                  cleaning, 1 s grid, projection, data audit
  pairs.py                    zone-entry intervals, pruning, encounters
  geometry.py                 flight phases, geometry classes
  probability.py              optional add-on: conflict probability (--probability)
  labels.py                   plain-language labels for figures and reports
  windows.py                  window metrics, exposure, quality, flows
  baseline.py                 regression baseline, Phase I/II, validation
  events.py                   event windows, clusters, matched controls
  stats.py                    IRR, SIR, equilibrium, power, epoch
  compare.py                  comparison with normal landings and its checks
  wizard.py                   the questions asked when started without arguments
  sensitivity.py              one-at-a-time sweep
  pipeline.py                 per-day cache and run orchestration
  goaround_adapter.py         the only bridge to goaround_pipeline
  viz.py / report_pdf.py      figures, summary.pdf, baseline_report.pdf
  validate_cpa.py             cross-check vs the traffic library
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
