# KDAB Go-Around Detection Pipeline

Detects and classifies go-around events in raw ADS-B arrival data
(OpenSky-style state vectors) for Daytona Beach Intl (KDAB). Built for the
ERAU-fleet yearlong arrivals dataset, but every airport-specific parameter
lives in `config.py`, so the method transfers to other airports/datasets.

## Method overview

1. **Load & clean** (`loading.py`) - daily parquet files; AGL computed from
   GNSS altitude minus field elevation (baro altitude is QNH-uncorrected and
   only used as a gap-filler after re-referencing to GNSS); altitude and
   vertical rate get a 7 s rolling-median smooth.
2. **Flight legs** (`loading.py`) - each aircraft-day is split into legs on
   20 min data gaps. Training aircraft fly several legs/day and several
   approaches/leg, so all detection is per approach, never per flight.
3. **Approach detection** (`approaches.py`) - contiguous intervals where the
   aircraft is on a runway's extended centerline (< 5 NM out, < 0.30 NM
   cross-track, track within 25 deg of runway heading, below 1,500 ft AGL,
   descending below 800 ft AGL at some point). Segments < 60 s apart merge;
   runway is a vote weighted toward low-altitude points (parallel-runway
   safe). A fallback catches arrivals whose final dips below the local ADS-B
   coverage floor (~50-250 ft AGL) before alignment is established.
4. **Outcome classification** (`classify.py`) - each approach becomes one of:

   | outcome | meaning |
   |---|---|
   | `full_stop` | touchdown (or data ends low/descending at leg end = landed below coverage floor) |
   | `touch_and_go` | touchdown evidence, then sustained climb |
   | `go_around` | **no touchdown; V-shaped low point (plateau <= 20 s); sustained climb (>= 300 fpm for >= 15 s within 120 s of the low point) regaining >= 250 ft; low point below 1,000 ft AGL** |
   | `low_approach` | no touchdown, climb-out, but a level plateau >= 30 s at the low point (deliberate low pass) |
   | `ga_ambiguous` | no touchdown + climb-out, plateau 20-30 s - kept, flagged, excluded from the strict count |
   | `continued_approach` | aligned fragment followed < 4 min later by another aligned segment on the same leg (S-turn/realignment) - not an independent attempt |
   | `unresolved` | outcome not classifiable (coverage loss, unusual maneuvering) |

   Touchdown evidence = any of: `onground` flag, AGL <= 25 ft, plateau
   >= 20 s below 120 ft AGL (GNSS bias makes "on the runway" read up to
   ~100 ft), groundspeed <= 50 kt below 200 ft, or a > 12 s data gap below
   150 ft (dip below the coverage floor).

   The **plateau duration** is the core shape metric. For climb-away events
   it is measured from the last descent into the low band (min AGL + 60 ft)
   until sustained climb start - anchoring on the climb makes it immune to
   slow altitude drift across the band edge during a level segment. (For
   events with no climb, e.g. landings, it falls back to the contiguous
   time within 60 ft of the profile minimum.) On this dataset the metric
   separates the classes with an empty gap: go-arounds all fall at 0-25 s
   (flare + rotate), low approaches and landing rolls at 30 s and above -
   the 20/30 s cutoffs sit inside the gap.

## Known limitations (state these in any publication)

- Intent is invisible: practice go-arounds, ATC-directed go-arounds, and
  pilot-initiated balked landings are indistinguishable in surveillance data.
- A go-around initiated *from* a sustained level-off is indistinguishable
  from a low approach; 20-30 s plateaus are therefore flagged ambiguous.
- Below the local ADS-B coverage floor, a very short touch cannot be
  distinguished from a low float; plateau + speed evidence decides.
- An aircraft whose entire final approach fell below the coverage floor and
  which reappears already climbing (no observed descent >= 150 ft before
  the low point) is classified `unresolved`, not `go_around` - the data
  cannot distinguish a go-around from a touch-and-go there. Go-arounds
  initiated fully below the coverage floor are therefore undetectable.
- Scope is go-arounds from established, runway-aligned approaches. Breakoffs
  from circling/power-off-180 maneuvers (never aligned) are out of scope and
  appear as `unresolved`.
- traffic's built-in `Flight.go_around` counts any re-approach (including
  after touch-and-goes) and requires a subsequent landing attempt; this
  pipeline's definition excludes touchdowns and keeps go-arounds followed by
  departures. Cross-check disagreements are definitional (see
  `validate_traffic.py`).

## Running it

Use the repo-root entry point (see the top-level README for details):

```bash
python run_analysis.py <dataset_name>            # e.g. KDAB_2025
python run_analysis.py <dataset_name> --report --calibrate
python run_analysis.py new-airport KBNA          # generate an airport profile
```

Research extras (run after configuring an airport profile):

```bash
python -m goaround_pipeline.sensitivity          # threshold sensitivity sweep
python -m goaround_pipeline.validate_traffic 20250108   # cross-check vs traffic lib
```

## Outputs

- `results/go_around_events.csv` - one row per go-around (strict +
  ambiguous, distinguished by `outcome`), with UTC/local time, callsign,
  icao24, runway, min AGL, climb-initiation altitude and distance from
  threshold, peak climb rate, altitude regained, plateau duration, and what
  followed (`next_action`: reapproach / leg_end).
- `results/all_approaches.csv` - every classified approach segment (the
  denominator for rates).
- `results/plots/` - per-event PNG (map + altitude/vertical-rate/groundspeed
  traces) for every go_around / ga_ambiguous / low_approach.
- `results/figures/` + `results/summary_report.md` - summary statistics.
- `results/sensitivity_analysis.csv` - event counts under one-at-a-time
  threshold perturbations.
