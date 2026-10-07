"""
Pipeline configuration: every threshold used by the proximity-risk method
lives here so the method is transparent, tunable, and citable. Units are
noted per parameter. Section numbers refer to FRAMEWORK.md.

Nothing in this file is airport-specific. Running through run_proximity.py
loads the airport profile (airports/<ICAO>.yaml) into the vendored
go-around config (runways, elevation, timezone) and reads the profile's
`airspace:` and `flows:` blocks into the AIRSPACE_* / FLOWS attributes
below via airspace.load_airspace().
"""

from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "output"

# ------------------------------------------------- airport profile (set) ----
# Filled by airspace.load_airspace() from airports/<ICAO>.yaml; the values
# here are placeholders that make the module importable on its own.
AIRSPACE_CLASS = "C"            # informational: B / C / D
AIRSPACE_CEILING_FT_MSL = 4000.0  # charted controlled-airspace ceiling
AIRSPACE_CEILING_MODE = "auto"  # auto | fixed   (section 3.3)
# Optional runway-flow labels: {label: [runway end, ...]}; None = derive
# flows automatically by grouping runway ends by heading (section 8.4).
FLOWS = None

# ------------------------------------------------------- study volume ----
# Cylinder around the airport reference point (section 3).
VOLUME_RADIUS_NM = 10.0
# Sensitivity radius (section 12).
VOLUME_RADIUS_SENSITIVITY_NM = 5.0
# Charted ceiling must lie in this band above field elevation, else the
# run stops (an input mistake, section 3.2).
CEILING_MIN_FT_AGL = 1500.0
CEILING_MAX_FT_AGL = 12000.0
# Encounters are always computed up to ceiling + this buffer and tagged
# inside/outside the nominal ceiling (section 3.3).
CEILING_BUFFER_FT = 2000.0
# Max share of go-around-involved T1 encounters allowed above the nominal
# ceiling before the ceiling is raised (section 3.3).
CEILING_SPILL_MAX = 0.10
# Ceiling override step and cap (ft above field).
CEILING_STEP_FT = 500.0
CEILING_CAP_FT_AGL = 10000.0
# Ceiling sensitivity values (ft above field); "effective" is added at
# run time (section 12).
CEILING_SENSITIVITY_FT_AGL = (2500.0, 5000.0, 10000.0)
# Raw data is kept out to this radius so that dead-reckoning error samples
# (section 7.1) and go-around containment (section 3.3) can look beyond
# the study volume. Anything further from the airport is dropped on load.
LOAD_RADIUS_NM = 25.0

# ----------------------------------------------------- data preparation ----
# Position freshness (section 4.2). OpenSky state vectors repeat an
# aircraft's last known position for up to 300 s after its last position
# report, so many rows can be frozen copies of an old position (on the
# first KMCO test day: 58% of rows). When a file has a position-time
# column, only rows carrying a NEW position report are kept, re-timed to
# the report time; reports older than MAX_POSITION_AGE_S at the row time
# are dropped too. Applied to both stages (go-around and proximity).
POSITION_FRESHNESS_FILTER = True
POSITION_TIME_COLUMNS = ("last_position", "lastposupdate", "time_position")
MAX_POSITION_AGE_S = 10.0
# Geometric altitude that disagrees with barometric altitude by more than
# this (after removing the leg's median geo-baro offset) is treated as
# missing; height above field then falls back to re-referenced barometric
# altitude (section 4.2). Catches isolated GNSS altitude spikes.
GEO_BARO_MAX_DEV_FT = 1000.0
# Airborne = onground false AND groundspeed at/above this (section 4.3).
AIRBORNE_MIN_GS_KT = 50.0
# Position fixes implying a speed above this between consecutive samples
# are dropped as jumps (section 4.2).
MAX_IMPLIED_SPEED_KT = 600.0
# Gaps up to this many seconds are linearly interpolated on the 1 s grid;
# longer gaps leave the aircraft absent (section 4.4).
MAX_INTERP_GAP_S = 10.0
# Grid step (s). The whole method assumes 1 s; kept as a named value so
# the assumption is visible.
GRID_STEP_S = 1.0
# Vertical-rate smoothing window (s), same as the go-around pipeline.
VRATE_SMOOTH_WINDOW_S = 7
# Flight legs split on gaps longer than this (same rule as the go-around
# pipeline, section 4.2).
SEGMENT_GAP_MINUTES = 20.0

# ------------------------------------------- go-around classification ----
# Climb-aways whose low point is hidden by a data gap (FRAMEWORK.md
# section 13). Where ADS-B coverage ends above the runway (KMCO: ~530 ft
# above field), a go-around started below the coverage floor shows as a
# gap at the bottom of the approach followed by a climb. The go-around
# pipeline measures that gap as level time at the low point and calls the
# climb-away a low approach / ambiguous. When the time NOT hidden by gaps
# is within the go-around cutoff (LEVEL_GA_MAX_S), the event is a
# go-around. A low pass that is actually SEEN level for longer stays a
# low approach. Gaps longer than GA_HIDDEN_GAP_MIN_S count as hidden time.
GA_HIDDEN_LOW_POINT_AS_GO_AROUND = True
GA_HIDDEN_GAP_MIN_S = 5.0

# ---------------------------------------------------- separation tiers ----
# (horizontal NM, vertical ft). Basis in section 5.
TIERS = {
    "T1": (3.0, 1000.0),      # terminal radar separation minimum
    "T2": (0.5, 500.0),       # close proximity / Class C IFR-VFR vertical
    "T3": (500.0 / 6076.12, 100.0),   # NMAC cylinder: 500 ft x 100 ft
}
TIER_ORDER = ("T1", "T2", "T3")
# Tier sensitivity multipliers applied to each H and V one at a time.
TIER_SENSITIVITY_FACTORS = (0.75, 1.25)
# Straight-line lookahead for predicted encounters (s), with sensitivity
# values (section 5).
T_LOOKAHEAD_S = 120.0
T_LOOKAHEAD_SENSITIVITY_S = (60.0, 180.0)

# ------------------------------------------------ pairwise computation ----
# Pair pruning radius = H_T1 + 2 * PRUNE_VMAX_KT * T_LOOKAHEAD (section 6.1).
PRUNE_VMAX_KT = 350.0
# Consecutive flagged seconds of a pair form one encounter episode; gaps up
# to this are merged (section 6.3).
EPISODE_MERGE_GAP_S = 10.0
# Days are processed in time chunks of this length to bound memory.
PAIR_CHUNK_S = 3600

# --------------------------------------------------- geometry classes ----
# Section 6.3. Relative track angle bands (deg).
GEOM_IN_TRAIL_MAX_DPSI_DEG = 30.0
GEOM_CROSSING_MAX_DPSI_DEG = 150.0
# Runway ends whose headings agree within this are "parallel".
GEOM_PARALLEL_HEADING_DEG = 15.0
# Runway-area class: both aircraft within this distance of a threshold and
# below this height above field.
GEOM_RUNWAY_AREA_MAX_DIST_NM = 1.0
GEOM_RUNWAY_AREA_MAX_HEIGHT_FT = 500.0
# Flight-phase alignment (section 6.3): an aircraft is on final to a runway
# when within this cross-track and track tolerance of the runway's
# approach course and below the height gate; a departure when aligned with
# the runway heading past the threshold and climbing.
PHASE_MAX_XTRACK_NM = 0.5
PHASE_MAX_TRACK_DELTA_DEG = 25.0
PHASE_ARRIVAL_MAX_DIST_NM = 12.0
PHASE_ARRIVAL_MAX_HEIGHT_FT = 4000.0
PHASE_DEPARTURE_MAX_DIST_NM = 8.0
PHASE_DEPARTURE_MIN_VRATE_FPM = 200.0
# Procedural classes excluded from T1 in the geometry sensitivity variant.
GEOM_PROCEDURAL_CLASSES = ("in_trail_same_runway", "parallel_runways")

# -------------------------------------------------- probability model ----
# Dead-reckoning error sampled at these horizons (s) (section 7.1).
ERROR_MODEL_TAU_STEP_S = 10.0
# Aircraft-seconds sampled per day for the error model.
ERROR_MODEL_SAMPLES_PER_DAY = 3000
# Position-noise floor for sigma at tau -> 0 (NM horizontal, ft vertical).
ERROR_MODEL_SIGMA0_NM = 0.02
ERROR_MODEL_SIGMA0_FT = 25.0
# Probability metrics are labelled "poorly calibrated" above this expected
# calibration error (section 7.4).
PROB_MAX_ECE = 0.05
PROB_RELIABILITY_BINS = 10
# Monte Carlo draws used by the unit test that checks the analytic result.
PROB_MC_DRAWS = 200000

# ----------------------------------------------------------- windows ----
WINDOW_MIN = 10.0
WINDOW_SENSITIVITY_MIN = (5.0, 15.0)
# Inner radius for the "pair-hours within 5 NM" traffic metric.
INNER_RADIUS_NM = 5.0
# Runway flow is derived from arrivals within +/- this of the window centre.
FLOW_LOOKAROUND_MIN = 30.0
# Share of arrivals a flow needs to be the window's label; else "mixed".
FLOW_DOMINANCE_SHARE = 0.70
# Runway ends within this heading difference belong to one auto-derived
# flow when the profile has no flows: block.
FLOW_AUTO_HEADING_DEG = 30.0

# ------------------------------------------------------ data quality ----
# Per-window quality (section 8.3).
QUALITY_GAP_MIN_S = 10.0             # a per-aircraft gap counts above this
# The interpolation share used for quality counts only seconds filled
# across report gaps LONGER than this. Under heavy traffic OpenSky reports
# positions every ~2 s instead of every 1 s (KMCO test day: 2-4 s gaps
# were two thirds of all interpolated seconds); filling those is
# essentially exact and must not mark busy windows as poor data.
QUALITY_INTERP_GAP_S = 4.0
QUALITY_OUTAGE_MIN_S = 60.0          # receiver outage: silence above this
QUALITY_DEGRADED_INTERP_FRAC = 0.20  # degraded above this interp share
QUALITY_BAD_INTERP_FRAC = 0.40       # bad above this interp share
QUALITY_DEGRADED_GAPS_PER_AC = 2.0   # degraded above this gaps/aircraft
QUALITY_BAD_GAPS_PER_AC = 5.0        # bad above this gaps/aircraft

# ------------------------------------------------------------ baseline ----
# Windows within this of any go-around (or ambiguous) are excluded.
BASELINE_EXCLUSION_MIN = 30.0
BASELINE_EXCLUSION_SENSITIVITY_MIN = (15.0, 60.0)
# Count metrics with fewer baseline events fall back to a pooled Poisson
# rate (section 9.2).
GLM_MIN_EVENTS = 50
# Local-hour covariate granularity (h). 1 = hour of day as in the
# formula; the fit coarsens to 3 h automatically when an hour level has no
# events at all (recorded in baseline.json).
BASELINE_HOUR_BLOCK_H = 1
BASELINE_HOUR_BLOCK_FALLBACK_H = 3
# A covariate level with fewer baseline windows than this, or with no
# events at all, cannot be estimated on its own and is merged into the
# reference level (the most frequent level of that factor).
BASELINE_MIN_LEVEL_WINDOWS = 10
# Dispersion test: negative binomial replaces Poisson when the boundary
# likelihood-ratio test is significant at this level.
BASELINE_DISPERSION_ALPHA = 0.05
# Phase I: windows beyond this predictive quantile are removed and the
# model refit; at most this many iterations and this share of windows.
PHASE1_QUANTILE = 0.999
PHASE1_MAX_ITER = 3
PHASE1_MAX_REMOVED_FRAC = 0.01
# Control limits (section 9.4).
LIMIT_WARNING = 0.95
LIMIT_ACTION = 0.99
# Non-parametric strata for the cross-check and continuous metrics:
# hour blocks (h) x traffic terciles x flow.
STRATA_HOUR_BLOCK_H = 6
# Model vs non-parametric limit agreement tolerance (relative, or 1 count).
CROSSCHECK_MAX_REL_DIFF = 0.25
# Validation (section 9.5).
VALIDATION_MIN_MONTHS_FOR_LOMO = 3
VALIDATION_PASS_95 = (0.03, 0.07)
VALIDATION_PASS_99 = (0.003, 0.02)
STABILITY_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
STABILITY_REPS = 10
STABILITY_PLATEAU_TOL = 0.05
SUFFICIENCY_MIN_WINDOWS_PER_STRATUM = 30
MIN_BASELINE_DAYS = 28
RECOMMENDED_BASELINE_DAYS = 90
BOOTSTRAP_REPS = 2000
SENSITIVITY_BOOTSTRAP_REPS = 500
RANDOM_SEED = 20250101

# -------------------------------------------------------------- events ----
# Events are go-arounds AND touch-and-goes: both put an aircraft back into
# the terminal airspace on a climb-out instead of landing, so both are
# analysed for proximity risk. They keep their own names (outcome column,
# plot titles, per-type result rows); the go-around definition itself is
# unchanged. --include-ambiguous adds ga_ambiguous.
EVENT_OUTCOMES = ("go_around", "touch_and_go")
EVENT_OUTCOMES_AMBIGUOUS = ("go_around", "touch_and_go", "ga_ambiguous")
EVENT_LABELS = {"go_around": "go-around", "touch_and_go": "touch-and-go",
                "ga_ambiguous": "ambiguous go-around"}
EVENT_LABELS_PLURAL = {"go_around": "go-arounds",
                       "touch_and_go": "touch-and-goes",
                       "ga_ambiguous": "ambiguous go-arounds"}
EVENT_ID_PREFIX = {"go_around": "ga", "touch_and_go": "tg",
                   "ga_ambiguous": "gx"}
# Per-type result rows (event_set "<type>_only") when events of more than
# one type are present.
EVENT_TYPE_BREAKDOWN = ("go_around", "touch_and_go")
# Matched landing controls: full-stop arrivals anchored at their low point,
# at most this many (random, seeded) to bound the control computation.
CONTROL_OUTCOMES = ("full_stop",)
CONTROL_MAX_EVENTS = 3000
# Superposed epoch (section 11.5): 1-min bins over this range.
EPOCH_RANGE_MIN = 30.0
EPOCH_BIN_MIN = 1.0
# Power calculation (section 11.4).
POWER_ALPHA = 0.05
POWER_TARGET = 0.80

# ------------------------------------------------------------ figures ----
FIGURE_DPI = 150
