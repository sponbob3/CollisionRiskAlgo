"""
Pipeline configuration: every threshold used by the method lives here so the
method is transparent, tunable, and citable. Units are noted per parameter.

The airport-specific values below (runways, elevation, ...) are DEFAULTS for
KDAB; running through run_analysis.py replaces them from the dataset's
airport profile (airports/<ICAO>.yaml) via profiles.load_profile(). The
threshold defaults are the `training_ga` preset, calibrated on the KDAB
training-fleet dataset; presets and per-airport overrides adjust them.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths ----
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "KDAB_2025"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"

# ---------------------------------------------------------- environment ----
# Set by profiles.load_profile(); documented in airports/<ICAO>.yaml.
PRESET = "training_ga"
# AGL reference: "flat" = field elevation everywhere; "dem" = terrain model
# sampled under each position (see terrain.py).
TERRAIN_MODE = "flat"
# True only when the dataset was queried as ARRIVALS at the airport, so a
# flight leg that ends low & descending means "landed below the local ADS-B
# coverage floor". Enables the coverage-truncation landing rules.
ASSUME_ARRIVALS_DATASET = True

# -------------------------------------------------------------- airport ----
AIRPORT_ICAO = "KDAB"
AIRPORT_LATLON = (29.18255, -81.059464)
FIELD_ELEVATION_FT = 34.0  # KDAB field elevation (ft MSL)
LOCAL_TZ = "America/New_York"

# Runway thresholds and true bearings (from ourairports.com via traffic).
# Each entry: name -> (threshold_lat, threshold_lon, true_bearing_deg)
RUNWAYS = {
    "07L": (29.172600, -81.077797, 65.236208),
    "25R": (29.184700, -81.047897, 245.250785),
    "07R": (29.176001, -81.056801, 64.902047),
    "25L": (29.179701, -81.047798, 244.906436),
    "16":  (29.190901, -81.056297, 157.164818),
    "34":  (29.175699, -81.049004, 337.168375),
}

# ------------------------------------------------- flight segmentation ----
# A gap in a single aircraft's data longer than this starts a new flight leg.
SEGMENT_GAP_MINUTES = 20.0
# Discard legs with fewer points than this (too short to contain an approach).
MIN_LEG_POINTS = 60

# ------------------------------------------------------------ smoothing ----
# Rolling median window (seconds) applied to altitude before profile logic.
ALT_SMOOTH_WINDOW_S = 7
# Rolling median window (seconds) for vertical rate.
VRATE_SMOOTH_WINDOW_S = 7

# ---------------------------------------------------- approach detection ----
# A point is "on approach" to a runway when ALL of:
#   - along-track distance to threshold within [-0.3, APPROACH_MAX_DIST_NM]
#     (negative = past the threshold, i.e. over the runway)
#   - |cross-track distance| < APPROACH_MAX_XTRACK_NM
#   - track within APPROACH_MAX_TRACK_DELTA_DEG of runway bearing
#   - AGL below APPROACH_MAX_AGL_FT
APPROACH_MAX_DIST_NM = 5.0
APPROACH_MAX_XTRACK_NM = 0.30
APPROACH_MAX_TRACK_DELTA_DEG = 25.0
APPROACH_MAX_AGL_FT = 1500.0
# Approach candidate segments separated by less than this many seconds on the
# same runway are merged (handles brief dropouts on final).
APPROACH_MERGE_GAP_S = 60.0
# Minimum duration (s) of an alignment segment to count as an approach.
APPROACH_MIN_DURATION_S = 20.0
# The aircraft must get below this AGL during the segment for it to count as
# a genuine approach attempt (filters high overflights of the extended
# centerline, e.g. crosswind legs over a parallel runway).
APPROACH_MIN_REACHED_AGL_FT = 800.0

# --------------------------------------------------- profile plateau ------
# The core low-point shape metric: duration of the contiguous stretch of
# samples within PLATEAU_BAND_FT of the profile's minimum AGL.
# V-shaped go-around: seconds.  Landing roll or low approach: tens of seconds.
PLATEAU_BAND_FT = 60.0

# ------------------------------------------------------ touchdown logic ----
# Evidence that wheels touched (any of these within the approach low segment):
TOUCHDOWN_ONGROUND = True          # onground flag observed near runway
TOUCHDOWN_AGL_FT = 25.0            # AGL at/below this ==> treated as touchdown
TOUCHDOWN_GS_KT = 50.0             # groundspeed at/below this near the runway
# GNSS altitude bias means "on the runway" can read up to ~100 ft AGL, so a
# sustained plateau at the bottom of the profile below this AGL is treated
# as a landing roll (touchdown), even with no other evidence:
TOUCHDOWN_PLATEAU_MAX_AGL_FT = 120.0
TOUCHDOWN_PLATEAU_MIN_S = 20.0
# ADS-B coverage often drops out below ~50-100 ft AGL. If the trajectory has a
# data gap longer than this while last seen below TOUCHDOWN_GAP_AGL_FT on
# short final, we treat it as a probable touchdown (aircraft went below
# coverage floor).
TOUCHDOWN_GAP_S = 12.0
TOUCHDOWN_GAP_AGL_FT = 150.0

# ADS-B coverage at KDAB typically ends between ~50 and 250 ft AGL. When an
# approach's data simply stops low and descending and the leg ends there
# (this is an arrivals dataset: legs end at the final landing), it is a
# full-stop landing that finished below the coverage floor.
END_TRUNCATED_MAX_AGL_FT = 350.0
END_TRUNCATED_MAX_GAP_S = 60.0
# Fallback for arrivals whose entire final approach happened below/outside
# alignment coverage (e.g. data lost during the base turn): leg ends within
# this distance of the field, below the AGL gate, not climbing.
TRUNCATED_FINAL_MAX_DIST_NM = 3.0
TRUNCATED_FINAL_MAX_AGL_FT = 600.0

# ------------------------------------------------- go-around definition ----
# After the low point of an approach (no touchdown), a go-around requires:
#   - sustained climb: vertical rate >= GA_CLIMB_VRATE_FPM for at least
#     GA_CLIMB_DURATION_S (within the 120 s after the low point), and
#   - altitude regain of at least GA_ALT_REGAIN_FT above the minimum AGL.
GA_CLIMB_VRATE_FPM = 300.0
GA_CLIMB_DURATION_S = 15.0
GA_ALT_REGAIN_FT = 250.0
# The low point must be below this AGL for the event to count at all
# (above it, a discontinued approach is just a normal breakoff/vector).
GA_MAX_LOW_AGL_FT = 1000.0
# The trajectory must actually be observed DESCENDING at least this much
# before the low point. Without this, an aircraft reappearing from below
# the coverage floor in its climb-out (final approach swallowed by a data
# gap - could equally be a touch-and-go) would count as a go-around.
GA_MIN_OBSERVED_DESCENT_FT = 150.0

# --------------------------------- go-around vs low-approach separation ----
# Plateau duration at the low point (see PLATEAU_BAND_FT above):
#   <= LEVEL_GA_MAX_S            -> go-around (V-shaped profile)
#   >= LEVEL_LOWAPP_MIN_S        -> low approach (U/plateau profile)
#   in between                   -> ambiguous (kept, flagged)
# Cutoffs calibrated on the Jan-2025 plateau-duration distribution, which is
# bimodal: climb-away events cluster at 5-20 s (flare + rotate), landing
# rolls at 35-60 s.
LEVEL_GA_MAX_S = 20.0
LEVEL_LOWAPP_MIN_S = 30.0

# ------------------------------------------------ chain post-processing ----
# An "unresolved" aligned segment followed by another aligned segment on the
# same leg within this gap is one continued approach (S-turn, sidestep,
# realignment), not an independent attempt.
CONTINUED_APPROACH_MAX_GAP_S = 240.0
