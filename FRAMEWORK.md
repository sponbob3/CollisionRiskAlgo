# Proximity Risk After Go-Arounds — Method and Build Framework

This document is the specification for the proximity-risk pipeline in this
repository. It defines the metric, the baseline and noise framework, the
statistical tests, the integration with the go-around detector, the folder
layout, the command-line interface, and the figures. Where a value is given
it is the default; every value is a named, documented config parameter.

---

## 1. Research question

> Does proximity risk in the terminal airspace around an airport increase
> after a go-around, relative to (a) the same airspace immediately before the
> go-around and (b) the airport's own baseline behaviour?

Two scopes are always reported side by side:

- **Airspace-wide** — all aircraft pairs inside the study volume.
- **Go-around-involved** — pairs in which one aircraft is the go-around
  aircraft.

Design principles (inherited from the go-around pipeline):

- Rule-based and explainable: pandas / NumPy / SciPy / statsmodels, no
  machine learning. Every threshold is a named parameter in `config.py`.
- **Universal**: nothing is airport-specific in code. Everything
  airport-specific comes from `airports/<ICAO>.yaml`.
- One entry point (`run_proximity.py`) runs everything.
- Every run writes a fresh numbered folder with a full provenance record.
- Rigor over convenience: a window with zero encounters is a valid result;
  a missed encounter is not. Data-quality problems are detected and
  reported, never silently absorbed.

---

## 2. Terminology

| term | meaning |
|---|---|
| **proximity risk** | the rate and severity of aircraft pairs coming within defined separation thresholds (observed or predicted). A surrogate safety measure — this is *not* a probability of collision. |
| **study volume** | the cylinder around the airport in which pairs are evaluated (§3). |
| **tier** | one of three nested separation thresholds (§5). |
| **encounter** | one contiguous episode of one aircraft pair inside (or predicted to enter) a tier's zone; counted once regardless of duration (§6). |
| **window** | a fixed time interval over which metrics are aggregated (default 10 min). |
| **exposure** | pair-hours: the summed time that aircraft pairs coexist airborne inside the study volume. The denominator of every rate. |
| **baseline** | the airport's expected behaviour in windows unaffected by go-arounds, conditional on traffic and operating conditions (§9). |
| **event** | one detected go-around or touch-and-go (each keeps its own name), anchored at its climb start `t0`. |
| **pre / post window** | `[t0 − W, t0)` and `[t0, t0 + W)`, W = 10 min (sensitivity 5, 15). |

"Collision risk" is not used in outputs; the literature definition (the
probability that two aircraft come into physical contact) requires sample
sizes and physical-contact models outside this study's scope (see the
review of CRM methods, ref. [2]).

---

## 3. Study volume

### 3.1 Definition

A vertical cylinder centred on the airport reference point:

- **Radius**: 10 NM (`VOLUME_RADIUS_NM`), sensitivity 5 NM.
- **Floor**: airborne aircraft only (§4.3).
- **Ceiling**: the airport's controlled-airspace ceiling (§3.2), subject to
  the data-driven adequacy check (§3.3).

### 3.2 Ceiling basis — Class C design

The ceiling is anchored on the airport's charted controlled-airspace
ceiling. For Class C airspace (AIM 3-2-4; FAA JO 7400.11) the standard shape
is:

- surface area: 5 NM radius, surface to 4,000 ft above airport elevation;
- shelf area: 5–10 NM radius, ~1,200 ft AGL to the same 4,000 ft ceiling.

The 10 NM radius therefore matches the outer edge of standard Class C, and
the ceiling is the Class C ceiling. The airport profile carries it:

```yaml
airspace:
  class: C                    # informational (B / C / D)
  ceiling_ft_msl: 4600        # CHARTED ceiling from the sectional; verify
  radius_nm: 10
  ceiling_mode: auto          # auto | fixed   (see 3.3)
```

`new-airport` pre-fills `ceiling_ft_msl` as field elevation + 4,000 ft and
marks it "verify against the sectional chart". Internally the ceiling is
converted to feet above field elevation.

Input sanity check (hard error): the ceiling must lie between 1,500 and
12,000 ft above field elevation; anything else is treated as an input
mistake and the run stops with a clear message.

### 3.3 Ceiling adequacy check and automatic override

The charted value is a regulatory boundary, not a guarantee that it contains
the traffic that matters. Every run tests it against the data:

1. **Go-around containment.** For every go-around event, take the maximum
   height (above field) of the go-around aircraft during its post window
   while within the radius. Compute the 95th percentile across events
   (`P95_GA`).
2. **Encounter spill-over.** Encounters are always computed in an
   *extended* volume (ceiling + 2,000 ft, `CEILING_BUFFER_FT`) and tagged
   inside/outside the nominal ceiling. Compute the share of
   go-around-involved T1 encounters whose closest point lies above the
   nominal ceiling (`SPILL`).

Decision (`ceiling_mode: auto`, the default):

- If `P95_GA` ≤ ceiling and `SPILL` ≤ 10 %: the charted ceiling is used.
- Otherwise the effective ceiling is raised to
  `max(ceiling, round_up_500(P95_GA))`, capped at 10,000 ft above field; if
  `SPILL` is still > 10 % it is raised further in 500 ft steps until it is
  ≤ 10 % or the cap is reached.
- The effective ceiling can never go below the charted value.
- Any override is printed on the console, written to `ceiling_check.csv`,
  shown on the first page of `summary.pdf`, and recorded in
  `run_config.txt`.

`ceiling_mode: fixed` uses the charted value unconditionally but still runs
and reports the check.

The horizontal radius is not auto-changed; its containment (share of
go-around post-window time spent beyond 10 NM) is reported alongside.

---

## 4. Data preparation

### 4.1 Input

`datasets/<ICAO>_<label>/` holds one file per day (`.parquet` preferred,
`.csv` accepted) of OpenSky-style state vectors containing **all traffic**
around the airport, with the columns used by the go-around pipeline:

`timestamp, icao24, callsign, latitude, longitude, altitude (barometric),
geoaltitude, vertical_rate, groundspeed, track, onground`

Daily files may sit directly in the dataset folder or in subfolders (e.g.
one per month); hidden files are ignored, and two files with the same day
name are an error. When a file also has a position-time column
(`last_position`, as delivered by OpenSky), it is used by the freshness
filter (§4.2).

### 4.2 Cleaning

Per aircraft, per day:

- **position freshness** (both stages, go-around and proximity): OpenSky
  state vectors repeat an aircraft's last known position for up to 300 s
  after its last position report (on the first KMCO test day 58 % of rows
  were such frozen copies, typically right after an arrival dropped below
  the coverage floor, where a frozen "ghost" would otherwise sit on the
  final approach path). Only rows that carry a new position report are
  kept, re-timed to the time of that report; reports older than 10 s at
  the row time are dropped;
- drop rows without position; drop duplicate timestamps;
- drop position jumps (implied speed > 600 kt between consecutive fixes);
- flight legs split on 20-min gaps (same rule as the go-around pipeline);
- smooth vertical rate with a short rolling median (7 s, as in the
  go-around pipeline); positions are not smoothed;
- geometric altitude that disagrees with barometric altitude by more than
  1,000 ft beyond the leg's median offset is treated as missing (isolated
  GNSS spikes); height above field then uses the re-referenced barometric
  fallback.

Altitude references:

- **Vertical separation between two aircraft uses barometric altitude**
  when both have it: it is the reference ATC separates on, and the shared
  pressure-setting bias cancels in the difference. Fallback: geometric
  altitude for both aircraft of the pair (never mixed within a pair).
- **Height above field** (volume ceiling, airborne test) uses geometric
  altitude minus field elevation, with the go-around pipeline's barometric
  re-referencing fallback.

### 4.3 Airborne filter

A sample is airborne when `onground` is false **and** groundspeed ≥ 50 kt
(`AIRBORNE_MIN_GS_KT`). Pairs in which either aircraft is on the ground are
never evaluated. Low-altitude airborne pairs near the runway are kept and
tagged (`runway_area` geometry, §6.3), because runway-area proximity is
often exactly what precedes or follows a go-around.

### 4.4 Time grid and interpolation

- All aircraft are placed on a common 1-second UTC grid.
- Linear interpolation fills gaps ≤ 10 s (`MAX_INTERP_GAP_S`). Longer gaps
  are **not** filled; the aircraft is absent for those seconds and the gap
  is logged.
- Each grid sample carries a flag `interpolated` (true/false) so data
  quality can be measured per window (§8.3).

### 4.5 Coordinates

Positions are projected to a local azimuthal-equidistant plane centred on
the airport reference point (x east, y north, NM). Distortion within the
volume is negligible. Velocity vectors come from groundspeed/track and
vertical rate.

### 4.6 Role of the traffic library

Used for: the `Flight`/`Traffic` containers where convenient, resampling,
projection (via pyproj), airport and runway data (`new-airport`), and an
independent cross-check of closest-point-of-approach results
(`validate_cpa.py`, analogous to the go-around pipeline's
`validate_traffic.py`). The core pairwise geometry is implemented
explicitly in NumPy so that every step is transparent and testable.

---

## 5. Separation tiers

Three nested zones. A pair is *inside* a tier when its horizontal distance
is below H **and** its vertical separation is below V.

| tier | H | V | basis |
|---|---|---|---|
| **T1 — separation proximity** | 3 NM | 1,000 ft | terminal radar separation minimum and standard vertical separation (FAA JO 7110.65) |
| **T2 — close proximity** | 0.5 NM | 500 ft | 500 ft = Class C IFR/VFR vertical separation (FAA JO 7110.65, Class C service); 0.5 NM close-proximity threshold |
| **T3 — NMAC** | 500 ft | 100 ft | near mid-air collision definition (FAA; RTCA TCAS/DAA NMAC cylinder) |

Each tier is evaluated two ways:

- **observed** — the pair actually penetrated the zone;
- **predicted** — a straight-line projection of both aircraft (Paielli &
  Erzberger style, ref. [4]) enters the zone within the lookahead horizon
  `T_LOOKAHEAD_S` = 120 s (sensitivity 60, 180), while the pair is not yet
  inside it. This corresponds to Ghorbani's "projected near mid-air
  collision" concept (ref. [3]).

The normalised separation `s = max(d_h / H, |d_z| / V)` is recorded for T1;
`s < 1` means inside the zone. It provides a continuous severity measure.

Tier thresholds are config parameters and are swept in the sensitivity
analysis (§12). Before publication, the citations for T1–T3 must be
verified against the current editions of the cited orders.

---

## 6. Pairwise geometry and encounters

### 6.1 Candidate pairs

At each second, every pair of airborne aircraft both inside the (extended)
volume is a candidate. To keep this efficient, pairs are pruned with a
spatial grid / KD-tree to those within
`H_T1 + 2 · v_max · T_LOOKAHEAD` horizontally (v_max = 350 kt,
`PRUNE_VMAX_KT`); pruning must never drop a pair that could enter T1
within the lookahead (unit-tested).

### 6.2 Zone-entry intervals (the core computation)

For a pair with relative position **r** (x, y), relative horizontal velocity
**v**, vertical separation `dz` and relative vertical rate `vz`, over a time
span `[0, τ_max]`:

- horizontal inside-interval: solve `|r + v·t| < H` (a quadratic in t);
- vertical inside-interval: solve `|dz + vz·t| < V` (linear in t);
- the pair is inside the zone during the **intersection** of the two
  intervals, clipped to `[0, τ_max]`.

Uses:

- **observed**, with `τ_max` = the grid step (1 s): detects penetrations
  that occur *between* samples, so a fast pass between two fixes is never
  missed;
- **predicted**, with `τ_max = T_LOOKAHEAD_S`: time-to-entry, predicted
  minimum horizontal distance and vertical separation at closest approach.

This is the analytic interval method used in detect-and-avoid "well clear"
evaluation; it avoids the error of evaluating horizontal and vertical
closest approach at different times.

### 6.3 Encounter episodes

Per pair and per tier (observed and predicted separately), consecutive
flagged seconds form an episode; episodes separated by ≤ 10 s
(`EPISODE_MERGE_GAP_S`) merge. Each encounter row records:

- pair (icao24, callsign ×2), tier, observed/predicted;
- start, end, duration, time of minimum separation;
- minimum horizontal distance, vertical separation at that time, minimum
  `s`;
- position (lat/lon/height) of both aircraft at minimum separation, and
  whether that point is inside the nominal ceiling;
- **geometry class**, from runway alignment (reusing the go-around
  pipeline's runway-frame geometry) and relative track angle Δψ:
  - `in_trail_same_runway` — both aligned with the same runway end
    (arrival or departure), Δψ < 30°;
  - `parallel_runways` — aligned with different runway ends of the same
    heading (within 15°);
  - `crossing` — 30° ≤ Δψ ≤ 150°;
  - `head_on` — Δψ > 150°;
  - `runway_area` — both within 1 NM of a threshold and below 500 ft;
  - `other`;
- flight phase of each aircraft (arrival on final to runway X / departure
  from runway X / other);
- whether either aircraft is a go-around aircraft, and if so its event id
  and whether the encounter falls in its pre or post window.

An encounter is attributed to the window containing its time of minimum
separation.

Geometry classes are never used to drop encounters from the primary
metric. They are reported as breakdowns, and a sensitivity variant
excludes `in_trail_same_runway` and `parallel_runways` from T1 (these are
often procedurally separated operations).

---

## 7. Probability model (Paielli & Erzberger)

### 7.1 Error model estimated from the data

Straight-line prediction error is measured, not assumed. For a large random
sample of aircraft-seconds in the volume (all days of the dataset), predict
the position at `t + τ` by dead reckoning for τ = 10, 20, …, T_LOOKAHEAD s
and compare with the actual position. Decompose the error into along-track,
cross-track and vertical components and estimate robust standard deviations
σ_a(τ), σ_c(τ), σ_z(τ) (MAD-based) — separately for arrival / departure /
other phases. Store as `error_model.csv` with the baseline.

### 7.2 Conflict probability

For each pair-second with a predicted closest approach within the lookahead,
the relative position at closest approach is treated as Gaussian with
covariance built from both aircraft's σ_a, σ_c (rotated to each aircraft's
track) and σ_z at that lookahead time. Following Paielli & Erzberger, the
combined covariance is transformed to make horizontal uncertainty circular,
and the probability that the relative trajectory passes through the
(transformed) zone is computed analytically; the vertical probability
`P(|dz| < V)` is multiplied in. Implementation must be verified against
Monte Carlo on a sample of geometries (unit test, agreement within 0.01).

### 7.3 Metric

Per window and tier: **expected conflicts** = sum over pairs of the maximum
conflict probability that pair reached in the window.

### 7.4 Showing the model works (required output)

- **Reliability diagram**: bin pair-seconds by predicted probability
  (deciles); plot mean predicted probability vs the observed fraction that
  actually entered the zone within the lookahead.
- **Brier score** and **Brier skill score** versus a constant-rate
  reference; **expected calibration error**.
- **Error-growth plot**: σ_a, σ_c, σ_z vs τ.

If the expected calibration error exceeds 0.05 (`PROB_MAX_ECE`) the
probability metrics are labelled "poorly calibrated" in every table and in
the summary, and are not used for any headline claim.

---

## 8. Windows, exposure, and data quality

### 8.1 Window grid

Two sets of windows are computed:

- **clock windows**: contiguous 10-min windows over every day of the
  dataset (for the baseline);
- **event windows**: pre and post windows for each go-around (§10).

### 8.2 Per-window metrics

- traffic: unique aircraft, aircraft-seconds, **pair-hours** (exposure),
  pair-hours within 5 NM;
- encounter counts per tier × {observed, predicted}, with geometry-class
  breakdown;
- minimum normalised separation `s_min`;
- expected conflicts per tier (§7.3);
- encounter rates = counts / pair-hours.

### 8.3 Data quality per window

- `interp_fraction`: share of aircraft-seconds interpolated across report
  gaps longer than 4 s. Filling the 1–4 s gaps that OpenSky produces
  under heavy traffic is essentially exact and does not count (counting
  it marked the busiest windows of the first KMCO test day as bad data);
- `gap_count`: number of per-aircraft gaps > 10 s;
- `outage`: any interval > 60 s with no messages from any aircraft while
  aircraft were present before and after (receiver outage);
- `quality` = `good` / `degraded` / `bad` from these, with thresholds in
  config.

`bad` windows are excluded from the baseline and flag any pre/post window
they belong to. Counts by quality class are reported in every summary.

### 8.4 Operating covariates (per window)

- local hour of day; weekday / weekend;
- month (season);
- **runway flow configuration** — the set of runway ends used for arrivals
  (and departures) within ±30 min, derived from the go-around pipeline's
  `all_approaches.csv` (and departure alignment), mapped to a small number
  of flow labels per airport (e.g. "north flow" / "south flow" / "mixed");
- traffic level (unique aircraft, pair-hours).

---

## 9. Baseline and noise framework

Ghorbani et al. (ref. [3]) compare a monitored period against historical
danger-zone counts with a noise filter. This pipeline keeps that concept and
makes it more robust in seven ways.

### 9.1 All data, clean exclusions

- Every clock window of every day in the dataset is a candidate.
- Excluded: windows overlapping ±30 min (`BASELINE_EXCLUSION_MIN`) of any
  go-around or ambiguous go-around; `bad`-quality windows; windows with
  zero pair exposure (kept in counts, excluded from rate models).
- The exclusion log (how many windows removed, for which reason) is part of
  the baseline report.

### 9.2 Conditional expectation instead of a single average

For each count metric, a **negative binomial regression** with exposure
offset:

```
log E[count] = log(pair_hours) + hour + daytype + month + flow + β·log(n_aircraft)
```

- Negative binomial captures over-dispersion (the "noise"); Poisson is used
  only if dispersion is not significant.
- This yields, for *any* window, the expected count and its full predictive
  distribution given that window's conditions — a risk-adjusted control
  chart (Montgomery, ref. [7]). Busy periods are compared with busy
  periods.
- **Sparse metrics** (e.g. T3, which may be almost always zero): if the
  baseline contains fewer than 50 encounters of that metric
  (`GLM_MIN_EVENTS`), the model falls back to a pooled exposure-adjusted
  Poisson rate with exact limits. The fallback is recorded.
- **Continuous metrics** (`s_min`, expected conflicts): empirical
  conditional quantiles within strata (hour block × traffic tercile ×
  flow).
- **Cross-check**: a non-parametric stratified-quantile baseline is
  computed for the count metrics too; model-based and non-parametric
  control limits must agree (reported; disagreement flagged).

### 9.3 Phase I / Phase II

- **Phase I** (baseline establishment): fit, then remove windows beyond the
  99.9 % predictive limit and refit; at most 3 iterations and at most 1 % of
  windows removed. Removed windows are listed in
  `baseline/phase1_removed.csv` with their values (they are reported, not
  hidden).
- **Phase II** (monitoring): the frozen model scores pre, post and control
  windows.

### 9.4 Scoring a window

For a window with covariates X and observed count y:

- expected count μ(X) and predictive distribution;
- upper-tail mid-p-value, and a **randomized quantile residual** z
  (Dunn & Smyth, ref. [8]) — a standard-normal score under the baseline;
- in-control flag: within the 95 % (warning) and 99 % (action) limits.

### 9.5 Proving the baseline is accurate (validation)

- **Out-of-sample calibration**: leave-one-month-out (or leave-one-week-out
  if the dataset is shorter than 3 months). Fit on the rest, score the
  held-out windows. Under a correct baseline ~5 % exceed the 95 % limit and
  ~1 % the 99 % limit. Report observed exceedance rates with binomial CIs;
  pass criteria: 95 % limit exceedance in [3 %, 7 %], 99 % in [0.3 %, 2 %].
- **PIT histogram / z-score QQ-plot** of held-out windows (should be
  uniform / on the diagonal).
- **Stability curve**: refit on random subsets of 10 %, 20 %, …, 100 % of
  days (bootstrap by day); plot the baseline rate and 95 % limit versus the
  number of days used. A plateau demonstrates there is enough data; the
  report states the number of days at which estimates stabilise within
  ±5 %.
- **Data sufficiency table**: windows and encounters per stratum; strata
  with < 30 windows are flagged; overall minimum 28 days of data (hard
  warning below), recommended ≥ 90 days.
- **Uncertainty**: all baseline confidence intervals use a **day-block
  bootstrap** (neighbouring windows are correlated).

The baseline report prints an overall verdict: `VALID`, `VALID WITH
WARNINGS` (listing them), or `NOT VALID` (analysis still runs but every
output is labelled accordingly).

### 9.6 Storage and reuse

The baseline is stored per airport and dataset:

```
output/<ICAO>/proximity_risk/baseline/<label>/
  baseline.json          model coefficients, dispersion, limits, fallbacks,
                         data date range, n days/windows, parameter hash
  baseline_windows.parquet
  error_model.csv
  phase1_removed.csv
  validation/            calibration tables + figures
  baseline_report.pdf
```

It is reused automatically when the dataset contents and all
baseline-relevant parameters (hash) match; otherwise it is rebuilt.
`--rebuild-baseline` forces a rebuild; `--baseline-from <ICAO>_<label>`
scores events against another dataset's baseline for the same airport.

### 9.7 Matched landing controls (go-around-involved scope)

The airspace-wide baseline has no natural equivalent for "encounters
involving one specific aircraft". For the go-around-involved scope the
control group is **normal arrivals**: approaches classified `full_stop` in
`all_approaches.csv`, anchored at their low point, with the same pre/post
windows, excluding any within ±30 min of a go-around. Encounters involving
the control aircraft are counted exactly as for go-around aircraft. The
same regression/validation machinery (§9.2–9.5) is applied, with
covariates from the window.

---

## 10. Events: go-arounds and touch-and-goes

- Source: the vendored go-around pipeline (§13), run on the same dataset.
- Primary set: `outcome` in {`go_around`, `touch_and_go`}. A touch-and-go
  puts an aircraft back into the terminal airspace on a climb-out just as
  a go-around does, so both are analysed for proximity risk. Each event
  keeps its own name (its `outcome`, its event id prefix `ga_` / `tg_`,
  its plot title and file name); the go-around definition is unchanged.
  When both types occur, every result is also given per type
  (`event_set` = `go_around_only`, `touch_and_go_only`). Sensitivity:
  include `ga_ambiguous`.
- Anchor `t0 = climb_start_utc` (fallback `t_low_utc`).
- Windows: pre `[t0 − 10 min, t0)`, post `[t0, t0 + 10 min)`; sensitivity
  W = 5 and 15 min.
- **Clusters**: events within W of each other form a cluster. The
  primary analysis uses the first event of each cluster (independent
  events), with cluster size recorded; a sensitivity variant drops all
  clustered events.
- **Contaminated pre windows**: if another event occurred in the W
  before `t0`, the pre window is flagged.
- The baseline exclusion (§9.1) and the matched landing controls (§9.7)
  keep away from touch-and-goes as well as go-arounds.
- **Quality**: events whose pre or post window is `bad` are excluded and
  listed.

---

## 11. Statistical analysis

### 11.1 Per event

For pre and post windows, each metric: observed, expected under the
baseline, O/E, z-score, p-value, in-control flag; plus post − pre
difference. Written to `go_around_risk.csv`.

### 11.2 Equilibrium check (pre within baseline)

Required, reported on page 1 of the summary:

- share of events whose pre window is in control (expected ≈ 95 %), with
  binomial CI;
- **standardised incidence ratio** of pre windows, SIR_pre = Σ observed /
  Σ expected, with day-block bootstrap CI — should include 1;
- KS test of pre-window z-scores against N(0, 1) (and PIT histogram).

Verdict line: `EQUILIBRIUM: PASS / FAIL`. The primary results are always
computed twice: all events, and only events with in-control pre windows. If
the conclusions differ, the summary says so.

### 11.3 Primary comparisons

1. **Post vs pre** (paired): conditional negative binomial / Poisson model
   with event fixed effects and exposure offset →
   **rate ratio (IRR) post/pre** with day-clustered bootstrap CI; Wilcoxon
   signed-rank test on (post − pre) z-scores as a non-parametric
   confirmation.
2. **Post vs baseline**: **SIR_post** = Σ observed / Σ expected in post
   windows, with CI; Wilcoxon one-sample test of post z-scores against 0.

Both are run for both scopes (airspace-wide; go-around-involved against
matched landing controls).

### 11.4 Endpoints

- **Primary**: T1 encounters (observed ∪ predicted) — airspace-wide and
  go-around-involved.
- **Secondary**: T2, T3 (observed and predicted separately), expected
  conflicts (if calibrated), `s_min`. Holm correction across secondary
  endpoints.
- For every endpoint the **minimum detectable effect** (80 % power, α =
  0.05) given the number of events is reported, so a null result can be
  interpreted.

### 11.5 Time-course (superposed epoch)

O/E of encounter rate in 1-min bins from −30 to +30 min around `t0`,
averaged across events, with a day-block bootstrap band. The baseline is
O/E = 1 by construction.

---

## 12. Sensitivity analysis

One-at-a-time variations, each re-running §11 and reporting IRR and SIR
with CIs:

| parameter | values |
|---|---|
| window W | 5, **10**, 15 min |
| radius | 5, **10** NM |
| ceiling | 2,500 ft AGL, **effective ceiling**, 5,000, 10,000 ft AGL |
| lookahead | 60, **120**, 180 s |
| tier thresholds | each H and V ×0.75 and ×1.25 |
| event set | strict, + ambiguous, no clusters, in-control pre only |
| geometry | all, excluding procedural classes (T1) |
| baseline exclusion | ±15, **±30**, ±60 min |

Encounters are computed once in the widest volume, longest lookahead and
loosest thresholds, then filtered, so sweeps do not repeat the pairwise
computation.

---

## 13. Integration with the go-around detector

- The package `goaround_pipeline/` is vendored **unchanged** from
  `github.com/sponbob3/GoAroundAlgo`, commit
  `6d1aef96f263bbbcaeb35c7e0347ff9291c79270`. The source commit is recorded
  in `goaround_pipeline/VENDORED_FROM.txt`. It is not edited; all
  integration goes through `proximity_pipeline/goaround_adapter.py`.
- The adapter reproduces GoAroundAlgo's `run_analysis.py` behaviour: load
  the airport profile, set the output directory, dump `run_config.txt`,
  run the pipeline, write summaries (and optional report / calibration).
  With the two deliberate additions below switched off, outputs are
  identical to a GoAroundAlgo run on the same data (test 8).
- Airport profiles use the same YAML format; the `airspace:` block (§3.2)
  and an optional `flows:` block (§8.4) are additions that the go-around
  loader ignores.
- Input handling for BOTH stages comes from the adapter: `data_files()`
  (daily files in subfolders) and `load_day()` (the vendored loader plus
  the position-freshness filter, §4.2). The go-around stage gets them for
  the duration of its run, without editing the vendored files. On files
  without a position-time column `load_day()` is the vendored loader
  exactly.
- **Hidden low point = go-around** (go-around stage only;
  `GA_HIDDEN_LOW_POINT_AS_GO_AROUND`). Where ADS-B coverage ends above
  the runway (first KMCO test day: ~530 ft above field), a go-around
  started below the coverage floor shows as a data gap at the bottom of
  the approach followed by a climb, and the go-around pipeline measures
  the gap as level time at the low point, calling the climb-away a
  `low_approach` or `ga_ambiguous`. The adapter's classifier hook
  subtracts the hidden time (report gaps longer than 5 s inside the
  plateau, from the last descent into the low band to the climb start)
  from the measured plateau; if the level time actually observed is
  within the go-around cutoff (`LEVEL_GA_MAX_S`, 20 s), the climb-away is
  a `go_around`. A low pass actually seen level for longer stays a
  `low_approach`, so training airports with genuine low approaches keep
  them. A touch-and-go whose touchdown is hidden below the coverage floor
  becomes a go-around; both are events (§10), so the proximity analysis
  is unaffected. Reclassified approaches carry `reclassified_from` and
  `low_point_hidden_s` in `all_approaches.csv` and
  `go_around_events.csv`. On synthetic traffic with every sample below
  600 ft within 5 NM removed, the rule recovered all 11 true go-arounds
  (4 without it, 7 mislabelled low approaches) with no false positives,
  and it changes nothing on data with full coverage.
- Updating the vendored copy later = replace the folder, update
  `VENDORED_FROM.txt`, re-run the test suite.

---

## 14. Command-line interface

One file runs everything:

```bash
python run_proximity.py KBNA_2025
```

This (1) runs go-around detection → `output/KBNA/goaround/run_NN/`,
(2) builds or reuses the baseline, (3) computes encounters and window
metrics, (4) analyses every event, (5) runs the sensitivity sweep,
(6) writes tables, figures, per-event plots and `summary.pdf` →
`output/KBNA/proximity_risk/run_NN/` (same NN).

| option | effect |
|---|---|
| `--goaround-run N` | reuse existing `goaround/run_N` instead of re-detecting |
| `--rebuild-baseline` | force a baseline rebuild |
| `--baseline-only` | build/validate the baseline and stop |
| `--baseline-from ICAO_label` | score against another dataset's baseline |
| `--window MIN` | pre/post window length (default 10) |
| `--include-ambiguous` | primary event set includes `ga_ambiguous` |
| `--no-sensitivity` | skip the sweep |
| `--no-plots` / `--no-event-plots` | skip all plots / per-event plots only |
| `--goaround-report` / `--calibrate` | passed to the go-around stage |
| `--limit N` | first N days only (testing) |
| `--workers N` | parallel day processing (default: CPU count − 1) |

Sub-commands:

```bash
python run_proximity.py new-airport KMCO KMCO_2025Q1  # profile only (optional)
python run_proximity.py check-data KMCO_2025Q1        # data audit only
```

**Automatic airport profiles** (`proximity_pipeline/profile_setup.py`).
When `airports/<ICAO>.yaml` does not exist, the run (or `check-data`)
creates it with no manual editing:

- geometry (name, reference point, elevation, open runways with threshold
  coordinates; true bearing = geodesic azimuth between the two ends) from
  the OurAirports database (ourairports.com, falling back to its GitHub
  mirror), cached in `airports/.ourairports/`;
- `timezone` from the airportsdata table (offline, pure Python;
  timezonefinder on the coordinates as an optional fallback);
- `preset` from the data: median groundspeed on final approach
  (descending within 4 NM, 200–1,500 ft above field, up to 5 days spread
  over the dataset); ≥ 100 kt → `air_carrier`, else `training_ga`.
  Without data: the OurAirports airport type;
- `assume_arrivals_dataset` from the data: if fewer than half of the
  tracks that end low (< 1,000 ft) near the field (< 3 NM) show on-ground
  samples at their end, landings are not seen on the ground and a track
  ending low near the field is treated as a landing;
- the ADS-B coverage floor near the field (P90 of those last-seen heights
  + 100 ft, rounded up to 50 ft): when above the go-around pipeline's
  landing-truncation heights, `END_TRUNCATED_MAX_AGL_FT` and
  `TRUNCATED_FINAL_MAX_AGL_FT` are raised to it in the profile's
  `overrides` (capped at 1,200 ft). On the first KMCO test day coverage
  ended at ~530 ft above field and every landing was `unresolved` without
  this;
- the `airspace:` block (field elevation + 4,000 ft, §3.2).

Every detected value is printed and written into the profile with the
evidence it came from. The profile is the single place to change any of
them and is never overwritten (delete it to regenerate). An existing
profile without an `airspace:` block gets the default block added.

`check-data` reports: days found, columns present, the share of stale
(repeated-position) rows removed, per-day message counts,
aircraft counts, coverage gaps/outages, altitude-reference availability, and
whether the data volume meets the baseline minimum.

**Terminal commands PDF**: `TERMINAL_COMMANDS.pdf` at the repo root lists
every command and option with examples (setup, venv, running, reusing runs,
rebuilding baselines, new airport, data audit, tests). It is generated by
`tools/make_terminal_commands_pdf.py` (reportlab) from a single source so it
stays in sync with the CLI; regenerate it whenever the CLI changes.

---

## 15. Repository layout

```
run_proximity.py              <- the only file you run
FRAMEWORK.md                  <- this document
README.md
TERMINAL_COMMANDS.pdf
requirements.txt
airports/<ICAO>.yaml          <- airport profiles (+ airspace, flows)
datasets/<ICAO>_<label>/      <- daily parquet/csv (gitignored)
output/                       <- runs (gitignored)
goaround_pipeline/            <- vendored, unchanged (+ VENDORED_FROM.txt)
proximity_pipeline/
  config.py                   every threshold, documented
  airspace.py                 volume, ceiling check
  loading.py                  cleaning, 1 s grid, projection, quality flags
  pairs.py                    pruning, zone-entry intervals, encounters
  geometry.py                 geometry classes, flight phase
  probability.py              error model, P&E conflict probability
  windows.py                  window metrics, covariates, flows
  baseline.py                 regression baseline, Phase I/II, validation
  events.py                   event windows, clusters, controls
  stats.py                    IRR, SIR, equilibrium, power, epoch
  sensitivity.py
  goaround_adapter.py
  viz.py                      all figures (shared style)
  report_pdf.py               summary.pdf, baseline_report.pdf
  validate_cpa.py             cross-check vs traffic
tools/make_terminal_commands_pdf.py
tests/                        pytest suite (§18)
```

Output layout:

```
output/KBNA/
  goaround/run_NN/                 go-around pipeline outputs (§13)
  proximity_risk/
    baseline/<label>/              stored baseline (§9.6)
    cache/<label>/<day>.parquet    per-day pairwise results (param-hashed)
    run_NN/
      run_config.txt               every resolved parameter + dataset + versions
      ceiling_check.csv
      data_quality.csv
      window_metrics.parquet       every clock window
      encounters.parquet (+ .csv)  every encounter
      go_around_risk.csv           one row per event (§11.1)
      results_primary.csv          IRR / SIR per endpoint & scope
      equilibrium.csv
      sensitivity.csv
      figures/
      events/                      one PNG per go-around / touch-and-go
      summary.pdf
```

---

## 16. Figures

Shared style with the go-around pipeline (clean, light background, minimal
chart junk). Each figure answers one question; no figure has more than four
panels; every axis is labelled with units; colours are consistent across
all figures (pre, post, baseline, and the three tiers each have one fixed
colour).

### 16.1 Per event (`events/<outcome>_<t0>_<callsign>_<runway>.png`)

One page, three rows:

1. **Maps, pre | post** side by side, same extent and scale: tracks of all
   aircraft in the window (grey), the go-around aircraft highlighted,
   encounter pairs linked at their closest point and coloured by tier,
   5 and 10 NM range rings, runways.
2. **Timeline** from −W to +W: aircraft count; minimum normalised
   separation `s` over time with tier bands shaded; a bar per active
   encounter; vertical line at `t0`.
3. **Against the baseline**: pre and post values of the primary metric as
   two markers on the baseline predictive distribution for their
   conditions, with 95 % / 99 % limits; a small table of observed, expected,
   O/E and z for pre and post.

### 16.2 Study-level (`figures/`)

1. **Superposed epoch** (headline figure): O/E vs minutes from `t0`, −30 to
   +30, with bootstrap band and the O/E = 1 line.
2. **Forest plot**: IRR (post/pre) and SIR (pre, post) with CIs for every
   endpoint and both scopes, reference line at 1.
3. **Equilibrium**: histogram of pre-window z-scores over the N(0, 1)
   curve, and the in-control share with CI.
4. **Control chart**: z-score of every clock window over the dataset with
   control limits; go-around windows marked.
5. **Where encounters happen**: density maps (same colour scale) of
   encounter locations — baseline vs post-go-around.
6. **Sensitivity**: IRR and SIR with CIs across every sweep setting.
7. **Ceiling check**: distribution of go-around max post-window height with
   the charted and effective ceilings.

### 16.3 Baseline report (`baseline/<label>/validation/`)

Out-of-sample exceedance vs nominal, PIT histogram / QQ-plot, stability
curve, data-sufficiency table; probability-model reliability diagram and
error-growth curves.

---

## 17. Configuration defaults (summary)

| parameter | default |
|---|---|
| `VOLUME_RADIUS_NM` | 10 |
| ceiling | profile `ceiling_ft_msl` (default field + 4,000 ft), `ceiling_mode: auto` |
| `CEILING_BUFFER_FT` | 2,000 |
| `CEILING_SPILL_MAX` | 0.10 |
| `CEILING_CAP_FT_AGL` | 10,000 |
| `AIRBORNE_MIN_GS_KT` | 50 |
| `MAX_INTERP_GAP_S` | 10 |
| `MAX_IMPLIED_SPEED_KT` | 600 |
| `POSITION_FRESHNESS_FILTER` / `MAX_POSITION_AGE_S` | on / 10 |
| `GEO_BARO_MAX_DEV_FT` | 1,000 |
| `QUALITY_INTERP_GAP_S` | 4 |
| `EVENT_OUTCOMES` | go_around, touch_and_go |
| tiers (H, V) | T1 3 NM/1,000 ft · T2 0.5 NM/500 ft · T3 500 ft/100 ft |
| `T_LOOKAHEAD_S` | 120 |
| `EPISODE_MERGE_GAP_S` | 10 |
| `WINDOW_MIN` | 10 |
| `BASELINE_EXCLUSION_MIN` | 30 |
| `GLM_MIN_EVENTS` | 50 |
| `PROB_MAX_ECE` | 0.05 |
| `MIN_BASELINE_DAYS` | 28 (warning), 90 recommended |
| `BOOTSTRAP_REPS` | 2,000 |

---

## 18. Verification and tests (`tests/`, pytest)

The method must be shown to work on data where the answer is known:

1. **Zone-entry intervals**: analytic results match brute-force sampling at
   10 ms resolution on random geometries.
2. **No missed passes**: a synthetic head-on pass whose closest point falls
   between two 1 s samples is detected as observed T3.
3. **Synthetic scenarios**: head-on, crossing, in-trail, parallel-runway
   and runway-area encounters are detected with the correct tier, class and
   minimum separation.
4. **Pruning safety**: pruning never removes a pair that the unpruned
   computation flags.
5. **P&E probability**: analytic result vs Monte Carlo within 0.01.
6. **Baseline calibration**: on simulated negative binomial windows with
   known covariate effects, the fitted limits achieve nominal exceedance
   (within binomial error).
7. **Injected effect**: a synthetic dataset with a known post-go-around
   rate increase (e.g. IRR = 1.5) is detected; with no injected effect,
   the false-positive rate is ≈ α across repeated simulations.
8. **Go-around parity**: the adapter reproduces GoAroundAlgo outputs on a
   small fixture (hidden-low-point rule off).
9. **Traffic cross-check**: `validate_cpa.py` closest-approach distances
   agree with the traffic library on sample days.
10. **Input handling**: daily files are found in month subfolders; a
    synthetic day with injected frozen-position ghost rows loads
    identically to the clean day; re-timing to the report time is exact.
11. **Events**: touch-and-goes enter the event set with their own name,
    and landing controls keep away from them.
12. **Automatic profile**: created offline from a stubbed OurAirports
    database and the synthetic data (timezone, preset, arrivals
    assumption, runways, airspace block), never overwritten.
13. **Hidden low point**: the hidden/observed plateau split is exact on
    constructed tracks; only climb-aways whose observed level time is
    within the cutoff are reclassified; on synthetic days with a 600 ft
    coverage floor every true go-around is recovered with no low
    approaches left.

---

## 19. Performance

- Days are processed independently and in parallel (`--workers`).
- Per-day pairwise results are cached (`cache/<label>/<day>.parquet`) keyed
  by a hash of the relevant parameters; re-runs and sensitivity sweeps
  reuse them.
- Pairwise computation is vectorised in NumPy per second (pairs × 1 s),
  with pruning; memory is bounded by processing one day at a time.

---

## 20. Known limitations (state these in any publication)

- Proximity risk is a surrogate for safety, not a collision probability.
- Surveillance only: aircraft without ADS-B, or below the local coverage
  floor, are invisible; coverage gaps are measured but cannot be recovered.
- Straight-line prediction ignores clearances and intent; the
  probability model quantifies, but does not remove, this error.
- Tier thresholds are regulatory reference values, not the separation that
  was actually required for each pair (visual separation, parallel-approach
  procedures and reduced separation on final are not observable). Geometry
  classes and the sensitivity analysis address this.
- Go-around intent (ATC-directed, pilot-initiated, practice) is invisible.
- Observational design: association, not causation; the pre window and
  the conditional baseline control for, but do not eliminate, confounding.

---

## 21. Build order

1. Scaffolding: repo layout, `config.py`, profiles with `airspace:` block,
   `new-airport`, `check-data`, `.gitignore`, `requirements.txt`.
2. Vendor `goaround_pipeline`, adapter, output folder structure; verify
   parity.
3. Loading, cleaning, 1 s grid, projection, quality flags, airborne filter.
4. Pairwise zone-entry intervals, pruning, encounters, geometry classes
   (+ tests 1–4).
5. Window metrics and covariates (flows).
6. Ceiling adequacy check.
7. Baseline: regression, Phase I/II, validation, storage (+ test 6).
8. Events, clusters, matched landing controls; statistics, equilibrium
   (+ test 7).
9. Probability model and its calibration outputs (+ test 5).
10. Sensitivity sweep.
11. Figures, per-event plots, `summary.pdf`, `baseline_report.pdf`.
12. `validate_cpa.py`, README, `TERMINAL_COMMANDS.pdf`.

Each step is committed separately and tested with `--limit` on real data
before moving on.

---

## 22. References

1. Generating Synthetic Flight Tracks for Collision Risk Analysis Using
   Artificial Intelligence: Case Study of Chicago O'Hare Runway 27C (2026)
   — Shortle, Sherry et al.
2. Review of Current State of AI/ML and Other Advanced Techniques Related
   to Air-to-Air Collision Risk Models (CRM) in the Terminal Airspace
   (2023).
3. Ghorbani et al., Separating the Signal from the Noise: Statistical
   Process Control (SPC) for Steady-State Monitoring of Airspace Collision
   Risk (2025).
4. Paielli, R. A. & Erzberger, H., Conflict Probability Estimation for Free
   Flight, NASA (1997), NTRS 19970001686.
5. FAA JO 7110.65, Air Traffic Control (separation minima).
6. FAA JO 7400.11 and Aeronautical Information Manual 3-2-4 (Class C
   airspace).
7. Montgomery, D. C., Introduction to Statistical Quality Control (Phase
   I/II, risk-adjusted control charts).
8. Dunn, P. K. & Smyth, G. K. (1996), Randomized quantile residuals,
   Journal of Computational and Graphical Statistics.
9. Olive, X., traffic: a toolbox for processing and analysing air traffic
   data, Journal of Open Source Software (2019).
10. GoAroundAlgo — go-around detection from ADS-B data
    (github.com/sponbob3/GoAroundAlgo).
