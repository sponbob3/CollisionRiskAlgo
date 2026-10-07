"""
Orchestration: per-day processing with a parameter-hashed cache, and
run_all(), which runs every stage in the order of FRAMEWORK.md section 14.

Per-day cache (section 19): output/<ICAO>/proximity_risk/cache/<label>/
holds, per daily file <day>:
    <day>.pairs.parquet     flagged pair-seconds (pairs.PAIR_COLUMNS)
    <day>.presence.parquet  airborne aircraft-seconds in the volume
    <day>.legs.parquet      flight legs (leg index -> icao24, callsign)
    <day>.gaps.parquet      per-leg gaps > MAX_INTERP_GAP_S
    <day>.errors.parquet    dead-reckoning error samples (section 7.1)
    <day>.meta.json         provenance: parameter hash, file signature,
                            outages, counts
A day is recomputed when the raw file or any parameter that enters the
per-day computation changes; re-runs and sensitivity sweeps reuse it.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import goaround_adapter as ga
from . import loading, pairs, probability

CACHE_VERSION = 4

# proximity parameters that change the per-day computation
DAY_PARAMS = (
    "LOAD_RADIUS_NM", "AIRBORNE_MIN_GS_KT", "MAX_IMPLIED_SPEED_KT",
    "MAX_INTERP_GAP_S", "GRID_STEP_S", "VRATE_SMOOTH_WINDOW_S",
    "SEGMENT_GAP_MINUTES", "TIERS", "TIER_SENSITIVITY_FACTORS",
    "T_LOOKAHEAD_S", "T_LOOKAHEAD_SENSITIVITY_S", "PRUNE_VMAX_KT",
    "PAIR_CHUNK_S", "PHASE_MAX_XTRACK_NM", "PHASE_MAX_TRACK_DELTA_DEG",
    "PHASE_ARRIVAL_MAX_DIST_NM", "PHASE_ARRIVAL_MAX_HEIGHT_FT",
    "PHASE_DEPARTURE_MAX_DIST_NM", "PHASE_DEPARTURE_MIN_VRATE_FPM",
    "ERROR_MODEL_TAU_STEP_S", "ERROR_MODEL_SAMPLES_PER_DAY",
    "CEILING_CAP_FT_AGL", "CEILING_BUFFER_FT", "CEILING_SENSITIVITY_FT_AGL",
    "VOLUME_RADIUS_NM", "VOLUME_RADIUS_SENSITIVITY_NM", "RANDOM_SEED",
    "POSITION_FRESHNESS_FILTER", "POSITION_TIME_COLUMNS",
    "MAX_POSITION_AGE_S", "GEO_BARO_MAX_DEV_FT", "QUALITY_INTERP_GAP_S",
)
GA_DAY_PARAMS = ("AIRPORT_ICAO", "AIRPORT_LATLON", "FIELD_ELEVATION_FT",
                 "RUNWAYS", "TERRAIN_MODE")


# ------------------------------------------------------ config state ----

def config_state() -> dict:
    """Every resolved parameter of both config modules (for workers and
    for run_config.txt)."""
    return {
        "prox": {k: getattr(config, k) for k in dir(config)
                 if k.isupper() and not isinstance(getattr(config, k), Path)},
        "ga": {k: getattr(ga.config, k) for k in dir(ga.config)
               if k.isupper() and not isinstance(getattr(ga.config, k), Path)},
    }


def apply_config_state(state: dict) -> None:
    for k, v in state["prox"].items():
        setattr(config, k, v)
    for k, v in state["ga"].items():
        setattr(ga.config, k, v)


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, Path):
        return str(v)
    return v


def param_hash(names: tuple, ga_names: tuple = ()) -> str:
    d = {n: _jsonable(getattr(config, n)) for n in names}
    d.update({f"ga.{n}": _jsonable(getattr(ga.config, n)) for n in ga_names})
    d["cache_version"] = CACHE_VERSION
    blob = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def day_param_hash() -> str:
    return param_hash(DAY_PARAMS, GA_DAY_PARAMS)


def file_signature(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


# ------------------------------------------------------- day cache ------

def cache_root(icao: str, label: str) -> Path:
    return config.OUTPUT_DIR / icao / "proximity_risk" / "cache" / label


def cache_paths(cache_dir: Path, day: str) -> dict:
    return {k: cache_dir / f"{day}.{k}.{'json' if k == 'meta' else 'parquet'}"
            for k in ("pairs", "presence", "legs", "gaps", "errors", "meta")}


def cache_is_current(cache_dir: Path, path: Path) -> bool:
    p = cache_paths(cache_dir, path.stem)
    if not all(f.exists() for f in p.values()):
        return False
    try:
        meta = json.loads(p["meta"].read_text())
    except Exception:
        return False
    return (meta.get("param_hash") == day_param_hash()
            and meta.get("file_signature") == file_signature(path))


def process_day(path: Path, cache_dir: Path) -> dict:
    """Load, grid, pair and sample one day; write its cache; return the
    meta record."""
    path = Path(path)
    t_start = time.time()
    dg = loading.build_day_grid(path)
    dp = pairs.compute_day(dg)
    seed = (config.RANDOM_SEED + int(hashlib.sha1(path.stem.encode())
                                     .hexdigest()[:8], 16)) % (2 ** 31)
    errors = probability.sample_prediction_errors(dg, dp.presence, seed)
    outs = loading.outages(dg.messages_per_s, dg.legs, dg.t0)
    meta = {
        "day": path.stem,
        "file": path.name,
        "file_signature": file_signature(path),
        "param_hash": day_param_hash(),
        "date": f"{dg.date:%Y-%m-%d}",
        "t0": dg.t0,
        "n_raw": int(dg.n_raw),
        "n_rows_in_file": int(dg.n_rows_in_file),
        "stale_position_share": round(float(dg.stale_share), 4),
        "n_dropped_jumps": int(dg.n_dropped_jumps),
        "n_legs": int(len(dg.legs)),
        "n_grid_rows": int(len(dg.grid)),
        "n_presence_rows": int(len(dp.presence)),
        "n_pair_rows": int(len(dp.pairs)),
        "n_pair_candidates": int(dp.n_candidates),
        "n_pair_pruned": int(dp.n_pruned),
        "n_gaps": int(len(dg.gaps)),
        "outages": [[int(a), int(b)] for a, b in outs],
        "interp_share": float(dg.grid["interpolated"].mean()),
        "interp_long_share": float(dg.grid["interp_long"].mean()),
        "elapsed_s": round(time.time() - t_start, 2),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_paths(cache_dir, path.stem)
    dp.pairs.to_parquet(p["pairs"], index=False)
    dp.presence.to_parquet(p["presence"], index=False)
    dg.legs.to_parquet(p["legs"], index=False)
    dg.gaps.to_parquet(p["gaps"], index=False)
    errors.to_parquet(p["errors"], index=False)
    p["meta"].write_text(json.dumps(meta, indent=1))
    return meta


def _worker_init(state: dict) -> None:
    apply_config_state(state)


def _worker(args):
    path, cache_dir = args
    return process_day(Path(path), Path(cache_dir))


def build_day_caches(files: list[Path], cache_dir: Path, workers: int = 1,
                     force: bool = False, quiet: bool = False) -> list[dict]:
    """Ensure every day's cache is current; compute missing/stale days in
    parallel. Returns the meta records in file order."""
    todo = [f for f in files if force or not cache_is_current(cache_dir, f)]
    if not quiet:
        print(f"pairwise stage: {len(files)} days, {len(todo)} to compute "
              f"({len(files) - len(todo)} cached), workers={workers}")
    if todo:
        cache_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(cache_dir)) for f in todo]
        if workers > 1 and len(todo) > 1:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_worker_init,
                                     initargs=(config_state(),)) as ex:
                for i, meta in enumerate(ex.map(_worker, jobs)):
                    if not quiet:
                        _print_day(i, len(todo), meta)
        else:
            for i, job in enumerate(jobs):
                meta = _worker(job)
                if not quiet:
                    _print_day(i, len(todo), meta)
    return [json.loads(cache_paths(cache_dir, f.stem)["meta"].read_text())
            for f in files]


def _print_day(i, n, meta):
    print(f"[{i + 1}/{n}] {meta['file']}: {meta['n_legs']} legs, "
          f"{meta['n_presence_rows']:,} aircraft-s in volume, "
          f"{meta['n_pair_rows']:,} flagged pair-s, "
          f"{len(meta['outages'])} outages  ({meta['elapsed_s']} s)",
          flush=True)


class DayCache:
    """Lazy reader of one day's cache."""

    def __init__(self, cache_dir: Path, day: str):
        self.paths = cache_paths(cache_dir, day)
        self.day = day
        self._meta = None
        self._frames: dict = {}

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self._meta = json.loads(self.paths["meta"].read_text())
        return self._meta

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._frames:
            self._frames[name] = pd.read_parquet(self.paths[name])
        return self._frames[name]

    @property
    def pairs(self): return self.table("pairs")
    @property
    def presence(self): return self.table("presence")
    @property
    def legs(self): return self.table("legs")
    @property
    def gaps(self): return self.table("gaps")
    @property
    def errors(self): return self.table("errors")


# ------------------------------------------------------- run folder -----

def dump_run_config(run_dir: Path, dataset: Path, n_files: int,
                    argv: list[str], extra: dict | None = None) -> None:
    """Full provenance record: every resolved parameter of both config
    modules, the dataset, versions, and the resolved run facts."""
    import matplotlib, scipy, statsmodels
    lines = [
        f"run_time: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"dataset: {dataset.name}  ({n_files} daily files)",
        f"command: {' '.join(argv)}",
        f"python: {sys.version.split()[0]}  pandas: {pd.__version__}  "
        f"numpy: {np.__version__}  scipy: {scipy.__version__}  "
        f"statsmodels: {statsmodels.__version__}  "
        f"matplotlib: {matplotlib.__version__}",
        f"go-around pipeline: vendored, see goaround_pipeline/VENDORED_FROM.txt",
        "",
    ]
    if extra:
        lines.append("[resolved run]")
        for k, v in extra.items():
            lines.append(f"{k} = {v!r}")
        lines.append("")
    lines.append("[resolved proximity configuration]")
    for key in sorted(dir(config)):
        if key.isupper():
            lines.append(f"{key} = {getattr(config, key)!r}")
    lines += ["", "[resolved go-around configuration]"]
    for key in sorted(dir(ga.config)):
        if key.isupper():
            lines.append(f"{key} = {getattr(ga.config, key)!r}")
    (run_dir / "run_config.txt").write_text("\n".join(lines) + "\n")


def rel(path: Path) -> str:
    """Path relative to the repository root when it is inside it."""
    try:
        return str(Path(path).relative_to(config.ROOT))
    except ValueError:
        return str(path)


def dataset_signature(files: list[Path]) -> str:
    blob = "|".join(f"{f.name}:{file_signature(f)}" for f in files)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _day_of(t: int, days: dict) -> str:
    for name, dc in days.items():
        t0 = int(dc.meta["t0"])
        if t0 <= t < t0 + 86400:
            return name
    return ""


class Run:
    """Everything one run computes, so that the sensitivity sweep and the
    figures can reuse it."""

    def __init__(self):
        self.tables: dict = {}


def analyse(days: dict, app: pd.DataFrame, spec, error_model, ev: pd.DataFrame,
            ctl: pd.DataFrame, window_min: float, exclusion_min: float,
            baseline_dir: Path | None, rebuild: bool, validate: bool,
            reps: int, quiet: bool = False, ds_sig: str = "",
            baseline_from: Path | None = None) -> dict:
    """Encounters -> windows -> baseline -> events -> statistics for one
    Spec. Returns a dict of tables (the sensitivity sweep calls this with
    varied inputs and validate=False)."""
    from . import windows as W, events as E, stats as S, baseline as B
    from . import probability as P
    tiers = spec.tiers
    exp = W.Exposure(days, spec)
    enc = pd.concat([W.day_encounters(dc, spec) for dc in days.values()],
                    ignore_index=True)
    enc = E.tag_encounters(enc, ev, "ga")
    enc = E.tag_encounters(enc, ctl, "ctl")
    usage = W.runway_usage(app, days)

    # conflict probabilities (section 7) per tier
    prob_tables = {tier: {d: P.day_pair_probabilities(dc, spec, error_model,
                                                      tier)
                          for d, dc in days.items()}
                   for tier in tiers}
    calib = {tier: P.calibration_from_tables(prob_tables[tier])
             for tier in tiers}

    def expected_cols(intervals):
        return pd.DataFrame({
            f"expected_conflicts_{tier}": P.expected_conflicts(
                prob_tables[tier], intervals, tier) for tier in tiers},
            index=intervals.index)

    # clock windows (section 8)
    clock = W.clock_windows(days, window_min)
    wm = W.interval_metrics(clock, exp, enc, days, usage, expected_cols(clock))

    # baseline, airspace scope (section 9)
    t_excl = E.event_table(app, config.EVENT_OUTCOMES_AMBIGUOUS,
                           window_min)["t0"].to_numpy()
    bl = None
    reused = False
    if baseline_from is not None:
        bl = B.load_baseline(baseline_from)
        reused = True
    elif baseline_dir is not None and not rebuild:
        if B.baseline_is_reusable(baseline_dir, baseline_param_hash_for(spec, window_min, exclusion_min), ds_sig):
            bl = B.load_baseline(baseline_dir)
            reused = True
            if not quiet:
                print(f"baseline reused from {baseline_dir}")
    if bl is None:
        bl = B.build_baseline(wm, t_excl, tiers, "airspace", validate,
                              exclusion_min, quiet=quiet)

    # event windows, airspace scope (sections 10-11)
    ew = E.event_windows(ev)
    ewm = W.interval_metrics(ew, exp, enc, days, usage, expected_cols(ew))
    ewm["day"] = [_day_of(int(t), days) for t in ewm["t_start"]]
    scored = S.score_event_windows(ewm, bl, config.RANDOM_SEED)
    metrics = B.count_metrics(tiers) + B.continuous_metrics(tiers)
    risk = S.event_risk_table(scored, ev, metrics)
    risk["day"] = [_day_of(int(t), days) for t in risk["t0"]]

    # go-around-involved scope against matched landing controls (9.7)
    cw = E.event_windows(ctl)
    cwm = W.interval_metrics(cw, exp, enc, days, usage, expected_cols(cw))
    cwm["day"] = [_day_of(int(t), days) for t in cwm["t_start"]]
    cinv = E.involved_frame(cwm, enc, ctl, exp, "ctl", tiers)
    cinv["window_id"] = cinv["event_id"] + "_" + cinv["window"]
    bl_inv = None
    if baseline_from is not None and (baseline_from / "ga_involved" / "baseline.json").exists():
        bl_inv = B.load_baseline(baseline_from / "ga_involved")
    elif reused and baseline_dir is not None and (baseline_dir / "ga_involved" / "baseline.json").exists():
        bl_inv = B.load_baseline(baseline_dir / "ga_involved")
    if bl_inv is None:
        bl_inv = B.build_baseline(cinv, np.array([]), tiers, "ga_involved",
                                  validate, exclusion_min, quiet=quiet)
    einv = E.involved_frame(ewm, enc, ev, exp, "ga", tiers)
    scored_inv = S.score_event_windows(einv, bl_inv, config.RANDOM_SEED)
    risk_inv = S.event_risk_table(scored_inv, ev, metrics)
    risk_inv["day"] = risk["day"]

    # statistics (section 11)
    results, equil = [], []
    for scope, r, b in (("airspace", risk, bl), ("ga_involved", risk_inv, bl_inv)):
        eq = S.equilibrium(r)
        eq["scope"] = scope
        equil.append(eq)
        results.append(S.primary_results(r, b, scope, "all", reps))
        in_ctl = r[r[f"pre_{S.PRIMARY}_in_control"].fillna(False).astype(bool)]
        if len(in_ctl):
            results.append(S.primary_results(in_ctl, b, scope, "in_control_pre", reps))
        # per event type (go-arounds only, touch-and-goes only) when the
        # event set mixes types
        if "outcome" in r.columns and r["outcome"].nunique() > 1:
            for typ in config.EVENT_TYPE_BREAKDOWN:
                sub = r[r["outcome"] == typ]
                if len(sub):
                    results.append(S.primary_results(sub, b, scope,
                                                     f"{typ}_only", reps))
    results = pd.concat(results, ignore_index=True)
    epoch = S.superposed_epoch(ev.assign(day=risk["day"].to_numpy()) if len(ev) else ev,
                               exp, enc, wm, bl, S.PRIMARY, reps, config.RANDOM_SEED)
    return {"spec": spec, "exposure": exp, "encounters": enc, "usage": usage,
            "window_metrics": wm, "baseline": bl, "baseline_involved": bl_inv,
            "baseline_reused": reused, "event_windows": scored,
            "risk": risk, "risk_involved": risk_inv, "results": results,
            "equilibrium": pd.DataFrame(equil), "epoch": epoch,
            "calibration": calib, "prob_tables": prob_tables,
            "control_windows": cinv}


def baseline_param_hash_for(spec, window_min: float, exclusion_min: float) -> str:
    from .baseline import baseline_param_hash
    return baseline_param_hash({"radius": spec.radius_nm,
                                "ceiling": spec.ceiling_ft,
                                "lookahead": spec.lookahead_s,
                                "tiers": _jsonable(spec.tiers),
                                "window_min": window_min,
                                "exclusion_min": exclusion_min})


def run_all(dataset: Path, args, argv: list[str]) -> Path:
    """The whole run (section 14)."""
    from . import airspace, events as E, probability as P, windows as W
    from . import baseline as B
    t_run = time.time()
    dataset = Path(dataset)
    icao, label = dataset.name.split("_", 1)
    icao = icao.upper()
    from . import profile_setup
    profile_setup.ensure_profile(icao, dataset)
    airspace.load_airspace(icao)
    window_min = args.window or config.WINDOW_MIN
    outcomes = (config.EVENT_OUTCOMES_AMBIGUOUS if args.include_ambiguous
                else config.EVENT_OUTCOMES)
    out_root = config.OUTPUT_DIR / icao
    ga_base, px_base = out_root / "goaround", out_root / "proximity_risk"
    nn = max(ga.next_run_number(ga_base), ga.next_run_number(px_base))
    run_dir = px_base / f"run_{nn:02d}"
    run_dir.mkdir(parents=True)
    files = ga.data_files(dataset)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"no .parquet/.csv files in {dataset}")
    print(f"dataset: {dataset.name}  airport: {icao}  days: {len(files)}  "
          f"window: {window_min:g} min  -> {rel(run_dir)}")

    # 1. go-around detection (section 13)
    if args.goaround_run:
        ga_dir = ga_base / f"run_{args.goaround_run:02d}"
        if not (ga_dir / "all_approaches.csv").exists():
            raise FileNotFoundError(f"{ga_dir} has no all_approaches.csv")
        print(f"go-around stage: reusing {rel(ga_dir)}")
    else:
        ga_dir = ga_base / f"run_{nn:02d}"
        ga.run_goaround(dataset, ga_dir, plots=not args.no_plots,
                        limit=args.limit, calibrate=args.calibrate,
                        report=args.goaround_report, argv=argv)
        print()
    app = ga.read_approaches(ga_dir)

    # 2. per-day pairwise caches (sections 4-6, 19)
    cache_dir = cache_root(icao, label)
    build_day_caches(files, cache_dir, workers=args.workers)
    days = {f.stem: DayCache(cache_dir, f.stem) for f in files}
    ds_sig = dataset_signature(files)

    # 3. events and the ceiling adequacy check (sections 3.3, 10)
    ev = E.event_table(app, outcomes, window_min)
    ctl = E.control_table(app, window_min, config.BASELINE_EXCLUSION_MIN,
                          config.CONTROL_MAX_EVENTS, config.RANDOM_SEED)
    charted = airspace.charted_ceiling_ft_agl()
    spec0 = W.default_spec(charted)
    enc0 = pd.concat([W.day_encounters(dc, spec0) for dc in days.values()],
                     ignore_index=True)
    enc0 = E.tag_encounters(enc0, ev, "ga")
    ga_t1 = enc0[(enc0["tier"] == "T1") & (enc0["kind"] == "any")
                 & enc0["ga_event_id"].notna()]
    heights = E.event_aircraft_heights(ev, days, config.VOLUME_RADIUS_NM)
    beyond = (heights["post_beyond_radius_s"].sum()
              / max(heights["post_present_s"].sum(), 1)) if len(heights) else None
    chk = airspace.ceiling_check(heights["max_height_post_ft"],
                                 ga_t1["h_max_ft"], beyond)
    print()
    print("\n".join(airspace.ceiling_check_lines(chk)))
    pd.DataFrame([chk]).T.reset_index().rename(
        columns={"index": "item", 0: "value"}).to_csv(
        run_dir / "ceiling_check.csv", index=False)
    effective = chk["effective_ceiling_ft_agl"]
    spec = W.default_spec(effective)
    print(f"events: {E.event_count_text(ev)} "
          f"({len(ev[ev['cluster_first']])} cluster-first), {len(ctl)} "
          f"matched landing controls")

    # 4. error model (section 7.1)
    err = pd.concat([dc.errors for dc in days.values()], ignore_index=True)
    em_table = P.fit_error_model(err) if len(err) else pd.DataFrame()
    error_model = P.ErrorModel(em_table)

    # 5-8. encounters, windows, baseline, events, statistics
    baseline_dir = px_base / "baseline" / label
    baseline_from = None
    if args.baseline_from:
        other_label = args.baseline_from.split("_", 1)[1]
        baseline_from = px_base / "baseline" / other_label
        if not (baseline_from / "baseline.json").exists():
            raise FileNotFoundError(f"no stored baseline at {baseline_from}")
        print(f"baseline: scoring against {rel(baseline_from)}")
    print()
    res = analyse(days, app, spec, error_model, ev, ctl, window_min,
                  config.BASELINE_EXCLUSION_MIN, baseline_dir,
                  args.rebuild_baseline, validate=True,
                  reps=config.BOOTSTRAP_REPS, ds_sig=ds_sig,
                  baseline_from=baseline_from)
    bl, bl_inv = res["baseline"], res["baseline_involved"]
    res["files"] = files
    res["heights"] = heights
    if not res["baseline_reused"] and baseline_from is None:
        ph = baseline_param_hash_for(spec, window_min, config.BASELINE_EXCLUSION_MIN)
        B.save_baseline(bl, baseline_dir, em_table, ph, ds_sig)
        B.save_baseline(bl_inv, baseline_dir / "ga_involved", em_table, ph, ds_sig)
        print(f"baseline stored in {rel(baseline_dir)}")
    print(f"BASELINE: {bl.verdict}")
    for w in bl.warnings:
        print(f"  warning: {w}")
    if args.baseline_only:
        _write_common(run_dir, dataset, files, argv, res, chk, ev, ctl, heights,
                      window_min, outcomes, ga_dir, baseline_dir, t_run)
        _report(run_dir, res, chk, ev, days, em_table, args, baseline_dir,
                baseline_only=True)
        print(f"\nbaseline only: outputs in {rel(run_dir)}/")
        return run_dir

    for _, eq in res["equilibrium"].iterrows():
        print(f"EQUILIBRIUM [{eq['scope']}]: {eq['verdict']}  "
              f"(pre in control {100 * eq['in_control_share']:.0f}%, "
              f"SIR_pre {eq['sir_pre']:.2f} "
              f"[{eq['sir_pre_ci'][0]:.2f}, {eq['sir_pre_ci'][1]:.2f}], "
              f"KS p={eq['ks_p']:.2f})")
    prim = res["results"][(res["results"]["endpoint"] == "T1_any")
                          & (res["results"]["event_set"] == "all")]
    for _, r in prim.iterrows():
        print(f"PRIMARY [{r['scope']}]: IRR post/pre {r['irr']:.2f} "
              f"[{r['irr_lo']:.2f}, {r['irr_hi']:.2f}]  SIR_post "
              f"{r['sir_post']:.2f} [{r['sir_post_lo']:.2f}, "
              f"{r['sir_post_hi']:.2f}]  n={r['n_events']}  "
              f"MDE IRR {r['mde_irr']:.2f}")

    # 9. sensitivity sweep (section 12)
    sens = pd.DataFrame()
    if not args.no_sensitivity:
        from . import sensitivity as SENS
        print()
        sens = SENS.sweep(days, app, error_model, ev, ctl, window_min, spec,
                          effective, charted, outcomes, res)
        sens.to_csv(run_dir / "sensitivity.csv", index=False)

    # 10. tables, figures, per-event plots, summary.pdf
    _write_common(run_dir, dataset, files, argv, res, chk, ev, ctl, heights,
                  window_min, outcomes, ga_dir, baseline_dir, t_run)
    _report(run_dir, res, chk, ev, days, em_table, args, baseline_dir,
            baseline_only=False, sens=sens)
    print(f"\noutputs in {rel(run_dir)}/  "
          f"({time.time() - t_run:.0f} s)")
    return run_dir


def _write_common(run_dir, dataset, files, argv, res, chk, ev, ctl, heights,
                  window_min, outcomes, ga_dir, baseline_dir, t_run):
    wm = res["window_metrics"]
    wm.to_parquet(run_dir / "window_metrics.parquet", index=False)
    enc = res["encounters"]
    enc.to_parquet(run_dir / "encounters.parquet", index=False)
    enc.to_csv(run_dir / "encounters.csv", index=False)
    risk = res["risk"].merge(heights, on="event_id", how="left")
    risk.to_csv(run_dir / "go_around_risk.csv", index=False)
    res["risk_involved"].to_csv(run_dir / "go_around_risk_involved.csv",
                                index=False)
    res["results"].to_csv(run_dir / "results_primary.csv", index=False)
    res["equilibrium"].to_csv(run_dir / "equilibrium.csv", index=False)
    res["epoch"].to_csv(run_dir / "epoch.csv", index=False)
    ctl.to_csv(run_dir / "controls.csv", index=False)
    # data quality: per window classes and per day summary
    dq = wm.groupby("day").agg(
        windows=("window_id", "size"),
        good=("quality", lambda s: int((s == "good").sum())),
        degraded=("quality", lambda s: int((s == "degraded").sum())),
        bad=("quality", lambda s: int((s == "bad").sum())),
        interp_fraction=("interp_fraction", "mean"),
        gaps=("gap_count", "sum"),
        outage_windows=("outage", "sum"),
        pair_hours=("pair_hours", "sum"),
        aircraft_max=("n_aircraft", "max"),
    ).reset_index()
    dq.to_csv(run_dir / "data_quality.csv", index=False)
    bl = res["baseline"]
    extra = {
        "go_around_run": str(ga_dir),
        "days_processed": len(files),
        "window_min": window_min,
        "event_outcomes": outcomes,
        "n_events": int(len(ev)),
        "n_controls": int(len(ctl)),
        "charted_ceiling_ft_agl": chk["charted_ceiling_ft_agl"],
        "effective_ceiling_ft_agl": chk["effective_ceiling_ft_agl"],
        "ceiling_override": chk["override"],
        "ceiling_reason": chk["reason"],
        "baseline_dir": str(baseline_dir),
        "baseline_reused": res["baseline_reused"],
        "baseline_verdict": bl.verdict,
        "baseline_warnings": bl.warnings,
        "baseline_involved_verdict": res["baseline_involved"].verdict,
        "probability_calibration_ece": {t: c["ece"] for t, c in
                                        res["calibration"].items()},
        "elapsed_s": round(time.time() - t_run, 1),
    }
    dump_run_config(run_dir, dataset, len(files), argv, extra)


def _report(run_dir, res, chk, ev, days, em_table, args, baseline_dir,
            baseline_only=False, sens=None):
    if args.no_plots:
        return
    from . import report_pdf, viz
    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)
    viz.baseline_figures(res["baseline"], res["calibration"], em_table,
                         baseline_dir / "validation")
    if not args.baseline_from:
        report_pdf.baseline_report(res["baseline"], res["baseline_involved"],
                                   res["calibration"], em_table, baseline_dir)
    if baseline_only:
        return
    viz.study_figures(res, chk, ev, figdir, sens)
    if not args.no_event_plots:
        viz.event_plots(res, ev, days, run_dir / "events")
    report_pdf.summary(run_dir, res, chk, ev, sens)
