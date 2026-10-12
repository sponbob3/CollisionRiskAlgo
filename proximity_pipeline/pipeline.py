"""
Orchestration: per-day processing with a parameter-hashed cache, and
run_all(), which runs every stage in the order of FRAMEWORK.md section 14.

Per-day cache (section 19): output/<ICAO>/proximity_risk/cache/ holds,
per daily file <day> (one folder per airport, so renaming the dataset
folder never loses it; caches left in older per-dataset subfolders are
moved up automatically when still valid):
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
import re
import shutil
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

# ------------------------------------------------- dataset identity -----

_ICAO_NAME = re.compile(r"^([A-Za-z][A-Za-z0-9]{3})(?:[\s_\-.]+(.*))?$")
_ICAO_FILE = re.compile(r"^([A-Za-z][A-Za-z0-9]{3})[_\-]?(?:19|20)\d{6}")


def dataset_identity(dataset: Path, icao: str | None = None
                     ) -> tuple[str, str]:
    """(ICAO, label) of a dataset folder, whatever it is called: KMCO,
    kmco_2025q1, KMCO-2025Q1 and "KMCO 2025 Q1" all work, and
    capitalisation never matters. The ICAO code comes from `icao` when
    given, else from the start of the folder name, else from the daily
    file names (KMCO_20250101.parquet). The label (lower case; "all"
    when the name has none) only names the run in reports and the
    stored baseline's folder; caches and baselines are found by their
    contents, not by this name."""
    dataset = Path(dataset)
    name = dataset.name
    m = _ICAO_NAME.match(name)
    code = icao.upper() if icao else None
    if code is None and m:
        code = m[1].upper()
    if code is None:
        for f in sorted(dataset.rglob("*")):
            fm = _ICAO_FILE.match(f.name)
            if fm and f.suffix in (".parquet", ".csv"):
                code = fm[1].upper()
                break
    if code is None:
        raise ValueError(
            f"cannot tell the airport from the folder name '{name}' or its "
            f"files: start the folder name with the ICAO code (e.g. "
            f"KMCO_{name}) or pass --airport KMCO")
    rest = (m[2] or "") if (m and m[1].upper() == code) else name
    label = re.sub(r"[^a-z0-9]+", "_", (rest or "").lower()).strip("_")
    return code, label or "all"


def cache_root(icao: str, label: str | None = None) -> Path:
    """One pairwise cache folder per airport (label kept for callers)."""
    return config.OUTPUT_DIR / icao.upper() / "proximity_risk" / "cache"


def adopt_old_caches(files: list[Path], cache_dir: Path) -> int:
    """Move valid day caches from older per-dataset subfolders
    (cache/<label>/) up into the airport's cache folder, so that a
    renamed dataset folder still finds them. Returns how many days were
    moved; stale copies are left where they are."""
    if not cache_dir.exists():
        return 0
    moved = 0
    for f in files:
        if cache_is_current(cache_dir, f):
            continue
        for meta in sorted(cache_dir.glob(f"*/{f.stem}.meta.json")):
            sub = meta.parent
            if cache_is_current(sub, f):
                for src in cache_paths(sub, f.stem).values():
                    shutil.move(str(src), str(cache_dir / src.name))
                moved += 1
                break
    for sub in cache_dir.iterdir():
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()
    return moved


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
    moved = adopt_old_caches(files, cache_dir)
    if moved and not quiet:
        print(f"pairwise stage: {moved} day caches moved from an older "
              f"dataset-name folder into {cache_dir.name}/")
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


def analyse(days: dict, app: pd.DataFrame, spec, ev: pd.DataFrame,
            ctl: pd.DataFrame, window_min: float, exclusion_min: float,
            baseline_dir: Path | None, rebuild: bool, validate: bool,
            reps: int, quiet: bool = False, ds_sig: str = "",
            baseline_from: Path | None = None,
            error_model=None, full: bool = True) -> dict:
    """Encounters -> windows -> baseline -> events -> statistics for one
    Spec. Returns a dict of tables (the sensitivity sweep calls this with
    varied inputs, validate=False and full=False). The conflict-
    probability model (section 7) runs only when an error_model is given
    (--probability). full=False skips the checks around the comparison
    with normal landings (before-window scoring of the landings,
    pre-trend, placebo, regression, robustness, minute-by-minute)."""
    from . import windows as W, events as E, stats as S, baseline as B
    from . import compare as C
    tiers = spec.tiers
    exp = W.Exposure(days, spec)
    enc = pd.concat([W.day_encounters(dc, spec) for dc in days.values()],
                    ignore_index=True)
    enc = E.tag_encounters(enc, ev, "ga")
    enc = E.tag_encounters(enc, ctl, "ctl")
    usage = W.runway_usage(app, days)

    # optional add-on: conflict probabilities (section 7) per tier
    prob_tables, calib = {}, {}
    if error_model is not None:
        from . import probability as P
        prob_tables = {tier: {d: P.day_pair_probabilities(dc, spec,
                                                          error_model, tier)
                              for d, dc in days.items()}
                       for tier in tiers}
        calib = {tier: P.calibration_from_tables(prob_tables[tier])
                 for tier in tiers}

    def expected_cols(intervals):
        if not prob_tables:
            return None
        from . import probability as P
        return pd.DataFrame({
            f"expected_conflicts_{tier}": P.expected_conflicts(
                prob_tables[tier], intervals, tier) for tier in tiers},
            index=intervals.index)

    # clock windows (section 8)
    clock = W.clock_windows(days, window_min)
    wm = W.interval_metrics(clock, exp, enc, days, usage, expected_cols(clock))

    # baseline (section 9)
    t_excl = E.event_table(app, config.EVENT_OUTCOMES_AMBIGUOUS,
                           window_min)["t0"].to_numpy()
    bl = None
    reused = False
    if baseline_from is not None:
        bl = B.load_baseline(baseline_from)
        reused = True
    elif baseline_dir is not None and not rebuild:
        if B.baseline_is_reusable(baseline_dir, baseline_param_hash_for(
                spec, window_min, exclusion_min), ds_sig):
            bl = B.load_baseline(baseline_dir)
            reused = True
            if not quiet:
                print(f"baseline reused from {baseline_dir}")
    if bl is None:
        bl = B.build_baseline(wm, t_excl, tiers, "airspace", validate,
                              exclusion_min, quiet=quiet)

    # event windows (sections 10-11)
    ew = E.event_windows(ev)
    ewm = W.interval_metrics(ew, exp, enc, days, usage, expected_cols(ew))
    ewm["day"] = [_day_of(int(t), days) for t in ewm["t_start"]]
    scored = S.score_event_windows(ewm, bl, config.RANDOM_SEED)
    if f"{S.PRIMARY}_exceed95" in scored.columns:
        scored[f"{S.PRIMARY}_in_control"] = ~scored[
            f"{S.PRIMARY}_exceed95"].astype(bool)
    metrics = B.count_metrics(tiers) + B.continuous_metrics(tiers)
    risk = S.event_risk_table(scored, ev, metrics)
    risk["day"] = [_day_of(int(t), days) for t in risk["t0"]]

    # normal landings: the same windows, airspace-wide (data only; each
    # event plot shows its most similar landings). With full=True also
    # the window before the before-window (pre-trend check) and the
    # baseline scores of the landing windows, from models that did not
    # see the landing's own month (score_held_out).
    which = ("early", "pre", "post") if full else ("pre", "post")
    cw = E.event_windows(ctl, which)
    cwm = W.interval_metrics(cw, exp, enc, days, usage, with_smin=False)
    cwm["day"] = [_day_of(int(t), days) for t in cwm["t_start"]]
    ctl_scoring = "not scored"
    if full and S.PRIMARY in bl.count_models and len(cwm):
        sc, ctl_scoring = B.score_held_out(bl, cwm, S.PRIMARY,
                                           config.RANDOM_SEED)
        cwm = pd.concat([cwm, sc], axis=1)
        cwm[f"{S.PRIMARY}_in_control"] = ~cwm[
            f"{S.PRIMARY}_exceed95"].astype(bool)
    gwin = scored
    if full and len(ev):
        eew = W.interval_metrics(E.event_windows(ev, ("early",)), exp, enc,
                                 days, usage, with_smin=False)
        eew["day"] = [_day_of(int(t), days) for t in eew["t_start"]]
        gwin = pd.concat([scored, eew], ignore_index=True)

    # statistics (section 11): per unit of traffic against the baseline
    results = [S.primary_results(risk, bl, "airspace", "all", reps)]
    in_ctl = risk[risk[f"pre_{S.PRIMARY}_in_control"].fillna(False)
                  .astype(bool)]
    if len(in_ctl):
        results.append(S.primary_results(in_ctl, bl, "airspace",
                                         "in_control_pre", reps))
    if "outcome" in risk.columns and risk["outcome"].nunique() > 1:
        for typ in config.EVENT_TYPE_BREAKDOWN:
            sub = risk[risk["outcome"] == typ]
            if len(sub):
                results.append(S.primary_results(sub, bl, "airspace",
                                                 f"{typ}_only", reps))
    results = pd.concat(results, ignore_index=True)
    eq = S.equilibrium(risk)
    eq["scope"] = "airspace"

    # totals, traffic and the comparison with normal landings (11.6)
    cols = (["aircraft_s", "pair_hours"] + [f"{t}_any" for t in tiers]
            + ["T1_observed"])
    seed = config.RANDOM_SEED
    ga_pairs = C.paired_windows(gwin, ev, cols, which)
    ctl_pairs = C.paired_windows(cwm, ctl, cols, which)
    C.add_traffic_band(ga_pairs, ctl_pairs)
    change = C.change_summary(ga_pairs, ctl_pairs, reps, seed)
    comparison = {}
    if full and len(ev):
        comparison = _comparison(ga_pairs, ctl_pairs, ev, ctl, exp, enc,
                                 days, usage, reps, seed, ctl_scoring)

    # the go-around aircraft itself vs the other aircraft present (11.7)
    einv = E.involved_frame(ewm, enc, ev, exp, "ga", tiers)
    involvement = S.ga_aircraft_involvement(ewm, einv, risk, tiers, reps,
                                            config.RANDOM_SEED)

    epoch = S.superposed_epoch(ev.assign(day=risk["day"].to_numpy())
                               if len(ev) else ev, exp, enc, wm, bl,
                               S.PRIMARY, reps, config.RANDOM_SEED)
    return {"spec": spec, "exposure": exp, "encounters": enc, "usage": usage,
            "window_metrics": wm, "baseline": bl, "baseline_reused": reused,
            "event_windows": scored, "risk": risk, "results": results,
            "equilibrium": pd.DataFrame([eq]), "epoch": epoch,
            "change": change, "involvement": involvement,
            "ga_pairs": ga_pairs, "ctl_pairs": ctl_pairs,
            "comparison": comparison,
            "calibration": calib, "prob_tables": prob_tables}


def _comparison(g, c, ev, ctl, exp, enc, days, usage, reps, seed,
                ctl_scoring) -> dict:
    """The checks around the comparison with normal landings
    (FRAMEWORK.md section 11.6): before-window checks for both groups,
    pre-trend, placebo, regression confirmation, robustness set,
    leave-one-out and the raw minute-by-minute curves."""
    from . import compare as C, events as E, windows as W
    out = {"landing_scoring": ctl_scoring}
    out["before"] = C.before_checks(g, c, reps, seed)
    out["pretrend"] = C.change_summary(g, c, reps, seed, frm="early",
                                       to="pre")
    out["placebo"] = {col: C.placebo(g, c, col, reps, seed)
                      for col in ("T1_any", "pair_hours")}
    out["regression"] = pd.DataFrame(
        [r for r in (C.regression_check(g, c, "T1_any"),
                     C.regression_check(g, c, "pair_hours")) if r])
    # other window lengths: same events and landings, windows re-cut
    extra = []
    cols = ["aircraft_s", "pair_hours", "T1_any", "T1_observed"]
    for wmin in config.WINDOW_SENSITIVITY_MIN:
        frames = []
        for tab in (ev, ctl):
            t2 = tab.copy()
            t2["t_pre_start"] = t2["t0"] - int(wmin * 60)
            t2["t_post_end"] = t2["t0"] + int(wmin * 60)
            wm2 = W.interval_metrics(E.event_windows(t2), exp, enc, days,
                                     usage, with_smin=False)
            wm2["day"] = [_day_of(int(t), days) for t in wm2["t_start"]]
            frames.append(C.paired_windows(wm2, t2, cols))
        extra.append((f"{wmin:g}-minute windows", frames[0], frames[1]))
    out["robustness"] = C.robustness(g, c, "T1_any", reps, seed, extra)
    out["leave_one_out"] = C.leave_one_out(g, c, "T1_any")
    out["raw_epoch"] = C.raw_epoch(g, c, exp, enc, reps, seed)
    return out


def find_baseline(root: Path, label: str, param_hash: str | None,
                  ds_sig: str | None) -> Path:
    """Folder of the stored baseline for this data: baseline/<label>/
    when it matches; otherwise any stored baseline of this airport built
    from exactly the same files with the same settings (a renamed
    dataset folder); otherwise baseline/<label>/ (built there). With no
    hash and signature, the folder whose name matches the label
    regardless of capitalisation."""
    from .baseline import baseline_is_reusable
    own = root / label
    if param_hash is None:
        if own.exists() or not root.exists():
            return own
        for d in root.iterdir():
            if d.is_dir() and d.name.lower() == label.lower():
                return d
        return own
    if baseline_is_reusable(own, param_hash, ds_sig):
        return own
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir() and baseline_is_reusable(d, param_hash, ds_sig):
                return d
    return own


def goaround_run_info(ga_dir: Path) -> dict:
    """Dataset name and number of days a go-around run was made from
    (from its run_config.txt), and its go-around / touch-and-go counts
    and first / last approach date."""
    info = {"dir": ga_dir, "nn": int(ga_dir.name.split("_")[1]),
            "dataset": "?", "days": None, "go_arounds": 0,
            "touch_and_goes": 0, "first": None, "last": None}
    cfg = ga_dir / "run_config.txt"
    if cfg.exists():
        m = re.search(r"^dataset:\s*(.+?)\s*\((\d+) daily",
                      cfg.read_text(errors="ignore"), re.M)
        if m:
            info["dataset"], info["days"] = m[1], int(m[2])
    try:
        a = pd.read_csv(ga_dir / "all_approaches.csv",
                        usecols=["outcome", "t_low_utc"])
        vc = a["outcome"].value_counts()
        info["go_arounds"] = int(vc.get("go_around", 0))
        info["touch_and_goes"] = int(vc.get("touch_and_go", 0))
        t = pd.to_datetime(a["t_low_utc"], utc=True, errors="coerce").dropna()
        if len(t):
            info["first"] = f"{t.min():%Y-%m-%d}"
            info["last"] = f"{t.max():%Y-%m-%d}"
    except Exception:
        pass
    return info


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
    from . import airspace, events as E, windows as W
    from . import baseline as B
    t_run = time.time()
    dataset = Path(dataset)
    icao, label = dataset_identity(dataset, getattr(args, "airport", None))
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
        info = goaround_run_info(ga_dir)
        if info["days"] is not None and info["days"] != len(files):
            print(f"  note: {ga_dir.name} was made from {info['days']} days "
                  f"of data ({info['dataset']}); this dataset has "
                  f"{len(files)} days")
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

    # 4. optional add-on: error model for the conflict-probability model
    #    (section 7; only with --probability)
    em_table, error_model = pd.DataFrame(), None
    if getattr(args, "probability", False):
        from . import probability as P
        err = pd.concat([dc.errors for dc in days.values()], ignore_index=True)
        em_table = P.fit_error_model(err) if len(err) else pd.DataFrame()
        error_model = P.ErrorModel(em_table)

    # 5-8. encounters, windows, baseline, events, statistics
    baseline_dir = find_baseline(px_base / "baseline", label,
                                 baseline_param_hash_for(
                                     spec, window_min,
                                     config.BASELINE_EXCLUSION_MIN), ds_sig)
    baseline_from = None
    if args.baseline_from:
        _, other_label = dataset_identity(Path(args.baseline_from),
                                          icao=icao)
        baseline_from = find_baseline(px_base / "baseline", other_label,
                                      None, None)
        if not (baseline_from / "baseline.json").exists():
            raise FileNotFoundError(f"no stored baseline at {baseline_from}")
        print(f"baseline: scoring against {rel(baseline_from)}")
    print()
    res = analyse(days, app, spec, ev, ctl, window_min,
                  config.BASELINE_EXCLUSION_MIN, baseline_dir,
                  args.rebuild_baseline, validate=True,
                  reps=config.BOOTSTRAP_REPS, ds_sig=ds_sig,
                  baseline_from=baseline_from, error_model=error_model)
    bl = res["baseline"]
    res["files"] = files
    res["heights"] = heights
    if not res["baseline_reused"] and baseline_from is None:
        ph = baseline_param_hash_for(spec, window_min, config.BASELINE_EXCLUSION_MIN)
        B.save_baseline(bl, baseline_dir, em_table, ph, ds_sig)
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

    for line in result_lines(res):
        print(line)

    # 9. sensitivity sweep (section 12)
    sens = pd.DataFrame()
    if not args.no_sensitivity:
        from . import sensitivity as SENS
        print()
        sens = SENS.sweep(days, app, ev, ctl, window_min, spec,
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
    res["change"].to_csv(run_dir / "change_summary.csv", index=False)
    _write_comparison(run_dir, res)
    res["involvement"].to_csv(run_dir / "go_around_aircraft.csv", index=False)
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
        "probability_model": ({t: c["ece"] for t, c in
                               res["calibration"].items()}
                              if res["calibration"] else "not run"),
        "elapsed_s": round(time.time() - t_run, 1),
    }
    dump_run_config(run_dir, dataset, len(files), argv, extra)


def _write_comparison(run_dir: Path, res: dict) -> None:
    """CSV outputs of the comparison with normal landings."""
    res["ctl_pairs"].to_csv(run_dir / "landing_windows.csv", index=False)
    comp = res.get("comparison") or {}
    if not comp:
        return
    d = run_dir / "landing_comparison"
    d.mkdir(exist_ok=True)
    comp["before"].to_csv(d / "before_window_checks.csv", index=False)
    comp["pretrend"].to_csv(d / "pretrend.csv", index=False)
    comp["regression"].to_csv(d / "regression_check.csv", index=False)
    comp["robustness"].to_csv(d / "robustness.csv", index=False)
    comp["leave_one_out"].to_csv(d / "leave_one_out.csv", index=False)
    comp["raw_epoch"].to_csv(d / "minute_by_minute.csv", index=False)
    pl = comp["placebo"]
    pd.DataFrame([{k: v for k, v in r.items() if k != "ratios"}
                  for r in pl.values() if r]).to_csv(d / "placebo.csv",
                                                     index=False)
    pd.DataFrame({k: pd.Series(r["ratios"]) for k, r in pl.items() if r}
                 ).to_csv(d / "placebo_sets.csv", index=False)


def result_lines(res: dict) -> list[str]:
    """Console summary of the main results, in plain words: the
    comparison with normal landings first, then the checks behind it,
    then the per-traffic (mechanism) results."""
    out = []
    ch = res["change"].set_index("measure") if len(res["change"]) else None
    if ch is not None:
        for key in ("T1_any", "pair_time"):
            if key in ch.index:
                r = ch.loc[key]
                out.append(f"{r['label'].upper()}: go-around vs normal landing "
                           f"{r['vs_normal']:.2f} [{r['vs_normal_lo']:.2f}, "
                           f"{r['vs_normal_hi']:.2f}]  (after/before: "
                           f"go-arounds {r['ratio']:.2f} [{r['ratio_lo']:.2f}"
                           f", {r['ratio_hi']:.2f}], normal landings "
                           f"{r['normal_ratio']:.2f})")
    comp = res.get("comparison") or {}
    if comp:
        b = comp["before"].set_index("group")
        for grp in ("go-arounds", "normal landings"):
            if grp in b.index and "in_normal_range" in b.columns:
                r = b.loc[grp]
                out.append(f"BEFORE-WINDOW, {grp.upper()}: "
                           f"{100 * r['in_normal_range']:.0f}% in the normal "
                           f"range; observed/expected {r['oe_before']:.2f} "
                           f"[{r['oe_before_lo']:.2f}, {r['oe_before_hi']:.2f}]")
        last = comp["before"].iloc[-1]
        if "traffic_ratio" in last and np.isfinite(last["traffic_ratio"]):
            out.append(f"BEFORE-WINDOW TRAFFIC, go-arounds / landings: "
                       f"{last['traffic_ratio']:.2f} "
                       f"[{last['traffic_ratio_lo']:.2f}, "
                       f"{last['traffic_ratio_hi']:.2f}]")
        pt = comp["pretrend"].set_index("measure")
        if "T1_any" in pt.index:
            r = pt.loc["T1_any"]
            out.append(f"PRE-TREND (T1, 20-10 min before -> last 10 min): "
                       f"go-around vs normal {r['vs_normal']:.2f} "
                       f"[{r['vs_normal_lo']:.2f}, {r['vs_normal_hi']:.2f}]"
                       f"  (about 1 = parallel before the event)")
        p = comp["placebo"].get("T1_any") or {}
        if p:
            out.append(f"PLACEBO (T1): {100 * p['share_at_least_as_large']:.1f}"
                       f"% of {p['n_sets']} random sets of hour-matched "
                       f"landings changed as much as the go-arounds "
                       f"(p = {p['p_value']:.3f})")
        rg = comp["regression"]
        if len(rg):
            r = rg.iloc[0]
            out.append(f"REGRESSION CHECK (T1, adjusted for hour and flow): "
                       f"{r['ratio']:.2f} [{r['ratio_lo']:.2f}, "
                       f"{r['ratio_hi']:.2f}]  p = {r['p_value']:.3f}")
        rb = comp["robustness"]
        if len(rb):
            out.append(f"ROBUSTNESS (T1, go-around vs normal): "
                       f"{rb['vs_normal'].min():.2f} to "
                       f"{rb['vs_normal'].max():.2f} across "
                       f"{len(rb)} variations; "
                       f"{int((rb['vs_normal_lo'] > 1).sum())} with the "
                       f"interval above 1")
    eq = res["equilibrium"].iloc[0] if len(res["equilibrium"]) else None
    if eq is not None:
        out.append(f"BEFORE-GO-AROUND CHECK: {eq['verdict']}  "
                   f"({100 * eq['in_control_share']:.0f}% of before-windows "
                   f"in the normal range; observed/expected "
                   f"{eq['sir_pre']:.2f} [{eq['sir_pre_ci'][0]:.2f}, "
                   f"{eq['sir_pre_ci'][1]:.2f}])")
    prim = res["results"][(res["results"]["endpoint"] == "T1_any")
                          & (res["results"]["event_set"] == "all")]
    for _, r in prim.iterrows():
        out.append(f"T1 PER UNIT OF TRAFFIC: after/before {r['irr']:.2f} "
                   f"[{r['irr_lo']:.2f}, {r['irr_hi']:.2f}]  after vs normal "
                   f"{r['sir_post']:.2f} [{r['sir_post_lo']:.2f}, "
                   f"{r['sir_post_hi']:.2f}]  n={r['n_events']}")
    inv = res["involvement"]
    if len(inv):
        for _, r in inv[inv["tier"] == "T1"].iterrows():
            out.append(f"GO-AROUND AIRCRAFT ({r['window']}): involved in "
                       f"{r['ratio']:.2f}x the T1 encounters of an average "
                       f"aircraft present [{r['ratio_lo']:.2f}, "
                       f"{r['ratio_hi']:.2f}]")
    return out


def _report(run_dir, res, chk, ev, days, em_table, args, baseline_dir,
            baseline_only=False, sens=None):
    if args.no_plots:
        return
    from . import report_pdf, viz
    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)
    viz.baseline_figures(res["baseline"], baseline_dir / "validation")
    if not args.baseline_from:
        report_pdf.baseline_report(res["baseline"], baseline_dir)
    if res["calibration"]:
        viz.probability_figure(res["calibration"], em_table,
                               run_dir / "probability")
    if baseline_only:
        return
    viz.study_figures(res, chk, ev, figdir, sens)
    if not args.no_event_plots:
        viz.event_plots(res, ev, days, run_dir / "events")
    report_pdf.summary(run_dir, res, chk, ev, sens)
