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

CACHE_VERSION = 1

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


def run_all(dataset: Path, args, argv: list[str]) -> Path:
    """The whole run (section 14). Filled in stage by stage."""
    raise NotImplementedError("run_all is assembled in later build steps")
