"""
Stage 1: load a daily file, clean it, and place every aircraft on a common
1-second grid in a local plane around the airport (FRAMEWORK.md section 4).

The raw loader and the flight-leg rules are the go-around pipeline's
(through goaround_adapter), so both pipelines see the same aircraft. On top
of that this stage:

- drops position jumps (implied speed > MAX_IMPLIED_SPEED_KT);
- computes height above field (geometric altitude minus field elevation,
  with the go-around pipeline's barometric re-referencing fallback);
- projects positions to an azimuthal-equidistant plane centred on the
  airport (x east, y north, NM) and derives velocity vectors from
  groundspeed / track / vertical rate;
- resamples every flight leg to 1 s, linearly interpolating gaps up to
  MAX_INTERP_GAP_S and leaving longer gaps empty (logged);
- flags airborne samples (onground false and groundspeed >= 50 kt).

The result is a DayGrid: one row per aircraft-second, plus a legs table,
a gap log and a per-second message count (for outage detection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

from . import config
from . import goaround_adapter as ga

NM_PER_M = 1.0 / 1852.0
FT_PER_M = 3.28084


@dataclass
class DayGrid:
    day: str                       # file stem, the cache key
    date: pd.Timestamp             # UTC date of the first sample
    t0: int                        # epoch seconds of 00:00 UTC that day
    grid: pd.DataFrame             # one row per aircraft-second
    legs: pd.DataFrame             # one row per flight leg
    gaps: pd.DataFrame             # per-leg gaps > MAX_INTERP_GAP_S
    messages_per_s: np.ndarray     # raw message count per second of day
    n_raw: int = 0
    n_dropped_jumps: int = 0
    notes: list = field(default_factory=list)


def epoch_s(ts) -> np.ndarray:
    """tz-aware timestamps -> int64 epoch seconds."""
    ts = pd.to_datetime(ts, utc=True)
    if isinstance(ts, pd.Timestamp):
        return np.int64(ts.value // 10**9)
    return (pd.DatetimeIndex(ts).tz_convert("UTC").tz_localize(None)
            .to_numpy().astype("datetime64[s]").astype(np.int64))


def projection() -> pyproj.Proj:
    lat, lon = ga.config.AIRPORT_LATLON
    return pyproj.Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84",
                       units="m")


def project(lat, lon):
    """Lat/lon -> local plane (x east, y north) in NM."""
    x, y = projection()(np.asarray(lon, float), np.asarray(lat, float))
    return x * NM_PER_M, y * NM_PER_M


def unproject(x, y):
    """Local plane (NM) -> lat, lon."""
    lon, lat = projection()(np.asarray(x, float) / NM_PER_M,
                            np.asarray(y, float) / NM_PER_M, inverse=True)
    return lat, lon


# ----------------------------------------------------------- cleaning ----

def _drop_jumps(g: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop fixes that imply an impossible speed from the previous kept
    fix (a corrupted position). Sequential so that one bad fix does not
    also condemn the good fix after it."""
    if len(g) < 2:
        return g, 0
    x, y = project(g["latitude"].to_numpy(), g["longitude"].to_numpy())
    t = epoch_s(g["timestamp"])
    keep = np.ones(len(g), dtype=bool)
    last = 0
    for i in range(1, len(g)):
        dt = t[i] - t[last]
        if dt <= 0:
            keep[i] = False
            continue
        d_nm = np.hypot(x[i] - x[last], y[i] - y[last])
        if d_nm / dt * 3600.0 > config.MAX_IMPLIED_SPEED_KT:
            keep[i] = False
        else:
            last = i
    return g[keep], int((~keep).sum())


def clean_day(raw: pd.DataFrame) -> tuple[list[pd.DataFrame], int]:
    """Per-aircraft cleaning and leg splitting. Returns legs (each with
    height-above-field `h`, smoothed `vrate`, and attrs) and the number of
    dropped jump fixes."""
    raw = raw.copy()
    raw["timestamp"] = raw["timestamp"].dt.floor("s")
    raw = raw.drop_duplicates(["icao24", "timestamp"], keep="first")
    # keep only traffic within the load radius (the data may cover a
    # much larger box; anything further can never enter the volume)
    x, y = project(raw["latitude"].to_numpy(), raw["longitude"].to_numpy())
    raw = raw[np.hypot(x, y) <= config.LOAD_RADIUS_NM]
    legs, n_jumps = [], 0
    gap = pd.Timedelta(minutes=config.SEGMENT_GAP_MINUTES)
    for icao24, g in raw.groupby("icao24", sort=False):
        g = g.sort_values("timestamp")
        g, dropped = _drop_jumps(g)
        n_jumps += dropped
        if len(g) < 2:
            continue
        new_leg = (g["timestamp"].diff() > gap).to_numpy(dtype=bool)
        for _, leg in g.groupby(np.cumsum(new_leg)):
            if len(leg) < 2:
                continue
            leg = leg.reset_index(drop=True).copy()
            elev = ga.config.FIELD_ELEVATION_FT
            h = leg["geoaltitude"] - elev
            # barometric fallback where geometric altitude is missing,
            # re-referenced per leg (same rule as the go-around loader)
            missing = h.isna()
            if missing.any() and leg["geoaltitude"].notna().any():
                offset = (leg["geoaltitude"] - leg["altitude"]).median()
                h[missing] = (leg["altitude"] + offset - elev)[missing]
            elif missing.all():
                # no geometric altitude at all: barometric minus elevation,
                # uncorrected (flagged in the legs table)
                h = leg["altitude"] - elev
            leg["h"] = h
            leg["vrate"] = (leg["vertical_rate"]
                            .rolling(config.VRATE_SMOOTH_WINDOW_S,
                                     center=True, min_periods=1).median())
            callsign = (leg["callsign"].mode().iat[0]
                        if leg["callsign"].notna().any() else "")
            leg.attrs["icao24"] = icao24
            leg.attrs["callsign"] = str(callsign).strip()
            leg.attrs["leg_id"] = (
                f"{icao24}_{leg['timestamp'].iat[0]:%Y%m%d_%H%M%S}")
            leg.attrs["geo_missing"] = bool(missing.all())
            legs.append(leg)
    return legs, n_jumps


# --------------------------------------------------------- resampling ----

def _interp_angle(t_new, t, deg):
    """Linear interpolation of a heading through its sine and cosine."""
    rad = np.radians(deg)
    s = np.interp(t_new, t, np.sin(rad))
    c = np.interp(t_new, t, np.cos(rad))
    return np.degrees(np.arctan2(s, c)) % 360.0


def _interp_nan(t_new, t, v):
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.full(len(t_new), np.nan)
    return np.interp(t_new, t[ok], v[ok], left=np.nan, right=np.nan)


def leg_to_grid(leg: pd.DataFrame, leg_idx: int):
    """Resample one leg to the 1 s grid. Returns (grid rows, gap rows)."""
    t = epoch_s(leg["timestamp"])
    x, y = project(leg["latitude"].to_numpy(), leg["longitude"].to_numpy())
    t_new = np.arange(t[0], t[-1] + 1, dtype=np.int64)
    n = len(t_new)
    # which grid seconds are real samples, and which fall in long gaps
    is_sample = np.zeros(n, dtype=bool)
    is_sample[t - t[0]] = True
    dts = np.diff(t)
    long_gaps = np.flatnonzero(dts > config.MAX_INTERP_GAP_S)
    absent = np.zeros(n, dtype=bool)
    gap_rows = []
    for i in long_gaps:
        a, b = t[i] - t[0], t[i + 1] - t[0]
        absent[a + 1:b] = True
        gap_rows.append((leg_idx, int(t[i]), int(t[i + 1]),
                         int(t[i + 1] - t[i])))

    cols = {
        "x": np.interp(t_new, t, x),
        "y": np.interp(t_new, t, y),
        "z_baro": _interp_nan(t_new, t, leg["altitude"].to_numpy(float)),
        "z_geo": _interp_nan(t_new, t, leg["geoaltitude"].to_numpy(float)),
        "h": _interp_nan(t_new, t, leg["h"].to_numpy(float)),
        "gs": _interp_nan(t_new, t, leg["groundspeed"].to_numpy(float)),
        "track": _interp_angle(t_new, t, leg["track"].to_numpy(float)),
        "vrate": _interp_nan(t_new, t, leg["vrate"].to_numpy(float)),
    }
    onground = np.interp(t_new, t, leg["onground"].to_numpy(float)) > 0.5
    keep = ~absent
    g = pd.DataFrame({k: v[keep] for k, v in cols.items()})
    g.insert(0, "t", t_new[keep])
    g.insert(0, "leg", np.int32(leg_idx))
    g["interpolated"] = ~is_sample[keep]
    g["onground"] = onground[keep]
    return g, gap_rows


def build_day_grid(path: Path) -> DayGrid:
    """Load, clean and resample one daily file (section 4)."""
    path = Path(path)
    raw = ga.load_day(path)
    n_raw = len(raw)
    legs, n_jumps = clean_day(raw)
    if not legs:
        raise ValueError(f"{path.name}: no usable aircraft legs")
    date = raw["timestamp"].min().floor("D")
    t0 = int(date.value // 10**9)

    # raw message count per second of the day (outage detection, 8.3)
    sec = epoch_s(raw["timestamp"]) - t0
    counts = np.bincount(sec[(sec >= 0) & (sec < 86400)], minlength=86400)

    grids, gap_rows, leg_rows = [], [], []
    for i, leg in enumerate(legs):
        g, gaps = leg_to_grid(leg, i)
        grids.append(g)
        gap_rows.extend(gaps)
        leg_rows.append({
            "leg": i,
            "leg_id": leg.attrs["leg_id"],
            "icao24": leg.attrs["icao24"],
            "callsign": leg.attrs["callsign"],
            "t_first": int(g["t"].iat[0]),
            "t_last": int(g["t"].iat[-1]),
            "n_raw": len(leg),
            "geo_missing": leg.attrs["geo_missing"],
        })
    grid = pd.concat(grids, ignore_index=True)

    # velocities in the plane: NM/s and ft/s
    b = np.radians(grid["track"].to_numpy())
    gs_nm_s = grid["gs"].to_numpy() / 3600.0
    grid["vx"] = gs_nm_s * np.sin(b)
    grid["vy"] = gs_nm_s * np.cos(b)
    grid["vz"] = grid["vrate"].to_numpy() / 60.0
    grid["r"] = np.hypot(grid["x"], grid["y"])
    grid["airborne"] = ((~grid["onground"])
                        & (grid["gs"] >= config.AIRBORNE_MIN_GS_KT)
                        & np.isfinite(grid["h"]))
    for c in ("x", "y", "z_baro", "z_geo", "h", "gs", "track", "vrate",
              "vx", "vy", "vz", "r"):
        grid[c] = grid[c].astype(np.float32)
    grid = grid.sort_values(["t", "leg"], kind="stable").reset_index(drop=True)

    gaps = pd.DataFrame(gap_rows, columns=["leg", "t_start", "t_end",
                                           "gap_s"])
    return DayGrid(day=path.stem, date=date, t0=t0, grid=grid,
                   legs=pd.DataFrame(leg_rows), gaps=gaps,
                   messages_per_s=counts, n_raw=n_raw,
                   n_dropped_jumps=n_jumps)


def outages(messages_per_s: np.ndarray, legs: pd.DataFrame | None = None,
            t0: int = 0, min_s: float | None = None) -> list:
    """Receiver outages: intervals longer than min_s with no messages from
    any aircraft while aircraft were present before and after. "Present"
    is read strictly: at least one flight leg must straddle the silent
    interval (an aircraft in flight went silent), otherwise a quiet night
    with no traffic would count as an outage. Returns [(start_s, end_s)]
    in seconds of day."""
    min_s = config.QUALITY_OUTAGE_MIN_S if min_s is None else min_s
    present = np.flatnonzero(messages_per_s > 0)
    out = []
    if len(present) < 2:
        return out
    d = np.diff(present)
    first = last = None
    if legs is not None and len(legs):
        first = legs["t_first"].to_numpy() - t0
        last = legs["t_last"].to_numpy() - t0
    for i in np.flatnonzero(d > min_s):
        a, b = int(present[i]), int(present[i + 1])
        if first is not None and not np.any((first < a) & (last > b)):
            continue
        out.append((a, b))
    return out


# ------------------------------------------------------- data audit -----

def audit_dataset(dataset: Path, limit: int | None = None) -> pd.DataFrame:
    """check-data: per-day audit of a dataset folder (section 14).
    Returns a per-day table; prints a summary."""
    files = ga.data_files(dataset)
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no .parquet/.csv files in {dataset}")
    rows = []
    for i, f in enumerate(files):
        try:
            if f.suffix.lower() == ".csv":
                head = pd.read_csv(f, nrows=5)
            else:
                head = pd.read_parquet(f)
            missing = [c for c in ga.RAW_COLUMNS if c not in head.columns]
            if missing:
                rows.append({"day": f.stem, "ok": False,
                             "problem": f"missing columns {missing}"})
                print(f"[{i + 1}/{len(files)}] {f.name}: MISSING {missing}")
                continue
            dg = build_day_grid(f)
        except Exception as e:  # report, never absorb
            rows.append({"day": f.stem, "ok": False, "problem": str(e)})
            print(f"[{i + 1}/{len(files)}] {f.name}: ERROR {e}")
            continue
        g = dg.grid
        air = g[g["airborne"]]
        in_vol = air[(air["r"] <= config.VOLUME_RADIUS_NM)]
        outs = outages(dg.messages_per_s, dg.legs, dg.t0)
        raw = ga.load_day(f)
        row = {
            "day": f.stem,
            "ok": True,
            "date": f"{dg.date:%Y-%m-%d}",
            "messages": dg.n_raw,
            "messages_in_load_radius": int(sum(dg.legs["n_raw"])),
            "aircraft": int(dg.legs["icao24"].nunique()),
            "legs": int(len(dg.legs)),
            "airborne_ac_in_volume": int(in_vol["leg"].nunique()),
            "dropped_jumps": dg.n_dropped_jumps,
            "gaps_over_10s": int(len(dg.gaps)),
            "interp_share": float(g["interpolated"].mean()),
            "outages": len(outs),
            "outage_s": int(sum(b - a for a, b in outs)),
            "baro_alt_share": float(raw["altitude"].notna().mean()),
            "geo_alt_share": float(raw["geoaltitude"].notna().mean()),
            "onground_share": float(raw["onground"].mean()),
            "problem": "",
        }
        rows.append(row)
        print(f"[{i + 1}/{len(files)}] {f.name}: {row['messages']:,} msgs, "
              f"{row['aircraft']} aircraft, {row['airborne_ac_in_volume']} "
              f"airborne in volume, {row['gaps_over_10s']} gaps, "
              f"{row['outages']} outages, geo-alt "
              f"{100 * row['geo_alt_share']:.0f}%", flush=True)
    table = pd.DataFrame(rows)
    n_ok = int(table["ok"].sum()) if len(table) else 0
    print()
    print(f"days found: {len(files)}   readable: {n_ok}")
    if n_ok:
        ok = table[table["ok"]]
        print(f"messages/day: median {ok['messages'].median():,.0f}   "
              f"aircraft/day: median {ok['aircraft'].median():.0f}   "
              f"interpolated share: {ok['interp_share'].mean():.3f}")
        print(f"altitude references: barometric "
              f"{100 * ok['baro_alt_share'].mean():.1f}%   geometric "
              f"{100 * ok['geo_alt_share'].mean():.1f}%")
    verdict = ("meets the baseline minimum" if n_ok >= config.MIN_BASELINE_DAYS
               else f"BELOW the baseline minimum of "
                    f"{config.MIN_BASELINE_DAYS} days")
    rec = ("" if n_ok >= config.RECOMMENDED_BASELINE_DAYS
           else f" (recommended: >= {config.RECOMMENDED_BASELINE_DAYS} days)")
    print(f"data volume: {n_ok} days {verdict}{rec}")
    return table
