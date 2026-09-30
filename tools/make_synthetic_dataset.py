#!/usr/bin/env python
"""
Synthetic ADS-B traffic for development and testing.

Produces daily files in the same OpenSky-style format as the real data
(RAW_COLUMNS of the go-around pipeline), with arrivals, go-arounds,
departures, overflights and - optionally - injected close-passing pairs,
flown with a small point-mass kinematic model around a real airport
profile. The go-around pipeline detects the synthetic go-arounds; the
proximity pipeline finds the resulting encounters. Nothing here is used by
the analysis itself; it exists so the method can be exercised on data
where the answer is known (FRAMEWORK.md section 18).

Usage:
    python tools/make_synthetic_dataset.py KBNA_synth --days 6
    python tools/make_synthetic_dataset.py KBNA_synth --days 6 \
        --movements 200 --ga-rate 0.04 --seed 7
    python tools/make_synthetic_dataset.py KBNA_effect --days 6 \
        --intruder-rate 2.0 --post-ga-irr 3.0    # injected effect

Writes datasets/<name>/<label>_YYYYMMDD.parquet (gitignored).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proximity_pipeline import goaround_adapter as ga  # noqa: E402

NM_PER_DEG_LAT = 60.0
FT_PER_NM = 6076.12
TURN_RATE_DEG_S = 3.0
ACCEL_KT_S = 2.0
GLIDE_FT_PER_NM = 318.0            # 3 degree glide path
HOURLY_WEIGHTS = np.array([        # local-hour demand profile
    0.2, 0.1, 0.1, 0.1, 0.2, 0.6, 1.4, 2.0, 1.8, 1.4, 1.3, 1.4,
    1.5, 1.4, 1.5, 1.7, 2.0, 2.1, 1.8, 1.5, 1.2, 0.9, 0.6, 0.3,
])


# ----------------------------------------------------------- geometry ----

@dataclass
class Airport:
    icao: str
    lat: float
    lon: float
    elev: float
    runways: dict          # name -> (thr_lat, thr_lon, bearing)
    tz: str
    flows: dict | None = None

    def xy(self, lat, lon):
        coslat = math.cos(math.radians(self.lat))
        return ((lon - self.lon) * NM_PER_DEG_LAT * coslat,
                (lat - self.lat) * NM_PER_DEG_LAT)

    def latlon(self, x, y):
        coslat = math.cos(math.radians(self.lat))
        return (self.lat + y / NM_PER_DEG_LAT,
                self.lon + x / (NM_PER_DEG_LAT * coslat))

    def threshold_xy(self, rwy):
        lat, lon, _ = self.runways[rwy]
        return self.xy(lat, lon)

    def bearing(self, rwy):
        return self.runways[rwy][2]


def load_airport(icao: str) -> Airport:
    prof = ga.load_profile(icao)
    c = ga.config
    apt = Airport(c.AIRPORT_ICAO, c.AIRPORT_LATLON[0], c.AIRPORT_LATLON[1],
                  c.FIELD_ELEVATION_FT, dict(c.RUNWAYS), c.LOCAL_TZ)
    # the profile's flows: block (if any) says which runway ends are used
    # together; otherwise group runway ends by heading
    apt.flows = ({k: [str(r) for r in v] for k, v in prof["flows"].items()}
                 if prof.get("flows") else _auto_flows(apt))
    return apt


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def along_cross(apt: Airport, rwy: str, x, y):
    """Along-track (positive = on final, before the threshold) and
    cross-track (NM) relative to a runway threshold."""
    tx, ty = apt.threshold_xy(rwy)
    b = math.radians(apt.bearing(rwy))
    ux, uy = math.sin(b), math.cos(b)
    dx, dy = x - tx, y - ty
    along = -(dx * ux + dy * uy)
    cross = dx * uy - dy * ux
    return along, cross


# --------------------------------------------------------- simulator ----

@dataclass
class State:
    x: float
    y: float
    z: float       # ft MSL
    gs: float      # kt
    track: float   # deg
    vr: float = 0.0
    onground: bool = False


@dataclass
class Segment:
    """One flight-plan leg: targets (None = hold current) and an end
    condition. `z` may be a callable of the state (a glide-path law)."""
    done: Callable[[State, float], bool]
    track: float | None = None
    gs: float | None = None
    z: float | Callable | None = None
    vrate: float = 1500.0
    onground: bool = False
    max_s: float = 1800.0


def fly(state: State, segments: list[Segment], dt: float = 1.0) -> list:
    rows = []
    for seg in segments:
        t = 0.0
        while True:
            if seg.track is not None:
                d = wrap180(seg.track - state.track)
                state.track = (state.track
                               + float(np.clip(d, -TURN_RATE_DEG_S * dt,
                                               TURN_RATE_DEG_S * dt))) % 360
            if seg.gs is not None:
                state.gs += float(np.clip(seg.gs - state.gs,
                                          -ACCEL_KT_S * dt * 2, ACCEL_KT_S * dt))
            target_z = seg.z(state) if callable(seg.z) else seg.z
            if target_z is not None:
                want = (target_z - state.z) / dt * 60.0
                state.vr = float(np.clip(want, -abs(seg.vrate),
                                         abs(seg.vrate)))
            else:
                state.vr = 0.0
            state.z += state.vr * dt / 60.0
            b = math.radians(state.track)
            state.x += state.gs / 3600.0 * dt * math.sin(b)
            state.y += state.gs / 3600.0 * dt * math.cos(b)
            state.onground = seg.onground
            rows.append((state.x, state.y, state.z, state.gs, state.track,
                         state.vr, state.onground))
            t += dt
            if seg.done(state, t) or t >= seg.max_s:
                break
    return rows


# ------------------------------------------------------ flight kinds ----

def _glide_law(apt: Airport, rwy: str):
    def law(st: State):
        along, _ = along_cross(apt, rwy, st.x, st.y)
        return apt.elev + 50.0 + max(along, 0.0) * GLIDE_FT_PER_NM
    return law


def _final_segments(apt: Airport, rwy: str, ga_height: float | None,
                    rng) -> list[Segment]:
    """From an established final to touchdown and roll-out, or to a
    go-around initiated at ga_height ft above field."""
    b = apt.bearing(rwy)
    segs = []
    if ga_height is None:
        segs.append(Segment(  # glide to the threshold
            track=b, gs=140.0, z=_glide_law(apt, rwy), vrate=1200.0,
            done=lambda st, t: along_cross(apt, rwy, st.x, st.y)[0] < 0.0))
        segs.append(Segment(  # flare / touchdown
            track=b, gs=130.0, z=apt.elev, vrate=250.0,
            done=lambda st, t: st.z <= apt.elev + 1.0, max_s=40))
        segs.append(Segment(  # roll-out
            track=b, gs=12.0, z=apt.elev, vrate=0.0, onground=True,
            done=lambda st, t: st.gs <= 15.0, max_s=90))
        segs.append(Segment(  # taxi in
            track=b, gs=12.0, z=apt.elev, vrate=0.0, onground=True,
            done=lambda st, t: t >= 25, max_s=25))
    else:
        low = apt.elev + ga_height
        segs.append(Segment(
            track=b, gs=140.0, z=_glide_law(apt, rwy), vrate=1200.0,
            done=lambda st, t: st.z <= low
            or along_cross(apt, rwy, st.x, st.y)[0] < 0.0))
        hold = float(rng.uniform(2.0, 6.0))
        segs.append(Segment(   # brief level-off before the climb
            track=b, gs=145.0, z=low, vrate=300.0,
            done=lambda st, t: t >= hold, max_s=hold))
        segs.append(Segment(   # initial climb on runway heading
            track=b, gs=170.0, z=apt.elev + 3000.0, vrate=2000.0,
            done=lambda st, t: st.z >= apt.elev + 1500.0))
    return segs


def arrival(apt: Airport, rwy: str, rng, go_around: bool,
            reapproach: bool = True) -> list:
    """An arrival from 12 NM on the extended centreline; optionally a
    go-around followed by a traffic pattern and a second approach."""
    b = apt.bearing(rwy)
    tx, ty = apt.threshold_xy(rwy)
    br = math.radians(b)
    d0 = 12.0
    x0 = tx - d0 * math.sin(br) + rng.normal(0, 0.05)
    y0 = ty - d0 * math.cos(br) + rng.normal(0, 0.05)
    st = State(x0, y0, apt.elev + 3000.0, 170.0, b)
    segs = [Segment(   # level until the glide path
        track=b, gs=160.0, z=apt.elev + 3000.0, vrate=500.0,
        done=lambda s, t: (along_cross(apt, rwy, s.x, s.y)[0]
                           * GLIDE_FT_PER_NM + 50.0) <= 3000.0)]
    ga_height = float(rng.uniform(150.0, 800.0)) if go_around else None
    segs += _final_segments(apt, rwy, ga_height, rng)
    rows = fly(st, segs)
    if not go_around:
        return rows
    # after the initial climb: left crosswind, downwind, base, final
    if not reapproach:
        rows += fly(st, [
            Segment(track=(b - 90) % 360, gs=220.0, z=apt.elev + 8000.0,
                    vrate=2000.0,
                    done=lambda s, t: math.hypot(s.x, s.y) > 26.0)])
        return rows
    cw = (b - 90) % 360
    dw = (b + 180) % 360
    base = (b + 90) % 360
    pattern_alt = apt.elev + 3000.0
    rows += fly(st, [
        Segment(track=cw, gs=180.0, z=pattern_alt, vrate=1500.0,
                done=lambda s, t: abs(along_cross(apt, rwy, s.x, s.y)[1])
                >= 4.0),
        Segment(track=dw, gs=180.0, z=pattern_alt, vrate=500.0,
                done=lambda s, t: along_cross(apt, rwy, s.x, s.y)[0]
                >= 9.0),
        Segment(track=base, gs=160.0, z=apt.elev + 2500.0, vrate=800.0,
                done=lambda s, t: abs(along_cross(apt, rwy, s.x, s.y)[1])
                <= 0.6),
        Segment(track=b, gs=150.0, z=apt.elev + 2500.0, vrate=500.0,
                done=lambda s, t: abs(wrap180(s.track - b)) < 2.0),
    ])
    rows += fly(st, _final_segments(apt, rwy, None, rng))
    return rows


def departure(apt: Airport, rwy: str, rng) -> list:
    b = apt.bearing(rwy)
    tx, ty = apt.threshold_xy(rwy)
    st = State(tx, ty, apt.elev, 0.0, b, onground=True)
    heading = (b + rng.choice([-1, 1]) * rng.uniform(20.0, 110.0)) % 360
    return fly(st, [
        Segment(track=b, gs=150.0, z=apt.elev, vrate=0.0, onground=True,
                done=lambda s, t: s.gs >= 140.0, max_s=90),
        Segment(track=b, gs=180.0, z=apt.elev + 1500.0, vrate=2500.0,
                done=lambda s, t: s.z >= apt.elev + 1500.0),
        Segment(track=heading, gs=250.0, z=apt.elev + 10000.0, vrate=2200.0,
                done=lambda s, t: math.hypot(s.x, s.y) > 26.0),
    ])


def overflight(apt: Airport, rng) -> list:
    ang = rng.uniform(0, 2 * math.pi)
    x0, y0 = 26.0 * math.cos(ang), 26.0 * math.sin(ang)
    px, py = rng.uniform(-8, 8), rng.uniform(-8, 8)
    track = math.degrees(math.atan2(px - x0, py - y0)) % 360
    z = apt.elev + rng.uniform(5000.0, 11000.0)
    st = State(x0, y0, z, 250.0, track)
    return fly(st, [Segment(
        track=track, gs=250.0, z=z, vrate=0.0,
        done=lambda s, t: t > 60 and math.hypot(s.x, s.y) > 26.0)])


def intruder_pair(apt: Airport, rng) -> tuple[list, list]:
    """Two aircraft passing each other closely (crossing or head-on) at
    a random point in the volume: a known close pair, used to inject
    encounters at a known rate."""
    px, py = rng.uniform(-7, 7), rng.uniform(-7, 7)
    z = apt.elev + rng.uniform(1500.0, 3500.0)
    theta = rng.uniform(0, 360)
    kind = rng.choice(["crossing", "head_on"])
    dpsi = 90.0 if kind == "crossing" else 180.0
    miss = rng.uniform(0.2, 2.5)
    dz = rng.uniform(0.0, 600.0) * rng.choice([-1, 1])
    gs = 160.0
    reach = 10.0

    def straight(track, cx, cy, alt):
        b = math.radians(track)
        st = State(cx - reach * math.sin(b), cy - reach * math.cos(b),
                   alt, gs, track)
        return fly(st, [Segment(track=track, gs=gs, z=alt, vrate=0.0,
                                done=lambda s, t: t >= 2 * reach / gs
                                * 3600.0)])

    tb = (theta + dpsi) % 360
    nb = math.radians(tb + 90.0)
    rows_a = straight(theta, px, py, z)
    rows_b = straight(tb, px + miss * math.sin(nb), py + miss * math.cos(nb),
                      z + dz)
    return rows_a, rows_b


# ----------------------------------------------------------- the day ----

def _hex_icao(rng) -> str:
    return "".join(rng.choice(list("0123456789abcdef"), 6))


def _to_frame(apt: Airport, rows, t_start: pd.Timestamp, icao24: str,
              callsign: str, rng, baro_bias: float) -> pd.DataFrame:
    """Add ADS-B realism: noise, quantisation, dropouts, coverage floor."""
    arr = np.asarray(rows, dtype=float)
    n = len(arr)
    x, y, z, gs, track, vr, onground = arr.T
    lat, lon = apt.latlon(x + rng.normal(0, 0.005, n),
                          y + rng.normal(0, 0.005, n))
    geo = z + rng.normal(0, 20.0, n) + rng.normal(0, 30.0)
    baro = np.round((z + baro_bias + rng.normal(0, 10.0, n)) / 25.0) * 25.0
    keep = rng.random(n) > 0.03                       # random dropouts
    agl = z - apt.elev
    keep &= ~((agl < 80.0) & (rng.random(n) < 0.35))  # coverage floor
    # occasional longer gaps (12-30 s)
    for start in np.flatnonzero(rng.random(n) < 0.0015):
        keep[start:start + int(rng.integers(12, 31))] = False
    idx = np.flatnonzero(keep)
    ts = t_start.floor("s") + pd.to_timedelta(idx, unit="s")
    return pd.DataFrame({
        "timestamp": ts,
        "icao24": icao24,
        "callsign": callsign,
        "latitude": lat[idx],
        "longitude": lon[idx],
        "altitude": baro[idx],
        "geoaltitude": geo[idx],
        "vertical_rate": vr[idx] + rng.normal(0, 60.0, len(idx)),
        "groundspeed": gs[idx] + rng.normal(0, 1.5, len(idx)),
        "track": (track[idx] + rng.normal(0, 1.0, len(idx))) % 360,
        "onground": onground[idx].astype(bool),
    })


def _schedule(rng, n: int, tz: str, day: pd.Timestamp,
              min_gap_s: float) -> np.ndarray:
    """n movement times (seconds from 00:00 UTC) following the local-hour
    demand profile, with a minimum spacing."""
    local_midnight = day.tz_convert(tz).normalize()
    offset = (local_midnight.tz_convert("UTC") - day).total_seconds()
    hours = rng.choice(24, size=n, p=HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum())
    secs = np.sort(hours * 3600.0 + rng.uniform(0, 3600.0, n) + offset)
    for i in range(1, len(secs)):
        if secs[i] - secs[i - 1] < min_gap_s:
            secs[i] = secs[i - 1] + min_gap_s
    return secs % 86400.0


def make_day(apt: Airport, day: pd.Timestamp, rng, movements: int = 200,
             ga_rate: float = 0.03, flows: dict | None = None,
             intruder_rate_per_h: float = 0.0, post_ga_irr: float = 1.0,
             window_min: float = 10.0) -> tuple[pd.DataFrame, dict]:
    """One synthetic day. Returns the raw frame and a dict of truth
    (go-around times, injected pairs)."""
    day = pd.Timestamp(day).tz_localize("UTC") if day.tzinfo is None \
        else day.tz_convert("UTC")
    if flows is None:
        flows = apt.flows or _auto_flows(apt)
    flow = rng.choice(list(flows))
    rwys = flows[flow]
    n_arr = int(movements * 0.5)
    n_dep = int(movements * 0.42)
    n_ovf = movements - n_arr - n_dep
    baro_bias = float(rng.uniform(-250.0, 250.0))
    frames, truth = [], {"flow": flow, "go_arounds": [], "intruders": []}
    seq = int(rng.integers(100, 900))

    for t in _schedule(rng, n_arr, apt.tz, day, 75.0):
        rwy = str(rng.choice(rwys))
        is_ga = rng.random() < ga_rate
        rows = arrival(apt, rwy, rng, is_ga, reapproach=rng.random() < 0.7)
        # the leg starts 12 NM out, i.e. ~4.5 min before the threshold
        t_start = day + pd.Timedelta(seconds=float(t) - 270.0)
        seq += 1
        frames.append(_to_frame(apt, rows, t_start, _hex_icao(rng),
                                f"SYN{seq}", rng, baro_bias))
        if is_ga:
            truth["go_arounds"].append(
                {"t_approx": day + pd.Timedelta(seconds=float(t)),
                 "runway": rwy, "callsign": f"SYN{seq}"})
    for t in _schedule(rng, n_dep, apt.tz, day, 60.0):
        rwy = str(rng.choice(rwys))
        rows = departure(apt, rwy, rng)
        seq += 1
        frames.append(_to_frame(apt, rows, day + pd.Timedelta(seconds=float(t)),
                                _hex_icao(rng), f"SYN{seq}", rng, baro_bias))
    for t in _schedule(rng, n_ovf, apt.tz, day, 30.0):
        rows = overflight(apt, rng)
        seq += 1
        frames.append(_to_frame(apt, rows, day + pd.Timedelta(seconds=float(t)),
                                _hex_icao(rng), f"OVF{seq}", rng, baro_bias))

    # injected close pairs: a Poisson process at intruder_rate_per_h, with
    # the rate multiplied by post_ga_irr during the post window of every
    # go-around (a known effect the analysis must recover)
    if intruder_rate_per_h > 0:
        n_base = rng.poisson(intruder_rate_per_h * 24.0)
        times = list(rng.uniform(0, 86400.0, n_base))
        if post_ga_irr != 1.0:
            extra = (post_ga_irr - 1.0) * intruder_rate_per_h * window_min / 60
            for g in truth["go_arounds"]:
                k = rng.poisson(max(extra, 0.0))
                t0 = (g["t_approx"] - day).total_seconds()
                times += list(t0 + rng.uniform(30.0, window_min * 60 - 30, k))
        for t in times:
            ra, rb = intruder_pair(apt, rng)
            t_cpa = day + pd.Timedelta(seconds=float(t))
            t_start = t_cpa - pd.Timedelta(seconds=len(ra) / 2)
            for rows in (ra, rb):
                seq += 1
                frames.append(_to_frame(apt, rows, t_start, _hex_icao(rng),
                                        f"INT{seq}", rng, baro_bias))
            truth["intruders"].append({"t_cpa": t_cpa})

    df = pd.concat(frames, ignore_index=True)
    # keep the file to this UTC day only, as real daily files are
    df = df[(df["timestamp"] >= day)
            & (df["timestamp"] < day + pd.Timedelta(days=1))]
    df = df.sort_values(["timestamp", "icao24"]).reset_index(drop=True)
    df["timestamp"] = df["timestamp"].astype("datetime64[ns, UTC]")
    return df, truth


def _auto_flows(apt: Airport) -> dict:
    """Group runway ends into flows by heading (same idea as the pipeline's
    automatic flow derivation), keeping only the parallel set(s)."""
    groups: dict[str, list[str]] = {}
    for name, (_, _, b) in apt.runways.items():
        key = None
        for k in groups:
            if abs(wrap180(b - apt.runways[groups[k][0]][2])) <= 30.0:
                key = k
        if key is None:
            groups[f"flow_{int(round(b / 10.0)) * 10:03d}"] = [name]
        else:
            groups[key].append(name)
    return groups


def write_dataset(name: str, days: int, start: str, seed: int,
                  movements: int, ga_rate: float, intruder_rate: float,
                  post_ga_irr: float, out_root: Path | None = None) -> Path:
    icao, label = name.split("_", 1)
    apt = load_airport(icao)
    out = (out_root or ROOT / "datasets") / name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truth_rows = []
    day0 = pd.Timestamp(start)
    for i in range(days):
        day = day0 + pd.Timedelta(days=i)
        df, truth = make_day(apt, day, rng, movements, ga_rate,
                             intruder_rate_per_h=intruder_rate,
                             post_ga_irr=post_ga_irr)
        path = out / f"{label}_{day:%Y%m%d}.parquet"
        df.to_parquet(path, index=False)
        for g in truth["go_arounds"]:
            truth_rows.append({"day": f"{day:%Y%m%d}", **g,
                               "flow": truth["flow"]})
        print(f"{path.name}: {len(df):,} rows, {len(truth['go_arounds'])} "
              f"go-arounds, {len(truth['intruders'])} injected pairs, "
              f"flow {truth['flow']}", flush=True)
    # (json, not csv: the dataset folder must hold nothing the pipelines
    # could mistake for a daily file)
    pd.DataFrame(truth_rows).to_json(out / "truth_go_arounds.json",
                                     orient="records", date_format="iso")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", help="dataset name <ICAO>_<label>")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--start", default="2025-03-01")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--movements", type=int, default=200,
                    help="movements per day (arrivals + departures + "
                         "overflights)")
    ap.add_argument("--ga-rate", type=float, default=0.03,
                    help="share of arrivals that go around")
    ap.add_argument("--intruder-rate", type=float, default=0.0,
                    help="injected close pairs per hour (0 = none)")
    ap.add_argument("--post-ga-irr", type=float, default=1.0,
                    help="rate ratio of injected pairs in post-go-around "
                         "windows (1 = no effect)")
    args = ap.parse_args()
    out = write_dataset(args.name, args.days, args.start, args.seed,
                        args.movements, args.ga_rate, args.intruder_rate,
                        args.post_ga_irr)
    print(f"wrote {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
