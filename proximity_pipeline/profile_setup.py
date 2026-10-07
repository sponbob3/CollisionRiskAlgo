"""
Automatic airport profiles (FRAMEWORK.md section 14).

When airports/<ICAO>.yaml does not exist, run_proximity.py creates it
here, without manual editing:

- geometry (name, reference point, elevation, runway thresholds and true
  bearings) from the OurAirports database (ourairports.com, falling back
  to its GitHub mirror), cached in airports/.ourairports/;
- timezone from the airportsdata package (offline, pure Python; falls
  back to timezonefinder on the coordinates when installed);
- preset (training_ga | air_carrier) from the data: the median
  groundspeed of aircraft on final approach (descending within 4 NM of
  the airport, 200-1,500 ft above field); airliners fly finals at
  ~130-150 kt, training aircraft at ~60-80 kt. Without data: the airport
  type (large airport or scheduled service -> air_carrier);
- assume_arrivals_dataset from the data: whether landings are visible
  on the ground. If most flights whose track ends low near the field show
  no on-ground samples at their end, the ADS-B coverage stops before
  touchdown, and a track ending low near the field is then treated as a
  landing (the go-around pipeline's coverage-truncation rule);
- the ADS-B coverage floor near the field, from the data: the height at
  which tracks ending near the field are last seen. When it is above the
  go-around pipeline's default landing-truncation heights, those two
  heights are raised (profile `overrides`) so that landings whose data
  stops at the coverage floor are still recognised as landings;
- the airspace block with the ceiling at field elevation + 4,000 ft (the
  standard Class C ceiling; the run's ceiling check raises it from the
  data when needed).

Every detected value is written into the profile with the evidence it
came from, and printed. The profile is the single place to change any of
them; it is never overwritten once it exists.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

AIRPORTS_DIR = config.ROOT / "airports"
OA_CACHE = AIRPORTS_DIR / ".ourairports"
OA_SOURCES = (
    "https://ourairports.com/data/",
    "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/",
)

# data-driven detection (section 14)
DETECT_MAX_DAYS = 5                  # days sampled, spread over the dataset
PRESET_FINAL_RADIUS_NM = 4.0
PRESET_FINAL_HEIGHT_FT = (200.0, 1500.0)
PRESET_FINAL_MAX_VRATE_FPM = -300.0  # descending
PRESET_AIR_CARRIER_MIN_GS_KT = 100.0
PRESET_MIN_SAMPLES = 200
ARRIVAL_END_RADIUS_NM = 3.0
ARRIVAL_END_MAX_HEIGHT_FT = 1000.0
ARRIVAL_END_LOOKBACK_S = 120.0
ARRIVAL_GROUND_VISIBLE_MIN = 0.5
ARRIVAL_MIN_LEGS = 10
LEG_GAP_S = 20 * 60
# coverage floor: this percentile of the last-seen heights, plus a margin,
# rounded up to 50 ft, sets the landing-truncation heights when it exceeds
# the go-around pipeline's defaults; capped (a floor above the cap means
# the data cannot support landing detection - reported, not hidden)
COVERAGE_FLOOR_PERCENTILE = 90
COVERAGE_FLOOR_MARGIN_FT = 100.0
COVERAGE_FLOOR_CAP_FT = 1200.0


# --------------------------------------------------------- OurAirports --

def _download(name: str) -> bytes:
    import httpx
    errors = []
    for base in OA_SOURCES:
        try:
            r = httpx.get(base + name, follow_redirects=True, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:   # try the next source, report if all fail
            errors.append(f"{base}{name}: {e}")
    raise RuntimeError("could not download the OurAirports database:\n  "
                       + "\n  ".join(errors))


def ourairports_table(name: str) -> pd.DataFrame:
    """airports.csv / runways.csv from OurAirports, cached locally."""
    OA_CACHE.mkdir(parents=True, exist_ok=True)
    path = OA_CACHE / name
    if not path.exists():
        print(f"downloading OurAirports {name} (first use only) ...",
              flush=True)
        path.write_bytes(_download(name))
    return pd.read_csv(path, low_memory=False)


def airport_record(icao: str) -> dict:
    """Name, reference point, elevation, type and runway ends of an
    airport. Runway ends: {name: (threshold lat, threshold lon, true
    bearing)}, bearing = geodesic azimuth from this end to the opposite
    end (closed runways and ends without coordinates are skipped)."""
    import pyproj
    icao = icao.upper()
    apt = ourairports_table("airports.csv")
    sel = apt[apt["ident"].astype(str).str.upper() == icao]
    if sel.empty and "gps_code" in apt:
        sel = apt[apt["gps_code"].astype(str).str.upper() == icao]
    if sel.empty:
        raise ValueError(f"airport {icao} not found in the OurAirports "
                         "database")
    a = sel.iloc[0]
    rw = ourairports_table("runways.csv")
    rw = rw[(rw["airport_ident"].astype(str).str.upper()
             == str(a["ident"]).upper())]
    geod = pyproj.Geod(ellps="WGS84")
    runways, skipped = {}, []
    for _, r in rw.iterrows():
        if int(r.get("closed", 0) or 0) == 1:
            skipped.append(f"{r['le_ident']}/{r['he_ident']} (closed)")
            continue
        ends = [(r["le_ident"], r["le_latitude_deg"], r["le_longitude_deg"]),
                (r["he_ident"], r["he_latitude_deg"], r["he_longitude_deg"])]
        if any(pd.isna(v) for e in ends for v in e):
            skipped.append(f"{r['le_ident']}/{r['he_ident']} "
                           "(no threshold coordinates)")
            continue
        (n1, la1, lo1), (n2, la2, lo2) = ends
        az12, az21, _ = geod.inv(lo1, la1, lo2, la2)
        runways[str(n1)] = (float(la1), float(lo1), az12 % 360.0)
        runways[str(n2)] = (float(la2), float(lo2), az21 % 360.0)
    if not runways:
        raise ValueError(f"{icao}: no open runway with threshold "
                         "coordinates in the OurAirports database")
    return {
        "icao": icao,
        "name": str(a["name"]),
        "latitude": float(a["latitude_deg"]),
        "longitude": float(a["longitude_deg"]),
        "elevation_ft": float(a["elevation_ft"]),
        "type": str(a.get("type", "")),
        "scheduled_service": str(a.get("scheduled_service", "")) == "yes",
        "runways": runways,
        "skipped_runways": skipped,
    }


def timezone_of(icao: str, lat: float, lon: float) -> str:
    """IANA timezone of the airport: the airportsdata table (offline, no
    compiled code), else timezonefinder on the coordinates if installed,
    else UTC with a warning."""
    try:
        import airportsdata
        rec = airportsdata.load("ICAO").get(icao.upper())
        if rec and rec.get("tz"):
            return rec["tz"]
    except ImportError:
        pass
    try:
        from timezonefinder import TimezoneFinder
        tz = TimezoneFinder().timezone_at(lat=lat, lng=lon)
        if tz:
            return tz
    except ImportError:
        pass
    print(f"warning: no timezone found for {icao}; set `timezone:` in the "
          "profile (it is used for the hour-of-day baseline covariate)")
    return "UTC"


# ------------------------------------------------- data-driven checks --

def _distance_nm(lat, lon, lat0, lon0):
    coslat = math.cos(math.radians(lat0))
    return np.hypot((lat - lat0) * 60.0, (lon - lon0) * 60.0 * coslat)


def sample_files(dataset: Path) -> list[Path]:
    from . import goaround_adapter as ga
    files = ga.data_files(dataset)
    if len(files) <= DETECT_MAX_DAYS:
        return files
    idx = np.linspace(0, len(files) - 1, DETECT_MAX_DAYS).round().astype(int)
    return [files[i] for i in sorted(set(idx))]


def detect_from_data(rec: dict, dataset: Path) -> dict:
    """Preset and arrivals assumption from a sample of days (fresh
    position reports only)."""
    from . import goaround_adapter as ga
    files = sample_files(dataset)
    gs_final, n_end, n_end_ground, end_heights = [], 0, 0, []
    for f in files:
        df = ga.load_day(f)
        if df.empty:
            continue
        d = _distance_nm(df["latitude"].to_numpy(), df["longitude"].to_numpy(),
                         rec["latitude"], rec["longitude"])
        h = (df["geoaltitude"].fillna(df["altitude"]).to_numpy()
             - rec["elevation_ft"])
        lo, hi = PRESET_FINAL_HEIGHT_FT
        fin = ((d < PRESET_FINAL_RADIUS_NM) & (h >= lo) & (h <= hi)
               & (df["vertical_rate"].to_numpy() <= PRESET_FINAL_MAX_VRATE_FPM)
               & ~df["onground"].to_numpy())
        gs_final.append(df["groundspeed"].to_numpy()[fin])

        # legs whose track ends low near the field: ground visible?
        t = df["timestamp"].astype("int64").to_numpy() // 10**9
        ac = df["icao24"].to_numpy()
        new_leg = np.ones(len(df), dtype=bool)
        new_leg[1:] = (ac[1:] != ac[:-1]) | (np.diff(t) > LEG_GAP_S)
        leg = np.cumsum(new_leg)
        last = np.flatnonzero(np.r_[new_leg[1:], True])
        onground = df["onground"].to_numpy()
        for i in last:
            if not (d[i] < ARRIVAL_END_RADIUS_NM
                    and h[i] < ARRIVAL_END_MAX_HEIGHT_FT):
                continue
            n_end += 1
            end_heights.append(h[i])
            j = i
            while (j > 0 and leg[j - 1] == leg[i]
                   and t[i] - t[j - 1] <= ARRIVAL_END_LOOKBACK_S):
                j -= 1
            if onground[j:i + 1].any():
                n_end_ground += 1
    gs = np.concatenate(gs_final) if gs_final else np.array([])
    gs = gs[np.isfinite(gs)]
    out = {"days_sampled": len(files), "final_samples": int(len(gs)),
           "final_gs_median_kt": float(np.median(gs)) if len(gs) else np.nan,
           "legs_ending_low": n_end,
           "ground_visible_share": (n_end_ground / n_end if n_end else np.nan),
           "end_height_median_ft": (float(np.median(end_heights))
                                    if end_heights else np.nan),
           "end_height_p_ft": (float(np.percentile(
               end_heights, COVERAGE_FLOOR_PERCENTILE))
               if end_heights else np.nan)}
    if len(gs) >= PRESET_MIN_SAMPLES:
        out["preset"] = ("air_carrier"
                         if out["final_gs_median_kt"]
                         >= PRESET_AIR_CARRIER_MIN_GS_KT else "training_ga")
        out["preset_reason"] = (
            f"median final-approach groundspeed "
            f"{out['final_gs_median_kt']:.0f} kt over {len(gs):,} samples "
            f"in {len(files)} day(s) (air_carrier at >= "
            f"{PRESET_AIR_CARRIER_MIN_GS_KT:.0f} kt)")
    if n_end >= ARRIVAL_MIN_LEGS:
        share = out["ground_visible_share"]
        out["assume_arrivals_dataset"] = bool(
            share < ARRIVAL_GROUND_VISIBLE_MIN)
        out["arrivals_reason"] = (
            f"{100 * share:.0f}% of {n_end} tracks ending low near the field "
            f"show on-ground samples (median last height "
            f"{out['end_height_median_ft']:.0f} ft above field); "
            + ("landings are mostly NOT seen on the ground, so a track "
               "ending low near the field is treated as a landing"
               if out["assume_arrivals_dataset"] else
               "landings are seen on the ground, so touchdown evidence "
               "decides"))
    if n_end >= ARRIVAL_MIN_LEGS and out["assume_arrivals_dataset"]:
        from . import goaround_adapter as ga
        need = (math.ceil((out["end_height_p_ft"] + COVERAGE_FLOOR_MARGIN_FT)
                          / 50.0) * 50.0)
        defaults = {"END_TRUNCATED_MAX_AGL_FT":
                    ga.config_defaults("END_TRUNCATED_MAX_AGL_FT"),
                    "TRUNCATED_FINAL_MAX_AGL_FT":
                    ga.config_defaults("TRUNCATED_FINAL_MAX_AGL_FT")}
        raised = {k: min(need, COVERAGE_FLOOR_CAP_FT)
                  for k, v in defaults.items() if need > v}
        if raised:
            out["overrides"] = raised
            out["overrides_reason"] = (
                f"ADS-B coverage near the field ends at "
                f"~{out['end_height_median_ft']:.0f} ft above field (median; "
                f"P{COVERAGE_FLOOR_PERCENTILE} "
                f"{out['end_height_p_ft']:.0f} ft), above the go-around "
                f"pipeline's landing-truncation defaults "
                f"({', '.join(f'{k} {v:.0f}' for k, v in defaults.items())})"
                + (" - CAPPED: landing detection is unreliable at this "
                   "coverage" if need > COVERAGE_FLOOR_CAP_FT else ""))
    return out


# --------------------------------------------------------- the profile --

def profile_path(icao: str) -> Path:
    return AIRPORTS_DIR / f"{icao.upper()}.yaml"


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(config.ROOT))
    except ValueError:
        return str(path)


def airspace_block_lines(elevation_ft: float, cls: str = "unknown") -> list:
    ceiling = int(round((elevation_ft + 4000.0) / 100.0) * 100)
    return [
        "# Study volume for the proximity-risk pipeline (FRAMEWORK.md "
        "section 3).",
        "# The go-around pipeline ignores this block.",
        "airspace:",
        f"  class: {cls}             # informational only (B / C / D)",
        f"  ceiling_ft_msl: {ceiling}       # field elevation "
        f"({elevation_ft:.0f} ft) + 4,000 ft, the",
        "                             # standard Class C ceiling; the run's "
        "ceiling",
        "                             # check raises it from the data when "
        "needed",
        "  radius_nm: 10",
        "  ceiling_mode: auto         # auto | fixed   (section 3.3)",
    ]


def create_profile(icao: str, dataset: Path | None = None) -> tuple[
        Path, list[str]]:
    """Write airports/<ICAO>.yaml (never overwrites). Returns the path and
    the console lines describing every detected value."""
    icao = icao.upper()
    path = profile_path(icao)
    if path.exists():
        raise FileExistsError(f"{path} already exists; delete it to "
                              "regenerate")
    rec = airport_record(icao)
    tz = timezone_of(icao, rec["latitude"], rec["longitude"])
    det = detect_from_data(rec, dataset) if dataset is not None else {}

    if "preset" in det:
        preset, preset_why = det["preset"], det["preset_reason"]
    else:
        preset = ("air_carrier" if rec["type"] == "large_airport"
                  or rec["scheduled_service"] else "training_ga")
        preset_why = (f"airport type '{rec['type']}', scheduled service "
                      f"{'yes' if rec['scheduled_service'] else 'no'}"
                      + ("" if dataset is None else
                         " (too few final-approach samples in the data)"))
    if "assume_arrivals_dataset" in det:
        arrivals, arr_why = (det["assume_arrivals_dataset"],
                             det["arrivals_reason"])
    else:
        arrivals = False
        arr_why = ("no data to check" if dataset is None else
                   "too few tracks ending low near the field to check")

    L = [
        f"# Airport profile: {rec['name']} ({icao})",
        "# AUTO-GENERATED by run_proximity.py. Geometry from the OurAirports",
        "# database; timezone from the airportsdata table; preset and",
        "# assume_arrivals_dataset detected from the data (evidence below).",
        "# This file is the place to change any of these values; it is never",
        "# overwritten (delete it to regenerate).",
        "",
        f"icao: {icao}",
        f"name: {rec['name']}",
        f"latitude: {rec['latitude']}",
        f"longitude: {rec['longitude']}",
        f"elevation_ft: {rec['elevation_ft']}",
        f"timezone: {tz}       # from the airportsdata table",
        "",
        "# training_ga | air_carrier",
        f"# detected: {preset_why}",
        f"preset: {preset}",
        "",
        "# flat | dem  (dem = terrain-model AGL, auto-downloaded on first "
        "use;",
        "# use dem where there is real terrain under the final approaches)",
        "terrain: flat",
        "",
        "# true: a track that ends low near the field counts as a landing "
        "below",
        "# the ADS-B coverage floor (needed when landings are not seen on "
        "the ground)",
        f"# detected: {arr_why}",
        f"assume_arrivals_dataset: {'true' if arrivals else 'false'}",
        "",
        "# name: [threshold_latitude, threshold_longitude, true_bearing_deg]",
        "runways:",
    ]
    for name, (la, lo, b) in rec["runways"].items():
        L.append(f'  "{name}": [{la}, {lo}, {b:.6f}]')
    if rec["skipped_runways"]:
        L.append(f"# skipped: {', '.join(rec['skipped_runways'])}")
    L += [""] + airspace_block_lines(rec["elevation_ft"]) + [
        "",
        "# Optional runway-flow labels (FRAMEWORK.md section 8.4), e.g.",
        "#   flows:",
        '#     north: ["36L", "36R"]',
        '#     south: ["18L", "18R"]',
        "# Without this block flows are derived automatically from runway "
        "headings.",
        "",
        "# Optional overrides of any go-around parameter "
        "(goaround_pipeline/config.py)",
    ]
    ov = det.get("overrides") or {}
    if ov:
        L.append(f"# detected: {det['overrides_reason']}")
        L.append("overrides:")
        L += [f"  {k}: {v:.0f}" for k, v in ov.items()]
    else:
        L.append("overrides: {}")
    L.append("")
    AIRPORTS_DIR.mkdir(exist_ok=True)
    path.write_text("\n".join(L))

    lines = [
        f"created {_rel(path)} for {rec['name']}",
        f"  runways: {', '.join(rec['runways'])}   elevation "
        f"{rec['elevation_ft']:.0f} ft   timezone {tz}",
        f"  preset: {preset}  ({preset_why})",
        f"  assume_arrivals_dataset: {str(arrivals).lower()}  ({arr_why})",
        "  airspace ceiling: field + 4,000 ft (checked against the data "
        "every run)",
    ]
    if ov:
        lines.append(f"  go-around overrides: "
                     f"{', '.join(f'{k}={v:.0f}' for k, v in ov.items())}  "
                     f"({det['overrides_reason']})")
    lines += [
        "  edit the file to change any of these; it is never overwritten",
    ]
    return path, lines


def ensure_profile(icao: str, dataset: Path | None = None) -> None:
    """Create the profile when missing; add a default airspace block to
    an older profile that lacks one."""
    path = profile_path(icao)
    if not path.exists():
        print(f"no airport profile for {icao.upper()}: creating it from "
              "the OurAirports database and the data")
        _, lines = create_profile(icao, dataset)
        print("\n".join(lines))
        print()
        return
    import yaml
    prof = yaml.safe_load(path.read_text()) or {}
    if not prof.get("airspace"):
        text = path.read_text().rstrip("\n") + "\n\n" + "\n".join(
            airspace_block_lines(float(prof["elevation_ft"]))) + "\n"
        path.write_text(text)
        print(f"{_rel(path)}: added the default airspace "
              "block (field elevation + 4,000 ft)")
