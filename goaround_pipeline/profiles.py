"""
Airport profiles: per-airport YAML files under airports/ that supply the
geometry (runways, elevation, location), the operating-environment preset,
and any parameter overrides. Loading a profile configures the pipeline by
setting attributes on the config module, which every stage reads at runtime.

A new airport's profile can be generated automatically from the OurAirports
database (via the traffic library) with create_profile("KBNA").
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from . import config

AIRPORTS_DIR = Path(__file__).resolve().parent.parent / "airports"

# Operating-environment presets: named bundles of threshold overrides.
# `training_ga` is the calibrated baseline (config.py defaults, tuned on the
# KDAB training-fleet dataset). `air_carrier` adapts the speed-dependent
# thresholds to jet/turboprop operations at Class B/C/D airline airports.
PRESETS: dict[str, dict] = {
    "training_ga": {},
    "air_carrier": {
        # jets cross the fence at 120-140 kt and roll out through 80 kt;
        # a GA-style 50 kt gate would fire far too late (or never, with an
        # early high-speed turnoff)
        "TOUCHDOWN_GS_KT": 80.0,
        # airliners fly longer stabilized finals; extending the alignment
        # window improves approach capture without extra false positives
        "APPROACH_MAX_DIST_NM": 8.0,
    },
}


def profile_path(icao: str) -> Path:
    return AIRPORTS_DIR / f"{icao.upper()}.yaml"


def load_profile(icao: str) -> dict:
    """Read airports/<ICAO>.yaml and configure the pipeline from it."""
    path = profile_path(icao)
    if not path.exists():
        raise FileNotFoundError(
            f"No airport profile at {path}. Generate one with:\n"
            f"    python run_analysis.py new-airport {icao.upper()}"
        )
    with open(path) as f:
        prof = yaml.safe_load(f)

    preset = prof.get("preset", "training_ga")
    if preset not in PRESETS:
        raise ValueError(
            f"{path}: unknown preset '{preset}' "
            f"(available: {', '.join(PRESETS)})"
        )

    # airport geometry
    config.AIRPORT_ICAO = prof["icao"].upper()
    config.AIRPORT_LATLON = (float(prof["latitude"]),
                             float(prof["longitude"]))
    config.FIELD_ELEVATION_FT = float(prof["elevation_ft"])
    config.LOCAL_TZ = prof.get("timezone", "UTC")
    config.RUNWAYS = {
        str(name): (float(v[0]), float(v[1]), float(v[2]))
        for name, v in prof["runways"].items()
    }

    # environment
    config.PRESET = preset
    config.TERRAIN_MODE = prof.get("terrain", "flat")
    config.ASSUME_ARRIVALS_DATASET = bool(
        prof.get("assume_arrivals_dataset", False)
    )

    # preset overrides, then explicit per-airport overrides on top
    applied = dict(PRESETS[preset])
    applied.update(prof.get("overrides") or {})
    for key, value in applied.items():
        if not hasattr(config, key):
            raise ValueError(f"{path}: override '{key}' is not a known "
                             "config parameter")
        setattr(config, key, value)

    _warn_parallel_spacing()

    if config.TERRAIN_MODE == "dem":
        from . import terrain
        terrain.ensure_dem(config.AIRPORT_ICAO, *config.AIRPORT_LATLON)

    return prof


def _parallel_spacing_nm() -> float | None:
    """Smallest lateral spacing between parallel runway centerlines (NM)."""
    items = list(config.RUNWAYS.items())
    best = None
    for i, (na, (lata, lona, ba)) in enumerate(items):
        for nb, (latb, lonb, bb) in items[i + 1:]:
            d = abs((ba - bb + 180.0) % 360.0 - 180.0)
            if d > 10.0:
                continue  # not parallel
            # perpendicular offset of b's threshold from a's centerline
            coslat = math.cos(math.radians(lata))
            dx = (lonb - lona) * 60.0 * coslat
            dy = (latb - lata) * 60.0
            b = math.radians(ba)
            cross = abs(dx * math.cos(b) - dy * math.sin(b))
            if 0.01 < cross and (best is None or cross < best):
                best = cross
    return best


def _warn_parallel_spacing() -> None:
    spacing = _parallel_spacing_nm()
    if spacing is not None and config.APPROACH_MAX_XTRACK_NM > 0.6 * spacing:
        print(f"note: parallel runways ~{spacing:.2f} NM apart; the "
              f"{config.APPROACH_MAX_XTRACK_NM:.2f} NM cross-track window "
              "spans both centerlines. Runway assignment relies on the "
              "low-altitude-weighted vote (this is normal, and correct "
              "for KDAB-style close parallels).")


def create_profile(icao: str) -> Path:
    """Generate airports/<ICAO>.yaml from the OurAirports database via the
    traffic library. The generated file should be reviewed by hand:
    timezone, preset, terrain mode, and assume_arrivals_dataset in
    particular."""
    icao = icao.upper()
    from traffic.data import airports  # network on first use, then cached

    apt = airports[icao]
    if apt is None:
        raise ValueError(f"airport {icao} not found in OurAirports data")
    rwy = apt.runways.data

    lines = [
        f"# Airport profile: {apt.name} ({icao})",
        "# AUTO-GENERATED from ourairports.com via the traffic library.",
        "# REVIEW BEFORE USE - especially: timezone (auto-set to UTC),",
        "# preset, terrain, and assume_arrivals_dataset.",
        "",
        f"icao: {icao}",
        f"name: {apt.name}",
        f"latitude: {apt.latlon[0]}",
        f"longitude: {apt.latlon[1]}",
        f"elevation_ft: {float(apt.altitude)}",
        "timezone: UTC          # EDIT: e.g. America/Chicago",
        "",
        "# training_ga | air_carrier",
        "preset: air_carrier    # EDIT if this is a GA/training field",
        "",
        "# flat | dem  (dem = terrain-model AGL, auto-downloaded on first "
        "use)",
        "terrain: flat",
        "",
        "# true only for datasets queried as ARRIVALS at this airport",
        "assume_arrivals_dataset: false",
        "",
        "# name: [threshold_latitude, threshold_longitude, true_bearing_deg]",
        "runways:",
    ]
    for _, r in rwy.iterrows():
        lines.append(f'  "{r["name"]}": [{r.latitude}, {r.longitude}, '
                     f'{r.bearing}]')
    lines += ["", "overrides: {}", ""]

    AIRPORTS_DIR.mkdir(exist_ok=True)
    path = profile_path(icao)
    path.write_text("\n".join(lines))
    return path
