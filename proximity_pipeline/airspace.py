"""
Study volume and airport profile extensions (FRAMEWORK.md section 3).

The airport profile airports/<ICAO>.yaml is the go-around pipeline's
format plus an `airspace:` block (charted controlled-airspace ceiling,
radius, ceiling mode) and an optional `flows:` block. load_airspace()
reads those two blocks into proximity_pipeline.config after the vendored
loader has configured the go-around config from the same file.

The ceiling adequacy check (section 3.3) also lives here: it takes the
go-around events and the encounters and decides the effective ceiling.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config
from . import goaround_adapter as ga

AIRPORTS_DIR = config.ROOT / "airports"


def profile_path(icao: str) -> Path:
    return AIRPORTS_DIR / f"{icao.upper()}.yaml"


def load_airspace(icao: str) -> dict:
    """Configure the go-around config from the profile (vendored loader),
    then read the proximity-specific blocks. Returns the raw profile."""
    prof = ga.load_profile(icao)
    path = profile_path(icao)
    block = prof.get("airspace")
    if not block:
        raise ValueError(
            f"{path}: missing the `airspace:` block. Add it (see "
            f"airports/KBNA.yaml) or regenerate the profile with "
            f"`python run_proximity.py new-airport {icao.upper()}`.")
    config.AIRSPACE_CLASS = str(block.get("class", "C"))
    config.AIRSPACE_CEILING_FT_MSL = float(block["ceiling_ft_msl"])
    config.VOLUME_RADIUS_NM = float(block.get("radius_nm",
                                              config.VOLUME_RADIUS_NM))
    mode = str(block.get("ceiling_mode", "auto")).lower()
    if mode not in ("auto", "fixed"):
        raise ValueError(f"{path}: ceiling_mode must be auto or fixed")
    config.AIRSPACE_CEILING_MODE = mode

    flows = prof.get("flows")
    if flows:
        unknown = [r for rws in flows.values() for r in rws
                   if str(r) not in ga.config.RUNWAYS]
        if unknown:
            raise ValueError(f"{path}: flows: reference unknown runway "
                             f"ends {unknown}")
        config.FLOWS = {str(k): [str(r) for r in v]
                        for k, v in flows.items()}
    else:
        config.FLOWS = None

    # proximity-specific overrides (the go-around loader already raised on
    # unknown go-around keys; keys that exist here are applied here)
    for key, value in (prof.get("proximity_overrides") or {}).items():
        if not hasattr(config, key):
            raise ValueError(f"{path}: proximity_overrides '{key}' is not "
                             "a known proximity config parameter")
        setattr(config, key, value)

    check_ceiling_input()
    return prof


def field_elevation_ft() -> float:
    return float(ga.config.FIELD_ELEVATION_FT)


def charted_ceiling_ft_agl() -> float:
    """Charted ceiling converted to feet above field elevation."""
    return config.AIRSPACE_CEILING_FT_MSL - field_elevation_ft()


def check_ceiling_input() -> None:
    """Hard error on an implausible charted ceiling (section 3.2)."""
    agl = charted_ceiling_ft_agl()
    if not (config.CEILING_MIN_FT_AGL <= agl <= config.CEILING_MAX_FT_AGL):
        raise ValueError(
            f"airspace ceiling_ft_msl = {config.AIRSPACE_CEILING_FT_MSL:.0f}"
            f" ft MSL is {agl:.0f} ft above field elevation "
            f"({field_elevation_ft():.0f} ft); expected between "
            f"{config.CEILING_MIN_FT_AGL:.0f} and "
            f"{config.CEILING_MAX_FT_AGL:.0f} ft above field. Check the "
            "profile (ceiling is in ft MSL, from the sectional chart).")


def computation_ceiling_ft_agl() -> float:
    """Ceiling of the volume in which encounters are actually computed:
    the widest ceiling any sweep can ask for, plus the buffer, so that the
    pairwise computation is done once (sections 3.3, 12)."""
    widest = max(config.CEILING_CAP_FT_AGL, charted_ceiling_ft_agl(),
                 *config.CEILING_SENSITIVITY_FT_AGL)
    return widest + config.CEILING_BUFFER_FT


def computation_radius_nm() -> float:
    return max(config.VOLUME_RADIUS_NM, config.VOLUME_RADIUS_SENSITIVITY_NM)


def round_up_500(x: float) -> float:
    return math.ceil(x / config.CEILING_STEP_FT) * config.CEILING_STEP_FT


def ceiling_check(ga_max_heights: pd.Series, ga_t1_heights: pd.Series,
                  ga_beyond_radius_share: float | None) -> dict:
    """Ceiling adequacy check and automatic override (section 3.3).

    ga_max_heights:  per go-around event, max height above field of the
                     go-around aircraft in its post window within the
                     radius (ft).
    ga_t1_heights:   height above field (max of the pair) at the closest
                     point of every go-around-involved T1 encounter (ft).
    Returns a dict with every number the decision used.
    """
    charted = charted_ceiling_ft_agl()
    n_events = int(ga_max_heights.notna().sum())
    p95 = float(np.nanpercentile(ga_max_heights, 95)) if n_events else np.nan

    def spill(ceiling: float) -> float:
        if len(ga_t1_heights) == 0:
            return 0.0
        return float((ga_t1_heights > ceiling).mean())

    spill_charted = spill(charted)
    effective = charted
    reason = "charted ceiling used"
    contained = (n_events == 0) or (p95 <= charted)
    if config.AIRSPACE_CEILING_MODE == "auto" and (
            not contained or spill_charted > config.CEILING_SPILL_MAX):
        cap = config.CEILING_CAP_FT_AGL
        if not contained:
            effective = min(max(charted, round_up_500(p95)), cap)
            reason = "raised to contain go-around climb-outs (P95)"
        while (spill(effective) > config.CEILING_SPILL_MAX
               and effective + config.CEILING_STEP_FT <= cap):
            effective += config.CEILING_STEP_FT
            reason = "raised until encounter spill-over <= limit"
        effective = max(effective, charted)   # never below charted
    elif config.AIRSPACE_CEILING_MODE == "fixed" and (
            not contained or spill_charted > config.CEILING_SPILL_MAX):
        reason = ("ceiling_mode fixed: charted ceiling kept although the "
                  "check failed")
    return {
        "charted_ceiling_ft_msl": config.AIRSPACE_CEILING_FT_MSL,
        "charted_ceiling_ft_agl": charted,
        "field_elevation_ft": field_elevation_ft(),
        "ceiling_mode": config.AIRSPACE_CEILING_MODE,
        "n_go_arounds": n_events,
        "p95_ga_max_height_ft_agl": p95,
        "ga_contained": bool(contained),
        "n_ga_t1_encounters": int(len(ga_t1_heights)),
        "spill_share_at_charted": spill_charted,
        "spill_share_at_effective": spill(effective),
        "spill_limit": config.CEILING_SPILL_MAX,
        "effective_ceiling_ft_agl": effective,
        "effective_ceiling_ft_msl": effective + field_elevation_ft(),
        "override": bool(effective > charted),
        "reason": reason,
        "radius_nm": config.VOLUME_RADIUS_NM,
        "ga_post_time_beyond_radius_share": (
            np.nan if ga_beyond_radius_share is None
            else float(ga_beyond_radius_share)),
    }


def ceiling_check_lines(chk: dict) -> list[str]:
    """Console / report text for the ceiling check."""
    lines = [
        f"ceiling check: charted {chk['charted_ceiling_ft_msl']:.0f} ft MSL "
        f"= {chk['charted_ceiling_ft_agl']:.0f} ft above field "
        f"(mode {chk['ceiling_mode']})",
        f"  go-around post-window max height P95: "
        f"{chk['p95_ga_max_height_ft_agl']:.0f} ft above field "
        f"(n={chk['n_go_arounds']}) -> "
        f"{'contained' if chk['ga_contained'] else 'NOT contained'}",
        f"  go-around T1 encounters above charted ceiling: "
        f"{100 * chk['spill_share_at_charted']:.1f}% "
        f"(limit {100 * chk['spill_limit']:.0f}%, "
        f"n={chk['n_ga_t1_encounters']})",
        f"  go-around post-window time beyond {chk['radius_nm']:.0f} NM: "
        f"{100 * chk['ga_post_time_beyond_radius_share']:.1f}%",
    ]
    if chk["override"]:
        lines.append(
            f"  ** EFFECTIVE CEILING RAISED to "
            f"{chk['effective_ceiling_ft_agl']:.0f} ft above field "
            f"({chk['effective_ceiling_ft_msl']:.0f} ft MSL): "
            f"{chk['reason']}")
    else:
        lines.append(f"  effective ceiling = charted "
                     f"({chk['effective_ceiling_ft_agl']:.0f} ft above "
                     f"field): {chk['reason']}")
    return lines


def create_profile(icao: str) -> Path:
    """new-airport: generate the go-around profile via the vendored
    generator, then append the airspace block with the ceiling pre-filled
    as field elevation + 4,000 ft (standard Class C), to be verified
    against the sectional chart."""
    icao = icao.upper()
    path = ga.create_profile(icao)
    with open(path) as f:
        prof = yaml.safe_load(f)
    elev = float(prof["elevation_ft"])
    ceiling = int(round((elev + 4000.0) / 100.0) * 100)
    block = [
        "",
        "# Study volume for the proximity-risk pipeline (FRAMEWORK.md "
        "section 3).",
        "# The go-around pipeline ignores this block.",
        "airspace:",
        "  class: C                   # EDIT: B / C / D (informational)",
        f"  ceiling_ft_msl: {ceiling}       # PRE-FILLED as field elevation "
        f"({elev:.0f} ft) + 4,000 ft;",
        "                             # VERIFY AGAINST THE SECTIONAL CHART",
        "  radius_nm: 10",
        "  ceiling_mode: auto         # auto | fixed   (section 3.3)",
        "",
        "# Optional runway-flow labels (FRAMEWORK.md section 8.4), e.g.",
        "#   flows:",
        '#     north: ["02L", "02R"]',
        '#     south: ["20L", "20R"]',
        "# Without this block flows are derived automatically from runway "
        "headings.",
        "",
    ]
    text = path.read_text().rstrip("\n") + "\n" + "\n".join(block)
    path.write_text(text)
    return path
