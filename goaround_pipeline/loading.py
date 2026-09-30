"""
Stage 1: load daily raw files, clean, and split into flight legs.

A "flight leg" is one continuous airborne operation by one aircraft: the unit
we search for approaches. Training aircraft fly several legs per day and many
approaches per leg, so segmentation is by (icao24, time gap), never by day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

RAW_COLUMNS = [
    "timestamp", "icao24", "callsign", "latitude", "longitude",
    "altitude", "geoaltitude", "vertical_rate", "groundspeed",
    "track", "onground",
]


def load_day(path) -> pd.DataFrame:
    """Load one daily file (parquet or csv) with the pipeline's columns."""
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS)
    else:
        df = pd.read_parquet(path, columns=RAW_COLUMNS)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    # pyarrow-backed dtypes -> plain numpy for speed and compatibility
    for col in ["latitude", "longitude", "altitude", "geoaltitude",
                "vertical_rate", "groundspeed", "track"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["onground"] = df["onground"].astype("boolean").fillna(False).astype(bool)
    df["callsign"] = df["callsign"].astype(str)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    ).astype("datetime64[ns, UTC]")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df.sort_values(["icao24", "timestamp"]).reset_index(drop=True)
    return df


def _smooth(series: pd.Series, window_s: int) -> pd.Series:
    # data is ~1 Hz, so a point-based window approximates a time window
    return series.rolling(window_s, center=True, min_periods=1).median()


def _ground_elevation(leg: pd.DataFrame):
    """Ground reference under each point (ft MSL): the field elevation in
    `flat` terrain mode, or a terrain-model sample in `dem` mode."""
    if config.TERRAIN_MODE == "dem":
        from . import terrain
        return terrain.ground_elevation_ft(
            config.AIRPORT_ICAO,
            leg["latitude"].to_numpy(),
            leg["longitude"].to_numpy(),
        )
    return config.FIELD_ELEVATION_FT


def split_into_legs(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a day's data into per-aircraft flight legs on time gaps."""
    legs = []
    gap = pd.Timedelta(minutes=config.SEGMENT_GAP_MINUTES)
    for icao24, g in df.groupby("icao24", sort=False):
        g = g.sort_values("timestamp")
        new_leg = (g["timestamp"].diff() > gap).to_numpy(dtype=bool)
        for _, leg in g.groupby(np.cumsum(new_leg)):
            if len(leg) < config.MIN_LEG_POINTS:
                continue
            leg = leg.reset_index(drop=True).copy()
            ground = _ground_elevation(leg)
            leg["agl"] = _smooth(
                leg["geoaltitude"] - ground,
                config.ALT_SMOOTH_WINDOW_S,
            )
            # fall back to baro altitude where geoaltitude is missing;
            # baro is QNH-uncorrected so re-reference it per leg by matching
            # its median offset from geoaltitude where both exist
            missing = leg["agl"].isna()
            if missing.any() and leg["geoaltitude"].notna().any():
                offset = (leg["geoaltitude"] - leg["altitude"]).median()
                fallback = leg["altitude"] + offset - ground
                leg.loc[missing, "agl"] = _smooth(
                    fallback, config.ALT_SMOOTH_WINDOW_S
                )[missing]
            leg["vrate"] = _smooth(
                leg["vertical_rate"], config.VRATE_SMOOTH_WINDOW_S
            )
            callsign = (
                leg["callsign"].mode().iat[0]
                if leg["callsign"].notna().any() else ""
            )
            leg.attrs["leg_id"] = (
                f"{icao24}_{leg['timestamp'].iat[0]:%Y%m%d_%H%M%S}"
            )
            leg.attrs["icao24"] = icao24
            leg.attrs["callsign"] = str(callsign).strip()
            legs.append(leg)
    return legs
