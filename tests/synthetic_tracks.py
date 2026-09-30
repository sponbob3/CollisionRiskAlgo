"""Helpers to build a DayGrid from hand-made straight-line tracks, so
that scenario tests know the exact geometry of every pair."""

from __future__ import annotations

import numpy as np
import pandas as pd

from proximity_pipeline import loading
from proximity_pipeline import goaround_adapter as ga

DAY0 = pd.Timestamp("2025-06-01", tz="UTC")


def straight_track(icao24: str, t_start: float, x0: float, y0: float,
                   h0: float, gs_kt: float, track_deg: float,
                   duration_s: float, vrate_fpm: float = 0.0,
                   step_s: float = 1.0, callsign: str | None = None,
                   onground: bool = False) -> pd.DataFrame:
    """Raw OpenSky-style rows for a constant-velocity flight starting at
    (x0, y0) NM from the airport, h0 ft above field."""
    n = int(duration_s / step_s) + 1
    t = t_start + np.arange(n) * step_s
    b = np.radians(track_deg)
    x = x0 + gs_kt / 3600.0 * (t - t_start) * np.sin(b)
    y = y0 + gs_kt / 3600.0 * (t - t_start) * np.cos(b)
    h = h0 + vrate_fpm / 60.0 * (t - t_start)
    lat, lon = loading.unproject(x, y)
    elev = ga.config.FIELD_ELEVATION_FT
    return pd.DataFrame({
        "timestamp": DAY0 + pd.to_timedelta(t, unit="s"),
        "icao24": icao24,
        "callsign": callsign or icao24.upper(),
        "latitude": lat, "longitude": lon,
        "altitude": h + elev + 150.0,       # baro with a shared bias
        "geoaltitude": h + elev,
        "vertical_rate": vrate_fpm,
        "groundspeed": gs_kt,
        "track": track_deg % 360.0,
        "onground": onground,
    })


def day_grid_from(tracks: list[pd.DataFrame], tmp_path) -> loading.DayGrid:
    df = pd.concat(tracks, ignore_index=True)
    path = tmp_path / "scenario_20250601.parquet"
    df.to_parquet(path, index=False)
    return loading.build_day_grid(path)
