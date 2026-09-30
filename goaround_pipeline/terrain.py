"""
Optional terrain-model AGL support (profile setting `terrain: dem`).

Ground elevation comes from the AWS Terrain Tiles public dataset
("terrarium" encoding, ~30-40 m resolution at the zoom used here). On the
first run for an airport, the tiles covering the detection area (airport
+/- DEM_RADIUS_NM) are downloaded once and cached as a NumPy grid under
airports/dem_cache/<ICAO>.npz; afterwards everything works offline.

With `terrain: flat` (the default) this module is never imported and AGL is
GNSS altitude minus field elevation - appropriate for flat sites like KDAB.
"""

from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / "airports" / "dem_cache"
TILE_URL = ("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/"
            "{z}/{x}/{y}.png")
ZOOM = 12          # ~38 m/pixel at the equator
DEM_RADIUS_NM = 8.0
FT_PER_M = 3.28084

_samplers: dict[str, "callable"] = {}


def _tile_xy(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Continuous slippy-map tile coordinates."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r))
         / math.pi) / 2.0 * n
    return x, y


def _ssl_context():
    """macOS system Pythons often lack CA certs for urllib; use certifi's
    bundle when available (it ships with the traffic/requests stack)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch_tile(z: int, x: int, y: int) -> np.ndarray:
    """One 256x256 terrarium tile as elevation in metres."""
    from PIL import Image

    with urllib.request.urlopen(TILE_URL.format(z=z, x=x, y=y),
                                timeout=60, context=_ssl_context()) as resp:
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    rgb = np.asarray(img, dtype=np.float64)
    return rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 \
        - 32768.0


def ensure_dem(icao: str, lat: float, lon: float) -> Path:
    """Download and cache the DEM grid for an airport if not present."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{icao.upper()}.npz"
    if path.exists():
        return path

    dlat = DEM_RADIUS_NM / 60.0
    dlon = dlat / math.cos(math.radians(lat))
    x0f, y1f = _tile_xy(lat + dlat, lon - dlon, ZOOM)  # NW corner
    x1f, y0f = _tile_xy(lat - dlat, lon + dlon, ZOOM)  # SE corner
    x0, x1 = int(x0f), int(x1f)
    y0, y1 = int(y1f), int(y0f)

    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    print(f"terrain: downloading {nx * ny} DEM tiles for {icao} "
          f"(one-time, ~{nx * ny * 40} kB) ...", flush=True)
    grid = np.zeros((ny * 256, nx * 256))
    for iy in range(y0, y1 + 1):
        for ix in range(x0, x1 + 1):
            tile = _fetch_tile(ZOOM, ix, iy)
            grid[(iy - y0) * 256:(iy - y0 + 1) * 256,
                 (ix - x0) * 256:(ix - x0 + 1) * 256] = tile

    np.savez_compressed(path, grid=grid, x0=x0, y0=y0, zoom=ZOOM)
    print(f"terrain: cached {path.name} "
          f"({grid.shape[1]}x{grid.shape[0]} px)", flush=True)
    return path


def ground_elevation_ft(icao: str, lat: np.ndarray,
                        lon: np.ndarray) -> np.ndarray:
    """Bilinear-sampled ground elevation (ft MSL) under each position.
    Positions outside the cached area get the nearest edge value."""
    icao = icao.upper()
    if icao not in _samplers:
        with np.load(CACHE_DIR / f"{icao}.npz") as d:
            grid, x0, y0, zoom = (d["grid"], int(d["x0"]), int(d["y0"]),
                                  int(d["zoom"]))
        _samplers[icao] = (grid, x0, y0, zoom)
    grid, x0, y0, zoom = _samplers[icao]

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = 2.0 ** zoom
    fx = ((lon + 180.0) / 360.0 * n - x0) * 256.0
    lat_r = np.radians(np.clip(lat, -85.0, 85.0))
    fy = ((1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / np.pi)
          / 2.0 * n - y0) * 256.0

    fx = np.clip(fx, 0.0, grid.shape[1] - 1.001)
    fy = np.clip(fy, 0.0, grid.shape[0] - 1.001)
    ix, iy = fx.astype(int), fy.astype(int)
    tx, ty = fx - ix, fy - iy
    elev_m = (grid[iy, ix] * (1 - tx) * (1 - ty)
              + grid[iy, ix + 1] * tx * (1 - ty)
              + grid[iy + 1, ix] * (1 - tx) * ty
              + grid[iy + 1, ix + 1] * tx * ty)
    return elev_m * FT_PER_M
