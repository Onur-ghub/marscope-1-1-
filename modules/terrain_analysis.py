"""Terrain analysis derived from the CTX DTM mosaic.

All outputs here are DERIVED products: computed by MARSCOPE from the
observed elevation (and THEMIS) data, with the formulas documented inline
and in the README.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import config
from modules.data_manager import MarsDataset, RegionGrid, region_grid

# Terrain classes (DTM-derived physiography, NOT a geologic map)
TERRAIN_CLASSES = {
    0: ("Smooth low plains / crater floor", "#8d6e63"),
    1: ("Elevated benches & mesas (e.g. fan deposits, buttes)", "#ffb74d"),
    2: ("Moderate slopes & hummocky terrain", "#a1887f"),
    3: ("Steep scarps & crater walls", "#e53935"),
    4: ("High-standing rim & plains", "#bcaaa4"),
}


@dataclass
class TerrainModel:
    fine_grid: RegionGrid
    graph_grid: RegionGrid
    elevation: np.ndarray        # fine, metres
    slope_deg: np.ndarray        # fine
    roughness_m: np.ndarray      # fine, local relief (std of detrended elevation)
    difficulty: np.ndarray       # fine, 0..1 terrain difficulty index
    terrain_class: np.ndarray    # fine, int codes (TERRAIN_CLASSES)
    hillshade: np.ndarray        # fine, 0..1
    # Graph-resolution aggregates used by the route optimiser
    g_elevation: np.ndarray
    g_slope_p90: np.ndarray
    g_difficulty: np.ndarray
    g_passable: np.ndarray


def slope_degrees(elev: np.ndarray, grid: RegionGrid) -> np.ndarray:
    """Slope from central differences with metric spacing on the Mars sphere."""
    dy_m = np.deg2rad(grid.res) * config.MARS_RADIUS_M
    lats = grid.lat_max - (np.arange(grid.height) + 0.5) * grid.res
    dx_m = dy_m * np.cos(np.deg2rad(lats))[:, None]
    gy, gx = np.gradient(elev)
    return np.degrees(np.arctan(np.hypot(gx / dx_m, gy / dy_m))).astype(np.float32)


def hillshade(elev: np.ndarray, grid: RegionGrid, azimuth=315.0, altitude=40.0, z=2.0) -> np.ndarray:
    dy_m = np.deg2rad(grid.res) * config.MARS_RADIUS_M
    dx_m = dy_m * np.cos(np.deg2rad(0.5 * (grid.lat_min + grid.lat_max)))
    gy, gx = np.gradient(elev * z)
    gx, gy = gx / dx_m, -gy / dy_m
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.deg2rad(360 - azimuth + 90), np.deg2rad(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1).astype(np.float32)


def roughness(elev: np.ndarray, window: int = 5) -> np.ndarray:
    """Local relief: standard deviation of elevation minus a smoothed surface."""
    detrended = elev - ndimage.gaussian_filter(elev, sigma=window)
    mean = ndimage.uniform_filter(detrended, window)
    mean_sq = ndimage.uniform_filter(detrended ** 2, window)
    return np.sqrt(np.clip(mean_sq - mean ** 2, 0, None)).astype(np.float32)


def difficulty_index(slope: np.ndarray, rough: np.ndarray) -> np.ndarray:
    """Terrain difficulty in [0, 1].

    difficulty = 0.7 * min(slope / max_slope, 1)^1.5 + 0.3 * min(roughness / 15 m, 1)

    The slope term is convex so that gentle slopes cost little while slopes
    approaching the traversability limit dominate. 15 m of local relief at
    ~60 m sampling is treated as maximally rough.
    """
    max_slope = config.ASSUMPTIONS["max_traversable_slope_deg"]
    s = np.clip(slope / max_slope, 0, 1) ** 1.5
    r = np.clip(rough / 15.0, 0, 1)
    return (0.7 * s + 0.3 * r).astype(np.float32)


def classify_terrain(elev: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Simple, transparent physiographic classification from the DTM.

    * steep (slope > caution threshold)                      -> scarps/walls
    * high-standing (elevation above the regional 80th pct.) -> rim & plains
    * locally elevated (> +40 m above 3 km-scale mean)       -> benches/mesas
    * moderate slopes (5 deg to caution)                     -> hummocky
    * else                                                   -> smooth low plains
    """
    caution = config.ASSUMPTIONS["caution_slope_deg"]
    local_mean = ndimage.uniform_filter(elev, size=51)  # ~3 km at 59 m/px
    rel = elev - local_mean
    high = elev > np.nanpercentile(elev, 80)
    cls = np.zeros(elev.shape, np.uint8)
    cls[(slope > 5)] = 2
    cls[(rel > 40) & (slope <= caution)] = 1
    cls[high & (slope <= caution)] = 4
    cls[slope > caution] = 3
    return ndimage.median_filter(cls, size=3)


def _block_reduce(arr: np.ndarray, factor: int, func) -> np.ndarray:
    h, w = arr.shape
    h2, w2 = h // factor, w // factor
    blocks = arr[: h2 * factor, : w2 * factor].reshape(h2, factor, w2, factor)
    return func(blocks, axis=(1, 3))


def build_terrain_model(ds: MarsDataset) -> TerrainModel:
    fine = ds.fine_grid
    elev = ds.elevation.astype(np.float32)
    slope = slope_degrees(elev, fine)
    rough = roughness(elev)
    diff = difficulty_index(slope, rough)
    cls = classify_terrain(elev, slope)
    hs = hillshade(elev, fine)

    f = config.AGG_FACTOR
    g_elev = _block_reduce(elev, f, np.mean)
    g_slope = _block_reduce(slope, f, lambda b, axis: np.percentile(b, 90, axis=axis))
    g_diff = _block_reduce(diff, f, np.mean)
    # Planning grid = exact aggregation of the fine grid (same north-west corner).
    gh, gw = g_elev.shape
    res = fine.res * f
    graph = RegionGrid(fine.lon_min, fine.lat_max - gh * res, fine.lon_min + gw * res, fine.lat_max, res)
    passable = g_slope <= config.ASSUMPTIONS["max_traversable_slope_deg"]
    return TerrainModel(fine, graph, elev, slope, rough, diff, cls, hs,
                        g_elev.astype(np.float32), g_slope.astype(np.float32),
                        g_diff.astype(np.float32), passable)


def sample_along(arr: np.ndarray, grid: RegionGrid, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Bilinear sample of a fine-grid array at arbitrary lat/lon points."""
    rows = (grid.lat_max - np.asarray(lats)) / grid.res - 0.5
    cols = (np.asarray(lons) - grid.lon_min) / grid.res - 0.5
    return ndimage.map_coordinates(arr, [rows, cols], order=1, mode="nearest")


def point_summary(tm: TerrainModel, lat: float, lon: float) -> dict:
    r, c = tm.fine_grid.lonlat_to_rowcol(lat, lon)
    win = (slice(max(r - 8, 0), r + 9), slice(max(c - 8, 0), c + 9))  # ~1 km box
    return {
        "elevation_m": float(tm.elevation[r, c]),
        "slope_deg": float(tm.slope_deg[r, c]),
        "max_slope_1km_deg": float(tm.slope_deg[win].max()),
        "roughness_m": float(tm.roughness_m[r, c]),
        "difficulty": float(tm.difficulty[r, c]),
        "local_relief_1km_m": float(tm.elevation[win].max() - tm.elevation[win].min()),
        "terrain_class": TERRAIN_CLASSES[int(tm.terrain_class[r, c])][0],
        "passable": bool(tm.slope_deg[r, c] <= config.ASSUMPTIONS["max_traversable_slope_deg"]),
    }
