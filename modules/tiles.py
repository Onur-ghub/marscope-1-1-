"""Local map-tile pyramids for high-resolution imagery.

Leaflet's EPSG:4326 CRS uses 256-px tiles where zoom z spans 180 / 2^z degrees
per tile (2 x 1 tiles at z = 0), the same scheme as NASA Trek's equirectangular
WMTS. We render tiles for:

* ``ctx``    - MRO CTX orthoimage mosaic (USGS ARD, ~20 m/px), zooms 8-11, whole study area.
* ``hirise`` - MRO HiRISE orthoimages (USGS ARD, 0.25 m/px native, read at ~1-2 m),
               zooms 12-15 over the landing site / delta / Neretva corridor. Gaps
               between HiRISE strips are filled with CTX so the layer is seamless,
               and the fill is recorded in the provenance file.

Tiles are written to ``static/tiles/<layer>/<z>/<x>/<y>.jpg`` and served by
Streamlit's static file server (``/app/static/...``).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

import config
from modules.data_manager import RegionGrid, _read_resampled, haversine_m  # noqa: F401

TILE = 256
STATIC_URL_PREFIX = "/app/static/tiles"   # absolute URL served by Streamlit

# HiRISE stereo-pair orthoimages (USGS ARD, CC0) covering the demo corridor.
HIRISE_ORTHOS = [
    ("DTEEC_045994_1985_046060_1985_U01",
     "https://astrogeo-ard.s3-us-west-2.amazonaws.com/mars/mro/hirise/controlled/dtm/"
     "ESP_045994_1985_ESP_046060_1985/ESP_045994_1985_RED_A_01_ORTHO.tif"),
    ("DTEEC_002387_1985_003798_1985_A01", None),  # resolved from the STAC index at build time
]
HIRISE_BOX = (77.320, 18.420, 77.470, 18.530)   # lon_min, lat_min, lon_max, lat_max


def tile_deg(z: int) -> float:
    return 180.0 / (2 ** z)


def tile_range(bounds, z):
    lon_min, lat_min, lon_max, lat_max = bounds
    t = tile_deg(z)
    x0 = int(math.floor((lon_min + 180) / t)); x1 = int(math.floor((lon_max + 180) / t - 1e-9))
    y0 = int(math.floor((90 - lat_max) / t)); y1 = int(math.floor((90 - lat_min) / t - 1e-9))
    return x0, x1, y0, y1


def aligned_grid(bounds, z) -> tuple[RegionGrid, tuple]:
    x0, x1, y0, y1 = tile_range(bounds, z)
    t = tile_deg(z)
    g = RegionGrid(lon_min=-180 + x0 * t, lat_min=90 - (y1 + 1) * t,
                   lon_max=-180 + (x1 + 1) * t, lat_max=90 - y0 * t, res=t / TILE)
    return g, (x0, x1, y0, y1)


def layer_dir(layer: str) -> Path:
    return config.ROOT_DIR / "static" / "tiles" / layer


def layer_info(layer: str) -> dict | None:
    p = layer_dir(layer) / "tiles.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def url_template(layer: str) -> str:
    return f"{STATIC_URL_PREFIX}/{layer}/{{z}}/{{x}}/{{y}}.jpg"


def _stretch(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.clip((arr - lo) / max(hi - lo, 1e-6) * 254 + 1, 1, 255)
    return np.where(np.isfinite(arr), out, 0).astype(np.uint8)


def _write_tiles(img: np.ndarray, rng: tuple, z: int, out: Path, tint=(1.0, 0.93, 0.85)) -> int:
    x0, x1, y0, y1 = rng
    n = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            r0, c0 = (ty - y0) * TILE, (tx - x0) * TILE
            tile = img[r0:r0 + TILE, c0:c0 + TILE]
            if tile.shape != (TILE, TILE) or (tile > 0).mean() < 0.01:
                continue
            rgb = np.dstack([tile * tint[0], tile * tint[1], tile * tint[2]]).astype(np.uint8)
            d = out / str(z) / str(tx)
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(d / f"{ty}.jpg", "JPEG", quality=80, optimize=True)
            n += 1
    return n


def build_ctx_tiles(ctx_urls: list[str], zooms=range(8, 12),
                    progress: Callable[[str], None] = print) -> dict:
    out = layer_dir("ctx")
    bounds = config.REGION_BOUNDS
    stretch = None
    total = 0
    for z in zooms:
        grid, rng = aligned_grid(bounds, z)
        mosaic = np.full(grid.shape, np.nan, np.float32)
        for url in ctx_urls:
            arr = _read_resampled(url, grid, 20.0)
            fill = np.isfinite(arr) & ~np.isfinite(mosaic)
            mosaic[fill] = arr[fill]
        if stretch is None:
            stretch = tuple(np.nanpercentile(mosaic, [1, 99.5]))
        n = _write_tiles(_stretch(mosaic, *stretch), rng, z, out)
        total += n
        progress(f"CTX z{z}: {n} tiles")
    info = {"layer": "ctx", "min_zoom": min(zooms), "max_zoom": max(zooms), "bounds": list(bounds),
            "tiles": total, "kind": "observation",
            "title": "MRO CTX orthoimage mosaic (USGS ARD, ~20 m/px)",
            "sources": ctx_urls, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "tiles.json").write_text(json.dumps(info, indent=1))
    return info


def build_hirise_tiles(hirise_urls: list[str], ctx_urls: list[str], zooms=range(12, 16),
                       box=HIRISE_BOX, progress: Callable[[str], None] = print) -> dict:
    out = layer_dir("hirise")
    total, hi_stretch, ctx_stretch = 0, None, None
    hirise_fraction = []
    for z in zooms:
        grid, rng = aligned_grid(box, z)
        gsd = grid.cell_size_m()[1]
        hi = np.full(grid.shape, np.nan, np.float32)
        for url in hirise_urls:
            arr = _read_resampled(url, grid, 0.25)
            arr[arr <= 0] = np.nan
            fill = np.isfinite(arr) & ~np.isfinite(hi)
            hi[fill] = arr[fill]
        base = np.full(grid.shape, np.nan, np.float32)
        for url in ctx_urls:
            arr = _read_resampled(url, grid, 20.0, resampling_name="bilinear")
            fill = np.isfinite(arr) & ~np.isfinite(base)
            base[fill] = arr[fill]
        if hi_stretch is None:
            hi_stretch = tuple(np.nanpercentile(hi, [1, 99.5]))
            ctx_stretch = tuple(np.nanpercentile(base, [1, 99.5]))
        img = _stretch(base, *ctx_stretch)
        hv = np.isfinite(hi)
        img[hv] = _stretch(hi, *hi_stretch)[hv]
        hirise_fraction.append(float(hv.mean()))
        n = _write_tiles(img, rng, z, out)
        total += n
        progress(f"HiRISE z{z} (~{gsd:.1f} m/px): {n} tiles, {hv.mean():.0%} HiRISE pixels")
    info = {"layer": "hirise", "min_zoom": min(zooms), "max_zoom": max(zooms), "bounds": list(box),
            "tiles": total, "kind": "observation",
            "title": "MRO HiRISE orthoimages (USGS ARD), gaps filled with CTX",
            "hirise_pixel_fraction": round(float(np.mean(hirise_fraction)), 3),
            "sources": hirise_urls, "fill_sources": ctx_urls,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "tiles.json").write_text(json.dumps(info, indent=1))
    return info


def build_landing_image(box=(77.20, 18.30, 77.68, 18.62), z: int = 11, width: int = 1600) -> Path:
    """Stitch bundled CTX tiles over the delta into static/landing_terrain.jpg (landing-screen ground)."""
    x0, x1, y0, y1 = tile_range(box, z)
    img = Image.new("RGB", ((x1 - x0 + 1) * TILE, (y1 - y0 + 1) * TILE), (40, 30, 25))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            p = layer_dir("ctx") / str(z) / str(x) / f"{y}.jpg"
            if p.exists():
                img.paste(Image.open(p), ((x - x0) * TILE, (y - y0) * TILE))
    img = img.resize((width, int(width * img.height / img.width)), Image.LANCZOS)
    out = config.ROOT_DIR / "static" / "landing_terrain.jpg"
    img.save(out, "JPEG", quality=78, optimize=True)
    return out
