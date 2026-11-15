"""Data acquisition layer for MARSCOPE.

Responsibilities
----------------
* Discover and read *real* public Mars data products for Jezero Crater:
  - USGS Astrogeology "Analysis Ready Data" (ARD) on AWS S3 (CC0):
      MRO CTX stereo DTMs (20 m/px, absolutely controlled to MOLA, aeroid heights)
      MRO CTX orthoimages (co-registered with the DTMs)
      Mars Odyssey THEMIS controlled IR mosaics, MC-13 Syrtis Major (100 m/px)
  These are Cloud Optimized GeoTIFFs; Rasterio reads only the bytes we need
  (HTTP range requests on overview levels), so a full refresh is ~30-60 MB.
* Resample everything onto one common lat/lon grid over Jezero.
* Cache results locally, validate them, and record provenance per layer.
* Fall back gracefully:  local cache -> bundled snapshot -> live download ->
  clearly-labelled synthetic DEMONSTRATION terrain (only if all else fails).

Every layer carries a ``kind`` of ``observation`` (instrument data product),
``derived`` (computed by MARSCOPE from observations) or ``demonstration``
(synthetic, illustrative only).
"""
from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import config

log = logging.getLogger("marscope.data")

DATASET_NAME = "jezero_dataset"
MARS_GEOGRAPHIC_PROJ4 = f"+proj=longlat +R={config.MARS_RADIUS_M:.0f} +no_defs"


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionGrid:
    """A regular lat/lon grid (north-up) over the study region."""

    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    res: float

    @property
    def width(self) -> int:
        return int(round((self.lon_max - self.lon_min) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.lat_max - self.lat_min) / self.res))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def lonlat_to_rowcol(self, lat: float, lon: float) -> tuple[int, int]:
        row = int((self.lat_max - lat) / self.res)
        col = int((lon - self.lon_min) / self.res)
        return (min(max(row, 0), self.height - 1), min(max(col, 0), self.width - 1))

    def rowcol_to_lonlat(self, row: float, col: float) -> tuple[float, float]:
        """Return (lat, lon) of the cell centre."""
        lat = self.lat_max - (row + 0.5) * self.res
        lon = self.lon_min + (col + 0.5) * self.res
        return lat, lon

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max

    def cell_size_m(self) -> tuple[float, float]:
        """(dy, dx) metres at the grid's mid-latitude (spherical Mars)."""
        dy = np.deg2rad(self.res) * config.MARS_RADIUS_M
        mid_lat = np.deg2rad(0.5 * (self.lat_min + self.lat_max))
        return dy, dy * np.cos(mid_lat)

    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.lon_min, self.lat_max, self.res, self.res)


def region_grid(res: float) -> RegionGrid:
    lon_min, lat_min, lon_max, lat_max = config.REGION_BOUNDS
    return RegionGrid(lon_min, lat_min, lon_max, lat_max, res)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance on the Mars sphere (metres). Vectorised."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1
    dlmb = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * config.MARS_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------
@dataclass
class MarsDataset:
    """All gridded layers for the study area plus provenance metadata."""

    fine_grid: RegionGrid
    image_grid: RegionGrid
    elevation: np.ndarray            # float32 metres (aeroid), FINE grid; NaN = no data
    dtm_coverage: np.ndarray         # bool, FINE grid: True where measured by a CTX DTM
    ctx_image: Optional[np.ndarray]  # uint8, IMAGE grid (0 = no data)
    themis_day: Optional[np.ndarray]   # uint8 DN, FINE grid
    themis_night: Optional[np.ndarray]  # uint8 DN, FINE grid
    provenance: dict = field(default_factory=dict)
    status: str = "unknown"          # "cache" | "bundled" | "live" | "demonstration"
    status_detail: str = ""
    validation: list = field(default_factory=list)

    @property
    def is_demonstration(self) -> bool:
        return self.status == "demonstration"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _paths(directory: Path) -> tuple[Path, Path]:
    return directory / f"{DATASET_NAME}.npz", directory / f"{DATASET_NAME}.json"


def save_dataset(ds: MarsDataset, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _paths(directory)
    arrays = {"elevation": ds.elevation.astype(np.float32),
              "dtm_coverage": ds.dtm_coverage.astype(np.uint8)}
    for name in ("ctx_image", "themis_day", "themis_night"):
        arr = getattr(ds, name)
        if arr is not None:
            arrays[name] = arr.astype(np.uint8)
    np.savez_compressed(npz_path, **arrays)
    meta = {"provenance": ds.provenance,
            "fine_grid": ds.fine_grid.__dict__,
            "image_grid": ds.image_grid.__dict__,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta_path.write_text(json.dumps(meta, indent=2))


def load_dataset(directory: Path, status: str) -> Optional[MarsDataset]:
    npz_path, meta_path = _paths(directory)
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        with np.load(npz_path) as z:
            ds = MarsDataset(
                fine_grid=RegionGrid(**meta["fine_grid"]),
                image_grid=RegionGrid(**meta["image_grid"]),
                elevation=z["elevation"].astype(np.float32),
                dtm_coverage=z["dtm_coverage"].astype(bool),
                ctx_image=z["ctx_image"] if "ctx_image" in z else None,
                themis_day=z["themis_day"] if "themis_day" in z else None,
                themis_night=z["themis_night"] if "themis_night" in z else None,
                provenance=meta.get("provenance", {}),
                status=status,
                status_detail=f"Loaded from {directory.name}/ (saved {meta.get('saved_at', '?')})",
            )
        return ds
    except Exception as exc:  # corrupted cache must never crash the app
        log.warning("Could not load dataset from %s: %s", directory, exc)
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(ds: MarsDataset) -> list[tuple[str, bool, str]]:
    """Return a list of (check, passed, detail)."""
    checks = []
    shape_ok = ds.elevation.shape == ds.fine_grid.shape
    checks.append(("Elevation grid shape", shape_ok, f"{ds.elevation.shape} vs {ds.fine_grid.shape}"))
    finite = np.isfinite(ds.elevation)
    checks.append(("Elevation has values", bool(finite.mean() > 0.95), f"{finite.mean():.1%} finite"))
    if finite.any():
        lo, hi = float(np.nanmin(ds.elevation)), float(np.nanmax(ds.elevation))
        # Jezero lies roughly between -3 km and -1 km relative to the Mars aeroid.
        plausible = -4500 < lo < -1500 and -3000 < hi < 500
        checks.append(("Elevation range plausible for Jezero", plausible, f"{lo:.0f} m to {hi:.0f} m"))
    cov = float(ds.dtm_coverage.mean())
    checks.append(("CTX DTM coverage of region", cov > 0.6, f"{cov:.1%} measured, rest gap-filled"))
    for name in ("ctx_image", "themis_day", "themis_night"):
        arr = getattr(ds, name)
        ok = arr is not None and (arr > 0).mean() > 0.5
        detail = "missing" if arr is None else f"{(arr > 0).mean():.1%} valid pixels"
        checks.append((f"{name} layer", ok, detail))
    return checks


# ---------------------------------------------------------------------------
# Live acquisition from USGS ARD (Cloud Optimized GeoTIFFs)
# ---------------------------------------------------------------------------
def _http_get(url: str) -> bytes:
    import requests
    resp = requests.get(url, timeout=config.HTTP_TIMEOUT_S * 6)
    resp.raise_for_status()
    return resp.content


def _cached_index(url: str) -> Path:
    """Download a STAC parquet index once and keep it in the cache."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CACHE_DIR / Path(url).name
    if not path.exists():
        path.write_bytes(_http_get(url))
    return path


def select_ctx_dtms(region: tuple[float, float, float, float]) -> list[dict]:
    """Choose CTX DTM stereo pairs from the USGS STAC index that best cover the region.

    Greedy set cover on footprints: repeatedly take the DTM that adds the most
    uncovered area until TARGET_DTM_COVERAGE or MAX_CTX_DTMS is reached.
    """
    import geopandas as gpd
    from shapely.geometry import box

    index = gpd.read_parquet(_cached_index(config.CTX_DTM_INDEX_URL))
    aoi = box(*region)
    cand = index[index.intersects(aoi)].copy()
    if cand.empty:
        return []
    chosen, covered = [], None
    for _ in range(config.MAX_CTX_DTMS):
        best, best_gain = None, 0.0
        for idx, row in cand.iterrows():
            piece = row.geometry.intersection(aoi)
            gain = piece.area if covered is None else piece.difference(covered).area
            if gain > best_gain:
                best, best_gain = idx, gain
        if best is None or best_gain < 1e-4 * aoi.area:
            break
        row = cand.loc[best]
        covered = row.geometry.intersection(aoi) if covered is None else covered.union(row.geometry.intersection(aoi))
        assets = row["assets"]
        chosen.append({
            "id": row["id"],
            "dtm": assets["dtm"]["href"],
            "orthoimage": (assets.get("orthoimage") or {}).get("href"),
            "gsd_m": float(row.get("gsd") or 20.0),
            "bounds": [round(v, 4) for v in row.geometry.bounds],
        })
        cand = cand.drop(index=best)
        if covered.area / aoi.area >= config.TARGET_DTM_COVERAGE:
            break
    return chosen


def _read_resampled(url: str, grid: RegionGrid, src_gsd_m: float, dtype=np.float32,
                    resampling_name: str = "average") -> np.ndarray:
    """Read a remote COG at an appropriate overview and warp it onto ``grid``.

    Returns float32 with NaN outside the product footprint / nodata.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds

    dst_crs = CRS.from_proj4(MARS_GEOGRAPHIC_PROJ4)
    target_m = grid.cell_size_m()[1]
    factor = max(1, int(target_m // src_gsd_m))
    dst = np.full(grid.shape, np.nan, dtype=np.float32)
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "3",
           "GDAL_HTTP_RETRY_DELAY": "1", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    with rasterio.Env(**env), rasterio.open(f"/vsicurl/{url}") as src:
        # Only read the window that overlaps our region (plus a margin).
        left, bottom, right, top = transform_bounds(dst_crs, src.crs, *grid.bounds, densify_pts=21)
        pad = 5 * src_gsd_m * factor
        win = from_bounds(left - pad, bottom - pad, right + pad, top + pad, src.transform)
        try:
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        except rasterio.errors.WindowError:
            return dst  # product does not overlap this grid
        win = win.round_offsets().round_lengths()
        out_h = max(1, int(win.height // factor))
        out_w = max(1, int(win.width // factor))
        data = src.read(1, window=win, out_shape=(out_h, out_w), masked=True,
                        resampling=Resampling.average).astype(np.float32)
        arr = data.filled(np.nan)
        from affine import Affine
        base = src.window_transform(win)
        win_tf = base * Affine.scale(win.width / out_w, win.height / out_h)
        reproject(arr, dst, src_transform=win_tf, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=grid.transform(), dst_crs=dst_crs, dst_nodata=np.nan,
                  resampling=getattr(Resampling, resampling_name))
    return dst


def _mosaic(layers: list[np.ndarray], harmonise: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mosaic in priority order. Optionally remove vertical offsets between DTMs
    (median difference over the overlap) to suppress seams."""
    mosaic = np.full(layers[0].shape, np.nan, dtype=np.float32)
    for arr in layers:
        arr = arr.copy()
        valid_new = np.isfinite(arr)
        overlap = valid_new & np.isfinite(mosaic)
        if harmonise and overlap.sum() > 500:
            arr -= float(np.nanmedian(arr[overlap] - mosaic[overlap]))
        fill = valid_new & ~np.isfinite(mosaic)
        mosaic[fill] = arr[fill]
    return mosaic, np.isfinite(mosaic)


def fill_gaps(arr: np.ndarray) -> np.ndarray:
    """Fill NaN gaps by nearest valid neighbour followed by light smoothing of the
    filled pixels only. Filled pixels are flagged separately (dtm_coverage)."""
    from scipy import ndimage
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr
    idx = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
    filled = arr[tuple(idx)]
    smooth = ndimage.gaussian_filter(filled, sigma=3)
    out = arr.copy()
    out[missing] = smooth[missing]
    return out


def fetch_live_dataset(progress: Callable[[str, float], None] = lambda m, f: None) -> MarsDataset:
    """Download and assemble the dataset from USGS ARD. Raises on failure."""
    fine = region_grid(config.FINE_RES_DEG)
    image = region_grid(config.IMAGE_RES_DEG)
    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    progress("Querying USGS STAC index for CTX stereo DTMs...", 0.05)
    dtms = select_ctx_dtms(config.REGION_BOUNDS)
    if not dtms:
        raise RuntimeError("No CTX DTMs intersect the region in the USGS index")

    elev_layers, img_layers = [], []
    for i, item in enumerate(dtms):
        progress(f"Reading CTX DTM {i + 1}/{len(dtms)}: {item['id'][:32]}...", 0.1 + 0.6 * i / len(dtms))
        elev_layers.append(_read_resampled(item["dtm"], fine, item["gsd_m"]))
        if item["orthoimage"] and i < 4:
            try:
                img_layers.append(_read_resampled(item["orthoimage"], image, 6.0))
            except Exception as exc:
                log.warning("Orthoimage failed for %s: %s", item["id"], exc)

    elev, coverage = _mosaic(elev_layers, harmonise=True)
    elevation = fill_gaps(elev).astype(np.float32)

    ctx_u8 = None
    if img_layers:
        img, img_valid = _mosaic(img_layers, harmonise=False)
        lo, hi = np.nanpercentile(img, [1, 99.5])
        ctx_u8 = np.zeros(img.shape, np.uint8)
        ctx_u8[img_valid] = np.clip(1 + 254 * (img[img_valid] - lo) / (hi - lo), 1, 255).astype(np.uint8)

    themis = {}
    for key, url in (("themis_day", config.THEMIS_DAY_URL), ("themis_night", config.THEMIS_NIGHT_URL)):
        progress(f"Reading THEMIS {key.split('_')[1]}time IR mosaic (MC-13)...", 0.75 if key.endswith("day") else 0.85)
        try:
            arr = _read_resampled(url, fine, 100.0, resampling_name="bilinear")
            arr[arr <= 0] = np.nan
            themis[key] = np.nan_to_num(arr, nan=0).clip(0, 255).astype(np.uint8)
        except Exception as exc:
            log.warning("THEMIS %s failed: %s", key, exc)
            themis[key] = None

    provenance = {
        "elevation": {
            "kind": "observation",
            "title": "MRO CTX stereo DTM mosaic (USGS ARD, absolutely controlled to MGS MOLA)",
            "source": "USGS Astrogeology Science Center, Analysis Ready Data (CC0)",
            "items": [d["id"] for d in dtms],
            "urls": [d["dtm"] for d in dtms],
            "native_resolution": "20 m/px (resampled to ~59 m/px for analysis)",
            "notes": "Heights in metres relative to the Mars aeroid. Vertical offsets between "
                     "overlapping DTMs removed by median differencing. Pixels outside all DTM "
                     "footprints are gap-filled by nearest-neighbour + smoothing and flagged.",
            "fetched_at": fetched_at,
        },
        "ctx_image": {
            "kind": "observation",
            "title": "MRO CTX orthoimages co-registered with the DTMs",
            "source": "USGS Astrogeology ARD (CC0); MRO CTX: NASA/JPL-Caltech/MSSS",
            "urls": [d["orthoimage"] for d in dtms[:4] if d["orthoimage"]],
            "native_resolution": "~6 m/px (resampled to ~30 m/px)",
            "fetched_at": fetched_at,
        },
        "themis_day": {
            "kind": "observation",
            "title": "Mars Odyssey THEMIS daytime IR controlled mosaic, MC-13 Syrtis Major",
            "source": "USGS Astrogeology ARD; THEMIS: NASA/JPL-Caltech/ASU",
            "urls": [config.THEMIS_DAY_URL],
            "native_resolution": "100 m/px",
            "notes": "8-bit stretched brightness; tracks relative daytime surface temperature "
                     "(dominated by slope/albedo). NOT calibrated temperature.",
            "fetched_at": fetched_at,
        },
        "themis_night": {
            "kind": "observation",
            "title": "Mars Odyssey THEMIS nighttime IR controlled mosaic, MC-13 Syrtis Major",
            "source": "USGS Astrogeology ARD; THEMIS: NASA/JPL-Caltech/ASU",
            "urls": [config.THEMIS_NIGHT_URL],
            "native_resolution": "100 m/px",
            "notes": "8-bit stretched brightness; brighter at night = warmer = qualitatively "
                     "higher thermal inertia (rockier / more indurated surfaces). NOT a "
                     "calibrated thermal-inertia product.",
            "fetched_at": fetched_at,
        },
    }
    ds = MarsDataset(fine, image, elevation, coverage, ctx_u8,
                     themis.get("themis_day"), themis.get("themis_night"),
                     provenance=provenance, status="live",
                     status_detail=f"Downloaded from USGS ARD at {fetched_at}")
    progress("Validating...", 0.95)
    ds.validation = validate_dataset(ds)
    return ds


# ---------------------------------------------------------------------------
# Synthetic DEMONSTRATION fallback (used only if no real data is available)
# ---------------------------------------------------------------------------
def synthetic_dataset() -> MarsDataset:
    """Parametric crater terrain loosely shaped like Jezero (45 km crater with a
    western fan). It is ILLUSTRATIVE ONLY and is labelled as such everywhere."""
    fine = region_grid(config.FINE_RES_DEG)
    image = region_grid(config.IMAGE_RES_DEG)
    rows, cols = np.indices(fine.shape)
    lat = fine.lat_max - (rows + 0.5) * fine.res
    lon = fine.lon_min + (cols + 0.5) * fine.res
    clat, clon = config.JEZERO_CENTER
    r_km = haversine_m(lat, lon, clat, clon) / 1000.0
    R = config.JEZERO_RIM_RADIUS_KM
    floor, rim_h, plain = -2650.0, -1450.0, -2100.0
    x = r_km / R
    elev = np.where(x < 0.85, floor + 120 * x ** 4,
                    np.where(x < 1.0, floor + (rim_h - floor) * ((x - 0.85) / 0.15) ** 2,
                             plain + (rim_h - plain) * np.exp(-(x - 1.0) / 0.25)))
    # western fan bench
    fan_lat, fan_lon = 18.49, 77.40
    d_fan = haversine_m(lat, lon, fan_lat, fan_lon) / 1000.0
    elev += 110 * np.exp(-(d_fan / 5.0) ** 4)
    rng = np.random.default_rng(42)
    from scipy import ndimage
    elev += ndimage.gaussian_filter(rng.normal(0, 25, fine.shape), 4)
    elev = elev.astype(np.float32)
    ds = MarsDataset(fine, image, elev, np.zeros(fine.shape, bool), None, None, None,
                     provenance={"elevation": {
                         "kind": "demonstration",
                         "title": "SYNTHETIC parametric terrain (demonstration only)",
                         "source": "Generated by MARSCOPE; NOT NASA/USGS data",
                         "notes": "Used only because neither cached, bundled nor live USGS data "
                                  "were available. Shapes approximate Jezero's published size; "
                                  "all derived numbers are illustrative."}},
                     status="demonstration",
                     status_detail="No real data available - showing synthetic demonstration terrain")
    ds.validation = validate_dataset(ds)
    return ds


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_or_fetch(force_live: bool = False,
                  progress: Callable[[str, float], None] = lambda m, f: None) -> MarsDataset:
    """Resolve the best available dataset.

    Order: live (if forced) -> local cache -> bundled snapshot -> live -> synthetic.
    """
    errors = []
    if force_live and not config.OFFLINE_MODE:
        try:
            ds = fetch_live_dataset(progress)
            save_dataset(ds, config.CACHE_DIR)
            return ds
        except Exception as exc:
            errors.append(f"live download failed: {exc}")

    for directory, status in ((config.CACHE_DIR, "cache"), (config.BUNDLED_DIR, "bundled")):
        ds = load_dataset(directory, status)
        if ds is not None:
            ds.validation = validate_dataset(ds)
            if errors:
                ds.status_detail += " | " + "; ".join(errors)
            return ds

    if not config.OFFLINE_MODE and not force_live:
        try:
            ds = fetch_live_dataset(progress)
            save_dataset(ds, config.CACHE_DIR)
            return ds
        except Exception as exc:
            errors.append(f"live download failed: {exc}")

    ds = synthetic_dataset()
    if errors:
        ds.status_detail += " | " + "; ".join(errors)
    return ds


def check_endpoints() -> list[dict]:
    """Probe the remote services used by MARSCOPE (server-side reachability)."""
    import requests
    probes = [
        ("USGS ARD STAC index (CTX DTMs)", config.CTX_DTM_INDEX_URL),
        ("USGS ARD THEMIS day IR COG (MC-13)", config.THEMIS_DAY_URL),
        ("NASA Trek WMTS (MOLA shaded relief tile)",
         list(config.TREK_LAYERS.values())[0][0].format(z=0, y=0, x=0)),
    ]
    out = []
    for name, url in probes:
        t0 = time.time()
        try:
            r = requests.head(url, timeout=config.HTTP_TIMEOUT_S, allow_redirects=True)
            ok = r.status_code < 400
            detail = f"HTTP {r.status_code}"
        except Exception as exc:
            ok, detail = False, type(exc).__name__
        out.append({"service": name, "reachable": ok, "detail": detail,
                    "latency_s": round(time.time() - t0, 2), "url": url})
    return out
