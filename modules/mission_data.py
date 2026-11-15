"""Live mission data from NASA / USGS web services, with a local cache.

* Perseverance traverse path and waypoints - NASA/JPL MMGIS map service
  (the same data behind NASA's "Where is Perseverance?" map).
* Perseverance raw images by sol - NASA Mars 2020 raw-image feed
  (https://mars.nasa.gov/mars2020/multimedia/raw-images/).
* Global geologic map of Mars (Tanaka et al. 2014, USGS SIM 3292) - USGS
  Astrogeology ArcGIS feature service, clipped to the Jezero study area.

Every fetch is cached in data/cache/. If the network is unavailable the last
cached copy is used; if there is none, the caller gets ``None`` and the UI
says the layer is unavailable (nothing is invented).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

import config
from modules.data_manager import haversine_m

log = logging.getLogger("marscope.mission")

MMGIS_BASE = "https://mars.nasa.gov/mmgis-maps/M20/"
TRAVERSE_URL = MMGIS_BASE + "Layers/json/M20_traverse.json"
WAYPOINTS_URL = MMGIS_BASE + "Layers/json/M20_waypoints.json"
RAW_IMAGES_URL = "https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json&sol={sol}"
RAW_IMAGES_PAGE = "https://mars.nasa.gov/mars2020/multimedia/raw-images/"
GEOLOGY_URL = ("https://services.arcgis.com/nzS0F0zdNLvs7nc8/ArcGIS/rest/services/"
               "Mars_Geology_feature_layer/FeatureServer/0/query")

# Preferred cameras for galleries (colour context first)
CAMERA_PRIORITY = ["MCZ_LEFT", "MCZ_RIGHT", "NAVCAM_LEFT", "NAVCAM_RIGHT", "SUPERCAM_RMI",
                   "SHERLOC_WATSON", "FRONT_HAZCAM_LEFT_A", "REAR_HAZCAM_LEFT"]

# SIM 3292 unit names (from the map's own unit descriptions)
GEOLOGY_UNIT_NAMES = {
    "HNt": "Hesperian and Noachian transition unit",
    "mNhm": "Middle Noachian highland massif unit",
    "eNh": "Early Noachian highland unit",
    "mNh": "Middle Noachian highland unit",
    "lNh": "Late Noachian highland unit",
    "AHi": "Amazonian and Hesperian impact unit",
}


def _cache_path(name: str) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / name


def fetch_json(url: str, cache_name: str, max_age_h: float = 24.0, params: dict | None = None) -> tuple:
    """Return (data, status) where status is 'live', 'cache', 'stale-cache' or 'unavailable'."""
    path = _cache_path(cache_name)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600
    if fresh:
        try:
            return json.loads(path.read_text()), "cache"
        except Exception:
            pass
    if not config.OFFLINE_MODE:
        try:
            import requests
            r = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT_S,
                             headers={"User-Agent": "MARSCOPE/1.0 (NASA Space Apps project)"})
            r.raise_for_status()
            data = r.json()
            path.write_text(json.dumps(data))
            return data, "live"
        except Exception as exc:
            log.warning("fetch failed %s: %s", url, exc)
    if path.exists():
        try:
            return json.loads(path.read_text()), "stale-cache"
        except Exception:
            pass
    return None, "unavailable"


# ---------------------------------------------------------------------------
# Perseverance traverse
# ---------------------------------------------------------------------------
def _in_region(lon, lat) -> bool:
    lo0, la0, lo1, la1 = config.REGION_BOUNDS
    return lo0 <= lon <= lo1 and la0 <= lat <= la1


def perseverance_traverse() -> tuple[list[dict], str]:
    """List of drive segments: {sol, coords [(lat, lon)], length_m}."""
    data, status = fetch_json(TRAVERSE_URL, "m20_traverse.json", max_age_h=24)
    if not data:
        return [], status
    segs = []
    for f in data.get("features", []):
        geom = f.get("geometry") or {}
        props = f.get("properties") or {}
        lines = geom.get("coordinates") or []
        if geom.get("type") == "LineString":
            lines = [lines]
        elif geom.get("type") != "MultiLineString":
            continue
        for line in lines:
            pts = [(float(c[1]), float(c[0])) for c in line if len(c) >= 2 and _in_region(c[0], c[1])]
            if len(pts) >= 2:
                segs.append({"sol": int(props.get("sol", -1)), "coords": pts,
                             "length_m": float(props.get("length", 0) or 0)})
    return sorted(segs, key=lambda s: s["sol"]), status


def perseverance_waypoints() -> tuple[list[dict], str]:
    data, status = fetch_json(WAYPOINTS_URL, "m20_waypoints.json", max_age_h=24)
    if not data:
        return [], status
    wps = []
    for f in data.get("features", []):
        p = f.get("properties") or {}
        try:
            lat, lon = float(p.get("lat")), float(p.get("lon"))
        except (TypeError, ValueError):
            coords = (f.get("geometry") or {}).get("coordinates") or [None, None]
            lon, lat = coords[0], coords[1]
        if lat is None or not _in_region(lon, lat):
            continue
        pano = next((i for i in (p.get("images") or []) if i.get("url")), None)
        wps.append({"sol": int(p.get("sol", -1)), "site": p.get("site"), "drive": p.get("drive"),
                    "lat": lat, "lon": lon, "elev_m": p.get("elev_geoid"),
                    "dist_total_km": p.get("dist_km"), "rmc": p.get("RMC"),
                    "panorama_url": (MMGIS_BASE + pano["url"]) if pano else None})
    return sorted(wps, key=lambda w: w["sol"]), status


def visits_near(lat: float, lon: float, waypoints: list[dict], radius_m: float = 400) -> list[dict]:
    """Waypoints within radius of a location, closest first per sol."""
    if not waypoints:
        return []
    lats = np.array([w["lat"] for w in waypoints]); lons = np.array([w["lon"] for w in waypoints])
    d = haversine_m(lat, lon, lats, lons)
    out = {}
    for w, dist in zip(waypoints, d):
        if dist <= radius_m and (w["sol"] not in out or dist < out[w["sol"]]["distance_m"]):
            out[w["sol"]] = {**w, "distance_m": float(dist)}
    return sorted(out.values(), key=lambda w: w["sol"])


# ---------------------------------------------------------------------------
# Raw images
# ---------------------------------------------------------------------------
def raw_images_for_sol(sol: int, limit: int = 12) -> tuple[list[dict], str]:
    data, status = fetch_json(RAW_IMAGES_URL.format(sol=int(sol)), f"m20_raw_sol{int(sol)}.json",
                              max_age_h=24 * 30)
    if not data:
        return [], status
    imgs = []
    for im in data.get("images", []):
        files = im.get("image_files") or {}
        cam = (im.get("camera") or {}).get("instrument", "UNK")
        if not files.get("small"):
            continue
        imgs.append({"id": im.get("imageid"), "camera": cam, "title": im.get("title", ""),
                     "thumb": files.get("medium") or files.get("small"), "large": files.get("large"),
                     "full": files.get("full_res"), "link": im.get("link"),
                     "date": im.get("date_taken_utc", ""), "credit": im.get("credit", "NASA/JPL-Caltech"),
                     "sample_type": im.get("sample_type", "")})
    def rank(i):
        c = i["camera"]
        pri = next((k for k, name in enumerate(CAMERA_PRIORITY) if c.startswith(name)), len(CAMERA_PRIORITY))
        return (i["sample_type"] != "Full", pri)
    imgs.sort(key=rank)
    picked, seen = [], {}
    for i in imgs:  # at most 4 per camera so the gallery shows variety
        if seen.get(i["camera"], 0) < 4:
            picked.append(i); seen[i["camera"]] = seen.get(i["camera"], 0) + 1
        if len(picked) >= limit:
            break
    return picked, status


# ---------------------------------------------------------------------------
# USGS global geologic map (SIM 3292)
# ---------------------------------------------------------------------------
def geologic_units() -> tuple[dict | None, str]:
    lo0, la0, lo1, la1 = config.REGION_BOUNDS
    params = {"where": "1=1", "geometry": f"{lo0},{la0},{lo1},{la1}",
              "geometryType": "esriGeometryEnvelope", "spatialRel": "esriSpatialRelIntersects",
              "outFields": "Unit,UnitDesc", "returnGeometry": "true", "outSR": "4326",
              "f": "geojson"}
    data, status = fetch_json(GEOLOGY_URL, "usgs_sim3292_jezero.geojson", max_age_h=24 * 30, params=params)
    if not data or not data.get("features"):
        return None, status if data is None else "unavailable"
    try:
        import geopandas as gpd
        from shapely.geometry import box
        gdf = gpd.GeoDataFrame.from_features(data["features"])
        gdf = gdf.set_geometry(gdf.geometry.intersection(box(lo0, la0, lo1, la1)))
        gdf = gdf[~gdf.geometry.is_empty]
        gdf["UnitName"] = gdf["Unit"].map(GEOLOGY_UNIT_NAMES).fillna(gdf["Unit"])
        if "UnitDesc" not in gdf:
            gdf["UnitDesc"] = ""
        return json.loads(gdf.to_json()), status
    except Exception as exc:
        log.warning("geology parse failed: %s", exc)
        return None, "unavailable"
