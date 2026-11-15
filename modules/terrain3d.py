"""Interactive 3D colour map of Jezero crater (three.js component with two-way selection).

Every colour mode is a real or documented MARSCOPE layer sampled on the same mesh:

* Elevation      CTX DTM (USGS ARD), metres relative to the Mars aeroid          -> observation
* Slope          central differences of the DTM (terrain_analysis.slope_degrees)  -> derived
* Terrain class  DTM physiography (terrain_analysis.classify_terrain)             -> derived
* Geology        zones around catalogued, published features (jezero_pois.json);
                 an approximate literature overlay, NOT a geologic map            -> literature
* Science        MARSCOPE science-value surface (science_discovery)               -> derived
* Imagery        MRO CTX orthoimage mosaic (USGS ARD)                             -> observation

The rover playback in the component is SIMULATED along a planned route at the assumed pace; it is
never presented as mission telemetry. The real Perseverance traverse (NASA/JPL MMGIS) is drawn only
when it has been downloaded.

The component is a plain HTML file (components/terrain_explorer/index.html) that speaks the
Streamlit component protocol directly, so no frontend build step is needed.
"""
from __future__ import annotations

import base64
import math

import numpy as np
import streamlit.components.v1 as components

import config
from modules import science_discovery as sd
from modules import terrain_analysis as ta
from modules import view3d
from modules.data_manager import haversine_m

COMPONENT_DIR = config.ROOT_DIR / "components" / "terrain_explorer"
_component = components.declare_component("terrain_explorer", path=str(COMPONENT_DIR))

MESH_STEP = view3d.MESH_STEP          # same ~300 m mesh as the navigation 3D view
KM_PER_DEG = view3d.KM_PER_DEG

# Muted palette (charcoal/slate UI): colours carry meaning, not decoration.
CLASS_COLORS = {0: "#6f7f86", 1: "#d9c39c", 2: "#9fb3c8", 3: "#c98b6b", 4: "#b8a6e3"}

# Literature overlay: catalogued features grouped by geological setting (radius in km).
GEO_GROUPS = [
    ("Delta / fan sediments", "#d4b483", 1.5, ["delta_front", "delta_top_belva", "kodiak"]),
    ("Fluvial channel / inlet valley", "#7fa7c9", 1.5, ["bright_angel", "neretva_inlet"]),
    ("Crater-floor igneous units (Séítah, Máaz)", "#8fb49a", 1.5, ["seitah", "octavia_butler_landing"]),
    ("Marginal carbonates", "#b8a6e3", 1.5, ["margin_unit"]),
    ("Crater rim / ancient crust", "#a99c8e", 1.5, ["witch_hazel_hill"]),
    ("Impact craters", "#c98b6b", 0.7, ["belva_crater", "floor_crater_north", "floor_crater_east"]),
]

MODES = [
    {"id": "elevation", "label": "Elevation", "tag": "real",
     "about": "CTX DTM elevation (m, relative to the Mars aeroid). Low crater floor in slate, high rim and "
              "fan surfaces in sand."},
    {"id": "slope", "label": "Slope", "tag": "derived",
     "about": "Slope in degrees from the DTM (~59 m sampling). Sand above the 15° caution slope, terracotta "
              "above the 25° traversability limit."},
    {"id": "class", "label": "Terrain class", "tag": "derived",
     "about": "DTM physiography from slope and local relief: a planning aid, not a geologic map."},
    {"id": "geology", "label": "Geological features", "tag": "lit",
     "about": "Approximate zones around catalogued, published features (positions ±0.5–1 km). Not a geologic "
              "map: uncoloured ground is simply not catalogued."},
    {"id": "science", "label": "Scientific interest", "tag": "derived",
     "about": "MARSCOPE science value (0–100): proximity to published targets, terrain exposure and THEMIS "
              "night-IR. A ranking aid, not a measurement."},
    {"id": "imagery", "label": "CTX image", "tag": "real",
     "about": "MRO CTX orthoimage (~20 m/px, NASA/JPL-Caltech/MSSS via USGS ARD) draped on the DTM."},
]


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def mesh_latlon(ds) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of the mesh vertices (rows north -> south)."""
    g = ds.fine_grid
    rows = np.arange(0, g.height, MESH_STEP)
    cols = np.arange(0, g.width, MESH_STEP)
    lat = g.lat_max - (rows + 0.5) * g.res
    lon = g.lon_min + (cols + 0.5) * g.res
    return lat, lon


def geology_zones(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """1-based GEO_GROUPS index of the nearest catalogued feature within its radius, else 0."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    best = np.full(la.shape, np.inf, np.float32)
    zone = np.zeros(la.shape, np.uint8)
    pois = {p["id"]: p for p in sd.pois()}
    for gi, (_, _, radius_km, ids) in enumerate(GEO_GROUPS, start=1):
        for pid in ids:
            p = pois.get(pid)
            if p is None:
                continue
            ratio = haversine_m(la, lo, p["lat"], p["lon"]) / (radius_km * 1000)
            take = (ratio < 1) & (ratio < best)
            best[take] = ratio[take]
            zone[take] = gi
    return zone


def terrain_data(ds, tm, sm) -> dict:
    """Static mesh payload (cache it): geometry, every colour layer and the CTX texture."""
    s = MESH_STEP
    elev = ds.elevation[::s, ::s].astype(np.float32)
    elev = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    lat, lon = mesh_latlon(ds)
    g = ds.fine_grid
    lat_c, lon_c = (g.lat_min + g.lat_max) / 2, (g.lon_min + g.lon_max) / 2
    kx = KM_PER_DEG * math.cos(math.radians(lat_c))
    lo, hi = np.percentile(elev, [2, 98])
    tex = view3d.terrain_payload(ds)["tex"]
    return {
        "id": f"{ds.status}-{elev.shape}-{float(elev.mean()):.1f}",
        "nx": int(elev.shape[1]), "ny": int(elev.shape[0]),
        "lat0": float(lat[0]), "lon0": float(lon[0]), "dlat": float(lat[1] - lat[0]), "dlon": float(lon[1] - lon[0]),
        "lat_c": lat_c, "lon_c": lon_c, "kx": kx, "ky": KM_PER_DEG, "zmid": float(np.median(elev)),
        "elev_lo": float(lo), "elev_hi": float(hi),
        "elev": _b64(np.round(elev).astype("<i2")),
        "slope": _b64(np.clip(np.round(tm.slope_deg[::s, ::s]), 0, 90).astype(np.uint8)),
        "cls": _b64(tm.terrain_class[::s, ::s].astype(np.uint8)),
        "sci": _b64(np.clip(np.round(sm.value[::s, ::s] * 100), 0, 100).astype(np.uint8)),
        "geo": _b64(geology_zones(lat, lon)),
        "tex": tex,
        "synthetic": bool(ds.is_demonstration),
        "caution": config.ASSUMPTIONS["caution_slope_deg"],
        "max_slope": config.ASSUMPTIONS["max_traversable_slope_deg"],
        "legend": {
            "class": [[k, ta.TERRAIN_CLASSES[k][0], CLASS_COLORS[k]] for k in sorted(ta.TERRAIN_CLASSES)],
            "geology": [[i, name, col] for i, (name, col, _, _) in enumerate(GEO_GROUPS, start=1)],
        },
        "modes": MODES,
    }


def _route_record(r) -> dict:
    p = r.profile
    n = len(p["lat"])
    k = max(1, n // 700)
    idx = list(range(0, n, k))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return {"key": r.key, "name": r.name, "color": r.color,
            "pts": [[round(float(p["lat"][i]), 5), round(float(p["lon"][i]), 5), round(float(p["elevation_m"][i]), 1),
                     round(float(p["distance_m"][i]), 1), round(float(p["slope_deg"][i]), 1)] for i in idx]}


def scene(routes, selected_key, start, destination, selection, section, traverse=None, focus=None) -> dict:
    """Dynamic scene: routes, catalogued features, endpoints, selection and the cross-section line."""
    out = {"routes": [_route_record(r) for r in (routes or []) if r.found and r.profile],
           "selected": selected_key,
           "pois": [{"id": p["id"], "name": p["name"], "lat": p["lat"], "lon": p["lon"], "category": p["category"],
                     "priority": p["priority"]} for p in sd.pois()],
           "start": list(start[:2]) if start else None, "dest": list(destination[:2]) if destination else None,
           "selection": list(selection) if selection else None,
           "section": section, "focus": focus,
           "pace": config.SIM_PACE_M_PER_SOL, "visit_m": config.ASSUMPTIONS["poi_visit_radius_m"],
           "traverse": None}
    if traverse:
        segs = []
        for sgm in traverse:
            c = sgm["coords"]
            k = max(1, len(c) // 30)
            segs.append([[round(a, 5), round(b, 5)] for a, b in c[::k]] + [[round(c[-1][0], 5), round(c[-1][1], 5)]])
        out["traverse"] = segs
    return out


def cross_section(ds, tm, sm, lat: float, lon: float, azimuth_deg: float, length_km: float, n: int = 240) -> dict:
    """Surface profile through (lat, lon) at full DTM resolution. azimuth 0 = N-S, 90 = W-E."""
    half = length_km / 2
    az = math.radians(azimuth_deg)
    d = np.linspace(-half, half, n)                  # km along the section
    kx = KM_PER_DEG * math.cos(math.radians(lat))
    lats = lat + d * math.cos(az) / KM_PER_DEG
    lons = lon + d * math.sin(az) / kx
    g = ds.fine_grid
    inside = (lats > g.lat_min) & (lats < g.lat_max) & (lons > g.lon_min) & (lons < g.lon_max)
    lats, lons, d = lats[inside], lons[inside], d[inside]
    elev = ta.sample_along(tm.elevation, g, lats, lons)
    slope = ta.sample_along(tm.slope_deg, g, lats, lons)
    sci = ta.sample_along(sm.value, g, lats, lons)
    r = np.clip(((g.lat_max - lats) / g.res).astype(int), 0, g.height - 1)
    c = np.clip(((lons - g.lon_min) / g.res).astype(int), 0, g.width - 1)
    cls = tm.terrain_class[r, c]
    crossings = []
    for p in sd.pois():
        dist = haversine_m(lats, lons, p["lat"], p["lon"])
        i = int(np.argmin(dist)) if len(dist) else 0
        if len(dist) and dist[i] < 800:
            crossings.append({"name": p["name"], "d_km": float(d[i] + half), "elev": float(elev[i]),
                              "off_m": float(dist[i])})
    return {"d_km": (d + half).round(3).tolist(), "lat": lats.round(5).tolist(), "lon": lons.round(5).tolist(),
            "elev": np.round(elev, 1).tolist(), "slope": np.round(slope, 1).tolist(),
            "sci": np.round(sci * 100).tolist(), "cls": cls.astype(int).tolist(), "crossings": crossings,
            "azimuth": azimuth_deg, "length_km": length_km, "centre": [lat, lon],
            "ends": [[float(lats[0]), float(lons[0])], [float(lats[-1]), float(lons[-1])]] if len(lats) else None}


def render(data: dict, scn: dict, height: int = 700, key: str = "terrain3d"):
    """Draw the component. Returns the latest event dict ({type: select|poi, ...}) or None."""
    return _component(data=data, scene=scn, height=height, key=key, default=None)
