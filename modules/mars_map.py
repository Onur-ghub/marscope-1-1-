"""Interactive Mars map (Folium / Leaflet) for MARSCOPE.

The map uses Leaflet's EPSG:4326 (plate carree) CRS. Mars coordinates are
planetocentric latitude / east longitude, so the gridded layers (which are on a
regular lat/lon grid) overlay exactly, and NASA Solar System Treks WMTS tiles
(equirectangular, 2x1 tiles at zoom 0) line up as well.
"""
from __future__ import annotations

import base64
import io

import folium
import numpy as np
from PIL import Image

import config
from modules import tiles
from modules.data_manager import MarsDataset, haversine_m
from modules.science_discovery import mineral_zones, pois
from modules.terrain_analysis import TERRAIN_CLASSES, TerrainModel

# ---------------------------------------------------------------------------
# Colour maps (small, dependency-free)
# ---------------------------------------------------------------------------
CMAPS = {
    "mars_elev": [(0.0, (25, 20, 35)), (0.25, (70, 45, 70)), (0.5, (160, 70, 45)),
                  (0.75, (225, 140, 70)), (1.0, (250, 225, 180))],
    "slope": [(0.0, (30, 140, 80)), (0.35, (200, 210, 60)), (0.6, (245, 150, 40)),
              (0.8, (225, 60, 40)), (1.0, (140, 0, 60))],
    "inferno": [(0.0, (0, 0, 4)), (0.25, (87, 16, 110)), (0.5, (188, 55, 84)),
                (0.75, (249, 142, 9)), (1.0, (252, 255, 164))],
    "science": [(0.0, (20, 30, 60)), (0.4, (40, 110, 140)), (0.7, (240, 180, 60)),
                (1.0, (255, 90, 50))],
}


def apply_cmap(values01: np.ndarray, name: str) -> np.ndarray:
    stops = CMAPS[name]
    xs = np.array([s[0] for s in stops])
    out = np.zeros(values01.shape + (3,), np.float32)
    v = np.clip(np.nan_to_num(values01, nan=0), 0, 1)
    for ch in range(3):
        out[..., ch] = np.interp(v, xs, [s[1][ch] for s in stops])
    return out


def _to_data_url(rgb: np.ndarray, alpha: np.ndarray | None = None, fmt: str = "PNG",
                 max_width: int = 1800) -> str:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if alpha is not None:
        img = Image.fromarray(np.dstack([rgb, np.clip(alpha, 0, 255).astype(np.uint8)]), "RGBA")
        fmt = "PNG"
    else:
        img = Image.fromarray(rgb, "RGB")
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)), Image.BILINEAR)
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, "JPEG", quality=82)
    else:
        img.save(buf, "PNG", optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Layer rendering (pure functions -> cached by the app)
# ---------------------------------------------------------------------------
LAYER_SPECS = {
    # key: (label, kind, legend)
    "elevation": ("Elevation - CTX DTM, colour + hillshade", "observation",
                  "Dark purple = low crater floor, pale = high rim"),
    "slope": ("Terrain slope (derived)", "derived", "Green < 5 deg, yellow ~10 deg, red > 20 deg"),
    "difficulty": ("Terrain difficulty index (derived)", "derived", "0 (easy, transparent) to 1 (hard, red)"),
    "impassable": ("Impassable cells (slope limit)", "derived", "Red: exceeds the max traversable slope"),
    "terrain_class": ("Terrain classes - DTM physiography (derived)", "derived",
                      "Not a geologic map; see legend in the sidebar"),
    "themis_day": ("THEMIS daytime IR (Mars Odyssey)", "observation",
                   "Bright = warm by day (sun-facing slopes, low albedo)"),
    "themis_night": ("THEMIS nighttime IR (Mars Odyssey)", "observation",
                     "Bright = warm at night = rockier / higher thermal inertia (qualitative)"),
    "science": ("Science-value surface (derived)", "derived", "Blue low to orange/red high"),
    "minerals": ("Literature-reported mineral detections (CRISM & rover)", "literature",
                 "Approximate zones around published targets, not CRISM pixels"),
}


def render_layers(ds: MarsDataset, tm: TerrainModel, science_value: np.ndarray) -> dict[str, str]:
    """Render every raster overlay to a data URL once."""
    layers = {}
    if ds.ctx_image is not None:
        g = ds.ctx_image.astype(np.float32)
        layers["ctx"] = _to_data_url(np.dstack([g, g * 0.93, g * 0.85]), fmt="JPEG")
    elev = tm.elevation
    lo, hi = np.percentile(elev, [1, 99])
    col = apply_cmap((elev - lo) / (hi - lo), "mars_elev")
    shade = 0.35 + 0.65 * tm.hillshade[..., None]
    layers["elevation"] = _to_data_url(col * shade, fmt="JPEG")
    layers["hillshade"] = _to_data_url(np.repeat(tm.hillshade[..., None] * 230 + 15, 3, axis=2), fmt="JPEG")

    sl = np.clip(tm.slope_deg / 30.0, 0, 1)
    layers["slope"] = _to_data_url(apply_cmap(sl, "slope"), alpha=np.full(sl.shape, 175))
    d = tm.difficulty
    layers["difficulty"] = _to_data_url(apply_cmap(d, "slope"), alpha=np.clip(d * 2.2, 0, 1) * 220)
    imp = tm.slope_deg > config.ASSUMPTIONS["max_traversable_slope_deg"]
    rgb = np.zeros(imp.shape + (3,)); rgb[..., 0] = 255; rgb[..., 1] = 30; rgb[..., 2] = 60
    layers["impassable"] = _to_data_url(rgb, alpha=imp * 210)

    cls_rgb = np.zeros(tm.terrain_class.shape + (3,))
    for code, (_, hexcol) in TERRAIN_CLASSES.items():
        c = tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
        cls_rgb[tm.terrain_class == code] = c
    layers["terrain_class"] = _to_data_url(cls_rgb, alpha=np.full(tm.terrain_class.shape, 150))

    for key in ("themis_day", "themis_night"):
        arr = getattr(ds, key)
        if arr is not None:
            a = arr.astype(np.float32)
            valid = a > 0
            p2, p98 = np.percentile(a[valid], [2, 98])
            layers[key] = _to_data_url(apply_cmap((a - p2) / (p98 - p2), "inferno"),
                                       alpha=valid * 215)
    layers["science"] = _to_data_url(apply_cmap(science_value, "science"),
                                     alpha=np.clip(science_value, 0.15, 1) * 210)
    return layers


# ---------------------------------------------------------------------------
# Map assembly
# ---------------------------------------------------------------------------
def build_map(ds: MarsDataset, layers: dict[str, str], active: list[str], base: str,
              routes: list | None = None, selected_route: str | None = None,
              start=None, destination=None, inspect=None, height_hint: int = 680,
              mission: dict | None = None, opacity: float = 1.0, rover=None,
              completed: list | None = None, target=None, recommended: str | None = None) -> folium.Map:
    """``mission`` may hold live NASA/USGS layers: traverse, waypoints, geology (see mission_data)."""
    mission = mission or {}
    g = ds.fine_grid
    bounds = [[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]]
    m = folium.Map(location=list(config.JEZERO_CENTER), zoom_start=10, crs="EPSG4326",
                   tiles=None, min_zoom=7, max_zoom=18, control_scale=False,
                   prefer_canvas=True, attribution_control=True)

    # ---- base layers (radio buttons in the layer control) -----------------
    bases = []
    ctx_info, hi_info = tiles.layer_info("ctx"), tiles.layer_info("hirise")
    if ctx_info:
        fg = folium.FeatureGroup(name="High-res imagery: CTX ~20 m + HiRISE ~1.2 m (MRO, via USGS)",
                                 overlay=False, show=base == "hires")
        folium.TileLayer(tiles=tiles.url_template("ctx"), name="CTX", no_wrap=True,
                         attr="MRO CTX: NASA/JPL-Caltech/MSSS; orthos: USGS Astrogeology ARD",
                         min_zoom=7, min_native_zoom=ctx_info["min_zoom"],
                         max_native_zoom=ctx_info["max_zoom"], max_zoom=18,
                         bounds=[[ctx_info["bounds"][1], ctx_info["bounds"][0]],
                                 [ctx_info["bounds"][3], ctx_info["bounds"][2]]]).add_to(fg)
        if hi_info:
            hb = hi_info["bounds"]
            folium.TileLayer(tiles=tiles.url_template("hirise"), name="HiRISE", no_wrap=True,
                             attr="MRO HiRISE: NASA/JPL-Caltech/UArizona; orthos: USGS Astrogeology ARD",
                             min_zoom=hi_info["min_zoom"], min_native_zoom=hi_info["min_zoom"],
                             max_native_zoom=hi_info["max_zoom"], max_zoom=18,
                             bounds=[[hb[1], hb[0]], [hb[3], hb[2]]]).add_to(fg)
        bases.append(fg)
    if "ctx" in layers:
        fg = folium.FeatureGroup(name="CTX orthoimage mosaic, single image (MRO, via USGS)", overlay=False,
                                 show=base == "ctx")
        folium.raster_layers.ImageOverlay(layers["ctx"], bounds=bounds, opacity=1.0,
                                          attr="MRO CTX: NASA/JPL-Caltech/MSSS; mosaic: USGS ARD").add_to(fg)
        bases.append(fg)
    fg = folium.FeatureGroup(name="Hillshade from CTX DTM (derived)", overlay=False, show=base == "hillshade")
    folium.raster_layers.ImageOverlay(layers["hillshade"], bounds=bounds).add_to(fg)
    bases.append(fg)
    for label, (url, attr) in config.TREK_LAYERS.items():
        key = "trek_mola" if "MOLA" in label else "trek_viking"
        fg = folium.FeatureGroup(name=label + " [loads from trek.nasa.gov]", overlay=False, show=base == key)
        folium.TileLayer(tiles=url, attr=attr, name=label, max_native_zoom=7,
                         max_zoom=18, no_wrap=True).add_to(fg)
        bases.append(fg)
    for fg in bases:
        fg.add_to(m)

    # ---- USGS global geologic map (SIM 3292), drawn under everything else ----
    geo = mission.get("geology")
    if geo and "usgs_geology" in active:
        palette = {"HNt": "#c9a35a", "mNhm": "#8d6e63", "eNh": "#795548", "mNh": "#a1887f",
                   "lNh": "#bcaaa4", "AHi": "#ffcc80"}
        fg = folium.FeatureGroup(name="USGS geologic map of Mars (Tanaka et al. 2014, SIM 3292)", show=True)
        folium.GeoJson(geo, style_function=lambda f: {
            "color": "#ffe082", "weight": 1.2,
            "fillColor": palette.get(f["properties"].get("Unit"), "#e0e0e0"),
            "fillOpacity": 0.28 * opacity},
            tooltip=folium.GeoJsonTooltip(["Unit", "UnitName"], aliases=["Unit", "Name"], sticky=True)
        ).add_to(fg)
        fg.add_to(m)

    # ---- overlays ----------------------------------------------------------
    for key in active:
        if key == "minerals":
            fg = folium.FeatureGroup(name=LAYER_SPECS[key][0], show=True)
            for z in mineral_zones():
                lines = [f"<b>{z['name']}</b>", "<i>Literature-reported, approximate zone</i>"]
                lines += [f"Orbital: {o}" for o in z["orbital"]] + [f"Rover: {r}" for r in z["rover"]]
                lines.append("Refs: " + ", ".join(z["refs"]))
                folium.Circle([z["lat"], z["lon"]], radius=z["radius_m"], color="#26c6da",
                              weight=1.5, dash_array="4 4", fill=True, fill_opacity=0.12,
                              tooltip=folium.Tooltip("<br>".join(lines), sticky=True)).add_to(fg)
            fg.add_to(m)
            continue
        if key not in layers or key not in LAYER_SPECS:
            continue
        fg = folium.FeatureGroup(name=LAYER_SPECS[key][0], show=True)
        folium.raster_layers.ImageOverlay(layers[key], bounds=bounds,
                                          opacity=(0.85 if key == "elevation" else 1.0) * opacity).add_to(fg)
        fg.add_to(m)

    # ---- optional user-supplied geologic map (GeoJSON / shapefile) -------------
    geo = load_user_geology()
    if geo is not None:
        fg = folium.FeatureGroup(name="User-supplied geologic map (data/geology/)", show=True)
        label_col = next((c for c in geo.columns if c.lower() in ("unit", "unitsymbol", "unit_name", "name", "label")), None)
        folium.GeoJson(geo.__geo_interface__, style_function=lambda f: {"color": "#ffd54f", "weight": 1,
                                                                      "fillOpacity": 0.15},
                       tooltip=folium.GeoJsonTooltip([label_col]) if label_col else None).add_to(fg)
        fg.add_to(m)

    # ---- Perseverance actual traverse (NASA/JPL MMGIS) --------------------
    trav, wps = mission.get("traverse") or [], mission.get("waypoints") or []
    if "traverse" in active and (trav or wps):
        fg = folium.FeatureGroup(name="Perseverance actual traverse (NASA/JPL)", show=True)
        sols = [s["sol"] for s in trav] or [0]
        s0, s1 = min(sols), max(max(sols), min(sols) + 1)
        for seg in trav:
            t = (seg["sol"] - s0) / (s1 - s0)
            col = "#%02x%02x%02x" % (255, int(255 - 120 * t), int(255 - 225 * t))   # white -> orange
            folium.PolyLine(seg["coords"], color="#000", weight=5, opacity=0.45).add_to(fg)
            folium.PolyLine(seg["coords"], color=col, weight=3, opacity=0.95,
                            tooltip=f"Perseverance drive, sol {seg['sol']} ({seg['length_m']:.0f} m)").add_to(fg)
        for i, w in enumerate(wps):
            last = i == len(wps) - 1
            km = w.get("dist_total_km")
            html = (f"<div style='width:230px'><b>Perseverance, sol {w['sol']}</b><br>"
                    f"Site {w.get('site')} · drive {w.get('drive')}<br>"
                    + (f"Odometry: {float(km):.2f} km<br>" if km not in (None, "") else "")
                    + (f"Elevation: {float(w['elev_m']):.0f} m<br>" if w.get("elev_m") not in (None, "") else "")
                    + (f"<a href='{w['panorama_url']}' target='_blank'>Navcam panorama</a><br>"
                       if w.get("panorama_url") else "")
                    + "<small>Source: NASA/JPL-Caltech MMGIS. Select a nearby POI in the Science tab "
                      "for raw images from this sol.</small></div>")
            folium.CircleMarker([w["lat"], w["lon"]], radius=6 if last else 2.5,
                                color="#000" if last else "#ffb74d", weight=1 if last else 0,
                                fill=True, fill_color="#ffffff" if last else "#ffb74d", fill_opacity=0.95,
                                tooltip=("Latest position, " if last else "") + f"sol {w['sol']}",
                                popup=folium.Popup(html, max_width=260)).add_to(fg)
        fg.add_to(m)

    # ---- POIs --------------------------------------------------------------
    poi_fg = folium.FeatureGroup(name="Points of interest", show=True)
    for p in pois():
        colour = "#D9C39C" if p["origin"] == "published" else "#9e9e9e"
        html = (f"<div style='width:260px'><b>{p['name']}</b><br><i>{p['category']}</i><br>"
                f"MARSCOPE priority: {p['priority']:.2f}<br>"
                f"{p['observations'][0]['text'] if p['observations'] else ''} "
                f"<small>[{p['observations'][0]['source'] if p['observations'] else ''}]</small><br>"
                f"<small>Position approximate. Select it in the Science tab for details.</small></div>")
        folium.CircleMarker([p["lat"], p["lon"]], radius=6, color="#111", weight=1.5,
                            fill=True, fill_color=colour, fill_opacity=0.95,
                            tooltip=p["name"], popup=folium.Popup(html, max_width=300)).add_to(poi_fg)
    poi_fg.add_to(m)

    # ---- routes ------------------------------------------------------------
    if routes:
        rfg = folium.FeatureGroup(name="Planned routes", show=True)
        ordered = sorted(routes, key=lambda r: r.key == selected_route)  # selected drawn last
        for r in ordered:
            if not r.found or r.lats is None:
                continue
            sel = r.key == selected_route
            pts = list(zip(r.lats.tolist(), r.lons.tolist()))
            if sel:
                folium.PolyLine(pts, color="#000", weight=9, opacity=0.6).add_to(rfg)
            folium.PolyLine(pts, color=r.color, weight=6 if sel else 3,
                            opacity=1.0 if sel else 0.75, dash_array=None if sel else "6 6",
                            tooltip=("★ Recommended · " if r.key == recommended else "") + f"Route {r.key} - {r.name}: {r.metrics['distance_km']:.2f} km, "
                                    f"science {r.metrics['science_score']:.0f}/100").add_to(rfg)
        rfg.add_to(m)
        sel = next((r for r in routes if r.key == selected_route and r.found and r.lats is not None), None)
        if sel is not None:
            _numbered_stops(sel).add_to(m)

    # ---- simulated rover progress (Mission Control) --------------------------
    if completed and len(completed) >= 2:
        fg = folium.FeatureGroup(name="Completed segment (SIMULATED progress)", show=True)
        folium.PolyLine(completed, color="#ffffff", weight=4, opacity=0.95,
                        tooltip="Driven part of the route (simulated)").add_to(fg)
        fg.add_to(m)
    if rover:
        folium.Marker(rover, tooltip="Rover position (SIMULATED along the planned route)",
                      icon=folium.DivIcon(html="<div style='width:18px;height:18px;background:#d9b48c;"
                                               "border:3px solid #ffffff;box-shadow:0 0 0 2px #000;"
                                               "transform:rotate(45deg)'></div>",
                                          icon_size=(24, 24), icon_anchor=(12, 12))).add_to(m)
    if target:
        folium.CircleMarker(target, radius=13, color="#ffd54f", weight=2.5, fill=False,
                            tooltip="Selected science target").add_to(m)

    def marker(latlon, label, colour, icon):
        folium.Marker(latlon, tooltip=label,
                      icon=folium.Icon(color=colour, icon=icon, prefix="fa")).add_to(m)

    if start:
        marker(start, "START", "green", "play")
    if destination:
        marker(destination, "DESTINATION", "red", "flag-checkered")
    if inspect:
        marker(inspect, "Inspected location", "blue", "crosshairs")

    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    m.fit_bounds(bounds if not (start and destination) else
                 _padded_bounds([start, destination] + ([] if not routes else [
                     (float(lat), float(lon)) for r in routes if r.found and r.lats is not None
                     for lat, lon in zip(r.lats[::5], r.lons[::5])])))
    return m


def _numbered_stops(route) -> folium.FeatureGroup:
    """NASA "proposed path" style: numbered stops at each POI the route passes, in driving order."""
    fg = folium.FeatureGroup(name=f"Route {route.key} stops (numbered)", show=True)
    by_id = {p["id"]: p for p in pois()}
    order = []
    for pid, name, *_ in route.metrics.get("visited", []):
        p = by_id.get(pid)
        if p is None:
            continue
        idx = int(np.argmin(haversine_m(p["lat"], p["lon"], route.lats, route.lons)))
        order.append((idx, p))
    for n, (idx, p) in enumerate(sorted(order, key=lambda t: t[0]), 1):
        html = (f"<div style='display:flex;align-items:center;gap:6px;white-space:nowrap'>"
                f"<div style='width:22px;height:22px;border-radius:11px;background:{route.color};"
                f"color:#111;font:700 12px sans-serif;display:flex;align-items:center;justify-content:center;"
                f"border:2px solid #111'>{n}</div>"
                f"<div style='font:600 11px sans-serif;color:#fff;text-shadow:0 0 3px #000,0 0 3px #000'>"
                f"{p['name']}</div></div>")
        folium.Marker([p["lat"], p["lon"]], icon=folium.DivIcon(html=html, icon_size=(220, 24),
                                                               icon_anchor=(11, 12)),
                      tooltip=f"Stop {n}: {p['name']}").add_to(fg)
    return fg


def load_user_geology():
    """Load the first GeoJSON/shapefile in data/geology/ (lat/lon), or return None."""
    folder = config.USER_GEOLOGY_DIR
    if not folder.exists():
        return None
    files = sorted(list(folder.glob("*.geojson")) + list(folder.glob("*.json")) + list(folder.glob("*.shp")))
    if not files:
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_file(files[0])
        if gdf.crs is not None and not gdf.crs.is_geographic:
            from modules.data_manager import MARS_GEOGRAPHIC_PROJ4
            gdf = gdf.to_crs(MARS_GEOGRAPHIC_PROJ4)
        return gdf.cx[config.REGION_BOUNDS[0]:config.REGION_BOUNDS[2], config.REGION_BOUNDS[1]:config.REGION_BOUNDS[3]]
    except Exception:
        return None


def _padded_bounds(points, pad=0.02):
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return [[min(lats) - pad, min(lons) - pad], [max(lats) + pad, max(lons) + pad]]
