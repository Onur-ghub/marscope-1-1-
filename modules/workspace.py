"""Workspace sections that connect the modules: 3D colour map, discovery catalogue, route science.

Shared state (st.session_state):
* ``focus``     {"lat", "lon", "label", "poi"}: the location the user is looking at. Written by a 3D
                click or pin, the Mission Control map (Inspect), the discovery catalogue and the
                Science Discovery picker; read by the 3D map, the cross-section, Science Discovery
                and the Assistant context.
* ``focus3d``   {"lat", "lon", "n"}: a request for the 3D camera to fly to a point.
* ``ws``        the open workspace section.
Everything shown is computed from MARSCOPE data and tagged by evidence type; nothing is invented.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from modules import science_discovery as sd
from modules import terrain3d
from modules import terrain_analysis as ta
from modules import visualizations as viz

SECTIONS = ["Mission Control", "3D Terrain", "Science Discovery", "Exploration Routes", "Assistant",
            "Data & Methods"]
GEO_OF_POI = {pid: name for name, _, _, ids in terrain3d.GEO_GROUPS for pid in ids}


# ---------------------------------------------------------------------------
# Shared focus
# ---------------------------------------------------------------------------
def set_focus(lat: float, lon: float, label: str, poi: str | None = None, fly: bool = False) -> None:
    ss = st.session_state
    ss.focus = {"lat": float(lat), "lon": float(lon), "label": label, "poi": poi}
    if poi:
        ss.science_select = poi
        ss.mc_target = poi
    else:
        ss.inspect = (float(lat), float(lon))
        ss.science_select = "__inspect__"
    if fly:
        ss.focus3d = {"lat": float(lat), "lon": float(lon), "n": time.time()}


def focus_point(destination=None) -> dict:
    """The current focus, falling back to the destination, then the landing site."""
    f = st.session_state.get("focus")
    if f:
        return f
    if destination:
        return {"lat": destination[0], "lon": destination[1], "label": destination[2], "poi": None}
    p = sd.poi_by_id(config.DEMO_START_POI)
    return {"lat": p["lat"], "lon": p["lon"], "label": p["name"], "poi": p["id"]}


def open_section(name: str) -> None:
    if name == "Assistant":                      # the Assistant lives in the map's mission panel
        from modules import flow_ui
        if flow_ui.open_assistant():
            return
    st.session_state.ws = name
    st.session_state.view = "all"


# ---------------------------------------------------------------------------
# Evidence basis for catalogued features (from the catalogue text, never guessed)
# ---------------------------------------------------------------------------
_ROVER = ("rover", "perseverance", "sample", "pixl", "sherloc", "supercam")
_ORBIT = ("crism", "orbital", "orbit", "ctx", "themis", "hirise")


def evidence_basis(p: dict) -> tuple[str, str]:
    """(confidence label, explanation) from the kind of evidence the catalogue cites."""
    texts = [o["text"].lower() for o in p["observations"]] + [m.lower() for m in p.get("minerals_reported", [])]
    srcs = {o["source"] for o in p["observations"]}
    rover = any(any(k in t for k in _ROVER) for t in texts)
    orbit = any(any(k in t for k in _ORBIT) for t in texts)
    if srcs == {"MARSCOPE"}:
        return "Unverified", "Identified by MARSCOPE in the DTM/imagery; no published analysis."
    if rover and orbit:
        return "Observed in situ + orbit", "Rover observations and orbital remote sensing both cited."
    if rover:
        return "Observed in situ", "Rover observations cited."
    if orbit:
        return "Orbital detection", "Orbital remote sensing only; not yet confirmed on the ground."
    return "Published", "Published mission or literature source."


def water_flag(p: dict) -> str:
    w = (p.get("water_evidence") or "").lower()
    if not w or w.startswith("unknown") or w.startswith("not primarily"):
        return "—"
    return "yes"


def discovery_table(pois: list[dict]) -> pd.DataFrame:
    rows = []
    for p in sorted(pois, key=lambda p: -p["priority"]):
        conf, _ = evidence_basis(p)
        rows.append({"id": p["id"], "Feature": p["name"], "Setting": p["category"],
                     "Geological group": GEO_OF_POI.get(p["id"], "Mission landmark"),
                     "Location": f"{p['lat']:.4f}° N, {p['lon']:.4f}° E",
                     "Evidence": conf, "Water evidence": water_flag(p), "Priority": p["priority"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Imagery crop (real CTX orthoimage)
# ---------------------------------------------------------------------------
def tile_crop(lat: float, lon: float, half_km: float = 3.0):
    """(RGB array, source label) stitched from the bundled CTX/HiRISE map tiles, or (None, None)."""
    from PIL import Image
    from modules import tiles
    dlat = half_km / terrain3d.KM_PER_DEG
    dlon = dlat / np.cos(np.radians(lat))
    bounds = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    hb = tiles.HIRISE_BOX
    choices = []
    if hb[0] <= lon <= hb[2] and hb[1] <= lat <= hb[3]:
        choices.append(("hirise", 13, "MRO HiRISE (NASA/JPL-Caltech/UArizona) over CTX, via USGS ARD"))
    choices.append(("ctx", 11, "MRO CTX (NASA/JPL-Caltech/MSSS), via USGS ARD"))
    for layer, z, label in choices:
        base = tiles.layer_dir(layer) / str(z)
        if not base.exists():
            continue
        x0, x1, y0, y1 = tiles.tile_range(bounds, z)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 64:
            continue
        canvas = Image.new("RGB", ((x1 - x0 + 1) * tiles.TILE, (y1 - y0 + 1) * tiles.TILE))
        found = 0
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                f = base / str(x) / f"{y}.jpg"
                if f.exists():
                    canvas.paste(Image.open(f).convert("RGB"), ((x - x0) * tiles.TILE, (y - y0) * tiles.TILE))
                    found += 1
        if found < (x1 - x0 + 1) * (y1 - y0 + 1) * 0.6:
            continue
        t = tiles.tile_deg(z)
        px = tiles.TILE / t
        left = (bounds[0] - (-180 + x0 * t)) * px
        top = ((90 - y0 * t) - bounds[3]) * px
        img = canvas.crop((int(left), int(top), int(left + 2 * dlon * px), int(top + 2 * dlat * px)))
        return np.asarray(img), label
    return None, None


def ctx_crop(ds, lat: float, lon: float, half_km: float = 2.0):
    if ds.ctx_image is None:
        return None
    g = ds.image_grid
    dlat = half_km / terrain3d.KM_PER_DEG
    dlon = dlat / np.cos(np.radians(lat))
    r0, c0 = g.lonlat_to_rowcol(lat + dlat, lon - dlon)
    r1, c1 = g.lonlat_to_rowcol(lat - dlat, lon + dlon)
    r0, r1 = sorted((max(0, r0), min(g.height, r1)))
    c0, c1 = sorted((max(0, c0), min(g.width, c1)))
    if r1 - r0 < 8 or c1 - c0 < 8:
        return None
    crop = ds.ctx_image[r0:r1, c0:c1]
    return None if (crop > 0).mean() < 0.3 else crop


# ---------------------------------------------------------------------------
# Charts (all from MARSCOPE layers; units and provenance in the titles)
# ---------------------------------------------------------------------------
def section_figure(sec: dict, label: str) -> go.Figure:
    fig = go.Figure()
    d, z = sec["d_km"], sec["elev"]
    names = [ta.TERRAIN_CLASSES[c][0].split(" (")[0] for c in sec["cls"]]
    fig.add_trace(go.Scatter(x=d, y=z, mode="lines", line=dict(color="#d6cbef", width=2), fill="tozeroy",
                             fillcolor="rgba(184,166,227,.16)", name="Surface (CTX DTM)",
                             customdata=np.stack([sec["slope"], names, sec["sci"]], axis=1),
                             hovertemplate="%{x:.2f} km · %{y:,.0f} m<br>slope %{customdata[0]}°<br>"
                                           "%{customdata[1]}<br>science value %{customdata[2]}/100<extra></extra>"))
    zmin, zmax = min(z), max(z)
    fig.update_yaxes(range=[zmin - 0.08 * (zmax - zmin + 20), zmax + 0.15 * (zmax - zmin + 20)],
                     title_text="Elevation (m, aeroid)")
    mid = sec["length_km"] / 2
    fig.add_vline(x=mid, line=dict(color="#7fb8ae", dash="dot"), annotation_text=label[:28],
                  annotation_font_color="#9fb3c8")
    for q in sec["crossings"]:
        fig.add_trace(go.Scatter(x=[q["d_km"]], y=[q["elev"]], mode="markers+text", text=[q["name"].split(" (")[0]],
                                 textposition="top center", marker=dict(color="#d8cfe9", size=8),
                                 textfont=dict(size=10, color="#e4e7ea"), showlegend=False,
                                 hovertemplate=f"{q['name']}<br>{q['off_m']:.0f} m from the section line<extra></extra>"))
    fig.update_xaxes(title_text=f"Distance along section (km), azimuth {sec['azimuth']:.0f}°")
    return viz._theme(fig, height=300, showlegend=False, margin=dict(l=60, r=16, t=40, b=45),
                      title=dict(text="Topographic cross-section through the selected location (CTX DTM, ~59 m)",
                                 font=dict(size=13)))


def composition_figures(tm, zones: np.ndarray) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Area per terrain class with its elevation range, and area per catalogued feature zone."""
    cell_km2 = (tm.fine_grid.cell_size_m()[0] * tm.fine_grid.cell_size_m()[1]) / 1e6
    names, area, cols, box = [], [], [], go.Figure()
    for k, (name, _) in ta.TERRAIN_CLASSES.items():
        m = tm.terrain_class == k
        names.append(name.split(" (")[0])
        area.append(float(m.sum() * cell_km2))
        cols.append(terrain3d.CLASS_COLORS[k])
        vals = tm.elevation[m]
        if vals.size:
            q = np.percentile(vals, [5, 25, 50, 75, 95])
            box.add_trace(go.Box(name=name.split(" (")[0][:22], x=[name.split(" (")[0][:22]], q1=[q[1]], median=[q[2]], q3=[q[3]],
                                 lowerfence=[q[0]], upperfence=[q[4]], marker_color=terrain3d.CLASS_COLORS[k],
                                 hovertemplate="%{x}<br>median %{median:,.0f} m<extra></extra>"))
    bar = go.Figure(go.Bar(x=area, y=names, orientation="h", marker_color=cols,
                           hovertemplate="%{y}: %{x:,.0f} km²<extra></extra>",
                           text=[f"{a / sum(area) * 100:.0f}%" for a in area], textposition="outside"))
    bar.update_xaxes(title_text="Area (km²)")
    bar = viz._theme(bar, height=260, margin=dict(l=190, r=40, t=40, b=40),
                     title=dict(text="Terrain classes in the study area (DTM-derived)", font=dict(size=13)))
    box.update_yaxes(title_text="Elevation (m)")
    box = viz._theme(box, height=260, showlegend=False, margin=dict(l=60, r=16, t=40, b=60),
                     title=dict(text="Elevation range of each terrain class (5–95th percentile)", font=dict(size=13)))
    zone_area = []
    mesh_km2 = cell_km2 * terrain3d.MESH_STEP ** 2
    for i, (name, col, _, _) in enumerate(terrain3d.GEO_GROUPS, start=1):
        zone_area.append((name, float((zones == i).sum() * mesh_km2), col))
    zfig = go.Figure(go.Bar(x=[a for _, a, _ in zone_area], y=[n for n, _, _ in zone_area], orientation="h",
                            marker_color=[c for _, _, c in zone_area],
                            hovertemplate="%{y}: ~%{x:,.0f} km²<extra></extra>"))
    zfig.update_xaxes(title_text="Approximate zone area (km²)")
    zfig = viz._theme(zfig, height=260, margin=dict(l=230, r=20, t=40, b=40),
                      title=dict(text="Catalogued geological settings (literature zones, approximate)",
                                 font=dict(size=13)))
    return bar, box, zfig


def route_terrain_shares(route, tm) -> dict:
    """Share of the route's distance in each terrain class (sampled along the profile)."""
    p = route.profile
    g = tm.fine_grid
    lat, lon = np.asarray(p["lat"]), np.asarray(p["lon"])
    r = np.clip(((g.lat_max - lat) / g.res).astype(int), 0, g.height - 1)
    c = np.clip(((lon - g.lon_min) / g.res).astype(int), 0, g.width - 1)
    cls = tm.terrain_class[r, c]
    seg = np.diff(np.asarray(p["distance_m"]), prepend=0.0)
    tot = max(seg.sum(), 1)
    return {k: float(seg[cls == k].sum() / tot) for k in ta.TERRAIN_CLASSES}


def route_terrain_figure(routes, tm) -> go.Figure:
    fig = go.Figure()
    found = [r for r in routes if r.found and r.profile]
    shares = {r.key: route_terrain_shares(r, tm) for r in found}
    for k, (name, _) in ta.TERRAIN_CLASSES.items():
        fig.add_trace(go.Bar(y=[f"Route {r.key}" for r in found], x=[shares[r.key][k] * 100 for r in found],
                             orientation="h", name=name.split(" (")[0], marker_color=terrain3d.CLASS_COLORS[k],
                             hovertemplate=name.split(" (")[0] + ": %{x:.0f}% of distance<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Share of route distance (%)", range=[0, 100])
    return viz._theme(fig, height=250, margin=dict(l=70, r=20, t=40, b=70),
                      legend=dict(orientation="h", y=-0.35),
                      title=dict(text="Terrain crossed by each route (DTM-derived classes)", font=dict(size=13)))


def route_story(route, tm) -> list[str]:
    """Why this route is scientifically interesting: features, settings, terrain, from the data."""
    m = route.metrics
    visited = m.get("visited", [])
    out = []
    groups = {}
    for pid, name, dist, _ in visited:
        groups.setdefault(GEO_OF_POI.get(pid, "Mission landmark"), []).append((pid, name, dist))
    n_settings = len([g for g in groups if g != "Mission landmark"])
    out.append(f"**Route {route.key} · {route.name}** covers {m['distance_km']:.2f} km (estimate) and passes "
               f"{len(visited)} catalogued feature(s) within {config.ASSUMPTIONS['poi_visit_radius_m']:.0f} m, "
               f"sampling {n_settings} of the {len(terrain3d.GEO_GROUPS)} geological settings in MARSCOPE's catalogue.")
    for gname, items in groups.items():
        for pid, name, dist in items:
            p = sd.poi_by_id(pid)
            interp = p["interpretations"][0] if p["interpretations"] else None
            obs = p["observations"][0] if p["observations"] else None
            line = f"- **{name}** ({gname}, {dist:.0f} m from the route)"
            if obs:
                line += f": observed, {obs['text'].rstrip('.')} [{obs['source']}]"
            if interp:
                line += f". Interpretation (hypothesis): {interp['text'].rstrip('.')} [{interp['source']}]"
            out.append(line + ".")
    if not visited:
        out.append("- It passes no catalogued feature; its value is the terrain itself and the destination.")
    sh = route_terrain_shares(route, tm)
    top = sorted(sh.items(), key=lambda kv: -kv[1])[:2]
    out.append("- Terrain (derived): " + ", ".join(f"{v * 100:.0f}% {ta.TERRAIN_CLASSES[k][0].split(' (')[0].lower()}"
                                                   for k, v in top)
               + f"; climb {m['elev_gain_m']:.0f} m, descent {m['elev_loss_m']:.0f} m, max slope {m['max_slope_deg']:.0f}°.")
    return out


# ---------------------------------------------------------------------------
# 3D Terrain section
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=2)
def _terrain_data(_ds, _tm, _sm, key):
    return terrain3d.terrain_data(_ds, _tm, _sm)


@st.cache_data(show_spinner=False, max_entries=2)
def _zones(_ds, key):
    lat, lon = terrain3d.mesh_latlon(_ds)
    return terrain3d.geology_zones(lat, lon)


def handle_3d_event(key: str = "terrain3d") -> None:
    """Apply the component's last event before the scene is built (so the scene is never stale)."""
    ss = st.session_state
    ev = ss.get(key)
    if not ev or ev.get("n") == ss.get("_t3d_n"):
        return
    ss._t3d_n = ev.get("n")
    from modules import mission_log as mlog
    if ev.get("type") == "poi":
        p = sd.poi_by_id(ev.get("id"))
        if p:
            set_focus(p["lat"], p["lon"], p["name"], p["id"])
            mlog.log("Feature selected in 3D", p["name"])
    elif ev.get("type") == "select":
        set_focus(ev["lat"], ev["lon"], "Selected point")
        mlog.log("Location selected in 3D", f"{ev['lat']:.4f}° N, {ev['lon']:.4f}° E")


def terrain_section(ds, tm, sm, routes, selected_key, start, dest, mission, badge, on_plan_here) -> None:
    ss = st.session_state
    ss.setdefault("sec_az", 90)
    ss.setdefault("sec_len", 10.0)
    handle_3d_event()
    f = focus_point(dest)
    data = _terrain_data(ds, tm, sm, (ds.status, ds.elevation.shape))
    sec = terrain3d.cross_section(ds, tm, sm, f["lat"], f["lon"], float(ss.sec_az), float(ss.sec_len))
    scn = terrain3d.scene(routes, selected_key, start, dest, (f["lat"], f["lon"]),
                          {k: sec[k] for k in ("d_km", "lat", "lon", "elev", "slope", "cls", "crossings", "azimuth",
                                               "ends")},
                          traverse=mission.get("traverse") if ss.get("lyr_traverse", True) else None,
                          focus=ss.get("focus3d"))
    left, right = st.columns([3.15, 1.1], gap="small")
    with left, st.container(key="t3d_stage"):
        terrain3d.render(data, scn, height=720)
    with right, st.container(key="t3d_side"):
        info = sd.inspect_location(ds, tm, sm, f["lat"], f["lon"])
        t = info["terrain"]
        st.markdown(f"<div class='t3-h'>{f['label']}</div><div class='t3-sub'>{f['lat']:.4f}° N · {f['lon']:.4f}° E"
                    + (" · catalogued position ±0.5–1 km" if f.get("poi") else "") + "</div>", unsafe_allow_html=True)
        rows = [("Elevation", f"{t['elevation_m']:,.0f} m", "observation" if info["dtm_measured"] else "derived"),
                ("Slope", f"{t['slope_deg']:.1f}° (max {t['max_slope_1km_deg']:.0f}° within 1 km)", "derived"),
                ("Terrain class", t["terrain_class"].split(" (")[0], "derived"),
                ("Science value", f"{info['science_value'] * 100:.0f}/100", "derived"),
                ("Nearest feature", (f"{info['nearby'][0][0].split(' (')[0]} · {info['nearby'][0][1] / 1000:.1f} km")
                 if info["nearby"] else "none within 3 km", "literature")]
        st.markdown("<div class='t3-rows'>" + "".join(
            f"<div><span>{k}</span><b>{v}</b>{badge(tag)}</div>" for k, v, tag in rows) + "</div>",
            unsafe_allow_html=True)
        if info["why"]:
            st.markdown("<div class='t3-why'>" + "<br>".join(f"• {w}" for w in info["why"][:3]) + "</div>",
                        unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.button("Science details", key="t3_sci", on_click=open_section, args=("Science Discovery",),
                  width="stretch", help="Observations, interpretations, minerals, imagery and citations")
        c2.button("Ask Assistant", key="t3_ask", width="stretch", on_click=_ask_about, args=(f["label"],))
        st.button("Plan a route to this point", key="t3_plan", width="stretch", on_click=on_plan_here,
                  args=(f["lat"], f["lon"], f["label"]))
        st.markdown("<div class='fl-label'>Jump to a catalogued feature</div>", unsafe_allow_html=True)
        ids = ["—"] + [p["id"] for p in sorted(sd.pois(), key=lambda p: -p["priority"])]
        st.selectbox("Feature", ids, key="t3_jump", label_visibility="collapsed",
                     format_func=lambda i: "Choose a feature…" if i == "—" else sd.poi_by_id(i)["name"],
                     on_change=_jump)
        st.markdown("<div class='fl-label'>Cross-section through this point</div>", unsafe_allow_html=True)
        st.slider("Azimuth (°, 0 = north–south, 90 = west–east)", 0, 175, step=5, key="sec_az")
        st.slider("Length (km)", 2.0, 20.0, step=1.0, key="sec_len")
        st.caption("The lavender curtain in 3D marks the section. ▶ Scan in the 3D view animates it; the chart "
                   "below is the same profile at full DTM resolution.")
    st.plotly_chart(section_figure(sec, f["label"]), width="stretch", key="sec_fig")
    zones = _zones(ds, (ds.status, ds.elevation.shape))
    bar, box, zfig = composition_figures(tm, zones)
    a, b, c = st.columns(3)
    a.plotly_chart(bar, width="stretch", key="comp_bar")
    b.plotly_chart(box, width="stretch", key="comp_box")
    c.plotly_chart(zfig, width="stretch", key="comp_zone")
    st.caption("How to read the 3D map: every colour mode is a data layer on the same ~300 m mesh of the CTX DTM. "
               "Elevation and the CTX image are observations; slope, terrain class and science value are MARSCOPE "
               "derivations; geological features are approximate zones around published targets. The rover in the "
               "3D view is a SIMULATED playback along the planned route at "
               f"{config.SIM_PACE_M_PER_SOL:.0f} m/sol; it is not mission telemetry. Vertical exaggeration is "
               "adjustable and stated on screen; the rover symbol is not to scale.")


def _jump() -> None:
    pid = st.session_state.get("t3_jump")
    if pid and pid != "—":
        p = sd.poi_by_id(pid)
        set_focus(p["lat"], p["lon"], p["name"], pid, fly=True)
    st.session_state.t3_jump = "—"


def _ask_about(label: str) -> None:
    ss = st.session_state
    ss.assistant_pending = f"What is scientifically interesting about the selected location ({label})?"
    open_section("Assistant")
