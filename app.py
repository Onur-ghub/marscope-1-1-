"""MARSCOPE - Intelligent Mars exploration & mission planning for Jezero Crater.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules import st_compat

st_compat.install()       # older Streamlit (>=1.50): drop/emulate keyword arguments it does not know

import config
from modules import data_manager as dm
from modules import mars_map, route_optimizer as ro, science_discovery as sd
from modules import terrain_analysis as ta, visualizations as viz
from modules import landing, mission_data as md, tiles
from modules import ai_service, assistant_ui, mission_context as mc, mission_planner as mp
from modules import mission_control as mcc, mission_log as mlog, rover_sim
from modules import flow_ui, workspace as wsp

st.set_page_config(page_title="MARSCOPE - Jezero Mission Planner", page_icon="🔴",
                   layout="wide", initial_sidebar_state="collapsed")

CSS_PATH = config.ASSETS_DIR / "style.css"
st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

KIND_BADGE = {
    "observation": ("OBSERVATION", "badge-obs"),
    "derived": ("DERIVED", "badge-der"),
    "literature": ("LITERATURE", "badge-lit"),
    "demonstration": ("DEMONSTRATION", "badge-demo"),
}


def badge(kind: str) -> str:
    label, cls = KIND_BADGE.get(kind, (kind.upper(), "badge-der"))
    return f"<span class='badge {cls}'>{label}</span>"


# ---------------------------------------------------------------------------
# Cached core objects
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_dataset(refresh_token: int):
    progress = st.progress(0.0, text="Loading Jezero datasets...")

    def cb(msg, frac):
        progress.progress(min(frac, 1.0), text=msg)

    ds = dm.load_or_fetch(force_live=refresh_token > 0, progress=cb)
    progress.empty()
    return ds


@st.cache_resource(show_spinner=False)
def build_models(refresh_token: int, max_slope: float):
    ds = load_dataset(refresh_token)
    config.ASSUMPTIONS["max_traversable_slope_deg"] = max_slope
    tm = ta.build_terrain_model(ds)
    sm = sd.build_science_model(ds, tm)
    planner = ro.RoutePlanner(tm, sm)
    layers = mars_map.render_layers(ds, tm, sm.value)
    return tm, sm, planner, layers


@st.cache_data(ttl=3600, show_spinner=False)
def load_mission_layers(refresh_token: int) -> dict:
    """Live NASA/USGS mission layers (cached on disk too). Missing layers stay empty, never invented."""
    trav, s_trav = md.perseverance_traverse()
    wps, s_wps = md.perseverance_waypoints()
    geo, s_geo = md.geologic_units()
    return {"traverse": trav, "waypoints": wps, "geology": geo,
            "status": {"traverse": s_trav, "waypoints": s_wps, "geology": s_geo}}


@st.cache_data(ttl=3600, show_spinner=False)
def load_raw_images(sol: int) -> tuple:
    return md.raw_images_for_sol(sol)


def _tiles_note() -> str:
    c, h = tiles.layer_info("ctx"), tiles.layer_info("hirise")
    if not c:
        return "Not built. Run <code>python scripts/build_tiles.py</code> to create them from USGS ARD."
    txt = f"CTX: {c['tiles']} tiles, zoom {c['min_zoom']}–{c['max_zoom']} (~20 m/px). "
    if h:
        txt += (f"HiRISE: {h['tiles']} tiles, zoom {h['min_zoom']}–{h['max_zoom']} (~1.2 m/px at max zoom) over "
                f"the landing site, delta front and Belva crater; {h['hirise_pixel_fraction']:.0%} of pixels are "
                f"HiRISE, gaps between strips are filled with CTX. ")
    return txt + "Rendered from USGS Astrogeology ARD orthoimages (CC0), built " + c["built_at"][:10] + "."


def init_state():
    defaults = {"refresh_token": 0, "start": None, "destination": None, "inspect": None,
                "routes": None, "route_sig": None, "selected_route": "C", "last_click": None,
                "click_mode": "Inspect location", "ai_cache": {}, "science_target": None,
                "w_science": 0.50, "w_safety": 0.30, "w_distance": 0.20, "strength": 0.70,
                "max_slope": config.ASSUMPTIONS["max_traversable_slope_deg"],
                "base_layer": "hires" if tiles.layer_info("ctx") else "ctx", "overlay_opacity": 1.0,
                "lyr_traverse": True, "lyr_usgs_geology": False,
                "view": "setup", "landing_request": None, "landing_error": None,
                "max_route_km": 0.0, "chat": [], "chat_cache": {}, "assistant_pending": None,
                "sim_progress_pct": 0, "mc_target": None, "mc_inspected": set(), "mc_passed": set(),
                "lyr_impassable": True, "ws": "Mission Control", "focus": None, "focus3d": None}
    defaults.update({"start_select": "—", "dest_select": "—"})
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    # Widget values changed by actions that happen after the widget was drawn are applied here,
    # at the top of the next run (Streamlit forbids changing a drawn widget's state).
    for k, v in st.session_state.pop("_pending", {}).items():
        st.session_state[k] = v


def set_later(**kwargs):
    st.session_state.setdefault("_pending", {}).update(kwargs)


def poi_point(poi_id: str):
    p = sd.poi_by_id(poi_id)
    return (p["lat"], p["lon"], p["name"])


def load_demo():
    mlog.log("Demo mission loaded", "Octavia E. Butler Landing → Bright Angel")
    st.session_state.start = poi_point(config.DEMO_START_POI)
    st.session_state.destination = poi_point(config.DEMO_DESTINATION_POI)
    st.session_state.w_science, st.session_state.w_safety, st.session_state.w_distance = 0.50, 0.30, 0.20
    st.session_state.strength = 0.70
    st.session_state.selected_route = "C"
    st.session_state.science_target = config.DEMO_DESTINATION_POI
    st.session_state.science_select = config.DEMO_DESTINATION_POI
    st.session_state.start_select = config.DEMO_START_POI
    st.session_state.dest_select = config.DEMO_DESTINATION_POI
    st.session_state.inspect = None


def route_signature(start, dest, status) -> tuple:
    """Identifies a route computation, so routes are only recomputed when an input changes."""
    w = ro.Weights(st.session_state.w_distance, st.session_state.w_safety, st.session_state.w_science)
    return (tuple(start[:2]), tuple(dest[:2]), w.normalised().as_tuple(), st.session_state.strength,
            st.session_state.max_slope, status)


init_state()
if st.session_state.view not in ("setup", "routes", "navigate", "all"):
    st.session_state.view = "setup"

# ---------------------------------------------------------------------------
# Mission navigation flow (setup -> routes -> navigate). "all" = Mission Control & all data.
# ---------------------------------------------------------------------------
_ds = load_dataset(st.session_state.refresh_token)
_tm, _sm, _planner, _layers = build_models(st.session_state.refresh_token, float(st.session_state.max_slope))
config.ASSUMPTIONS["max_traversable_slope_deg"] = float(st.session_state.max_slope)
FLOW = flow_ui.App(ds=_ds, tm=_tm, sm=_sm, planner=_planner, layers=_layers, pois=sd.pois(),
                   poi_by_id={p["id"]: p for p in sd.pois()}, route_signature=route_signature,
                   set_later=set_later, poi_point=poi_point, badge=badge,
                   mission_waypoints=lambda: load_mission_layers(st.session_state.refresh_token)["waypoints"])
flow_ui.init(FLOW)
if st.session_state.view != "all":
    flow_ui.run(FLOW)
    st.stop()
flow_ui.inject_css("all")
flow_ui.top_bar(FLOW)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='brand'>MARS<span>COPE</span></div>"
                "<div class='brand-sub'>Jezero Crater mission planner</div>", unsafe_allow_html=True)
    if st.button("▶  Load demo mission", width="stretch", type="primary",
                 help="Landing site → Bright Angel (Neretva Vallis), across the ancient delta"):
        load_demo()
    if st.button("↩  Mission setup", width="stretch", help="Back to the mission setup screen"):
        st.session_state.view = "setup"
        st.rerun()

ds = load_dataset(st.session_state.refresh_token)
tm, sm, planner, layers = build_models(st.session_state.refresh_token, float(st.session_state.max_slope))
mission = load_mission_layers(st.session_state.refresh_token)
config.ASSUMPTIONS["max_traversable_slope_deg"] = float(st.session_state.max_slope)
poi_list = sd.pois()
poi_names = {p["id"]: p["name"] for p in poi_list}

with st.sidebar:
    status_kind = "demonstration" if ds.is_demonstration else "observation"
    status_text = {"bundled": "Real USGS data - bundled snapshot", "cache": "Real USGS data - local cache",
                   "live": "Real USGS data - downloaded now",
                   "demonstration": "SYNTHETIC demonstration terrain"}[ds.status]
    st.markdown(f"<div class='status-card {'demo' if ds.is_demonstration else ''}'>{badge(status_kind)}"
                f"<div class='status-title'>{status_text}</div>"
                f"<div class='status-detail'>{ds.status_detail}</div></div>", unsafe_allow_html=True)

    with st.expander("🎯 Mission setup", expanded=True):
        options = ["—"] + [p["id"] for p in poi_list]

        def fmt(pid):
            return "Choose a point of interest…" if pid == "—" else poi_names[pid]

        def _on_select(widget_key, state_key):
            v = st.session_state[widget_key]
            if v == "—":            # the placeholder never clears a location (use ✕ Clear); some Streamlit
                return              # versions report it spuriously when the widget re-mounts
            st.session_state[state_key] = poi_point(v)
            if v != "—":
                mlog.log("Destination selected" if state_key == "destination" else "Start selected", poi_names[v])

        # keep the widgets in step with the shared endpoints (set on the route map, the 3D view or here)
        for _wk, _sk in (("start_select", "start"), ("dest_select", "destination")):
            _pt = st.session_state.get(_sk)
            _pid = next((p["id"] for p in poi_list if _pt and p["name"] == _pt[2]), "—")
            if st.session_state.get(_wk) != _pid:
                st.session_state[_wk] = _pid
        st.selectbox("Start", options, format_func=fmt, key="start_select",
                     on_change=_on_select, args=("start_select", "start"))
        st.selectbox("Destination", options, format_func=fmt, key="dest_select",
                     on_change=_on_select, args=("dest_select", "destination"))
        for label, key in (("Start", "start"), ("Destination", "destination")):
            pt = st.session_state[key]
            if pt and pt[2] == "Map point":
                st.caption(f"{label}: map point {pt[0]:.4f}°N, {pt[1]:.4f}°E")
        c1, c2 = st.columns(2)
        if c1.button("⇅ Swap", width="stretch"):
            st.session_state.start, st.session_state.destination = (st.session_state.destination,
                                                                    st.session_state.start)
            set_later(start_select=st.session_state.dest_select, dest_select=st.session_state.start_select)
            st.rerun()
        if c2.button("✕ Clear", width="stretch"):
            st.session_state.start = st.session_state.destination = None
            st.session_state.routes = None
            set_later(start_select="—", dest_select="—")
            st.rerun()
        st.caption("Tip: you can also click the map to place the start, destination or an inspection point.")

    with st.expander("⚖️ Mission priorities", expanded=True):
        st.slider("Scientific value", 0.0, 1.0, key="w_science", step=0.05)
        st.slider("Terrain safety", 0.0, 1.0, key="w_safety", step=0.05)
        st.slider("Distance efficiency", 0.0, 1.0, key="w_distance", step=0.05)
        st.slider("Strategy strength", 0.0, 1.0, key="strength", step=0.05,
                  help="How strongly each route keeps its own strategy (A safest, B shortest, C science) "
                       "versus your priorities. 0 = all routes use your weights; 1 = pure strategies. "
                       "Route D always uses exactly your weights.")
        uw = ro.Weights(st.session_state.w_distance, st.session_state.w_safety,
                        st.session_state.w_science).normalised()
        st.caption(f"Normalised: science {uw.science:.2f} · safety {uw.safety:.2f} · distance {uw.distance:.2f}. "
                   "These weights change the edge costs of the A*/Dijkstra search, not just the display.")

    with st.expander("🛞 Traverse assumptions", expanded=False):
        st.slider("Max traversable slope (deg)", 10.0, 35.0, key="max_slope", step=1.0,
                  help="Cells whose steepest 10% of slopes exceed this are impassable.")
        lo, hi = config.ASSUMPTIONS["indicative_speed_m_per_sol"]
        st.markdown(f"<div class='small-note'>Caution slope: {config.ASSUMPTIONS['caution_slope_deg']:.0f}°<br>"
                    f"POI counted as visited within {config.ASSUMPTIONS['poi_visit_radius_m']:.0f} m<br>"
                    f"Science detour budget: ×{config.ASSUMPTIONS['science_detour_budget']}<br>"
                    f"Indicative pace {lo:.0f}–{hi:.0f} m/sol (planning assumption, not a capability claim)"
                    f"</div>", unsafe_allow_html=True)

    with st.expander("🗺️ Data layers", expanded=False):
        base_opts = {"hires": "High-res CTX + HiRISE tiles (real, zoom in)",
                     "ctx": "CTX orthoimage, single image (real)", "hillshade": "DTM hillshade (derived)",
                     "trek_mola": "NASA Trek MOLA relief (online)", "trek_viking": "NASA Trek Viking colour (online)"}
        if "ctx" not in layers:
            base_opts.pop("ctx")
        if not tiles.layer_info("ctx"):
            base_opts.pop("hires")
        if st.session_state.get("base_layer") not in base_opts:
            st.session_state.base_layer = next(iter(base_opts))
        base = st.radio("Base map", list(base_opts), format_func=base_opts.get, key="base_layer")
        active = []
        for key, (label, kind, legend) in mars_map.LAYER_SPECS.items():
            if key not in layers and key != "minerals":
                continue
            default = key in ("minerals",)
            if st.checkbox(label, value=st.session_state.get(f"lyr_{key}", default), key=f"lyr_{key}",
                           help=f"{KIND_BADGE[kind][0]} · {legend}"):
                active.append(key)
        st.markdown("<div class='panel-title'>Live NASA / USGS layers</div>", unsafe_allow_html=True)
        mst = mission["status"]
        status_txt = {"live": "live", "cache": "cached", "stale-cache": "cached (offline)",
                      "unavailable": "unavailable (offline, no cache)"}
        if st.checkbox("Perseverance actual traverse (NASA/JPL)", key="lyr_traverse",
                       disabled=not (mission["traverse"] or mission["waypoints"]),
                       help="Drive path and waypoints from NASA's 'Where is Perseverance?' map (MMGIS).") \
                and (mission["traverse"] or mission["waypoints"]):
            active.append("traverse")
        st.caption(f"Traverse: {status_txt[mst['traverse']]}"
                   + (f" · {len(mission['traverse'])} drives, up to sol "
                      f"{max(w['sol'] for w in mission['waypoints'])}" if mission["waypoints"] else ""))
        if st.checkbox("USGS geologic map (SIM 3292)", key="lyr_usgs_geology",
                       disabled=mission["geology"] is None,
                       help="Tanaka et al. 2014 global geologic map, from the USGS Astrogeology ArcGIS service. "
                            "Global 1:20M scale, so units are coarse at Jezero."):
            active.append("usgs_geology")
        st.caption(f"Geology: {status_txt[mst['geology']]}")
        st.slider("Overlay opacity", 0.2, 1.0, step=0.05, key="overlay_opacity")
        st.markdown("<div class='legend'>" + "".join(
            f"<div><span style='background:{c}'></span>{n}</div>" for n, c in ta.TERRAIN_CLASSES.values())
            + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Route computation (automatic whenever inputs change)
# ---------------------------------------------------------------------------
user_w = ro.Weights(st.session_state.w_distance, st.session_state.w_safety, st.session_state.w_science)
start, dest = st.session_state.start, st.session_state.destination
if start and dest:
    sig = route_signature(start, dest, ds.status)
    if st.session_state.route_sig != sig:
        with st.spinner("Optimising routes (A* + science orienteering)..."):
            t0 = time.time()
            st.session_state.routes = ro.plan_all(planner, start[:2], dest[:2], user_w,
                                                  st.session_state.strength, include_custom=True)
            st.session_state.route_time = time.time() - t0
            first = st.session_state.route_sig is None
            st.session_state.route_sig = sig
            _rk = ro.rank_routes(st.session_state.routes, user_w, st.session_state.max_route_km or None)
            mlog.log("Route generated" if first else "Route recalculated",
                     f"{dest[2]}: {len(st.session_state.routes)} candidates in "
                     f"{st.session_state.route_time:.2f} s" + (f"; recommended Route {_rk[0][0].key}" if _rk else ""))
routes = st.session_state.routes if (start and dest) else None
evaluation = mp.evaluate_routes(
    routes, mp.ScoringModel(st.session_state.w_science, st.session_state.w_safety, st.session_state.w_distance),
    mp.MissionConstraints(max_route_km=st.session_state.max_route_km or None,
                          max_slope_deg=float(st.session_state.max_slope))) if routes else None
ranking = [(c.route, c.score) for c in evaluation.ranked] if evaluation else []
route_by_key = {r.key: r for r in routes} if routes else {}
selected = route_by_key.get(st.session_state.selected_route) if routes else None

# ---------------------------------------------------------------------------
# Mission Control state (rover simulation, targets, AI context) and callbacks
# ---------------------------------------------------------------------------
ai_ok = ai_service.status()[0]
real_rover = mission["waypoints"][-1] if mission["waypoints"] else None
sel_route_ = route_by_key.get(st.session_state.selected_route) if routes else None
rover = rover_sim.state_at(sel_route_, st.session_state.sim_progress_pct / 100, poi_list) if sel_route_ else None
if not mlog.events():
    mlog.log("Mission initialized", f"terrain: {ds.status} ({'synthetic' if ds.is_demonstration else 'USGS'})")
for pid in (rover.passed_targets if rover else []):
    if pid not in st.session_state.mc_passed:
        st.session_state.mc_passed.add(pid)
        mlog.log("Target reached", f"{poi_names[pid]} (simulated rover within "
                                   f"{config.ASSUMPTIONS['poi_visit_radius_m']:.0f} m)", "sim")

mission_sig = mc.signature(start, dest, st.session_state.route_sig, st.session_state.selected_route,
                           st.session_state.max_route_km) + f"|{st.session_state.sim_progress_pct}"
mission_ctx = mc.build_context(ds, tm, sm, start, dest, evaluation, st.session_state.selected_route,
                               mission["waypoints"])
if rover:
    mission_ctx["simulated_rover"] = {
        "note": "SIMULATED position along the planned route; MARSCOPE is not connected to a real rover.",
        "route": sel_route_.key, "progress_pct": round(rover.progress * 100), "lat": round(rover.lat, 4),
        "lon_east": round(rover.lon, 4), "traveled_km": round(rover.traveled_m / 1000, 2),
        "remaining_km": round(rover.remaining_m / 1000, 2), "elevation_m_derived": round(rover.elevation_m),
        "slope_deg_derived": round(rover.slope_deg, 1),
        "energy_used_kwh_simulated": round(rover.energy_used_wh / 1000, 2),
        "targets_passed": [poi_names[p] for p in rover.passed_targets],
        "next_target_on_route": ({"name": rover.next_target[1], "km_ahead": round(rover.next_target[2] / 1000, 2)}
                                 if rover.next_target else None)}
    mission_ctx["route_energy_kwh_simulated"] = {
        r.key: round(rover_sim.route_energy_wh(r) / 1000, 2) for r in routes if r.found}
_focus = st.session_state.get("focus")
if _focus:
    _finfo = sd.inspect_location(ds, tm, sm, _focus["lat"], _focus["lon"])
    mission_ctx["selected_location"] = {
        "name": _focus["label"], "lat": round(_focus["lat"], 4), "lon_east": round(_focus["lon"], 4),
        "catalogued_point_of_interest": _focus.get("poi"),
        "note": "The location the user selected on the 3D colour map, the Mission Control map or in Science "
                "Discovery. 'Selected location', 'this location' and 'here' (when no destination is meant) refer to it.",
        **sd.build_fact_sheet(_finfo), "why_interesting_marscope": _finfo["why"]}
    mission_sig += f"|focus:{_focus['lat']:.4f},{_focus['lon']:.4f}"


def _advance_sol(route):
    new = rover_sim.progress_after_one_sol(route, st.session_state.sim_progress_pct / 100)
    st.session_state.sim_progress_pct = int(round(new * 100))
    mlog.log("Simulated sol advanced", f"Route {route.key}: {st.session_state.sim_progress_pct}%", "sim")


def _reset_sim():
    st.session_state.sim_progress_pct = 0
    mlog.log("Simulation reset", "rover back at the start", "sim")


def _route_changed():
    mlog.log("Route selected", f"Route {st.session_state.selected_route}")


def _target_selected():
    st.session_state.mc_inspected.add(st.session_state.mc_target)
    mlog.log("Target inspected", poi_names.get(st.session_state.mc_target, ""))


def _apply_scenario(r):
    w = r["weights"]
    st.session_state.w_science, st.session_state.w_safety, st.session_state.w_distance = (
        round(w["science"], 2), round(w["safety"], 2), round(w["distance"], 2))
    mlog.log("Scenario applied", f"{r['scenario']}: weights {w['science']:.0%}/{w['safety']:.0%}/{w['distance']:.0%}")


def _target_rows(route, rover_state) -> list[dict]:
    stops = {}
    if route is not None and route.found:
        order = sorted(((int(np.argmin(dm.haversine_m(sd.poi_by_id(v[0])["lat"], sd.poi_by_id(v[0])["lon"],
                                                       route.lats, route.lons))), v[0])
                        for v in route.metrics.get("visited", [])))
        stops = {pid: n + 1 for n, (_, pid) in enumerate(order)}
    out = []
    for p in poi_list:
        r_, c_ = ds.fine_grid.lonlat_to_rowcol(p["lat"], p["lon"])
        if rover_state and p["id"] in rover_state.passed_targets:
            status = "reached (sim)"
        elif p["id"] in stops:
            status = f"stop {stops[p['id']]}"
        else:
            status = "off route"
        if p["id"] in st.session_state.mc_inspected:
            status += " · inspected"
        out.append({"id": p["id"], "short": p["name"].split(" (")[0].split(" / ")[0][:24],
                    "priority": p["priority"], "science_value": float(sm.value[r_, c_]),
                    "n_obs": len(p.get("observations", [])), "n_int": len(p.get("interpretations", [])),
                    "n_min": len(p.get("minerals_reported", [])), "status": status,
                    "order": stops.get(p["id"], 99)})
    return sorted(out, key=lambda r: (r["order"], -r["priority"]))


MC_QUESTIONS = ["Which route has the highest scientific value?", "What are the major terrain risks?",
                "Which target should we investigate next?"]


def _mission_intelligence():
    qs = list(MC_QUESTIONS)
    ranked = evaluation.ranked if evaluation else []
    if len(ranked) > 1:
        qs.append(f"What is the trade-off between Route {ranked[0].key} and Route {ranked[1].key}?")
    with st.container(key="ai_suggest_mc"):
        for i, q in enumerate(qs):
            st.button(q, key=f"mcq_{i}", on_click=lambda q=q: st.session_state.update(mc_question=q))
    with st.form("mc_ask", clear_on_submit=True, border=False):
        typed = st.text_input("Ask mission intelligence", placeholder="Ask about routes, risks or targets…",
                              label_visibility="collapsed")
        sent = st.form_submit_button("Ask", width="stretch")
    question = (typed if sent and typed.strip() else None) or st.session_state.pop("mc_question", None)
    if question:
        reply = assistant_ui.get_reply(question, mission_ctx, evaluation, mission_sig, ai_ok)
        assistant_ui.record(question, reply)
        mlog.log("AI query", question[:80])
    last = next((i for i in range(len(st.session_state.chat) - 1, -1, -1)
                 if st.session_state.chat[i]["role"] == "assistant"), None)
    if last is not None:
        q_ = st.session_state.chat[last - 1]["content"] if last > 0 else ""
        a_ = st.session_state.chat[last]
        with st.container(key="mc_answer", height=300):
            st.markdown(f"**{q_}**")
            st.markdown(a_["content"])
            if a_.get("note"):
                st.caption(a_["note"])
        st.caption("Shared with the Assistant in the map's mission panel, which shows the full conversation.")
    else:
        st.caption("Answers are grounded in MARSCOPE data. " + ("" if ai_ok else
                   "Offline mode: set ANTHROPIC_API_KEY for full AI answers."))


if ds.is_demonstration:
    st.error("No real NASA/USGS data could be loaded, so MARSCOPE is showing SYNTHETIC demonstration terrain. "
             "All numbers are illustrative. Use **Data & Methods → Refresh from USGS** when online.")

_ws = st.session_state.ws if st.session_state.ws in wsp.SECTIONS else wsp.SECTIONS[0]
if _ws == "Mission Control":
  mcc.top_bar(
    f"Jezero traverse · {start[2] if start else '—'} → {dest[2] if dest else '—'}",
    [("Terrain", "USGS CTX/THEMIS" if not ds.is_demonstration else "synthetic",
      "real" if not ds.is_demonstration else "sim"),
     ("Rover traverse", {"live": "live", "cache": "cached", "stale-cache": "cached"}.get(
         mission["status"]["traverse"], "offline"),
      "real" if mission["waypoints"] else "na"),
     ("AI", "online" if ai_ok else "offline mode", "derived" if ai_ok else "na")],
    rover.sols_elapsed if rover else None)

# ---------------------------------------------------------------------------
# TAB 1 - Map
# ---------------------------------------------------------------------------
if _ws == "Mission Control":
    sel_route = route_by_key.get(st.session_state.selected_route) if routes else None
    col_side, col_map, col_sci = st.columns([1.05, 2.55, 1.35], gap="small")

    # ---- LEFT: rover, navigation, mission parameters ----------------------------
    with col_side:
        mcc.telemetry(rover, real_rover, sel_route)
        if sel_route is not None:
            st.slider("Simulated progress along the route (%)", 0, 100, key="sim_progress_pct",
                      on_change=lambda: mlog.log("Simulated progress", f"{st.session_state.sim_progress_pct}% of "
                                                 f"Route {st.session_state.selected_route}", "sim"))
            c1, c2 = st.columns(2)
            c1.button("Advance 1 sol", on_click=_advance_sol, args=(sel_route,), width="stretch",
                      help=f"Moves the simulated rover {config.SIM_PACE_M_PER_SOL:.0f} m along the route.")
            c2.button("Reset", on_click=_reset_sim, width="stretch")
        mcc.section("Navigation")
        st.radio("Clicking the map will…", ["Inspect location", "Set start", "Set destination"],
                 key="click_mode")
        if routes:
            rec = ranking[0][0].key if ranking else None
            keys = [r.key for r in routes if r.found]
            if st.session_state.selected_route not in keys and keys:
                st.session_state.selected_route = keys[0]
            st.radio("Route on the map", keys, key="selected_route", on_change=_route_changed,
                     format_func=lambda kk: f"{kk} · {config.STRATEGIES[kk]['name'] if kk in config.STRATEGIES else 'Your weights'}"
                                            + (" ★" if kk == rec else ""))
            if rec:
                st.caption(f"★ Recommended by the Mission Planner: Route {rec} ({route_by_key[rec].name}). "
                           "Ask why in the Assistant (mission panel on the map).")
        elif start or dest:
            st.info("Set both a start and a destination to generate routes.")
        else:
            st.info("Go to ↩ Mission setup (sidebar), click the map, or press **Load demo mission**.")
        mcc.section("Mission parameters")
        w_ = evaluation.model.as_dict() if evaluation else None
        st.markdown("<div class='mc-panel'>"
                    + (mcc.row("Scoring weights", f"sci {w_['science']:.0%} · safe {w_['safety']:.0%} · dist "
                                                  f"{w_['distance']:.0%}", "derived") if w_ else "")
                    + mcc.row("Max traversable slope", f"{st.session_state.max_slope:.0f}°", "derived")
                    + mcc.row("Max route length", f"{st.session_state.max_route_km:g} km"
                              if st.session_state.max_route_km else "no limit", "derived")
                    + mcc.row("Assumed pace", f"{config.SIM_PACE_M_PER_SOL:.0f} m/sol", "sim")
                    + "<div class='mc-foot'>Change these in the sidebar (‹ top left) or the Assistant (mission panel on the map).</div></div>",
                    unsafe_allow_html=True)

    # ---- CENTRE: surface operations map -------------------------------------------
    with col_map:
        mcc.section("Mars surface operations map")
        tgt_poi = sd.poi_by_id(st.session_state.mc_target) if st.session_state.get("mc_target") else None
        completed = None
        if rover and rover.index > 0:
            pr = sel_route.profile
            completed = list(zip(pr["lat"][: rover.index + 1].tolist(), pr["lon"][: rover.index + 1].tolist()))
        fmap = mars_map.build_map(
            ds, layers, active, base, routes=routes, selected_route=st.session_state.selected_route,
            start=start[:2] if start else None, destination=dest[:2] if dest else None,
            inspect=st.session_state.inspect, mission=mission, opacity=st.session_state.overlay_opacity,
            rover=(rover.lat, rover.lon) if rover and rover.progress > 0 else None, completed=completed,
            target=(tgt_poi["lat"], tgt_poi["lon"]) if tgt_poi else None,
            recommended=ranking[0][0].key if ranking else None)
        out = st_folium(fmap, height=640, use_container_width=True, returned_objects=["last_clicked"],
                        key="marsmap")
        legend_bits = [f"<b>{mars_map.LAYER_SPECS[k][0]}</b>: {mars_map.LAYER_SPECS[k][2]}" for k in active if k in mars_map.LAYER_SPECS]
        if rover and rover.progress > 0:
            legend_bits.append("<b>◆ Rover</b> and <b>white line</b>: SIMULATED position and driven segment.")
        if "traverse" in active:
            legend_bits.append("<b>Perseverance actual traverse</b>: drive path coloured from white (early sols) to "
                               "orange (recent); white dot = latest position. NASA/JPL-Caltech.")
        if "usgs_geology" in active:
            legend_bits.append("<b>USGS geologic map (SIM 3292)</b>: hover a unit for its name. 1:20M global map.")
        if legend_bits:
            st.markdown("<div class='map-legend'>" + "<br>".join(legend_bits) + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='map-caption'>Imagery: MRO CTX (NASA/JPL-Caltech/MSSS) and HiRISE "
                    "(NASA/JPL-Caltech/UArizona) via USGS Astrogeology ARD. Layers: sidebar or the layer control "
                    "(top right). Red cells = terrain hazards over the slope limit.</div>", unsafe_allow_html=True)
        click = (out or {}).get("last_clicked")
        if click and click != st.session_state.last_click:
            st.session_state.last_click = click
            lat, lon = float(click["lat"]), float(click["lng"])
            if not ds.fine_grid.contains(lat, lon):
                st.warning("That point is outside the Jezero study area.")
            else:
                mode = st.session_state.click_mode
                if mode == "Set start":
                    st.session_state.start = (lat, lon, "Map point")
                    set_later(start_select="—")
                    mlog.log("Start selected", f"map point {lat:.4f}° N, {lon:.4f}° E")
                elif mode == "Set destination":
                    st.session_state.destination = (lat, lon, "Map point")
                    set_later(dest_select="—")
                    mlog.log("Destination selected", f"map point {lat:.4f}° N, {lon:.4f}° E")
                else:
                    st.session_state.inspect = (lat, lon)
                    st.session_state.focus = {"lat": lat, "lon": lon, "label": "Inspected point", "poi": None}
                    set_later(science_select="__inspect__")
                    mlog.log("Location inspected", f"{lat:.4f}° N, {lon:.4f}° E")
                st.rerun()

    # ---- RIGHT: science operations, inspection, AI ---------------------------------
    with col_sci:
        mcc.section("Science operations")
        target_rows = _target_rows(sel_route, rover)
        ids = [r_["id"] for r_ in target_rows]
        if st.session_state.get("mc_target") not in ids:
            st.session_state.mc_target = (rover.next_target[0] if rover and rover.next_target
                                          else ids[0] if ids else None)
        st.selectbox("Science target", ids, key="mc_target", on_change=_target_selected,
                     format_func=lambda i: poi_names[i])
        mcc.target_table(target_rows, st.session_state.mc_target)
        tp = sd.poi_by_id(st.session_state.mc_target)
        tinfo = sd.inspect_location(ds, tm, sm, tp["lat"], tp["lon"])
        r_, c_ = ds.fine_grid.lonlat_to_rowcol(tp["lat"], tp["lon"])
        mcc.target_evidence(tinfo, tp, bool(ds.dtm_coverage[r_, c_]))
        if st.session_state.inspect:
            lat, lon = st.session_state.inspect
            info = sd.inspect_location(ds, tm, sm, lat, lon)
            t = info["terrain"]
            mcc.section("Inspected location")
            st.markdown("<div class='mc-panel'>"
                        + mcc.row("Coordinates", f"{lat:.4f}° N · {lon:.4f}° E", "real")
                        + mcc.row("Elevation", f"{t['elevation_m']:,.0f} m", "real" if info["dtm_measured"] else "derived")
                        + mcc.row("Slope", f"{t['slope_deg']:.1f}°", "derived")
                        + mcc.row("Terrain", t["terrain_class"].split(" (")[0], "derived")
                        + mcc.row("Science value", f"{info['science_value']:.2f}", "derived")
                        + mcc.row("Nearest target", (info['nearby'][0][0][:26] + f" · {info['nearby'][0][1]/1000:.1f} km")
                                  if info['nearby'] else "none within 3 km", "lit")
                        + "<div class='mc-foot'>Full analysis in Science Discovery; shown in 3D Terrain too.</div></div>",
                        unsafe_allow_html=True)
        mcc.section("AI mission intelligence")
        _mission_intelligence()

    # ---- BOTTOM: strategy, timeline, communications, statistics -------------------
    st.markdown("<div class='mc-divider'></div>", unsafe_allow_html=True)
    b1, b2 = st.columns([1.55, 1], gap="medium")
    with b1:
        mcc.section("Mission strategy")
        if evaluation:
            mcc.strategy(evaluation, {r.key: rover_sim.route_energy_wh(r) for r in routes if r.found},
                         st.session_state.selected_route)
            mcc.section("Alternative mission scenarios", " <small>same routes, re-scored; apply to re-plan</small>")
            mcc.scenarios(mp.scenario_table(routes, evaluation.constraints), _apply_scenario)
        else:
            st.markdown("<div class='mc-empty'>Routes appear here once a destination is set.</div>",
                        unsafe_allow_html=True)
        if sel_route is not None:
            m_ = sel_route.metrics
            mcc.section(f"Route {sel_route.key} and terrain summary")
            st.markdown("<div class='mc-panel mc-stats'>"
                        + mcc.row("Distance / straight line", f"{m_['distance_km']:.2f} / {m_['straight_line_km']:.2f} km", "derived")
                        + mcc.row("Climb / descent", f"{m_['elev_gain_m']:.0f} / {m_['elev_loss_m']:.0f} m", "derived")
                        + mcc.row("Mean / max slope", f"{m_['mean_slope_deg']:.1f}° / {m_['max_slope_deg']:.0f}°", "derived")
                        + mcc.row("Mean terrain difficulty", f"{m_['mean_difficulty']:.2f}", "derived")
                        + mcc.row("Targets passed", f"{m_['pois_visited']}", "derived")
                        + mcc.row("Indicative duration", f"{m_['sols_range'][0]:.0f}–{m_['sols_range'][1]:.0f} sols", "sim")
                        + "</div>", unsafe_allow_html=True)
    with b2:
        mcc.section("Mission activity timeline")
        mcc.timeline(mlog.events())
        mcc.section("Communications")
        mcc.comms()

# ---------------------------------------------------------------------------
# 3D Terrain: interactive colour map, scientific animations, cross-section, distributions
# ---------------------------------------------------------------------------
if _ws == "3D Terrain":
    wsp.terrain_section(ds, tm, sm, routes, st.session_state.selected_route if routes else None, start, dest,
                        mission, badge, lambda la, lo, lb: flow_ui.plan_to(FLOW, la, lo, lb))

# ---------------------------------------------------------------------------
# TAB - Astronaut Assistant (Mission Planner + Science Assistant)
# ---------------------------------------------------------------------------
def _show_route(key):
    st.session_state.selected_route = key


def _choose_destination():
    st.session_state.view = "setup"


if _ws == "Assistant":
    assistant_ui.render(
        mission_ctx, evaluation, st.session_state.selected_route if routes else None,
        mission_sig, ai_service.status(), badge, _show_route, _choose_destination)

# ---------------------------------------------------------------------------
# TAB 2 - Route planner
# ---------------------------------------------------------------------------
if _ws == "Exploration Routes":
    if not routes:
        st.info("Choose a start and destination (sidebar or map) - or press **Load demo mission** - "
                "to generate and compare routes.")
        st.button("▶ Load demo mission (landing site → Bright Angel)", key="rt_demo", on_click=load_demo,
                  type="primary")
    else:
        st.markdown(f"<div class='route-head'>From <b>{start[2]}</b> to <b>{dest[2]}</b> · straight-line "
                    f"{routes[0].metrics.get('straight_line_km', 0):.2f} km</div>", unsafe_allow_html=True)
        cols = st.columns(len(routes))
        rank_pos = {r.key: (i, s) for i, (r, s) in enumerate(ranking)}
        for col, r in zip(cols, routes):
            with col:
                if not r.found:
                    st.markdown(f"<div class='route-card' style='border-top-color:{r.color}'>"
                                f"<div class='rc-title'>Route {r.key} · {r.name}</div>"
                                f"<div class='rc-warn'>{r.message}</div></div>", unsafe_allow_html=True)
                    continue
                m = r.metrics
                cand = next(c for c in evaluation.candidates if c.key == r.key)
                pos, score = rank_pos.get(r.key, (None, cand.score or 0))
                star = ("<span class='best'>★ recommended by the Mission Planner</span>" if pos == 0 else
                        f"<span class='rc-way'>Same path as Route {cand.duplicate_of}</span>" if cand.duplicate_of else
                        "" if pos is not None else
                        f"<span class='rc-warn'>Excluded: {cand.exclusions[0] if cand.exclusions else ''}</span>")
                w = r.weights
                pros = "".join(f"<li>{p}</li>" for p in r.pros)
                cons = "".join(f"<li>{c}</li>" for c in r.cons)
                way = (f"<div class='rc-way'>Planned stops: {', '.join(poi_names[p].split('(')[0].strip() for p in r.waypoints)}</div>"
                       if r.waypoints else "")
                st.markdown(f"""<div class='route-card' style='border-top-color:{r.color}'>
                <div class='rc-title'>Route {r.key} · {r.name}</div>{star}
                <div class='rc-score'>{score:.0f}<small>/100 mission score</small></div>
                <table class='rc-table'>
                <tr><td>Total distance</td><td>{m['distance_km']:.2f} km</td></tr>
                <tr><td>Terrain difficulty (mean / p90)</td><td>{m['mean_difficulty']:.2f} / {m['p90_difficulty']:.2f}</td></tr>
                <tr><td>Max slope · % above {config.ASSUMPTIONS['caution_slope_deg']:.0f}°</td><td>{m['max_slope_deg']:.0f}° · {m['pct_above_caution']:.1f}%</td></tr>
                <tr><td>Cumulative elevation change</td><td>{m['cumulative_elev_change_m']:.0f} m</td></tr>
                <tr><td>Climb / descent</td><td>+{m['elev_gain_m']:.0f} / −{m['elev_loss_m']:.0f} m</td></tr>
                <tr><td>POIs visited (≤{config.ASSUMPTIONS['poi_visit_radius_m']:.0f} m)</td><td>{m['pois_visited']}</td></tr>
                <tr><td>Science-value score</td><td>{m['science_score']:.0f} / 100</td></tr>
                <tr><td>Optimiser weights d/s/c</td><td>{w.distance:.2f} / {w.safety:.2f} / {w.science:.2f}</td></tr>
                </table>{way}
                <div class='rc-pc'><b>Advantages</b><ul>{pros}</ul><b>Limitations</b><ul>{cons}</ul></div>
                </div>""", unsafe_allow_html=True)
                if st.button(f"Select route {r.key}", key=f"sel_{r.key}", width="stretch",
                             type="primary" if r.key == st.session_state.selected_route else "secondary"):
                    set_later(selected_route=r.key)
                    st.rerun()

        selected = route_by_key.get(st.session_state.selected_route)
        st.plotly_chart(viz.elevation_profile(routes, st.session_state.selected_route), width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.tradeoff_radar(routes), width="stretch")
        with c2:
            st.plotly_chart(viz.slope_distribution(routes), width="stretch")
            if selected and selected.found:
                st.plotly_chart(viz.science_along_route(selected), width="stretch")

        if selected and selected.found:
            st.markdown(f"<div class='panel-title'>Scientific points of interest along Route {selected.key}</div>",
                        unsafe_allow_html=True)
            visited = selected.metrics["visited"]
            if visited:
                vcols = st.columns(min(len(visited), 4))
                for i, (pid, name, dist, prio) in enumerate(visited):
                    p = sd.poi_by_id(pid)
                    with vcols[i % len(vcols)]:
                        st.markdown(f"<div class='poi-chip'><b>{name}</b><br><small>{p['category']} · "
                                    f"{dist:.0f} m from route · priority {prio:.2f}</small></div>",
                                    unsafe_allow_html=True)
                        if st.button("Inspect →", key=f"insp_{pid}", width="stretch"):
                            set_later(science_select=pid)
                            st.rerun()
            else:
                st.caption("No catalogued points of interest within the visit radius of this route.")

            gj = {"type": "FeatureCollection", "features": [{
                "type": "Feature",
                "properties": {"route": selected.key, "strategy": selected.name,
                               **{k: v for k, v in selected.metrics.items() if k not in ("visited", "sols_range")},
                               "pois_visited_ids": [v[0] for v in visited],
                               "note": "Coordinates: Mars planetocentric lat / east lon (IAU 2015 sphere)."},
                "geometry": {"type": "LineString",
                             "coordinates": [[float(a), float(b)] for a, b in zip(selected.lons, selected.lats)]}}]}
            dl1, dl2 = st.columns(2)
            dl1.download_button("⬇ Download selected route (GeoJSON)", json.dumps(gj, indent=1),
                                file_name=f"marscope_route_{selected.key}.geojson", mime="application/geo+json",
                                width="stretch")
            prof = pd.DataFrame({k: selected.profile[k] for k in
                                 ("distance_m", "lat", "lon", "elevation_m", "slope_deg", "difficulty", "science")})
            dl2.download_button("⬇ Download elevation profile (CSV)", prof.to_csv(index=False),
                                file_name=f"marscope_profile_{selected.key}.csv", mime="text/csv",
                                width="stretch")

        if selected and selected.found:
            st.markdown(f"<div class='panel-title'>Why Route {selected.key} is scientifically interesting</div>",
                        unsafe_allow_html=True)
            st_l, st_r = st.columns([1.35, 1], gap="medium")
            with st_l:
                st.markdown("\n".join(wsp.route_story(selected, tm)))
                st.caption("Observations and interpretations are quoted from the catalogue with their sources; "
                           "terrain numbers are MARSCOPE-derived estimates.")
                st.button("Play this route in 3D (simulated rover)", key="rt_3d", type="primary",
                          on_click=wsp.open_section, args=("3D Terrain",),
                          help="Opens the 3D colour map with this route; press ▶ Play there.")
            with st_r:
                st.plotly_chart(wsp.route_terrain_figure(routes, tm), width="stretch", key="rt_terrain")
        with st.expander("🏔️ 3D terrain view with routes", expanded=False):
            ex = st.slider("Vertical exaggeration", 1.0, 10.0, 4.0, 0.5)
            st.plotly_chart(viz.terrain_3d(tm, routes, st.session_state.selected_route, ex),
                            width="stretch")
        st.caption("How routes are built: each strategy blends its own weight profile with your priorities, "
                   "then A* (Routes A, B, D) or Dijkstra + greedy orienteering (Route C) minimises "
                   "length × (distance + safety·hazard + science·(1−value)) on a ~237 m grid of the CTX DTM. "
                   "Durations are indicative only and assume a pace stated in Traverse assumptions.")

# ---------------------------------------------------------------------------
# TAB 3 - Science discovery
# ---------------------------------------------------------------------------
if _ws == "Science Discovery":
    st.markdown("<div class='panel-title'>Discovery catalogue</div><div class='sd-sub'>Catalogued geological "
                "features of Jezero from published results (plus two MARSCOPE identifications, marked "
                "<b>Unverified</b>). <i>Evidence</i> states what kind of evidence the catalogue cites: rover in situ, "
                "orbital remote sensing, or neither. It is not a statistical confidence. Select a row to analyse "
                "it below.</div>", unsafe_allow_html=True)
    _disc = wsp.discovery_table(poi_list)
    _dsel = st.dataframe(_disc, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                         key="disc_tbl", height=300,
                         column_config={"id": None,
                                        "Priority": st.column_config.ProgressColumn(
                                            "Planning priority", min_value=0, max_value=1, format="%.2f",
                                            help="MARSCOPE expert-judgement weight, not a NASA ranking")})
    _rows = (_dsel.selection.rows if _dsel and _dsel.selection else [])
    if _rows:
        _pid = _disc.iloc[_rows[0]]["id"]
        if _pid != st.session_state.get("_disc_last"):
            st.session_state._disc_last = _pid
            st.session_state.science_select = _pid
            mlog.log("Discovery opened", poi_names[_pid])
    _foc = st.session_state.get("focus")
    if _foc and _foc.get("poi") and st.session_state.get("_sd_synced") != _foc:
        st.session_state.science_select = _foc["poi"]
    elif _foc and not _foc.get("poi") and st.session_state.get("_sd_synced") != _foc:
        st.session_state.inspect = (_foc["lat"], _foc["lon"])
        st.session_state.science_select = "__inspect__"
    targets = (["__inspect__"] if st.session_state.inspect else []) + [p["id"] for p in poi_list]
    if st.session_state.get("science_select") not in targets:
        st.session_state.science_select = (
            selected.metrics["visited"][-1][0] if selected and selected.found and selected.metrics["visited"]
            else targets[0])
    tgt = st.selectbox("Location to analyse", targets, key="science_select",
                       format_func=lambda t: "📍 Inspected map location" if t == "__inspect__" else poi_names[t])
    if tgt == "__inspect__":
        lat, lon = st.session_state.inspect
        poi = None
    else:
        poi = sd.poi_by_id(tgt)
        lat, lon = poi["lat"], poi["lon"]
    info = sd.inspect_location(ds, tm, sm, lat, lon)
    st.session_state.focus = {"lat": lat, "lon": lon, "label": poi["name"] if poi else "Inspected point",
                              "poi": poi["id"] if poi else None}
    st.session_state._sd_synced = st.session_state.focus

    head_l, head_r = st.columns([2.2, 1])
    with head_l:
        title = poi["name"] if poi else "Inspected map location"
        origin = (badge("literature") if poi and poi["origin"] == "published"
                  else badge("derived"))
        st.markdown(f"<div class='sci-title'>{title}</div><div class='sci-sub'>{origin} "
                    f"{(poi['category'] + ' · ') if poi else ''}{lat:.4f}°N, {lon:.4f}°E"
                    f"{' · position approximate (±0.5–1 km)' if poi else ''}</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Why this location could be scientifically interesting</div>"
                    + "".join(f"<div class='why'>• {w}</div>" for w in info["why"]), unsafe_allow_html=True)
    with head_r:
        st.metric("MARSCOPE science value", f"{info['science_value']:.2f}",
                  help="Derived: 0.65·POI proximity + 0.20·terrain exposure + 0.15·THEMIS night-IR (normalised)")
        if poi:
            st.metric("Planning priority", f"{poi['priority']:.2f}", help="MARSCOPE expert-judgement weight, not a NASA ranking")

    im_col, ev_col = st.columns([1.05, 1.25], gap="medium")
    with im_col:
        _img, _src = wsp.tile_crop(lat, lon, 1.5)
        if _img is None:
            _img, _src = wsp.ctx_crop(ds, lat, lon, 3.0), "MRO CTX (NASA/JPL-Caltech/MSSS), via USGS ARD"
        if _img is not None:
            st.image(_img, width=min(560, max(360, _img.shape[1])), clamp=True)
            st.caption(f"Supporting imagery: {_src}; real orbital image centred on this location, north up "
                       f"({'3 × 3 km' if _img.shape[0] > 200 else '6 × 6 km'}). Position of catalogued targets ±0.5–1 km.")
        else:
            st.caption("No orbital image coverage here.")
    with ev_col:
        if poi:
            _conf, _why = wsp.evidence_basis(poi)
            _grp = wsp.GEO_OF_POI.get(poi["id"], "Mission landmark")
            _sig = poi["interpretations"][0]["text"] if poi["interpretations"] else (info["why"][0] if info["why"] else "")
            st.markdown(f"<div class='card'><div class='card-h'>Evidence and significance</div>"
                        f"<div class='obs'><b>Evidence basis:</b> {_conf}. {_why}</div>"
                        f"<div class='obs'><b>Geological setting:</b> {_grp} · {poi['category']}</div>"
                        f"<div class='obs'><b>Why it matters (interpretation, not a confirmed finding):</b> {_sig}</div>"
                        f"<div class='obs'><b>Water-related evidence:</b> {poi.get('water_evidence') or 'none catalogued'}"
                        f"</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'><div class='card-h'>Evidence and significance</div><div class='obs'>"
                        "An uncatalogued point: everything below is derived from the DTM, THEMIS and the "
                        "catalogued features within 3 km. Nothing here is a published detection.</div></div>",
                        unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        a1.button("Show in 3D terrain", key="sd_3d", width="stretch",
                  on_click=lambda: (wsp.set_focus(lat, lon, poi["name"] if poi else "Inspected point",
                                                  poi["id"] if poi else None, fly=True),
                                    wsp.open_section("3D Terrain")))
        a2.button("Ask the Assistant", key="sd_ask", width="stretch",
                  on_click=lambda: (st.session_state.update(assistant_pending=(
                      f"What is scientifically interesting about the selected location "
                      f"({poi['name'] if poi else f'{lat:.4f} N, {lon:.4f} E'})?")), wsp.open_section("Assistant")))
        a3.button("Plan a route here", key="sd_plan", width="stretch",
                  on_click=lambda: flow_ui.plan_to(FLOW, lat, lon, poi["name"] if poi else "Map point"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='card'><div class='card-h'>{badge('observation')} Observations "
                    f"(reported measurements)</div>" + ("".join(
                        f"<div class='obs'>{o['text']} <span class='ref'>[{o['source']}]</span>"
                        f"<div class='near'>{o['poi']}</div></div>" for o in info["observations"])
                        or "<div class='obs'>No published observations catalogued within 3 km.</div>")
                    + "</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><div class='card-h'>{badge('literature')} Scientific interpretations</div>"
                    + ("".join(f"<div class='obs'>{i['text']} <span class='ref'>[{i['source']}]</span></div>"
                               for i in info["interpretations"])
                       or "<div class='obs'>No published interpretations catalogued nearby.</div>")
                    + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='card-h'>💧 Evidence potentially associated with ancient water</div>"
                    + ("".join(f"<div class='obs'>{w['text']}<div class='near'>{w['poi']}</div></div>"
                               for w in info["water_evidence"])
                       or "<div class='obs'>No water-related evidence catalogued within 3 km. MARSCOPE does not "
                          "infer water activity from topography alone.</div>")
                    + "</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><div class='card-h'>{badge('derived')} Elevation &amp; terrain "
                    f"(from CTX DTM / THEMIS)</div>" + "".join(f"<div class='obs'>{d}</div>" for d in info["derived"])
                    + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='card-h'>🧪 Mineralogical observations (published only)</div>"
                    + ("".join(f"<div class='obs'>{m['text']}<div class='near'>{m['poi']}</div></div>"
                               for m in info["minerals"])
                       or "<div class='obs'>No published mineral detections catalogued here. MARSCOPE does not "
                          "create mineral detections; CRISM spectral products would be needed.</div>")
                    + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='card-h'>🎯 Suggested investigation objectives</div>"
                    + "".join(f"<div class='obs'>▸ {o}</div>" for o in info["objectives"]) + "</div>",
                    unsafe_allow_html=True)

    st.plotly_chart(viz.location_context_chart(info, tm), width="stretch")

    # ---- Perseverance at this location: actual visits + raw images ---------------
    st.markdown("<div class='panel-title'>📷 Perseverance at this location (actual rover data)</div>",
                unsafe_allow_html=True)
    if not mission["waypoints"]:
        st.markdown(f"<div class='obs'>Rover traverse data is {mission['status']['waypoints']} "
                    "(it is fetched from NASA/JPL when online and then cached). No visits are shown rather "
                    "than guessed.</div>", unsafe_allow_html=True)
    else:
        visits = md.visits_near(lat, lon, mission["waypoints"], radius_m=config.ASSUMPTIONS["poi_visit_radius_m"])
        if not visits:
            near = md.visits_near(lat, lon, mission["waypoints"], radius_m=1e9)
            closest = min(near, key=lambda w: w["distance_m"]) if near else None
            st.markdown(f"<div class='obs'>{badge('observation')} Perseverance has not stopped within "
                        f"{config.ASSUMPTIONS['poi_visit_radius_m']:.0f} m of this location (traverse data up to sol "
                        f"{mission['waypoints'][-1]['sol']})."
                        + (f" Closest stop: sol {closest['sol']}, {closest['distance_m']/1000:.1f} km away."
                           if closest else "") + "</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='obs'>{badge('observation')} Perseverance stopped within "
                        f"{config.ASSUMPTIONS['poi_visit_radius_m']:.0f} m of here on {len(visits)} sol(s): "
                        + ", ".join(str(v["sol"]) for v in visits[:30])
                        + (" …" if len(visits) > 30 else "") + "</div>", unsafe_allow_html=True)
            sol = st.selectbox("Show raw images from sol", [v["sol"] for v in visits],
                               index=len(visits) // 2 if len(visits) > 2 else 0, key=f"sol_pick_{tgt}",
                               format_func=lambda s_: next(f"Sol {v['sol']} · {v['distance_m']:.0f} m away"
                                                           + (" · panorama" if v.get("panorama_url") else "")
                                                           for v in visits if v["sol"] == s_))
            v = next(v for v in visits if v["sol"] == sol)
            if v.get("panorama_url"):
                st.markdown(f"[Open the Navcam panorama for sol {sol}]({v['panorama_url']}) (NASA/JPL-Caltech, large image)")
            imgs, ist = load_raw_images(int(sol))
            if imgs:
                cols = st.columns(4)
                for i, im in enumerate(imgs):
                    with cols[i % 4]:
                        st.markdown(f"<a href='{im['link'] or im['large']}' target='_blank'>"
                                    f"<img src='{im['thumb']}' style='width:100%;border-radius:6px' loading='lazy'></a>"
                                    f"<div class='near'>{im['camera']} · {im['date'][:10]}</div>",
                                    unsafe_allow_html=True)
                st.caption(f"Raw, unprocessed images from the NASA Mars 2020 raw-image feed ({ist}). "
                           f"Credit: {imgs[0]['credit']}. Images load from mars.nasa.gov. "
                           f"[Browse all raw images]({md.RAW_IMAGES_PAGE})")
            else:
                st.caption(f"Raw images for sol {sol} are {ist}. "
                           f"[Browse them on NASA's raw-image site]({md.RAW_IMAGES_PAGE}).")

    st.markdown("<div class='panel-title'>🤖 AI explanation</div>", unsafe_allow_html=True)
    if sd.ai_available():
        cache_key = f"{lat:.4f},{lon:.4f}"
        if st.button("Explain this location in plain language", type="primary"):
            with st.spinner("Asking Claude (using only the facts above)..."):
                try:
                    st.session_state.ai_cache[cache_key] = sd.ai_explain(info)
                except RuntimeError as exc:
                    st.error(f"AI explanation unavailable: {exc}")
        if cache_key in st.session_state.ai_cache:
            st.markdown(f"<div class='ai-box'>{badge('derived')} <i>AI-generated from the fact sheet above; "
                        f"check citations.</i></div>", unsafe_allow_html=True)
            st.markdown(st.session_state.ai_cache[cache_key])
    else:
        st.markdown("<div class='ai-box'>AI explanations are optional. Set the <code>ANTHROPIC_API_KEY</code> "
                    "environment variable (and <code>pip install anthropic</code>) to enable plain-language "
                    "explanations generated strictly from the fact sheet below. Everything else works without it."
                    "</div>", unsafe_allow_html=True)
    with st.expander("Fact sheet sent to the AI (and references)"):
        st.json(sd.build_fact_sheet(info))

# ---------------------------------------------------------------------------
# TAB 4 - Data & methods
# ---------------------------------------------------------------------------
if _ws == "Data & Methods":
    a, b = st.columns([1.3, 1])
    with a:
        st.markdown("<div class='panel-title'>Data provenance</div>", unsafe_allow_html=True)
        for key, p in ds.provenance.items():
            urls = "".join(f"<div class='url'>{u}</div>" for u in (p.get("urls") or [])[:4])
            items = (f"<div class='near'>Products: {', '.join(p['items'])}</div>" if p.get("items") else "")
            st.markdown(f"<div class='card'><div class='card-h'>{badge(p.get('kind', 'derived'))} {p.get('title', key)}</div>"
                        f"<div class='obs'>{p.get('source', '')}</div>"
                        f"<div class='near'>{p.get('native_resolution', '')} · fetched {p.get('fetched_at', 'n/a')}</div>"
                        f"<div class='obs'>{p.get('notes', '')}</div>{items}{urls}</div>", unsafe_allow_html=True)
        st.markdown(f"""<div class='card'><div class='card-h'>{badge('literature')} Points of interest & mineral zones</div>
        <div class='obs'>{len(poi_list)} locations compiled from peer-reviewed literature and NASA mission reports
        (see references). Positions are approximate. Mineral zones are drawn around published detections; they are
        not CRISM pixel maps.</div></div>
        <div class='card'><div class='card-h'>{badge('derived')} Slope, roughness, difficulty, terrain classes,
        science value, routes</div><div class='obs'>Computed by MARSCOPE from the observations above. Formulas below.</div></div>
        <div class='card'><div class='card-h'>{badge('observation')} High-resolution map tiles (bundled)</div>
        <div class='obs'>{_tiles_note()}</div></div>
        <div class='card'><div class='card-h'>{badge('observation')} Perseverance traverse &amp; waypoints</div>
        <div class='obs'>NASA/JPL-Caltech MMGIS service behind "Where is Perseverance?" ({mission['status']['traverse']}).
        Raw images per sol: NASA Mars 2020 raw-image feed. Both fetched when online, cached in data/cache/.</div>
        <div class='url'>{md.TRAVERSE_URL}</div><div class='url'>{md.RAW_IMAGES_PAGE}</div></div>
        <div class='card'><div class='card-h'>{badge('observation')} USGS global geologic map of Mars</div>
        <div class='obs'>Tanaka et al. (2014), USGS SIM 3292, 1:20M, via the USGS Astrogeology ArcGIS service
        ({mission['status']['geology']}). Coarse at Jezero scale: the crater floor and delta fall in a few units.</div>
        <div class='url'>https://astrogeology-usgs.hub.arcgis.com/pages/interactive-maps-mars</div></div>
        <div class='card'><div class='card-h'>Browser-loaded tiles</div><div class='obs'>NASA Solar System Treks WMTS
        (MOLA colour shaded relief, Viking MDIM 2.1) are fetched directly by your browser from trek.nasa.gov and
        are optional base maps.</div></div>""", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='panel-title'>Validation checks</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(ds.validation or dm.validate_dataset(ds), columns=["Check", "Passed", "Detail"]),
                     hide_index=True, width="stretch")
        st.markdown("<div class='panel-title'>Live service status</div>", unsafe_allow_html=True)
        if st.button("Check NASA / USGS endpoints"):
            with st.spinner("Probing services..."):
                st.dataframe(pd.DataFrame(dm.check_endpoints()), hide_index=True, width="stretch")
        if st.button("Refresh data from USGS (≈30–60 MB)", help="Re-downloads the CTX DTMs, orthoimages and "
                                                             "THEMIS mosaics and stores them in data/cache/"):
            st.session_state.refresh_token += 1
            st.session_state.route_sig = None
            st.rerun()

    st.markdown("<div class='panel-title'>Methods</div>", unsafe_allow_html=True)
    st.markdown(f"""
* **Slope** – central differences of the CTX DTM mosaic (~59 m/px) with metric spacing on the IAU Mars sphere (R = 3 396.19 km).
* **Roughness** – standard deviation of elevation minus a Gaussian-smoothed surface in a 5×5 window.
* **Terrain difficulty** = 0.7 · min(slope / max slope, 1)^1.5 + 0.3 · min(roughness / 15 m, 1).
* **Terrain classes** – thresholds on slope, local relief vs a 3 km mean, and regional elevation percentile. A physiographic aid, **not** a geologic map.
* **Science value** = 0.65 · POI proximity (priority · exp(−(d/1.5 km)²)) + 0.20 · terrain exposure + 0.15 · THEMIS night-IR brightness, rescaled to 0–1.
* **Route cost** per edge = length · (w_d + {ro.K_SAFETY:g}·w_s·hazard + {ro.K_SCIENCE:g}·w_c·(1 − science)); cells whose 90th-percentile slope exceeds {config.ASSUMPTIONS['max_traversable_slope_deg']:.0f}° are impassable.
* **Routes A, B, D** – A* on an 8-connected ~237 m grid with heuristic w_d · straight-line distance (admissible). **Route C** – multi-source Dijkstra (SciPy) between start, destination and POIs, then greedy insertion of POIs by priority per added cost within ×{config.ASSUMPTIONS['science_detour_budget']} of the direct cost.
* **Mission Planner score** (0–100, recommends the route) = w_sci · science score + w_safe · safety index + w_dist · (100 · shortest eligible distance / route distance). Weights come from Mission priorities (default 50 / 30 / 20). Safety index = 100 · (1 − (0.5 · mean difficulty + 0.3 · p90 difficulty + 0.2 · min(% distance above the caution slope / 20 %, 1))): a relative hazard measure, never a safety guarantee. Routes longer than the optional maximum length are excluded; a missing criterion is reported and the remaining weights are renormalised.
* **Astronaut Assistant** – the language model only explains these deterministic results and the catalogued science, using a JSON mission context built from the same data. Without an AI key, answers come straight from that data.
* **Science-value score** (0–100) = 40 · mean science value along the route + 60 · (Σ priority of POIs within the visit radius / Σ priority of all POIs in the study area).
""")
    st.markdown("<div class='panel-title'>How the data become the 3D colour map, animations and charts</div>",
                unsafe_allow_html=True)
    st.markdown(f"""
* **Mesh** – the CTX DTM mosaic (~59 m) is sampled every {wsp.terrain3d.MESH_STEP}th cell (~300 m) into a {wsp.terrain3d.MESH_STEP}× coarser grid of vertices in kilometres (x east, z south, y = (elevation − median)/1000 × vertical exaggeration). The browser draws it with three.js (bundled, MIT).
* **Colour modes**, each the same grid sampled from one layer: *Elevation* (observation, 2nd–98th percentile stretch), *Slope* (derived; breaks at {config.ASSUMPTIONS['caution_slope_deg']:.0f}° caution and {config.ASSUMPTIONS['max_traversable_slope_deg']:.0f}° limit), *Terrain class* (derived physiography), *Geological features* (each vertex takes the setting of the nearest catalogued, published feature within 1.5 km, 0.7 km for craters; a literature overlay, **not** a geologic map), *Scientific interest* (science value × 100) and *CTX image* (orthoimage texture).
* **Selection** – a click is ray-cast onto the mesh and converted back to latitude/longitude; the readout uses the nearest vertex, and the right-hand panel recomputes everything at full resolution with the same functions as Science Discovery.
* **Cross-section** – {240} points along a great-circle approximation through the selected point (azimuth and length chosen by the user), bilinearly sampled from the full-resolution DTM, slope and science layers; catalogued features within 800 m of the line are marked. There are no subsurface data, so no layers below the surface are drawn.
* **Animations** – *Reveal relief* scales the vertical axis from 0 to the chosen exaggeration; *Tour layers* and mode changes interpolate vertex colours between two layers; *Compare/Sweep* colours vertices west of a moving split with one layer and east of it with another; *Scan* moves along the section in the chart and in 3D together; *rover playback* interpolates along the planned route's profile by distance, 1 s = 1 simulated sol at {config.SIM_PACE_M_PER_SOL:.0f} m/sol (adjustable 0.5–8×), and marks a catalogued feature as reached when the route passes within {config.ASSUMPTIONS['poi_visit_radius_m']:.0f} m. All playback is **SIMULATED** and labelled so. The real Perseverance traverse (NASA/JPL MMGIS) is drawn only when it has been downloaded.
* **Distribution charts** – areas are cell counts × cell area on the Mars sphere; route terrain shares weight each profile segment by its length.
* **Evidence labels in the discovery catalogue** come from the catalogue's own citations: *Observed in situ* when a cited observation is from the rover, *Orbital detection* when only orbital data are cited, *Unverified* for MARSCOPE's own identifications. They describe the type of evidence, not a statistical confidence.
""")
    st.markdown("<div class='panel-title'>Limitations</div>", unsafe_allow_html=True)
    st.markdown("""
* The planning grid (~237 m) and DTM sampling (~59 m) cannot resolve rover-scale hazards (rocks, small scarps, sand ripples). Real traverse planning uses HiRISE (~1 m) DTMs and onboard hazard detection.
* THEMIS layers are 8-bit stretched mosaics: useful for relative comparisons, not calibrated temperature or thermal inertia.
* MARSCOPE does not process CRISM spectra; mineral information is limited to published detections.
* Points of interest and priorities are curated and approximate; the two unnamed craters are MARSCOPE identifications.
* Durations are indicative ranges from an assumed pace and ignore science stops, communication windows and power.
""")
    st.markdown("<div class='panel-title'>References</div>", unsafe_allow_html=True)
    for key, ref in sd.load_poi_catalog()["references"].items():
        st.markdown(f"<div class='refline'><b>[{key}]</b> {ref}</div>", unsafe_allow_html=True)
    st.markdown("<div class='refline'><b>[USGS ARD]</b> USGS Astrogeology Science Center, Analysis Ready Data "
                "(CTX DTMs/orthoimages, THEMIS controlled mosaics), s3://astrogeo-ard, CC0.</div>"
                "<div class='refline'><b>[Tanaka2014]</b> Tanaka, K.L. et al. (2014) Geologic map of Mars. "
                "USGS Scientific Investigations Map 3292.</div>"
                "<div class='refline'><b>[MMGIS]</b> NASA/JPL-Caltech, Mars 2020 'Where is Perseverance?' map "
                "(MMGIS), https://mars.nasa.gov/maps/location/?mission=M20</div>"
                "<div class='refline'><b>[M20 raw]</b> NASA/JPL-Caltech, Mars 2020 raw images, "
                "https://mars.nasa.gov/mars2020/multimedia/raw-images/</div>",
                unsafe_allow_html=True)
