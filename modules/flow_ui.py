"""Streamlit screens of the mission navigation flow (presentation and state only).

Screens: setup -> routes -> navigate. Mission Control and every analysis tab stay in the
"Mission Control & all data" view (app.py). Route planning, evaluation and terrain all come from
the existing modules; positions during navigation are SIMULATED.
"""
from __future__ import annotations

import html
import time
from dataclasses import dataclass

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

import config
from modules import dock, flow, guide, landing, nav_map, rover_sim
from modules import mission_center as mc
from modules import mission_control as mcc
from modules import mission_log as mlog
from modules import mission_planner as mp
from modules import route_optimizer as ro
from modules import science_discovery as sd
from modules.data_manager import haversine_m

LAYER_OPTIONS = ["Imagery", "Elevation", "Slope", "Science targets", "Hazards", "Route"]
LAYER_KEYS = {"Imagery": "imagery", "Elevation": "elevation", "Slope": "slope", "Science targets": "targets",
              "Hazards": "hazards", "Route": "route"}


@dataclass
class App:
    """Everything the flow needs from app.py (built once per run there)."""
    ds: object
    tm: object
    sm: object
    planner: object
    layers: dict
    pois: list
    poi_by_id: dict
    route_signature: object      # callable(start, dest, status) -> tuple
    set_later: object            # callable(**state)
    poi_point: object            # callable(poi_id) -> (lat, lon, name)
    badge: object = None         # callable(kind) -> html (data-kind badge, used by the Assistant)
    mission_waypoints: object = None   # callable() -> real Perseverance waypoints (optional)


def _e(x) -> str:
    return html.escape(str(x))


def short(name: str) -> str:
    return name.split(" (")[0]


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def init(app: App) -> None:
    ss = st.session_state
    for k, v in {"click_target": "Destination", "nav_mode": "2D", "nav_playing": False, "nav_detail": None,
                 "nav_layers": ["Imagery", "Science targets", "Route"], "guide_step": None,
                 "flow_err": {}, "_3d_intro": True, "nav_last_obj": None, "setup_last_click": None,
                 "flow_w_science": ss.get("w_science", .5), "flow_w_safety": ss.get("w_safety", .3),
                 "flow_w_distance": ss.get("w_distance", .2), "click_mode_flow": "Inspect",
                 "flow_inspect": None, "nav_panel": True,
                 "dock_n": "Assistant", "dock_r": "Routes"}.items():
        ss.setdefault(k, v)
    if not ss.get("_flow_started"):
        ss._flow_started = True
        if ss.get("start") is None:
            ss.start = app.poi_point(flow.START_DEFAULT_POI)
            ss.start_select = flow.START_DEFAULT_POI


def current_sig(app: App):
    ss = st.session_state
    if not (ss.get("start") and ss.get("destination")):
        return None
    return app.route_signature(ss.start, ss.destination, app.ds.status)


def have_routes(app: App) -> bool:
    ss = st.session_state
    return flow.routes_valid(ss.get("routes"), ss.get("route_sig"), current_sig(app))


def set_endpoint(app: App, kind: str, pt, source: str) -> None:
    """Change the start or destination. Any change clears the routes (they no longer match)."""
    ss = st.session_state
    if pt == ss.get(kind):
        return
    ss[kind] = pt
    ss.flow_inspect = None
    poi = next((p for p in app.pois if pt and p["name"] == pt[2]), None)
    sel = poi["id"] if poi else ("__map__" if pt else "—")
    if kind == "start":
        ss.start_select = sel if poi else "—"
    else:
        ss.dest_select = sel if poi else "—"
        if poi:
            ss.science_select = poi["id"]
    if ss.get("routes"):
        mlog.log("Routes cleared", f"{'start' if kind == 'start' else 'destination'} changed")
    ss.routes, ss.route_sig = None, None
    ss.sim_progress_pct, ss.nav_playing, ss.nav_detail, ss.sim_route = 0, False, None, None
    mlog.log("Start selected" if kind == "start" else "Destination selected",
             f"{flow.endpoint_label(pt)} ({source})")


def _swap(app: App) -> None:
    ss = st.session_state
    s, d = ss.start, ss.destination
    ss.start, ss.destination = d, s
    ss.start_select, ss.dest_select = ss.get("dest_select", "—"), ss.get("start_select", "—")
    ss.routes, ss.route_sig, ss.sim_progress_pct, ss.sim_route = None, None, 0, None
    mlog.log("Endpoints swapped", f"{flow.endpoint_label(d)} → {flow.endpoint_label(s)}")


def calculate(app: App) -> bool:
    """Run the existing planner for the current endpoints. Returns True if any route was found."""
    ss = st.session_state
    start, dest = ss.start, ss.destination
    t0 = time.time()
    user_w = ro.Weights(ss.w_distance, ss.w_safety, ss.w_science)
    routes = ro.plan_all(app.planner, start[:2], dest[:2], user_w, ss.strength, include_custom=True)
    elapsed = time.time() - t0
    ranking = ro.rank_routes(routes, user_w, ss.max_route_km or None)
    if not ranking:
        ss.flow_err["calc"] = (routes[0].message if routes and routes[0].message else
                               "No traversable route was found under the current slope limit. "
                               "Try another destination.")
        return False
    ss.flow_err.pop("calc", None)
    snapped = app.planner.snap(dest[0], dest[1])
    s_lat, s_lon = app.planner.grid.rowcol_to_lonlat(*snapped)
    off = float(haversine_m(dest[0], dest[1], s_lat, s_lon))
    ss.flow_snap_note = (f"The destination is on terrain steeper than {ss.max_slope:.0f}°, so routes end at the "
                         f"nearest traversable cell, {off:.0f} m away." if off > 400 else "")
    ss.update(routes=routes, route_time=elapsed, route_sig=app.route_signature(start, dest, app.ds.status),
              selected_route=ranking[0][0].key, sim_progress_pct=0, nav_playing=False, nav_detail=None,
              sim_route=None)
    mlog.log("Route generated", f"{short(dest[2])}: {len(routes)} candidates in {elapsed:.2f} s; "
                                f"recommended Route {ranking[0][0].key}")
    return True


def evaluation_for(app: App):
    ss = st.session_state
    return mp.evaluate_routes(ss.routes, mp.ScoringModel(ss.w_science, ss.w_safety, ss.w_distance),
                              mp.MissionConstraints(max_route_km=ss.max_route_km or None,
                                                    max_slope_deg=float(ss.max_slope)))


def go(view: str) -> None:
    ss = st.session_state
    ss.view = view
    if view != "navigate":
        ss.nav_playing = False
        if ss.get("sim_status") == mc.PLAYING and ss.get("sim_total"):     # leaving pauses the simulation
            d = mc.clock_d(ss.sim_d0, ss.sim_t0, ss.get("_sim_speed_eff", 1), time.time(), ss.sim_total)
            ss.sim_d0, ss.sim_t0, ss.sim_status = d, None, mc.PAUSED
            ss.sim_progress_pct = int(round(100 * d / ss.sim_total))


# ---------------------------------------------------------------------------
# Shared chrome: a slim bar on the map screens (nothing on the start screen)
# ---------------------------------------------------------------------------
def _edit_locations() -> None:
    go("setup")


NAV_MAP = "Route map"


def _nav_change(app: App) -> None:
    ss = st.session_state
    v = ss.get("ws_nav")
    if not v:
        return
    if v == NAV_MAP:
        go(ss.get("_map_view", "routes") if have_routes(app) else "setup")
    else:
        ss.ws = v
        go("all")


def open_section(name: str) -> None:
    if name == "Assistant" and open_assistant():
        return
    st.session_state.ws = name
    go("all")


def open_assistant() -> bool:
    """Show the Assistant in the right-hand dock of the map screen. False when no route exists yet (then the
    full Assistant page in the workspace is used, as before)."""
    ss = st.session_state
    if not (ss.get("routes") and ss.get("start") and ss.get("destination") and ss.get("route_sig")):
        return False
    ss.dock_n = ss.dock_r = "Assistant"
    ss.nav_panel = True
    go(ss.get("_map_view", "routes"))
    return True


def plan_to(app: App, lat: float, lon: float, label: str) -> None:
    """Make a point the destination and plan routes with the existing setup -> map transition."""
    ss = st.session_state
    if not ss.get("start"):
        ss.start = app.poi_point(config.DEMO_START_POI)
    poi = next((p for p in app.pois if p["name"] == label), None)
    set_endpoint(app, "destination", (lat, lon, poi["name"] if poi else "Map point"), "workspace")
    ss._leaving = True
    go("setup")


def top_bar(app: App) -> None:
    """Slim bar on every screen but the start screen: brand + trip, section navigation, actions."""
    from modules.workspace import SECTIONS
    ss = st.session_state
    if ss.view in ("routes", "navigate"):
        ss._map_view = ss.view
    sections = [x for x in SECTIONS if x != "Assistant"]       # the Assistant is in the map's right dock
    ws = ss.get("ws", SECTIONS[0])
    ss.ws_nav = NAV_MAP if ss.view in ("routes", "navigate") else (ws if ws in sections else None)
    trip = ""
    if ss.get("start") and ss.get("destination"):
        trip = (f"<i class='s'></i>{_e(short(flow.endpoint_label(ss.start)))}<span>→</span><i class='d'></i>"
                f"{_e(short(flow.endpoint_label(ss.destination)))}")
    with st.container(key="flow_top"):
        c0, c1, c2 = st.columns([1.25, 3.7, 1.75], vertical_alignment="center")
        c0.markdown(f"<div class='fl-brand'><i class='hud-radar' aria-hidden='true'></i>MARSCOPE<small class='fl-trip'>{trip or 'Jezero crater'}</small></div>",
                    unsafe_allow_html=True)
        with c1, st.container(key="ws_nav_box"):
            st.segmented_control("Section", [NAV_MAP] + sections, key="ws_nav", on_change=_nav_change, args=(app,),
                                 label_visibility="collapsed", required=True,
                                 help="Every MARSCOPE module: the route map (with the Assistant in its right "
                                      "panel), Mission Control, the 3D colour map, Science Discovery, "
                                      "Exploration Routes and Data & Methods")
        with c2, st.container(key="flow_actions", horizontal=True, horizontal_alignment="right"):
            if ss.view == "navigate":
                st.button("← Routes", key="flow_to_routes", on_click=go, args=("routes",))
            if ss.view in ("routes", "navigate"):
                st.button("Hide panel" if ss.get("nav_panel", True) else "Show panel", key="panel_btn",
                          on_click=_toggle_panel, help="Fold the right-hand mission panel away to widen the map")
            st.button("Edit locations", key="flow_edit", on_click=_edit_locations)
            label = "Resume guide" if ss.get("guide_step") is None and ss.get("_guide_paused") is not None \
                else "Guide"
            st.button(label, key="guide_open", on_click=open_guide, args=(app,),
                      help="The 10-step Mission Guide")


# ---------------------------------------------------------------------------
# Screen 1: centred start screen (visor template): start, destination, swap
# ---------------------------------------------------------------------------
COORD_HINT = "Choose a site or type 18.47, 77.40"


def _labels(app: App, kind: str) -> dict:
    """Option label -> POI. Landing sites first for the start; science targets by priority otherwise."""
    if kind == "start":
        ids = flow.start_options(app.pois)
    else:
        ids = [p["id"] for p in sorted(app.pois, key=lambda p: -p["priority"])]
    return {app.poi_by_id[i]["name"]: app.poi_by_id[i] for i in ids}


def _pt_label(pt) -> str | None:
    if not pt:
        return None
    return pt[2] if pt[2] != "Map point" else f"{pt[0]:.4f}, {pt[1]:.4f}"


def _apply_field(app: App, kind: str, key: str) -> None:
    ss = st.session_state
    choice = ss.get(key)
    ss.flow_err.pop(kind, None)
    if choice in (None, ""):
        return
    if choice == _pt_label(ss.get(kind)):
        return
    pt, err = landing.resolve_destination(choice, _labels(app, kind), app.ds.fine_grid.contains)
    if err:
        ss.flow_err[kind] = err.replace("Choose a destination first", "Choose a location")
        return
    set_endpoint(app, kind, pt, "coordinates" if pt[2] == "Map point" else "list")


def _field(app: App, kind: str, title: str) -> None:
    ss = st.session_state
    key = f"fld_{kind}"
    labels = list(_labels(app, kind))
    cur = _pt_label(ss.get(kind))
    if cur and cur not in labels:
        labels = [cur] + labels
    ss[key] = cur                       # the field always shows the current location
    with st.container(key=f"{'start' if kind == 'start' else 'dest'}_panel"):
        dot = "s" if kind == "start" else "d"
        st.markdown(f"<div class='vz-label'><i class='{dot}'></i>{title}</div>", unsafe_allow_html=True)
        st.selectbox(title, labels, key=key, index=None, placeholder=COORD_HINT, accept_new_options=True,
                     label_visibility="collapsed", on_change=_apply_field, args=(app, kind, key),
                     help="Pick a landing site or science target, or type latitude, longitude in degrees "
                          "(east longitude), e.g. 18.47, 77.40")
        if ss.flow_err.get(kind):
            st.markdown(f"<div class='vz-err' role='alert'>{_e(ss.flow_err[kind])}</div>", unsafe_allow_html=True)


def _show_route() -> None:
    st.session_state._leaving = True


def setup_screen(app: App) -> None:
    ss = st.session_state
    st.markdown(
        "<div class='visor-scene' aria-hidden='true'><div class='visor-sky'></div><div class='visor-ground'></div>"
        "<div class='visor-haze'></div></div><div class='visor-frame' aria-hidden='true'><i class='c tl'></i>"
        "<i class='c tr'></i><i class='c bl'></i><i class='c br'></i></div>"
        "<div class='visor-credit'>Ground: MRO CTX orthoimage of the Jezero delta (NASA/JPL-Caltech/MSSS, USGS "
        "Astrogeology), shown in perspective.</div>", unsafe_allow_html=True)
    leaving = bool(ss.get("_leaving")) and bool(ss.start and ss.destination)
    with st.container(key="setup_wrap"):
        with st.container(key="setup_panel_leave" if leaving else "setup_panel"):
            st.markdown("<div class='vz-brand'>MARSCOPE</div><div class='vz-tag'>Jezero crater · route planning</div>",
                        unsafe_allow_html=True)
            _field(app, "start", "Start")
            with st.container(key="swap_row", horizontal_alignment="center"):
                st.button("⇅  Swap", key="swap_btn", on_click=_swap, args=(app,),
                          disabled=not (ss.start and ss.destination) or leaving,
                          help="Swap start and destination")
            _field(app, "destination", "Destination")
            with st.container(key="calc_btn"):
                st.button("Planning routes…" if leaving else "Show route", key="calc", on_click=_show_route,
                          width="stretch", disabled=not (ss.start and ss.destination) or leaving)
            stages = st.empty()
            if ss.flow_err.get("calc"):
                st.markdown(f"<div class='vz-err' role='alert'>{_e(ss.flow_err['calc'])}</div>",
                            unsafe_allow_html=True)
    with st.container(key="setup_guide_link", horizontal=True, horizontal_alignment="right"):
        for sec, lbl in (("3D Terrain", "3D terrain"), ("Science Discovery", "Science"),
                         ("Assistant", "Assistant"), ("Data & Methods", "Data & Methods")):
            st.button(lbl, key=f"explore_{lbl}", on_click=open_section, args=(sec,), type="tertiary",
                      help=f"Open {sec} without planning a route first")
        st.button("Mission Guide", key="guide_open_setup", on_click=open_guide, args=(app,), type="tertiary",
                  help="A 10-step walkthrough")
    if leaving:
        t0 = time.time()
        landing.show_stages(stages, [("Analysing terrain", "done"), ("Planning four routes", "active"),
                                     ("Drawing the map", "todo")])
        ok = calculate(app)
        ss._leaving = False
        if ok:
            time.sleep(max(0.0, 0.45 - (time.time() - t0)))   # let the panel finish fading out
            ss._enter_routes = True
            go("routes")
        st.rerun()


def _map_fallback(app: App, exc: Exception) -> None:
    st.warning(f"The interactive map could not be shown ({type(exc).__name__}). Locations can still be edited "
               "(Edit locations). Static CTX overview below.")
    if app.layers.get("ctx"):
        st.image(app.layers["ctx"], caption="MRO CTX orthoimage mosaic of the study area (NASA/JPL-Caltech/MSSS, "
                                            "USGS Astrogeology ARD). North is up.")


# ---------------------------------------------------------------------------
# Screen 2: route preview. Map first (route draws), then the route-choice panel.
# ---------------------------------------------------------------------------
CLICK_MODES = ["Inspect", "Set start", "Set destination"]


def _apply_custom(app: App) -> None:
    ss = st.session_state
    ss.w_science, ss.w_safety, ss.w_distance = ss.flow_w_science, ss.flow_w_safety, ss.flow_w_distance
    mlog.log("Custom weights applied", f"science {ss.w_science:.2f} · safety {ss.w_safety:.2f} · "
                                       f"distance {ss.w_distance:.2f}")
    calculate(app)
    ss.selected_route = "D" if any(r.key == "D" and r.found for r in ss.routes or []) else ss.selected_route


def _select(key: str) -> None:
    st.session_state.selected_route = key
    mlog.log("Route selected", f"Route {key} (preview)")


def _card(app: App, d: dict) -> None:
    ss = st.session_state
    sel = d["key"] == ss.selected_route
    with st.container(key=f"card_{d['key']}"):
        badge = "<span class='fl-rec'>Recommended</span>" if d["recommended"] else ""
        if d["duplicate_of"]:
            body = f"<div class='rc3-x'>Same path as Route {d['duplicate_of']} with these weights.</div>"
        elif d["excluded"]:
            body = f"<div class='rc3-x'>Excluded: {_e(d['excluded'])}</div>"
        else:
            sci = "n/a" if d["science"] is None else f"{d['science']:.0f}"
            stops = ", ".join(d["stops"][:3]) + (f" +{len(d['stops']) - 3}" if len(d["stops"]) > 3 else "") \
                if d["stops"] else "no catalogued stops"
            body = (f"<div class='rc3-m'><span><b>{d['distance_km']:.1f} km</b> est.</span>"
                    f"<span><b class='risk-{d['risk'].lower()}'>{d['risk']}</b> risk</span>"
                    f"<span><b>{sci}</b>/100 science</span></div>"
                    f"<div class='rc3-s'>{_e(stops)}</div><div class='rc3-w'>{_e(d['explanation'])}</div>")
        st.markdown(f"<div class='rc3{' sel' if sel else ''}' style='--c:{d['color']}'><div class='rc3-h'><i></i>{_e(d['title'])}"
                    f"<em>{d['key']}</em>{badge}</div>{body}</div>", unsafe_allow_html=True)
        if d["eligible"] and not sel:
            st.button(f"Preview {d['title']}", key=f"show_{d['key']}", on_click=_select, args=(d["key"],),
                      width="stretch", type="tertiary")
        if d["key"] == "D":
            with st.expander("Adjust weights"):
                st.slider("Science", 0.0, 1.0, step=0.05, key="flow_w_science")
                st.slider("Safety", 0.0, 1.0, step=0.05, key="flow_w_safety")
                st.slider("Distance", 0.0, 1.0, step=0.05, key="flow_w_distance")
                st.button("Recalculate with these weights", key="apply_custom", on_click=_apply_custom,
                          args=(app,), width="stretch")


def _start_nav(key: str) -> None:
    ss = st.session_state
    if ss.get("selected_route") != key or ss.get("view") != "navigate":
        ss.sim_progress_pct, ss.sim_route = 0, None
    ss.selected_route = key
    ss.nav_playing, ss.nav_detail = False, None
    mlog.log("Route selected", f"Route {key} for navigation")
    go("navigate")


def _handle_click(app: App, click: dict | None, last_key: str) -> None:
    """Map clicks on the route preview: inspect, or move the start / destination (routes re-planned)."""
    ss = st.session_state
    if not click or click == ss.get(last_key):
        return
    ss[last_key] = click
    lat, lon = float(click["lat"]), float(click["lng"])
    if not app.ds.fine_grid.contains(lat, lon):
        st.toast("That point is outside the Jezero study area.")
        return
    mode = ss.get("click_mode_flow", "Inspect")
    if mode == "Inspect":
        ss.flow_inspect = (lat, lon)
        _want_tab("dock_r", "Science")
        mlog.log("Location inspected", f"{lat:.4f}° N, {lon:.4f}° E")
        st.rerun()
    kind = "start" if mode == "Set start" else "destination"
    poi = flow.nearest_poi(lat, lon, app.pois)
    set_endpoint(app, kind, (poi["lat"], poi["lon"], poi["name"]) if poi else (lat, lon, "Map point"), "map")
    if ss.start and ss.destination:
        calculate(app)          # never show stale routes: re-plan for the new endpoints
    st.rerun()


def _inspect_card(app: App) -> None:
    ss = st.session_state
    lat, lon = ss.flow_inspect
    info = sd.inspect_location(app.ds, app.tm, app.sm, lat, lon)
    t = info["terrain"]
    near = info["nearby"][0] if info["nearby"] else None
    st.markdown(
        f"<div class='fl-detail'><div class='fd-h'>Inspected point</div>"
        f"<div class='fd-sub'>{lat:.4f}° N · {lon:.4f}° E</div>"
        f"<div class='fd-s'>Derived here {mcc.tag('derived')}</div>"
        f"<p>Elevation {t['elevation_m']:,.0f} m ({'CTX DTM' if info['dtm_measured'] else 'gap-filled'}) · slope "
        f"{t['slope_deg']:.1f}° · {_e(t['terrain_class'].split(' (')[0])} · science value {info['science_value']:.2f}</p>"
        + (f"<div class='fd-s'>Nearest catalogued target {mcc.tag('lit')}</div><p>{_e(short(near[0]))}, "
           f"{near[1] / 1000:.1f} km away</p>" if near else "") + "</div>", unsafe_allow_html=True)
    st.button("Close", key="close_inspect", type="tertiary",
              on_click=lambda: st.session_state.update(flow_inspect=None))


def routes_screen(app: App) -> None:
    ss = st.session_state
    if not have_routes(app):
        st.info("The routes no longer match the start and destination. Edit the locations to plan again.")
        st.button("Edit locations", on_click=go, args=("setup",), key="back_setup2")
        return
    _apply_dock_next()
    ev = evaluation_for(app)
    cards = flow.card_data(ev, app.poi_by_id)
    s, d = ss.start, ss.destination
    sel = next((r for r in ss.routes if r.key == ss.selected_route and r.found), None)
    if sel is None:
        sel = ev.recommended.route if ev.recommended else None
        ss.selected_route = sel.key if sel else ss.selected_route
    entering = ss.pop("_enter_routes", False)
    if ss.get("nav_mode") not in MODE_LABELS:
        ss.nav_mode = "2D"
    if sel is not None:
        _handle_nav3d(app, "routes3d", "routes")
    open_panel = ss.get("nav_panel", True)
    with st.container(key="map_row"):
        st.markdown(f"<div class='dock-{'open' if open_panel else 'closed'}' hidden></div>", unsafe_allow_html=True)
        map_col, panel_col = st.columns([2.55, 1], gap="small")
    with map_col, st.container(key="routes_stage" + ("_enter" if entering else "")):
        mode = ss.nav_mode
        st.markdown(f"<div class='nav-mode-{mode.lower()}' hidden></div>", unsafe_allow_html=True)
        _view_switch()
        if not open_panel:
            _dock_handle()
        focus = app.poi_by_id.get(ss.get("nav_detail"))
        focus_pt = (focus["lat"], focus["lon"]) if focus else ss.get("flow_inspect")
        try:
            m = nav_map.build(app.ds, app.layers, routes=ss.routes, selected=ss.selected_route, start=s,
                              destination=d, stops=flow.notable_stops(sel, app.poi_by_id) if sel else [],
                              pois=app.pois, show={"imagery", "targets", "route"},
                              route_bounds=_focus_bounds(*focus_pt) if focus_pt and mode == "2D"
                              else nav_map.route_fit_bounds(sel, s, d),
                              recommended=ev.recommended.key if ev.recommended else None,
                              inspect=ss.get("flow_inspect"), fill=True)
            out = st_folium(m, height=760, use_container_width=True, returned_objects=["last_clicked"],
                            key="routes_map_f")
        except Exception as exc:  # noqa: BLE001
            _map_fallback(app, exc)
            out = None
        if sel is not None:
            with st.container(key="routes_3d_box"):
                _nav_3d(app, sel, 0.0, active=mode == "3D", key="routes3d", preview=True)
    with panel_col, st.container(key="route_panel" + ("_enter" if entering else "")):
        if open_panel:
            _dock_head("Mission panel", "ROUTE PREVIEW · NO LIVE ROVER LINK")
            if ss.get("dock_r") not in dock.ROUTE_TABS:
                ss.dock_r = "Routes"
            _dock_tabs("dock_r", dock.ROUTE_TABS)
            tab = ss.dock_r
            with st.container(key="dock_body_r_" + tab.lower()):
                if tab == "Routes":
                    _routes_tab(app, ev, cards, sel)
                elif tab == "Assistant":
                    ctx, sig = dock.context(app, ev, screen="route preview", mode=ss.nav_mode,
                                            detail=ss.get("nav_detail"))
                    with st.container(key="dock_chat"):
                        dock.chat(ctx, sig, ev, history_height=380)
                    dock.recommendation(ctx, ev, app.badge or (lambda k: ""), _show_route_key, _back_to_setup)
                elif tab == "Science":
                    _science_preview_tab(app, sel, ev)
                else:
                    st.markdown(_sec("", "System status") + dock.system_rows(app, [
                        ("View", MODE_LABELS[ss.nav_mode], "ok"),
                        ("Routes", f"{sum(r.found for r in ss.routes)} planned on the CTX DTM", "ok"),
                        ("Simulation", "not started · choose Start navigation", "dim")]), unsafe_allow_html=True)
    _handle_click(app, (out or {}).get("last_clicked"), "routes_last_click")


def _show_route_key(key: str) -> None:
    st.session_state.selected_route = key


def _back_to_setup() -> None:
    go("setup")


def _routes_tab(app: App, ev, cards, sel) -> None:
    ss = st.session_state
    rec = ev.recommended
    st.markdown(f"<div class='rp-sub'>Planned on the CTX elevation model · straight line "
                f"{ss.routes[0].metrics.get('straight_line_km', 0):.1f} km</div>", unsafe_allow_html=True)
    if rec is not None:
        st.markdown(f"<div class='rp-why'><b>{_e(flow.CARD_TITLES.get(rec.key))}</b> is recommended. "
                    f"{_e(_why_short(ev))}</div>", unsafe_allow_html=True)
    if ss.get("flow_snap_note"):
        st.caption(ss.flow_snap_note)
    with st.container(key="route_cards"):
        for c in cards:
            _card(app, c)
    if sel is not None:
        with st.container(key="nav_go"):
            st.button(f"Start navigation · {flow.CARD_TITLES.get(sel.key)}", key="nav_start", type="primary",
                      on_click=_start_nav, args=(sel.key,), width="stretch")
    st.markdown("<div class='fl-foot'>Distance and sols are estimates (planning grid ~237 m; assumed pace "
                f"{config.ASSUMPTIONS['indicative_speed_m_per_sol'][0]:.0f}–"
                f"{config.ASSUMPTIONS['indicative_speed_m_per_sol'][1]:.0f} m/sol). Risk is a relative terrain "
                f"index, never a guarantee of safety. {mcc.tag('derived')}</div>", unsafe_allow_html=True)
    with st.expander("Map tools"):
        with st.container(key="click_mode"):
            st.segmented_control("A map click will", CLICK_MODES, key="click_mode_flow", required=True,
                                 help="Inspect a point, or move the start or destination (routes are "
                                      "re-planned at once).")
        st.markdown(nav_map.legend_html({"imagery"}, ("start", "dest", "stop", "target")) +
                    "<div class='fl-cap'>Imagery: MRO CTX (~20 m/px) everywhere, HiRISE (~1.2 m/px) over "
                    "the landing site, delta front and Belva crater, via USGS Astrogeology.</div>",
                    unsafe_allow_html=True)


def _science_preview_tab(app: App, sel, ev) -> None:
    """Science on the route-preview screen: the inspected point, the selected target, the destination."""
    ss = st.session_state
    if ss.get("flow_inspect"):
        _inspect_card(app)
    pid = ss.get("nav_detail") or (flow.nearest_poi(ss.destination[0], ss.destination[1], app.pois, radius_m=400)
                                   or {}).get("id")
    if pid in app.poi_by_id and sel is not None:
        st.markdown(detail_card_html(app, app.poi_by_id[pid], sel, flow.notable_stops(sel, app.poi_by_id)),
                    unsafe_allow_html=True)
        with st.container(horizontal=True, key="sci_actions"):
            st.button("Ask the Assistant", key="sci_ask", icon=":material/forum:",
                      on_click=_ask_about, args=(app.poi_by_id[pid]["name"],))
            if ss.get("nav_detail"):
                st.button("Close details", key="sci_close", type="tertiary", on_click=_select_wp, args=(None,))
    elif not ss.get("flow_inspect"):
        st.markdown("<div class='ai-empty'>Click a target on the 2D map (Map tools → Inspect) or a pin in the 3D "
                    "view to read its published observations here.</div>", unsafe_allow_html=True)


def _ask_about(name: str) -> None:
    ss = st.session_state
    ss.assistant_pending = f"What is scientifically interesting about {name}, and what should I look at there?"
    ss.dock_n = ss.dock_r = "Assistant"


def _focus_bounds(lat: float, lon: float, half_km: float = 1.6):
    """2D map bounds around a selected point, so switching 3D -> 2D keeps the same focus."""
    dlat = half_km / 59.0
    dlon = dlat / max(0.2, np.cos(np.radians(lat)))
    return [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]


def _view_switch() -> None:
    ss = st.session_state
    ss.nav_mode = ss.nav_mode          # re-assert: older Streamlit shows no selection on first mount otherwise
    with st.container(key="view_switch"):
        st.segmented_control("Map view", list(MODE_LABELS), key="nav_mode", format_func=MODE_LABELS.get,
                             on_change=_set_mode, required=True, label_visibility="collapsed",
                             help="Switch between the 2D map and the 3D terrain. The route, waypoints, "
                                  "selection and the simulated rover stay in sync.")


def _dock_handle() -> None:
    with st.container(key="dock_show"):
        st.button("Panel", key="dock_show_btn", on_click=_toggle_panel, icon=":material/left_panel_open:",
                  help="Show the mission panel")


def _dock_head(title: str, sub: str) -> None:
    with st.container(key="dock_head", horizontal=True, vertical_alignment="center"):
        st.markdown(f"<div class='mx-top' role='heading' aria-level='2'>{_e(title)}<em>{_e(sub)}</em></div>",
                    unsafe_allow_html=True)
        st.button("", key="dock_hide_btn", on_click=_toggle_panel, icon=":material/right_panel_close:",
                  type="tertiary", help="Hide the mission panel (widens the map)")


def _dock_tabs(key: str, tabs: list[str]) -> None:
    st.markdown(dock.tab_icon_css(), unsafe_allow_html=True)
    st.session_state[key] = st.session_state[key]      # see _view_switch
    with st.container(key="dock_tabs"):
        st.segmented_control("Mission panel section", tabs, key=key, format_func=dock.tab_label, required=True,
                             label_visibility="collapsed")


def _why_short(ev) -> str:
    """One plain sentence from the deterministic evaluation: what the recommendation wins and gives up."""
    rec, ranked = ev.recommended, ev.ranked
    w = ev.model.as_dict()
    if len(ranked) < 2:
        return f"It is the only route that meets the mission constraints (score {rec.score:.0f}/100)."
    nxt = ranked[1]
    gains, losses = [], []
    for k, label in (("science", "science value"), ("safety", "terrain safety"), ("distance", "distance")):
        a, b = rec.criteria.get(k), nxt.criteria.get(k)
        if a is None or b is None or abs(a - b) < 1:
            continue
        (gains if a > b else losses).append(label)
    txt = (f"Its overall score is {rec.score:.0f}/100 against {nxt.score:.0f} for "
           f"{flow.CARD_TITLES.get(nxt.key, nxt.name)}")
    if gains:
        txt += f", winning on {' and '.join(gains)}"
    if losses:
        txt += f" while giving up some {' and '.join(losses)}"
    return txt + f" (weights: science {w['science']:.0%}, safety {w['safety']:.0%}, distance {w['distance']:.0%})."


# ---------------------------------------------------------------------------
# Screen 3: navigation (simulated). Map stage (2D map / 3D terrain, switch on the map) and the
# MARSCOPE Mission Center sidebar: overview, assistant, exploration guide, simulation, waypoints.
# ---------------------------------------------------------------------------
STAGE_TICK_S = 1.0     # 2D map refresh while the simulation plays (3D animates itself in the browser)
PANEL_TICK_S = 0.6     # Mission Center refresh while the simulation plays
MODE_LABELS = {"2D": "2D MAP", "3D": "3D TERRAIN"}
SPEED_LABELS = {s: f"{s}×" for s in mc.SPEEDS}


def _route_title(route) -> str:
    return f"Route {route.key} · {flow.CARD_TITLES.get(route.key, route.name)}"


def _status_for(d: float, total: float) -> str:
    return mc.COMPLETED if d >= total - mc.EPS_M else mc.READY if d <= mc.EPS_M else mc.PAUSED


def _sim_sync(route) -> float:
    """Current SIMULATED distance driven (m). Adopts progress changed elsewhere (Mission Control slider)."""
    ss = st.session_state
    total = mc.total_m(route)
    sig = f"{route.key}:{total:.0f}"
    ss.sim_total = total
    ss.setdefault("sim_speed", 1)
    if ss.get("sim_route") != sig:
        ss.sim_route, ss.sim_t0 = sig, None
        ss.sim_d0 = ss.get("sim_progress_pct", 0) / 100 * total
        ss._sim_speed_eff = ss.sim_speed
        ss.sim_status = _status_for(ss.sim_d0, total)
    elif ss.sim_t0 is None and int(round(100 * ss.sim_d0 / total)) != ss.get("sim_progress_pct", 0):
        ss.sim_d0 = ss.sim_progress_pct / 100 * total
        ss.sim_status = _status_for(ss.sim_d0, total)
    return mc.clock_d(ss.sim_d0, ss.sim_t0, ss.get("_sim_speed_eff", 1), time.time(), total)


def _sim_set(route, d: float, status: str, t0: float | None = None) -> None:
    ss = st.session_state
    total = mc.total_m(route)
    ss.sim_d0, ss.sim_t0, ss.sim_status = float(min(max(d, 0.0), total)), t0, status
    ss.sim_progress_pct = int(round(100 * ss.sim_d0 / total))
    ss.nav_playing = status == mc.PLAYING
    ss._nav_app_rerun = True          # the map stage and the 3D view must see the new state


def _sim_play(route) -> None:
    ss = st.session_state
    d, total = _sim_sync(route), mc.total_m(route)
    if ss.sim_status == mc.PLAYING:
        _sim_set(route, d, mc.PAUSED)
        mlog.log("Simulation paused", f"Route {route.key} at {d / 1000:.2f} km (simulated)", "sim")
        return
    if ss.sim_status in (mc.STOPPED, mc.COMPLETED) or d >= total - mc.EPS_M:
        d = 0.0
    ss._sim_speed_eff = ss.sim_speed
    _sim_set(route, d, mc.PLAYING, t0=time.time())
    mlog.log("Simulation started", f"Route {route.key} from {d / 1000:.2f} km at {ss.sim_speed}× (simulated)", "sim")


def _sim_stop(route) -> None:
    d = _sim_sync(route)
    _sim_set(route, d, mc.STOPPED)
    mlog.log("Simulation stopped", f"Route {route.key} at {d / 1000:.2f} km (simulated)", "sim")


def _sim_reset(route) -> None:
    ss = st.session_state
    ss.sim_speed = ss._sim_speed_eff = 1
    ss.nav_detail, ss.nav_detail_sel = None, "—"
    _sim_set(route, 0.0, mc.READY)
    mlog.log("Simulation reset", f"Route {route.key} back to the start", "sim")


def _sim_step(route, wps, direction: int) -> None:
    ss = st.session_state
    total = mc.total_m(route)
    target = mc.step_target(wps, _sim_sync(route), direction)
    _sim_set(route, target, _status_for(target, total))
    w = next((w for w in wps if abs(w["d"] - target) < 1), None)
    if w and w["poi_id"]:
        ss.nav_detail = ss.nav_detail_sel = w["poi_id"]
    mlog.log("Simulation stepped", f"{'Next' if direction > 0 else 'Previous'} waypoint: "
             f"{mc.wp_label(w) if w else f'{target / 1000:.2f} km'} (simulated)", "sim")


def _sim_jump(route, d: float) -> None:
    _sim_set(route, d, _status_for(d, mc.total_m(route)))
    mlog.log("Simulated rover moved", f"Route {route.key} to {d / 1000:.2f} km", "sim")


def _sim_speed(route, speed: int) -> None:
    ss = st.session_state
    ss.sim_speed = speed
    d = _sim_sync(route)                       # position at the old speed
    ss._sim_speed_eff = ss.sim_speed
    if ss.sim_status == mc.PLAYING:
        _sim_set(route, d, mc.PLAYING, t0=time.time())
    ss._nav_app_rerun = True
    mlog.log("Simulation speed", f"{ss.sim_speed}× ({ss.sim_speed:g} sol per second, simulated)", "sim")


def _end_mission() -> None:
    ss = st.session_state
    pct = ss.sim_progress_pct
    mlog.log("Mission ended", f"Route {ss.selected_route} stopped at {pct}% (simulated)", "sim")
    ss.nav_playing, ss.sim_progress_pct, ss.sim_route, ss.sim_t0 = False, 0, None, None
    ss._flash = f"Simulated mission on Route {ss.selected_route} ended at {pct}% of the route."
    go("routes")


def _set_mode() -> None:
    mlog.log("View switched", f"{MODE_LABELS.get(st.session_state.nav_mode, '')} (simulation state kept)")


def _toggle_panel() -> None:
    st.session_state.nav_panel = not st.session_state.get("nav_panel", True)


def _want_tab(key: str, tab: str) -> None:
    """Open a dock section on the next run (a widget's state cannot change after it is drawn)."""
    st.session_state._dock_next = (key, tab)
    st.session_state.nav_panel = True


def _apply_dock_next() -> None:
    nxt = st.session_state.pop("_dock_next", None)
    if nxt:
        st.session_state[nxt[0]] = nxt[1]


def _select_wp(pid: str | None, source: str = "list") -> None:
    ss = st.session_state
    if pid and source in ("2D map", "3D terrain"):
        _want_tab("dock_n" if ss.get("view") == "navigate" else "dock_r",
                  "Route" if ss.get("view") == "navigate" else "Science")
    ss.nav_detail = pid
    ss.nav_detail_sel = pid or "—"
    ss._nav_app_rerun = True
    if pid:
        mlog.log("Target inspected", f"{pid} ({source})")


def _ask_assistant(label: str) -> None:
    ss = st.session_state
    ss.assistant_pending = (f"What is scientifically interesting near the simulated rover position ({label}), "
                            "and what should I look at next?")
    ss._nav_app_rerun = True
    _want_tab("dock_n", "Assistant")


def _handle_nav3d(app: App, key: str = "nav3d", view: str = "navigate") -> None:
    """Pin or terrain clicks in the 3D view select a waypoint / catalogued target."""
    ss = st.session_state
    ev = ss.get(key)
    if not isinstance(ev, dict) or ev.get("n") == ss.get(f"_{key}_n"):
        return
    ss[f"_{key}_n"] = ev.get("n")
    if ev.get("type") == "poi":
        pid = ev.get("id")
    else:
        poi = flow.nearest_poi(float(ev["lat"]), float(ev["lon"]), app.pois, radius_m=500)
        pid = poi["id"] if poi else None
        if pid is None:
            st.toast("No catalogued target within 500 m of that point.")
    if pid in app.poi_by_id:
        _select_wp(pid, "3D terrain")
        _apply_dock_next()
        ss._nav_app_rerun = False


def navigate_screen(app: App) -> None:
    ss = st.session_state
    if not have_routes(app):
        st.info("Plan a route first.")
        st.button("Edit locations", on_click=go, args=("setup",), key="back_setup3")
        return
    route = next((r for r in ss.routes if r.key == ss.selected_route and r.found), None)
    if route is None:
        go("routes")
        st.rerun()
    ss.pop("_nav_app_rerun", None)            # this full run already redraws everything
    if ss.get("nav_mode") not in MODE_LABELS:
        ss.nav_mode = "2D"
    _handle_nav3d(app)
    _apply_dock_next()
    stops = flow.notable_stops(route, app.poi_by_id)
    wps = mc.waypoints(route, stops, ss.start, ss.destination, app.pois)
    d = _sim_sync(route)
    open_panel = ss.get("nav_panel", True)
    with st.container(key="map_row"):
        st.markdown(f"<div class='dock-{'open' if open_panel else 'closed'}' hidden></div>", unsafe_allow_html=True)
        cols = st.columns([2.55, 1], gap="small")
    with cols[0]:
        _nav_stage(app, route, stops, d, open_panel)
    with cols[1], st.container(key="screen_nav_side"):
        if not open_panel:
            return
        _dock_head("Mission panel", "SIMULATION · NO LIVE ROVER LINK")
        st.markdown("<style>" + "".join(
            f".st-key-wpt_{i}{{--wp:{_wp_color(w)}}}" for i, w in enumerate(wps)) + "</style>", unsafe_allow_html=True)
        tick = PANEL_TICK_S if ss.sim_status == mc.PLAYING else None
        st.fragment(run_every=tick)(_dock_strip)(app, route, wps)
        if ss.get("dock_n") not in dock.NAV_TABS:
            ss.dock_n = "Assistant"
        _dock_tabs("dock_n", dock.NAV_TABS)
        tab = ss.dock_n
        with st.container(key="dock_body_n_" + tab.lower()):
            if tab == "Assistant":
                ev = evaluation_for(app)
                ctx, sig = dock.context(app, ev, route, _sim_sync(route), "navigation", ss.nav_mode,
                                        ss.get("nav_detail"))
                with st.container(key="dock_chat"):
                    dock.chat(ctx, sig, ev)
                dock.recommendation(ctx, ev, app.badge or (lambda k: ""), _show_route_key, _back_to_setup)
            elif tab == "System":
                _dock_system(app, route)
            else:
                st.fragment(run_every=tick)(_mission_center)(app, route, stops, wps, tab)


def _dock_system(app: App, route) -> None:
    ss = st.session_state
    from modules import st_compat
    st.markdown(_sec("", "System status") + dock.system_rows(app, [
        ("Simulation", f"{mc.STATUS_TEXT[ss.sim_status]} · {ss.get('_sim_speed_eff', 1)}×", "ok"),
        ("View", MODE_LABELS[ss.nav_mode], "ok"),
        ("Active route", _route_title(route), "ok")]), unsafe_allow_html=True)
    with st.container(key="layer_panel"):
        with st.expander("Map layers (2D map)", expanded=True):
            st.pills("Map layers", LAYER_OPTIONS, selection_mode="multi", key="nav_layers",
                     label_visibility="collapsed", disabled=ss.nav_mode == "3D")
            st.markdown(nav_map.legend_html({LAYER_KEYS[x] for x in ss.nav_layers or []}) +
                        ("<div class='fl-cov'>2D: CTX ~20 m/px everywhere, HiRISE ~1.2 m/px over the landing "
                         "site, delta front and Belva crater (zoom in). 3D: the CTX DTM as a ~300 m mesh with "
                         "its own colour modes (top left of the 3D view).</div>"), unsafe_allow_html=True)
    if st_compat.dropped():
        st.caption("Streamlit compatibility: ignored on this version · " + ", ".join(st_compat.dropped()))
    st.button("Data & Methods", key="dock_methods", icon=":material/dataset:", on_click=open_section,
              args=("Data & Methods",))
    with st.container(key="end_btn"):
        st.button("End mission and return to routes", key="end_mission", on_click=_end_mission,
                  type="tertiary", width="stretch")


def _wp_color(w: dict) -> str:
    from modules import terrain3d
    for _, col, _, ids in terrain3d.GEO_GROUPS:
        if w.get("poi_id") in ids:
            return col
    return {"start": "#4f8f86", "dest": "#c98b6b"}.get(w["kind"], "#d9c39c")


# ---------------------------------------------------------------- map stage
def _nav_stage(app: App, route, stops, d: float, open_panel: bool = True) -> None:
    ss = st.session_state
    mode = ss.nav_mode
    playing = ss.sim_status == mc.PLAYING
    with st.container(key="nav_stage"):
        st.markdown(f"<div class='nav-mode-{mode.lower()}' hidden></div>", unsafe_allow_html=True)
        _view_switch()
        if not open_panel:
            _dock_handle()
        st.fragment(run_every=STAGE_TICK_S if playing and mode == "2D" else None)(_nav_map2d)(app, route, stops)
        with st.container(key="nav_3d_box"):
            _nav_3d(app, route, d, active=mode == "3D")


def _nav_map2d(app: App, route, stops) -> None:
    ss = st.session_state
    total = mc.total_m(route)
    d = _sim_sync(route)
    state = rover_sim.state_at(route, d / total, app.pois)
    show = {LAYER_KEYS[x] for x in (ss.nav_layers or [])}
    try:
        m = _nav_base_map(app, route, stops, show)
        fg = nav_map.rover_layer(state, route)
        sel = app.poi_by_id.get(ss.get("nav_detail"))
        if sel:
            folium.CircleMarker([sel["lat"], sel["lon"]], radius=18, color="#d9c39c", weight=2.5, fill=False,
                                tooltip=f"Selected: {short(sel['name'])}").add_to(fg)
        out = st_folium(m, height=760, use_container_width=True, returned_objects=["last_object_clicked"],
                        feature_group_to_add=fg, key="nav_map_f")
    except Exception as exc:  # noqa: BLE001
        _map_fallback(app, exc)
        return
    obj = (out or {}).get("last_object_clicked")
    if obj and obj != ss.nav_last_obj:
        ss.nav_last_obj = obj
        poi = flow.nearest_poi(float(obj["lat"]), float(obj["lng"]), app.pois, radius_m=60)
        if poi and poi["id"] != ss.nav_detail:
            _select_wp(poi["id"], "2D map")
            ss.nav_panel = True
            st.rerun(scope="app")


def _nav_base_map(app: App, route, stops, show):
    ss = st.session_state
    focus = app.poi_by_id.get(ss.get("nav_detail"))       # keep the selected waypoint in view (3D -> 2D)
    return nav_map.build(app.ds, app.layers, routes=ss.routes, selected=route.key, start=ss.start,
                         destination=ss.destination, stops=stops, pois=app.pois, show=show,
                         route_bounds=_focus_bounds(focus["lat"], focus["lon"]) if focus else
                         nav_map.route_fit_bounds(route, ss.start, ss.destination), dim_others=False,
                         rover_control=True, fill=True)


def _nav_3d(app: App, route, d: float, active: bool, key: str = "nav3d", preview: bool = False) -> None:
    """The existing 3D colour-map engine in navigation mode, driven by the shared simulation clock."""
    from modules import terrain3d
    from modules import workspace as wsp
    ss = st.session_state
    data = wsp._terrain_data(app.ds, app.tm, app.sm, (app.ds.status, app.ds.elevation.shape))
    scn = terrain3d.scene(ss.routes, route.key, ss.start, ss.destination, None, None)
    playing = ss.get("sim_status") == mc.PLAYING and not preview
    scn.update(ui="nav", active=active, highlight=ss.get("nav_detail"),
               controlled={"d": round(d, 1), "rate": mc.rate_m_per_s(ss.get("_sim_speed_eff", 1)) if playing else 0,
                           "status": "ROUTE PREVIEW" if preview else ss.sim_status,
                           "speed": ss.get("_sim_speed_eff", 1)})
    terrain3d.render(data, scn, height=760, key=key)


# ---------------------------------------------------------------- Mission Center (live while playing)
def _sec(num: str, title: str, extra: str = "") -> str:
    return f"<div class='mx-sec'>{f'<span>{num}</span>' if num else ''}{_e(title)}{extra}</div>"


def _live(app: App, route, wps):
    """Shared per-tick state for the dock (completion is detected here)."""
    ss = st.session_state
    if ss.pop("_nav_app_rerun", False):
        st.rerun(scope="app")
    total = mc.total_m(route)
    d = _sim_sync(route)
    if ss.sim_status == mc.PLAYING:
        ss.sim_progress_pct = int(round(100 * d / total))
        if d >= total - mc.EPS_M:
            _sim_set(route, total, mc.COMPLETED)
            ss.pop("_nav_app_rerun", None)
            mlog.log("Destination reached", f"Route {route.key} after ~{total / config.SIM_PACE_M_PER_SOL:.0f} "
                                            "sols (simulated)", "sim")
            st.rerun(scope="app")
    return total, d


def _dock_strip(app: App, route, wps) -> None:
    """Always-visible mission strip: status, progress ring, next waypoint, play/pause and waypoint steps."""
    ss = st.session_state
    total, d = _live(app, route, wps)
    status = ss.sim_status
    playing = status == mc.PLAYING
    cur, nxt = mc.current_next(wps, d)
    with st.container(key="dock_strip"):
        st.markdown(
            f"<div class='dk-strip dk-{status.lower()}'>{_gauge_html(d / total if total else 0.0)}"
            f"<div class='dk-s'><div class='dk-s1'><em class='mx-st st-{status.lower()}'>{_e(status)}</em>"
            f"<span>sol {mc.sim_sol(d):.1f} · {d / 1000:.2f}/{total / 1000:.2f} km</span></div>"
            f"<div class='dk-s2'>{'Next · ' + _e(mc.wp_label(nxt)) if nxt else 'Destination reached'}"
            + (f" <b>{(nxt['d'] - d) / 1000:.2f} km</b>" if nxt else "") + "</div>"
            + _progress_html(wps, d, total, "Simulated progress along the route") + "</div></div>",
            unsafe_allow_html=True)
        with st.container(key="dock_ctl", horizontal=True):
            st.button("Pause" if playing else ("Again" if status in (mc.STOPPED, mc.COMPLETED) else "Play"),
                      key="dk_play", on_click=_sim_play, args=(route,), type="primary",
                      icon=":material/pause:" if playing else ":material/play_arrow:",
                      help="Play or pause the simulated drive")
            st.button("", key="dk_back", on_click=_sim_step, args=(route, wps, -1), icon=":material/skip_previous:",
                      disabled=d <= mc.EPS_M, help="Previous waypoint")
            st.button("", key="dk_fwd", on_click=_sim_step, args=(route, wps, 1), icon=":material/skip_next:",
                      disabled=d >= total - mc.EPS_M, help="Next waypoint")


def _mission_center(app: App, route, stops, wps, tab: str = "Mission") -> None:
    ss = st.session_state
    total, d = _live(app, route, wps)
    status = ss.sim_status
    state = rover_sim.state_at(route, d / total, app.pois)
    cur, nxt = mc.current_next(wps, d)
    hz = flow.hazard_ahead(route, state.index)
    near = mc.nearest_feature(app.pois, state.lat, state.lon)
    title = _route_title(route)
    if tab == "Mission":
        _mc_overview(route, wps, d, total, status, title)
        _mc_assistant(app, route, state, d, total, status, cur, nxt, hz, near, title)
    elif tab == "Navigation":
        _mc_navigation(route, state, d, total, status, cur, nxt, hz)
        _mc_simulation(route, wps, d, total, status, cur, nxt)
    elif tab == "Route":
        _mc_route(route, wps, d, total, hz)
        _mc_waypoints(app, route, stops, wps, d, nxt)
    elif tab == "Science":
        _mc_guide(app, route, state, wps, d, total, status, cur, nxt, hz, near, title)


def _mc_navigation(route, state, d, total, status, cur, nxt, hz) -> None:
    head = mc.bearing_text(route, state.index)
    eta = mc.eta_sols(nxt["d"] - d) if nxt else 0
    st.markdown(
        _sec("", "Navigation", "<em class='mx-sim'>SIMULATED</em>") +
        "<div class='mx-card dk-nav'>"
        f"<div class='dk-head'>{_e(head) if head and status != mc.COMPLETED else 'Holding at the destination'}</div>"
        "<div class='mx-grid'>"
        f"<div><span>Next waypoint</span>{_e(mc.wp_label(nxt)) if nxt else 'None, destination reached'}</div>"
        f"<div><span>Distance to it</span>{(nxt['d'] - d) / 1000 if nxt else 0:.2f} km</div>"
        f"<div><span>Estimated arrival</span>~{eta} sol{'s' if eta != 1 else ''} at {config.SIM_PACE_M_PER_SOL:.0f} m/sol</div>"
        f"<div><span>Remaining route</span>{(total - d) / 1000:.2f} km · ~{mc.eta_sols(total - d)} sols</div>"
        f"<div><span>Simulated position</span>{state.lat:.4f}° N {state.lon:.4f}° E</div>"
        f"<div><span>Terrain ahead</span>{_e(mc.terrain_phrase(hz))}</div></div>"
        "<div class='mx-foot'>Heading and distances follow the planned route geometry; the position is simulated.</div>"
        "</div>", unsafe_allow_html=True)


def _mc_route(route, wps, d, total, hz) -> None:
    m = route.metrics
    warn = hz["level"] != "gentle"
    st.markdown(
        _sec("", "Route intelligence", f"<em class='mx-n'>{_e(route.key)}</em>") +
        "<div class='mx-card'>" + _progress_html(wps, d, total, "Route progress") +
        "<div class='mx-kv'>"
        f"<span><b class='hud-count' style='--to:{int(round(100 * d / total))}'></b>% driven</span>"
        f"<span><b>{m.get('max_slope_deg', 0):.0f}°</b> max slope</span>"
        f"<span><b>{m.get('pct_above_caution', 0):.1f}%</b> above {config.ASSUMPTIONS['caution_slope_deg']:.0f}°</span>"
        f"<span><b>+{m.get('elev_gain_m', 0):.0f} / −{m.get('elev_loss_m', 0):.0f}</b> m climb / descent</span></div>"
        f"<div class='mx-grid'><div><span>Slope here / ahead</span><b class='{'mx-warn' if warn else 'mx-ok'}'>"
        f"{hz['here_deg']:.0f}° / max {hz['max_ahead_deg']:.0f}° in {hz['window_m'] / 1000:.1f} km</b></div>"
        f"<div><span>Terrain ahead</span>{_e(mc.terrain_phrase(hz))}</div></div>"
        + _stream_html(route) +
        f"<div class='mx-foot'>Derived from the CTX DTM profile of the planned route {mcc.tag('derived')}</div>"
        "</div>", unsafe_allow_html=True)


def _progress_html(wps, d: float, total: float, label: str) -> str:
    pct = 100 * d / total
    ticks = "".join(f"<i class='tk {mc.wp_state(w, d)}' style='left:{100 * w['d'] / total:.2f}%' "
                    f"title='{_e(mc.wp_label(w))}'></i>" for w in wps)
    return (f"<div class='mx-bar' role='progressbar' aria-valuenow='{pct:.0f}' aria-valuemin='0' aria-valuemax='100' "
            f"aria-label='{_e(label)}'><b style='width:{pct:.2f}%'></b>{ticks}</div>")


def _mc_overview(route, wps, d, total, status, title) -> None:
    ss = st.session_state
    n_stops = sum(w["kind"] == "stop" for w in wps)
    with st.container(key="nav_overview"):
        st.markdown(
            _sec("", "Mission overview", f"<em class='mx-st st-{status.lower()}'>{_e(status)}</em>") +
            f"<div class='mx-card'>"
            f"<div class='mx-route' style='--c:{route.color}'><i></i>{_e(title)}</div>"
            f"<div class='mx-trip'><i class='s'></i>{_e(wps[0]['name'])}<span>→</span><i class='d'></i>"
            f"{_e(wps[-1]['name'])}</div>"
            f"<div class='mx-kv'><span><b>{total / 1000:.2f}</b> km</span><span><b class='hud-count' style='--to:{n_stops}'></b> stops</span>"
            f"<span>~<b>{total / config.SIM_PACE_M_PER_SOL:.0f}</b> sols at {config.SIM_PACE_M_PER_SOL:.0f} m/sol</span>"
            f"<span>max slope <b>{route.metrics.get('max_slope_deg', 0):.0f}°</b></span></div>"
            f"<div class='mx-foot'>{100 * d / total:.0f}% · {_e(mc.STATUS_TEXT[status])} · "
            f"{ss.get('_sim_speed_eff', 1)}× · simulated</div></div>", unsafe_allow_html=True)
        st.markdown(_stream_html(route), unsafe_allow_html=True)   # unchanged between ticks: no restart


def _gauge_html(frac: float) -> str:
    """Ring gauge of the SIMULATED progress (fraction of the planned route driven)."""
    c = 2 * np.pi * 31
    frac = min(max(float(frac), 0.0), 1.0)
    return ("<div class='hud-gauge' role='img' aria-label='Simulated progress "
            f"{100 * frac:.0f} percent'><svg viewBox='0 0 74 74'><circle class='bg' cx='37' cy='37' r='31'/>"
            "<circle class='tk' cx='37' cy='37' r='31'/>"
            f"<circle class='fg' cx='37' cy='37' r='31' stroke-dasharray='{c:.1f}' "
            f"stroke-dashoffset='{c * (1 - frac):.1f}'/></svg><b>{100 * frac:.0f}%<small>SIM</small></b></div>")


def _stream_html(route) -> str:
    """Slow ticker of the planned route's CTX DTM profile (real derived values, not telemetry)."""
    p = route.profile
    n = len(p["distance_m"])
    idx = np.unique(np.linspace(0, n - 1, min(n, 14)).astype(int))
    items = " · ".join(f"<em>{p['distance_m'][i] / 1000:.1f} km</em> {p['elevation_m'][i]:.0f} m "
                       f"{p['slope_deg'][i]:.0f}°" for i in idx)
    return (f"<div class='hud-stream' aria-hidden='true'><span>CTX DTM ROUTE PROFILE (derived, not telemetry) · "
            f"{items}</span></div>")


def _mc_assistant(app, route, state, d, total, status, cur, nxt, hz, near, title) -> None:
    head = mc.bearing_text(route, state.index)
    warn = hz["level"] != "gentle"
    narr_title = f"Route {route.key} ({flow.CARD_TITLES.get(route.key, route.name)})"
    next_txt = (f"{_e(mc.wp_label(nxt))} · {(nxt['d'] - d) / 1000:.2f} km" if nxt else "None, destination reached")
    with st.container(key="nav_hud"):
        st.markdown(
            _sec("", "Situation report", "<em class='mx-sim'>SIMULATED · no live rover link</em>") +
            f"<div class='mx-card mx-as'><p class='mx-narr' aria-live='polite'>"
            f"{_e(mc.narrative(status, narr_title, d, total, cur, nxt, hz, near))}</p>"
            "<div class='mx-grid'>"
            f"<div><span>Route</span>{_e(title)}</div>"
            f"<div><span>Simulated rover position</span>{state.lat:.4f}° N {state.lon:.4f}° E</div>"
            f"<div><span>Current waypoint</span>{_e(mc.wp_label(cur))}</div>"
            f"<div><span>Next waypoint</span>{next_txt}</div>"
            f"<div><span>Remaining</span>{(total - d) / 1000:.2f} km · ~{mc.eta_sols(total - d)} sols</div>"
            f"<div><span>Terrain</span>{_e(mc.terrain_class_at(app.ds, app.tm, state.lat, state.lon))}</div>"
            f"<div><span>Slope here / ahead</span><b class='{'mx-warn' if warn else 'mx-ok'}'>{hz['here_deg']:.0f}° / "
            f"max {hz['max_ahead_deg']:.0f}° in {hz['window_m'] / 1000:.1f} km</b></div>"
            f"<div><span>Progress · status</span>{100 * d / total:.0f}% · {_e(mc.STATUS_TEXT[status])}</div>"
            "</div>"
            + (f"<div class='mx-head'>{_e(head)}</div>" if head and status != mc.COMPLETED else "") +
            (f"<div class='mx-alert'>Slope warning: {hz['max_ahead_deg']:.0f}° ahead, above the "
             f"{hz['caution_deg']:.0f}° caution slope (CTX DTM, derived).</div>" if warn and status != mc.COMPLETED else "")
            + "<div class='mx-foot'>Built from the planned route, the CTX DTM profile and the published catalogue. "
              "Position, sols and progress are simulated, not rover telemetry.</div></div>",
            unsafe_allow_html=True)
        st.button("Ask the Assistant about this position", key="nav_ask", type="tertiary",  # -> Assistant tab
                  on_click=_ask_assistant, args=(f"{state.lat:.4f}° N {state.lon:.4f}° E",),
                  icon=":material/forum:")


def _cite(item: dict) -> str:
    return f"{_e(item['text'])} <span class='cite'>[{_e(item['source'])}]</span>"


def _mc_guide(app, route, state, wps, d, total, status, cur, nxt, hz, near, title) -> None:
    dest_poi = app.poi_by_id.get(wps[-1]["poi_id"]) if wps[-1]["poi_id"] else None
    n_stops = sum(w["kind"] == "stop" for w in wps)
    obj = (f"Drive {_e(title)} from {_e(wps[0]['name'])} to {_e(wps[-1]['name'])}: {total / 1000:.2f} km past "
           f"{n_stops} catalogued stop{'s' if n_stops != 1 else ''}.")
    if dest_poi and dest_poi.get("objectives"):
        obj += "<ul>" + "".join(f"<li>{_e(o)}</li>" for o in dest_poi["objectives"][:3]) + "</ul>" \
               f"<div class='mx-src'>Destination objectives from the MARSCOPE catalogue {mcc.tag('lit')}</div>"
    if near:
        p, dist = near
        loc = (f"<b>{_e(short(p['name']))}</b> · {_e(p['category'])} · {dist / 1000:.2f} km from the simulated "
               "position.")
        if p.get("observations"):
            loc += f"<p>{_cite(p['observations'][0])} {mcc.tag('real')}</p>"
        if p.get("interpretations"):
            loc += f"<p><i>Hypothesis:</i> {_cite(p['interpretations'][0])} {mcc.tag('lit')}</p>"
    else:
        loc = (f"No catalogued feature within {mc.NEAR_M / 1000:.1f} km. Ground here: "
               f"{_e(mc.terrain_class_at(app.ds, app.tm, state.lat, state.lon))} {mcc.tag('derived')}")
    if nxt:
        np_ = app.poi_by_id.get(nxt["poi_id"]) if nxt["poi_id"] else None
        tgt = (f"<b>{_e(mc.wp_label(nxt))}</b>, {(nxt['d'] - d) / 1000:.2f} km ahead, about "
               f"{mc.eta_sols(nxt['d'] - d)} sols at the assumed pace {mcc.tag('sim')}")
        if np_ and np_.get("objectives"):
            tgt += f"<p>Why go: {_e(np_['objectives'][0])}</p>"
        elif np_ and np_.get("observations"):
            tgt += f"<p>{_cite(np_['observations'][0])}</p>"
    else:
        tgt = "All waypoints reached. Open a waypoint below to review it."
    ter = (f"Elevation {state.elevation_m:,.0f} m (CTX DTM) · slope here {hz['here_deg']:.1f}° · "
           f"{_e(mc.terrain_class_at(app.ds, app.tm, state.lat, state.lon))} {mcc.tag('derived')}"
           f"<p>Next {hz['window_m'] / 1000:.1f} km: max {hz['max_ahead_deg']:.0f}°, {hz['pct_caution']:.0f}% above "
           f"the {hz['caution_deg']:.0f}° caution slope; limit {config.ASSUMPTIONS['max_traversable_slope_deg']:.0f}°."
           f" Terrain ahead is <b class='{'mx-warn' if hz['level'] != 'gentle' else 'mx-ok'}'>"
           f"{_e(mc.terrain_phrase(hz))}</b>.</p>")
    act = _e(mc.recommended_action(status, hz, nxt, (nxt["d"] - d) if nxt else None))
    marks = {"passed": "✓", "here": "●", "ahead": "○"}
    prog = (f"{100 * d / total:.0f}% · sol {mc.sim_sol(d):.1f} · {d / 1000:.2f} of {total / 1000:.2f} km "
            f"{mcc.tag('sim')}<ul class='mx-check'>" + "".join(
                f"<li class='{mc.wp_state(w, d)}'>{marks[mc.wp_state(w, d)]} {_e(mc.wp_label(w))}</li>" for w in wps)
            + "</ul>")
    with st.container(key="nav_guide"):
        st.markdown(_sec("", "Exploration guide"), unsafe_allow_html=True)
        for i, (label, body, open_) in enumerate([
                ("Current objective", obj, False), ("Current location", loc, True), ("Next target", tgt, False),
                ("Terrain", ter, False), ("Recommended next action", act, True), ("Overall progress", prog, False)]):
            with st.expander(f"{i + 1} · {label}", expanded=open_):
                st.markdown(f"<div class='mx-gd'>{body}</div>", unsafe_allow_html=True)


def _mc_simulation(route, wps, d, total, status, cur, nxt) -> None:
    ss = st.session_state
    playing = status == mc.PLAYING
    with st.container(key="sim_controls"):
        st.markdown(_sec("", "Route simulation", "<em class='mx-sim'>SIMULATED</em>"), unsafe_allow_html=True)
        with st.container(key="sim_row1", horizontal=True):
            st.button("Pause" if playing else ("Play again" if status in (mc.STOPPED, mc.COMPLETED) else "Play"),
                      key="sim_play", on_click=_sim_play, args=(route,), type="primary",
                      icon=":material/pause:" if playing else ":material/play_arrow:",
                      help="Pause keeps the rover where it is" if playing else
                      "Drive the planned route at the chosen speed (simulated)")
            st.button("Stop", key="sim_stop", on_click=_sim_stop, args=(route,), icon=":material/stop:",
                      disabled=status not in (mc.PLAYING, mc.PAUSED),
                      help="End this run and keep the rover where it stopped; Play then starts from the start")
            st.button("Reset", key="sim_reset", on_click=_sim_reset, args=(route,), icon=":material/restart_alt:",
                      disabled=status == mc.READY and ss.get("_sim_speed_eff", 1) == 1,
                      help="Back to the start at 1×, selection cleared")
        with st.container(key="sim_row2", horizontal=True):
            st.button("Previous waypoint", key="sim_back", on_click=_sim_step, args=(route, wps, -1),
                      icon=":material/skip_previous:", disabled=d <= mc.EPS_M)
            st.button("Next waypoint", key="sim_fwd", on_click=_sim_step, args=(route, wps, 1),
                      icon=":material/skip_next:", disabled=d >= total - mc.EPS_M)
        st.markdown("<div class='mx-lbl'>Simulation speed · 1× = one simulated sol per second</div>",
                    unsafe_allow_html=True)
        with st.container(key="sim_speed_box", horizontal=True):
            for sp in mc.SPEEDS:
                st.button(SPEED_LABELS[sp], key=f"sim_speed_{sp}", on_click=_sim_speed, args=(route, sp),
                          type="primary" if ss.get("_sim_speed_eff", 1) == sp else "secondary",
                          help=f"{sp} simulated sol{'s' if sp > 1 else ''} per second")
        st.markdown(
            "<div class='mx-card mx-tl'>" + _progress_html(wps, d, total, "Simulation timeline") +
            "<div class='mx-grid'>"
            f"<div><span>Sim time</span>sol {mc.sim_sol(d):.1f} of ~{mc.sim_sol(total):.0f}</div>"
            f"<div><span>Speed</span>{ss.get('_sim_speed_eff', 1)}× · {ss.get('_sim_speed_eff', 1):g} sol/s</div>"
            f"<div><span>Current waypoint</span>{_e(mc.wp_label(cur))}</div>"
            f"<div><span>Next waypoint</span>{_e(mc.wp_label(nxt)) if nxt else '—'}</div>"
            f"<div><span>Distance travelled</span>{d / 1000:.2f} km</div>"
            f"<div><span>Remaining</span>{(total - d) / 1000:.2f} km</div></div></div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=64)
def _thumb(lat: float, lon: float) -> tuple[str | None, str | None]:
    """Small real CTX/HiRISE preview (data URI) from the bundled map tiles."""
    import base64
    import io

    from PIL import Image

    from modules import workspace as wsp
    img, src = wsp.tile_crop(lat, lon, 0.9)
    if img is None:
        return None, None
    im = Image.fromarray(img)
    w, h = im.size
    side = min(w, h)
    im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)).resize((132, 132))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), \
        ("HiRISE" if "HiRISE" in src else "CTX")


def _mc_waypoints(app, route, stops, wps, d, nxt) -> None:
    ss = st.session_state
    sel = ss.get("nav_detail")
    with st.container(key="detail_panel"):
        st.markdown(_sec("", "Waypoint information", f"<em class='mx-n'>{len(wps)}</em>"), unsafe_allow_html=True)
        for i, w in enumerate(wps):
            stt = mc.wp_state(w, d)
            if nxt is not None and w is nxt:
                stt = "next"
            is_sel = bool(w["poi_id"]) and w["poi_id"] == sel
            badge = {"start": "S", "dest": "D"}.get(w["kind"], str(w["n"]))
            thumb = ""
            if stt in ("next", "here") or is_sel:
                uri, src = _thumb(round(w["lat"], 4), round(w["lon"], 4))
                if uri:
                    thumb = (f"<div class='wp-img' role='img' aria-label='{src} image of {_e(w['name'])}' "
                             f"style=\"background-image:url('{uri}')\"></div><em class='wp-src'>{src}</em>")
            word = {"passed": "Passed", "here": "Rover here", "next": "Next", "ahead": ""}[stt]
            with st.container(key=f"wpt_{i}"):
                st.markdown(
                    f"<div class='wp-row' data-state='{stt}' data-sel='{int(is_sel)}'><b class='wp-dot'>{badge}</b>"
                    f"<div class='wp-main'><b>{_e(w['name'])}</b><span>{_e(mc.wp_label(w).split(' · ')[0])} · "
                    f"{w['d'] / 1000:.2f} km · sol {mc.sim_sol(w['d']):.0f}</span></div>"
                    f"<span class='wp-st'>{word}</span>{thumb}</div>", unsafe_allow_html=True)
                st.button(f"Show {mc.wp_label(w)}", key=f"wpb_{i}", on_click=_select_wp,
                          args=(w["poi_id"], "waypoint list"), disabled=not w["poi_id"], type="tertiary")
        others = [p["id"] for p in sorted(app.pois, key=lambda p: -p["priority"])
                  if p["id"] not in {w["poi_id"] for w in wps}]
        opts = ["—"] + others
        if ss.get("nav_detail_sel") not in opts:
            ss.nav_detail_sel = "—"
        st.selectbox("Off-route science targets", opts, key="nav_detail_sel", label_visibility="collapsed",
                     format_func=lambda i: "Off-route science targets…" if i == "—" else
                     "Off route · " + short(app.poi_by_id[i]["name"]),
                     on_change=lambda: _select_wp(None if st.session_state.nav_detail_sel == "—"
                                                  else st.session_state.nav_detail_sel, "target list"))
        if sel in app.poi_by_id:
            st.markdown(detail_card_html(app, app.poi_by_id[sel], route, stops), unsafe_allow_html=True)
            w = next((w for w in wps if w["poi_id"] == sel), None)
            with st.container(horizontal=True, key="detail_actions"):
                if w is not None:
                    st.button("Move simulated rover here", key="wp_jump", on_click=_sim_jump, args=(route, w["d"]),
                              icon=":material/my_location:", disabled=abs(w["d"] - d) < 1)
                st.button("Close details", key="close_detail", on_click=_select_wp, args=(None,), type="tertiary")


def detail_card_html(app: App, poi: dict, route, stops) -> str:
    info = sd.inspect_location(app.ds, app.tm, app.sm, poi["lat"], poi["lon"])
    t = info["terrain"]
    r_, c_ = app.ds.fine_grid.lonlat_to_rowcol(poi["lat"], poi["lon"])
    measured = bool(app.ds.dtm_coverage[r_, c_])
    obs = "".join(f"<li>{_e(o['text'])} <span class='cite'>[{_e(o['source'])}]</span></li>"
                  for o in poi.get("observations", [])) or "<li>None catalogued.</li>"
    hyp = "".join(f"<li>{_e(i['text'])} <span class='cite'>[{_e(i['source'])}]</span></li>"
                  for i in poi.get("interpretations", [])) or "<li>None catalogued.</li>"
    mins = ", ".join(_e(m) for m in poi.get("minerals_reported", [])) or "none reported"
    stop = next(((n, idx) for n, p, idx in stops if p["id"] == poi["id"]), None)
    if stop:
        d_m = float(route.profile["distance_m"][stop[1]])
        sim = (f"Stop {stop[0]} on Route {route.key}: {d_m / 1000:.2f} km from the start, reached after "
               f"~{d_m / config.SIM_PACE_M_PER_SOL:.0f} sols at the assumed pace.")
    else:
        sim = f"Not on Route {route.key} (no part of the route passes within " \
              f"{config.ASSUMPTIONS['poi_visit_radius_m']:.0f} m)."
    origin = "Published target" if poi.get("origin") == "published" else "MARSCOPE identification from the DTM"
    return f"""<div class='fl-detail'>
  <div class='fd-h'>{_e(poi['name'])}</div>
  <div class='fd-sub'>{_e(poi['category'])} · {origin} · {poi['lat']:.4f}° N {poi['lon']:.4f}° E (±0.5–1 km)</div>
  <div class='fd-s'>Observations (measured, published) {mcc.tag('real')}</div><ul>{obs}</ul>
  <div class='fd-s'>Interpretations (hypotheses) {mcc.tag('lit')}</div><ul>{hyp}</ul>
  <div class='fd-s'>Reported minerals {mcc.tag('lit')}</div><p>{mins}</p>
  <div class='fd-s'>Derived here {mcc.tag('derived')}</div>
  <p>Elevation {t['elevation_m']:,.0f} m ({'CTX DTM' if measured else 'gap-filled'}) · slope {t['slope_deg']:.1f}° ·
     {_e(t['terrain_class'].split(' (')[0])} · science value {info['science_value']:.2f}</p>
  <div class='fd-s'>On this route {mcc.tag('sim')}</div><p>{_e(sim)}</p>
  <div class='fd-foot'>Observations are published measurements; interpretations are what the cited authors
  think they mean. MARSCOPE does not process CRISM spectra.</div></div>"""


# ---------------------------------------------------------------------------
# Start Mission Guide
# ---------------------------------------------------------------------------
def open_guide(app: App) -> None:
    ss = st.session_state
    step = ss.pop("_guide_paused", None)
    _guide_to(app, step if step is not None else 0)


def _guide_to(app: App, i: int) -> None:
    ss = st.session_state
    i = max(0, min(guide.N - 1, i))
    ss.guide_step = i
    s = guide.STEPS[i]
    if s.dock:
        ss[{"routes": "dock_r", "navigate": "dock_n"}[s.view]] = s.dock
        ss.nav_panel = True
    if not s.needs_routes or have_routes(app):
        go(s.view)
    else:
        go("setup")


def _guide_close(skip: bool) -> None:
    ss = st.session_state
    ss._guide_paused = None if skip else ss.guide_step
    ss.guide_step = None


def _guide_demo(app: App) -> None:
    ss = st.session_state
    if not ss.get("start"):
        set_endpoint(app, "start", app.poi_point(flow.START_DEFAULT_POI), "guide")
    if not ss.get("destination"):
        set_endpoint(app, "destination", app.poi_point(config.DEMO_DESTINATION_POI), "guide")
    if calculate(app):
        _guide_to(app, ss.guide_step)


def guide_card(app: App) -> None:
    ss = st.session_state
    i = ss.get("guide_step")
    if i is None:
        return
    s = guide.STEPS[i]
    st.markdown(guide.highlight_css(s), unsafe_allow_html=True)
    with st.container(key="guide_card"):
        st.markdown(f"<div class='gd-n'>Mission Guide · {i + 1} of {guide.N}</div>"
                    f"<div class='gd-dots'>{''.join('<i class=on></i>' if k <= i else '<i></i>' for k in range(guide.N))}</div>"
                    f"<div class='gd-t'>{_e(s.title)}</div><div class='gd-b'>{_e(s.body)}</div>",
                    unsafe_allow_html=True)
        if s.needs_routes and not have_routes(app):
            st.markdown("<div class='gd-warn'>This step needs calculated routes.</div>", unsafe_allow_html=True)
            st.button("Calculate routes for me" if ss.get("start") and ss.get("destination")
                      else "Use the demo mission (landing site → Bright Angel)", key="gd_demo",
                      on_click=_guide_demo, args=(app,), width="stretch")
        b = st.columns([1, 1.15, 1, 1])
        b[0].button("← Back", key="gd_back", on_click=_guide_to, args=(app, i - 1), disabled=i == 0, width="stretch")
        if i < guide.N - 1:
            b[1].button("Next →", key="gd_next", type="primary", on_click=_guide_to, args=(app, i + 1), width="stretch")
        else:
            b[1].button("Finish", key="gd_finish", type="primary", on_click=_guide_close, args=(True,), width="stretch")
        b[2].button("Skip tour", key="gd_skip", on_click=_guide_close, args=(True,), width="stretch")
        b[3].button("Close", key="gd_close", on_click=_guide_close, args=(False,), width="stretch",
                    help="Hide the guide; “Resume Mission Guide” brings it back at this step")
        st.button("↺ Restart from step 1", key="gd_restart", on_click=_guide_to, args=(app, 0))


# ---------------------------------------------------------------------------
def inject_css(view: str) -> None:
    """Theme for every view; the start screen adds the visor template."""
    css = (config.ASSETS_DIR / "flow.css").read_text()
    if view == "setup":
        css = (config.ASSETS_DIR / "landing.css").read_text() + css
    else:                                   # astronaut-helmet HUD theme: every screen after the start screen
        css += (config.ASSETS_DIR / "hud.css").read_text()
    st.markdown(f"<style>{css}</style><div class='view-{view}' hidden></div>", unsafe_allow_html=True)


def run(app: App) -> None:
    """Draw the current flow screen (setup / routes / navigate)."""
    ss = st.session_state
    if ss.view in ("routes", "navigate") and not have_routes(app):
        ss.view = "setup"
    inject_css(ss.view)
    if ss.view != "setup":
        top_bar(app)
    if ss.get("_flash"):
        st.toast(ss.pop("_flash"))
    if app.ds.is_demonstration:
        st.warning("Real terrain data could not be loaded: routes use labelled SYNTHETIC demonstration terrain.")
    {"setup": setup_screen, "routes": routes_screen, "navigate": navigate_screen}[ss.view](app)
    guide_card(app)
