"""Right-side mission dock: the Astronaut Assistant inside the map screens (presentation only).

The answers come from the same code as before (assistant_ui.get_reply -> assistant.answer), with the same
shared conversation (st.session_state.chat). This module only builds the context for the map screens
(selected route, simulated rover position, active 2D/3D view, selected waypoint) and draws a compact chat.
"""
from __future__ import annotations

import html

import streamlit as st

import config
from modules import ai_service, assistant, assistant_ui, rover_sim
from modules import mission_center as mc
from modules import mission_context as mctx
from modules import science_discovery as sd

# Dock sections (one shown at a time). The route-preview screen has no simulation yet, so it has fewer.
NAV_TABS = ["Assistant", "Mission", "Navigation", "Route", "Science", "System"]
ROUTE_TABS = ["Routes", "Assistant", "Science", "System"]
TAB_ICONS = {"Assistant": ":material/forum:", "Mission": ":material/radar:", "Navigation": ":material/explore:",
             "Route": ":material/route:", "Science": ":material/science:", "System": ":material/memory:",
             "Routes": ":material/alt_route:"}
TAB_SHORT = {"Navigation": "Nav"}


def tab_label(t: str) -> str:
    # Icons are drawn by CSS (hud.css): an icon inside a segmented-control label breaks the selected
    # state on older Streamlit versions.
    return TAB_SHORT.get(t, t)


def tab_icon_css() -> str:
    """Material-symbol icon above each section label, by position in each tab bar."""
    rules = []
    for key, tabs in (("dock_r", ROUTE_TABS), ("dock_n", NAV_TABS)):
        for i, t in enumerate(tabs):
            name = TAB_ICONS[t].split("/")[-1].strip(":")
            rules.append(f".st-key-{key} button:nth-of-type({i + 1})::after{{content:'{name}'}}")
    return "<style>" + "".join(rules) + "</style>"


def _e(x) -> str:
    return html.escape(str(x))


def context(app, evaluation, route=None, d: float | None = None, screen: str = "navigation",
            mode: str = "2D", detail: str | None = None) -> tuple[dict, str]:
    """Mission context for the Assistant, from MARSCOPE data only (simulated values are labelled)."""
    ss = st.session_state
    try:
        waypoints = app.mission_waypoints() if app.mission_waypoints else None
    except Exception:  # noqa: BLE001  (optional real traverse; the assistant works without it)
        waypoints = None
    ctx = mctx.build_context(app.ds, app.tm, app.sm, ss.start, ss.destination, evaluation, ss.selected_route,
                             waypoints)
    sig = mctx.signature(ss.start, ss.destination, ss.get("route_sig"), ss.selected_route, ss.get("max_route_km"))
    if route is not None and d is not None:
        total = mc.total_m(route)
        rv = rover_sim.state_at(route, d / total if total else 0.0, app.pois)
        names = {p["id"]: p["name"] for p in app.pois}
        ctx["simulated_rover"] = {
            "note": "SIMULATED position along the planned route; MARSCOPE is not connected to a real rover.",
            "route": route.key, "progress_pct": round(rv.progress * 100), "lat": round(rv.lat, 4),
            "lon_east": round(rv.lon, 4), "traveled_km": round(rv.traveled_m / 1000, 2),
            "remaining_km": round(rv.remaining_m / 1000, 2), "elevation_m_derived": round(rv.elevation_m),
            "slope_deg_derived": round(rv.slope_deg, 1),
            "simulation_status": ss.get("sim_status"), "simulated_sol": round(mc.sim_sol(d), 1),
            "targets_passed": [names.get(p, p) for p in rv.passed_targets],
            "next_target_on_route": ({"name": rv.next_target[1], "km_ahead": round(rv.next_target[2] / 1000, 2)}
                                     if rv.next_target else None)}
        sig += f"|{int(round(100 * d / total)) if total else 0}"
    ctx["interface"] = {
        "screen": screen, "visualization_mode": "3D terrain" if mode == "3D" else "2D map",
        "note": "What the user is looking at in MARSCOPE. Positions are simulated; there is no live rover link."}
    poi = app.poi_by_id.get(detail) if detail else None
    if poi:
        info = sd.inspect_location(app.ds, app.tm, app.sm, poi["lat"], poi["lon"])
        ctx["selected_location"] = {
            "name": poi["name"], "lat": round(poi["lat"], 4), "lon_east": round(poi["lon"], 4),
            "catalogued_point_of_interest": poi["id"],
            "note": "The waypoint or target the user selected on the map or in the waypoint list. "
                    "'Selected location', 'this location' and 'here' refer to it.",
            **sd.build_fact_sheet(info), "why_interesting_marscope": info["why"]}
        sig += f"|focus:{poi['id']}"
    return ctx, sig


def chat(ctx: dict, sig: str, evaluation, history_height: int = 330) -> None:
    """Compact chat: suggested questions, the latest exchange, earlier messages folded away, input."""
    ss = st.session_state
    ok, provider, _ = ai_service.status()
    st.markdown(
        "<div class='dk-ai'><span class='dk-dot " + ("on" if ok else "off") + "'></span>" + (
            f"{_e(provider)} · grounded in MARSCOPE mission data" if ok else
            "Offline mode · answers come straight from MARSCOPE data (set ANTHROPIC_API_KEY for full "
            "conversation)") + "</div>", unsafe_allow_html=True)
    log = ss.chat
    if ss.get("_chat_sig") != sig.split("|")[0]:
        if log and ctx.get("destination"):
            log.append({"role": "event", "content": f"Mission updated · destination {ctx['destination']['name']}"})
        ss._chat_sig = sig.split("|")[0]
    with st.container(key="dock_suggest"):
        for i, q in enumerate(assistant.suggested_questions(ctx)[:4]):
            st.button(q, key=f"dq_{i}", on_click=lambda q=q: ss.update(assistant_pending=q))
    msgs = [m for m in log]
    last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=None)
    older, recent = (msgs[:last_user], msgs[last_user:]) if last_user else ([], msgs)
    box = st.container(height=history_height, key="dock_history")
    with box:
        if older:
            with st.expander(f"Earlier conversation · {sum(m['role'] == 'user' for m in older)} questions"):
                for m in older:
                    _msg(m)
        if not msgs and not (ss.get("assistant_pending") or ss.get("dock_chat_input")):
            st.markdown("<div class='ai-empty'>Ask about the route, the terrain ahead, the next waypoint or the "
                        "science here. The assistant knows the selected route, the simulated rover position and "
                        "the 2D/3D view.</div>", unsafe_allow_html=True)
        for m in recent:
            _msg(m)
    typed = st.chat_input("Ask the mission assistant…", key="dock_chat_input")
    question = typed or ss.get("assistant_pending")
    ss.assistant_pending = None
    if question:
        with box:
            with st.chat_message("user", avatar=assistant_ui.AVATARS["user"]):
                st.markdown(question)
            with st.chat_message("assistant", avatar=assistant_ui.AVATARS["assistant"]):
                reply = assistant_ui.get_reply(question, ctx, evaluation, sig, ok)
                st.markdown(reply.text)
                if reply.note:
                    st.caption(reply.note + (" · sources: " + ", ".join(reply.sources) if reply.sources else ""))
        assistant_ui.record(question, reply)
        from modules import mission_log as mlog
        mlog.log("AI query", question[:80])
    if log:
        st.button("Clear conversation", key="dock_chat_clear", type="tertiary",
                  on_click=lambda: ss.update(chat=[], chat_cache={}))


def _msg(m: dict) -> None:
    if m["role"] == "event":
        st.markdown(f"<div class='ai-event'>{_e(m['content'])}</div>", unsafe_allow_html=True)
        return
    with st.chat_message(m["role"], avatar=assistant_ui.AVATARS[m["role"]]):
        st.markdown(m["content"])
        if m.get("note"):
            st.caption(m["note"] + (" · sources: " + ", ".join(m["sources"]) if m.get("sources") else ""))


def recommendation(ctx, evaluation, badge, on_show_route, on_choose_dest) -> None:
    """The Assistant's mission recommendation (scores, candidates, max length) folded into the dock."""
    with st.expander("Mission recommendation · all candidates", expanded=False):
        assistant_ui._recommendation(ctx, evaluation, st.session_state.selected_route, badge, on_show_route,
                                     on_choose_dest)


def system_rows(app, extra: list[tuple[str, str, str]]) -> str:
    """System status card: only states MARSCOPE actually knows (data source, AI mode, simulation)."""
    ok, provider, _ = ai_service.status()
    ds = app.ds
    rows = [("Terrain data", "SYNTHETIC demonstration" if ds.is_demonstration else f"USGS CTX DTM · {ds.status}",
             "warn" if ds.is_demonstration else "ok"),
            ("Assistant", provider if ok else "offline mode (MARSCOPE data only)", "ok" if ok else "dim"),
            ("Rover link", "none · positions are simulated", "dim")] + extra
    return "<div class='dk-sys'>" + "".join(
        f"<div class='{c}'><span>{_e(k)}</span><b>{_e(v)}</b></div>" for k, v, c in rows) + \
        f"<div class='dk-foot'>Assumed pace {config.SIM_PACE_M_PER_SOL:.0f} m/sol · 1× = one simulated sol per " \
        "second.</div></div>"
