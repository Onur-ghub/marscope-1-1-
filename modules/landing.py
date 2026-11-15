"""Astronaut-visor landing screen for MARSCOPE.

UI only: this module draws the screen, reads and validates the destination, and
shows progress stages. It does no terrain analysis or route planning itself; the
app passes in callables that run the existing pipeline (terrain model, A*/Dijkstra
planner), so every stage shown corresponds to a real operation.
"""
from __future__ import annotations

import re
from typing import Callable

import streamlit as st

import config

TYPE_HINT = "Choose a site or type coordinates: 18.47, 77.40"
_COORD_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*°?\s*([NS])?\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*°?\s*([EW])?\s*$", re.I)


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Parse 'lat, lon' in planetocentric degrees, east-positive longitude.

    Accepts '18.47, 77.40', '18.47 77.40' and '18.47N 77.40E'. Returns None if the
    text is not a coordinate pair.
    """
    m = _COORD_RE.match(text or "")
    if not m:
        return None
    lat, ns, lon, ew = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    if ns and ns.upper() == "S":
        lat = -abs(lat)
    if ew and ew.upper() == "W":
        lon = 360 - abs(lon)  # MARSCOPE uses 0-360 east longitude
    return lat, lon


def resolve_destination(choice: str | None, poi_by_label: dict, contains: Callable[[float, float], bool]):
    """Return ((lat, lon, name), None) or (None, error message)."""
    if not choice or not str(choice).strip():
        return None, "Choose a destination first: pick a site from the list or type coordinates."
    if choice in poi_by_label:
        p = poi_by_label[choice]
        return (p["lat"], p["lon"], p["name"]), None
    ll = parse_coordinates(str(choice))
    if ll is None:
        return None, (f"“{choice}” is not a known site or a coordinate pair. "
                      "Use decimal degrees, latitude first, e.g. 18.47, 77.40.")
    lat, lon = ll
    if not contains(lat, lon):
        lo0, la0, lo1, la1 = config.REGION_BOUNDS
        return None, (f"{lat:.4f}°N, {lon:.4f}°E is outside the area MARSCOPE has terrain data for "
                      f"(latitude {la0}–{la1}°N, longitude {lo0}–{lo1}°E, Jezero crater).")
    return (lat, lon, "Map point"), None


def inject_css(css_text: str) -> None:
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


def hud_frame(readouts_left: list[tuple[str, str]], readouts_right: list[tuple[str, str]], credit: str) -> None:
    """Background terrain, visor edge and corner readouts (pure HTML, no widgets)."""
    def block(items, cls):
        return f"<div class='hud-readout {cls}'>" + "".join(
            f"<div><span>{k}</span>{v}</div>" for k, v in items) + "</div>"
    st.markdown(
        "<div class='visor-scene' aria-hidden='true'><div class='visor-sky'></div>"
        "<div class='visor-ground'></div><div class='visor-haze'></div></div>"
        "<div class='visor-frame' aria-hidden='true'><i class='c tl'></i><i class='c tr'></i>"
        "<i class='c bl'></i><i class='c br'></i><div class='visor-reticle'></div></div>"
        + block(readouts_left, "left") + block(readouts_right, "right")
        + f"<div class='visor-credit'>{credit}</div>",
        unsafe_allow_html=True)


def render(options: list[str], start_label: str, on_submit: Callable[[str | None], None],
           busy: bool, error: str | None) -> st.delta_generator.DeltaGenerator:
    """Draw the destination panel. Returns a container where progress stages are written."""
    panel = st.container(key="visor_panel")
    with panel:
        st.markdown("<div class='visor-brand'>MARSCOPE</div>"
                    "<div class='visor-tag'>Mars surface exploration system</div>",
                    unsafe_allow_html=True)
        st.markdown("<label class='visor-label' for='dest'>Destination</label>", unsafe_allow_html=True)
        st.selectbox("Destination", options, index=None, key="landing_dest", placeholder=TYPE_HINT,
                     accept_new_options=True, label_visibility="collapsed", disabled=busy)
        st.markdown(f"<div class='visor-start'>From <b>{start_label}</b></div>", unsafe_allow_html=True)
        st.button("Generating route…" if busy else "Generate route", key="landing_go", type="primary",
                  width="stretch", disabled=busy,
                  on_click=lambda: on_submit(st.session_state.get("landing_dest")))
        if error:
            st.markdown(f"<div class='visor-error' role='alert'>{error}</div>", unsafe_allow_html=True)
        stages = st.empty()
    return stages


def show_stages(slot, stages: list[tuple[str, str]], detail: str = "") -> None:
    """stages: [(label, state)] with state in {'done', 'active', 'todo'}."""
    mark = {"done": "✓", "active": "›", "todo": "·"}
    slot.markdown("<div class='visor-stages' aria-live='polite'>" + "".join(
        f"<div class='st-{s}'><b>{mark[s]}</b>{label}</div>" for label, s in stages)
        + (f"<div class='visor-stage-detail'>{detail}</div>" if detail else "") + "</div>",
        unsafe_allow_html=True)
