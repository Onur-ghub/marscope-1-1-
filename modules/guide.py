"""Start Mission Guide: a 10-step walkthrough of the mission flow.

Each step names the screen it belongs to and the control it explains. The app switches to that
screen and outlines the control (CSS on its ``st-key-…`` container). Steps that need calculated
routes say so and offer to calculate them; the guide never fakes a result.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    title: str
    body: str
    view: str                 # setup | routes | navigate
    targets: tuple = ()       # st.container keys to highlight
    needs_routes: bool = False
    dock: str | None = None   # mission-panel section to open for this step


STEPS = [
    Step("The start screen",
         "Planning starts here with only two things to choose: where you start and where you want to go. "
         "Everything else appears once a route exists.", "setup", ("setup_panel",)),
    Step("Start",
         "The start defaults to the Perseverance landing site (Octavia E. Butler Landing). Open the field to pick "
         "another landing site or target, or type coordinates.", "setup", ("start_panel",)),
    Step("Destination and coordinates",
         "Pick a science target from MARSCOPE's catalogue of published Jezero results, or type latitude, "
         "longitude in degrees east, e.g. 18.47, 77.40. Points outside the terrain data are refused.",
         "setup", ("dest_panel",)),
    Step("Swap and show the route",
         "Swap exchanges the two locations. Show route plans four routes on the CTX elevation model, then "
         "opens the map. Changing a location later re-plans, so a route never belongs to old endpoints.",
         "setup", ("swap_row", "calc_btn")),
    Step("The route preview",
         "The map fills the screen and the selected route draws itself first. Map tools let a click inspect a "
         "point or move the start or destination.", "routes", ("routes_stage", "routes_stage_enter"), True),
    Step("Four route options",
         "Safest, Shortest, Science-focused and Balanced/custom, each with estimated distance, risk, science "
         "value and stops, in the Routes section of the mission panel on the right. The recommended option says "
         "why. Preview one to see it on the map.", "routes", ("route_panel", "route_panel_enter"), True, "Routes"),
    Step("2D map and 3D terrain",
         "The switch on the map changes between the detailed 2D map and 3D terrain built from the real CTX "
         "elevation model. Route, waypoints, the selected waypoint and the simulated rover stay in sync, and "
         "switching keeps the simulation and the selected waypoint in focus. The switch is on the route preview "
         "too.", "navigate", ("view_switch",), True),
    Step("Mission panel and the AI assistant",
         "The panel on the right of the map holds the Assistant, which knows the selected route, the simulated "
         "rover position and the 2D/3D view, plus Mission, Navigation, Route, Science and System sections. The "
         "strip at its top always shows status, progress and the next waypoint. Every position is SIMULATED "
         "along the planned route: MARSCOPE has no live rover connection and Mars has no GPS.",
         "navigate", ("dock_tabs", "dock_strip", "dock_chat"), True, "Assistant"),
    Step("Route simulation and layers",
         "Play, pause, stop or reset the simulation, step to the next or previous waypoint and choose 1x to "
         "10x speed in the Navigation section; Play and the waypoint steps are also in the strip. The timeline "
         "shows sols, distance driven and remaining. Hide panel widens the map.", "navigate",
         ("sim_controls", "dock_strip"), True, "Navigation"),
    Step("Waypoints, science details and every module",
         "Pick a waypoint in the list or on either map to open its details: observations, interpretations, "
         "derived and simulated values stay separate, with citations. Map layers and End mission are in System. "
         "The section bar at the top opens Mission Control, the 3D colour map with its animations, Science "
         "Discovery, Exploration Routes and Data & Methods.", "navigate",
         ("detail_panel", "ws_nav_box"), True, "Route"),
]

N = len(STEPS)


def highlight_css(step: Step) -> str:
    sel = ", ".join(f".st-key-{t}" for t in step.targets)
    if not sel:
        return ""
    return (f"<style>{sel}{{outline:2px solid #7FB8AE!important;outline-offset:4px;border-radius:10px;"
            f"box-shadow:0 0 0 5px rgba(127,184,174,.16);animation:guidePulse 1.6s ease-in-out 2}}"
            "@media (prefers-reduced-motion: reduce){" + sel + "{animation:none}}</style>")
