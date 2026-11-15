"""Mission Center logic for the navigation screen (pure functions, no Streamlit).

Everything here is computed from the planned route, the CTX-DTM-derived route profile and the
published POI catalogue. The rover position is SIMULATED: a clock moves it along the planned route at
the assumed pace (config.SIM_PACE_M_PER_SOL). MARSCOPE has no live rover link and no real-time
telemetry; nothing returned here is presented as such.

Simulation clock: the position is ``d0 + rate * (now - t0)`` metres while playing (t0 set) and ``d0``
while paused, so the server-side panels and the 3D view (which runs the same formula in the browser)
stay in step without streaming positions.
"""
from __future__ import annotations

import numpy as np

import config
from modules import flow
from modules.data_manager import haversine_m

SOLS_PER_SECOND = 1.0                   # playback rate at 1x
SPEEDS = (1, 2, 5, 10)
READY, PLAYING, PAUSED, STOPPED, COMPLETED = "READY", "PLAYING", "PAUSED", "STOPPED", "COMPLETED"
STATUS_TEXT = {READY: "Ready · at the start", PLAYING: "Playing", PAUSED: "Paused",
               STOPPED: "Stopped", COMPLETED: "Completed · destination reached"}
NEAR_M = 1500.0                          # "approaching / near" a catalogued feature
EPS_M = 1.0


def short(name: str) -> str:
    return name.split(" (")[0]


def rate_m_per_s(speed: float) -> float:
    return float(speed) * SOLS_PER_SECOND * config.SIM_PACE_M_PER_SOL


def clock_d(d0: float, t0: float | None, speed: float, now: float, total: float) -> float:
    """Simulated distance driven (m) at wall-clock ``now``."""
    if t0 is None:
        return float(min(max(d0, 0.0), total))
    return float(min(total, max(0.0, d0 + rate_m_per_s(speed) * max(0.0, now - t0))))


def total_m(route) -> float:
    return float(route.profile["distance_m"][-1])


def index_at(route, d: float) -> int:
    dist = route.profile["distance_m"]
    return min(int(np.searchsorted(dist, d, side="left")), len(dist) - 1)


# ---------------------------------------------------------------------------
# Waypoints: start, numbered catalogued stops, destination (in driving order)
# ---------------------------------------------------------------------------
def waypoints(route, stops, start, destination, pois: list[dict]) -> list[dict]:
    p = route.profile
    total = total_m(route)

    def endpoint(kind, pt, d, idx):
        poi = flow.nearest_poi(pt[0], pt[1], pois, radius_m=400) if pt else None
        name = pt[2] if pt and pt[2] != "Map point" else (poi["name"] if poi else flow.endpoint_label(pt))
        return {"kind": kind, "n": None, "name": short(name), "poi_id": poi["id"] if poi else None,
                "d": d, "idx": idx, "lat": float(p["lat"][idx]), "lon": float(p["lon"][idx])}

    out = [endpoint("start", start, 0.0, 0)]
    for n, poi, idx in stops:
        out.append({"kind": "stop", "n": n, "name": short(poi["name"]), "poi_id": poi["id"],
                    "d": float(p["distance_m"][idx]), "idx": idx, "lat": poi["lat"], "lon": poi["lon"]})
    out.append(endpoint("dest", destination, total, len(p["distance_m"]) - 1))
    return out


def wp_label(w: dict) -> str:
    return {"start": "Start", "dest": "Destination"}.get(w["kind"], f"Stop {w['n']}") + f" · {w['name']}"


def wp_state(w: dict, d: float) -> str:
    """passed / here / ahead for one waypoint at distance ``d``."""
    if abs(w["d"] - d) <= EPS_M + 0.5:
        return "here"
    return "passed" if w["d"] < d else "ahead"


def current_next(wps: list[dict], d: float) -> tuple[dict, dict | None]:
    """The last waypoint reached and the next one ahead (None after the destination)."""
    cur = wps[0]
    for w in wps:
        if w["d"] <= d + EPS_M:
            cur = w
    nxt = next((w for w in wps if w["d"] > d + EPS_M), None)
    return cur, nxt


def step_target(wps: list[dict], d: float, direction: int) -> float:
    """Distance of the next (+1) or previous (-1) waypoint from ``d``."""
    if direction > 0:
        w = next((w for w in wps if w["d"] > d + EPS_M), wps[-1])
        return w["d"]
    prev = [w for w in wps if w["d"] < d - EPS_M]
    return prev[-1]["d"] if prev else 0.0


# ---------------------------------------------------------------------------
# Terrain and context at the simulated position
# ---------------------------------------------------------------------------
def terrain_phrase(hz: dict) -> str:
    return {"gentle": "gentle", "caution": "moderately steep",
            "steep": "steep, with slopes over the traversability limit"}[hz["level"]]


def nearest_feature(pois: list[dict], lat: float, lon: float, radius_m: float = NEAR_M):
    if not pois:
        return None
    dist = np.array([haversine_m(lat, lon, p["lat"], p["lon"]) for p in pois])
    i = int(np.argmin(dist))
    return (pois[i], float(dist[i])) if dist[i] <= radius_m else None


def terrain_class_at(ds, tm, lat: float, lon: float) -> str:
    from modules.terrain_analysis import TERRAIN_CLASSES
    r, c = ds.fine_grid.lonlat_to_rowcol(lat, lon)
    return short(TERRAIN_CLASSES[int(tm.terrain_class[r, c])][0])


def recommended_action(status: str, hz: dict, nxt: dict | None, ahead_m: float | None) -> str:
    if status == COMPLETED:
        return ("Review the destination's catalogued observations below, or open Science Discovery to compare "
                "it with other targets.")
    if status == STOPPED:
        return "The run is stopped. Press Play to drive the route again from the start, or Reset."
    if hz["level"] == "steep":
        act = ("Slow down: the next stretch exceeds the traversability limit in places. Step through it "
               "waypoint by waypoint, or compare the safer Route A on the Routes screen.")
    elif hz["level"] == "caution":
        act = f"Continue with care: {hz['pct_caution']:.0f}% of the next {hz['window_m'] / 1000:.1f} km is above " \
              f"the {hz['caution_deg']:.0f}° caution slope."
    else:
        act = "Continue along the planned route."
    if status in (READY, PAUSED) and nxt is not None:
        act += f" Press Play or step forward to {wp_label(nxt)}."
    elif nxt is not None and ahead_m is not None and ahead_m < config.ASSUMPTIONS["poi_visit_radius_m"] \
            and nxt["kind"] == "stop":
        act += f" {nxt['name']} is close: a good point to pause and read its observations."
    return act


def narrative(status: str, route_title: str, d: float, total: float, cur: dict, nxt: dict | None, hz: dict,
              near) -> str:
    """Plain-language summary of the SIMULATED situation, built only from route and catalogue data."""
    if status == COMPLETED:
        return (f"The simulated rover has reached {cur['name']} at the end of {route_title}, "
                f"{total / 1000:.2f} km and about {total / config.SIM_PACE_M_PER_SOL:.0f} sols after the start.")
    s, nxt_said = [], False
    if status == READY or d <= EPS_M:
        s.append(f"The simulated rover is at the start, {cur['name']}, ready to drive {route_title}.")
    elif status == STOPPED:
        s.append(f"The simulation is stopped {d / 1000:.2f} km along {route_title}.")
    else:
        lead = "Paused: the" if status == PAUSED else "The"
        if nxt is not None and nxt["d"] - d < 1000:
            s.append(f"{lead} rover is approaching {nxt['name']}, about {(nxt['d'] - d) / 1000:.1f} km ahead.")
            nxt_said = True
        elif near is not None and near[1] < 800:
            s.append(f"{lead} rover is passing {short(near[0]['name'])}, about {near[1] / 1000:.1f} km away.")
        else:
            s.append(f"{lead} rover has left {cur['name']} and is {d / 1000:.2f} km along {route_title}.")
    if nxt is not None and not nxt_said:
        kind = "destination" if nxt["kind"] == "dest" else "next waypoint"
        s.append(f"The {kind} is {nxt['name']}, approximately {(nxt['d'] - d) / 1000:.1f} km ahead.")
    s.append(f"The upcoming terrain is {terrain_phrase(hz)} (up to {hz['max_ahead_deg']:.0f}° in the next "
             f"{hz['window_m'] / 1000:.1f} km).")
    if status != STOPPED:
        s.append({"steep": "Proceed with caution along the planned route.",
                  "caution": "Continue carefully along the planned route."}.get(hz["level"],
                                                                                 "Continue along the planned route."))
    return " ".join(s)


def bearing_text(route, index: int) -> str | None:
    dr = flow.direction(route, index)
    if dr is None:
        return None
    t = f"Heading {dr['compass']} {dr['bearing']:.0f}° for {dr['for_m'] / 1000:.2f} km"
    if dr["turn"]:
        t += f", then bear {dr['turn'][0]}"
    return t


def sim_sol(d: float) -> float:
    return d / config.SIM_PACE_M_PER_SOL


def fmt_km(m: float) -> str:
    return f"{m / 1000:.2f} km"


def eta_sols(m: float) -> int:
    return int(round(max(0.0, m) / config.SIM_PACE_M_PER_SOL))
