"""Simulated rover state along a planned route (no real telemetry exists in MARSCOPE).

The rover position is SIMULATED: the user chooses how far along the selected route it is.
Elevation and slope at that position are DERIVED from the real CTX DTM (the route profile).
Energy is a SIMULATED physics estimate from stated assumptions (config.ENERGY_MODEL).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from modules.data_manager import haversine_m


def energy_wh(distance_m: np.ndarray, elevation_m: np.ndarray) -> np.ndarray:
    """Cumulative traction energy (Wh): rolling resistance plus climbing, divided by drive efficiency."""
    e = config.ENERGY_MODEL
    mg = e["rover_mass_kg"] * e["gravity_m_s2"]
    step = np.diff(distance_m, prepend=distance_m[:1])
    climb = np.clip(np.diff(elevation_m, prepend=elevation_m[:1]), 0, None)
    joules = mg * (e["rolling_resistance"] * step + climb) / e["drive_efficiency"]
    return np.cumsum(joules) / 3600.0


def route_energy_wh(route) -> float | None:
    p = getattr(route, "profile", None) or {}
    if "distance_m" not in p:
        return None
    return float(energy_wh(p["distance_m"], p["elevation_m"])[-1])


@dataclass
class RoverState:
    progress: float            # 0-1 along the route (SIMULATED)
    lat: float
    lon: float
    index: int                 # index into the route profile
    traveled_m: float
    remaining_m: float
    elevation_m: float
    slope_deg: float
    energy_used_wh: float
    energy_total_wh: float
    sols_elapsed: float
    status: str
    passed_targets: list       # POI ids within the visit radius of the driven part
    next_target: tuple | None  # (poi id, name, metres ahead)


def state_at(route, progress: float, pois: list[dict]) -> RoverState | None:
    p = getattr(route, "profile", None) or {}
    if not getattr(route, "found", False) or "distance_m" not in p:
        return None
    d, z = p["distance_m"], p["elevation_m"]
    total = float(d[-1])
    progress = float(np.clip(progress, 0.0, 1.0))
    i = int(np.searchsorted(d, progress * total, side="left"))
    i = min(i, len(d) - 1)
    e = energy_wh(d, z)
    radius = config.ASSUMPTIONS["poi_visit_radius_m"]
    passed, ahead = [], []
    for poi in pois:
        dist = haversine_m(poi["lat"], poi["lon"], p["lat"], p["lon"])
        j = int(np.argmin(dist))
        if dist[j] > radius:
            continue
        (passed if j <= i else ahead).append((j, poi))
    nxt = min(ahead, key=lambda t: t[0]) if ahead else None
    status = ("At start (simulation not started)" if progress == 0 else
              "Arrived at destination" if progress >= 1 else "En route")
    return RoverState(
        progress=progress, lat=float(p["lat"][i]), lon=float(p["lon"][i]), index=i,
        traveled_m=float(d[i]), remaining_m=total - float(d[i]), elevation_m=float(z[i]),
        slope_deg=float(p["slope_deg"][i]), energy_used_wh=float(e[i]), energy_total_wh=float(e[-1]),
        sols_elapsed=float(d[i]) / config.SIM_PACE_M_PER_SOL, status=status,
        passed_targets=[poi["id"] for _, poi in sorted(passed, key=lambda t: t[0])],
        next_target=(nxt[1]["id"], nxt[1]["name"], float(d[nxt[0]] - d[i])) if nxt else None)


def progress_after_one_sol(route, progress: float) -> float:
    total = float(route.profile["distance_m"][-1])
    return min(1.0, progress + config.SIM_PACE_M_PER_SOL / max(total, 1.0))
