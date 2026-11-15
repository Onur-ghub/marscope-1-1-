"""Mission navigation flow: pure helpers (no Streamlit), so they can be unit tested.

Setup -> Calculate routes -> Compare route cards -> Navigate (simulated) -> Mission Control.
Everything here is computed from the route profiles produced by route_optimizer and the
deterministic evaluation in mission_planner. Rover positions are SIMULATED.
"""
from __future__ import annotations

import math

import numpy as np

import config
from modules.data_manager import haversine_m

VIEWS = ("setup", "routes", "navigate", "all")
STAGES = [("setup", "1 · Set up"), ("routes", "2 · Compare routes"), ("navigate", "3 · Navigate")]

# What each card is called in the flow (the planner's internal keys stay A-D).
CARD_TITLES = {"A": "Safest", "B": "Shortest", "C": "Science-focused", "D": "Balanced / custom"}

START_DEFAULT_POI = config.DEMO_START_POI          # Perseverance landing site
START_CATEGORIES = ("Mission landmark",)            # listed first in the start picker
ENDPOINT_GAP_M = 150.0                              # POIs this close to either end are endpoints
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def start_options(pois: list[dict]) -> list[str]:
    """POI ids for the start picker: landing sites / mission landmarks first."""
    first = [p["id"] for p in pois if p["category"].startswith(START_CATEGORIES)]
    return first + [p["id"] for p in pois if p["id"] not in first]


def nearest_poi(lat: float, lon: float, pois: list[dict], radius_m: float = 300.0):
    """The catalogued POI within ``radius_m`` of a map click, or None (used to name clicked points)."""
    best = None
    for p in pois:
        d = float(haversine_m(lat, lon, p["lat"], p["lon"]))
        if d <= radius_m and (best is None or d < best[1]):
            best = (p, d)
    return best[0] if best else None


def endpoint_label(pt) -> str:
    if not pt:
        return "not set"
    return f"{pt[0]:.4f}° N, {pt[1]:.4f}° E" if pt[2] == "Map point" else pt[2]


def routes_valid(routes, route_sig, current_sig) -> bool:
    """Routes may only be shown or navigated if they were computed for the current endpoints/settings."""
    return bool(routes) and route_sig == current_sig


# ---------------------------------------------------------------------------
# Route cards
# ---------------------------------------------------------------------------
def notable_stops(route, pois_by_id: dict) -> list[tuple[int, dict, int]]:
    """[(stop number, poi, profile index)] for catalogued POIs the route passes, in driving order."""
    p = route.profile or {}
    if not route.found or "lat" not in p:
        return []
    order = []
    total = float(p["distance_m"][-1])
    for pid, *_ in route.metrics.get("visited", []):
        poi = pois_by_id.get(pid)
        if poi is None:
            continue
        idx = int(np.argmin(haversine_m(poi["lat"], poi["lon"], p["lat"], p["lon"])))
        along = float(p["distance_m"][idx])
        # the start and destination themselves are not "stops"
        if along < ENDPOINT_GAP_M or total - along < ENDPOINT_GAP_M:
            continue
        order.append((idx, poi))
    order.sort(key=lambda t: t[0])
    return [(n, poi, idx) for n, (idx, poi) in enumerate(order, 1)]


def risk_label(level: str) -> str:
    return {"lower hazard": "Low", "moderate hazard": "Moderate", "elevated hazard": "Elevated"}.get(level, "Unknown")


def card_data(evaluation, pois_by_id: dict) -> list[dict]:
    """One dict per candidate route, in A-D order, with everything a route card shows."""
    rec = evaluation.recommended.key if evaluation and evaluation.recommended else None
    out = []
    for c in sorted(evaluation.candidates, key=lambda c: c.key):
        r, m = c.route, c.raw
        d = {"key": c.key, "title": CARD_TITLES.get(c.key, c.name), "planner_name": c.name, "color": c.color,
             "eligible": c.eligible and not c.duplicate_of, "duplicate_of": c.duplicate_of,
             "excluded": c.exclusions[0] if not c.eligible and c.exclusions else None,
             "recommended": c.key == rec, "rank": c.rank}
        if c.eligible:
            lo, hi = m.get("sols_range", (0, 0))
            d.update({
                "distance_km": m["distance_km"], "sols": (lo, hi),
                "sim_sols": m["distance_km"] * 1000 / config.SIM_PACE_M_PER_SOL,
                "risk": risk_label(c.safety_level), "safety_level": c.safety_level,
                "safety": c.criteria.get("safety"), "science": c.criteria.get("science"),
                "score": c.score, "max_slope": m["max_slope_deg"], "pct_caution": m["pct_above_caution"],
                "stops": [poi["name"].split(" (")[0] for _, poi, _ in notable_stops(r, pois_by_id)],
                "explanation": short_explanation(c, evaluation),
                "warnings": list(c.warnings)})
        out.append(d)
    return out


def short_explanation(c, evaluation) -> str:
    """One or two sentences, built only from the route's own pros/cons and criteria."""
    r = c.route
    pros = [p for p in getattr(r, "pros", []) if not p.startswith("Balanced compromise")]
    cons = [p for p in getattr(r, "cons", []) if not p.startswith("No major drawbacks")]
    intro = {"A": "Minimises terrain difficulty and steep ground.",
             "B": "Takes the most direct traversable path.",
             "C": "Detours to pass catalogued science targets.",
             "D": "Uses your own weights for science, safety and distance."}.get(c.key, "")
    txt = intro
    if pros:
        txt += " " + pros[0]
    if cons:
        txt += " Trade-off: " + cons[0][0].lower() + cons[0][1:]
    return txt.strip()


# ---------------------------------------------------------------------------
# Navigation (simulated)
# ---------------------------------------------------------------------------
def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def compass(b: float) -> str:
    return COMPASS[int((b + 22.5) // 45) % 8]


def _index_at(p: dict, metres: float) -> int:
    return min(int(np.searchsorted(p["distance_m"], metres, side="left")), len(p["distance_m"]) - 1)


def direction(route, index: int, look_m: float = 250.0) -> dict | None:
    """Heading of the next stretch of route and whether it turns afterwards."""
    p = route.profile
    d = p["distance_m"]
    if index >= len(d) - 1:
        return None
    j = _index_at(p, d[index] + look_m)
    k = _index_at(p, d[j] + look_m)
    b1 = bearing_deg(p["lat"][index], p["lon"][index], p["lat"][j], p["lon"][j])
    out = {"bearing": b1, "compass": compass(b1), "for_m": float(d[j] - d[index]), "turn": None}
    if k > j:
        b2 = bearing_deg(p["lat"][j], p["lon"][j], p["lat"][k], p["lon"][k])
        delta = (b2 - b1 + 540) % 360 - 180
        if abs(delta) >= 30:
            out["turn"] = ("right" if delta > 0 else "left", abs(delta), compass(b2))
    return out


def next_waypoint(route, index: int, stops: list[tuple[int, dict, int]]) -> dict:
    """Next numbered stop ahead of the simulated position, else the destination."""
    p = route.profile
    here = float(p["distance_m"][index])
    for n, poi, idx in stops:
        if idx > index:
            return {"label": f"Stop {n} · {poi['name'].split(' (')[0]}", "poi_id": poi["id"],
                    "ahead_m": float(p["distance_m"][idx]) - here}
    return {"label": "Destination", "poi_id": None, "ahead_m": float(p["distance_m"][-1]) - here}


def hazard_ahead(route, index: int, ahead_m: float = 500.0) -> dict:
    """Slope summary for the next ``ahead_m`` of the route (DERIVED from the CTX DTM profile)."""
    p = route.profile
    d = p["distance_m"]
    j = max(_index_at(p, d[index] + ahead_m), index)
    sl = p["slope_deg"][index: j + 1]
    caution = config.ASSUMPTIONS["caution_slope_deg"]
    limit = config.ASSUMPTIONS["max_traversable_slope_deg"]
    mx = float(sl.max()) if len(sl) else float(p["slope_deg"][index])
    steps = np.diff(d[index: j + 1], prepend=d[index])
    pct = float(100 * steps[sl > caution].sum() / max(d[j] - d[index], 1.0)) if len(sl) > 1 else 0.0
    level = "steep" if mx > limit else "caution" if mx > caution else "gentle"
    return {"here_deg": float(p["slope_deg"][index]), "max_ahead_deg": mx, "pct_caution": pct,
            "window_m": float(d[j] - d[index]), "level": level, "caution_deg": caution}


def step_progress(route, progress: float, sols: float = 1.0) -> float:
    total = float(route.profile["distance_m"][-1])
    return float(min(1.0, progress + sols * config.SIM_PACE_M_PER_SOL / max(total, 1.0)))


def fit_view(points: list[tuple[float, float]], width_px: int = 900, height_px: int = 600,
             pad: float = 0.12) -> tuple[tuple[float, float], int]:
    """Centre and Leaflet zoom (EPSG:4326 CRS, 256 px tiles of 180/2^z deg) that fit the points."""
    lats = [a for a, _ in points]
    lons = [b for _, b in points]
    c = ((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)
    dlat = max(max(lats) - min(lats), 0.005) * (1 + 2 * pad)
    dlon = max(max(lons) - min(lons), 0.005) * (1 + 2 * pad)
    px_per_deg_z0 = 256 / 180.0
    z = math.floor(math.log2(min(width_px / (dlon * px_per_deg_z0), height_px / (dlat * px_per_deg_z0))))
    return c, int(max(7, min(17, z)))
