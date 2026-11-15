"""Route optimisation over the Jezero terrain graph.

Graph
-----
Nodes are cells of the ~237 m planning grid (config.GRAPH_RES_DEG); each node
links to its 8 neighbours. Every edge gets a cost per metre built from three
normalised terms, weighted by the mission priorities:

    cost(edge) = length_m * ( w_dist
                              + K_SAFETY  * w_safety  * hazard(edge)
                              + K_SCIENCE * w_science * (1 - science(edge)) )

    hazard(edge)  = max(mean cell hazard of both cells,
                        0.7 * (edge grade / max slope)^1.5)
    science(edge) = mean science value of both cells (0..1)

Edges into cells whose 90th-percentile slope exceeds the traversability limit,
or whose own grade exceeds it, are removed (impassable).

Algorithms
----------
* Point-to-point routes use A* with the admissible heuristic
  w_dist * straight-line distance (every edge costs at least w_dist per metre).
* The science-focused route additionally solves a small orienteering problem:
  multi-source Dijkstra (SciPy csgraph) gives the cost between start, end and
  every point of interest, then a greedy insertion heuristic adds the POIs
  with the best priority-per-added-cost ratio while the total stays within a
  detour budget (config.ASSUMPTIONS["science_detour_budget"]).

The weights genuinely change the optimisation: they change edge costs, hence
which path is cheapest, and (for Route C) which POIs fit in the budget.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

import config
from modules.data_manager import RegionGrid, haversine_m
from modules.science_discovery import ScienceModel, pois
from modules.terrain_analysis import TerrainModel, _block_reduce, sample_along

K_SAFETY = 10.0
K_SCIENCE = 2.0
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class Weights:
    distance: float
    safety: float
    science: float

    def normalised(self) -> "Weights":
        d, s, c = max(self.distance, 0.02), max(self.safety, 0.0), max(self.science, 0.0)
        total = d + s + c
        return Weights(d / total, s / total, c / total)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.distance, self.safety, self.science)


def blend_weights(profile: tuple[float, float, float], user: Weights, strategy_strength: float) -> Weights:
    """Combine a strategy's profile with the user's mission priorities."""
    u = user.normalised()
    a = float(np.clip(strategy_strength, 0, 1))
    return Weights(*(a * p + (1 - a) * q for p, q in zip(profile, u.as_tuple()))).normalised()


@dataclass
class Route:
    key: str
    name: str
    color: str
    weights: Weights
    cells: list                      # [(row, col)] on the planning grid
    lats: np.ndarray = field(default=None)
    lons: np.ndarray = field(default=None)
    waypoints: list = field(default_factory=list)   # POI ids deliberately visited (Route C)
    metrics: dict = field(default_factory=dict)
    profile: dict = field(default_factory=dict)     # distance/elevation/slope samples
    pros: list = field(default_factory=list)
    cons: list = field(default_factory=list)
    found: bool = True
    message: str = ""


class RoutePlanner:
    def __init__(self, tm: TerrainModel, sm: ScienceModel):
        self.tm = tm
        self.sm = sm
        self.grid: RegionGrid = tm.graph_grid
        self.H, self.W = self.grid.shape
        f = config.AGG_FACTOR
        self.g_science = _block_reduce(sm.value, f, np.mean)[: self.H, : self.W]
        dy, dx = self.grid.cell_size_m()
        self.dy, self.dx = dy, dx
        self.step_len = np.array([np.hypot(dr * dy, dc * dx) for dr, dc in NEIGHBOURS])
        self._precompute_edges()

    # ------------------------------------------------------------------
    def _precompute_edges(self) -> None:
        """Per-direction edge arrays: length, hazard, science, validity."""
        H, W = self.H, self.W
        max_slope = config.ASSUMPTIONS["max_traversable_slope_deg"]
        elev, sci, passable = self.tm.g_elevation, self.g_science, self.tm.g_passable
        # Cell hazard: mean difficulty blended with the cell's steepest 10% of slopes,
        # so cells containing short steep scarps are penalised even if mostly flat.
        diff = 0.5 * self.tm.g_difficulty + 0.5 * np.clip(self.tm.g_slope_p90 / max_slope, 0, 1) ** 2
        self.e_hazard = np.zeros((8, H, W), np.float32)
        self.e_science = np.zeros((8, H, W), np.float32)
        self.e_valid = np.zeros((8, H, W), bool)
        for k, (dr, dc) in enumerate(NEIGHBOURS):
            r0, r1 = max(0, -dr), H - max(0, dr)
            c0, c1 = max(0, -dc), W - max(0, dc)
            src = (slice(r0, r1), slice(c0, c1))
            dst = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
            grade = np.degrees(np.arctan(np.abs(elev[dst] - elev[src]) / self.step_len[k]))
            hz = np.maximum(0.5 * (diff[src] + diff[dst]),
                            0.7 * np.clip(grade / max_slope, 0, 1) ** 1.5)
            self.e_hazard[k][src] = hz
            self.e_science[k][src] = 0.5 * (sci[src] + sci[dst])
            self.e_valid[k][src] = passable[dst] & passable[src] & (grade <= max_slope)

    def edge_costs(self, w: Weights) -> np.ndarray:
        per_m = w.distance + K_SAFETY * w.safety * self.e_hazard + K_SCIENCE * w.science * (1 - self.e_science)
        cost = per_m * self.step_len[:, None, None]
        return np.where(self.e_valid, cost, np.inf).astype(np.float64)

    # ------------------------------------------------------------------
    def snap(self, lat: float, lon: float) -> tuple[int, int]:
        """Nearest passable planning cell to a coordinate."""
        r, c = self.grid.lonlat_to_rowcol(lat, lon)
        if self.tm.g_passable[r, c]:
            return r, c
        rr, cc = np.nonzero(self.tm.g_passable)
        i = np.argmin((rr - r) ** 2 + (cc - c) ** 2)
        return int(rr[i]), int(cc[i])

    def astar(self, start: tuple[int, int], goal: tuple[int, int], costs: np.ndarray,
              w_dist: float) -> tuple[list, float]:
        H, W = self.H, self.W
        n = H * W
        s, g = start[0] * W + start[1], goal[0] * W + goal[1]
        gr, gc = goal
        best = np.full(n, np.inf)
        parent = np.full(n, -1, np.int64)
        closed = np.zeros(n, bool)
        best[s] = 0.0

        def h(r, c):
            return w_dist * np.hypot((r - gr) * self.dy, (c - gc) * self.dx)

        heap = [(h(*start), 0.0, s)]
        flat_costs = costs.reshape(8, n)
        while heap:
            f, g_cost, u = heapq.heappop(heap)
            if closed[u]:
                continue
            if u == g:
                break
            closed[u] = True
            ur, uc = divmod(u, W)
            for k, (dr, dc) in enumerate(NEIGHBOURS):
                vr, vc = ur + dr, uc + dc
                if not (0 <= vr < H and 0 <= vc < W):
                    continue
                step = flat_costs[k, u]
                if step == np.inf:
                    continue
                v = vr * W + vc
                ng = g_cost + step
                if ng < best[v]:
                    best[v] = ng
                    parent[v] = u
                    heapq.heappush(heap, (ng + h(vr, vc), ng, v))
        if best[g] == np.inf:
            return [], np.inf
        path, u = [], g
        while u != -1:
            path.append(divmod(int(u), W))
            u = parent[u]
        return path[::-1], float(best[g])

    def sparse_graph(self, costs: np.ndarray) -> sparse.csr_matrix:
        H, W = self.H, self.W
        idx = np.arange(H * W).reshape(H, W)
        rows, cols, data = [], [], []
        for k, (dr, dc) in enumerate(NEIGHBOURS):
            r0, r1 = max(0, -dr), H - max(0, dr)
            c0, c1 = max(0, -dc), W - max(0, dc)
            src = idx[r0:r1, c0:c1].ravel()
            dst = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc].ravel()
            cst = costs[k][r0:r1, c0:c1].ravel()
            ok = np.isfinite(cst)
            rows.append(src[ok]); cols.append(dst[ok]); data.append(cst[ok])
        return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(H * W, H * W))

    # ------------------------------------------------------------------
    def plan_point_to_point(self, key, name, color, w: Weights, start_ll, end_ll) -> Route:
        costs = self.edge_costs(w)
        s, g = self.snap(*start_ll), self.snap(*end_ll)
        cells, cost = self.astar(s, g, costs, w.distance)
        route = Route(key, name, color, w, cells)
        if not cells:
            route.found = False
            route.message = "No traversable path under the current slope limit."
        route.metrics["optimiser_cost"] = cost
        return route

    def plan_science(self, key, name, color, w: Weights, start_ll, end_ll,
                     exclude: tuple = ("three_forks_depot",)) -> Route:
        """Orienteering: visit high-priority POIs within a detour budget."""
        costs = self.edge_costs(w)
        graph = self.sparse_graph(costs)
        W = self.W
        s, g = self.snap(*start_ll), self.snap(*end_ll)
        cand = [p for p in pois() if self.grid.contains(p["lat"], p["lon"]) and p["id"] not in exclude]
        cand_cells = [self.snap(p["lat"], p["lon"]) for p in cand]
        nodes = [s, g] + cand_cells
        node_idx = [r * W + c for r, c in nodes]
        dist, pred = dijkstra(graph, directed=True, indices=node_idx, return_predecessors=True)
        D = dist[:, node_idx]
        if not np.isfinite(D[0, 1]):
            return Route(key, name, color, w, [], found=False,
                         message="No traversable path under the current slope limit.")

        budget = config.ASSUMPTIONS["science_detour_budget"] * D[0, 1]
        order = [0, 1]
        total = D[0, 1]
        remaining = set(range(2, len(nodes)))
        # skip POIs that coincide with start/end cells
        remaining = {i for i in remaining if nodes[i] not in (s, g)}
        while remaining:
            best = None
            for i in remaining:
                prio = cand[i - 2]["priority"]
                for pos in range(1, len(order)):
                    a, b = order[pos - 1], order[pos]
                    added = D[a, i] + D[i, b] - D[a, b]
                    if not np.isfinite(added) or total + added > budget:
                        continue
                    ratio = prio / (added + 0.05 * D[0, 1])
                    if best is None or ratio > best[0]:
                        best = (ratio, i, pos, added)
            if best is None:
                break
            _, i, pos, added = best
            order.insert(pos, i)
            total += added
            remaining.discard(i)
            # POIs already within the visit radius of the new waypoint count as visited
            remaining = {j for j in remaining if nodes[j] != nodes[i]}

        cells = []
        for a, b in zip(order[:-1], order[1:]):
            seg, u = [], node_idx[b]
            while u != node_idx[a] and u >= 0:
                seg.append(divmod(int(u), W))
                u = pred[a, u]
            seg.append(divmod(node_idx[a], W))
            seg = seg[::-1]
            cells.extend(seg if not cells else seg[1:])
        route = Route(key, name, color, w, cells,
                      waypoints=[cand[i - 2]["id"] for i in order[1:-1]])
        route.metrics["optimiser_cost"] = float(total)
        return route

    # ------------------------------------------------------------------
    def evaluate(self, route: Route, start_ll, end_ll) -> Route:
        """Compute transparent metrics for a route on the fine DTM grid."""
        if not route.found or not route.cells:
            return route
        latlon = [self.grid.rowcol_to_lonlat(r, c) for r, c in route.cells]
        lats = np.array([start_ll[0]] + [p[0] for p in latlon[1:-1]] + [end_ll[0]])
        lons = np.array([start_ll[1]] + [p[1] for p in latlon[1:-1]] + [end_ll[1]])
        if len(latlon) == 1:
            lats, lons = np.array([start_ll[0], end_ll[0]]), np.array([start_ll[1], end_ll[1]])
        route.lats, route.lons = lats, lons

        # densify to ~30 m for profiles
        seg = haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
        dl, dn, dd = [lats[0]], [lons[0]], [0.0]
        for i, L in enumerate(seg):
            n = max(1, int(np.ceil(L / 30.0)))
            t = np.arange(1, n + 1) / n
            dl.extend(lats[i] + t * (lats[i + 1] - lats[i]))
            dn.extend(lons[i] + t * (lons[i + 1] - lons[i]))
            dd.extend(dd[-1] + t * L)
        dl, dn, dd = np.array(dl), np.array(dn), np.array(dd)
        fg = self.tm.fine_grid
        z = sample_along(self.tm.elevation, fg, dl, dn)
        slope = sample_along(self.tm.slope_deg, fg, dl, dn)
        diff = sample_along(self.tm.difficulty, fg, dl, dn)
        sci = sample_along(self.sm.value, fg, dl, dn)
        dz = np.diff(z)

        caution = config.ASSUMPTIONS["caution_slope_deg"]
        step = np.diff(dd, prepend=0)
        visited = []
        for p in pois():
            d = haversine_m(dl, dn, p["lat"], p["lon"]).min()
            if d <= config.ASSUMPTIONS["poi_visit_radius_m"]:
                visited.append((p["id"], p["name"], float(d), p["priority"]))
        total_prio = sum(p["priority"] for p in pois() if self.grid.contains(p["lat"], p["lon"]))
        poi_term = sum(v[3] for v in visited) / max(total_prio, 1e-6)
        science_score = 100 * (0.4 * float(sci.mean()) + 0.6 * poi_term)
        lo_speed, hi_speed = config.ASSUMPTIONS["indicative_speed_m_per_sol"]
        dist_m = float(dd[-1])
        route.metrics.update({
            "distance_km": dist_m / 1000,
            "straight_line_km": float(haversine_m(*start_ll, *end_ll)) / 1000,
            "mean_difficulty": float(diff.mean()),
            "p90_difficulty": float(np.percentile(diff, 90)),
            "max_slope_deg": float(slope.max()),
            "mean_slope_deg": float(slope.mean()),
            "pct_above_caution": float(100 * step[slope > caution].sum() / max(dist_m, 1)),
            "elev_gain_m": float(dz[dz > 0].sum()),
            "elev_loss_m": float(-dz[dz < 0].sum()),
            "cumulative_elev_change_m": float(np.abs(dz).sum()),
            "net_elev_change_m": float(z[-1] - z[0]),
            "pois_visited": len(visited),
            "visited": visited,
            "mean_science_value": float(sci.mean()),
            "science_score": science_score,
            "sols_range": (dist_m / hi_speed, dist_m / lo_speed),
        })
        route.profile = {"distance_m": dd, "elevation_m": z, "slope_deg": slope,
                         "difficulty": diff, "science": sci, "lat": dl, "lon": dn}
        return route


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def annotate_tradeoffs(routes: list[Route]) -> None:
    """Fill pros/cons for each route by comparison with the others (transparent rules)."""
    ok = [r for r in routes if r.found and r.metrics.get("distance_km")]
    if not ok:
        return
    dmin = min(r.metrics["distance_km"] for r in ok)
    hmin = min(r.metrics["mean_difficulty"] for r in ok)
    smax = max(r.metrics["science_score"] for r in ok)
    emin = min(r.metrics["cumulative_elev_change_m"] for r in ok)
    for r in ok:
        m, pros, cons = r.metrics, [], []
        if m["distance_km"] <= dmin * 1.03:
            pros.append("Shortest (or near-shortest) distance of the options.")
        elif m["distance_km"] > dmin * 1.25:
            cons.append(f"{100 * (m['distance_km'] / dmin - 1):.0f}% longer than the shortest option.")
        if m["mean_difficulty"] <= hmin * 1.05:
            pros.append("Lowest average terrain difficulty.")
        elif m["mean_difficulty"] > hmin * 1.3:
            cons.append("Noticeably rougher / steeper terrain on average.")
        if m["pct_above_caution"] < 1:
            pros.append(f"Under 1% of the distance on slopes above {config.ASSUMPTIONS['caution_slope_deg']:.0f} deg.")
        elif m["pct_above_caution"] > 5:
            cons.append(f"{m['pct_above_caution']:.0f}% of the distance on slopes above "
                        f"{config.ASSUMPTIONS['caution_slope_deg']:.0f} deg.")
        if m["science_score"] >= smax * 0.97:
            pros.append(f"Highest science value ({m['pois_visited']} points of interest within "
                        f"{config.ASSUMPTIONS['poi_visit_radius_m']:.0f} m).")
        elif m["science_score"] < smax * 0.7:
            cons.append("Passes fewer scientifically valuable sites.")
        if m["cumulative_elev_change_m"] <= emin * 1.05:
            pros.append("Least cumulative climbing and descending.")
        if m["max_slope_deg"] > config.ASSUMPTIONS["caution_slope_deg"]:
            cons.append(f"Briefly crosses slopes up to {m['max_slope_deg']:.0f} deg (DTM ~59 m sampling; "
                        "local slopes can be steeper).")
        r.pros, r.cons = pros or ["Balanced compromise between the other options."], cons or ["No major drawbacks under the current model."]


def rank_routes(routes: list[Route], user: Weights, max_route_km: float | None = None) -> list[tuple[Route, float]]:
    """Rank routes (0-100) with the Mission Planner's scoring model, using the user's priorities as weights."""
    from modules import mission_planner as mp   # local import: mission_planner must not depend on this module
    ev = mp.evaluate_routes(routes, mp.ScoringModel(user.science, user.safety, user.distance),
                            mp.MissionConstraints(max_route_km=max_route_km,
                                                  max_slope_deg=config.ASSUMPTIONS["max_traversable_slope_deg"]))
    return [(c.route, c.score) for c in ev.ranked]


def plan_all(planner: RoutePlanner, start_ll, end_ll, user: Weights,
             strategy_strength: float = 0.7, include_custom: bool = True) -> list[Route]:
    routes = []
    for key, spec in config.STRATEGIES.items():
        w = blend_weights(spec["profile"], user, strategy_strength)
        if key == "C":
            r = planner.plan_science(key, spec["name"], spec["color"], w, start_ll, end_ll)
        else:
            r = planner.plan_point_to_point(key, spec["name"], spec["color"], w, start_ll, end_ll)
        routes.append(planner.evaluate(r, start_ll, end_ll))
    if include_custom:
        w = user.normalised()
        r = planner.plan_point_to_point("D", config.CUSTOM_ROUTE["name"], config.CUSTOM_ROUTE["color"],
                                        w, start_ll, end_ll)
        routes.append(planner.evaluate(r, start_ll, end_ll))
    annotate_tradeoffs(routes)
    return routes
