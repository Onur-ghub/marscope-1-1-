"""AI Mission Planner - deterministic route evaluation (no language model here).

Destination -> route candidates (route_optimizer) -> numerical evaluation (this module)
-> ranked recommendation -> optional AI explanation (assistant.py).

Every number used comes from ``Route.metrics`` computed by the existing planner from the
CTX DTM, the science-value surface and the POI catalogue. Nothing is estimated here; a
criterion that a candidate lacks is reported as missing and the remaining weights are
renormalised for that candidate (and the candidate is flagged as incompletely assessed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config


@dataclass(frozen=True)
class ScoringModel:
    """Criterion weights. Defaults follow the mission brief: science 50 %, safety 30 %, distance 20 %."""
    science: float = 0.50
    safety: float = 0.30
    distance: float = 0.20

    def normalised(self) -> "ScoringModel":
        s = self.science + self.safety + self.distance
        if s <= 0:
            return ScoringModel()
        return ScoringModel(self.science / s, self.safety / s, self.distance / s)

    def as_dict(self) -> dict:
        n = self.normalised()
        return {"science": n.science, "safety": n.safety, "distance": n.distance}


@dataclass(frozen=True)
class MissionConstraints:
    max_route_km: float | None = None          # optional hard limit on route length
    max_slope_deg: float | None = None          # the planner's traversability limit (for reporting)


@dataclass
class Candidate:
    key: str
    name: str
    color: str
    eligible: bool
    exclusions: list = field(default_factory=list)
    criteria: dict = field(default_factory=dict)   # name -> 0-100 or None
    raw: dict = field(default_factory=dict)        # the underlying metrics, for explanations
    missing: list = field(default_factory=list)
    score: float | None = None
    rank: int | None = None
    safety_level: str = "unknown"
    warnings: list = field(default_factory=list)
    duplicate_of: str | None = None                # same path as another candidate
    route: object = None


@dataclass
class MissionEvaluation:
    model: ScoringModel
    constraints: MissionConstraints
    candidates: list

    @property
    def ranked(self) -> list[Candidate]:
        return sorted([c for c in self.candidates if c.rank is not None], key=lambda c: c.rank)

    @property
    def recommended(self) -> Candidate | None:
        r = self.ranked
        return r[0] if r else None

    @property
    def excluded(self) -> list[Candidate]:
        return [c for c in self.candidates if not c.eligible]


# ---------------------------------------------------------------------------
# Criterion definitions (transparent, documented in the Data & Methods tab)
# ---------------------------------------------------------------------------
def safety_index(m: dict) -> float | None:
    """0-100, higher = lower terrain hazard along the route (MARSCOPE-derived, not a safety guarantee).

    100 * (1 - (0.5 * mean difficulty + 0.3 * 90th-percentile difficulty
                + 0.2 * min(share of distance above the caution slope / 20 %, 1)))
    """
    need = ("mean_difficulty", "p90_difficulty", "pct_above_caution")
    if any(m.get(k) is None for k in need):
        return None
    hazard = (0.5 * m["mean_difficulty"] + 0.3 * m["p90_difficulty"]
              + 0.2 * min(m["pct_above_caution"] / 20.0, 1.0))
    return 100.0 * max(0.0, 1.0 - hazard)


def safety_level(index: float | None) -> str:
    if index is None:
        return "unknown"
    if index >= 75:
        return "lower hazard"
    if index >= 55:
        return "moderate hazard"
    return "elevated hazard"


def evaluate_routes(routes: list, model: ScoringModel | None = None,
                    constraints: MissionConstraints | None = None) -> MissionEvaluation:
    model = (model or ScoringModel()).normalised()
    constraints = constraints or MissionConstraints(
        max_slope_deg=config.ASSUMPTIONS["max_traversable_slope_deg"])
    cands = []
    for r in routes or []:
        m = r.metrics or {}
        c = Candidate(key=r.key, name=r.name, color=r.color, eligible=True, route=r, raw=dict(m))
        if not r.found or not m.get("distance_km"):
            c.eligible = False
            c.exclusions.append(r.message or "No traversable path was found.")
        elif constraints.max_route_km and m["distance_km"] > constraints.max_route_km:
            c.eligible = False
            c.exclusions.append(f"{m['distance_km']:.2f} km exceeds the {constraints.max_route_km:g} km "
                                "maximum route length.")
        if constraints.max_slope_deg and m.get("max_slope_deg", 0) > constraints.max_slope_deg:
            # The planner already rejects grid cells over the limit; finer DTM sampling can still
            # show short steeper stretches, so this is a warning, not an exclusion.
            c.warnings.append(f"Fine-scale DTM sampling shows slopes up to {m['max_slope_deg']:.0f}° "
                              f"(limit {constraints.max_slope_deg:.0f}°) over short stretches.")
        twin = next((o for o in cands if o.eligible and c.eligible and o.route.cells == r.cells), None)
        if twin is not None:
            c.duplicate_of = twin.key
        cands.append(c)

    ok = [c for c in cands if c.eligible]
    dmin = min((c.raw["distance_km"] for c in ok), default=None)
    for c in ok:
        m = c.raw
        c.criteria = {
            "science": float(m["science_score"]) if m.get("science_score") is not None else None,
            "safety": safety_index(m),
            "distance": 100.0 * dmin / m["distance_km"] if dmin else None,
        }
        c.missing = [k for k, v in c.criteria.items() if v is None]
        w = {k: getattr(model, k) for k, v in c.criteria.items() if v is not None}
        total_w = sum(w.values())
        c.score = (sum(w[k] * c.criteria[k] for k in w) / total_w) if total_w else None
        c.safety_level = safety_level(c.criteria["safety"])
    for i, c in enumerate(sorted([c for c in ok if c.score is not None and not c.duplicate_of],
                                 key=lambda c: -c.score)):
        c.rank = i + 1
    return MissionEvaluation(model=model, constraints=constraints, candidates=cands)


# ---------------------------------------------------------------------------
# Plain-language explanation (deterministic; the AI only rephrases / discusses it)
# ---------------------------------------------------------------------------
CRITERION_LABEL = {"science": "scientific value", "safety": "terrain safety", "distance": "distance"}


def explain_recommendation(ev: MissionEvaluation) -> list[str]:
    rec = ev.recommended
    if rec is None:
        return ["No route satisfies the mission constraints, so nothing can be recommended."]
    w = ev.model.as_dict()
    lines = [f"Route {rec.key} ({rec.name}) has the highest overall score, {rec.score:.0f}/100, "
             f"with weights science {w['science']:.0%}, safety {w['safety']:.0%}, distance {w['distance']:.0%}."]
    others = ev.ranked[1:]
    if others:
        nxt = others[0]
        diffs = []
        for k in ("science", "safety", "distance"):
            a, b = rec.criteria.get(k), nxt.criteria.get(k)
            if a is not None and b is not None and abs(a - b) >= 1:
                diffs.append((w[k] * (a - b), k, a, b))
        diffs.sort(reverse=True)
        gains = [f"{CRITERION_LABEL[k]} ({a:.0f} vs {b:.0f})" for d, k, a, b in diffs if d > 0]
        losses = [f"{CRITERION_LABEL[k]} ({a:.0f} vs {b:.0f})" for d, k, a, b in diffs if d < 0]
        txt = f"Compared with the runner-up, Route {nxt.key} ({nxt.score:.0f}/100), it is stronger on " + (
            ", ".join(gains) if gains else "no single criterion") + "."
        if losses:
            txt += " It gives up some " + ", ".join(losses) + "."
        lines.append(txt)
    m = rec.raw
    lines.append(f"It is {m['distance_km']:.1f} km long, passes {m.get('pois_visited', 0)} catalogued point(s) of "
                 f"interest, and is assessed as {rec.safety_level} by MARSCOPE's terrain model "
                 f"({m['pct_above_caution']:.1f}% of the distance above "
                 f"{config.ASSUMPTIONS['caution_slope_deg']:.0f}° slope).")
    if rec.missing:
        lines.append("Not assessed for: " + ", ".join(CRITERION_LABEL[k] for k in rec.missing) +
                     " (data unavailable); its score uses the remaining criteria only.")
    twins = [c for c in ev.candidates if c.duplicate_of]
    if twins:
        lines.append("; ".join(f"Route {c.key} follows the same path as Route {c.duplicate_of}" for c in twins) + ".")
    if ev.excluded:
        lines.append("Excluded: " + "; ".join(f"Route {c.key} - {c.exclusions[0]}" for c in ev.excluded) + ".")
    lines.append("No route is guaranteed safe: the terrain model (~59 m DTM, ~237 m planning grid) cannot "
                 "resolve rocks, small scarps or loose sand.")
    return lines


def compare(ev: MissionEvaluation, a: str, b: str) -> list[str]:
    """Plain-language comparison of two candidates, from the evaluation numbers only."""
    ca = next((c for c in ev.candidates if c.key == a), None)
    cb = next((c for c in ev.candidates if c.key == b), None)
    if ca is None or cb is None:
        return [f"Route {a if ca is None else b} is not among the current candidates."]
    for c in (ca, cb):
        if not c.eligible:
            return [f"Route {c.key} is excluded: {c.exclusions[0]}"]
        if c.duplicate_of:
            return [f"Route {c.key} follows exactly the same path as Route {c.duplicate_of}."]
    if ca.score < cb.score:
        ca, cb = cb, ca
    w = ev.model.as_dict()
    out = [f"Route {ca.key} ({ca.name}) scores {ca.score:.0f}/100 and Route {cb.key} ({cb.name}) {cb.score:.0f}/100."]
    for k in ("science", "safety", "distance"):
        x, y = ca.criteria.get(k), cb.criteria.get(k)
        if x is None or y is None:
            out.append(f"- {CRITERION_LABEL[k]}: not available for both routes.")
            continue
        better = ca.key if x > y else cb.key if y > x else "neither"
        out.append(f"- {CRITERION_LABEL[k]} (weight {w[k]:.0%}): {x:.0f} vs {y:.0f}, better: Route {better}"
                   if better != "neither" else f"- {CRITERION_LABEL[k]}: equal ({x:.0f}).")
    ma, mb = ca.raw, cb.raw
    out.append(f"- In plain terms: {ma['distance_km']:.1f} km vs {mb['distance_km']:.1f} km; "
               f"{ma.get('pois_visited', 0)} vs {mb.get('pois_visited', 0)} points of interest passed; "
               f"{ma['pct_above_caution']:.1f}% vs {mb['pct_above_caution']:.1f}% of the distance on slopes above "
               f"{config.ASSUMPTIONS['caution_slope_deg']:.0f}°.")
    return out


def uncertainties() -> list[str]:
    return [
        "Terrain is sampled at ~59 m and routes are planned on a ~237 m grid, so rover-scale hazards "
        "(boulders, small scarps, sand ripples) are not resolved.",
        "Point-of-interest positions are approximate (about 0.5-1 km).",
        "Science value combines published targets with a MARSCOPE weighting; priorities are expert judgement, "
        "not NASA rankings.",
        "THEMIS layers are stretched brightness, not calibrated temperature or thermal inertia.",
        "Drive durations use an assumed pace and ignore science stops, power and communications.",
    ]


# Alternative mission scenarios: the same candidate routes re-scored under other weightings.
SCENARIOS = {
    "Balanced (mission brief)": ScoringModel(0.50, 0.30, 0.20),
    "Science first": ScoringModel(0.70, 0.20, 0.10),
    "Safety first": ScoringModel(0.20, 0.70, 0.10),
    "Shortest drive": ScoringModel(0.20, 0.20, 0.60),
}


def scenario_table(routes: list, constraints: MissionConstraints | None = None) -> list[dict]:
    rows = []
    for name, model in SCENARIOS.items():
        ev = evaluate_routes(routes, model, constraints)
        rec = ev.recommended
        rows.append({"scenario": name, "weights": model.as_dict(), "recommended": rec.key if rec else None,
                     "score": rec.score if rec else None,
                     "runner_up": ev.ranked[1].key if len(ev.ranked) > 1 else None})
    return rows
