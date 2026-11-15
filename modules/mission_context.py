"""Mission context management and scientific data retrieval for the Astronaut Assistant.

``build_context`` gathers the current mission state (start, destination, route evaluation,
destination science) from the existing modules. ``retrieve`` selects the parts relevant to a
question, so the language model only ever reasons over MARSCOPE data it is given.
"""
from __future__ import annotations

import re

import config
from modules import mission_data as md
from modules import mission_planner as mp
from modules import science_discovery as sd


def _poi_record(p: dict) -> dict:
    return {
        "id": p["id"], "name": p["name"], "category": p["category"],
        "lat": p["lat"], "lon_east": p["lon"], "position_note": "approximate (~0.5-1 km)",
        "marscope_priority": p["priority"],
        "source_type": "published literature" if p.get("origin") == "published" else "MARSCOPE identification",
        "observations": [f"{o['text']} [{o['source']}]" for o in p.get("observations", [])],
        "interpretations": [f"{i['text']} [{i['source']}]" for i in p.get("interpretations", [])],
        "reported_minerals": p.get("minerals_reported", []),
        "water_evidence": p.get("water_evidence") or None,
        "suggested_objectives": p.get("objectives", []),
    }


def _candidate_record(c: mp.Candidate) -> dict:
    m = c.raw
    rec = {"route": c.key, "strategy": c.name, "eligible": c.eligible}
    if not c.eligible:
        rec["excluded_because"] = c.exclusions
        return rec
    if c.duplicate_of:
        rec["same_path_as_route"] = c.duplicate_of
    rec.update({
        "rank": c.rank, "overall_score_0_100": round(c.score, 1) if c.score is not None else None,
        "criteria_0_100": {k: (round(v, 1) if v is not None else "not available") for k, v in c.criteria.items()},
        "safety_assessment": c.safety_level,
        "distance_km": round(m["distance_km"], 2),
        "mean_terrain_difficulty_0_1": round(m["mean_difficulty"], 3),
        "max_slope_deg": round(m["max_slope_deg"], 1),
        "pct_distance_above_caution_slope": round(m["pct_above_caution"], 1),
        "cumulative_elevation_change_m": round(m["cumulative_elev_change_m"]),
        "points_of_interest_passed": [v[1] for v in m.get("visited", [])],
        "indicative_sols": [round(x) for x in m.get("sols_range", (0, 0))],
        "warnings": c.warnings, "pros": getattr(c.route, "pros", []), "cons": getattr(c.route, "cons", []),
    })
    return rec


def build_context(ds, tm, sm, start, destination, evaluation: mp.MissionEvaluation | None,
                  selected_route: str | None, waypoints: list | None = None) -> dict:
    """Snapshot of everything the assistant may use. All values come from MARSCOPE modules."""
    ctx = {
        "study_area": "Jezero crater, Mars (planetocentric latitude, east longitude)",
        "terrain_data": ("SYNTHETIC demonstration terrain - numbers are illustrative" if ds.is_demonstration
                         else f"Real USGS data ({ds.status}): CTX DTM ~{ds.fine_grid.cell_size_m()[0]:.0f} m, "
                              "CTX imagery, THEMIS IR mosaics"),
        "astronaut_position": None, "destination": None, "route_evaluation": None,
        "destination_science": None, "perseverance_visits_near_destination": None,
        "uncertainties": mp.uncertainties(),
        "assumptions": {"max_traversable_slope_deg": config.ASSUMPTIONS["max_traversable_slope_deg"],
                        "caution_slope_deg": config.ASSUMPTIONS["caution_slope_deg"],
                        "indicative_speed_m_per_sol": config.ASSUMPTIONS["indicative_speed_m_per_sol"]},
    }
    if start:
        ctx["astronaut_position"] = {"name": start[2], "lat": round(start[0], 4), "lon_east": round(start[1], 4),
                                     "note": "Planned mission start; MARSCOPE has no live position tracking."}
    if destination:
        poi = next((p for p in sd.pois() if p["name"] == destination[2]), None)
        ctx["destination"] = {"name": destination[2] if destination[2] != "Map point" else "Custom map point",
                              "lat": round(destination[0], 4), "lon_east": round(destination[1], 4),
                              "catalogued_point_of_interest": poi["id"] if poi else None}
        info = sd.inspect_location(ds, tm, sm, destination[0], destination[1])
        ctx["destination_science"] = sd.build_fact_sheet(info)
        ctx["destination_science"]["why_interesting_marscope"] = info["why"]
        if waypoints:
            visits = md.visits_near(destination[0], destination[1], waypoints,
                                    radius_m=config.ASSUMPTIONS["poi_visit_radius_m"])
            ctx["perseverance_visits_near_destination"] = (
                {"sols": [v["sol"] for v in visits][:40], "source": "NASA/JPL MMGIS traverse"} if visits
                else "Perseverance has not stopped within the visit radius (NASA/JPL MMGIS traverse).")
    if evaluation is not None:
        rec = evaluation.recommended
        ctx["route_evaluation"] = {
            "scoring_model_weights": evaluation.model.as_dict(),
            "constraints": {"max_route_km": evaluation.constraints.max_route_km or "none set",
                            "max_traversable_slope_deg": evaluation.constraints.max_slope_deg},
            "safety_index_definition": "100*(1-(0.5*mean difficulty + 0.3*p90 difficulty + "
                                       "0.2*min(%distance above caution slope/20,1))); MARSCOPE-derived, "
                                       "a relative hazard measure, never a guarantee of safety",
            "distance_criterion_definition": "100 * shortest eligible distance / this route's distance",
            "science_criterion_definition": "MARSCOPE science score: 40*mean science value along route + "
                                            "60*share of POI priority passed within the visit radius",
            "recommended_route": rec.key if rec else None,
            "route_shown_on_map": selected_route,
            "deterministic_explanation": mp.explain_recommendation(evaluation),
            "candidates": [_candidate_record(c) for c in evaluation.candidates],
        }
    return ctx


_ROUTE_WORDS = re.compile(r"\b(route|path|drive|alternative|stop|along|traverse|safer|shorter|option)", re.I)


def retrieve(question: str, ctx: dict) -> dict:
    """Context + catalogue entries relevant to the question (named POIs, route stops, POI index)."""
    q = question.lower()
    out = dict(ctx)
    pois = sd.pois()
    named = [p for p in pois if any(len(w) >= 5 and w.lower() in q
                                    for w in re.split(r"[\s/()\-,]+", p["name"]))]
    if _ROUTE_WORDS.search(question) and ctx.get("route_evaluation"):
        rec = ctx["route_evaluation"]["recommended_route"]
        cand = next((c for c in ctx["route_evaluation"]["candidates"] if c.get("route") == rec), None)
        stops = set(cand.get("points_of_interest_passed", [])) if cand else set()
        named += [p for p in pois if p["name"] in stops and p not in named]
    out["relevant_catalogue_entries"] = [_poi_record(p) for p in named[:6]]
    out["all_points_of_interest_index"] = [
        {"name": p["name"], "category": p["category"], "marscope_priority": p["priority"],
         "lat": p["lat"], "lon_east": p["lon"]} for p in pois]
    refs = set()
    for p in named:
        refs |= {o["source"] for o in p.get("observations", []) + p.get("interpretations", [])}
    out.setdefault("destination_science", None)
    if refs:
        out["references"] = {k: sd.reference_text(k) for k in sorted(refs)}
    return out


def signature(start, destination, route_sig, selected_route, max_route_km) -> str:
    """Short key identifying the mission state (used to cache answers and reset suggestions)."""
    return repr((start[:2] if start else None, destination[:2] if destination else None, route_sig,
                 selected_route, max_route_km))
