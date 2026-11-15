"""MARSCOPE Astronaut Assistant: grounded answers about the mission and Mars.

Workflow: question -> mission_context.retrieve (relevant MARSCOPE data) -> LLM reasoning over
that data (ai_service) -> grounded answer. Without an AI key, a deterministic fallback answers
mission and destination questions straight from the same data, and says plainly when a
question needs the AI service or data MARSCOPE does not have.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from modules import ai_service
from modules import mission_context as mc
from modules import mission_planner as mp

HISTORY_TURNS = 6   # previous question/answer pairs sent for follow-up questions

SYSTEM_PROMPT = """You are the MARSCOPE Astronaut Assistant, the onboard mission companion of an astronaut
exploring Jezero crater on Mars. You combine two roles: Mission Planner (routes, destinations, decisions)
and Science Assistant (geology, minerals, water history, what to investigate).

Each user turn includes MISSION DATA (JSON) produced by MARSCOPE: the astronaut's planned position, the
selected destination, a deterministic route evaluation, destination science, catalogue entries and
uncertainties. "Here", "this place" and "the destination" mean the selected destination, unless the data
contain selected_location (a point the user picked on the 3D map or in Science Discovery) and the question is
about "the selected location" or "this location": then use selected_location. visualization_guide explains
what each 3D colour mode, animation and chart shows; use it to help the user read the visualisations. "This route",
"the recommendation" and "the alternative" refer to route_evaluation.

Rules:
1. For anything about this mission, Jezero, routes, scores, terrain or catalogued sites, use ONLY the
   MISSION DATA. Never invent or recalculate scores, distances, slopes, coordinates, detections or findings.
   Quote numbers exactly as given.
2. If the data do not contain what was asked, say so plainly ("MARSCOPE's data do not include ...").
3. Keep confirmed observations separate from scientific interpretations, and say which is which.
   Cite sources with the reference keys in square brackets, e.g. [Hurowitz2025]. Label MARSCOPE-derived
   values as derived.
4. Never say or imply that a route is completely safe. Safety assessments are relative terrain-hazard
   measures from a coarse model; mention the main uncertainty when discussing safety.
5. For general Mars science that is not in the data (e.g. why the atmosphere is thin), you may answer from
   well-established knowledge, but start that part with "General background (not from MARSCOPE data):"
   and avoid specific numbers you are not sure of.
6. Write for an astronaut: plain language, short paragraphs or bullets, at most about 200 words unless
   asked for detail. Lead with the answer."""


VIS_GUIDE = {
    "3d_colour_map": "A ~300 m mesh of the CTX DTM. Colour modes: Elevation (CTX DTM, observation); Slope (derived, "
                     "sand above the 15 deg caution slope, terracotta above the 25 deg limit); Terrain class (derived "
                     "DTM physiography, not a geologic map); Geological features (approximate zones around catalogued "
                     "published features, literature); Scientific interest (MARSCOPE science value 0-100, derived); "
                     "CTX image (observation). Vertical exaggeration is adjustable. Click to select a location.",
    "animations": "Reveal relief raises the terrain from flat to the chosen exaggeration; Tour layers cycles every "
                  "data layer over the same region; Compare + Sweep splits the map between two layers and moves the "
                  "split across it; Scan moves along the cross-section in the chart and in 3D; the rover playback is "
                  "SIMULATED along the planned route at the assumed pace (illustrative, not mission telemetry).",
    "cross_section": "Surface elevation along a line through the selected point at full DTM resolution (~59 m), "
                     "with the terrain class underneath and catalogued features within 800 m. No subsurface data: "
                     "layers below the surface are not shown.",
    "elevation_profile": "Elevation (m) against distance (km) along each planned route, from the CTX DTM.",
    "terrain_charts": "Area of each DTM-derived terrain class, its elevation range, the share of each route's "
                      "distance in each class, and the approximate area of each catalogued geological setting.",
    "tradeoff_radar": "Each route's distance, safety and science criteria (0-100) from the Mission Planner.",
}


@dataclass
class Reply:
    text: str
    mode: str                     # "ai" | "offline" | "error"
    note: str = ""
    sources: list = field(default_factory=list)


def suggested_questions(ctx: dict) -> list[str]:
    dest = (ctx.get("destination") or {}).get("name")
    ev = ctx.get("route_evaluation") or {}
    rec = ev.get("recommended_route")
    ranked = sorted([c for c in ev.get("candidates", []) if c.get("rank")], key=lambda c: c["rank"])
    qs = []
    sel = (ctx.get("selected_location") or {}).get("name")
    if sel:
        qs.append(f"What is scientifically interesting about the selected location ({sel.split('(')[0].strip()})?")
    if dest:
        qs.append(f"Why is {dest.split('(')[0].strip()} scientifically interesting?")
    if rec:
        qs.append(f"Why did you recommend Route {rec}?")
    if len(ranked) > 1:
        qs.append(f"Why is Route {ranked[0]['route']} better than Route {ranked[1]['route']}?")
    if dest:
        qs.append("What should I investigate when I arrive?")
    qs.append("What are the main uncertainties?" if rec else "Which destination has the greatest scientific value?")
    qs.append("How did water shape Jezero crater?")
    qs.insert(min(2, len(qs)), "How do I read the 3D colour map and its animations?")
    return qs[:7]


def answer(question: str, history: list[dict], ctx: dict, evaluation=None) -> Reply:
    data = mc.retrieve(question, ctx)
    data["visualization_guide"] = VIS_GUIDE
    data["_evaluation"] = evaluation
    ok, provider, why = ai_service.status()
    if not ok:
        r = offline_answer(question, data)
        r.note = "Answered from MARSCOPE data (AI not configured)"
        return r
    msgs = []
    for h in history[-2 * HISTORY_TURNS:]:
        if h["role"] in ("user", "assistant") and h.get("mode") != "error":
            msgs.append({"role": h["role"], "content": h["content"]})
    payload = {k: v for k, v in data.items() if not k.startswith("_")}
    msgs.append({"role": "user", "content": "MISSION DATA (JSON):\n" + json.dumps(payload, default=str)
                 + "\n\nASTRONAUT QUESTION:\n" + question})
    try:
        text = ai_service.chat(SYSTEM_PROMPT, msgs, max_tokens=2000)
        return Reply(text, "ai", note=f"{provider} · grounded in MARSCOPE mission data",
                     sources=sorted((data.get("references") or {}).keys()))
    except ai_service.AIUnavailable as exc:
        r = offline_answer(question, data)
        r.note = f"AI unavailable: {exc} Answered from MARSCOPE data instead"
        return r


# ---------------------------------------------------------------------------
# Deterministic fallback (no language model): answers only from MARSCOPE data
# ---------------------------------------------------------------------------
def _has(q, *words):
    return any(w in q for w in words)


def offline_answer(question: str, data: dict) -> Reply:
    q = question.lower()
    ev = data.get("route_evaluation")
    sci = data.get("destination_science")
    dest = (data.get("destination") or {}).get("name", "the destination")
    out = []

    rover = data.get("simulated_rover")
    sel = data.get("selected_location")
    if _has(q, "3d", "colour", "color", "legend", "animation", "playback", "cross-section", "cross section",
            "chart", "graph", "profile", "visuali", "read the map", "radar"):
        g = data.get("visualization_guide") or VIS_GUIDE
        keys = [k for k, words in (("3d_colour_map", ("3d", "colour", "color", "legend", "map")),
                                   ("animations", ("animation", "playback", "simulat", "scan", "sweep", "tour")),
                                   ("cross_section", ("section",)), ("elevation_profile", ("profile",)),
                                   ("terrain_charts", ("chart", "graph", "class")), ("tradeoff_radar", ("radar",)))
                if _has(q, *words)] or list(g)
        names = {"3d_colour_map": "3D colour map", "animations": "Animations", "cross_section": "Cross-section",
                 "elevation_profile": "Elevation profile", "terrain_charts": "Terrain charts",
                 "tradeoff_radar": "Trade-off radar"}
        out += [f"- **{names.get(k, k)}**: {g[k]}" for k in dict.fromkeys(keys)]
        return Reply(_md(out), "offline")
    if sel and _has(q, "selected", "this location", "this point"):
        out.append(f"{sel['name']} ({sel['lat']}° N, {sel['lon_east']}° E):")
        out += [f"- {w}" for w in sel.get("why_interesting_marscope", [])]
        if sel["observations"]:
            out.append("Observations (measured, published):")
            out += [f"- {o}" for o in sel["observations"][:4]]
        if sel["interpretations"]:
            out.append("Interpretations (hypotheses, not confirmed findings):")
            out += [f"- {i}" for i in sel["interpretations"][:3]]
        out.append("Derived terrain (MARSCOPE):")
        out += [f"- {d}" for d in sel["marscope_derived"][:3]]
        return Reply(_md(out), "offline", sources=sorted((sel.get("references") or {}).keys()))
    named = data.get("relevant_catalogue_entries") or []
    if named and _has(q, "what is", "tell me", "interesting", "significan", "why", "explain") and not _has(q, "route"):
        e = named[0]
        out.append(f"{e['name']} ({e['category']}, {e['source_type']}):")
        if e["observations"]:
            out.append("Observations (measured):")
            out += [f"- {o}" for o in e["observations"]]
        if e["interpretations"]:
            out.append("Interpretations (hypotheses):")
            out += [f"- {i}" for i in e["interpretations"]]
        if e["reported_minerals"]:
            out.append("Reported minerals: " + "; ".join(e["reported_minerals"]))
        return Reply(_md(out), "offline", sources=sorted((data.get("references") or {}).keys()))
    if ev and "route" in q and _has(q, "highest scien", "most scien", "best scien", "scientific value"):
        cands = sorted([c for c in ev["candidates"] if c.get("rank")],
                       key=lambda c: -(c["criteria_0_100"]["science"] if isinstance(c["criteria_0_100"]["science"],
                                                                                     (int, float)) else -1))
        if cands:
            top = cands[0]
            out.append(f"Route {top['route']} ({top['strategy']}) has the highest science criterion, "
                       f"{top['criteria_0_100']['science']}/100, passing {len(top['points_of_interest_passed'])} "
                       "catalogued target(s).")
            out += [f"- Route {c['route']}: science {c['criteria_0_100']['science']}, overall "
                    f"{c['overall_score_0_100']}, {c['distance_km']} km" for c in cands]
            if top["route"] != ev["recommended_route"]:
                out.append(f"The overall recommendation is Route {ev['recommended_route']} because safety and "
                           "distance also count in the score.")
            return Reply(_md(out), "offline")
    if _has(q, "next") and _has(q, "target", "investigat", "visit", "stop"):
        if rover and rover.get("next_target_on_route"):
            nt = rover["next_target_on_route"]
            out.append(f"Next catalogued target on Route {rover['route']}: {nt['name']}, about {nt['km_ahead']} km "
                       f"ahead of the simulated rover position ({rover['progress_pct']}% along the route).")
            entry = next((e for e in data.get("relevant_catalogue_entries", []) if e["name"] == nt["name"]), None)
            if entry is None:
                from modules import science_discovery as sd
                p_ = next((p for p in sd.pois() if p["name"] == nt["name"]), None)
                entry = mc._poi_record(p_) if p_ else None
            if entry:
                out += [f"- Objective: {o}" for o in entry["suggested_objectives"]]
                out += [f"- Observed: {o}" for o in entry["observations"][:2]]
            return Reply(_md(out), "offline")
        if rover:
            return Reply("The simulated rover has passed every catalogued target on this route; the destination is "
                         "the remaining science stop.", "offline")
    if _has(q, "route", "recommend", "alternative", "better", "option", "influenc", "decision", "why did", "trade"):
        if not ev:
            return Reply("No routes have been generated yet. Choose a destination first.", "offline")
        named = list(dict.fromkeys(re.findall(r"\broute\s+([a-d])\b", q)))
        if len(named) >= 2 and data.get("_evaluation") is not None:
            return Reply(_md(mp.compare(data["_evaluation"], named[0].upper(), named[1].upper())), "offline")
        out += ev["deterministic_explanation"]
        cands = [c for c in ev["candidates"] if c.get("rank")]
        if "second" in q and len(cands) > 1 and data.get("_evaluation") is not None:
            r = sorted(cands, key=lambda c: c["rank"])
            return Reply(_md(mp.compare(data["_evaluation"], r[0]["route"], r[1]["route"])), "offline")
        if _has(q, "alternative", "better", "option", "compare", "second") and len(cands) > 1:
            out.append("Candidates (overall score / science / safety / distance, all 0-100):")
            for c in sorted(cands, key=lambda c: c["rank"]):
                cr = c["criteria_0_100"]
                out.append(f"- Route {c['route']} {c['strategy']}: {c['overall_score_0_100']} / {cr['science']} / "
                           f"{cr['safety']} / {cr['distance']} - {c['distance_km']} km, {c['safety_assessment']}")
        if _has(q, "influenc", "information", "data", "decision"):
            w = ev["scoring_model_weights"]
            out.append(f"Inputs: route metrics from the terrain model and science surface, weighted science "
                       f"{w['science']:.0%}, safety {w['safety']:.0%}, distance {w['distance']:.0%}. "
                       f"Terrain data: {data['terrain_data']}.")
    elif _has(q, "uncertain", "risk", "safe", "hazard", "consider", "before proceeding", "danger"):
        if ev and ev.get("recommended_route"):
            rec = next(c for c in ev["candidates"] if c.get("route") == ev["recommended_route"])
            out.append(f"Route {rec['route']} is assessed as {rec['safety_assessment']} by MARSCOPE's terrain model "
                       f"(max slope {rec['max_slope_deg']}°, {rec['pct_distance_above_caution_slope']}% of the "
                       "distance above the caution slope). That is a relative measure, not a guarantee of safety.")
            out += [f"Warning: {w}" for w in rec.get("warnings", [])]
        out.append("Main uncertainties:")
        out += [f"- {u}" for u in data["uncertainties"]]
    elif _has(q, "investigat", "look for", "observ", "objective", "sample", "arrive", "evidence")and sci:
        out.append(f"Suggested objectives at {dest} (from the MARSCOPE catalogue):")
        out += [f"- {o}" for o in sci["suggested_objectives"]] or ["- None catalogued."]
        if sci["observations"]:
            out.append("What has been observed nearby:")
            out += [f"- {o}" for o in sci["observations"]]
    elif _has(q, "mineral", "composition", "clay", "carbonate", "olivine") and sci:
        out.append(f"Mineral detections reported near {dest} (published; MARSCOPE does not process CRISM spectra):")
        out += [f"- {m}" for m in sci["reported_minerals"]] or ["- None reported in MARSCOPE's catalogue for this area."]
    elif _has(q, "which destination", "greatest", "most valuable", "highest scientific"):
        idx = sorted(data["all_points_of_interest_index"], key=lambda p: -p["marscope_priority"])[:5]
        out.append("Highest MARSCOPE planning priority (expert judgement from published results, not a NASA ranking):")
        out += [f"- {p['name']}: {p['marscope_priority']:.2f} ({p['category']})" for p in idx]
    elif _has(q, "interesting", "significan", "valuable", "why go", "this place", "geolog", "water", "formed",
              "feature") and sci:
        out.append(f"{dest}:")
        out += [f"- {w}" for w in sci.get("why_interesting_marscope", [])]
        if sci["observations"]:
            out.append("Observations (measured):")
            out += [f"- {o}" for o in sci["observations"]]
        if sci["interpretations"]:
            out.append("Interpretations (what scientists think it means):")
            out += [f"- {i}" for i in sci["interpretations"]]
        if sci["water_related_evidence"]:
            out.append("Water-related evidence:")
            out += [f"- {w}" for w in sci["water_related_evidence"]]
    if not out:
        return Reply("MARSCOPE's data do not answer this directly, and general Mars questions need the AI "
                     "service (set ANTHROPIC_API_KEY). Try one of the suggested questions about the mission "
                     "or the destination.", "offline")
    refs = sorted((data.get("references") or {}).keys() | ((sci or {}).get("references") or {}).keys())
    return Reply(_md(out), "offline", sources=refs)


def _md(lines: list[str]) -> str:
    """Bullets stay together; other lines become their own paragraphs."""
    text = ""
    for ln in lines:
        text += ("\n" if ln.startswith("- ") and text else "\n\n" if text else "") + ln
    return text
