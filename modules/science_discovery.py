"""Science discovery: points of interest, science-value surface, location
inspection and the optional AI explanation.

Strict separation is kept between
  * observations   - reported measurements with a source,
  * interpretations - published scientific interpretations,
  * MARSCOPE-derived quantities - computed here from the data.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import config
from modules.data_manager import MarsDataset, haversine_m
from modules.terrain_analysis import TerrainModel, point_summary


# ---------------------------------------------------------------------------
# Points of interest
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_poi_catalog() -> dict:
    with open(config.POI_FILE, encoding="utf-8") as fh:
        catalog = json.load(fh)
    for poi in catalog["pois"]:
        for key in ("id", "name", "lat", "lon", "priority"):
            if key not in poi:
                raise ValueError(f"POI missing '{key}': {poi}")
        if not 0 <= poi["priority"] <= 1:
            raise ValueError(f"POI priority out of range: {poi['id']}")
    return catalog


def pois() -> list[dict]:
    return load_poi_catalog()["pois"]


def poi_by_id(poi_id: str) -> dict | None:
    return next((p for p in pois() if p["id"] == poi_id), None)


def reference_text(key: str) -> str:
    if key == "MARSCOPE":
        return "MARSCOPE analysis of USGS CTX DTM / THEMIS data (derived, unpublished)."
    return load_poi_catalog()["references"].get(key, key)


# Literature-reported mineral zones, drawn as approximate circles around POIs.
def mineral_zones() -> list[dict]:
    zones = []
    for p in pois():
        orbital = [m for m in p.get("minerals_reported", []) if "orbital" in m.lower() or "crism" in m.lower()]
        rover = [m for m in p.get("minerals_reported", []) if "rover" in m.lower()]
        if orbital or rover:
            refs = sorted({o["source"] for o in p["observations"]})
            zones.append({"poi": p["id"], "name": p["name"], "lat": p["lat"], "lon": p["lon"],
                          "radius_m": 1200 if orbital else 500,
                          "orbital": orbital, "rover": rover, "refs": refs})
    return zones


# ---------------------------------------------------------------------------
# Science-value surface (DERIVED)
# ---------------------------------------------------------------------------
@dataclass
class ScienceModel:
    value: np.ndarray          # fine grid, 0..1
    components: dict           # name -> fine-grid array (0..1)
    weights: dict


def build_science_model(ds: MarsDataset, tm: TerrainModel) -> ScienceModel:
    """science_value = 0.65 * POI proximity + 0.20 * exposure + 0.15 * thermal

    * POI proximity: max over POIs of priority * exp(-(d / 1.5 km)^2) - places
      near published targets score highest.
    * Exposure: elevated benches and moderate slopes (terrain classes 1-2)
      expose stratigraphy; scarps (class 3) are informative but not drivable,
      so they get half weight.
    * Thermal: THEMIS night-IR brightness (normalised 2-98th percentile).
      Warmer at night = higher thermal inertia = more exposed rock/indurated
      material and less dust. Qualitative only (8-bit mosaic, not calibrated).
    """
    grid = ds.fine_grid
    rows, cols = np.indices(grid.shape)
    lat = grid.lat_max - (rows + 0.5) * grid.res
    lon = grid.lon_min + (cols + 0.5) * grid.res
    prox = np.zeros(grid.shape, np.float32)
    for p in pois():
        d = haversine_m(lat, lon, p["lat"], p["lon"])
        prox = np.maximum(prox, p["priority"] * np.exp(-(d / 1500.0) ** 2))

    exposure = np.select([tm.terrain_class == 1, tm.terrain_class == 2, tm.terrain_class == 3],
                         [1.0, 0.6, 0.5], default=0.1).astype(np.float32)

    if ds.themis_night is not None:
        n = ds.themis_night.astype(np.float32)
        valid = n > 0
        lo, hi = np.percentile(n[valid], [2, 98]) if valid.any() else (0, 1)
        thermal = np.clip((n - lo) / max(hi - lo, 1), 0, 1)
        thermal[~valid] = 0.5
    else:
        thermal = np.full(grid.shape, 0.5, np.float32)

    weights = {"poi_proximity": 0.65, "exposure": 0.20, "thermal_inertia_proxy": 0.15}
    value = (weights["poi_proximity"] * prox + weights["exposure"] * exposure
             + weights["thermal_inertia_proxy"] * thermal)
    value = (value - value.min()) / max(value.max() - value.min(), 1e-6)
    return ScienceModel(value.astype(np.float32),
                        {"poi_proximity": prox, "exposure": exposure, "thermal_inertia_proxy": thermal},
                        weights)


# ---------------------------------------------------------------------------
# Location inspector
# ---------------------------------------------------------------------------
def nearby_pois(lat: float, lon: float, radius_m: float = 3000) -> list[tuple[dict, float]]:
    out = [(p, float(haversine_m(lat, lon, p["lat"], p["lon"]))) for p in pois()]
    return sorted([o for o in out if o[1] <= radius_m], key=lambda t: t[1])


def thermal_percentiles(ds: MarsDataset, lat: float, lon: float) -> dict:
    r, c = ds.fine_grid.lonlat_to_rowcol(lat, lon)
    out = {}
    for key in ("themis_day", "themis_night"):
        arr = getattr(ds, key)
        if arr is None or arr[r, c] == 0:
            out[key] = None
            continue
        valid = arr[arr > 0]
        out[key] = float((valid < arr[r, c]).mean() * 100)
    return out


def inspect_location(ds: MarsDataset, tm: TerrainModel, sm: ScienceModel,
                     lat: float, lon: float) -> dict:
    """Assemble everything known about a location, tagged by evidence type."""
    terrain = point_summary(tm, lat, lon)
    r, c = ds.fine_grid.lonlat_to_rowcol(lat, lon)
    thermal = thermal_percentiles(ds, lat, lon)
    near = nearby_pois(lat, lon)
    measured = bool(ds.dtm_coverage[r, c])

    derived = [
        f"Elevation {terrain['elevation_m']:.0f} m relative to the Mars aeroid "
        f"({'CTX DTM measurement' if measured else 'gap-filled, no DTM coverage'}).",
        f"Slope {terrain['slope_deg']:.1f} deg (max {terrain['max_slope_1km_deg']:.1f} deg within ~1 km); "
        f"local relief {terrain['local_relief_1km_m']:.0f} m in a ~1 km box.",
        f"Terrain class (DTM-derived physiography): {terrain['terrain_class']}.",
        f"Terrain difficulty index {terrain['difficulty']:.2f} (0 easy - 1 hardest); "
        f"{'traversable' if terrain['passable'] else 'exceeds the slope limit'} under MARSCOPE assumptions.",
        f"Science-value score {sm.value[r, c]:.2f} (MARSCOPE weighting, see Methods).",
    ]
    if thermal.get("themis_night") is not None:
        derived.append(
            f"THEMIS night-IR brightness at the {thermal['themis_night']:.0f}th percentile of the region "
            f"(qualitative thermal-inertia proxy; higher suggests rockier / less dusty ground).")

    observations, interpretations, water, minerals, objectives = [], [], [], [], []
    for p, d in near[:3]:
        tag = f"{p['name']} ({d / 1000:.1f} km away)"
        observations += [{"text": o["text"], "source": o["source"], "poi": tag} for o in p["observations"]]
        interpretations += [{"text": i["text"], "source": i["source"], "poi": tag} for i in p["interpretations"]]
        if p.get("water_evidence"):
            water.append({"text": p["water_evidence"], "poi": tag})
        minerals += [{"text": m, "poi": tag} for m in p.get("minerals_reported", [])]
        objectives += p.get("objectives", [])

    if not near:
        objectives = _generic_objectives(terrain, thermal)
    why = _why_interesting(terrain, thermal, near, sm.value[r, c])
    return {"lat": lat, "lon": lon, "terrain": terrain, "thermal": thermal,
            "science_value": float(sm.value[r, c]), "dtm_measured": measured,
            "derived": derived, "observations": observations,
            "interpretations": interpretations, "water_evidence": water,
            "minerals": minerals, "objectives": list(dict.fromkeys(objectives)),
            "why": why, "nearby": [(p["name"], d, p["id"]) for p, d in near]}


def _generic_objectives(terrain: dict, thermal: dict) -> list[str]:
    obj = ["Acquire high-resolution imaging to characterise surface texture and layering"]
    if terrain["terrain_class"].startswith("Elevated"):
        obj.append("Look for exposed layering on bench margins (possible sedimentary stratigraphy)")
    if terrain["terrain_class"].startswith("Steep"):
        obj.append("Remote sensing of exposed section from a safe standoff distance")
    if (thermal.get("themis_night") or 0) > 75:
        obj.append("Investigate warm-at-night (high thermal inertia) material: possible bedrock exposure")
    obj.append("No published mineral detections used here - targeted spectroscopy would be needed")
    return obj


def _why_interesting(terrain: dict, thermal: dict, near: list, value: float) -> list[str]:
    why = []
    if near:
        p, d = near[0]
        if d < 100:
            why.append(f"Catalogued target: {p['category']}. " + (p.get("water_evidence") or ""))
            others = [n for n, _ in near[1:3]]
            if others:
                why.append("Close to other targets: " + ", ".join(
                    f"{o['name']} ({dd / 1000:.1f} km)" for o, dd in near[1:3]) + ".")
        else:
            why.append(f"Within {d / 1000:.1f} km of '{p['name']}', a published target ({p['category']}).")
    if terrain["terrain_class"].startswith("Elevated"):
        why.append("Sits on a locally elevated bench - such landforms can preserve sedimentary layers.")
    if (thermal.get("themis_night") or 0) > 75:
        why.append("Relatively warm at night in THEMIS IR, suggesting rock exposure rather than dust.")
    if value < 0.3 and not near:
        why.append("Low MARSCOPE science score: no published targets nearby and little exposed relief.")
    return why or ["Moderate interest: general context for the crater-floor and fan environment."]


# ---------------------------------------------------------------------------
# Optional AI explanation (Anthropic Claude)
# ---------------------------------------------------------------------------
AI_SYSTEM_PROMPT = """You are the science communicator inside MARSCOPE, a Mars mission-planning tool for Jezero Crater.
You receive a JSON fact sheet. Explain it to a non-specialist mission planner in clear language.

Rules:
- Use ONLY the facts in the fact sheet. Do not add mineral detections, discoveries, measurements or NASA findings that are not in it. If something is unknown, say it is unknown.
- Keep three clearly headed sections: "What the data show" (observations and MARSCOPE-derived numbers), "What scientists think it means" (published interpretations, stated as interpretations), and "Why go there" (investigation objectives).
- Cite sources inline using the reference keys given in the sheet, e.g. [Horgan2020] or [MARSCOPE-derived].
- Label MARSCOPE-derived quantities as derived, and note that POI positions are approximate.
- Be concise: at most about 250 words."""


def ai_available() -> bool:
    from modules import ai_service
    return ai_service.status()[0]


def build_fact_sheet(info: dict) -> dict:
    refs = sorted({o["source"] for o in info["observations"] + info["interpretations"]})
    return {
        "location": {"lat": round(info["lat"], 4), "lon_east": round(info["lon"], 4)},
        "marscope_derived": info["derived"],
        "observations": [f"{o['text']} [{o['source']}] (near {o['poi']})" for o in info["observations"]],
        "interpretations": [f"{i['text']} [{i['source']}] (near {i['poi']})" for i in info["interpretations"]],
        "reported_minerals": [f"{m['text']} (near {m['poi']})" for m in info["minerals"]],
        "water_related_evidence": [f"{w['text']} (near {w['poi']})" for w in info["water_evidence"]],
        "suggested_objectives": info["objectives"],
        "references": {k: reference_text(k) for k in refs},
        "caveats": ["POI positions are approximate (~0.5-1 km).",
                    "THEMIS values are 8-bit mosaic brightness, not calibrated temperatures.",
                    "Terrain classes are DTM-derived physiography, not a geologic map."],
    }


def ai_explain(info: dict) -> str:
    """Return an AI explanation, or raise RuntimeError with a readable message."""
    from modules import ai_service
    sheet = build_fact_sheet(info)
    return ai_service.chat(AI_SYSTEM_PROMPT,
                           [{"role": "user", "content": "Fact sheet:\n" + json.dumps(sheet, indent=2)}])
