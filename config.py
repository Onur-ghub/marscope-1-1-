"""MARSCOPE central configuration.

Everything that defines *where* we work, *which* data we use and *what
assumptions* the planner makes lives here, so that it can be reviewed in
one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"          # live downloads land here (git-ignored)
BUNDLED_DIR = DATA_DIR / "bundled"      # snapshot of real data shipped with the repo
POI_FILE = DATA_DIR / "jezero_pois.json"
USER_GEOLOGY_DIR = DATA_DIR / "geology"  # optional: drop a USGS geologic map GeoJSON/SHP here
ASSETS_DIR = ROOT_DIR / "assets"

# ---------------------------------------------------------------------------
# Mars body constants (IAU 2015 sphere used by the USGS products below)
# ---------------------------------------------------------------------------
MARS_RADIUS_M = 3_396_190.0
MARS_GRAVITY = 3.721  # m/s^2

# ---------------------------------------------------------------------------
# Study area: Jezero Crater
# ---------------------------------------------------------------------------
JEZERO_CENTER = (18.38, 77.58)          # lat, lon (planetocentric, east-positive)
JEZERO_RIM_RADIUS_KM = 22.5             # ~45 km diameter crater
# Analysis window (lon_min, lat_min, lon_max, lat_max)
REGION_BOUNDS = (77.10, 18.05, 78.00, 18.80)

# Analysis grids (degrees per pixel).
FINE_RES_DEG = 0.001      # ~59 m: elevation, slope, thermal layers
IMAGE_RES_DEG = 0.0005    # ~30 m: CTX orthoimage for the basemap
GRAPH_RES_DEG = 0.004     # ~237 m: route-planning graph nodes
# FINE cells aggregated into one GRAPH cell (must be an integer)
AGG_FACTOR = int(round(GRAPH_RES_DEG / FINE_RES_DEG))

# ---------------------------------------------------------------------------
# Data sources (all public, verified reachable: see README "Data sources")
# ---------------------------------------------------------------------------
USGS_ARD_BUCKET = "https://astrogeo-ard.s3.us-west-2.amazonaws.com"
CTX_DTM_INDEX_URL = f"{USGS_ARD_BUCKET}/collectionindices/mro_ctx_controlled_usgs_dtms.parquet"
THEMIS_INDEX_URL = f"{USGS_ARD_BUCKET}/collectionindices/mo_themis_controlled_mosaics.parquet"
THEMIS_DAY_URL = (f"{USGS_ARD_BUCKET}/mars/mo/themis/controlled_mosaics/"
                  "mc13_syrtis_major_dir/mc13_syrtis_major_dir.tif")
THEMIS_NIGHT_URL = (f"{USGS_ARD_BUCKET}/mars/mo/themis/controlled_mosaics/"
                    "mc13_syrtis_major_nir/mc13_syrtis_major_nir.tif")
MAX_CTX_DTMS = 8           # stop mosaicking once coverage is good or this many are used
TARGET_DTM_COVERAGE = 0.985

# Browser-side tile services (loaded by the user's browser, not by Python).
# NASA Solar System Treks WMTS, equirectangular (EPSG:4326-style) tile matrix.
TREK_BASE = "https://trek.nasa.gov/tiles/Mars/EQ"
TREK_LAYERS = {
    "NASA Trek: MOLA colour shaded relief (463 m)": (
        f"{TREK_BASE}/Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0/default/default028mm/{{z}}/{{y}}/{{x}}.jpg",
        "NASA/JPL-Caltech/GSFC MGS MOLA; NASA Solar System Treks"),
    "NASA Trek: Viking MDIM 2.1 colour mosaic (232 m)": (
        f"{TREK_BASE}/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{{z}}/{{y}}/{{x}}.jpg",
        "NASA/USGS Viking MDIM 2.1; NASA Solar System Treks"),
}

# ---------------------------------------------------------------------------
# Network behaviour
# ---------------------------------------------------------------------------
HTTP_TIMEOUT_S = 20
OFFLINE_MODE = os.environ.get("MARSCOPE_OFFLINE", "0") == "1"

# ---------------------------------------------------------------------------
# Traverse model ASSUMPTIONS (explicit, editable, shown in the UI)
# ---------------------------------------------------------------------------
ASSUMPTIONS = {
    "max_traversable_slope_deg": 25.0,   # cells steeper than this are impassable
    "caution_slope_deg": 15.0,           # slopes above this count as "hazardous"
    # Planning speed used ONLY for an indicative duration range; it is not a
    # capability claim. Perseverance's demonstrated autonomous drives are of
    # order 100-300 m per sol; we show a range rather than a single number.
    "indicative_speed_m_per_sol": (100.0, 300.0),
    "poi_visit_radius_m": 600.0,         # a POI counts as "visited" within this distance
    "science_detour_budget": 1.6,        # science route may be up to 1.6x the shortest-path cost
}

# Route strategy weight profiles: (distance, safety, science). They are blended
# with the user's mission-priority sliders before optimisation.
STRATEGIES = {
    "A": {"name": "Safest", "color": "#7FC4B8", "profile": (0.15, 0.80, 0.05)},
    "B": {"name": "Shortest", "color": "#92ABD6", "profile": (0.95, 0.05, 0.00)},
    "C": {"name": "Science-focused", "color": "#E0C390", "profile": (0.20, 0.25, 0.55)},
}
CUSTOM_ROUTE = {"name": "Mission-priority (your weights)", "color": "#B8A6E3"}

# ---------------------------------------------------------------------------
# Optional AI explanation (Anthropic Claude). Never hardcode keys.
# ---------------------------------------------------------------------------
ANTHROPIC_MODEL = os.environ.get("MARSCOPE_AI_MODEL", "claude-opus-5-5")

# Demo scenario: start at the Perseverance landing site, cross the delta to Bright Angel
# (Neretva Vallis), the site of the Cheyava Falls rock.
DEMO_START_POI = "octavia_butler_landing"
DEMO_DESTINATION_POI = "bright_angel"

# ---------------------------------------------------------------------------
# Mission Control simulation (every value derived from these is labelled SIMULATED)
# ---------------------------------------------------------------------------
ENERGY_MODEL = {
    "rover_mass_kg": 1025.0,          # Perseverance mass (NASA public figure); stands in for a crewed rover
    "gravity_m_s2": 3.71,             # Mars surface gravity
    "rolling_resistance": 0.10,       # ASSUMPTION: loose regolith, typical range 0.05-0.20
    "drive_efficiency": 0.30,         # ASSUMPTION: battery-to-wheel efficiency
}
SIM_PACE_M_PER_SOL = 200.0            # ASSUMPTION: midpoint of the indicative 100-300 m/sol pace
