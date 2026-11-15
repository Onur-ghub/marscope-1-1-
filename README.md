# MARSCOPE — Intelligent Mars Exploration & Mission Planning for Jezero Crater

MARSCOPE is a Streamlit web application that helps mission planners answer three questions about
**Jezero Crater, Mars** (Perseverance's field site):

1. **Where should astronauts explore?** A science-value surface built from published targets, terrain
   exposure and THEMIS thermal data, plus a location inspector for any point on the map.
2. **Which route should they take?** A graph-based route optimiser (A\* and Dijkstra with a small
   orienteering step) that produces and compares **Safest**, **Shortest** and **Science-focused**
   routes, plus a fourth route that uses exactly your mission weights.
3. **What might they discover along the way?** Science cards that separate *observations*,
   *published interpretations* and *MARSCOPE-derived numbers*, with references, and an optional
   AI explanation that is only allowed to use those facts.

It runs on **real NASA/USGS data**: CTX stereo terrain models, CTX orthoimages and THEMIS infrared
mosaics from the USGS Astrogeology *Analysis Ready Data* archive. A snapshot of that data ships with the
repository (`data/bundled/`, 4 MB), so the demo works offline.

---

## Quick start

```bash
cd marscope
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The app opens on a quiet **start screen** and moves through three screens:

1. **Start screen**: only Start, Destination and ⇅ Swap, centred in the astronaut-visor view. Start
   defaults to the Perseverance landing site. Pick a site from either list or type coordinates
   (`18.47, 77.40`), then press **Show route**.
2. **Route preview**: the map fills the screen and the selected route draws itself. Then a side panel
   with four route options slides in: Safest, Shortest, Science-focused and Balanced/custom. Each shows
   estimated distance, risk, science value, stops and a one-line reason, and the panel says why the
   recommended one was chosen. **Map tools** holds the Inspect, Set start and Set destination click
   modes. Moving an endpoint re-plans the routes.
3. **Navigation (simulated)**: the map with a 2D/3D switch, and the mission panel on the right
   (Assistant, Mission, Navigation, Route, Science, System).

**Mission Guide** (start-screen corner, or **Guide** in the top bar) walks through the flow in 10 steps.
**Section bar** (top of every screen after the start screen): **Route map** · **Mission Control** ·
**3D Terrain** · **Science Discovery** · **Exploration Routes** · **Data & Methods**; the Assistant is in the
map's mission panel. Every module is one click away, and the start screen links straight to 3D Terrain, Science, the Assistant and Data &
Methods without planning a route first. No API key is needed for any of this.

Optional:

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | Enables the "Explain this location" AI feature (Claude). Everything else works without it. |
| `MARSCOPE_AI_MODEL` | Overrides the AI model (default `claude-opus-5-5`). |
| `MARSCOPE_OFFLINE=1` | Never touches the network for data (uses cache or bundled snapshot). |

Run the tests with `pip install pytest && pytest -q tests`.

The map's JavaScript (Leaflet) is loaded from public CDNs, as with any Folium map, so the browser
needs internet access for the map widget itself. three.js (MIT, r128) for the 3D view is bundled in
`static/vendor/` and served by the app, with cdnjs as a fallback. All analysis runs locally.

---

## Demonstration scenario (about 2 minutes)

Mission objective: *investigate geological formations associated with Jezero's ancient lake and
river-delta environment.*

0. **Start screen → Route preview → Navigation** (see *Mission navigation flow* below). Choose
   Bright Angel as the destination, press **Show route**, preview a card and press **Start navigation**.
   The steps below use the sections in the top section bar.
1. **Open the app.** The map opens on Jezero with the real CTX orthoimage mosaic, the 13 points of
   interest and literature-reported mineral zones (dashed circles).
2. **Load demo mission.** Start = Octavia E. Butler Landing (Perseverance landing site),
   destination = Bright Angel / Cheyava Falls in Neretva Vallis (site of the "leopard spot" rock).
3. **Three routes are generated automatically** (plus Route D for your own weights), in ~0.2 s:
   * **A – Safest** accepts a longer path to keep slopes and roughness low.
   * **B – Shortest** takes the most direct traversable line.
   * **C – Science-focused** stops at Séítah, Kodiak, the delta front, the delta top and Belva crater.
4. **Compare them** in *Route Planner*: distance, terrain difficulty, cumulative elevation change,
   number of points of interest visited, a normalised science score, advantages and limitations, an
   elevation and slope profile, a trade-off radar and a 3D terrain view.
5. **Move the sliders** (Scientific value / Terrain safety / Distance efficiency). The routes are
   re-optimised immediately, because the weights change the edge costs used by the search.
6. **Select a preferred route** and click **Inspect →** on any point of interest along it.
7. **Science Discovery** shows observations, interpretations, water-related evidence, published
   mineralogy, terrain numbers and suggested objectives, with references.
8. **Click anywhere on the map** (mode "Inspect location", "Set start" or "Set destination") to
   plan from or analyse any location.

---

## Architecture

```
marscope/
├── app.py                      # Entry point: shared models, mission flow, workspace sections
├── components/terrain_explorer/index.html  # three.js 3D colour map (build-free Streamlit component)
├── config.py                   # Study area, grids, data URLs, ASSUMPTIONS, strategy weights
├── requirements.txt
├── modules/
│   ├── data_manager.py         # Data acquisition: STAC index, COG range reads, mosaicking,
│   │                           #   caching, validation, provenance, fallbacks, endpoint checks
│   ├── terrain_analysis.py     # Slope, roughness, difficulty index, terrain classes, hillshade
│   ├── science_discovery.py    # POIs, science-value surface, location inspector, AI explanation
│   ├── route_optimizer.py      # Graph costs, A*, Dijkstra + orienteering, metrics, trade-offs
│   ├── mars_map.py             # Folium/Leaflet map (EPSG:4326), layer rendering, routes, markers
│   ├── tiles.py                # Builds the CTX + HiRISE map-tile pyramids (static/tiles/)
│   ├── mission_control.py      # Mission Control panels (provenance-tagged)
│   ├── rover_sim.py            # Simulated rover state + energy model along a route
│   ├── mission_log.py          # Mission activity timeline (real app events)
│   ├── mission_planner.py      # Deterministic route evaluation, ranking, explanations
│   ├── mission_context.py      # Mission context + relevant-data retrieval for the assistant
│   ├── ai_service.py           # LLM provider interface (Claude default, configurable)
│   ├── assistant.py            # Grounded Q&A + offline fallback + suggested questions
│   ├── assistant_ui.py         # Assistant tab (Streamlit)
│   ├── terrain3d.py            # 3D colour map payloads (layers, scene, cross-section) + component
│   ├── workspace.py            # Shared focus, 3D Terrain section, discovery evidence, route science, charts
│   ├── flow.py                 # Mission flow helpers: route cards, stops, heading, slope ahead (pure)
│   ├── flow_ui.py              # Setup / Compare routes / Navigate screens, Mission Center UI, detail card, guide card
│   ├── mission_center.py       # Simulation clock, waypoints, narrative, recommended action (pure logic)
│   ├── nav_map.py              # 2D navigation map: symbols, map controls, route animation, bounds
│   ├── view3d.py               # 3D view: globe fly-in, Jezero terrain mesh from the CTX DTM (three.js)
│   ├── guide.py                # The 10-step Start Mission Guide
│   ├── landing.py              # Coordinate parsing (and the earlier visor destination screen, unused)
│   ├── mission_data.py         # Perseverance traverse/waypoints, raw images, USGS geologic map
│   └── visualizations.py       # Plotly charts: profiles, radar, histograms, 3D terrain
├── data/
│   ├── jezero_pois.json        # Curated points of interest with sourced observations/interpretations
│   ├── bundled/                # Snapshot of the real USGS data (npz + provenance json)
│   └── cache/                  # Live downloads (created on demand, git-ignored)
├── static/tiles/{ctx,hirise}/  # Bundled zoomable imagery tiles (~13 MB, served at /app/static/)
├── static/vendor/three.min.js  # three.js r128 (MIT) for the 3D view
├── scripts/build_bundled_snapshot.py   # Rebuilds the bundled snapshot from USGS
├── scripts/build_tiles.py      # Rebuilds the CTX + HiRISE tiles from USGS ARD
├── tests/                      # Core tests + mission-data parser tests
├── assets/style.css            # Dark Mars mission-control theme
├── assets/flow.css             # Mission flow styles: visor palette, transitions, Mission Center, guide
├── assets/landing.css          # Earlier destination-screen styles (unused)
└── .streamlit/config.toml
```

### Data resolution order (`data_manager.load_or_fetch`)

1. Local cache (`data/cache/`), written by a live download.
2. Bundled snapshot (`data/bundled/`), real USGS data shipped with the repo.
3. Live download from USGS (about 30–60 MB of HTTP range reads).
4. **Synthetic demonstration terrain**, only if all of the above fail. It is labelled
   "DEMONSTRATION" in red everywhere and all numbers are flagged as illustrative.

The *Data & Methods* tab has **Refresh data from USGS** (forces step 3) and **Check NASA / USGS
endpoints** (live reachability probe), plus validation checks and per-layer provenance.

---

## Data sources

Every layer is tagged in the UI as **OBSERVATION** (instrument data product), **DERIVED** (computed by
MARSCOPE), **LITERATURE** (published results) or **DEMONSTRATION** (synthetic).

| Layer | Kind | Source | Notes |
|---|---|---|---|
| Elevation | Observation | MRO **CTX stereo DTMs**, USGS ARD (`s3://astrogeo-ard/mars/mro/ctx/controlled/usgs/`), CC0 | 20 m/px, absolutely controlled to **MGS MOLA**, heights relative to the Mars aeroid. Three DTMs chosen automatically from the USGS STAC index by greedy footprint cover, vertical offsets harmonised, resampled to ~59 m. The bundled snapshot measures 99.97% of the study area directly. |
| Surface imagery | Observation | MRO **CTX orthoimages** co-registered with the DTMs (USGS ARD) | ~6 m/px, resampled to ~30 m. |
| Daytime / night-time IR | Observation | Mars Odyssey **THEMIS** controlled mosaics, MC-13 Syrtis Major (USGS ARD) | 100 m/px, 8-bit stretched brightness: a qualitative proxy, not calibrated temperature or thermal inertia. |
| High-resolution imagery tiles (default base map) | Observation | MRO **CTX** (~20 m/px, zoom 8–11) and **HiRISE** orthoimages (USGS ARD, read at ~1.2 m/px, zoom 12–15) | HiRISE covers the landing site, delta front, Belva crater and the Neretva corridor; gaps between HiRISE strips are filled with CTX (≈70% HiRISE pixels, recorded in `static/tiles/hirise/tiles.json`). |
| Perseverance traverse & waypoints | Observation | NASA/JPL-Caltech **MMGIS** ("Where is Perseverance?" map) | Fetched live, cached in `data/cache/`. If offline with no cache, the layer is shown as unavailable. |
| Perseverance raw images | Observation | NASA **Mars 2020 raw-image feed** (https://mars.nasa.gov/mars2020/multimedia/raw-images/) | Science tab: for any location the rover stopped near, pick a sol and see thumbnails (loaded by your browser from mars.nasa.gov) linking to the full images. |
| Geologic map | Observation (map) | **USGS SIM 3292** global geologic map (Tanaka et al. 2014), USGS Astrogeology ArcGIS service | 1:20M scale, so only a few units cover Jezero. Hover for unit names. |
| Optional base maps | Observation | **NASA Solar System Treks** WMTS (MOLA colour shaded relief 463 m, Viking MDIM 2.1) | Loaded directly by the browser from trek.nasa.gov. |
| Slope, roughness, difficulty, terrain classes, science value, routes | Derived | MARSCOPE | Formulas below. |
| Points of interest, mineralogy, water evidence | Literature | Peer-reviewed papers and NASA mission reports (references in `data/jezero_pois.json`) | Positions approximate (±0.5–1 km). |

**Not implemented as raster layers, and why:**

* **CRISM mineralogy.** MARSCOPE does not process CRISM spectral cubes, because doing that properly
  needs atmospheric correction and spectral-parameter work beyond this scope. It shows only
  *published* detections (Ehlmann 2008, Goudge 2015, Horgan 2020, rover results) as approximate
  zones around the relevant targets. It never generates a mineral detection.
* **Detailed Jezero geologic maps.** The global USGS map (SIM 3292) is shown, but the detailed Jezero
  maps (e.g. Stack et al. 2020; Sun & Stack, USGS SIM 3464) are not served as a machine-readable API
  that MARSCOPE could rely on. MARSCOPE adds a transparent **DTM-derived physiographic classification**,
  labelled "not a geologic map". If you place a GeoJSON or shapefile of a published map in `data/geology/`, MARSCOPE loads it with
  GeoPandas and adds it as a "User-supplied geologic map" overlay.

---

### About the other suggested sources

* **NASA Worldview** covers Earth only, so it has no Mars data. Its ideas (a base-map picker, overlay
  toggles and an opacity slider) were borrowed for the sidebar.
* **Curiosity's proposed path up Mount Sharp** is in Gale crater, not Jezero. Its style (a planned path
  with numbered, labelled stops) is used for the selected route on the MARSCOPE map.

---

## Methods

* **Slope:** central differences on the ~59 m DTM grid, with metric spacing on the IAU Mars sphere
  (R = 3,396.19 km).
* **Roughness:** standard deviation of (elevation − Gaussian-smoothed elevation) in a 5×5 window.
* **Terrain difficulty** = 0.7·min(slope/max_slope, 1)^1.5 + 0.3·min(roughness/15 m, 1).
* **Terrain classes:** thresholds on slope, relief relative to a 3 km mean, and regional elevation
  percentile.
* **Science value** = 0.65·POI proximity (priority·exp(−(d/1.5 km)²)) + 0.20·terrain exposure +
  0.15·THEMIS night-IR brightness, rescaled to 0–1.
* **Planning graph:** 187 × 225 cells (~237 m), 8-connected. Each cell keeps the mean elevation, the
  90th-percentile slope and the mean difficulty of its 4×4 fine cells.
* **Edge cost** = length·(w_d + 10·w_s·hazard + 2·w_c·(1 − science)). Hazard is the larger of the
  cell hazard (half difficulty, half (p90 slope / limit)²) and the edge grade term. Cells or edges
  above the slope limit are removed.
* **Routes A, B, D:** A\* with the admissible heuristic w_d × straight-line distance. A unit test
  checks that A\* returns exactly the Dijkstra optimum.
* **Route C:** multi-source Dijkstra (SciPy) between start, destination and all POIs, then greedy
  insertion of POIs by priority per added cost while the total stays within 1.6× the direct cost.
* **Strategy blending:** each route's weights = strength × strategy profile + (1 − strength) × your
  weights. The profiles are A (0.15, 0.80, 0.05), B (0.95, 0.05, 0), C (0.20, 0.25, 0.55) for
  (distance, safety, science). Route D uses your weights only.
* **Science-value score (0–100)** = 40 × mean science value along the route + 60 × (sum of the
  priorities of POIs within 600 m / sum of all POI priorities).

### Explicit assumptions (`config.ASSUMPTIONS`, shown in the UI)

* Maximum traversable slope 25° (adjustable in the sidebar) and caution slope 15°.
* A POI counts as visited within 600 m of the route.
* Indicative pace of 100–300 m per sol, used **only** to show a duration *range*. It is not a claim
  about any rover's or astronaut's capability, and it excludes science stops, communications and
  power constraints.

### Limitations

* A ~59 m DTM and a ~237 m planning grid cannot resolve rover-scale hazards (rocks, small scarps,
  sand ripples). Operational planning uses ~1 m HiRISE DTMs and onboard hazard avoidance.
* THEMIS mosaics are relative brightness, not calibrated physical quantities.
* POI positions and priorities are curated approximations. Two unnamed craters are MARSCOPE's own
  identifications and are labelled as such.
* Navigation is a simulation along the planned path at an assumed pace; it ignores science stops,
  power and communications. Playback runs in both 2D and 3D on one shared clock.
* Changing a layer toggle redraws the 2D map, which re-fits the route (the zoom level is not kept).
* The 3D terrain is coarser (~300 m mesh) than the 2D map and uses CTX only.

---

## Map-first workstation (route preview and navigation)

Both map screens are one workstation: the map fills everything except an integrated mission panel on the
right (`modules/dock.py`, `flow_ui.routes_screen` / `navigate_screen`).

* **2D / 3D on the map.** The switch at the top right of the map changes between the satellite map and the
  3D terrain (the real CTX DTM) on the route preview and during navigation; the choice is kept between them.
  The 2D map tilts and recedes while the 3D view settles in, and each switch to 3D starts overhead and
  descends onto the selected waypoint (or the whole route). Switching back to 2D keeps the selected waypoint
  in view. Route, waypoints, selection and the simulated rover stay in sync.
* **Mission panel.** Navigation: an always-visible strip (status, simulated progress ring, next waypoint,
  Play/Pause, previous/next waypoint) and six sections: Assistant, Mission, Navigation, Route, Science,
  System. Route preview: Routes, Assistant, Science, System. "Hide panel" (top bar or the fold button)
  slides it away and widens the map; "Panel" on the map brings it back.
* **Assistant in the panel.** The Assistant moved out of the top bar into the panel. It uses the same answer
  code and the same shared conversation, and its context now includes the selected route, the SIMULATED
  rover position and sim status, the active 2D/3D view and the selected waypoint. Earlier questions fold
  into "Earlier conversation". The mission recommendation, candidate table and maximum route length are in
  "Mission recommendation · all candidates". Every "Ask the Assistant" button in the other modules opens
  this panel; before any route exists they open the full Assistant page as before.
* **Motion (CSS only, no new dependencies).** Staggered section entrance, animated tab indicator, a running
  border on the strip only while the simulation plays, a sheen on the progress bar while playing, count-up
  numbers, tooltip/expander/dialog transitions, button press feedback and a loading shimmer under the 3D
  view. Framer Motion / Skiper UI are React libraries that cannot run inside Streamlit without a separate
  build, so their patterns are recreated in original CSS. All of it stops under `prefers-reduced-motion`.

## Astronaut HUD interface

Every screen after the start screen uses one helmet-HUD design language (`assets/hud.css`, loaded by
`flow_ui.inject_css` for every view except `setup`). The start screen keeps its own visor template
(`landing.css`) and is unchanged; a pixel comparison of the start screen before and after the HUD work is identical.

* Palette: deep charcoal / near-black with graphite panels, light text, muted teal-cyan accent, warm Mars
  highlights, amber for warnings and a restrained red for critical states.
* Effects (CSS only): slow background grid, a very faint scanline, corner brackets on panels, a radar sweep next to
  the brand, wireframe buttons with a scan-sweep hover, panel fade-in, and a cyan underline on the active module.
* Mission Center overview: a ring gauge of the simulated progress (labelled SIM) and a slow ticker of the planned
  route's CTX DTM profile (distance, elevation, slope), labelled "derived, not telemetry".
* Charts: only the shared Plotly theme (`visualizations._theme`) changed (grid, tick font, hover label); data and
  calculations are untouched. The 3D component's panels use the same palette.
* `prefers-reduced-motion`: every HUD animation is switched off; nothing depends on animation.

### Streamlit compatibility (`modules/st_compat.py`)

`requirements.txt` asks for `streamlit>=1.50`. Newer keyword arguments (for example
`segmented_control(required=True)`) would raise `TypeError` on older installs, so `st_compat.install()` wraps the
widget functions at start-up: unknown keywords are dropped, `required=True` is emulated (clicking the active option
keeps it selected) and `width="stretch"` maps to `use_container_width=True`. Tested on Streamlit 1.50 and 1.67.

## Mission navigation flow

**Screens.** `setup` → `routes` → `navigate`, plus `all` (the workspace sections, reached from the
top bar). If the start or destination changes, routes are cleared, and a screen that needs routes
sends you back to setup. A route therefore never belongs to old endpoints.

**1 · Start screen (setup).** This is the astronaut-visor template from the landing screen (CTX
orthoimage of the delta in perspective, with the visor frame), recoloured to slate. A single panel is
centred both ways. It holds the MARSCOPE name, a **Start** field, **⇅ Swap**, a **Destination** field
and one outlined **Show route** button, which is quieter than the fields. Nothing else is shown: no
route cards, telemetry, science, layers or status. A small **Mission Guide** link sits in the corner.
- Start defaults to Octavia E. Butler Landing (Perseverance landing site).
- Each field is a searchable list: landing sites and landmarks first for the start, science targets by
  priority for the destination.
- You can also type coordinates (`18.47, 77.40`, degrees N and degrees E). Points outside the terrain
  data are refused with a message under the field.

**2 · Route preview (routes).** **Show route** fades the setup panel out while the four routes are
planned on the CTX elevation model. The map then fades in at nearly the full viewport (the browser
height minus a slim 48 px top bar), fitted to the start, the destination and all routes. The selected
route draws itself along its length. After that, the route-choice side panel slides in.
- Each of the four cards (**Safest**, **Shortest**, **Science-focused**, **Balanced / custom**) shows:
  - estimated distance;
  - risk (a relative terrain index, never "safe");
  - science value out of 100;
  - notable stops in driving order;
  - a one-line explanation with its trade-off.
- The recommended card carries a badge. A sentence above the cards says why it was recommended: its
  score against the runner-up, what it wins and gives up, and the weights.
- A footnote labels distances and sols as estimates.
- **Preview** on a card redraws the map with that route highlighted. The endpoints stay put.
- **Balanced / custom** has *Adjust weights*.
- **Start navigation · <route>** opens navigation.
- **Map tools** (collapsed) holds the three click modes:
  - **Inspect** shows the derived elevation, slope and terrain class for the clicked point;
  - **Set start** and **Set destination** move that endpoint and re-plan at once. A click within
    300 m of a catalogued target uses that target.

**3 · Navigation (simulated).** The map stays dominant on the left. A **2D MAP / 3D TERRAIN** switch
sits on the map's top right. The right sidebar is the **MARSCOPE MISSION CENTER**: it scrolls on its
own, never covers the map, and **Hide controls** collapses it (on narrow screens it stacks below the
map). Its five sections (`modules/mission_center.py` holds the logic, `flow_ui.py` the UI):

1. **Mission overview**: the route, start → destination, distance, stops, sols at the assumed pace,
   maximum slope, the status (READY / PLAYING / PAUSED / STOPPED / COMPLETED) and a progress bar with
   a tick per waypoint.
2. **Mission assistant** (replaces the old HUD over the map). A plain-language narrative is built only
   from the route, the CTX DTM profile and the catalogue, for example "The rover is approaching Delta
   front, about 0.7 km ahead. The upcoming terrain is gentle (up to 11° in the next 0.5 km). Continue
   along the planned route." It also shows the route, the simulated position, the current and next
   waypoint with its distance, remaining km and sols, the terrain class, the slope here and ahead
   (amber warning above the 15° caution slope), the heading, progress and status. It is labelled
   **SIMULATED · no live rover link**. *Ask the Assistant about this position* opens the Assistant.
3. **Exploration guide**: six expandable parts that follow the selected route and simulation state.
   They are the current objective (destination objectives from the catalogue); the current location
   (nearest catalogued feature within 1.5 km, with a cited observation and a labelled hypothesis); the
   next target; terrain; a rule-based recommended next action; and overall progress with a waypoint
   checklist.
4. **Route simulation**: **Play / Pause**, **Stop**, **Reset**, **Previous / Next waypoint**, speed
   **1× 2× 5× 10×** (1× = one simulated sol per second at 200 m/sol), a timeline with waypoint ticks,
   sim time (sol), the current and next waypoint, distance travelled and remaining.
   - Pause holds the position.
   - Stop ends the run but keeps the rover where it stopped; Play then starts again from the start.
   - Reset goes back to the start at 1× and clears the selection.
   - Stepping jumps to the waypoint and opens its details.
   - Leaving the screen pauses the simulation. The Mission Control progress slider and this screen
     share the same progress.
5. **Waypoint information**: the start, the numbered stops and the destination. Each card is tinted
   with its geological group colour. The next waypoint, the rover's waypoint and the selected one ease
   into that tint and shift slightly, and a real HiRISE/CTX crop from the bundled tiles slides in.
   This is an original CSS recreation of the colour-transition idea in skiper-ui's "skiper24" list,
   not its code: that component is a licensed Pro React component. Clicking a card (or a pin on either
   map) opens the detail card, which offers **Move simulated rover here**. Off-route targets are in a
   list below.

**One simulation clock.** The position is `d0 + speed × 200 m/s × (now − t0)` while playing and `d0`
while paused. The sidebar refreshes every 0.6 s and the 2D map every 1 s. The 3D view runs the same
formula in the browser and is re-anchored on every control change, so the rover, route, selected
waypoint and status stay in sync in both views. The 2D map and the 3D terrain stay mounted on top of
each other, so switching crossfades without reloading or losing state. The 3D view is the existing
`terrain_explorer` engine in a navigation mode: colour modes, legend, vertical exaggeration, follow
rover, route/whole-crater/top views, the selected-waypoint highlight and a **SIMULATED rover position ·
no live rover link** badge. It pauses rendering while hidden. Its first opening descends from high
altitude onto the route (instant with reduced motion). This replaced the older globe fly-in on this
screen.

Science details stay hidden until a stop or target is clicked on the map or picked from the list. The
detail card keeps these apart, with citations:
- *observations* (REAL DATA, published measurements);
- *interpretations* and *reported minerals* (LITERATURE);
- *derived values* (DERIVED);
- *on-this-route* values (SIMULATED).

Map buttons: zoom ±, **Fit route** and **◎ Rover** (centre on the simulated position). All of them can
be reached with the keyboard.

**2D / 3D.**
- *2D* uses MRO CTX (~20 m/px) everywhere and HiRISE (~1.2 m/px) over the landing site, delta front
  and Belva crater (zoom in). Both come from USGS Astrogeology ARD. Elevation and slope come from the
  CTX DTM (~59 m).
- *3D* flies in from the Mars globe to Jezero. It then shows the same CTX DTM as a ~300 m mesh (×3
  vertical) draped with the CTX orthoimage, with the route, stops and simulated rover. HiRISE is not
  used in 3D.
- If WebGL or three.js is unavailable, the 3D panel says so and 2D still works.

**Palette.** Charcoal, graphite and slate surfaces (`#111316`, `#181b1f`, `#1e2227`) with muted teal
(`#7fb8ae`), desaturated blue (`#92abd6`), lavender (`#b8a6e3`) and pale sand (`#d9c39c`). A muted Mars
tone (`#c98b6b`) is kept for small details only, such as the destination flag and its dot. There is no
saturated orange, neon or glow. Fonts stay Inter + JetBrains Mono.

**Transitions actually implemented** (CSS and Leaflet; all of them are short and none blocks input):
- Setup → map: the setup panel fades out (≥0.45 s, shown while routes compute), then the map fades in
  (0.5 s).
- Route draw → panel: the selected route draws itself (SVG stroke, ~1.2 s). The side panel slides in
  after a 1.25 s delay. During the delay the map is already usable.
- Route selection: the card border and background ease (0.25 s). The map redraws the newly selected
  route.
- Panels: the navigation side panel slides in. The detail card and guide card fade in. Expanders use
  Streamlit's own open/close.
- 2D / 3D: a 0.45 s crossfade between the two stacked views. The first 3D opening descends onto the
  route (~2.2 s).
- Waypoint cards: background colour, border and position ease (0.55 s); the image slides in (0.6 s).
- The map eases in by one zoom level when drawn. **Fit route** / **◎ Rover** fly smoothly. The progress
  bars animate their width.
- With *prefers-reduced-motion* all of these are off: no fades, no route drawing, instant map moves,
  and 3D opens directly on the terrain.

**Limitations (honest).**
- *Full-screen map.* The map iframe is sized with CSS to `100vh − 70px`, and Leaflet re-measures on
  load and on resize. Very short windows (under ~500 px) fall back to a scrolling layout. On phones the
  side panel stacks below the map.
- *Data seams.* The 2D mosaic shows brightness seams where three separate CTX orthoimages meet. Areas
  outside the CTX/DTM coverage are dark. Both are real data boundaries, not rendering faults.
- *Map refits.* Changing layers, previewing another route or a playback step redraws the map. It
  re-fits to the route, so a manual pan or zoom is not kept across those redraws.
- *Rover motion.* The simulated rover marker fades in at each new position. It does not glide between
  positions, because streamlit-folium replaces the marker layer.
- *Setup fade.* The fade-out runs while the routes compute (~0.2–1 s), so on a slow machine the
  "Planning routes…" stage list stays visible a little longer.
- *3D.* The 3D view is coarser (~300 m mesh) and uses CTX only. Its rover is predicted in the
  browser between server updates, so it can differ from the sidebar by a fraction of a second until
  the next control change re-anchors it.
- *2D rover.* The 2D rover marker jumps once per second during playback, because streamlit-folium
  replaces the marker layer.
- *Fallbacks.* If the interactive map fails, the screen says so and shows the static CTX image. Without
  real data, routes run on labelled synthetic terrain.

### Start Mission Guide

Open **Mission Guide** from the corner of the start screen or **Guide** in the top bar. It has ten
steps:
1. the start screen;
2. the start;
3. the destination and coordinates;
4. swap and **Show route**;
5. the route preview;
6. the four route options;
7. the 2D map / 3D terrain switch;
8. the Mission Center assistant and exploration guide (simulated, no live rover connection);
9. route simulation controls and layers;
10. waypoints, science details, provenance, ending a mission and every module.

Each step switches to its screen and outlines the control it describes. You can go **← Back** or
**Next →**, **Skip tour**, or **Close** it, which keeps your place: the button becomes **Resume
guide**. **↺ Restart** goes back to step 1. Steps 5–10 need routes. If there are none, the guide offers
to calculate them; it never fakes a result.

---

## Integrated workspace (all modules, shared state)

Nothing from earlier versions was removed: the redesign had only hidden the older modules behind one
button. They are now sections of one workspace, and they share state:

* **`focus`** is the location being studied. It is set by a click or a feature pin in 3D, by Inspect on the
  Mission Control map, by a row of the discovery catalogue or by the Science Discovery picker. 3D Terrain,
  the cross-section, Science Discovery and the Assistant all read it.
* The **selected route** is the same everywhere: route map, Mission Control, Exploration Routes, the 3D
  playback and the Assistant.
* Cross-links:
  * Science Discovery → *Show in 3D terrain* (the camera flies there), *Ask the Assistant* (the question is
    asked with the location in context) and *Plan a route here* (re-plans with the normal transition).
  * 3D Terrain → *Science details*, *Ask Assistant* and *Plan a route to this point*.
  * Exploration Routes → *Play this route in 3D*.

### 3D Terrain: interactive colour map (`modules/terrain3d.py`, `components/terrain_explorer/index.html`)

A three.js view of the CTX DTM (~300 m mesh). It is a plain-HTML Streamlit component that speaks the
component protocol directly, so no frontend build is needed, and it returns clicks to Python.

* **Colour modes**, all data, never decoration:
  * Elevation (observation);
  * Slope (derived, with breaks at the 15° caution slope and the 25° limit);
  * Terrain class (derived physiography);
  * Geological features (approximate zones around catalogued, published features; *not* a geologic map);
  * Scientific interest (derived science value);
  * CTX image (observation).
* **Legend:** each mode has its own legend, with an evidence tag. In the class and geology modes, clicking a
  legend entry isolates it.
* **Navigation:** drag to rotate, shift-drag or right-drag to pan, scroll to zoom, or use the arrow keys.
  The **Route view**, **Whole crater** and **Top view** buttons jump to preset views. Vertical
  exaggeration runs from 1× to 8×.
* **Readouts:** the cursor readout shows coordinates, elevation, slope, class, science value and the
  catalogued zone. **Click** to select a point. A side panel then recomputes the point at full resolution
  and offers the cross-links listed above.
* **Scientific animations**, all with controls:
  * **Rover playback (SIMULATED):** ▶ Play / ⏸ Pause, ⟲ Reset, speed 0.5–8 sol/s, a scrubbable timeline
    and *Follow rover*. A live readout shows distance driven, simulated sol, elevation, slope, position,
    the next feature and features passed. Catalogued features pulse when reached, and the driven trail is
    drawn. It is labelled "SIMULATED · illustrative playback along the planned route, not mission
    telemetry". The real Perseverance traverse (NASA/JPL MMGIS) is drawn only when it has been downloaded.
  * **Reveal relief:** the terrain rises from flat to the chosen exaggeration.
  * **Tour layers:** steps through every data layer over the same region, cross-fading the colours, so you
    can see how the datasets line up.
  * **Compare + Sweep:** two layers are split at a line you can drag or sweep across the crater.
  * **Cross-section Scan:** a scan line moves along the section in the small chart while a marker moves
    along the lavender section curtain in 3D.
  * Every animation respects *prefers-reduced-motion*.
* **Charts below the 3D view** (Plotly, hover for values):
  * the topographic cross-section through the selected point at full DTM resolution, with azimuth and
    length sliders, catalogued features within 800 m and the terrain class under each sample;
  * area per terrain class;
  * the elevation range of each class;
  * the approximate area of each catalogued geological setting.

### Science Discovery
* A **discovery catalogue** table lists every catalogued feature with its setting, geological group,
  location, evidence, water evidence and planning priority. Selecting a row opens it.
* **Evidence** comes from the catalogue's citations: *Observed in situ* (rover), *Orbital detection*,
  *Observed in situ + orbit*, or *Unverified* (MARSCOPE's own identifications). It is not a statistical
  confidence.
* **Evidence and significance** keeps three things apart:
  * the evidence basis;
  * the geological setting;
  * why it matters, labelled as an interpretation and not a confirmed finding.
* **Supporting imagery** is a real HiRISE crop (inside HiRISE coverage) or a CTX crop from the bundled map
  tiles.
* Everything from before is kept: observations, interpretations, water evidence, minerals, objectives, the
  context chart, Perseverance visits and raw images, the AI explanation and the fact sheet.

### Exploration Routes
Everything from the Route Planner is kept: cards, the elevation and slope profile, the trade-off radar,
the slope distribution, science along the route, POIs, GeoJSON and CSV downloads, and the 3D Plotly
terrain. Two things are new:
* **Why Route X is scientifically interesting**: each catalogued feature passed, with its geological
  setting, a quoted observation and its interpretation (with citations), plus the terrain the route
  crosses.
* A chart of the **terrain crossed by each route**.

### Assistant
The full Mission Planner and Science Assistant is unchanged (conversation, suggested questions,
loading and error states, optional Claude via `ANTHROPIC_API_KEY`). Its context gains two things:
* the **selected location** (fact sheet and citations);
* a **visualisation guide**, so it can explain the 3D modes, animations and charts. This works offline too.

### Limitations of the new parts
* **Mesh resolution:** the 3D mesh is ~300 m and uses CTX only (no HiRISE in 3D). The rover symbol is not to
  scale, and vertical exaggeration is stated on screen.
* **Geological features mode:** it shows approximate circles around catalogued targets, not mapped unit
  boundaries. MARSCOPE has no Jezero-scale geologic map.
* **Cross-section:** it is the surface only, because MARSCOPE has no subsurface data.
* **Playback:** the 3D rover playback is a simulation along the planned route at an assumed pace. It is not
  Perseverance's actual drive.
* **Interaction cost:** the 3D view runs in the browser (WebGL). A Python-side change, such as a slider or a
  section switch, re-sends the scene. The camera and the colour mode are kept when that happens. Playback
  progress is also kept unless the selected route changes.

## Mission Control Center

After the destination screen, MARSCOPE opens on **🛰️ Mission Control**. This is the main operations
view, built on the existing map, planner, science catalogue and assistant.

* **Top bar:** mission name, data-system status (terrain, rover traverse, AI), session clock and
  simulated sols.
* **Left:** *Rover telemetry* for a **simulated** rover on the selected route. You can move it with
  the progress slider or *Advance 1 sol*. Elevation and slope at its position come from the real
  CTX DTM. Energy comes from a stated physics model: mass, rolling resistance and efficiency, in
  `config.ENERGY_MODEL`. When NASA's MMGIS data are loaded, Perseverance's last reported position is
  shown as REAL DATA. This panel also has navigation (click mode, route choice) and the mission
  parameters.
* **Centre:** the existing surface map. It shows the planned and recommended routes, numbered stops,
  the simulated rover and its driven segment, targets, mineral zones, impassable terrain (on by
  default) and optional geology and traverse layers.
* **Right:** *Science operations* lists every target with its interest score, science value, an
  evidence count and its status (stop n, reached, inspected). The evidence for the selected target is
  split into measured, literature, derived and hypotheses. *AI mission intelligence* asks the shared
  assistant mission questions.
* **Bottom:** *Mission strategy* compares every route on score, science, safety, slopes, distance,
  energy (simulated) and sols (simulated). *Alternative scenarios* re-scores the same routes under
  other weightings, and one click applies a scenario. Also here: the route and terrain summary, the
  *mission activity timeline* (real application events with UTC times; simulation steps are tagged)
  and *Communications*. Communications is conceptual reference material only, because MARSCOPE has no
  link or pass data.

Every value carries a label: REAL DATA, DERIVED VALUE, LITERATURE, SIMULATED, CONCEPTUAL or
UNAVAILABLE. Modules: `mission_control.py` (panels), `rover_sim.py` (simulation and energy) and
`mission_log.py` (timeline).

## Astronaut Assistant (Mission Planner + Science Assistant)

The **🧑‍🚀 Assistant** tab combines a deterministic Mission Planner with a grounded conversational
assistant.

* **Mission Planner** (`modules/mission_planner.py`): evaluates the route candidates produced by the
  existing planner. Scoring is science 50 %, safety 30 % and distance 20 % by default; the weights
  follow the *Mission priorities* sliders. Routes that break a constraint are excluded: no path, or
  longer than the optional *Maximum route length*. Duplicate paths are merged, the rest are ranked,
  and a plain-language explanation is written from the numbers. No language model is involved, and
  no route is ever called safe.
* **Mission context** (`modules/mission_context.py`): collects the start, destination, evaluation,
  destination science (POI catalogue, terrain, THEMIS) and Perseverance visits. For each question it
  selects the relevant catalogue entries.
* **AI service** (`modules/ai_service.py`): one provider interface. The default is Claude through
  `anthropic`; `openai_compatible` works with any `/v1/chat/completions` server.
* **Assistant** (`modules/assistant.py`): sends the system rules, the recent conversation and the
  mission data, then returns the answer with its sources. Its rules are: use only MARSCOPE data for
  mission facts; separate observations from interpretations; say when data are missing; label general
  background.
* **UI** (`modules/assistant_ui.py`): the recommendation card, all candidates, destination science,
  contextual suggested questions (one click asks them), the chat history and the input. The
  recommended route is the route highlighted on the existing Mission Map, with numbered stops.

Configuration (environment variables or `.streamlit/secrets.toml`; never commit keys):

| Setting | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | Enables Claude (default provider). |
| `MARSCOPE_AI_PROVIDER` | `anthropic` (default) or `openai_compatible`. |
| `MARSCOPE_AI_MODEL` | Model id (default `claude-opus-5-5`). |
| `MARSCOPE_AI_BASE_URL`, `MARSCOPE_AI_API_KEY` | For `openai_compatible`. |

**Without a key** the assistant still works in *offline mode*. It answers route, comparison,
uncertainty, investigation, mineral and "why is this interesting" questions directly from MARSCOPE
data. It says that general Mars questions need the AI service instead of guessing. Answers are cached
per mission state, so repeated questions make no new API calls.

## AI explanation (optional)

When `ANTHROPIC_API_KEY` is set, *Science Discovery → Explain this location* sends a JSON **fact sheet**
(derived numbers, sourced observations and interpretations, references, caveats) to Claude. The
system prompt requires it to use only those facts, to separate observations from interpretations, to
cite reference keys, and never to invent detections or discoveries. The fact sheet is shown in the UI
for transparency. Without a key, the app explains how to enable the feature and works normally.

---

## References

* Fassett & Head (2005) GRL 32, L14201: ancient deltas in a crater lake in the Nili Fossae region.
* Ehlmann et al. (2008) Nature Geoscience 1, 355–358: clay minerals in delta deposits.
* Goudge et al. (2015) JGR Planets 120, 775–808: mineralogy of the Jezero watershed and fan.
* Horgan et al. (2020) Icarus 339, 113526: the mineral diversity of Jezero crater.
* Stack et al. (2020) Space Science Reviews 216, 127: photogeologic map of the Perseverance field site.
* Mangold et al. (2021) Science 374, 711–717: the ancient delta-lake system seen by Perseverance.
* Farley et al. (2022) Science 377, eabo2196: aqueously altered igneous rocks on the crater floor.
* Liu et al. (2022) Science 377, 1513–1519: an olivine cumulate outcrop on the crater floor.
* Hurowitz et al. (2025) Nature: redox-driven mineral and organic associations in Jezero Crater.
* USGS Astrogeology Science Center, Analysis Ready Data (CTX DTMs/orthoimages, THEMIS mosaics), CC0.
* NASA/JPL-Caltech Mars 2020 Perseverance mission pages; NASA Solar System Treks.

Built for the NASA International Space Apps Challenge. MARSCOPE is not an official NASA product.
