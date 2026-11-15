"""Mission Control Center panels (presentation only).

Every value shown carries a provenance tag:
REAL DATA (instrument or mission data), DERIVED VALUE (computed by MARSCOPE from real data),
LITERATURE (published results), SIMULATED (model or user-driven simulation), CONCEPTUAL
(reference information, no live connection) or UNAVAILABLE.
"""
from __future__ import annotations

import html

import streamlit as st

import config
from modules import mission_log, mission_planner as mp, rover_sim

TAGS = {"real": ("REAL DATA", "t-real"), "derived": ("DERIVED VALUE", "t-der"), "lit": ("LITERATURE", "t-lit"),
        "sim": ("SIMULATED", "t-sim"), "concept": ("CONCEPTUAL", "t-con"), "na": ("UNAVAILABLE", "t-na")}


def tag(kind: str) -> str:
    label, cls = TAGS[kind]
    return f"<span class='mc-tag {cls}'>{label}</span>"


def _e(x) -> str:
    return html.escape(str(x))


def row(label: str, value: str, kind: str) -> str:
    return f"<div class='mc-row'><span class='k'>{_e(label)}</span><span class='v'>{value}</span>{tag(kind)}</div>"


def section(title: str, extra: str = "") -> None:
    st.markdown(f"<div class='mc-sec'>{_e(title)}{extra}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def top_bar(mission_name: str, statuses: list[tuple[str, str, str]], sim_sols: float | None) -> None:
    el = mission_log.mission_elapsed_s()
    el_txt = (f"{int(el // 3600):02d}:{int(el % 3600 // 60):02d}:{int(el % 60):02d}" if el is not None else "--:--:--")
    chips = "".join(f"<div class='mc-chip'><span>{_e(n)}</span>{_e(v)} {tag(k)}</div>" for n, v, k in statuses)
    sol_txt = f"{sim_sols:.1f}" if sim_sols is not None else "—"
    st.markdown(f"""<div class='mc-top'>
      <div class='mc-id'><b>MARSCOPE</b><span>Mission Control</span></div>
      <div class='mc-mission'><span>Mission</span>{_e(mission_name)}</div>
      <div class='mc-chips'>{chips}</div>
      <div class='mc-clock'><div><span>Session elapsed</span>{el_txt}</div>
        <div><span>Simulated sols</span>{sol_txt} {tag('sim')}</div></div>
    </div>""", unsafe_allow_html=True)


def telemetry(state: rover_sim.RoverState | None, real_rover: dict | None, route) -> None:
    section("Rover telemetry", " <small>simulated rover on the planned route</small>")
    if state is None:
        st.markdown("<div class='mc-empty'>No route yet: choose a destination to start a simulated traverse.</div>",
                    unsafe_allow_html=True)
    else:
        e = config.ENERGY_MODEL
        rows = [
            row("Position", f"{state.lat:.4f}N {state.lon:.4f}E", "sim"),
            row("Status", _e(state.status), "sim"),
            row("Progress", f"{state.progress:.0%} of Route {route.key}", "sim"),
            row("Distance traveled", f"{state.traveled_m / 1000:.2f} km", "sim"),
            row("Remaining", f"{state.remaining_m / 1000:.2f} km", "derived"),
            row("Elevation here", f"{state.elevation_m:,.0f} m", "derived"),
            row("Slope here", f"{state.slope_deg:.1f}°", "derived"),
            row("Energy used", f"{state.energy_used_wh / 1000:.2f} / {state.energy_total_wh / 1000:.2f} kWh", "sim"),
        ]
        st.markdown("<div class='mc-panel'>" + "".join(rows) + f"""<div class='mc-bar'><i style='width:{state.progress*100:.1f}%'></i></div>
        <div class='mc-foot'>Position and progress are a simulation along the planned route; elevation and slope
        come from the real CTX DTM at that point. Energy model: mass {e['rover_mass_kg']:.0f} kg, rolling
        resistance {e['rolling_resistance']} (assumed), drive efficiency {e['drive_efficiency']:.0%} (assumed).</div>
        </div>""", unsafe_allow_html=True)
    if real_rover:
        st.markdown("<div class='mc-panel'>" + row("Perseverance",
                    f"sol {real_rover['sol']} · {real_rover['lat']:.4f}N {real_rover['lon']:.4f}E", "real")
                    + "<div class='mc-foot'>Last reported position, NASA/JPL MMGIS traverse data. MARSCOPE is not connected to the rover.</div></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='mc-panel'>" + row("Perseverance", "not loaded (offline)", "na") + "</div>",
                    unsafe_allow_html=True)


def target_table(rows: list[dict], selected: str | None) -> None:
    head = "<tr><th>Target</th><th>Interest</th><th>Sci</th><th>Obs/Int/Min</th><th>Status</th></tr>"
    body = "".join(
        f"<tr class='{'sel' if r['id'] == selected else ''}'><td>{_e(r['short'])}</td><td>{r['priority']:.2f}</td>"
        f"<td>{r['science_value']:.2f}</td><td>{r['n_obs']}/{r['n_int']}/{r['n_min']}</td>"
        f"<td>{_e(r['status'])}</td></tr>" for r in rows)
    st.markdown(f"<div class='mc-scroll'><table class='mc-table'>{head}{body}</table></div><div class='mc-foot'>Interest = MARSCOPE priority from "
                f"published results {tag('lit')} · Sci = science value at the site {tag('derived')} · Evidence = "
                "observations / interpretations / minerals in the catalogue.</div>", unsafe_allow_html=True)


def target_evidence(info: dict, poi: dict, measured: bool) -> None:
    obs = "".join(f"<li>{_e(o['text'])} <i>[{_e(o['source'])}]</i></li>" for o in info["observations"]
                  if o["poi"].startswith(poi["name"])) or "<li>None catalogued.</li>"
    hyp = "".join(f"<li>{_e(i['text'])} <i>[{_e(i['source'])}]</i></li>" for i in info["interpretations"]
                  if i["poi"].startswith(poi["name"])) or "<li>None catalogued.</li>"
    mins = "".join(f"<li>{_e(m)}</li>" for m in poi.get("minerals_reported", [])) or "<li>None reported.</li>"
    t = info["terrain"]
    st.markdown(f"""<div class='mc-panel ev'>
      <div class='ev-h'>Measured / observed {tag('real')}</div><ul>{obs}</ul>
      <div class='ev-h'>Reported minerals {tag('lit')}</div><ul>{mins}</ul>
      <div class='ev-h'>Derived terrain {tag('derived')}</div>
      <ul><li>Elevation {t['elevation_m']:,.0f} m · slope {t['slope_deg']:.1f}° · {_e(t['terrain_class'])}</li>
      <li>Science value {info['science_value']:.2f}</li></ul>
      <div class='ev-h'>Hypotheses / interpretations {tag('lit')}</div><ul>{hyp}</ul>
      <div class='mc-foot'>Data quality: position approximate (±0.5–1 km); elevation
      {'measured by the CTX DTM' if measured else 'gap-filled (no DTM coverage)'}; mineral data are published
      detections, not CRISM processing by MARSCOPE.</div></div>""", unsafe_allow_html=True)


def strategy(evaluation: mp.MissionEvaluation, energy: dict, selected: str | None) -> None:
    head = ("<tr><th>Route</th><th>Score</th><th>Science</th><th>Safety</th><th>Max slope</th><th>&gt;15°</th>"
            "<th>km</th><th>Energy kWh</th><th>Sols</th></tr>")
    body = ""
    for c in sorted(evaluation.candidates, key=lambda c: (c.rank is None, c.rank or 0)):
        if not c.eligible or c.duplicate_of:
            why = f"same path as Route {c.duplicate_of}" if c.duplicate_of else f"excluded: {c.exclusions[0]}"
            body += f"<tr class='dim'><td><b>{c.key}</b> {_e(c.name)}</td><td colspan='8'>{_e(why)}</td></tr>"
            continue
        m = c.raw
        en = energy.get(c.key)
        star = " ★" if c.rank == 1 else ""
        body += (f"<tr class='{'sel' if c.key == selected else ''}'><td><b style='color:{c.color}'>{c.key}</b> "
                 f"{_e(c.name)}{star}</td><td>{c.score:.0f}</td><td>{c.criteria['science']:.0f}</td>"
                 f"<td>{c.criteria['safety']:.0f} <small>{_e(c.safety_level)}</small></td>"
                 f"<td>{m['max_slope_deg']:.0f}°</td><td>{m['pct_above_caution']:.1f}%</td>"
                 f"<td>{m['distance_km']:.2f}</td><td>{'' if en is None else f'{en / 1000:.2f}'}</td>"
                 f"<td>{m['distance_km'] * 1000 / config.SIM_PACE_M_PER_SOL:.0f}</td></tr>")
    st.markdown(f"<table class='mc-table wide'>{head}{body}</table><div class='mc-foot'>Score, science, safety, "
                f"slopes and distance {tag('derived')} from the CTX DTM and catalogue · Energy and sols {tag('sim')} "
                f"(energy model above; {config.SIM_PACE_M_PER_SOL:.0f} m/sol assumed) · ★ recommended.</div>",
                unsafe_allow_html=True)


def scenarios(rows: list[dict], on_apply) -> None:
    body = ""
    for r in rows:
        w = r["weights"]
        score = "" if r["score"] is None else f" ({r['score']:.0f})"
        body += (f"<tr><td>{_e(r['scenario'])}</td><td>{w['science']:.0%} / {w['safety']:.0%} / {w['distance']:.0%}"
                 f"</td><td><b>{r['recommended'] or '—'}</b>{score}</td><td>{r['runner_up'] or '—'}</td></tr>")
    st.markdown("<table class='mc-table'><tr><th>Scenario</th><th>Sci / Safe / Dist</th><th>Recommends</th>"
                f"<th>Runner-up</th></tr>{body}</table>", unsafe_allow_html=True)
    cols = st.columns(len(rows))
    for col, r in zip(cols, rows):
        col.button(r["scenario"].split(" (")[0], key=f"scn_{r['scenario']}", on_click=on_apply, args=(r,),
                   help="Apply these weights: routes are re-optimised and the change is logged.",
                   width="stretch")


COMMS_REFERENCE = [
    ("Mars Reconnaissance Orbiter (NASA)", "UHF relay for surface missions"),
    ("Mars Odyssey (NASA)", "UHF relay; longest-serving Mars orbiter"),
    ("MAVEN (NASA)", "UHF relay"),
    ("Trace Gas Orbiter (ESA)", "UHF relay"),
    ("Mars Express (ESA)", "UHF relay (backup)"),
]


def comms() -> None:
    items = "".join(f"<li><b>{_e(n)}</b>: {_e(r)}</li>" for n, r in COMMS_REFERENCE)
    st.markdown(f"""<div class='mc-panel'>
      {row('Link status', 'no link', 'na')}
      <div class='ev-h'>Relay infrastructure used by Mars surface missions {tag('concept')}</div><ul>{items}</ul>
      <div class='mc-foot'>MARSCOPE is a planning tool and is not connected to any spacecraft. Reference: surface rovers such as Perseverance send most data through UHF relay to
      orbiters, and can also talk directly to Earth over X-band via NASA's Deep Space Network. One-way light time
      between Earth and Mars is roughly 3 to 22 minutes depending on orbital positions. No pass schedules, link
      budgets or data volumes are shown because MARSCOPE has no data for them.</div></div>""",
                unsafe_allow_html=True)


def timeline(events: list[dict]) -> None:
    if not events:
        st.markdown("<div class='mc-empty'>No activity yet.</div>", unsafe_allow_html=True)
        return
    rows = "".join(
        f"<div class='tl {e['kind']}'><span class='t'>{e['utc']:%H:%M:%S}</span><span class='e'>{_e(e['event'])}"
        f"{' ' + tag('sim') if e['kind'] == 'sim' else ''}</span><span class='d'>{_e(e['detail'])}</span></div>"
        for e in reversed(events[-40:]))
    st.markdown(f"<div class='mc-timeline'>{rows}</div><div class='mc-foot'>Actions taken in this session (UTC, "
                "app clock). Nothing is generated: each line is a real application event.</div>",
                unsafe_allow_html=True)
