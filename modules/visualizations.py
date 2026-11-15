"""Plotly scientific visualisations for MARSCOPE (dark mission-control theme)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from modules.science_discovery import poi_by_id

FONT = dict(family="Inter, Segoe UI, sans-serif", color="#dde3e8", size=12)
GRID = "rgba(86,194,207,0.09)"            # HUD: faint cyan grid
TICK = dict(family="JetBrains Mono, ui-monospace, Menlo, monospace", size=10, color="#8d98a3")
HOVER = dict(bgcolor="rgba(14,16,19,0.94)", bordercolor="#56c2cf",
             font=dict(family="JetBrains Mono, ui-monospace, Menlo, monospace", size=11, color="#e6ebef"))


def _theme(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    layout = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(18,21,25,0.55)", font=FONT, height=height, hoverlabel=HOVER,
                  margin=dict(l=55, r=20, t=40, b=45),
                  legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=TICK, linecolor="rgba(86,194,207,0.25)")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=TICK, linecolor="rgba(86,194,207,0.25)")
    fig.update_annotations(font_color="#8fb4d8")
    return fig


def elevation_profile(routes: list, selected: str | None) -> go.Figure:
    """Elevation (top) and slope (bottom) against distance for every route."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.06,
                        subplot_titles=("Elevation along route (CTX DTM, m rel. aeroid)", "Slope (deg)"))
    for r in routes:
        if not r.found or not r.profile:
            continue
        sel = r.key == selected
        x = r.profile["distance_m"] / 1000
        fig.add_trace(go.Scatter(x=x, y=r.profile["elevation_m"], name=f"{r.key} - {r.name}",
                                 line=dict(color=r.color, width=3.5 if sel else 1.6),
                                 opacity=1 if sel else 0.6, legendgroup=r.key,
                                 hovertemplate="%{x:.2f} km<br>%{y:.0f} m<extra>" + r.key + "</extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=r.profile["slope_deg"], showlegend=False,
                                 line=dict(color=r.color, width=2 if sel else 1), opacity=1 if sel else 0.45,
                                 legendgroup=r.key, hovertemplate="%{x:.2f} km<br>%{y:.1f} deg<extra></extra>"),
                      row=2, col=1)
        if sel:
            for pid, name, _, _ in r.metrics.get("visited", []):
                p = poi_by_id(pid)
                d = np.hypot(r.profile["lat"] - p["lat"], r.profile["lon"] - p["lon"])
                i = int(np.argmin(d))
                fig.add_trace(go.Scatter(x=[x[i]], y=[r.profile["elevation_m"][i]], mode="markers+text",
                                         marker=dict(size=10, color="#ffffff", line=dict(color=r.color, width=2)),
                                         text=[name.split("(")[0].strip()[:22]], textposition="top center",
                                         textfont=dict(size=10), showlegend=False,
                                         hovertemplate=name + "<extra></extra>"), row=1, col=1)
    fig.add_hline(y=config.ASSUMPTIONS["caution_slope_deg"], line=dict(color="#d9c39c", dash="dot", width=1),
                  annotation_text="caution", annotation_font_size=10, row=2, col=1)
    fig.add_hline(y=config.ASSUMPTIONS["max_traversable_slope_deg"], line=dict(color="#c98b6b", dash="dash", width=1),
                  annotation_text="limit", annotation_font_size=10, row=2, col=1)
    fig.update_xaxes(title_text="Distance along route (km)", row=2, col=1)
    fig.update_yaxes(title_text="m", row=1, col=1)
    fig.update_yaxes(title_text="°", row=2, col=1)
    return _theme(fig, height=500, margin=dict(l=55, r=20, t=78, b=45),
                  legend=dict(orientation="h", y=1.16, x=0, bgcolor="rgba(0,0,0,0)"))


def tradeoff_radar(routes: list) -> go.Figure:
    ok = [r for r in routes if r.found and r.metrics.get("distance_km")]
    fig = go.Figure()
    if not ok:
        return _theme(fig)
    dmin = min(r.metrics["distance_km"] for r in ok)
    hmin = min(r.metrics["mean_difficulty"] for r in ok)
    emin = min(r.metrics["cumulative_elev_change_m"] for r in ok)
    smax = max(r.metrics["science_score"] for r in ok)
    cats = ["Distance efficiency", "Terrain safety", "Low climbing", "Science value", "Steep-slope avoidance"]
    for r in ok:
        m = r.metrics
        vals = [dmin / m["distance_km"], hmin / max(m["mean_difficulty"], 1e-6),
                emin / max(m["cumulative_elev_change_m"], 1e-6), m["science_score"] / max(smax, 1e-6),
                1 - min(m["pct_above_caution"] / 10, 1)]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=f"{r.key} - {r.name}",
                                      line=dict(color=r.color, width=2), fill="toself", opacity=0.55))
    fig.update_layout(polar=dict(bgcolor="rgba(20,18,20,0.6)",
                                 radialaxis=dict(range=[0, 1.05], gridcolor=GRID, tickfont=dict(size=9)),
                                 angularaxis=dict(gridcolor=GRID)))
    return _theme(fig, height=400, title=dict(text="Relative trade-offs (1 = best of the options)", font=dict(size=13)))


def science_along_route(route) -> go.Figure:
    fig = go.Figure()
    if not route or not route.profile:
        return _theme(fig)
    x = route.profile["distance_m"] / 1000
    fig.add_trace(go.Scatter(x=x, y=route.profile["science"], fill="tozeroy", name="Science value",
                             line=dict(color="#ff7043", width=2), fillcolor="rgba(255,112,67,0.25)"))
    fig.add_trace(go.Scatter(x=x, y=route.profile["difficulty"], name="Terrain difficulty",
                             line=dict(color="#4fc3f7", width=1.5, dash="dot")))
    fig.update_yaxes(range=[0, 1.05], title_text="0 - 1")
    fig.update_xaxes(title_text="Distance along route (km)")
    return _theme(fig, height=300, title=dict(text=f"Route {route.key}: science value vs terrain difficulty",
                                              font=dict(size=13)))


def slope_distribution(routes: list) -> go.Figure:
    fig = go.Figure()
    bins = np.arange(0, 32, 2)
    for r in routes:
        if not r.found or not r.profile:
            continue
        step = np.diff(r.profile["distance_m"], prepend=0)
        hist, _ = np.histogram(r.profile["slope_deg"], bins=bins, weights=step)
        fig.add_trace(go.Bar(x=bins[:-1] + 1, y=hist / 1000, name=f"{r.key}", marker_color=r.color, opacity=0.85))
    fig.add_vline(x=config.ASSUMPTIONS["caution_slope_deg"], line=dict(color="#ff9800", dash="dot"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Slope (deg)")
    fig.update_yaxes(title_text="Distance (km)")
    return _theme(fig, height=300, title=dict(text="Distance driven per slope class", font=dict(size=13)))


def terrain_3d(tm, routes: list, selected: str | None, exaggeration: float = 4.0) -> go.Figure:
    """3D view of the planning-grid DTM with routes draped on it."""
    g = tm.graph_grid
    lats = g.lat_max - (np.arange(g.height) + 0.5) * g.res
    lons = g.lon_min + (np.arange(g.width) + 0.5) * g.res
    z = tm.g_elevation
    fig = go.Figure(go.Surface(x=lons, y=lats, z=z, colorscale=[[0, "#1a1423"], [0.3, "#5a2d4a"],
                                                                 [0.6, "#b5523b"], [0.85, "#e8a061"], [1, "#f6e3c4"]],
                               showscale=False, opacity=0.97, hoverinfo="skip",
                               lighting=dict(ambient=0.55, diffuse=0.8, roughness=0.9, specular=0.1)))
    for r in routes:
        if not r.found or not r.profile:
            continue
        sel = r.key == selected
        fig.add_trace(go.Scatter3d(x=r.profile["lon"][::3], y=r.profile["lat"][::3],
                                   z=r.profile["elevation_m"][::3] + 25, mode="lines",
                                   line=dict(color=r.color, width=8 if sel else 4), name=f"{r.key} - {r.name}"))
    xr = (g.lon_max - g.lon_min) * np.cos(np.deg2rad(18.4)) * 59.27  # km
    yr = (g.lat_max - g.lat_min) * 59.27
    zr = (z.max() - z.min()) / 1000 * exaggeration
    fig.update_layout(scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=yr / xr, z=zr / xr),
                                 xaxis=dict(title="Lon (E)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID),
                                 yaxis=dict(title="Lat (N)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID),
                                 zaxis=dict(title="Elev (m)", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID),
                                 camera=dict(eye=dict(x=-0.9, y=-1.3, z=0.9))))
    return _theme(fig, height=520, title=dict(text=f"Jezero terrain, CTX DTM (vertical exaggeration x{exaggeration:g})",
                                              font=dict(size=13)))


def location_context_chart(info: dict, tm) -> go.Figure:
    """Small bar chart of a location's terrain numbers against regional percentiles."""
    t = info["terrain"]
    metrics = {
        "Elevation": (t["elevation_m"], tm.elevation),
        "Slope": (t["slope_deg"], tm.slope_deg),
        "Roughness": (t["roughness_m"], tm.roughness_m),
        "Difficulty": (t["difficulty"], tm.difficulty),
    }
    names, pct = [], []
    for k, (v, arr) in metrics.items():
        names.append(k)
        pct.append(float((arr < v).mean() * 100))
    for key, label in (("themis_day", "THEMIS day IR"), ("themis_night", "THEMIS night IR")):
        if info["thermal"].get(key) is not None:
            names.append(label)
            pct.append(info["thermal"][key])
    names.append("Science value")
    pct.append(info["science_value"] * 100)
    fig = go.Figure(go.Bar(x=pct, y=names, orientation="h",
                           marker=dict(color=pct, colorscale=[[0, "#4fc3f7"], [0.5, "#ffb74d"], [1, "#ff5722"]],
                                       cmin=0, cmax=100),
                           text=[f"{p:.0f}" for p in pct], textposition="outside"))
    fig.update_xaxes(range=[0, 110], title_text="Percentile within the Jezero study area (science value: 0-100)")
    return _theme(fig, height=280, margin=dict(l=110, r=20, t=30, b=45))
