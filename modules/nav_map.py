"""2D navigation map for the mission flow (Folium / Leaflet, EPSG:4326).

Differences from the full map in mars_map.py:
* fewer layers, each switched by the flow's layer toggles;
* distinct symbols: start (green triangle), destination (flag), numbered stops, science targets;
* the map cannot be panned or zoomed far outside the Jezero study area, and empty areas are dark,
  not grey;
* in-map controls (keyboard focusable): zoom in/out (Leaflet), "fit route" and "centre on the
  simulated rover";
* the selected route is drawn with a short stroke animation, and the view eases in once per
  render; both are skipped when the browser asks for reduced motion;
* the simulated rover is a separate layer (``rover_layer``) that streamlit-folium swaps in
  without rebuilding the map.
"""
from __future__ import annotations

import json

import folium
from branca.element import MacroElement
from jinja2 import Template

from modules import tiles
from modules.mars_map import LAYER_SPECS

MAP_CSS = """
<style>
.leaflet-container{background:#121417!important;font-family:Inter,system-ui,sans-serif}
.leaflet-control-attribution{font-size:10px!important;background:rgba(17,19,22,.82)!important;color:#c3c9cf!important;
  max-width:72%;white-space:normal;line-height:1.3;padding:2px 6px!important;border-radius:6px 0 0 0}
.leaflet-control-attribution a{color:#a9c9c3!important}
.leaflet-bar{border:1px solid rgba(255,255,255,.10)!important;border-radius:10px!important;overflow:hidden;
  box-shadow:0 4px 14px rgba(0,0,0,.35)!important}
.leaflet-bar a,.leaflet-bar a:hover{background:rgba(24,27,31,.92);color:#e4e7ea;border-bottom-color:rgba(255,255,255,.08)}
.leaflet-bar a:hover{background:#22272d}
.leaflet-bar a:focus-visible,.nav-ctl a:focus-visible{outline:2px solid #7fb8ae;outline-offset:1px}
.nav-ctl a{display:block;width:auto!important;min-width:30px;padding:0 10px;font:500 11px/30px Inter,sans-serif;
  text-align:center;text-decoration:none;white-space:nowrap}
.route-draw{transition:stroke-dashoffset 1.3s cubic-bezier(.4,0,.2,1)}
.nav-rover-wrap{animation:rov .45s ease-out}
@keyframes rov{from{opacity:0;transform:scale(.7)}to{opacity:1;transform:scale(1)}}
.nav-rover{width:16px;height:16px;border-radius:50%;background:#d9b48c;border:3px solid #fff;
  box-shadow:0 0 0 2px #15171a,0 0 0 7px rgba(217,180,140,.25)}
.nav-label{font:500 11px Inter,sans-serif;color:#f1f3f4;text-shadow:0 0 3px #000,0 0 3px #000;white-space:nowrap}
.nav-inspect{width:18px;height:18px;border:2px solid #e4e7ea;border-radius:50%;box-shadow:0 0 0 2px #15171a}
@media (prefers-reduced-motion: reduce){.route-draw{transition:none}.nav-rover-wrap{animation:none}}
</style>
"""

FILL_CSS = """<style>html,body{height:100%;margin:0;overflow:hidden}
#map_div,#map_div2,.folium-map{height:100vh!important}</style>"""

class _NavControls(MacroElement):
    """Fit-route / centre-rover buttons, route stroke animation and a gentle initial zoom."""
    _template = Template("""
{% macro script(this, kwargs) %}
(function(){
  var map = {{ this._parent.get_name() }};
  var fit = {{ this.fit_json }};
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function rover(){ var ll=null; map.eachLayer(function(l){
      if(l.options && l.options.icon && l.options.icon.options &&
         String(l.options.icon.options.className||'').indexOf('nav-rover-wrap')>=0){ ll=l.getLatLng(); } });
    return ll; }
  var Ctl = L.Control.extend({options:{position:'topleft'}, onAdd:function(){
    var box=L.DomUtil.create('div','leaflet-bar nav-ctl');
    function btn(txt,title,fn){ var a=L.DomUtil.create('a','',box); a.href='#'; a.innerHTML=txt; a.title=title;
      a.setAttribute('role','button'); a.setAttribute('aria-label',title);
      L.DomEvent.on(a,'click',function(e){L.DomEvent.preventDefault(e);L.DomEvent.stopPropagation(e);fn();});
      L.DomEvent.on(a,'keydown',function(e){ if(e.key===' '){L.DomEvent.preventDefault(e);fn();} }); }
    if(fit){ btn({{ this.fit_label }},{{ this.fit_label }},function(){
      map.flyToBounds(fit,{padding:[24,24],duration:reduce?0:0.6,animate:!reduce});}); }
    {% if this.rover_control %}btn('◎ Rover','Centre on the simulated rover position',function(){ var ll=rover();
      if(ll){ map.flyTo(ll, Math.max(map.getZoom(),12), {duration:reduce?0:0.6,animate:!reduce}); }
      else if(fit){ map.flyToBounds(fit,{padding:[24,24],animate:!reduce}); } });{% endif %}
    L.DomEvent.disableClickPropagation(box); return box; }});
  map.addControl(new Ctl());
  window.addEventListener('resize', function(){ map.invalidateSize(); });
  setTimeout(function(){
    map.invalidateSize();
    if(fit){ map.fitBounds(fit,{padding:[24,24],animate:false}); }
    if(!reduce){ var z=map.getZoom(); if(z>map.getMinZoom()){ map.setZoom(z-1,{animate:false});
      setTimeout(function(){ map.setZoom(z,{animate:true}); }, 60); } }
    document.querySelectorAll('path.route-draw').forEach(function(p){
      if(reduce || !p.getTotalLength) return;
      var n=p.getTotalLength(); p.style.transition='none'; p.style.strokeDasharray=n; p.style.strokeDashoffset=n;
      p.getBoundingClientRect(); p.style.transition=''; p.style.strokeDashoffset=0;
      setTimeout(function(){ p.style.strokeDasharray=''; }, 1600); });
  }, 30);
})();
{% endmacro %}
""")

    def __init__(self, fit_bounds, fit_label: str = "Fit route", rover_control: bool = False):
        super().__init__()
        self._name = "NavControls"
        self.fit_json = json.dumps(fit_bounds)
        self.fit_label = json.dumps("⤢ " + fit_label)
        self.rover_control = rover_control


def _div(html: str, size=(26, 26), anchor=None, cls="") -> folium.DivIcon:
    return folium.DivIcon(html=html, icon_size=size, icon_anchor=anchor or (size[0] // 2, size[1] // 2),
                          class_name=cls or "nav-icon")


START_HTML = ("<svg width='26' height='26' viewBox='0 0 26 26' aria-label='Start'><circle cx='13' cy='13' r='11' "
              "fill='#4f8f86' stroke='#fff' stroke-width='2.5'/><path d='M10 7.5 L18.5 13 L10 18.5 Z' fill='#fff'/></svg>")
DEST_HTML = ("<svg width='30' height='34' viewBox='0 0 30 34' aria-label='Destination'><path d='M8 32 V4' stroke='#fff' "
             "stroke-width='3'/><path d='M8 32 V4' stroke='#111' stroke-width='1'/><path d='M9 4 H26 L21 10 L26 16 H9 Z' "
             "fill='#c98b6b' stroke='#fff' stroke-width='1.5'/></svg>")


def stop_html(n: int, color: str, name: str) -> str:
    return (f"<div style='display:flex;align-items:center;gap:6px'><div style='width:22px;height:22px;"
            f"border-radius:7px;background:{color};color:#111;font:600 11px Inter,sans-serif;display:flex;"
            f"align-items:center;justify-content:center;border:2px solid #15171a'>{n}</div>"
            f"<div class='nav-label'>{name}</div></div>")


def build(ds, layers: dict, *, routes=None, selected: str | None = None, start=None, destination=None,
          stops=None, pois=None, show=None, route_bounds=None, recommended: str | None = None,
          dim_others: bool = True, rover_control: bool = False, inspect=None, fill: bool = False) -> folium.Map:
    """``fill``: the map fills its frame's full height (the page sizes the frame to the viewport)."""
    """``show``: set of toggles from {"imagery", "elevation", "slope", "targets", "hazards", "route"}."""
    show = set(show if show is not None else {"imagery", "targets", "route"})
    g = ds.fine_grid
    bounds = [[g.lat_min, g.lon_min], [g.lat_max, g.lon_max]]
    m = folium.Map(location=[(g.lat_min + g.lat_max) / 2, (g.lon_min + g.lon_max) / 2], zoom_start=10,
                   crs="EPSG4326", tiles=None, min_zoom=9, max_zoom=17, control_scale=False,
                   prefer_canvas=False, attribution_control=True, max_bounds=True,
                   min_lat=g.lat_min - 0.08, max_lat=g.lat_max + 0.08,
                   min_lon=g.lon_min - 0.12, max_lon=g.lon_max + 0.12, max_bounds_viscosity=0.9)
    m.get_root().header.add_child(folium.Element(MAP_CSS + (FILL_CSS if fill else "")))

    ctx_info, hi_info = tiles.layer_info("ctx"), tiles.layer_info("hirise")
    if "imagery" in show:
        if ctx_info:
            folium.TileLayer(tiles=tiles.url_template("ctx"), name="CTX", no_wrap=True,
                             attr="MRO CTX: NASA/JPL-Caltech/MSSS · USGS Astrogeology ARD",
                             min_zoom=8, min_native_zoom=ctx_info["min_zoom"], max_native_zoom=ctx_info["max_zoom"],
                             max_zoom=17, bounds=[[ctx_info["bounds"][1], ctx_info["bounds"][0]],
                                                 [ctx_info["bounds"][3], ctx_info["bounds"][2]]]).add_to(m)
            if hi_info:
                hb = hi_info["bounds"]
                folium.TileLayer(tiles=tiles.url_template("hirise"), name="HiRISE", no_wrap=True,
                                 attr="MRO HiRISE: NASA/JPL-Caltech/UArizona",
                                 min_zoom=hi_info["min_zoom"], min_native_zoom=hi_info["min_zoom"],
                                 max_native_zoom=hi_info["max_zoom"], max_zoom=17,
                                 bounds=[[hb[1], hb[0]], [hb[3], hb[2]]]).add_to(m)
        elif "ctx" in layers:
            folium.raster_layers.ImageOverlay(layers["ctx"], bounds=bounds,
                                              attr="MRO CTX: NASA/JPL-Caltech/MSSS · USGS ARD").add_to(m)
    else:
        folium.raster_layers.ImageOverlay(layers["hillshade"], bounds=bounds, opacity=0.9,
                                          attr="Hillshade derived from the USGS CTX DTM").add_to(m)
    if "elevation" in show:
        folium.raster_layers.ImageOverlay(layers["elevation"], bounds=bounds, opacity=0.6).add_to(m)
    if "slope" in show:
        folium.raster_layers.ImageOverlay(layers["slope"], bounds=bounds, opacity=0.75).add_to(m)
    if "hazards" in show:
        folium.raster_layers.ImageOverlay(layers["impassable"], bounds=bounds, opacity=0.85).add_to(m)

    if "targets" in show and pois:
        stop_ids = {poi["id"] for _, poi, _ in (stops or [])}
        for p in pois:
            if p["id"] in stop_ids:
                continue
            col = "#d8cfe9" if p["origin"] == "published" else "#9aa3ab"
            folium.CircleMarker([p["lat"], p["lon"]], radius=5, color="#15171a", weight=1.5, fill=True,
                                fill_color=col, fill_opacity=0.95,
                                tooltip=f"Science target: {p['name']} (click for details)").add_to(m)

    if routes and "route" in show:
        for r in sorted(routes, key=lambda r: r.key == selected):
            if not r.found or r.lats is None:
                continue
            sel = r.key == selected
            if not sel and not dim_others:
                continue
            pts = list(zip(r.lats.tolist(), r.lons.tolist()))
            tip = ("★ Recommended · " if r.key == recommended else "") + f"Route {r.key}: {r.metrics['distance_km']:.2f} km"
            if sel:
                folium.PolyLine(pts, color="#0d0f11", weight=9, opacity=0.55).add_to(m)
            folium.PolyLine(pts, color=r.color, weight=5 if sel else 2.5, opacity=1.0 if sel else 0.5,
                            dash_array=None if sel else "5 7", tooltip=tip,
                            class_name="route-draw" if sel else "route-alt").add_to(m)
        sel_r = next((r for r in routes if r.key == selected and r.found), None)
        if sel_r is not None:
            for n, poi, _ in stops or []:
                folium.Marker([poi["lat"], poi["lon"]],
                              icon=_div(stop_html(n, sel_r.color, poi["name"].split(" (")[0]), (200, 24), (11, 12)),
                              tooltip=f"Stop {n}: {poi['name']} (click for details)").add_to(m)

    if inspect:
        folium.Marker(list(inspect), icon=_div("<div class='nav-inspect'></div>", (22, 22)),
                      tooltip="Inspected point").add_to(m)
    if start:
        folium.Marker(start[:2], icon=_div(START_HTML), tooltip=f"Start: {start[2]}").add_to(m)
    if destination:
        folium.Marker(destination[:2], icon=_div(DEST_HTML, (30, 34), (8, 32)),
                      tooltip=f"Destination: {destination[2]}").add_to(m)

    fit = route_bounds
    if fit is None and start and destination:
        fit = _pad([start[:2], destination[:2]])
    if fit is None:
        fit = bounds
    m.fit_bounds(fit)
    has_route = bool(routes) and "route" in show
    _NavControls(fit, "Fit route" if has_route else "Fit view", rover_control).add_to(m)
    return m


def route_fit_bounds(route, start, destination):
    pts = [start[:2], destination[:2]]
    if route is not None and route.found and route.lats is not None:
        pts += list(zip(route.lats.tolist(), route.lons.tolist()))
    return _pad(pts)


def _pad(points, frac=0.08, min_pad=0.01):
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    pl = max((max(lats) - min(lats)) * frac, min_pad)
    pn = max((max(lons) - min(lons)) * frac, min_pad)
    return [[min(lats) - pl, min(lons) - pn], [max(lats) + pl, max(lons) + pn]]


def rover_layer(state, route) -> folium.FeatureGroup:
    """Dynamic layer: SIMULATED rover marker and the driven part of the route."""
    fg = folium.FeatureGroup(name="Simulated rover")
    if state is None:
        return fg
    p = route.profile
    if state.index > 0:
        folium.PolyLine(list(zip(p["lat"][: state.index + 1].tolist(), p["lon"][: state.index + 1].tolist())),
                        color="#ffffff", weight=4, opacity=0.95, tooltip="Driven (SIMULATED)").add_to(fg)
    folium.Marker([state.lat, state.lon], tooltip="SIMULATED rover position (no live rover connection)",
                  icon=_div("<div class='nav-rover' role='img' aria-label='Simulated rover'></div>", (26, 26),
                            cls="nav-rover-wrap")).add_to(fg)
    return fg


def legend_html(show: set, items: tuple = ("start", "dest", "stop", "target", "rover")) -> str:
    bits = ["<span class='lg'><svg width='14' height='14' viewBox='0 0 26 26'><circle cx='13' cy='13' r='11' "
            "fill='#4f8f86' stroke='#fff' stroke-width='2.5'/><path d='M10 7.5 L18.5 13 L10 18.5 Z' fill='#fff'/></svg> Start</span>",
            "<span class='lg'><svg width='12' height='14' viewBox='0 0 30 34'><path d='M9 4 H26 L21 10 L26 16 H9 Z' "
            "fill='#c98b6b'/><path d='M8 32 V4' stroke='#fff' stroke-width='3'/></svg> Destination</span>",
            "<span class='lg'><i class='sq'>1</i> Numbered stop</span>",
            "<span class='lg'><i class='dot'></i> Science target</span>",
            "<span class='lg'><i class='rov'></i> Simulated rover</span>"]
    bits = [b for b, k in zip(bits, ("start", "dest", "stop", "target", "rover")) if k in items]
    if "slope" in show:
        bits.append(f"<span class='lg'>Slope: {LAYER_SPECS['slope'][2]}</span>")
    if "hazards" in show:
        bits.append("<span class='lg'><i class='haz'></i> Over the slope limit</span>")
    if "elevation" in show:
        bits.append("<span class='lg'>Elevation: dark low → pale high</span>")
    return "<div class='nav-legend'>" + "".join(bits) + "</div>"
