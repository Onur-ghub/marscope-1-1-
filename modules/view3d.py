"""3D view: Mars globe -> fly-in -> Jezero terrain built from the real CTX DTM (three.js).

What is real and what is not:
* Terrain mesh: the USGS CTX DTM used everywhere in MARSCOPE, downsampled for the browser
  (vertical exaggeration is stated on screen).
* Terrain texture: the CTX orthoimage mosaic (USGS ARD). HiRISE is not used in 3D.
* Globe texture: NASA Trek Viking MDIM 2.1 colour mosaic tiles, downloaded once and cached. If
  they cannot be downloaded (offline), the globe is drawn untextured and says so; nothing is
  painted in to look like imagery.
* Route, stops, start/destination: the planned route; the rover marker is SIMULATED.

three.js r128 (MIT) is served from the app's own static folder, with cdnjs as a fallback.
"""
from __future__ import annotations

import base64
import io
import json
import math

import numpy as np
import requests
from PIL import Image

import config

THREE_LOCAL = "/app/static/vendor/three.min.js"
THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
GLOBE_CACHE = config.CACHE_DIR / "globe_viking_mdim21_1024x512.jpg"
VERTICAL_EXAGGERATION = 3.0
MESH_STEP = 5          # every 5th DTM sample (~300 m) -> ~150 x 180 vertices
KM_PER_DEG = 2 * math.pi * config.MARS_RADIUS_M / 1000 / 360


def _jpeg_url(img: Image.Image, quality=80) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def globe_texture() -> tuple[str | None, str]:
    """(data URL or None, status). Real Viking mosaic tiles only; None when unavailable."""
    if GLOBE_CACHE.exists():
        return _jpeg_url(Image.open(GLOBE_CACHE)), "Viking MDIM 2.1 colour mosaic (NASA/USGS, via NASA Trek; cached)"
    if config.OFFLINE_MODE:
        return None, "unavailable (offline mode)"
    url = next(u for k, (u, _) in config.TREK_LAYERS.items() if "Viking" in k)
    z, nx, ny = 1, 4, 2                        # Trek EQ tile matrix: 2^(z+1) x 2^z tiles of 256 px
    canvas = Image.new("RGB", (nx * 256, ny * 256))
    try:
        for y in range(ny):
            for x in range(nx):
                r = requests.get(url.format(z=z, x=x, y=y), timeout=6)
                r.raise_for_status()
                canvas.paste(Image.open(io.BytesIO(r.content)).convert("RGB").resize((256, 256)), (x * 256, y * 256))
    except Exception as exc:                    # network blocked or tiles missing: no texture
        return None, f"unavailable ({type(exc).__name__}); globe drawn untextured"
    GLOBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(GLOBE_CACHE, quality=85)
    return _jpeg_url(canvas), "Viking MDIM 2.1 colour mosaic (NASA/USGS, via NASA Trek)"


def terrain_payload(ds) -> dict:
    """Downsampled DTM heights (metres) and CTX texture for the Jezero mesh."""
    g = ds.fine_grid
    z = ds.elevation[::MESH_STEP, ::MESH_STEP].astype(np.float32)
    lat_c, lon_c = (g.lat_min + g.lat_max) / 2, (g.lon_min + g.lon_max) / 2
    kx = KM_PER_DEG * math.cos(math.radians(lat_c))
    tex = None
    if ds.ctx_image is not None:
        a = ds.ctx_image.astype(np.float32)
        rgb = np.dstack([a, a * 0.93, a * 0.85]).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        tex = _jpeg_url(img.resize((1024, int(1024 * img.height / img.width)), Image.BILINEAR), 78)
    cell_m = g.cell_size_m()[0] * MESH_STEP
    return {"nx": int(z.shape[1]), "ny": int(z.shape[0]), "h": np.round(z).astype(int).ravel().tolist(),
            "zmid": float(np.median(z)), "w_km": (g.lon_max - g.lon_min) * kx,
            "d_km": (g.lat_max - g.lat_min) * KM_PER_DEG, "lat_c": lat_c, "lon_c": lon_c, "kx": kx,
            "ky": KM_PER_DEG, "tex": tex, "cell_m": cell_m, "exag": VERTICAL_EXAGGERATION,
            "synthetic": bool(ds.is_demonstration)}


def scene_payload(route=None, start=None, destination=None, stops=None, rover=None) -> dict:
    out = {"route": None, "markers": []}
    if route is not None and route.found and route.profile:
        p = route.profile
        k = max(1, len(p["lat"]) // 600)
        out["route"] = {"color": route.color, "pts": [[round(float(a), 5), round(float(b), 5), round(float(e))]
                                                      for a, b, e in zip(p["lat"][::k], p["lon"][::k],
                                                                         p["elevation_m"][::k])]}
    if start:
        out["markers"].append({"kind": "start", "lat": start[0], "lon": start[1], "label": "Start"})
    if destination:
        out["markers"].append({"kind": "dest", "lat": destination[0], "lon": destination[1], "label": "Destination"})
    for n, poi, _ in stops or []:
        out["markers"].append({"kind": "stop", "lat": poi["lat"], "lon": poi["lon"], "label": str(n)})
    if rover is not None:
        out["markers"].append({"kind": "rover", "lat": rover.lat, "lon": rover.lon, "label": "Simulated rover"})
    return out


def html(terrain: dict, scene: dict, globe_url: str | None, globe_status: str, intro: bool,
         height: int = 600) -> str:
    data = json.dumps({"t": terrain, "s": scene, "globe": globe_url, "intro": intro,
                       "jezero": list(config.JEZERO_CENTER)})
    globe_note = ("Globe: " + globe_status)
    return TEMPLATE.replace("__DATA__", data).replace("__H__", str(height)) \
        .replace("__THREE_LOCAL__", THREE_LOCAL).replace("__THREE_CDN__", THREE_CDN) \
        .replace("__GLOBE_NOTE__", globe_note.replace("<", "&lt;"))


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;height:100%;background:#0f1114;color:#ece6e1;font-family:Inter,system-ui,sans-serif;overflow:hidden}
#wrap{position:relative;width:100%;height:100vh;min-height:320px}
canvas{display:block;outline:none}
canvas:focus-visible{outline:2px solid #7fb8ae;outline-offset:-2px}
#fade{position:absolute;inset:0;background:#0f1114;opacity:0;pointer-events:none;transition:opacity .4s ease}
@media (prefers-reduced-motion: reduce){#fade{transition:none}}
.hud{position:absolute;left:12px;top:10px;font-size:12px;line-height:1.45;background:rgba(15,17,20,.72);
  border:1px solid rgba(255,255,255,.12);border-radius:8px;padding:8px 10px;max-width:min(360px,calc(100% - 40px))}
.hud b{color:#a9c9c3;font-weight:600;letter-spacing:.04em}
.sim{display:inline-block;font:700 9px/1 'JetBrains Mono',monospace;letter-spacing:.08em;padding:3px 5px;border-radius:4px;
  background:#2d2a3d;color:#cfc4ee;margin-left:4px}
.btns{position:absolute;right:12px;top:10px;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
button{font:600 12px Inter,sans-serif;color:#ece6e1;background:#22262b;border:1px solid #3a3f45;border-radius:6px;
  padding:6px 10px;cursor:pointer}
button:hover{background:#2c3137}button:focus-visible{outline:2px solid #7fb8ae;outline-offset:1px}
#msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;padding:24px;
  font-size:14px;line-height:1.5;color:#d8cfc6}
.foot{position:absolute;left:12px;bottom:8px;right:12px;font-size:10.5px;color:#a49c93;line-height:1.35}
#phase{color:#a9c9c3}
</style></head><body><div id="wrap">
<div id="fade"></div>
<div class="hud" id="hud"><b>3D VIEW</b> · <span id="phase">loading…</span><div id="hudtxt"></div></div>
<div class="btns"><button id="replay" aria-label="Replay the globe to Jezero transition">↻ Globe fly-in</button>
<button id="reset" aria-label="Reset the terrain camera">⌖ Reset view</button></div>
<div id="msg" role="status"></div>
<div class="foot" id="foot">Drag to orbit · scroll or +/- to zoom · arrow keys rotate when the view is focused.
 __GLOBE_NOTE__.</div>
</div>
<script>
var D = __DATA__;
function fail(t){var m=document.getElementById('msg');m.style.display='flex';m.innerHTML=t;
  document.getElementById('phase').textContent='unavailable';}
function load(src, ok, bad){var s=document.createElement('script');s.src=src;s.onload=ok;s.onerror=bad;document.head.appendChild(s);}
load('__THREE_LOCAL__', start, function(){ load('__THREE_CDN__', start, function(){
  fail('The 3D view needs the three.js library, which could not be loaded (it is served from the app\'s static '+
       'folder, with cdnjs as a fallback). Use the 2D map; everything shown here is also there.'); }); });

function start(){
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var wrap=document.getElementById('wrap'), W=wrap.clientWidth, H=wrap.clientHeight;
  var renderer;
  try{ renderer=new THREE.WebGLRenderer({antialias:true}); }
  catch(e){ fail('This browser could not start WebGL, so the 3D view is unavailable. Use the 2D map.'); return; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2)); renderer.setSize(W,H);
  renderer.domElement.tabIndex=0; renderer.domElement.setAttribute('aria-label','3D view of Jezero crater');
  wrap.insertBefore(renderer.domElement, wrap.firstChild);
  var camera=new THREE.PerspectiveCamera(45,W/H,0.001,500);
  var T=D.t, S=D.s, deg=Math.PI/180;

  // ---------- globe scene (radius 1) ----------
  var gScene=new THREE.Scene();
  gScene.add(new THREE.AmbientLight(0xffffff,0.35));
  var sun=new THREE.DirectionalLight(0xffffff,1.0); sun.position.set(-3,1.5,4); gScene.add(sun);
  function onSphere(lat,lon,r){var phi=(lon+180)*deg, th=(90-lat)*deg;
    return new THREE.Vector3(-r*Math.cos(phi)*Math.sin(th), r*Math.cos(th), r*Math.sin(phi)*Math.sin(th));}
  var gMat=new THREE.MeshPhongMaterial({color:D.globe?0xffffff:0x8a6a5c, shininess:4});
  if(D.globe){ new THREE.TextureLoader().load(D.globe,function(tx){gMat.map=tx;gMat.needsUpdate=true;}); }
  gScene.add(new THREE.Mesh(new THREE.SphereGeometry(1,96,48),gMat));
  if(!D.globe){ var gl=new THREE.LineBasicMaterial({color:0xc9d3dc,transparent:true,opacity:.25});
    for(var la=-60;la<=60;la+=30){var pts=[];for(var lo=0;lo<=360;lo+=5)pts.push(onSphere(la,lo,1.002));
      gScene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),gl));}
    for(var lo2=0;lo2<360;lo2+=30){var p2=[];for(var la2=-90;la2<=90;la2+=5)p2.push(onSphere(la2,lo2,1.002));
      gScene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p2),gl));} }
  var jz=onSphere(D.jezero[0],D.jezero[1],1.0);
  var ring=new THREE.Mesh(new THREE.RingGeometry(0.018,0.026,40),
    new THREE.MeshBasicMaterial({color:0xd9c39c,side:THREE.DoubleSide}));
  ring.position.copy(jz.clone().multiplyScalar(1.003)); ring.lookAt(jz.clone().multiplyScalar(2)); gScene.add(ring);

  // ---------- terrain scene (km) ----------
  var tScene=new THREE.Scene(); tScene.background=new THREE.Color(0x0f1114);
  tScene.add(new THREE.HemisphereLight(0xfff4ea,0x2a2420,0.55));
  var tsun=new THREE.DirectionalLight(0xffffff,0.9); tsun.position.set(-30,40,-20); tScene.add(tsun);
  var geo=new THREE.PlaneGeometry(T.w_km,T.d_km,T.nx-1,T.ny-1); geo.rotateX(-Math.PI/2);
  var pos=geo.attributes.position;
  function hy(m){return (m-T.zmid)/1000*T.exag;}
  for(var i=0;i<pos.count;i++){pos.setY(i,hy(T.h[i]));}
  geo.computeVertexNormals();
  var tMat=new THREE.MeshLambertMaterial({color:T.tex?0xffffff:0xb07a5a});
  if(T.tex){ new THREE.TextureLoader().load(T.tex,function(tx){tMat.map=tx;tMat.needsUpdate=true;}); }
  tScene.add(new THREE.Mesh(geo,tMat));
  function xyz(lat,lon,elev,lift){return new THREE.Vector3((lon-T.lon_c)*T.kx, hy(elev)+(lift||0), -(lat-T.lat_c)*T.ky);}
  function heightAt(lat,lon){var c=Math.round((lon-(T.lon_c-T.w_km/2/T.kx))/(T.w_km/T.kx)*(T.nx-1));
    var r=Math.round(((T.lat_c+T.d_km/2/T.ky)-lat)/(T.d_km/T.ky)*(T.ny-1));
    c=Math.max(0,Math.min(T.nx-1,c)); r=Math.max(0,Math.min(T.ny-1,r)); return T.h[r*T.nx+c];}
  if(S.route){ var rp=S.route.pts.map(function(p){return xyz(p[0],p[1],p[2],0.08);});
    var tube=new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rp),Math.min(rp.length*2,1200),0.09,6,false);
    tScene.add(new THREE.Mesh(tube,new THREE.MeshBasicMaterial({color:new THREE.Color(S.route.color)}))); }
  var COLORS={start:0x4f8f86,dest:0xc98b6b,stop:0xd8cfe9,rover:0xd9b48c};
  S.markers.forEach(function(mk){ var base=xyz(mk.lat,mk.lon,heightAt(mk.lat,mk.lon),0);
    var hgt=mk.kind==='stop'?0.8:1.4;
    var pole=new THREE.Mesh(new THREE.CylinderGeometry(0.03,0.03,hgt,6),new THREE.MeshBasicMaterial({color:0xffffff}));
    pole.position.copy(base).add(new THREE.Vector3(0,hgt/2,0)); tScene.add(pole);
    var head=mk.kind==='rover'?new THREE.OctahedronGeometry(0.32):mk.kind==='dest'?new THREE.ConeGeometry(0.28,0.5,4):
      mk.kind==='start'?new THREE.ConeGeometry(0.26,0.5,3):new THREE.SphereGeometry(0.2,12,8);
    var hm=new THREE.Mesh(head,new THREE.MeshBasicMaterial({color:COLORS[mk.kind]}));
    hm.position.copy(base).add(new THREE.Vector3(0,hgt+0.2,0)); if(mk.kind!=='rover')hm.rotation.x=Math.PI; tScene.add(hm);});

  // ---------- camera control ----------
  var orbit={az:-0.6, el:0.75, dist:Math.max(T.w_km,T.d_km)*1.05, target:new THREE.Vector3(0,0,0)};
  var routeCenter=null; if(S.route){var bb=new THREE.Box3();S.route.pts.forEach(function(p){bb.expandByPoint(xyz(p[0],p[1],p[2]));});
    routeCenter=bb.getCenter(new THREE.Vector3()); var sz=bb.getSize(new THREE.Vector3());
    orbit.target.copy(routeCenter); orbit.dist=Math.max(8,Math.max(sz.x,sz.z)*1.9);}
  var home={az:orbit.az,el:orbit.el,dist:orbit.dist};
  function placeTerrainCam(){ var t=orbit.target;
    camera.position.set(t.x+orbit.dist*Math.cos(orbit.el)*Math.sin(orbit.az), t.y+orbit.dist*Math.sin(orbit.el),
                        t.z+orbit.dist*Math.cos(orbit.el)*Math.cos(orbit.az)); camera.lookAt(t); }
  var phase='terrain', t0=0, FLY=2600;
  var phaseEl=document.getElementById('phase'), hudtxt=document.getElementById('hudtxt');
  function setHud(){ if(phase==='terrain'){ phaseEl.textContent='Jezero crater terrain';
      hudtxt.innerHTML='CTX DTM mesh (~'+Math.round(T.cell_m)+' m spacing), vertical exaggeration ×'+T.exag+
      (T.synthetic?' · <b>SYNTHETIC demo terrain</b>':'')+'<br>Texture: CTX orthoimage · route and stops as planned'+
      (S.markers.some(function(m){return m.kind==='rover';})?'<br>◆ Rover position <span class="sim">SIMULATED</span>':'');}
    else { phaseEl.textContent='Mars → Jezero crater'; hudtxt.innerHTML='Jezero: 18.38° N, 77.58° E'+(D.globe?'':' · globe untextured (no global imagery loaded)'); } }
  var fade=document.getElementById('fade');
  function startFly(){ phase='globe'; t0=performance.now(); fade.style.opacity=0; setHud(); }
  function toTerrain(){ fade.style.opacity=1; setTimeout(function(){ phase='terrain'; placeTerrainCam(); setHud();
      fade.style.opacity=0; }, reduce?0:450); phase='fading'; }
  var startDir=onSphere(D.jezero[0]-12,D.jezero[1]-55,1).normalize();
  function globeCam(k){ var e=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;
    var dir=startDir.clone().lerp(jz.clone().normalize(),e).normalize();
    var r=3.4+(1.06-3.4)*e; camera.position.copy(dir.multiplyScalar(r)); camera.lookAt(jz.clone().multiplyScalar(1-e*0.0));
    camera.up.set(0,1,0); }

  var drag=null; var el=renderer.domElement;
  el.addEventListener('pointerdown',function(e){ if(phase!=='terrain')return; drag=[e.clientX,e.clientY]; el.setPointerCapture(e.pointerId);});
  el.addEventListener('pointermove',function(e){ if(!drag)return; orbit.az-=(e.clientX-drag[0])*0.006;
    orbit.el=Math.max(0.12,Math.min(1.45,orbit.el+(e.clientY-drag[1])*0.005)); drag=[e.clientX,e.clientY]; placeTerrainCam();});
  el.addEventListener('pointerup',function(){drag=null;});
  el.addEventListener('wheel',function(e){ if(phase!=='terrain')return; e.preventDefault();
    orbit.dist=Math.max(2,Math.min(120,orbit.dist*(e.deltaY>0?1.1:0.9))); placeTerrainCam();},{passive:false});
  el.addEventListener('keydown',function(e){ if(phase!=='terrain')return; var k=e.key;
    if(k==='ArrowLeft')orbit.az+=0.08; else if(k==='ArrowRight')orbit.az-=0.08;
    else if(k==='ArrowUp')orbit.el=Math.min(1.45,orbit.el+0.06); else if(k==='ArrowDown')orbit.el=Math.max(0.12,orbit.el-0.06);
    else if(k==='+'||k==='=')orbit.dist=Math.max(2,orbit.dist*0.9); else if(k==='-')orbit.dist=Math.min(120,orbit.dist*1.1);
    else return; e.preventDefault(); placeTerrainCam();});
  document.getElementById('replay').onclick=function(){ if(reduce){ phase='terrain'; placeTerrainCam(); setHud(); return; } startFly(); };
  document.getElementById('reset').onclick=function(){ orbit.az=home.az; orbit.el=home.el; orbit.dist=home.dist;
    if(routeCenter)orbit.target.copy(routeCenter); phase='terrain'; placeTerrainCam(); setHud(); };
  window.addEventListener('resize',function(){W=wrap.clientWidth;H=wrap.clientHeight;renderer.setSize(W,H);camera.aspect=W/H;camera.updateProjectionMatrix();});

  if(D.intro && !reduce){ startFly(); } else { placeTerrainCam(); setHud(); }
  window.__marscope3d={ready:true, phase:function(){return phase;}};
  (function loop(now){ requestAnimationFrame(loop);
    if(phase==='globe'){ var k=Math.min(1,(now-t0)/FLY); globeCam(k); renderer.render(gScene,camera); if(k>=1)toTerrain(); }
    else if(phase==='fading'){ renderer.render(gScene,camera); }
    else { renderer.render(tScene,camera); } })(performance.now());
}
</script></body></html>"""
