"""Download the real USGS data for Jezero and store it as the bundled offline snapshot.

Usage:  python scripts/build_bundled_snapshot.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from modules import data_manager as dm

t0 = time.time()
ds = dm.fetch_live_dataset(lambda m, f: print(f"[{f:4.0%}] {m}", flush=True))
ds.provenance["elevation"]["snapshot_note"] = "Bundled snapshot of the live USGS ARD download"
dm.save_dataset(ds, config.BUNDLED_DIR)
for name, ok, detail in dm.validate_dataset(ds):
    print(("PASS" if ok else "WARN"), name, "-", detail)
print("CTX DTMs:", *ds.provenance["elevation"]["items"], sep="\n  ")
print(f"done in {time.time()-t0:.0f}s")
