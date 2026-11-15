"""Build the local high-resolution imagery tiles (CTX + HiRISE) from USGS ARD.

Usage:
    python scripts/build_tiles.py                 # default: CTX z8-11, HiRISE z12-15 on the demo corridor
    python scripts/build_tiles.py --hirise-max-zoom 16   # ~1.3 m/px (about 4x more tiles)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from modules import tiles  # noqa: E402

HIRISE_PAIRS = ["DTEEC_045994_1985_046060_1985_U01", "DTEEC_002387_1985_003798_1985_A01"]


def hirise_urls() -> list[str]:
    import geopandas as gpd
    from modules.data_manager import _cached_index
    idx = gpd.read_parquet(_cached_index(f"{config.USGS_ARD_BUCKET}/collectionindices/mro_hirise_socet_dtms.parquet"))
    urls = []
    for pid in HIRISE_PAIRS:
        assets = idx[idx.id == pid].iloc[0]["assets"]
        orthos = [v["href"] for k, v in assets.items()
                  if v and k.startswith("ortho_") and k[6:].isdigit() and "RED_A" in v["href"]]
        urls += sorted(orthos)[:1]
    return urls


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hirise-max-zoom", type=int, default=15)
    ap.add_argument("--skip-ctx", action="store_true")
    args = ap.parse_args()
    meta = json.loads((config.BUNDLED_DIR / "jezero_dataset.json").read_text())
    ctx_urls = meta["provenance"]["ctx_image"]["urls"]
    if not args.skip_ctx:
        print(tiles.build_ctx_tiles(ctx_urls))
    print(tiles.build_hirise_tiles(hirise_urls(), ctx_urls, zooms=range(12, args.hirise_max_zoom + 1)))
    print("landing image:", tiles.build_landing_image())
