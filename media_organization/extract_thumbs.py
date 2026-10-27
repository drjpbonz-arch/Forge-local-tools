"""
extract_thumbs.py — Pull small preview thumbnails for untagged clusters so Claude
can visually classify them. Read-only on source media.

Requires ffmpeg (https://ffmpeg.org or `winget install ffmpeg`).

Usage:
    python extract_thumbs.py            # thumbs for all untagged clusters
    python extract_thumbs.py --cluster 2025-01-08_sd03   # one cluster only
    python extract_thumbs.py --per-cluster 5             # max thumbs per cluster (default 4)

Output: 01_inventory/thumbs/<cluster_id>/<filename>.jpg  (~50 KB each)
Note: the source card must be plugged in (paths in the catalog must be reachable).
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# --- Project layout -------------------------------------------------------- #
INV = Path(__file__).resolve().parent.parent / "01_inventory"
CATALOG = INV / "master_catalog.csv"
THUMBS = INV / "thumbs"

def make_thumb(src: Path, dst: Path) -> bool:
    """Render one ~480px JPG thumbnail via ffmpeg. Returns True if the thumb exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True  # resumable: never regenerate an existing thumb
    try:
        if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp", ".webp"}:
            # Still image: just scale it down to a preview.
            cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", "scale=480:-1", "-frames:v", "1", str(dst)]
        else:  # video: grab a frame 10 seconds in (or first frame for short clips)
            cmd = ["ffmpeg", "-y", "-ss", "10", "-i", str(src), "-vf", "scale=480:-1",
                   "-frames:v", "1", str(dst)]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if not dst.exists():  # retry at 0s (clip shorter than 10s)
            cmd[2] = "0"
            subprocess.run(cmd, capture_output=True, timeout=120)
        return dst.exists()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        # A missing ffmpeg is fatal (nothing will work); a timeout just fails this one.
        if isinstance(e, FileNotFoundError):
            sys.exit("ffmpeg not found. Install it:  winget install ffmpeg  (then reopen terminal)")
        return False

def main():
    # --- CLI arguments -------------------------------------------------------- #
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default=None, help="Only this cluster_id")
    ap.add_argument("--per-cluster", type=int, default=4, help="Max thumbs per cluster")
    ap.add_argument("--include-tagged", action="store_true", help="Also do already-tagged clusters")
    args = ap.parse_args()

    # --- Select clusters to thumbnail from the catalog ----------------------- #
    # Default is untagged photo/video clusters only; flags widen or narrow that.
    clusters = defaultdict(list)
    with open(CATALOG, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["media_type"] not in ("video", "photo"):
                continue
            if args.cluster and row["cluster_id"] != args.cluster:
                continue
            if row["story_arc"] and not args.include_tagged and not args.cluster:
                continue  # already tagged -> skip unless explicitly requested
            clusters[row["cluster_id"]].append(row)

    if not clusters:
        sys.exit("Nothing to do (no matching untagged clusters).")

    # --- Render a spread of thumbnails per cluster --------------------------- #
    done, missing, failed = 0, 0, 0
    for cid, members in sorted(clusters.items()):
        # spread picks across the cluster: first, last, and evenly between
        n = min(args.per_cluster, len(members))
        idxs = sorted({round(i * (len(members) - 1) / max(n - 1, 1)) for i in range(n)})
        for i in idxs:
            src = Path(members[i]["full_path"])
            if not src.exists():
                missing += 1  # source drive/card not currently mounted
                continue
            dst = THUMBS / cid / (src.stem + ".jpg")
            if make_thumb(src, dst):
                done += 1
            else:
                failed += 1
        print(f"[{cid}] ok")

    # --- Summary -------------------------------------------------------------- #
    print(f"\nThumbs created: {done} | source files not reachable: {missing} | failed: {failed}")
    if missing:
        print("Not-reachable usually means that card isn't currently plugged in — rerun with it inserted.")
    print(f"Output: {THUMBS}")
    print("Tell Claude when done — Claude will view and classify them.")

if __name__ == "__main__":
    main()
