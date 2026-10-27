#!/usr/bin/env python3
"""
prep_listing_photos.py - stage listing photos for an online marketplace from a session manifest.
Reads manifest.csv, makes per-item folders of downscaled JPEGs (max 1600px, the platform's
stated ceiling; min 620px), named <ITEM>_<n>.jpg. Source photos are never modified.
Usage: python prep_listing_photos.py <session_folder> [--out <dir>]
"""
import csv, os, sys, argparse
from PIL import Image

# Phone photos are often HEIC; register the optional decoder when it is installed.
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError: pass

# CLI: a session folder holding manifest.csv, plus an optional output directory.
AP=argparse.ArgumentParser(); AP.add_argument("session"); AP.add_argument("--out", default=None)
A=AP.parse_args()
out = A.out or os.path.join(A.session, "_listing_photos")
os.makedirs(out, exist_ok=True)

# --- One downscaled folder per manifest item; skip calibration frames and photo-less rows ---
n_items=n_ph=0
for row in csv.DictReader(open(os.path.join(A.session,"manifest.csv"), encoding="utf-8")):
    if row["category"] in ("CLOCK_CAL",) or not row["photos"]: continue
    slug = f"item{int(row['item']):02d}_{row['category']}"
    d = os.path.join(out, slug); os.makedirs(d, exist_ok=True); n_items+=1
    # Downscale each source photo (long edge <=1600px); warn but continue on undersized images.
    for i, ph in enumerate(row["photos"].split(";"), 1):
        src = os.path.join(A.session, ph)
        if not os.path.exists(src): print("MISSING:", ph); continue
        im = Image.open(src).convert("RGB"); im.thumbnail((1600,1600))
        if max(im.size) < 620: print("WARN small:", ph)
        im.save(os.path.join(d, f"{slug}_{i}.jpg"), "JPEG", quality=92); n_ph+=1
print(f"staged {n_ph} photos across {n_items} items -> {out}")
print("Next: host these (R2/Dropbox raw=1) -> URLs -> CSV product_image_N columns.")
