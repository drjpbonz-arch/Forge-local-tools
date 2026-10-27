"""
sanitize_manifest.py — Copy only media rows (video/photo/audio) from a local,
non-synced manifest into the project's manifests folder. Document/project rows
(patents, filings, etc.) stay in the local manifest only and never sync to cloud.

Usage:
    python sanitize_manifest.py "C:\\Users\\username\\inventory_local\\manifests\\example_manifest.csv"
"""

import csv
import sys
from pathlib import Path

# Row types allowed to sync into the shared/cloud project; everything else stays local.
KEEP = {"video", "photo", "audio"}
OUT_DIR = Path(__file__).resolve().parent.parent / "01_inventory" / "manifests"

def main():
    # --- Validate the single source-manifest argument ------------------------ #
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    if not src.exists():
        sys.exit(f"Not found: {src}")
    # Refuse to run on a manifest that already lives in the synced folder (nothing to strip).
    if OUT_DIR in src.parents:
        sys.exit("Source manifest is already inside the synced project folder — nothing to sanitize.")

    # --- Copy through only the media rows; count what's kept vs. held back --- #
    out = OUT_DIR / src.name
    kept = skipped = 0
    with open(src, newline="", encoding="utf-8") as f_in, \
         open(out, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["media_type"] in KEEP:
                writer.writerow(row)
                kept += 1
            else:
                skipped += 1  # document/project rows: left in the local manifest only
    print(f"Media rows copied to project: {kept}")
    print(f"Document/other rows kept local-only: {skipped}")
    print(f"Sanitized manifest: {out}")
    print("Run build_catalog.py to merge.")

if __name__ == "__main__":
    main()
