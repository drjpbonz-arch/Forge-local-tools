"""
sanitize_manifest.py — Copy only media rows (video/photo/audio) from a local,
non-synced manifest into the project's manifests folder. Document/project rows
(patents, filings, etc.) stay in the local manifest only and never sync to cloud.

Usage:
    python sanitize_manifest.py "C:\\Users\\username\\inventory_local\\manifests\\example_manifest.csv"
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

# Row types allowed to sync into the shared/cloud project; everything else stays local.
KEEP = {"video", "photo", "audio"}
OUT_DIR = Path(__file__).resolve().parent.parent / "01_inventory" / "manifests"

def main():
    # --- Validate the single source-manifest argument ------------------------ #
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1]).resolve()  # resolve: a relative path or symlink must not dodge the checks below
    if not src.exists():
        sys.exit(f"Not found: {src}")
    # Refuse to run on a manifest that already lives in the synced folder (nothing to strip).
    if OUT_DIR in src.parents:
        sys.exit("Source manifest is already inside the synced project folder — nothing to sanitize.")

    # --- Copy through only the media rows; count what's kept vs. held back --- #
    out = OUT_DIR / src.name
    if out.exists() and os.path.samefile(out, src):
        sys.exit("Output would overwrite the source manifest — refusing.")
    kept = skipped = 0
    # Write to a temporary file beside the output and move it into place only when complete,
    # so a crash or a bad source never leaves a truncated manifest behind.
    with open(src, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames or "media_type" not in reader.fieldnames:
            sys.exit("Source has no 'media_type' column — nothing written.")
        fd, tmp = tempfile.mkstemp(dir=OUT_DIR, prefix=".sanitize_", suffix=".csv")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    if row["media_type"] in KEEP:
                        writer.writerow(row)
                        kept += 1
                    else:
                        skipped += 1  # document/project rows: left in the local manifest only
            os.replace(tmp, out)
        except BaseException:
            os.unlink(tmp)
            raise
    print(f"Media rows copied to project: {kept}")
    print(f"Document/other rows kept local-only: {skipped}")
    print(f"Sanitized manifest: {out}")
    print("Next: merge the sanitized manifest into your catalog.")

if __name__ == "__main__":
    main()
