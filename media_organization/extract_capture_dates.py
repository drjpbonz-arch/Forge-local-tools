#!/usr/bin/env python3
"""
extract_capture_dates.py — recover REAL capture dates for a media manifest.

WHY: iCloud (and some exports) reset filesystem timestamps to the download/export
date, so build_catalog.py clusters everything into one bogus "today" shooting day.
The true capture date survives in metadata:
  - Photos: EXIF DateTimeOriginal (Exif sub-IFD 0x8769, tag 36867), else EXIF
    DateTime (tag 306). HEIC supported via pillow-heif.
  - Videos: QuickTime com.apple.quicktime.creationdate (authentic, preserved by
    iOS), else container/stream creation_time (LOWER confidence — for iCloud
    exports this is often the export date, not the capture date).

WHAT IT DOES: reads a manifest CSV, adds two columns (capture_date, date_source),
and REWRITES the 'created' field with the best real datetime so the existing
build_catalog.py clusters on it unchanged. When no metadata date exists it falls
back to the file's 'modified' date (never the hydration / "today" date) and tags
date_source=modified_fallback so you can spot soft clusters.

Read-only on the media files. Backs the manifest up once before rewriting.
Resumable: skips rows already tagged; --budget N writes progress and exits after
N seconds so it can be run in repeated short windows. Requires: Pillow,
pillow-heif (HEIC), and ffprobe on PATH (videos).

Usage:
    python extract_capture_dates.py --manifest 01_inventory/manifests/icloud_photos_manifest.csv
    # remap path prefix if the drive is mounted elsewhere than when scanned:
    python extract_capture_dates.py --manifest <csv> --root "C:\\old=D:\\new"
    # run in resumable 30s windows (re-run until it reports remaining 0):
    python extract_capture_dates.py --manifest <csv> --budget 30
"""
import argparse, csv, json, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

# Which extensions are treated as photos vs videos (decides the metadata reader used).
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp",
              ".bmp", ".gif", ".dng", ".cr2", ".cr3", ".nef", ".arw", ".avif"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".mts", ".m2ts", ".3gp",
              ".wmv", ".webm", ".hevc"}

# --- HEIC support is registered lazily, once, on first photo read ---------- #
_HEIF_READY = False
def _ensure_heif():
    """Register the HEIF opener with Pillow exactly once (no-op if unavailable)."""
    global _HEIF_READY
    if _HEIF_READY:
        return
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    _HEIF_READY = True

def parse_exif_dt(val):
    """Parse an EXIF datetime string in either ':'- or '-'-separated form -> datetime/None."""
    if not val:
        return None
    s = str(val).strip().strip(chr(0))  # trim stray NUL padding some cameras emit
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

def parse_iso(val):
    """Parse an ISO-ish timestamp (with assorted timezone quirks) -> datetime/None."""
    if not val:
        return None
    # Normalize 'Z' and colon-less offsets so fromisoformat can handle them.
    s = str(val).strip().replace("Z", "+0000")
    if len(s) >= 5 and (s[-5] in "+-") and s[-3] != ":":
        s = s[:-2] + ":" + s[-2:]
    # Try progressively looser forms: full string, sans-fraction, then date+time only.
    for attempt in (s, s.split(".")[0], s[:19]):
        try:
            return datetime.fromisoformat(attempt)
        except ValueError:
            continue
    return None

def photo_date(path):
    """Best capture datetime from a photo's EXIF -> (datetime|None, source_tag)."""
    _ensure_heif()
    try:
        from PIL import Image
    except Exception:
        return None, "no_pillow"
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            if not ex:
                return None, "none"
            # DateTimeOriginal lives in the Exif sub-IFD; fall back to top-level DateTime.
            try:
                sub = ex.get_ifd(0x8769)
            except Exception:
                sub = {}
            dt = parse_exif_dt(sub.get(36867))   # DateTimeOriginal
            if dt:
                return dt, "exif_dto"
            dt = parse_exif_dt(ex.get(306))       # DateTime
            if dt:
                return dt, "exif_dt"
            return None, "none"
    except Exception:
        return None, "unreadable"

def video_date(path):
    """Best capture datetime from a video via ffprobe -> (datetime|None, source_tag)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60)
        d = json.loads(out.stdout or "{}")
    except Exception:
        return None, "unreadable"
    # Prefer Apple's authentic capture tag; then container creation_time; then any stream's.
    ftags = (d.get("format") or {}).get("tags", {}) or {}
    dt = parse_iso(ftags.get("com.apple.quicktime.creationdate"))
    if dt:
        return dt, "qt_creationdate"
    dt = parse_iso(ftags.get("creation_time"))
    if dt:
        return dt, "video_creation_time"
    for s in d.get("streams", []):
        dt = parse_iso((s.get("tags") or {}).get("creation_time"))
        if dt:
            return dt, "video_creation_time"
    return None, "none"

def main():
    # --- CLI arguments -------------------------------------------------------- #
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None, help="Remap full_path prefix. Form OLD=NEW.")
    ap.add_argument("--budget", type=float, default=0,
                    help="Seconds before writing progress and exiting (0 = run to completion).")
    args = ap.parse_args()

    # --- Load manifest and set up optional path remapping -------------------- #
    man = Path(args.manifest)
    rows = list(csv.DictReader(open(man, encoding="utf-8")))
    old_pre = new_pre = None
    if args.root and "=" in args.root:
        old_pre, new_pre = args.root.split("=", 1)
    def real(p):
        # Rewrite the scanned path prefix to wherever the drive is mounted now.
        return p.replace(old_pre, new_pre, 1) if old_pre else p

    # --- Main pass: fill capture_date/date_source per row -------------------- #
    t0 = time.time()
    processed = 0
    for i, r in enumerate(rows, 1):
        if (r.get("date_source") or "").strip():
            continue  # resume: this row was handled in a previous run
        if args.budget and (time.time() - t0) > args.budget:
            print(f"  budget hit at row {i}")
            break  # time window exhausted; progress is saved below
        processed += 1
        ext = (r.get("extension") or "").lower()
        path = real(r.get("full_path", ""))
        dt, src = (None, "skip")
        if ext in IMAGE_EXTS:
            dt, src = photo_date(path)
        elif ext in VIDEO_EXTS:
            dt, src = video_date(path)
        if dt:
            # Real metadata date found: overwrite 'created' so downstream clustering uses it.
            r["created"] = dt.isoformat(timespec="seconds")
            r["capture_date"] = dt.date().isoformat()
            r["date_source"] = src
        else:
            # No metadata: fall back to 'modified' (never the "today" hydration date).
            mod = (r.get("modified") or "")[:10]
            r["capture_date"] = mod
            r["date_source"] = "modified_fallback"
            if mod:
                r["created"] = r.get("modified")
        if processed % 500 == 0:
            print(f"  ...processed {processed} this run ({time.time()-t0:.0f}s)")

    # --- Back up the manifest once, then rewrite it in place (atomic) -------- #
    existing_bak = list(man.parent.glob(man.stem + "_pre_capturedate_*.csv"))
    if existing_bak:
        bak = existing_bak[0]  # keep the first (oldest) backup; don't stack new ones
    else:
        bak = man.with_name(man.stem + f"_pre_capturedate_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy2(man, bak)
    # Ensure the two new columns exist in the header before writing.
    fields = list(rows[0].keys())
    for c in ("capture_date", "date_source"):
        if c not in fields:
            fields.append(c)
    tmp = man.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, man)

    # --- Summary: tally date sources and the real-metadata date span --------- #
    stats, real_dates = {}, []
    for r in rows:
        ds = (r.get("date_source") or "unprocessed").strip() or "unprocessed"
        stats[ds] = stats.get(ds, 0) + 1
        # Only count high-confidence metadata dates toward the reported span.
        if ds not in ("modified_fallback", "unprocessed", "none", "unreadable", "skip", "no_pillow"):
            cd = (r.get("capture_date") or "").strip()
            if cd:
                real_dates.append(cd)
    done = sum(v for k, v in stats.items() if k != "unprocessed")
    print(f"\nRun {time.time()-t0:.0f}s. Processed {processed}. "
          f"Total done {done}/{len(rows)} (remaining {stats.get('unprocessed', 0)}). Backup: {bak.name}")
    print("date_source (cumulative):")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:22} {v}")
    if real_dates:
        real_dates.sort()
        print(f"real-metadata span: {real_dates[0]} -> {real_dates[-1]} ({len(real_dates)} files)")
    # Exit code 2 signals "work remaining" so a wrapper can loop until done.
    sys.exit(2 if stats.get("unprocessed", 0) else 0)

if __name__ == "__main__":
    main()
