"""
backup_media.py — Hash-verified backup of single-physical-copy media to a target drive.
Reads master_catalog.csv, finds files that exist on only one physical device, copies
them to the target with structure <target>\\MediaVault\\<source_label>\\<original subpath>,
verifies each copy by re-hashing, and writes a backup log the catalog can ingest.

Usage (target = your backup drive letter):
    python backup_media.py F:\\            # backs up everything reachable & unbacked
    python backup_media.py F:\\ --source sdr   # one source at a time (e.g. per card)
    python backup_media.py F:\\ --dry-run      # show what would copy, no writes

Safe: never deletes or modifies sources. Re-runnable: skips files already backed up
and verified. Sources must be plugged in to be copied (unreachable ones are reported).
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

# --- Project layout -------------------------------------------------------- #
INV = Path(__file__).resolve().parent.parent / "01_inventory"
CATALOG = INV / "master_catalog.csv"
LOG = INV / "backup_log.csv"

# --- Source-label -> physical-device map ----------------------------------- #
# Collapses many source labels onto the physical device they live on, so
# "exists on only one device" is judged per device, not per label. Keys must
# match the source_label values recorded in the catalog.
DEVICE = {'sd01':'card01','sd02':'card02','sd03':'card03','sd04':'card04','sd928':'card928',
          'sdr':'cardR','forge':'SSD','ssd_porsche':'SSD','os_c':'PC_C','desktop2':'PC_C_onedrive',
          'pro_portfolio':'PC_C'}

def quick_hash(path: Path, size: int) -> str:
    """Fast fingerprint: size + first/last 64KB. Used to verify a copy matches its source."""
    h = hashlib.md5()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(65536))
        if size > 131072:
            f.seek(-65536, os.SEEK_END)
            h.update(f.read(65536))
    return h.hexdigest()

def load_done() -> set:
    """Set of quick_hashes already backed up AND verified, so re-runs skip them."""
    if not LOG.exists():
        return set()
    with open(LOG, newline="", encoding="utf-8") as f:
        return {r["quick_hash"] for r in csv.DictReader(f) if r.get("verified") == "yes"}

def main():
    # --- CLI arguments -------------------------------------------------------- #
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="Backup drive/folder, e.g. F:\\ ")
    ap.add_argument("--source", default=None,
                    help="Limit to these source_label(s), comma-separated: --source sd01,sdr")
    ap.add_argument("--remap", default=None,
                    help="Drive letter remap if card mounts differently than when scanned, e.g. D:=E:")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Optional drive-letter remap (source paths were recorded under an old letter).
    remap_from = remap_to = None
    if args.remap:
        remap_from, remap_to = [s.strip().rstrip("\\") for s in args.remap.split("=")]

    target = Path(args.target) / "MediaVault"
    rows = list(csv.DictReader(open(CATALOG, newline="", encoding="utf-8")))
    done = load_done()

    # --- Build the backup queue: only single-copy, not-yet-backed-up files ---- #
    queue = []
    for r in rows:
        # Which physical devices hold this file? (from the recorded locations)
        devs = {DEVICE.get(loc.split(":", 1)[0], "other") for loc in r["all_locations"].split(" | ")}
        if len(devs) > 1 or "PC_C_onedrive" in devs:
            continue  # already physically redundant or cloud-synced
        if r["quick_hash"] in done:
            continue  # already backed up and verified
        if args.source and r["source_label"] not in {s.strip() for s in args.source.split(",")}:
            continue  # not in the requested source subset
        queue.append(r)

    total_gb = sum(int(r["size_bytes"]) for r in queue) / 1024**3
    print(f"Queued: {len(queue)} files, {total_gb:.1f} GB -> {target}")

    # --- Dry-run: preview a sample and stop (no writes) ---------------------- #
    if args.dry_run:
        for r in queue[:20]:
            print("  ", r["source_label"], r["filename"], f"{float(r['size_mb']):.0f} MB")
        return

    # --- Pre-flight: target must exist and have room ------------------------- #
    if not Path(args.target).exists():
        sys.exit(f"Target not found: {args.target}")
    free = shutil.disk_usage(args.target).free / 1024**3
    if free < total_gb * 1.02:  # 2% headroom
        print(f"WARNING: target has {free:.0f} GB free, need ~{total_gb:.0f} GB. Use --source to batch.")

    # --- Copy + verify each file, appending to the backup log ---------------- #
    new_log = not LOG.exists()
    copied = verified = failed = missing = 0
    t0 = time.time()
    with open(LOG, "a", newline="", encoding="utf-8") as logf:
        log = csv.writer(logf)
        if new_log:
            log.writerow(["quick_hash", "source_label", "src_path", "backup_path",
                          "size_bytes", "verified", "timestamp"])
        for i, r in enumerate(queue, 1):
            src_str = r["full_path"]
            if remap_from and src_str.upper().startswith(remap_from.upper()):
                src_str = remap_to + src_str[len(remap_from):]
            src = Path(src_str)
            if not src.exists():
                missing += 1  # source device not currently plugged in
                continue
            # mirror original folder structure under source label
            sub = Path(*src.parts[1:]) if len(src.parts) > 1 else Path(src.name)
            dst = target / r["source_label"] / sub
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)  # copy with metadata; never touches the source
                copied += 1
                # Verify by re-hashing the copy against the catalog's recorded hash.
                ok = quick_hash(dst, dst.stat().st_size) == r["quick_hash"]
                verified += ok
                failed += (not ok)
                log.writerow([r["quick_hash"], r["source_label"], str(src), str(dst),
                              r["size_bytes"], "yes" if ok else "HASH_MISMATCH",
                              time.strftime("%Y-%m-%dT%H:%M:%S")])
                if i % 25 == 0:
                    # Periodic throughput readout (GB done and rough GB/hr).
                    done_gb = sum(int(q["size_bytes"]) for q in queue[:i]) / 1024**3
                    rate = done_gb / max(time.time() - t0, 1) * 3600
                    print(f"  {i}/{len(queue)}  {done_gb:.0f} GB done  (~{rate:.0f} GB/hr)")
            except OSError as e:
                failed += 1
                log.writerow([r["quick_hash"], r["source_label"], str(src), str(dst),
                              r["size_bytes"], f"ERROR:{e}", time.strftime("%Y-%m-%dT%H:%M:%S")])
    print(f"\nCopied: {copied} | Verified: {verified} | Failed: {failed} | Source not plugged in: {missing}")
    print(f"Log: {LOG.name}. Re-run with other sources plugged in to continue.")

if __name__ == "__main__":
    main()
