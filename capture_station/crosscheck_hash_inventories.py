#!/usr/bin/env python3
"""
crosscheck_hash_inventories.py  -  cross-location duplicate finder for hash inventories (2026-09-06)

Joins two or more RECORD_INVENTORY_*.csv files (produced by _record_inventory.ps1,
columns: path,name,ext,bytes,md5,created,modified,custody_suggested,dedupe_eligible)
and reports CROSS-LOCATION duplicates: the same content hash present in more than
one inventory. REPORT ONLY. Nothing is moved, nothing is deleted.

Toolbox search (2026-09-06, by function): dedupe_media.py finds duplicates WITHIN
one tree; dedupe_engine.py fingerprints backup-vs-target with hardcoded roots;
audio_dedup.py is perceptual. None joins N independent hash inventories, so this
extends the record-inventory family with a joiner.

Usage:
    python crosscheck_hash_inventories.py A.csv B.csv [C.csv ...] [--out DIR]
"""
import csv, sys, os, collections, datetime

# Read one RECORD_INVENTORY csv into (md5, bytes, path) tuples,
# skipping rows whose hash is blank, an error marker, or "excluded".
def load(p):
    rows = []
    with open(p, encoding="utf-8-sig", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            h = (r.get("md5") or "").strip().lower()
            if not h or h.startswith("error") or h == "excluded":
                continue
            try: b = int(float(r.get("bytes") or 0))
            except ValueError: b = 0
            rows.append((h, b, r.get("path", "")))
    return rows

def main():
    # CLI: positional args are the inventory csvs; --out DIR optionally redirects the report.
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = None
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]; args = [a for a in args if a != out]
    if len(args) < 2:
        sys.exit("need at least two RECORD_INVENTORY csv files")
    # Load every inventory, labelling each by a short name derived from its filename.
    inv = {}
    for p in args:
        label = os.path.basename(p).replace("RECORD_INVENTORY_", "").rsplit("_", 1)[0]
        inv[label] = load(p)
        print(f"{label}: {len(inv[label])} hashed files")
    # Index every file by content hash across all inventories, then keep only the
    # hashes that appear in MORE THAN ONE inventory -- the cross-location duplicates.
    by_hash = collections.defaultdict(list)
    for label, rows in inv.items():
        seen = set()
        for h, b, path in rows:
            by_hash[h].append((label, b, path))
    cross = {h: v for h, v in by_hash.items() if len({x[0] for x in v}) > 1}
    stamp = datetime.date.today().isoformat()
    outdir = out or os.path.dirname(os.path.abspath(args[0]))
    rep = os.path.join(outdir, f"CROSS_LOCATION_DUPES_{stamp}.csv")
    # Write the cross-location duplicate report (report only; nothing is moved).
    with open(rep, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["md5", "bytes", "locations", "copies", "paths"])
        for h, v in sorted(cross.items(), key=lambda kv: -kv[1][0][1]):
            locs = sorted({x[0] for x in v})
            w.writerow([h, v[0][1], "|".join(locs), len(v), " || ".join(x[2] for x in v)])
    # summary
    labels = list(inv.keys())
    print(f"\ncross-location duplicate hashes: {len(cross)}")
    dup_bytes = sum(v[0][1] * (len(v) - 1) for v in cross.values())
    print(f"bytes held redundantly across locations: {dup_bytes/1e9:.2f} GB (keeping one copy of each)")
    # Pairwise overlap: shared vs unique hashes for each pair of locations.
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            sa = {h for h, _, _ in inv[a]}; sb = {h for h, _, _ in inv[b]}
            both = sa & sb
            print(f"  {a} x {b}: {len(both)} shared hashes  |  unique to {a}: {len(sa-sb)}  unique to {b}: {len(sb-sa)}")
    print(f"\nreport: {rep}\nNothing moved. Decide per location AFTER reading the report.")

if __name__ == "__main__":
    main()
