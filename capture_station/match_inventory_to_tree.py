#!/usr/bin/env python3
"""
match_inventory_to_tree.py  -  record-inventory family (2026-09-07)

Given ONE hash inventory CSV (RECORD_INVENTORY_*.csv from _record_inventory.ps1 / tool 28,
columns incl. path,bytes,md5) and one or more LIVE folder trees, report which files in the trees
are byte-identical to files in the inventory. Size-first (cheap), then MD5 only on size matches.
Per-file timeout + IO-error capture so cloud-only placeholders (Sync.com) are COUNTED, not fatal.
REPORT ONLY. Nothing moved, nothing deleted.

Toolbox search (2026-09-07, by function): dedupe_media.py = duplicates within one tree;
crosscheck_hash_inventories.py (25c) = joins N inventories; nothing matched an inventory against a
live tree without first inventorying the tree (hours on 14 GB). This is the missing rung.

Usage:
  python match_inventory_to_tree.py INVENTORY.csv TREE1 [TREE2 ...] [--out REPORT.csv] [--filter SUBSTR]
    --filter  only inventory rows whose path contains SUBSTR (e.g. "\\Camera")
"""
import os, sys, csv, hashlib, signal, collections, re, datetime

class _T(Exception): pass
# Hash a file, but with a hard per-file wall-clock timeout (SIGALRM): a hung cloud-only
# placeholder returns 'TIMEOUT' rather than blocking the run; an unreadable file -> 'IOERR'.
def md5(p, limit=30):
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_T())); signal.alarm(limit)
    try:
        m = hashlib.md5()
        with open(p, 'rb') as g:
            for c in iter(lambda: g.read(1 << 20), b''): m.update(c)
        return m.hexdigest()
    except _T: return 'TIMEOUT'
    except OSError: return 'IOERR'
    finally: signal.alarm(0)

def main():
    # CLI: INVENTORY.csv then one or more TREE roots; optional --out REPORT.csv and --filter SUBSTR.
    a = sys.argv[1:]
    out = a[a.index('--out') + 1] if '--out' in a else None
    flt = a[a.index('--filter') + 1] if '--filter' in a else None
    args = [x for i, x in enumerate(a) if not x.startswith('--') and (i == 0 or a[i-1] not in ('--out', '--filter'))]
    inv_path, trees = args[0], args[1:]
    inv = [r for r in csv.DictReader(open(inv_path, encoding='utf-8-sig')) if (not flt or flt in r['path'])]
    # Index inventory rows by byte size (cheap prefilter) and by md5 (confirming compare).
    by_size = collections.defaultdict(list)
    for r in inv:
        try: by_size[int(float(r['bytes']))].append(r)
        except ValueError: pass
    camhash = {r['md5'].lower(): r for r in inv if r.get('md5')}
    # Size-first pass: walk each live tree; a file is a candidate only if its size
    # matches some inventory row (avoids hashing everything).
    cand = []
    for root in trees:
        for dp, dn, fn in os.walk(root):
            for f in fn:
                p = os.path.join(dp, f)
                try: s = os.path.getsize(p)
                except OSError: continue
                if s in by_size: cand.append((p, s))
    stamp = datetime.date.today().isoformat()
    rep = out or os.path.join(os.path.dirname(os.path.abspath(inv_path)), f'INVENTORY_VS_TREE_MATCH_{stamp}.csv')
    # Confirm candidates by md5; tally timeouts and IO errors instead of failing on them.
    hit = to = io = 0; hitb = 0; bytree = collections.Counter(); rows = []
    for p, s in cand:
        h = md5(p)
        if h == 'TIMEOUT': to += 1; continue
        if h == 'IOERR': io += 1; continue
        if h in camhash:
            hit += 1; hitb += s; bytree[os.path.dirname(p)[:80]] += 1
            rows.append([h, s, camhash[h]['path'], p])
    # Write the identical-file report (report only; nothing is moved or deleted).
    with open(rep, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['md5', 'bytes', 'inventory_path', 'tree_path']); w.writerows(rows)
    print(f'inventory rows {len(inv)} | size-matched candidates {len(cand)} | IDENTICAL {hit} files {hitb/1e9:.2f} GB | timeouts {to} | cloud-only/IO errors {io}')
    for k, v in bytree.most_common(10): print(f'  {v:5d}  {k}')
    print('report:', rep, '\nNothing moved.')

if __name__ == '__main__': main()
