#!/usr/bin/env python3
"""
blinding_harness.py — imaging blinding harness
================================================================
Enforces an objectivity protocol so a rater performs paired before/after image
measurement WITHOUT seeing the capture date, phase, filename, or chronological
order — removing the unconscious bias of knowing which image "should" look
better or worse.

A "harness" here = a wrapper that holds the real task (measuring the images)
inside a fixed set of conditions and won't let you break them. This one:
  1. STRIPS every date/phase/filename cue,
  2. RANDOMIZES the order (so sequence can't leak the timeline),
  3. relabels each item with a neutral code (Exam-A, Exam-B, …),
  4. SEALS the code->truth map behind a tamper-evident hash you can't quietly
     edit after the fact,
  5. refuses to UNBLIND until your reads are recorded — then re-attaches the
     truth and proves the seal was never altered.

Modes
-----
  seal    --manifest items.csv --outdir blinded/ [--images-dir imgs/] [--seed N]
  unblind --outdir blinded/ --reads reads.csv
  selftest                     # generates 13 mock items and round-trips them

manifest.csv columns (any extra columns are treated as hidden truth):
  item_id,filename,date,phase           # e.g. s1,IMG_2207.png,2022-05-13,before
reads.csv columns (produced by the rater, keyed by the BLIND code):
  blind_code,<your measurements...>     # e.g. Exam-D,measure_1,measure_2, ...
"""
import argparse, csv, hashlib, json, os, secrets, shutil, sys
from datetime import datetime, timezone

BLIND_ALPHABET = [f"Exam-{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# SEAL: shuffle items, assign neutral blind codes, write a truth-free worklist for
# the rater, and lock the code->truth map inside a hash-sealed key + receipt.
def seal(manifest, outdir, images_dir=None, seed=None):
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        sys.exit("manifest is empty")
    if n > len(BLIND_ALPHABET):
        sys.exit(f"only {len(BLIND_ALPHABET)} blind codes available; {n} exams given")

    # cryptographically-strong shuffle; permutation recorded ONLY in the sealed key
    rng = secrets.SystemRandom() if seed is None else __import__("random").Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    codes = BLIND_ALPHABET[:n]
    rng.shuffle(codes)  # codes not in exam order either

    os.makedirs(outdir, exist_ok=True)
    imgdir = os.path.join(outdir, "blinded_images")
    if images_dir:
        os.makedirs(imgdir, exist_ok=True)

    key = {"created_utc": _now(), "n_exams": n, "map": []}
    worklist = []
    for present_pos, src_idx in enumerate(order):
        row = rows[src_idx]
        code = codes[present_pos]
        blind_img = ""
        if images_dir and row.get("filename"):
            ext = os.path.splitext(row["filename"])[1] or ".img"
            blind_img = f"{code}{ext}"
            src = os.path.join(images_dir, row["filename"])
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(imgdir, blind_img))
        # worklist row carries the BLIND code + (optional) blinded image only — no truth
        worklist.append({"present_order": present_pos + 1, "blind_code": code,
                         "blinded_image": blind_img})
        # sealed key carries the full truth
        key["map"].append({"blind_code": code, "present_order": present_pos + 1,
                           **{k: v for k, v in row.items()}})

    # write the sealed key first, then hash it, then write a receipt + worklist that
    # embed that hash. Editing the key later changes its hash -> unblind will refuse.
    key_path = os.path.join(outdir, "_SEAL_KEY.local.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, sort_keys=True)
    key_hash = _sha256(key_path)

    with open(os.path.join(outdir, "blinded_worklist.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["present_order", "blind_code", "blinded_image"])
        w.writeheader(); w.writerows(worklist)

    receipt = {
        "sealed_utc": key["created_utc"], "n_exams": n,
        "seal_key_sha256": key_hash,
        "instructions": ("Digitize every Exam in blinded_worklist.csv order. Record reads "
                         "keyed by blind_code into reads.csv. Do NOT open _SEAL_KEY.local.json. "
                         "Then run: blinding_harness.py unblind --outdir . --reads reads.csv"),
        "protocol": "Objectivity Protocol — date/phase/order blinded; key sealed."
    }
    with open(os.path.join(outdir, "SEAL_RECEIPT.json"), "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(json.dumps({"sealed": True, "n_exams": n, "outdir": outdir,
                      "seal_key_sha256": key_hash[:16] + "…",
                      "worklist": "blinded_worklist.csv",
                      "key_LOCAL_ONLY": "_SEAL_KEY.local.json (do not open/share until reads recorded)"},
                     indent=2))

# UNBLIND: refuse unless reads exist and the key's live hash still matches the
# receipt (proving the seal was never edited), then re-join reads to the truth.
def unblind(outdir, reads):
    key_path = os.path.join(outdir, "_SEAL_KEY.local.json")
    receipt_path = os.path.join(outdir, "SEAL_RECEIPT.json")
    for p in (key_path, receipt_path, reads):
        if not os.path.exists(p):
            sys.exit(f"missing required file: {p}")
    with open(reads, newline="", encoding="utf-8") as f:
        read_rows = list(csv.DictReader(f))
    if not read_rows:
        sys.exit("reads.csv is empty — record your reads before unblinding (protocol: read first)")

    receipt = json.load(open(receipt_path, encoding="utf-8"))
    live_hash = _sha256(key_path)
    if live_hash != receipt["seal_key_sha256"]:
        sys.exit("SEAL BROKEN: key hash != receipt hash. The sealed key was altered after sealing. "
                 "Reads are NOT trustworthy under the protocol.")
    key = json.load(open(key_path, encoding="utf-8"))
    truth = {m["blind_code"]: m for m in key["map"]}
    reads_by_code = {r.get("blind_code"): r for r in read_rows}

    missing = [c for c in truth if c not in reads_by_code]
    out_rows = []
    for code, t in truth.items():
        r = reads_by_code.get(code, {})
        merged = {**t, **{k: v for k, v in r.items() if k != "blind_code"}}
        merged["read_recorded"] = code in reads_by_code
        out_rows.append(merged)
    out_rows.sort(key=lambda x: (x.get("date", ""), x.get("phase", "")))
    out_path = os.path.join(outdir, "unblinded_results.csv")
    cols = sorted({k for row in out_rows for k in row.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
    print(json.dumps({"unblinded": True, "seal_verified": True,
                      "exams": len(truth), "reads_matched": len(truth) - len(missing),
                      "reads_missing": missing, "results": out_path}, indent=2))

# SELFTEST: generate mock items, seal, simulate reads against blind codes only,
# unblind, and assert the worklist leaked no date/phase/filename.
def selftest():
    import tempfile
    d = tempfile.mkdtemp(prefix="blind_selftest_")
    man = os.path.join(d, "exams.csv")
    with open(man, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["exam_id", "filename", "date", "phase"])
        dates = ["2022-05-13", "2022-11-07", "2023-06-26", "2024-01-09"]
        phases = ["before", "after", "baseline"]
        k = 0
        for i in range(13):
            w.writerow([f"c{i}", f"IMG_{1000+i}.png", dates[i % 4], phases[i % 3]]); k += 1
    out = os.path.join(d, "blinded")
    seal(man, out, seed=42)
    # simulate the reader recording measurements against the BLIND codes only
    wl = list(csv.DictReader(open(os.path.join(out, "blinded_worklist.csv"))))
    reads = os.path.join(d, "reads.csv")
    with open(reads, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["blind_code", "measure_deg"])
        for row in wl:
            w.writerow([row["blind_code"], round(10 + 2.5 * len(row["blind_code"]), 1)])
    unblind(out, reads)
    # confirm the worklist leaked no truth
    header = open(os.path.join(out, "blinded_worklist.csv")).readline().strip()
    assert "date" not in header and "phase" not in header and "filename" not in header, "LEAK!"
    print(f"\nSELFTEST OK — no date/phase/filename in worklist; seal verified; round-trip clean.\n(artifacts in {d})")

def main():
    ap = argparse.ArgumentParser(description="imaging blinding harness (Objectivity Protocol)")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("seal"); s.add_argument("--manifest", required=True); s.add_argument("--outdir", required=True)
    s.add_argument("--images-dir"); s.add_argument("--seed", type=int)
    u = sub.add_parser("unblind"); u.add_argument("--outdir", required=True); u.add_argument("--reads", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.mode == "seal": seal(a.manifest, a.outdir, a.images_dir, a.seed)
    elif a.mode == "unblind": unblind(a.outdir, a.reads)
    elif a.mode == "selftest": selftest()

if __name__ == "__main__":
    main()
