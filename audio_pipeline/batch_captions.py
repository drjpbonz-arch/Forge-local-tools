#!/usr/bin/env python3
"""
batch_captions.py - manifest-first batch runner for captions.py

A large voice-memo / rambling-recording / interview corpus is not one file,
it's a tree: years of phone voice memos, m4a/mp3/wav/mp4/mov/aac/opus/amr,
mixed with music and lyric drafts, nested a dozen folders deep. This wraps
captions.py so that corpus gets transcribed WITHOUT scattering a new little
output folder next to every single file in the tree, and without babysitting
a loop by hand.

Two-step workflow (each is a double-clickable .cmd in this folder):

    1. --manifest   Recursively scan, ffprobe every candidate file, and write
                     batch_manifest.csv (the working queue) + BATCH_MANIFEST.md
                     (a human summary: totals, hours, est. GPU wall-time, the
                     20 longest files, and everything skipped and why). No GPU
                     work happens here - it is safe, fast, and repeatable.

    2. --run        Consume that queue and transcribe every file that still
                     needs it, one captions.py subprocess at a time, on the
                     venv's own Python (sys.executable) so the GPU wheels are
                     used. Resume-safe: already-transcribed files are skipped
                     by checking for their output files on disk, not by
                     trusting a stale manifest. One file's failure is logged
                     and the queue continues - Ctrl+C at any point is safe.

OUTPUT LAYOUT (the decision this file makes, and why)
    captions.py's own default drops "<name>/<name>.srt" etc. as a SIBLING of
    the source file. Fine for one clip; wrong for a whole tree - it would
    sprinkle hundreds of new folders throughout a corpus that may live under
    OneDrive, mixed in with music drafts and originals never meant to be
    touched by a transcription pass.

    Instead, everything mirrors into a single top-level folder under the
    scan root:

        <root>\\_transcripts\\<same relative folder>\\<stem>\\<stem>.srt
                                                               <stem>.txt
                                                               <stem>.transcript.json

    Same per-file-folder convention captions.py already uses (so nothing
    about the FILE layout is new) - just rooted in one predictable, greppable
    place instead of scattered through the source tree:
      - "must not scatter into cloud-synced surprise locations": there is
        exactly one new top-level folder to exclude from sync or relocate,
        not hundreds buried at random depths.
      - "must be greppable in one place": `grep -r foo _transcripts\` finds
        every transcript in the corpus in one shot.
      - "must be resume-safe": completion is checked by looking for the
        expected output files in this mirror, never by trusting a log.
      - forward-compatible with a future prosody-v2 sweep: that pass can
        write `<stem>.prosody.json` into this same per-file folder, right
        next to the transcript it describes, with zero new naming logic.

    batch_manifest.csv, BATCH_MANIFEST.md, and batch_run.log all live at the
    top of that same _transcripts\ folder - the entire batch's paper trail
    in one place.

TESTABILITY (no GPU, no model weights required)
    --dry-run substitutes an internal stub worker for captions.py: it writes
    small placeholder output files (clearly marked as stubs) instead of
    running faster-whisper, so the queueing / logging / resume / failure-
    isolation logic can be exercised on a machine with no GPU at all.
    --engine-cmd overrides the base command used to invoke the engine, for
    pointing at a different test double entirely.

Nothing is uploaded. Same offline guarantee as captions.py - this file only
ever shells out to ffprobe (to build the manifest) and to captions.py itself
(to transcribe), both already local.

Usage:
    python batch_captions.py --manifest
    python batch_captions.py --manifest --root D:\other\corpus
    python batch_captions.py --run
    python batch_captions.py --run --limit 10
See --help for every flag.
"""

import os
import sys
import csv
import json
import time
import shlex
import argparse
import tempfile
import datetime
import subprocess
from collections import Counter

# captions.py lives next to this file - reuse its probe/stem helpers directly
# rather than re-implementing them, so "already transcribed" and "no audio
# stream" mean exactly the same thing here as they do to captions.py itself.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    import captions
except ImportError:
    print("ERROR: captions.py not found next to batch_captions.py.")
    print(f"  Expected it at: {os.path.join(SCRIPT_DIR, 'captions.py')}")
    sys.exit(1)

CAPTIONS_PY = os.path.join(SCRIPT_DIR, "captions.py")

__version__ = "1.0.0"

# --- defaults ----------------------------------------------------------------
DEFAULT_ROOT = r"C:\media\audio_corpus"
OUTPUT_DIRNAME = "_transcripts"
MANIFEST_CSV_NAME = "batch_manifest.csv"
MANIFEST_MD_NAME = "BATCH_MANIFEST.md"
RUN_LOG_NAME = "batch_run.log"

# Voice memos / rambling recordings / lyric drafts come in all of these.
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".opus", ".amr", ".flac", ".wma",
              ".ogg", ".oga", ".aiff", ".aif", ".caf", ".3ga", ".wv", ".mp2"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp", ".3g2",
              ".wmv", ".ts", ".m2ts"}
DEFAULT_EXTS = AUDIO_EXTS | VIDEO_EXTS

DEFAULT_LANGUAGE = "en"
DEFAULT_PROMPT = ("Jane Doe, Example Organization, "
                   "Project Alpha, Project Beta")
# captions.py's own default --gap is 0.70s. Batch defaults to a slightly
# looser 1.0s: unscripted voice memos ramble and breathe more than a produced
# clip, and a looser gap avoids fragmenting every half-second breath into its
# own cue across hours of unstructured audio. Override with --gap.
DEFAULT_GAP = 1.0
# No burn-in in batch mode (transcripts only); vtt skipped by default to cut
# clutter across a whole corpus - add it back with --formats.
DEFAULT_FORMATS = "txt,srt,json"

FORMAT_EXT = {"srt": ".srt", "vtt": ".vtt", "json": ".transcript.json",
              "txt": ".txt", "ass": ".ass"}

# Rough planning numbers only - see captions.py README's own performance
# table (large-v3 on CUDA int8/int8_float16 runs ~8-15x realtime; 10x is the
# round midpoint used here). Measure your own first few files and re-project.
EST_REALTIME_MULTIPLE = 10.0
# captions.py README: "model load is a few seconds... happens once per
# invocation, so batching many files in one command is much faster than one
# command per file." This runner deliberately invokes captions.py ONCE PER
# FILE (for clean failure isolation and per-file resume), so that per-
# invocation load cost compounds across the corpus - folded into the estimate
# rather than pretending it away.
MODEL_LOAD_OVERHEAD_SEC = 5.0

CSV_FIELDS = ["path", "rel_path", "ext", "minutes", "size_MB", "duration_sec",
              "status", "note"]


# --- small shared helpers ----------------------------------------------------
def parse_formats(s):
    return [f.strip().lower() for f in s.split(",") if f.strip()]


def parse_exts(s):
    out = set()
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.add(tok if tok.startswith(".") else "." + tok)
    return out


def fmt_hms(total_seconds):
    """Seconds -> '1h 23m' / '4m 05s', for humans reading a summary."""
    total_seconds = max(0, int(round(total_seconds)))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def require_ffmpeg():
    if not (captions.have("ffmpeg") and captions.have("ffprobe")):
        print("ERROR: ffmpeg/ffprobe not found on PATH - same requirement as captions.py.")
        print("  Windows: grab a full build from https://www.gyan.dev/ffmpeg/builds/ "
              "and add its bin\\ folder to PATH.")
        sys.exit(1)


# --- stage 1: scan + probe ----------------------------------------------------
def scan_media_files(root, exts, out_root):
    """Recursively walk root; return (sorted absolute media paths, Counter of
    ignored non-media extensions). Prunes dotfolders and the output mirror
    itself so a re-run never treats its own previous outputs as source input."""
    media = []
    ignored_ext = Counter()
    out_root_norm = os.path.normcase(os.path.normpath(out_root))

    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in dirnames:
            full_norm = os.path.normcase(os.path.normpath(os.path.join(dirpath, d)))
            if d.startswith(".") or full_norm == out_root_norm:
                continue
            keep.append(d)
        dirnames[:] = keep

        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                media.append(os.path.join(dirpath, fn))
            else:
                ignored_ext[ext or "(no ext)"] += 1

    media.sort()
    return media, ignored_ext


def probe_media(path):
    """Return {duration, has_audio, size_bytes, error}. Never raises."""
    try:
        size_bytes = os.path.getsize(path)
    except OSError as e:
        return {"duration": None, "has_audio": False, "size_bytes": 0,
                "error": f"cannot stat file: {e.strerror or e}"}

    info = captions.ffprobe_json(path)
    if not info:
        return {"duration": None, "has_audio": False, "size_bytes": size_bytes,
                "error": "ffprobe could not read this file (corrupt/unsupported?)"}

    streams = info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    return {"duration": duration, "has_audio": has_audio, "size_bytes": size_bytes,
            "error": None}


def mirrored_out_dir(root, media_path, out_root):
    """<root>\\sub\\clip.m4a -> <out_root>\\sub\\<stem>  (captions.py's own
    per-file-folder convention, rooted under the mirror instead of the source)."""
    rel = os.path.relpath(media_path, root)
    rel_dir = os.path.dirname(rel)
    stem = captions.safe_stem(media_path)
    if rel_dir in ("", "."):
        return os.path.normpath(os.path.join(out_root, stem))
    return os.path.normpath(os.path.join(out_root, rel_dir, stem))


def expected_output_files(out_dir, stem, formats):
    return [os.path.join(out_dir, stem + FORMAT_EXT[f]) for f in formats if f in FORMAT_EXT]


def is_done(out_dir, stem, formats):
    """Done means EVERY requested format's file already exists - a partial
    write (e.g. killed mid-run) is correctly NOT done, and gets re-run."""
    files = expected_output_files(out_dir, stem, formats)
    return bool(files) and all(os.path.isfile(f) for f in files)


def build_row(root, out_root, media_path, formats):
    rel_path = os.path.relpath(media_path, root)
    ext = os.path.splitext(media_path)[1].lower()
    probe = probe_media(media_path)
    size_mb = round(probe["size_bytes"] / 1e6, 2)
    duration = probe["duration"]
    minutes = round(duration / 60.0, 2) if duration is not None else 0.0
    stem = captions.safe_stem(media_path)
    out_dir = mirrored_out_dir(root, media_path, out_root)

    if probe["error"]:
        status, note = "error", probe["error"]
    elif not probe["has_audio"]:
        status, note = "no-audio", "no audio stream (video-only or silent container)"
    elif is_done(out_dir, stem, formats):
        status, note = "done", "already transcribed"
    else:
        status, note = "queued", ""

    return {"path": media_path, "rel_path": rel_path, "ext": ext,
             "minutes": minutes, "size_MB": size_mb,
             "duration_sec": round(duration, 1) if duration is not None else "",
             "status": status, "note": note}


def build_inventory(root, out_root, exts, formats, quiet=False):
    media_files, ignored_ext = scan_media_files(root, exts, out_root)
    rows = []
    for i, m in enumerate(media_files, 1):
        rows.append(build_row(root, out_root, m, formats))
        if not quiet and i % 25 == 0:
            print(f"  scanned {i}/{len(media_files)} file(s)...", flush=True)
    return rows, ignored_ext


# --- stage 2: manifest read/write --------------------------------------------
def write_manifest_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def read_manifest_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["minutes"] = float(r["minutes"]) if r.get("minutes") not in ("", None) else 0.0
        r["size_MB"] = float(r["size_MB"]) if r.get("size_MB") not in ("", None) else 0.0
        r["duration_sec"] = (float(r["duration_sec"])
                              if r.get("duration_sec") not in ("", None) else None)
    return rows


def write_manifest_md(rows, ignored_ext, root, out_root, path, formats):
    queued = [r for r in rows if r["status"] == "queued"]
    done = [r for r in rows if r["status"] == "done"]
    no_audio = [r for r in rows if r["status"] == "no-audio"]
    errors = [r for r in rows if r["status"] == "error"]

    total_minutes = sum(r["minutes"] for r in rows)
    queued_minutes = sum(r["minutes"] for r in queued)
    queued_seconds = queued_minutes * 60.0
    est_wall_seconds = queued_seconds / EST_REALTIME_MULTIPLE + len(queued) * MODEL_LOAD_OVERHEAD_SEC

    # Only queued/done rows: a no-audio or unreadable file's duration doesn't
    # inform GPU wall-time, and would just muddy a table meant to.
    longest = sorted((r for r in rows if r["status"] in ("queued", "done")
                       and r["duration_sec"] not in (None, "")),
                      key=lambda r: r["duration_sec"], reverse=True)[:20]

    L = []
    L.append("# Batch Manifest")
    L.append("")
    L.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Scan root: `{root}`")
    L.append(f"Output mirror: `{out_root}`")
    L.append(f"Formats: `{','.join(formats)}` (this decides what \"already transcribed\" means)")
    L.append("")
    L.append("## Totals")
    L.append("")
    L.append(f"- Media files found: **{len(rows)}**")
    L.append(f"- Total duration, all candidates: **{fmt_hms(total_minutes * 60)}** ({total_minutes:.1f} min)")
    L.append(f"- Queued to transcribe: **{len(queued)}** files, {fmt_hms(queued_seconds)} ({queued_minutes:.1f} min)")
    L.append(f"- Already done: **{len(done)}**")
    L.append(f"- No audio track (skipped): **{len(no_audio)}**")
    L.append(f"- Errors / unreadable (skipped): **{len(errors)}**")
    if ignored_ext:
        other_total = sum(ignored_ext.values())
        breakdown = ", ".join(f"{ext} ({n})" for ext, n in
                               sorted(ignored_ext.items(), key=lambda kv: -kv[1]))
        L.append(f"- Other non-media files seen in the tree, ignored: **{other_total}** ({breakdown})")
    L.append("")
    L.append("## Estimated GPU wall-time (queued files only)")
    L.append("")
    L.append(f"**~{fmt_hms(est_wall_seconds)}** - assumes large-v3 (int8/int8_float16 on CUDA) at "
              f"~{EST_REALTIME_MULTIPLE:.0f}x realtime, the round midpoint of captions.py README's own "
              f"8-15x figure, plus ~{MODEL_LOAD_OVERHEAD_SEC:.0f}s model-load overhead per file (this "
              f"runner invokes captions.py once per file, on purpose, for clean failure isolation and "
              f"resume - not once for the whole batch). Actual speed depends on your GPU, audio "
              f"complexity, and beam size - measure the first handful of files and re-project.")
    L.append("")
    L.append("## Top 20 longest files")
    L.append("")
    if longest:
        L.append("| rel_path | minutes | size_MB |")
        L.append("|---|---|---|")
        for r in longest:
            L.append(f"| {r['rel_path']} | {r['minutes']:.1f} | {r['size_MB']:.1f} |")
    else:
        L.append("(none)")
    L.append("")
    if no_audio:
        L.append("## Skipped - no audio track")
        L.append("")
        for r in no_audio[:200]:
            L.append(f"- `{r['rel_path']}`")
        if len(no_audio) > 200:
            L.append(f"- ...and {len(no_audio) - 200} more (see batch_manifest.csv)")
        L.append("")
    if errors:
        L.append("## Skipped - errors")
        L.append("")
        for r in errors[:200]:
            L.append(f"- `{r['rel_path']}` - {r['note']}")
        if len(errors) > 200:
            L.append(f"- ...and {len(errors) - 200} more (see batch_manifest.csv)")
        L.append("")
    L.append("## Next step")
    L.append("")
    L.append("Review the counts above. If they look right, double-click "
              "`Batch_2_Run_All.cmd` to transcribe the queued files. Re-running "
              "`Batch_1_Make_Manifest.cmd` any time refreshes this file without "
              "touching anything already transcribed.")
    L.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


# --- stage 3: run queue --------------------------------------------------------
def load_queue(root, out_root, exts, formats, rescan):
    csv_path = os.path.join(out_root, MANIFEST_CSV_NAME)
    if rescan or not os.path.isfile(csv_path):
        print("Building inventory (no manifest found, or --rescan requested)...")
        rows, ignored_ext = build_inventory(root, out_root, exts, formats)
        write_manifest_csv(rows, csv_path)
        write_manifest_md(rows, ignored_ext, root, out_root,
                           os.path.join(out_root, MANIFEST_MD_NAME), formats)
    else:
        print(f"Reading existing manifest: {csv_path}")
        rows = read_manifest_csv(csv_path)
    return rows, csv_path


def refresh_status(row, out_root, formats):
    """Re-check ground truth right before deciding to queue a file. Never
    trusts the CSV's status text alone - only what's actually on disk now."""
    path = row["path"]
    stem = captions.safe_stem(path)
    rel_dir = os.path.dirname(row["rel_path"])
    out_dir = (os.path.normpath(os.path.join(out_root, stem)) if rel_dir in ("", ".")
               else os.path.normpath(os.path.join(out_root, rel_dir, stem)))

    if not os.path.isfile(path):
        return "error", "source file no longer found", out_dir, stem
    if not os.access(path, os.R_OK):
        return "error", "source file exists but is not readable (permissions?)", out_dir, stem
    if is_done(out_dir, stem, formats):
        return "done", "already transcribed", out_dir, stem
    if row.get("status") == "no-audio":
        return "no-audio", row.get("note", ""), out_dir, stem
    return "queued", "", out_dir, stem


def log_line(path, msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{ts}  {msg}\n")


def tail_text(s, n_chars=500):
    s = (s or "").strip()
    return s[-n_chars:] if s else ""


def indent_block(s, prefix="           "):
    return "\n".join(prefix + ln for ln in s.splitlines())


def build_argv(src_path, out_dir, args, engine_base):
    if args.dry_run:
        return [sys.executable, os.path.abspath(__file__), "--_stub-worker",
                src_path, "--out-dir", out_dir, "--formats", args.formats]

    base = engine_base if engine_base is not None else [sys.executable, CAPTIONS_PY]
    argv = list(base) + [src_path, "--out-dir", out_dir, "--formats", args.formats]
    if args.language:
        argv += ["--language", args.language]
    if args.prompt:
        argv += ["--prompt", args.prompt]
    if args.gap is not None:
        argv += ["--gap", str(args.gap)]
    if args.model:
        argv += ["--model", args.model]
    if args.device:
        argv += ["--device", args.device]
    if args.compute:
        argv += ["--compute", args.compute]
    if args.beam_size:
        argv += ["--beam-size", str(args.beam_size)]

    extra = shlex.split(args.extra_args) if args.extra_args else []
    if "--burn" in extra:
        print("WARNING: --burn is not allowed in batch mode (no burn-in for batch) - ignoring it.")
        extra = [a for a in extra if a != "--burn"]
    argv += extra
    return argv


def run_queue(rows, out_root, args):
    os.makedirs(out_root, exist_ok=True)
    log_path = os.path.join(out_root, RUN_LOG_NAME)
    formats = parse_formats(args.formats)

    # Refresh EVERY row's status against disk right now - not what the CSV
    # said last time. Rows not selected below still get their status/note
    # corrected in the manifest snapshot written at the end of this run.
    queue = []
    for r in rows:
        status, note, out_dir, stem = refresh_status(r, out_root, formats)
        r["status"], r["note"] = status, note
        r["_out_dir"], r["_stem"] = out_dir, stem
        if status == "queued":
            queue.append(r)

    queue.sort(key=lambda r: (r["duration_sec"] if r["duration_sec"] is not None else 0.0))
    if args.limit:
        queue = queue[:args.limit]

    total = len(queue)
    already_done = sum(1 for r in rows if r["status"] == "done")
    print(f"batch_captions.py  |  {total} file(s) queued to transcribe"
          + (f"  (--limit {args.limit})" if args.limit else "")
          + f"  |  {already_done} already done")

    if total == 0:
        print("Nothing to do - every candidate is done / no-audio / error, or the queue is empty.")
        return 0, 0, 0

    engine_base = shlex.split(args.engine_cmd) if args.engine_cmd else None
    ok_count = fail_count = 0
    t0_all = time.time()

    try:
        for i, r in enumerate(queue, 1):
            rel, out_dir, stem = r["rel_path"], r["_out_dir"], r["_stem"]
            print(f"\n[{i}/{total}] {rel}  (~{r['minutes']:.1f} min)")
            print(f"        -> {out_dir}")
            log_line(log_path, f"START  {rel}")
            t0 = time.time()
            try:
                os.makedirs(out_dir, exist_ok=True)
                argv = build_argv(r["path"], out_dir, args, engine_base)
                if args.verbose:
                    proc = subprocess.run(argv)
                    out_text, err_text = "", ""
                else:
                    proc = subprocess.run(argv, capture_output=True, text=True,
                                           timeout=(args.timeout or None))
                    out_text, err_text = proc.stdout or "", proc.stderr or ""
                elapsed = time.time() - t0

                if proc.returncode != 0:
                    fail_count += 1
                    r["status"], r["note"] = "failed", f"engine exited {proc.returncode}"
                    reason = tail_text(err_text or out_text)
                    last_line = reason.splitlines()[-1] if reason else "(no output captured)"
                    print(f"        -> FAILED (exit {proc.returncode}) in {elapsed:.1f}s: {last_line}")
                    log_line(log_path, f"FAIL   {rel}  ({elapsed:.1f}s)  exit={proc.returncode}")
                    if reason:
                        log_line(log_path, indent_block(reason))
                elif not is_done(out_dir, stem, formats):
                    fail_count += 1
                    r["status"], r["note"] = "failed", "engine exited 0 but expected output file(s) missing"
                    print(f"        -> FAILED in {elapsed:.1f}s: exited clean but expected output missing")
                    log_line(log_path, f"FAIL   {rel}  ({elapsed:.1f}s)  reason: exited 0, outputs missing")
                else:
                    ok_count += 1
                    r["status"], r["note"] = "done", "already transcribed"
                    print(f"        -> ok in {elapsed:.1f}s")
                    log_line(log_path, f"OK     {rel}  ({elapsed:.1f}s)")

            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                fail_count += 1
                r["status"], r["note"] = "failed", f"timeout after {args.timeout}s"
                print(f"        -> FAILED (timeout after {elapsed:.1f}s)")
                log_line(log_path, f"FAIL   {rel}  ({elapsed:.1f}s)  reason: timeout")
            except Exception as e:                    # noqa: BLE001 - one bad file must never kill the queue
                elapsed = time.time() - t0
                fail_count += 1
                r["status"], r["note"] = "failed", str(e)
                print(f"        -> FAILED in {elapsed:.1f}s: {e}")
                log_line(log_path, f"FAIL   {rel}  ({elapsed:.1f}s)  reason: {e}")

    except KeyboardInterrupt:
        remaining = total - ok_count - fail_count
        print("\n\nInterrupted by user (Ctrl+C). Progress is saved - re-run to resume; "
              "already-transcribed files are skipped automatically.")
        log_line(log_path, "INTERRUPTED by user (Ctrl+C)")
        print("\n" + "=" * 60)
        print(f"Stopped after {ok_count + fail_count} of {total} queued file(s) "
              f"({fmt_hms(time.time() - t0_all)} wall time): {ok_count} ok, "
              f"{fail_count} failed, {remaining} still queued for next run.")
        print("=" * 60)
        write_manifest_csv(rows, os.path.join(out_root, MANIFEST_CSV_NAME))
        sys.exit(130)

    _print_tally(ok_count, fail_count, total, time.time() - t0_all)
    return ok_count, fail_count, total


def _print_tally(ok, fail, total, elapsed_sec):
    print("\n" + "=" * 60)
    if fail:
        print(f"Finished with {fail} failure(s) of {total} queued file(s) "
              f"({fmt_hms(elapsed_sec)} wall time).")
        print("  See batch_run.log for failure reasons. Re-run Batch_2 to retry - "
              "done files are skipped automatically, failed ones are retried.")
    else:
        print(f"All {total} queued file(s) transcribed OK ({fmt_hms(elapsed_sec)} wall time).")
    print("=" * 60)


# --- internal: --dry-run stand-in worker --------------------------------------
def stub_worker(argv):
    """Stands in for captions.py when --dry-run is set: writes small, clearly-
    marked placeholder files instead of running faster-whisper. Lets the whole
    orchestration layer (queueing, mirroring, logging, resume, failure
    isolation) be exercised on a machine with no GPU and no model weights.

    Testing hook (inert unless you explicitly set it): if the environment
    variable BATCH_CAPTIONS_STUB_FAIL_SUBSTRING is set and its value appears
    in the source file's name, this exits non-zero instead of writing output,
    simulating an engine crash on that one file so failure isolation can be
    verified deterministically.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--formats", default=DEFAULT_FORMATS)
    ns, _unknown = ap.parse_known_args(argv)

    fail_marker = os.environ.get("BATCH_CAPTIONS_STUB_FAIL_SUBSTRING")
    if fail_marker and fail_marker in os.path.basename(ns.media):
        sys.stderr.write(f"[dry-run stub] simulated engine failure on "
                          f"{os.path.basename(ns.media)} "
                          f"(BATCH_CAPTIONS_STUB_FAIL_SUBSTRING test hook)\n")
        return 7

    os.makedirs(ns.out_dir, exist_ok=True)
    stem = captions.safe_stem(ns.media)
    formats = parse_formats(ns.formats)
    written = []
    for f in formats:
        ext = FORMAT_EXT.get(f)
        if not ext:
            continue
        out_path = os.path.join(ns.out_dir, stem + ext)
        if f == "json":
            payload = {"schema": "caption_engine.transcript/1", "stub": True,
                       "generator": {"tool": "batch_captions.py --dry-run",
                                     "version": __version__},
                       "media": {"path": os.path.abspath(ns.media),
                                 "name": os.path.basename(ns.media)}}
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
        elif f == "srt":
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("1\n00:00:00,000 --> 00:00:01,000\n"
                         "[STUB - dry run, no real transcription]\n\n")
        elif f == "vtt":
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n"
                         "[STUB - dry run, no real transcription]\n\n")
        else:  # txt, ass, or anything else -> plain placeholder text
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(f"[STUB TRANSCRIPT - dry run, no GPU/model used]\n"
                         f"source: {os.path.basename(ns.media)}\n")
        written.append(out_path)
    print(f"[dry-run stub] wrote {len(written)} placeholder file(s) to {ns.out_dir}")
    return 0


# --- driver --------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="batch_captions.py",
        description="Manifest-first batch runner for captions.py: inventory a "
                    "whole voice-memo/audio corpus, review it, then transcribe the "
                    "queue on the GPU. Two-step: --manifest, then --run.",
        epilog="Nothing is uploaded - same offline guarantee as captions.py. Outputs "
              "mirror into <root>\\_transcripts\\, never scattered next to the source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--root", default=DEFAULT_ROOT, help="corpus root to scan recursively")

    mo = p.add_argument_group("mode (pass --manifest, --run, or both)")
    mo.add_argument("--manifest", action="store_true",
                    help="scan + probe the corpus; write batch_manifest.csv and "
                         "BATCH_MANIFEST.md, then exit. No GPU work.")
    mo.add_argument("--run", action="store_true",
                    help="consume the manifest queue (or scan fresh if absent) and "
                         "transcribe every queued file, sequentially")

    sc = p.add_argument_group("scan")
    sc.add_argument("--exts", default=None,
                    help="comma list of extensions to treat as media, overriding the "
                         "built-in default (" + ", ".join(sorted(DEFAULT_EXTS)) + ")")
    sc.add_argument("--rescan", action="store_true",
                    help="ignore any existing batch_manifest.csv and rescan the whole "
                         "corpus before running (always implied when the manifest is "
                         "missing; --manifest itself always rescans)")

    en = p.add_argument_group("engine (forwarded to captions.py)")
    en.add_argument("--language", default=DEFAULT_LANGUAGE, help="forwarded as --language")
    en.add_argument("--prompt", default=DEFAULT_PROMPT, help="forwarded as --prompt")
    en.add_argument("--gap", type=float, default=DEFAULT_GAP, help="forwarded as --gap")
    en.add_argument("--formats", default=DEFAULT_FORMATS,
                    help="forwarded as --formats; ALSO decides what \"already "
                         "transcribed\" means for resume. No burn-in in batch mode - "
                         "--burn is never sent and is stripped from --extra-args.")
    en.add_argument("--model", default=None, help="forwarded as --model, if set")
    en.add_argument("--device", default=None, help="forwarded as --device, if set")
    en.add_argument("--compute", default=None, help="forwarded as --compute, if set")
    en.add_argument("--beam-size", type=int, default=None, help="forwarded as --beam-size, if set")
    en.add_argument("--extra-args", default="",
                    help="extra raw flags appended verbatim to every captions.py call "
                         "(shell-quoted, e.g. \"--no-condition --vad\"). Use forward "
                         "slashes for any embedded paths - this is tokenized with shlex.")

    rc = p.add_argument_group("run control")
    rc.add_argument("--limit", type=int, default=None,
                    help="process at most N queued files this run, shortest first - a "
                         "starter batch; the rest stay queued for next time")
    rc.add_argument("--timeout", type=float, default=None,
                    help="per-file subprocess timeout in seconds (default: none)")
    rc.add_argument("--verbose", action="store_true",
                    help="let captions.py's own staged output stream live to the "
                         "console instead of being captured/condensed")

    ts = p.add_argument_group("testability (no GPU / no model weights required)")
    ts.add_argument("--dry-run", action="store_true",
                    help="substitute an internal stub worker for captions.py: writes "
                         "placeholder outputs so queueing/logging/resume/failure-"
                         "isolation can be exercised without a GPU or model")
    ts.add_argument("--engine-cmd", default=None,
                    help="override the base command used to invoke the transcription "
                         "engine (shell-quoted string); default is this venv's python "
                         "+ captions.py")
    return p


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_stub-worker":
        sys.exit(stub_worker(sys.argv[2:]))

    args = build_parser().parse_args()

    if not args.manifest and not args.run:
        build_parser().print_help()
        print("\nERROR: pass --manifest, --run, or both.")
        sys.exit(2)
    if args.limit is not None and args.limit <= 0:
        print("ERROR: --limit must be a positive integer.")
        sys.exit(2)

    formats = parse_formats(args.formats)
    bad = [f for f in formats if f not in FORMAT_EXT]
    if bad:
        print(f"ERROR: unknown --formats value(s): {', '.join(bad)} "
              f"(known: {', '.join(FORMAT_EXT)})")
        sys.exit(2)
    args.formats = ",".join(formats)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"ERROR: scan root not found: {root}")
        print("  Pass --root <folder> to point at a different corpus.")
        sys.exit(1)
    out_root = os.path.join(root, OUTPUT_DIRNAME)
    exts = parse_exts(args.exts) if args.exts else DEFAULT_EXTS

    require_ffmpeg()

    print(f"batch_captions.py v{__version__}  |  wraps captions.py "
          f"v{getattr(captions, '__version__', '?')}")
    print(f"  root      : {root}")
    print(f"  outputs   : {out_root}")

    if args.manifest:
        print(f"\nbatch_captions.py  |  manifest mode  |  scanning: {root}")
        t0 = time.time()
        rows, ignored_ext = build_inventory(root, out_root, exts, formats)
        csv_path = os.path.join(out_root, MANIFEST_CSV_NAME)
        md_path = os.path.join(out_root, MANIFEST_MD_NAME)
        write_manifest_csv(rows, csv_path)
        write_manifest_md(rows, ignored_ext, root, out_root, md_path, formats)
        elapsed = time.time() - t0

        counts = Counter(r["status"] for r in rows)
        print(f"\n{len(rows)} media file(s) found in {elapsed:.1f}s "
              f"({counts['queued']} queued, {counts['done']} done, "
              f"{counts['no-audio']} no-audio, {counts['error']} error)")
        print(f"  manifest  : {csv_path}")
        print(f"  summary   : {md_path}")

        # Drop a pointer so Batch_1_Make_Manifest.cmd can open the right
        # BATCH_MANIFEST.md in notepad even if --root was overridden.
        try:
            ptr = os.path.join(tempfile.gettempdir(), "batch_captions_last_manifest.txt")
            with open(ptr, "w", encoding="utf-8") as f:
                f.write(md_path + "\n")
        except OSError:
            pass

    if args.run:
        print()
        rows, csv_path = load_queue(root, out_root, exts, formats, args.rescan)
        _ok, fail, _total = run_queue(rows, out_root, args)
        write_manifest_csv(rows, csv_path)
        sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
