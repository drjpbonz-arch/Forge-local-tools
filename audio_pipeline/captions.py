#!/usr/bin/env python3
"""
captions.py - offline, private captioning for a local media console.

Transcribes video or audio locally with faster-whisper (CTranslate2), keeps the
WORD-LEVEL timestamps, and writes broadcast-shaped caption files:

    <name>/<name>.srt              numbered cues, <=42 chars/line, <=2 lines
    <name>/<name>.vtt              valid WebVTT
    <name>/<name>.transcript.json  word-level data layer (click-to-edit UI)
    <name>/<name>.txt              plain reading transcript

Optionally burns styled captions into a copy of the video (--burn) with ffmpeg's
subtitles filter: white text, black outline, lower-third margin.

Nothing leaves the machine. No API keys, no upload, no telemetry. Point it at a
model folder with --local-only and it will refuse to touch the network at all.

PAUSE-RESPECT DOCTRINE (house rule, not negotiable)
    A speaker's pauses are weighted - they carry meaning. This tool NEVER collapses,
    trims, or "smart"-fills silence. Cue times are the spoken word times:
      * a pause longer than --gap simply becomes a gap BETWEEN cues (screen clear)
      * a cue ends when its last word ends - it is never stretched to meet the
        next cue (turn that on deliberately with --min-cue)
      * VAD silence-removal is OFF by default (--vad to opt in, with a warning)

Pipeline (each stage is its own function):
    1. probe          - ffprobe the media, confirm there is an audio stream
    2. load_model     - resolve device/model/compute, build the WhisperModel
    3. transcribe     - faster-whisper with word_timestamps=True
    4. build_cues     - pack words into readable cues on word/punctuation breaks
    5. write outputs  - srt / vtt / transcript.json / txt
    6. burn_in        - optional ffmpeg hardsub pass (CRF 18 video, audio copied)

Dependencies: faster-whisper + ffmpeg/ffprobe on PATH. No torch, no numpy, no
cloud SDKs - matching the house style of bed_prep.py / highlight_clip.py.

Usage:
    python captions.py clip.mov
    python captions.py clip.mov --language en --burn
    python captions.py *.mov --model large-v3 --device cuda
    python captions.py --selftest
See --help for every flag.
"""

import os
import re
import sys
import json
import glob
import shutil
import argparse
import tempfile
import subprocess

__version__ = "1.0.0"

# --- defaults ---------------------------------------------------------------
MAX_CHARS_PER_LINE = 42      # broadcast-ish; comfortable on a phone
MAX_LINES_PER_CUE = 2        # never a wall of text
MAX_CUE_SECONDS = 6.0        # hard ceiling on one cue's screen time
GAP_SECONDS = 0.70           # silence longer than this forces a cue break
SENTENCE_MIN_CHARS = 12      # don't flush on "." if the cue is still tiny
PARAGRAPH_GAP = 2.00         # .txt paragraph break at pauses this long

# burn-in house look
BURN_FONT = "Arial"
BURN_FONTSIZE = 48           # real pixels: we author PlayRes = the video's size
BURN_MARGIN_V = 90           # lower third, measured up from the bottom edge
BURN_MARGIN_H = 60           # keep text clear of the left/right edges
BURN_OUTLINE = 3
BURN_CRF = 18

SENTENCE_END = (".", "!", "?", "\u2026")          # . ! ? ...
CLAUSE_END = (",", ";", ":", "\u2014", "-")       # , ; : -- -


# --- small shared helpers ---------------------------------------------------
def have(cmd):
    return shutil.which(cmd) is not None


def run(cmd, cwd=None):
    """Run an ffmpeg/ffprobe command, returning the CompletedProcess (stderr captured)."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def ffprobe_json(path):
    r = run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
             "-of", "json", path])
    try:
        return json.loads(r.stdout)
    except (ValueError, TypeError):
        return None


def fmt_ts(seconds, sep=","):
    """Seconds -> HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT). Never negative."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def safe_stem(path):
    base = os.path.splitext(os.path.basename(os.path.normpath(path)))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "captions"


def human(seconds):
    if seconds is None:
        return "?"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _r(v, n):
    """Round for JSON, tolerating None / non-numerics."""
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


# --- stage 1: probe ---------------------------------------------------------
def probe(path, stage):
    """ffprobe the media; return (duration, has_video, has_audio, width, height)."""
    info = ffprobe_json(path)
    if not info:
        raise RuntimeError(f"ffprobe could not read: {path}")
    streams = info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    has_video = bool(vstreams)
    width = height = None
    if has_video:
        try:
            width = int(vstreams[0].get("width"))
            height = int(vstreams[0].get("height"))
        except (TypeError, ValueError):
            width = height = None
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    if not has_audio:
        raise RuntimeError(f"no audio stream in {os.path.basename(path)} - nothing to caption")
    kind = "video+audio" if has_video else "audio only"
    if width and height:
        kind += f" {width}x{height}"
    dur_txt = f", {human(duration)} ({duration:.2f}s)" if duration else ""
    print(f"  {stage} probe          : {kind}{dur_txt}")
    return duration, has_video, has_audio, width, height


# --- stage 2: model ---------------------------------------------------------
def register_cuda_dll_dirs():
    """Windows only: make the NVIDIA pip wheels' DLLs findable. Returns dirs added.

    `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` drops cublas64_12.dll and
    cudnn*_9.dll into site-packages\\nvidia\\<pkg>\\bin. Since Python 3.8, Windows
    no longer searches PATH when loading an extension module's dependencies, so
    CTranslate2 fails with "Library cublas64_12.dll is not found" even though the
    wheels are installed - the classic Windows+faster-whisper GPU trap. Registering
    the folders with os.add_dll_directory() fixes it without touching system PATH.

    Harmless no-op on Linux/macOS, or when the wheels aren't installed.
    """
    if os.name != "nt":
        return []
    added = []
    try:
        import sysconfig
        paths = sysconfig.get_paths()
        roots = {paths.get("purelib"), paths.get("platlib")}
        for root in filter(None, roots):
            nvidia = os.path.join(root, "nvidia")
            if not os.path.isdir(nvidia):
                continue
            for sub in sorted(os.listdir(nvidia)):
                bindir = os.path.join(nvidia, sub, "bin")
                if os.path.isdir(bindir) and bindir not in added:
                    try:
                        os.add_dll_directory(bindir)
                        added.append(bindir)
                    except OSError:
                        pass
        # Belt AND suspenders: some NVIDIA DLLs resolve their own dependencies
        # through PATH rather than the add_dll_directory search list (observed
        # with cublas/cudnn on Windows despite installed wheels). Prepend the
        # same folders to PATH for this process so both loaders succeed.
        if added:
            os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    return added


def cuda_device_count():
    """CUDA device count via CTranslate2 (no torch import). 0 if unavailable."""
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def supported_compute_types(device):
    try:
        import ctranslate2
        return ", ".join(sorted(ctranslate2.get_supported_compute_types(device)))
    except Exception:
        return "unknown"


def resolve_device(requested):
    """Return (device, note). 'auto' picks cuda when CTranslate2 can see a GPU."""
    n = cuda_device_count()
    if requested == "cuda":
        if n == 0:
            raise RuntimeError(
                "--device cuda requested but CTranslate2 sees 0 CUDA devices.\n"
                "  On Windows+NVIDIA the CUDA 12 / cuDNN 9 runtime comes from pip wheels:\n"
                "    pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12>=9,<10\"\n"
                "  (see caption_engine\\README.md, or run Setup_CaptionEngine_Windows.cmd)\n"
                "  Or drop the flag / use --device cpu to run on the CPU.")
        return "cuda", f"{n} CUDA device(s) visible"
    if requested == "cpu":
        return "cpu", "forced by --device cpu"
    if n > 0:
        return "cuda", f"auto-picked, {n} CUDA device(s) visible"
    return "cpu", "auto-picked, no CUDA visible"


def resolve_model(requested, device):
    """'auto' -> large-v3 on GPU, small on CPU. Anything else passes through
    (a size name like 'medium', or a path to a converted CTranslate2 folder)."""
    if requested != "auto":
        return requested
    return "large-v3" if device == "cuda" else "small"


def resolve_compute(requested, device):
    """'auto' -> int8_float16 on CUDA (house doctrine), int8 on CPU."""
    if requested != "auto":
        return requested
    return "int8_float16" if device == "cuda" else "int8"


def load_model(args, stage):
    """Build the WhisperModel, with human-readable failure modes."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed.\n"
            "    pip install faster-whisper\n"
            "  (Windows: run Setup_CaptionEngine_Windows.cmd instead - it also\n"
            "   installs the CUDA 12 / cuDNN 9 pip wheels for GPU speed.)")

    device, dev_note = resolve_device(args.device)
    model_name = resolve_model(args.model, device)
    compute = resolve_compute(args.compute, device)

    kwargs = dict(device=device, compute_type=compute,
                  local_files_only=args.local_only)
    if args.download_root:
        kwargs["download_root"] = args.download_root
    if device == "cpu" and args.cpu_threads:
        kwargs["cpu_threads"] = args.cpu_threads

    try:
        model = WhisperModel(model_name, **kwargs)
    except ValueError as e:
        # Most often an unsupported compute_type for this device.
        raise RuntimeError(
            f"could not load model '{model_name}' as {compute} on {device}: {e}\n"
            f"  Supported here: {supported_compute_types(device)}\n"
            f"  Try --compute int8 (CPU) or --compute float16 / int8_float16 (CUDA).")
    except Exception as e:
        msg = str(e).lower()
        offline_ish = ("connection", "resolve", "network", "offline", "proxy",
                       "timed out", "couldn't find", "no such file", "403",
                       "404", "ssl", "local_files_only")
        if any(k in msg for k in offline_ish):
            raise RuntimeError(
                f"could not obtain model '{model_name}': {e}\n"
                "  This machine may be offline or firewalled off from huggingface.co.\n"
                "  Fetch the model once on a connected machine, then either:\n"
                "    - copy the model cache over and pass --download-root <folder>, or\n"
                "    - pass the converted model folder directly:\n"
                "        --model D:\\models\\faster-whisper-large-v3\n"
                "  Add --local-only to guarantee no network access is attempted.")
        raise RuntimeError(f"could not load model '{model_name}': {e}")

    print(f"  {stage} model          : {model_name} | {device} ({dev_note}) | {compute}")
    return model, model_name, device, compute


# --- stage 3: transcribe ----------------------------------------------------
def transcribe(model, path, args, stage):
    """Run faster-whisper with word timestamps on. Returns (segments, info)."""
    opts = dict(
        word_timestamps=True,                 # the whole point: the data layer
        beam_size=args.beam_size,
        language=(None if args.language in (None, "auto") else args.language),
        task=args.task,
        initial_prompt=args.prompt,
        condition_on_previous_text=not args.no_condition,
        vad_filter=args.vad,                  # OFF by default - pause doctrine
    )
    if args.temperature is not None:
        opts["temperature"] = args.temperature
    if args.vad:
        print(f"  {stage} transcribe     : WARNING --vad is on; silence-removal can "
              f"shift timings.\n                        House pause doctrine prefers "
              f"--vad off.")

    try:
        seg_iter, info = model.transcribe(path, **opts)
    except Exception as e:
        raise RuntimeError(f"transcription failed: {e}")

    lang = getattr(info, "language", None) or "?"
    prob = getattr(info, "language_probability", None)
    langnote = str(lang) + (f" (p={prob:.2f})" if isinstance(prob, float) else "")
    if args.language not in (None, "auto"):
        langnote += " [forced]"
    print(f"  {stage} transcribe     : language {langnote} | beam {args.beam_size} | "
          f"word timestamps ON", flush=True)

    # The generator does the real work - drain it, reporting as we go.
    segments, words_total = [], 0
    for seg in seg_iter:
        segments.append(seg)
        words_total += len(seg.words or ())
        if args.progress:
            print(f"        [{fmt_ts(seg.start, '.')} -> {fmt_ts(seg.end, '.')}] "
                  f"{(seg.text or '').strip()[:70]}", flush=True)
    print(f"                        {len(segments)} segment(s), {words_total} word(s)")
    if segments and not any(s.words for s in segments):
        print("                        NOTE: model returned no word timings - "
              "falling back to segment-level cues")
    return segments, info


# --- stage 4: cue layout ----------------------------------------------------
class Cue(object):
    """One caption event: the words it covers, laid out over 1-2 lines."""

    __slots__ = ("words", "lines", "start", "end", "reason")

    def __init__(self, words, lines, start, end, reason=""):
        self.words = words
        self.lines = lines
        self.start = start
        self.end = end
        self.reason = reason

    @property
    def text(self):
        return "\n".join(self.lines)


def flatten_words(segments):
    """Segments -> flat [{word, start, end, confidence}] list, timings untouched."""
    out = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                txt = (w.word or "").strip()
                if not txt:
                    continue
                out.append({"word": txt,
                            "start": float(w.start),
                            "end": float(w.end),
                            "confidence": (None if w.probability is None
                                           else float(w.probability))})
        else:
            # No word timings for this segment - keep it whole rather than guess.
            txt = (seg.text or "").strip()
            if txt:
                out.append({"word": txt, "start": float(seg.start),
                            "end": float(seg.end), "confidence": None})
    return out


def split_lines(text, max_chars, max_lines):
    """Wrap a cue's text onto <=max_lines lines of <=max_chars, on word breaks.

    Two lines are balanced around the middle, preferring a break just after
    punctuation so the reader's eye lands somewhere natural. A single word longer
    than max_chars is allowed to overflow rather than be split mid-word.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars or max_lines <= 1:
        return [text]

    tokens = text.split(" ")
    best, best_score = None, None
    for i in range(1, len(tokens)):
        left = " ".join(tokens[:i])
        right = " ".join(tokens[i:])
        if len(left) > max_chars:
            break
        if len(right) > max_chars * (max_lines - 1):
            continue
        score = abs(len(left) - len(right))
        if left.endswith(SENTENCE_END):
            score -= 24
        elif left.endswith(CLAUSE_END):
            score -= 12
        if best_score is None or score < best_score:
            best, best_score = i, score

    if best is None:
        # Nothing fits cleanly: greedy-fill and let the tail overflow.
        lines, cur = [], ""
        for t in tokens:
            cand = f"{cur} {t}".strip()
            if cur and len(cand) > max_chars and len(lines) < max_lines - 1:
                lines.append(cur)
                cur = t
            else:
                cur = cand
        lines.append(cur)
        return lines[:max_lines]

    left = " ".join(tokens[:best])
    right = " ".join(tokens[best:])
    if max_lines == 2 or len(right) <= max_chars:
        return [left, right]
    return [left] + split_lines(right, max_chars, max_lines - 1)


def build_cues(words, args, stage):
    """Pack words into cues. Breaks are chosen by, in order of authority:

        1. a silence longer than --gap        (pause doctrine: screen goes clear)
        2. the --max-cue-seconds ceiling
        3. sentence-final punctuation         (once the cue has some substance)
        4. running out of character budget    (max-chars x max-lines)

    Cue start = first word's start. Cue end = last word's END - never stretched
    toward the next cue. --min-cue opts into padding short cues into the silence
    that follows them (still never into the next cue).
    """
    budget = args.max_chars * args.max_lines
    cues, cur = [], []
    reason = "start"
    gaps = 0

    def flush(why):
        nonlocal cur, reason
        if not cur:
            return
        text = " ".join(w["word"] for w in cur)
        lines = split_lines(text, args.max_chars, args.max_lines)
        cues.append(Cue(list(cur), lines, cur[0]["start"], cur[-1]["end"], reason))
        cur = []
        reason = why

    for w in words:
        if cur:
            prev = cur[-1]
            gap = w["start"] - prev["end"]
            cur_text = " ".join(x["word"] for x in cur)
            if gap > args.gap:
                gaps += 1
                flush(f"pause {gap:.2f}s")
            elif (w["end"] - cur[0]["start"]) > args.max_cue_seconds:
                flush("duration")
            elif (prev["word"].endswith(SENTENCE_END)
                  and len(cur_text) >= SENTENCE_MIN_CHARS):
                flush("sentence")
            elif len(cur_text) + 1 + len(w["word"]) > budget:
                flush("length")
        cur.append(w)
    flush("end")

    # Optional readability padding into trailing silence (off by default).
    padded = 0
    if args.min_cue > 0:
        for i, c in enumerate(cues):
            if (c.end - c.start) >= args.min_cue:
                continue
            next_start = cues[i + 1].start if i + 1 < len(cues) else None
            want = c.start + args.min_cue
            new_end = want if next_start is None else min(want, next_start - 0.001)
            if new_end > c.end:
                c.end = new_end
                padded += 1

    # Safety: non-overlapping, positive-length cues (1 ms of air between them).
    fixed = 0
    for i in range(len(cues) - 1):
        if cues[i].end > cues[i + 1].start:
            cues[i].end = max(cues[i].start + 0.001, cues[i + 1].start - 0.001)
            fixed += 1
    for c in cues:
        if c.end <= c.start:
            c.end = c.start + 0.30

    note = (f"  {stage} cue layout     : {len(cues)} cue(s) | <={args.max_chars} chars x "
            f"{args.max_lines} lines | {gaps} pause break(s)")
    if padded:
        note += f" | {padded} padded to --min-cue"
    if fixed:
        note += f" | {fixed} overlap(s) trimmed"
    print(note)
    if args.min_cue <= 0:
        print("                        pause doctrine: cue ends = spoken word ends "
              "(no stretching)")
    return cues


# --- stage 5: writers -------------------------------------------------------
def write_srt(cues, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, c in enumerate(cues, start=1):
            f.write(f"{i}\n")
            f.write(f"{fmt_ts(c.start, ',')} --> {fmt_ts(c.end, ',')}\n")
            f.write(c.text + "\n\n")
    return path


def vtt_escape(text):
    """WebVTT payload escaping (cue text is parsed as limited markup)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_vtt(cues, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("WEBVTT\n\n")
        for i, c in enumerate(cues, start=1):
            f.write(f"{i}\n")
            f.write(f"{fmt_ts(c.start, '.')} --> {fmt_ts(c.end, '.')}\n")
            f.write(vtt_escape(c.text) + "\n\n")
    return path


def write_txt(cues, path, para_gap=PARAGRAPH_GAP):
    """Plain reading transcript. Paragraph break where the speaker took a real breath."""
    paras, cur, prev_end = [], [], None
    for c in cues:
        if prev_end is not None and (c.start - prev_end) > para_gap and cur:
            paras.append(" ".join(cur))
            cur = []
        cur.append(" ".join(c.lines))
        prev_end = c.end
    if cur:
        paras.append(" ".join(cur))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(paras).strip() + "\n")
    return path


def write_json(path, media, words, segments, cues, info, meta):
    """The data layer for a future click-transcript-to-edit UI in the console.

    Top-level 'words' is exactly [{word, start, end, confidence}, ...].
    'segments' carries the model's own diagnostics and indexes back into 'words'.
    'cues' carries the laid-out caption events and their word ranges, so a UI can
    map a click in the transcript to a caption event and a media timecode.
    """
    seg_out, cursor = [], 0
    for s in segments:
        kept = [w for w in (s.words or ()) if (w.word or "").strip()]
        n = len(kept) if s.words else 1
        seg_out.append({
            "id": int(s.id),
            "start": _r(s.start, 3),
            "end": _r(s.end, 3),
            "text": (s.text or "").strip(),
            "word_index": [cursor, cursor + n],       # [start, end) into "words"
            "avg_logprob": _r(s.avg_logprob, 4),
            "no_speech_prob": _r(s.no_speech_prob, 4),
            "compression_ratio": _r(s.compression_ratio, 4),
            "temperature": _r(s.temperature, 2),
        })
        cursor += n

    idx = {id(w): i for i, w in enumerate(words)}     # identity map, not text match
    cue_out = []
    for i, c in enumerate(cues, start=1):
        first = idx.get(id(c.words[0])) if c.words else None
        last = idx.get(id(c.words[-1])) if c.words else None
        cue_out.append({
            "index": i,
            "start": _r(c.start, 3),
            "end": _r(c.end, 3),
            "lines": c.lines,
            "text": " ".join(c.lines),
            "word_index": [first, (last + 1) if last is not None else None],
            "break_before": c.reason,
        })

    doc = {
        "schema": "caption_engine.transcript/1",
        "generator": {"tool": "captions.py", "version": __version__,
                      "engine": "faster-whisper", **meta},
        "media": media,
        "language": {"code": getattr(info, "language", None),
                     "probability": _r(getattr(info, "language_probability", None), 4)},
        "duration": _r(getattr(info, "duration", None), 3),
        "pause_doctrine": {
            "vad_filter": bool(meta.get("vad")),
            "min_cue_padding": meta.get("min_cue", 0.0),
            "note": "cue times follow spoken word times; silence is never collapsed",
        },
        "counts": {"words": len(words), "segments": len(seg_out), "cues": len(cue_out)},
        "words": [{"word": w["word"],
                   "start": _r(w["start"], 3),
                   "end": _r(w["end"], 3),
                   "confidence": _r(w["confidence"], 4)} for w in words],
        "segments": seg_out,
        "cues": cue_out,
    }
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


# --- stage 6: burn-in -------------------------------------------------------
def ass_color(hexrgb):
    """'FFFFFF' or '#FFFFFF' -> ASS &HAABBGGRR (opaque)."""
    h = hexrgb.strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", h):
        raise ValueError(f"expected a 6-digit hex colour, got '{hexrgb}'")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def style_fields(args):
    """The house caption look as (key, value) pairs, in real pixels.

    These are ASS style values. They only mean *pixels* because we author the ASS
    header ourselves with PlayResX/PlayResY set to the video's true size - see
    write_ass() for why that matters.
    """
    border_style, back_colour = "1", "&H80000000"
    if args.box:
        # BorderStyle=4 fills a box behind the text; BackColour is &HAABBGGRR,
        # so the leading byte is transparency (00 = opaque, FF = invisible).
        border_style = "4"
        back_colour = f"&H{args.box_alpha:02X}000000"
    return [
        ("FontName", args.font),
        ("FontSize", f"{args.fontsize}"),
        ("PrimaryColour", ass_color(args.text_color)),
        ("OutlineColour", ass_color(args.outline_color)),
        ("BackColour", back_colour),
        ("Bold", "-1" if args.bold else "0"),
        ("Italic", "0"),
        ("BorderStyle", border_style),
        ("Outline", f"{args.outline:g}"),
        ("Shadow", "0"),
        ("Alignment", "2"),                  # bottom-centre
        ("MarginL", f"{args.margin_h}"),
        ("MarginR", f"{args.margin_h}"),
        ("MarginV", f"{args.margin_v}"),
    ]


def force_style(args):
    """style_fields() flattened into an ffmpeg force_style= override string."""
    return ",".join(f"{k}={v}" for k, v in style_fields(args))


def ass_time(seconds):
    """ASS wants H:MM:SS.cc (centiseconds, single-digit hour)."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    cs = int(round(seconds * 100.0))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def ass_text(lines):
    r"""Cue lines -> one ASS event payload. '\N' is a hard line break; '{'/'}'
    would open an override block, so they are neutralised."""
    safe = [ln.replace("\\", "\u2216").replace("{", "(").replace("}", ")")
            for ln in lines]
    return r"\N".join(safe)


def write_ass(cues, path, args, width, height):
    """Author a complete ASS subtitle file with the house style baked in.

    WHY NOT JUST force_style ON THE SRT:
        ffmpeg's SRT decoder synthesises an ASS header with a fixed virtual canvas
        of 384x288 (ASS_DEFAULT_PLAYRESX/Y). libass then scales that canvas up to
        the real frame, so on 1080p every size is multiplied by 1080/288 = 3.75:
        "FontSize=48" lands as ~180 px and "MarginV=90" as ~338 px. The flags would
        be lying. Writing our own header with PlayResX/PlayResY = the true frame
        size makes the scale factor exactly 1.0, so --fontsize / --margin-v /
        --outline are honest pixel values at any resolution.

    WrapStyle 0 (smart wrap) is deliberate: we already break lines ourselves, so
    it should never fire - but if someone passes a huge --fontsize it wraps instead
    of running the text off the side of the frame.
    """
    width = width or 1920
    height = height or 1080
    fields = dict(style_fields(args))
    order = ["Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour",
             "OutlineColour", "BackColour", "Bold", "Italic", "Underline",
             "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle",
             "Outline", "Shadow", "Alignment", "MarginL", "MarginR", "MarginV",
             "Encoding"]
    style = {
        "Name": "Default", "Fontname": fields["FontName"],
        "Fontsize": fields["FontSize"], "PrimaryColour": fields["PrimaryColour"],
        "SecondaryColour": "&H000000FF", "OutlineColour": fields["OutlineColour"],
        "BackColour": fields["BackColour"], "Bold": fields["Bold"],
        "Italic": fields["Italic"], "Underline": "0", "StrikeOut": "0",
        "ScaleX": "100", "ScaleY": "100", "Spacing": "0", "Angle": "0",
        "BorderStyle": fields["BorderStyle"], "Outline": fields["Outline"],
        "Shadow": fields["Shadow"], "Alignment": fields["Alignment"],
        "MarginL": fields["MarginL"], "MarginR": fields["MarginR"],
        "MarginV": fields["MarginV"], "Encoding": "1",
    }
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("[Script Info]\n")
        f.write("; Generated by captions.py\n")
        f.write("ScriptType: v4.00+\n")
        f.write(f"PlayResX: {width}\nPlayResY: {height}\n")
        f.write("WrapStyle: 0\nScaledBorderAndShadow: yes\nYCbCr Matrix: None\n\n")
        f.write("[V4+ Styles]\nFormat: " + ", ".join(order) + "\n")
        f.write("Style: " + ",".join(style[k] for k in order) + "\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, "
                "MarginR, MarginV, Effect, Text\n")
        for c in cues:
            f.write(f"Dialogue: 0,{ass_time(c.start)},{ass_time(c.end)},Default,,"
                    f"0,0,0,,{ass_text(c.lines)}\n")
    return path


def burn_in(video, cues, out_path, args, width, height, stage, keep_ass=None):
    """Hardsub `cues` into a copy of `video`: CRF 18 H.264, audio stream copied.

    ffmpeg's subtitles= filter takes a path INSIDE a filtergraph, where ':' and
    '\\' are syntax - which makes Windows paths ("C:\\Users\\...") a minefield.
    Rather than fight the escaping, we write the styled ASS into a temp dir under
    a plain ASCII name and run ffmpeg with cwd set there, so the filter only ever
    sees "burn.ass". Input and output stay absolute (ordinary args, no escaping).
    """
    tmp = tempfile.mkdtemp(prefix="burnsub_")
    try:
        ass_path = os.path.join(tmp, "burn.ass")
        write_ass(cues, ass_path, args, width, height)
        if keep_ass:
            shutil.copyfile(ass_path, keep_ass)
        # force_style is redundant with our own header but harmless, and keeps the
        # look correct even if someone hand-edits the .ass we hand back.
        vf = f"subtitles=burn.ass:force_style='{force_style(args)}'"
        cmd = ["ffmpeg", "-y", "-i", os.path.abspath(video),
               "-vf", vf,
               "-c:v", "libx264", "-crf", str(args.crf), "-preset", args.preset,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-c:a", "copy", os.path.abspath(out_path)]
        r = run(cmd, cwd=tmp)
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            tail = (r.stderr or "")[-1200:]
            if "No such filter" in tail or "Unknown filter" in tail:
                tail += ("\n  This ffmpeg build has no libass/subtitles filter - install a "
                         "full build (https://www.gyan.dev/ffmpeg/builds/).")
            elif "codec_copy" in tail or "Could not write header" in tail:
                tail += "\n  Try --audio-reencode if the source audio can't be copied to MP4."
            raise RuntimeError(f"burn-in failed:\n{tail}")
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  {stage} burn-in        : {os.path.basename(out_path)} "
              f"({size_mb:.1f} MB) | {args.font} {args.fontsize}px, outline "
              f"{args.outline:g}, MarginV {args.margin_v}, CRF {args.crf} "
              f"@ {width or '?'}x{height or '?'}")
        return out_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- driver -----------------------------------------------------------------
def selftest(args):
    """--selftest: report what WOULD be used, and whether the deps are healthy."""
    print("captions.py self-test")
    print(f"  captions.py     : v{__version__}")
    ok = True

    dlls = register_cuda_dll_dirs()
    if os.name == "nt":
        print(f"  CUDA dll dirs   : {len(dlls)} registered"
              + (f" ({os.path.dirname(dlls[0])}\\*)" if dlls else " (no nvidia wheels found)"))

    try:
        import faster_whisper
        print(f"  faster-whisper  : {faster_whisper.__version__}")
    except ImportError:
        print("  faster-whisper  : MISSING  ->  pip install faster-whisper")
        ok = False
    try:
        import ctranslate2
        print(f"  ctranslate2     : {ctranslate2.__version__}")
    except ImportError:
        print("  ctranslate2     : MISSING  (ships with faster-whisper)")
        ok = False

    n = cuda_device_count()
    print(f"  CUDA devices    : {n}" + ("" if n else "   (CPU mode)"))
    if n == 0:
        print("                    for GPU speed on Windows+NVIDIA, install the "
              "CUDA 12 / cuDNN 9 pip wheels")
    for dev in (["cuda"] if n else []) + ["cpu"]:
        print(f"  compute types   : {dev}: {supported_compute_types(dev)}")

    try:
        device, note = resolve_device(args.device)
        print(f"  WOULD USE       : model={resolve_model(args.model, device)}  "
              f"device={device}  compute={resolve_compute(args.compute, device)}")
        print(f"                    ({note})")
    except RuntimeError as e:
        print(f"  WOULD USE       : ERROR {e}")
        ok = False

    for tool in ("ffmpeg", "ffprobe"):
        present = have(tool)
        print(f"  {tool:<15} : {'on PATH' if present else 'MISSING - required'}")
        ok = ok and present

    cache = (args.download_root or os.environ.get("HF_HUB_CACHE")
             or os.environ.get("HF_HOME")
             or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    print(f"  model cache     : {cache}")
    print(f"  local-only      : {args.local_only}   (True = never touch the network)")
    print("\nSELF-TEST OK" if ok else
          "\nSELF-TEST INCOMPLETE - see the MISSING lines above")
    return 0 if ok else 1


def caption_one(path, args, model_bundle):
    """Full pipeline for one media file. Returns a list of written paths."""
    _model, model_name, device, compute = model_bundle
    total = 6 if args.burn else 5
    stem = safe_stem(path)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(path)), stem)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nCaptioning: {path}")
    print(f"Output dir: {out_dir}")

    duration, has_video, _, width, height = probe(path, f"[1/{total}]")
    print(f"  [2/{total}] model          : {model_name} | {device} | {compute}")
    segments, info = transcribe(_model, path, args, f"[3/{total}]")

    words = flatten_words(segments)
    if not words:
        print("  ! no speech found - writing empty caption files anyway")
    cues = build_cues(words, args, f"[4/{total}]")

    made = []
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    if "srt" in formats:
        made.append(write_srt(cues, os.path.join(out_dir, f"{stem}.srt")))
    if "vtt" in formats:
        made.append(write_vtt(cues, os.path.join(out_dir, f"{stem}.vtt")))
    if "json" in formats:
        meta = {"model": model_name, "device": device, "compute_type": compute,
                "beam_size": args.beam_size, "language_requested": args.language,
                "vad": args.vad, "min_cue": args.min_cue, "gap": args.gap,
                "max_chars": args.max_chars, "max_lines": args.max_lines}
        media = {"path": os.path.abspath(path), "name": os.path.basename(path),
                 "duration": _r(duration, 3), "has_video": has_video}
        made.append(write_json(os.path.join(out_dir, f"{stem}.transcript.json"),
                               media, words, segments, cues, info, meta))
    if "txt" in formats:
        made.append(write_txt(cues, os.path.join(out_dir, f"{stem}.txt"), args.para_gap))
    keep_ass = None
    if "ass" in formats:
        keep_ass = os.path.join(out_dir, f"{stem}.ass")
        made.append(write_ass(cues, keep_ass, args, width, height))
    print(f"  [5/{total}] outputs        : "
          f"{'  '.join(os.path.basename(m) for m in made)}")

    if args.burn:
        if not has_video:
            print(f"  [6/{total}] burn-in        : skipped - "
                  f"{os.path.basename(path)} has no video stream")
        elif not cues:
            print(f"  [6/{total}] burn-in        : skipped - no cues to burn")
        else:
            made.append(burn_in(path, cues,
                                os.path.join(out_dir, f"{stem}_captioned.mp4"),
                                args, width, height, f"[6/{total}]", keep_ass=None))
    return made


def expand_inputs(patterns):
    """Accept literal paths and shell-style globs (Windows cmd doesn't expand)."""
    out = []
    for p in patterns:
        hits = sorted(h for h in glob.glob(p) if os.path.isfile(h))
        if hits:
            out.extend(hits)
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"WARNING: no such file: {p}")
    seen, uniq = set(), []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def build_parser():
    p = argparse.ArgumentParser(
        prog="captions.py",
        description="Offline word-level captioning with faster-whisper: srt / vtt / "
                    "transcript.json / txt, plus optional styled burn-in.",
        epilog="Nothing is uploaded. Add --local-only to forbid network access outright.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("media", nargs="*", help="one or more video/audio files (globs OK)")
    p.add_argument("--selftest", action="store_true",
                   help="print the device/model/compute that would be used, and exit")

    g = p.add_argument_group("model")
    g.add_argument("--model", default="auto",
                   help="'auto' (large-v3 on CUDA, small on CPU), a size name "
                        "(tiny/base/small/medium/large-v3/distil-large-v3), or a path "
                        "to a converted CTranslate2 model folder")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="where to run inference")
    g.add_argument("--compute", default="auto",
                   help="compute type: 'auto' (int8_float16 on CUDA, int8 on CPU), "
                        "or float16 / int8 / int8_float32 / float32")
    g.add_argument("--language", default="auto",
                   help="'auto' to detect, or a code like en / es / fr")
    g.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                   help="transcribe in-language, or translate to English")
    g.add_argument("--beam-size", type=int, default=5, help="decoder beam size")
    g.add_argument("--temperature", type=float, default=None,
                   help="fixed sampling temperature (default: whisper's fallback ladder)")
    g.add_argument("--prompt", default=None,
                   help="initial prompt to bias spelling of names/jargon, e.g. "
                        "\"Jane Doe, Example Organization\"")
    g.add_argument("--no-condition", action="store_true",
                   help="don't condition on previous text (reduces repetition loops)")
    g.add_argument("--cpu-threads", type=int, default=0,
                   help="CPU worker threads (0 = CTranslate2 default)")
    g.add_argument("--download-root", default=None,
                   help="folder to cache/read models from (keeps big models off C:)")
    g.add_argument("--local-only", action="store_true",
                   help="never touch the network; use only already-downloaded models")

    c = p.add_argument_group("cue layout (pause doctrine)")
    c.add_argument("--max-chars", type=int, default=MAX_CHARS_PER_LINE,
                   help="max characters per caption line")
    c.add_argument("--max-lines", type=int, default=MAX_LINES_PER_CUE,
                   help="max lines per caption cue")
    c.add_argument("--max-cue-seconds", type=float, default=MAX_CUE_SECONDS,
                   help="hard ceiling on one cue's screen time")
    c.add_argument("--gap", type=float, default=GAP_SECONDS,
                   help="silence longer than this forces a cue break (the pause is "
                        "kept - the screen simply goes clear)")
    c.add_argument("--min-cue", type=float, default=0.0,
                   help="OPT-IN: pad cues shorter than this into the silence that "
                        "follows (0 = strict; a cue ends when its last word ends)")
    c.add_argument("--para-gap", type=float, default=PARAGRAPH_GAP,
                   help="pause length that starts a new paragraph in the .txt")
    c.add_argument("--vad", action="store_true",
                   help="OPT-IN: enable Silero VAD silence-removal (can shift timings; "
                        "against house pause doctrine)")

    o = p.add_argument_group("output")
    o.add_argument("--out-dir", default=None,
                   help="write here instead of a sibling folder named after the input")
    o.add_argument("--formats", default="srt,vtt,json,txt",
                   help="comma list of outputs to write: srt, vtt, json, txt, ass "
                        "('ass' keeps the styled subtitle file used for burn-in)")
    o.add_argument("--progress", action="store_true",
                   help="print each segment as it is decoded")

    b = p.add_argument_group("burn-in")
    b.add_argument("--burn", action="store_true",
                   help="also render <name>_captioned.mp4 with the captions hardsubbed")
    b.add_argument("--font", default=BURN_FONT, help="font family name for burn-in")
    b.add_argument("--fontsize", type=int, default=BURN_FONTSIZE,
                   help="font size in REAL pixels at any resolution (we author the "
                        "ASS PlayRes to match the frame - see write_ass)")
    b.add_argument("--margin-v", type=int, default=BURN_MARGIN_V,
                   help="bottom margin in pixels (lower third)")
    b.add_argument("--margin-h", type=int, default=BURN_MARGIN_H,
                   help="left/right margin in pixels (keeps text off the edges)")
    b.add_argument("--outline", type=float, default=BURN_OUTLINE,
                   help="black outline thickness in pixels")
    b.add_argument("--text-color", default="FFFFFF", help="caption text colour (hex RGB)")
    b.add_argument("--outline-color", default="000000", help="outline colour (hex RGB)")
    b.add_argument("--bold", action="store_true", help="bold the caption text")
    b.add_argument("--box", action="store_true",
                   help="translucent box behind the text instead of a bare outline")
    b.add_argument("--box-alpha", type=lambda v: int(v, 0), default=0x80,
                   help="box transparency 0=opaque .. 255=invisible (with --box)")
    b.add_argument("--crf", type=int, default=BURN_CRF,
                   help="x264 CRF for the burned copy (lower = better/bigger)")
    b.add_argument("--preset", default="medium", help="x264 speed preset")
    return p


def main():
    args = build_parser().parse_args()

    if args.selftest:
        sys.exit(selftest(args))

    register_cuda_dll_dirs()   # Windows GPU DLL discovery; no-op elsewhere

    if not (have("ffmpeg") and have("ffprobe")):
        print("ERROR: ffmpeg/ffprobe not found on PATH.")
        print("  Windows: grab a full build from https://www.gyan.dev/ffmpeg/builds/ "
              "and add its bin\\ folder to PATH.")
        print("  Linux/macOS: apt install ffmpeg  /  brew install ffmpeg")
        sys.exit(1)

    files = expand_inputs(args.media)
    if not files:
        print("ERROR: no input media. Pass one or more files, or --selftest.")
        sys.exit(2)
    if args.out_dir and len(files) > 1:
        print("ERROR: --out-dir with multiple inputs would overwrite outputs; drop "
              "--out-dir and let each file get its own folder.")
        sys.exit(2)
    if args.max_lines < 1 or args.max_chars < 8:
        print("ERROR: --max-lines must be >=1 and --max-chars >=8.")
        sys.exit(2)

    print(f"captions.py v{__version__}  |  offline faster-whisper  |  {len(files)} file(s)")
    try:
        bundle = load_model(args, "[--]")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    failures = 0
    for path in files:
        try:
            made = caption_one(path, args, bundle)
            if made:
                print(f"  done           : {os.path.dirname(os.path.abspath(made[0]))}")
        except (RuntimeError, OSError, ValueError) as e:
            failures += 1
            print(f"  ERROR on {os.path.basename(path)}: {e}")

    print(f"\nFinished with {failures} failure(s) of {len(files)}." if failures
          else f"\nAll {len(files)} file(s) captioned.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
