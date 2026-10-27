#!/usr/bin/env python3
"""
bed_prep.py - turn a Demucs-separated track into a voiceover-ready music bed.

Takes one of the separated_stems/htdemucs/<TRACK> folders, sums the instrumental
stems (bass + drums + other, DROPPING vocals per doctrine), fits it to a target
video runtime, carves EQ room for speech, loudness-normalizes it to a bed level,
and either ducks it under a supplied voiceover or leaves it as a low static bed
ready to drop on a timeline that already has VO.

The result is a bed that sits UNDER spoken narration and never fights the voice.

Pipeline (each stage is its own function):
    1. mix_instrumental   - sum bass+drums+other -> stereo instrumental (no vocals)
    2. fit_duration       - trim, or loop with an equal-power crossfade, to --duration
    3. eq_carve           - high-pass ~80 Hz + a gentle 2-4 kHz dip for intelligibility
    4. loudness_normalize - two-pass EBU R128 loudnorm to --lufs (default -23)
    5. duck / static      - sidechain-duck against --vo, else a fixed low static bed
    6. fades              - --fade-in / --fade-out
    7. export             - beds/<track>_bed_<duration>s.wav (48 kHz stereo PCM), +.m4a if --aac

Dependencies: numpy + ffmpeg & ffprobe on PATH. No librosa / pydub / heavy deps,
matching the house style of highlight_clip.py.

Usage:
    python bed_prep.py --track "Edge" --duration 135
    python bed_prep.py --track "Edge" --duration 135 --vo takes/voiceover.wav --premix
    python bed_prep.py --list
See --help for every flag.
"""

import os
import sys
import re
import json
import math
import argparse
import tempfile
import subprocess
import shutil

import numpy as np  # kept as a house-style dependency / used for tolerance math

# --- defaults ---------------------------------------------------------------
STEMS_ROOT = os.path.join("separated_stems", "htdemucs")
OUTPUT_DIR = "beds"
INSTRUMENTAL_STEMS = ("bass", "drums", "other")   # vocals.wav is intentionally dropped

OUT_SR = 48000            # export sample rate
XFADE_SECONDS = 2.0       # equal-power crossfade length at each loop seam
HPF_HZ = 80               # high-pass to kill rumble
DIP_HZ = 3000             # center of the broad presence dip (2-4 kHz)
DIP_Q = 0.7               # broad bell (low Q = wide)
DIP_GAIN_DB = -3.0        # how much to scoop for speech room


def have(cmd):
    return shutil.which(cmd) is not None


def run(cmd):
    """Run an ffmpeg/ffprobe command, returning the CompletedProcess (stderr captured)."""
    return subprocess.run(cmd, capture_output=True, text=True)


def ffprobe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", path])
    try:
        return float(r.stdout.strip())
    except (ValueError, AttributeError):
        return None


def probe_stream(path):
    """Return (sample_rate, channels) of the first audio stream, or (None, None)."""
    r = run(["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "default=nw=1:nk=1", path])
    vals = r.stdout.split()
    try:
        return int(vals[0]), int(vals[1])
    except (IndexError, ValueError):
        return None, None


# --- stage 1: instrumental --------------------------------------------------
def mix_instrumental(track_dir, out_wav):
    """Sum bass+drums+other into one stereo instrumental (vocals dropped).

    Intermediate is 32-bit float so summing three stems can't clip before we
    normalize; loudnorm/true-peak limiting later brings it back into range.
    """
    inputs = []
    for stem in INSTRUMENTAL_STEMS:
        p = os.path.join(track_dir, f"{stem}.wav")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing stem: {p}")
        inputs.append(p)
    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    # amix with normalize=0 = a true sum (we keep headroom via float output).
    cmd += ["-filter_complex", f"amix=inputs={len(inputs)}:normalize=0[m]",
            "-map", "[m]", "-c:a", "pcm_f32le", out_wav]
    r = run(cmd)
    if not os.path.isfile(out_wav):
        raise RuntimeError(f"instrumental mix failed:\n{r.stderr[-800:]}")
    print(f"  [1/7] instrumental  : summed {'+'.join(INSTRUMENTAL_STEMS)} "
          f"(vocals dropped)")
    return out_wav


# --- stage 2: duration fit --------------------------------------------------
def fit_duration(in_wav, target, out_wav, xfade=XFADE_SECONDS):
    """Trim (if long enough) or loop with equal-power crossfades to `target` sec."""
    src = ffprobe_duration(in_wav) or 0.0
    if src >= target:
        run(["ffmpeg", "-y", "-i", in_wav, "-t", f"{target:.3f}",
             "-c:a", "pcm_f32le", out_wav])
        print(f"  [2/7] duration fit  : trimmed {src:.1f}s -> {target:.1f}s")
        return out_wav

    # Loop: chain equal-power (qsin) crossfades over N copies until we exceed
    # target, then trim to exact length. total = N*src - (N-1)*xfade.
    d = min(xfade, max(0.1, src * 0.4))
    n = max(2, math.ceil((target - d) / max(0.001, (src - d))))
    n = min(n, 64)  # sanity cap

    cmd = ["ffmpeg", "-y"]
    for _ in range(n):
        cmd += ["-i", in_wav]
    parts, prev = [], "[0:a]"
    for i in range(1, n):
        lbl = f"[x{i}]"
        parts.append(f"{prev}[{i}:a]acrossfade=d={d:.3f}:c1=qsin:c2=qsin{lbl}")
        prev = lbl
    fc = ";".join(parts) + f";{prev}atrim=0:{target:.3f},asetpts=N/SR/TB[o]"
    cmd += ["-filter_complex", fc, "-map", "[o]", "-c:a", "pcm_f32le", out_wav]
    r = run(cmd)
    if not os.path.isfile(out_wav):
        raise RuntimeError(f"loop/fit failed:\n{r.stderr[-800:]}")
    print(f"  [2/7] duration fit  : looped {src:.1f}s x{n} "
          f"(equal-power {d:.1f}s crossfades) -> {target:.1f}s")
    return out_wav


# --- stage 3: EQ carve ------------------------------------------------------
def eq_carve(in_wav, out_wav):
    """High-pass ~80 Hz + a gentle broad dip around 2-4 kHz for speech room."""
    af = (f"highpass=f={HPF_HZ},"
          f"equalizer=f={DIP_HZ}:width_type=q:w={DIP_Q}:g={DIP_GAIN_DB}")
    run(["ffmpeg", "-y", "-i", in_wav, "-af", af, "-c:a", "pcm_f32le", out_wav])
    if not os.path.isfile(out_wav):
        raise RuntimeError("EQ carve failed")
    print(f"  [3/7] EQ carve      : HPF {HPF_HZ} Hz + {DIP_GAIN_DB:+.0f} dB "
          f"dip @ {DIP_HZ/1000:.0f} kHz")
    return out_wav


# --- stage 4: loudness normalize -------------------------------------------
def _loudnorm_measure(in_wav, target_i, tp, lra):
    """First loudnorm pass: measure integrated loudness / true-peak (JSON)."""
    r = run(["ffmpeg", "-y", "-i", in_wav,
             "-af", f"loudnorm=I={target_i}:TP={tp}:LRA={lra}:print_format=json",
             "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def loudness_normalize(in_wav, target_i, out_wav, tp=-1.5, lra=11):
    """Two-pass EBU R128 loudnorm to `target_i` LUFS. Outputs 48 kHz float."""
    stats = _loudnorm_measure(in_wav, target_i, tp, lra)
    af = f"loudnorm=I={target_i}:TP={tp}:LRA={lra}"
    if stats:
        af += (f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
               f":measured_LRA={stats['input_lra']}"
               f":measured_thresh={stats['input_thresh']}"
               f":offset={stats['target_offset']}:linear=true")
        mode = "two-pass"
    else:
        mode = "single-pass (measure failed)"
    run(["ffmpeg", "-y", "-i", in_wav, "-af", af,
         "-ar", str(OUT_SR), "-ac", "2", "-c:a", "pcm_f32le", out_wav])
    if not os.path.isfile(out_wav):
        raise RuntimeError("loudnorm failed")
    print(f"  [4/7] loudness      : {mode} loudnorm -> {target_i:.0f} LUFS "
          f"(TP {tp} dBTP)")
    return out_wav


# --- stage 5a: static bed ---------------------------------------------------
def static_bed(in_wav, gain_db, out_wav):
    """No VO: apply a fixed low gain trim so the bed sits under a timeline's voice."""
    run(["ffmpeg", "-y", "-i", in_wav, "-af", f"volume={gain_db}dB",
         "-c:a", "pcm_f32le", out_wav])
    if not os.path.isfile(out_wav):
        raise RuntimeError("static gain failed")
    tag = "no extra trim" if abs(gain_db) < 1e-6 else f"{gain_db:+.1f} dB trim"
    print(f"  [5/7] static bed    : {tag}")
    return out_wav


# --- stage 5b: sidechain duck ----------------------------------------------
def duck_bed(bed_wav, vo_path, out_wav, ratio, thresh, attack, release):
    """VO present: sidechain-compress the bed so music dips when the voice speaks."""
    fc = (f"[0:a]aformat=sample_rates={OUT_SR}:channel_layouts=stereo[bed];"
          f"[1:a]aformat=sample_rates={OUT_SR}:channel_layouts=stereo[vo];"
          f"[bed][vo]sidechaincompress=threshold={thresh}:ratio={ratio}"
          f":attack={attack}:release={release}[o]")
    run(["ffmpeg", "-y", "-i", bed_wav, "-i", vo_path,
         "-filter_complex", fc, "-map", "[o]", "-c:a", "pcm_f32le", out_wav])
    if not os.path.isfile(out_wav):
        raise RuntimeError("sidechain duck failed")
    print(f"  [5/7] duck          : sidechain ratio {ratio}:1 thresh {thresh} "
          f"atk {attack}ms rel {release}ms")
    return out_wav


def premix(bed_wav, vo_path, out_wav):
    """Sum ducked bed + VO into a single premix (VO on top, bed underneath)."""
    fc = (f"[0:a]aformat=sample_rates={OUT_SR}:channel_layouts=stereo[bed];"
          f"[1:a]aformat=sample_rates={OUT_SR}:channel_layouts=stereo[vo];"
          f"[bed][vo]amix=inputs=2:normalize=0:dropout_transition=0[o]")
    run(["ffmpeg", "-y", "-i", bed_wav, "-i", vo_path,
         "-filter_complex", fc, "-map", "[o]", "-c:a", "pcm_f32le", out_wav])
    if not os.path.isfile(out_wav):
        raise RuntimeError("premix failed")
    print(f"  [5/7] premix        : ducked bed + VO summed")
    return out_wav


# --- stage 6: fades ---------------------------------------------------------
def apply_fades(in_wav, fade_in, fade_out, out_wav):
    dur = ffprobe_duration(in_wav) or 0.0
    filts = []
    if fade_in > 0:
        filts.append(f"afade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0 and dur > fade_out:
        filts.append(f"afade=t=out:st={dur - fade_out:.3f}:d={fade_out:.3f}")
    if not filts:
        shutil.copyfile(in_wav, out_wav)
    else:
        run(["ffmpeg", "-y", "-i", in_wav, "-af", ",".join(filts),
             "-c:a", "pcm_f32le", out_wav])
    if not os.path.isfile(out_wav):
        raise RuntimeError("fades failed")
    print(f"  [6/7] fades         : in {fade_in:.1f}s / out {fade_out:.1f}s")
    return out_wav


# --- stage 7: export --------------------------------------------------------
def export(in_wav, out_wav, want_aac):
    """Write 48 kHz stereo PCM (s16) WAV, and optionally an AAC .m4a alongside."""
    run(["ffmpeg", "-y", "-i", in_wav, "-ar", str(OUT_SR), "-ac", "2",
         "-c:a", "pcm_s16le", out_wav])
    if not os.path.isfile(out_wav):
        raise RuntimeError("WAV export failed")
    made = [out_wav]
    if want_aac:
        m4a = os.path.splitext(out_wav)[0] + ".m4a"
        run(["ffmpeg", "-y", "-i", in_wav, "-ar", str(OUT_SR), "-ac", "2",
             "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", m4a])
        if os.path.isfile(m4a):
            made.append(m4a)
    print(f"  [7/7] export        : {'  '.join(os.path.basename(m) for m in made)}")
    return made


def measure_lufs(path, target_i=-23):
    """Report-only: measure integrated LUFS of a finished file via loudnorm pass."""
    stats = _loudnorm_measure(path, target_i, -1.5, 11)
    try:
        return float(stats["input_i"]) if stats else None
    except (KeyError, ValueError, TypeError):
        return None


# --- driver -----------------------------------------------------------------
def resolve_track_dir(track, stems_root):
    """Accept a track name, an exact folder, or a full path."""
    if os.path.isdir(track) and any(
            os.path.isfile(os.path.join(track, f"{s}.wav")) for s in INSTRUMENTAL_STEMS):
        return track
    cand = os.path.join(stems_root, track)
    if os.path.isdir(cand):
        return cand
    return None


def list_tracks(stems_root):
    if not os.path.isdir(stems_root):
        print(f"No stems folder at {os.path.abspath(stems_root)}")
        return
    names = sorted(d for d in os.listdir(stems_root)
                   if os.path.isdir(os.path.join(stems_root, d)))
    print(f"{len(names)} track folder(s) in {os.path.abspath(stems_root)}:")
    for n in names:
        print(f"  - {n}")


def safe_name(track):
    base = os.path.basename(os.path.normpath(track))
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "track"


def build_bed(args):
    track_dir = resolve_track_dir(args.track, args.stems_root)
    if not track_dir:
        print(f"ERROR: could not find track '{args.track}' under "
              f"{os.path.abspath(args.stems_root)} (try --list).")
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    dur = int(round(args.duration))
    out_wav = os.path.join(args.out_dir, f"{safe_name(track_dir)}_bed_{dur}s.wav")

    print(f"Building bed from: {track_dir}")
    print(f"Target duration  : {args.duration:.1f}s   |   mode: "
          f"{'DUCK vs VO' if args.vo else 'STATIC'}"
          f"{'  (+premix)' if (args.vo and args.premix) else ''}")

    tmp = tempfile.mkdtemp(prefix="bed_prep_")
    try:
        t = lambda n: os.path.join(tmp, n)
        mix_instrumental(track_dir, t("instr.wav"))
        fit_duration(t("instr.wav"), args.duration, t("fit.wav"), args.xfade)
        eq_carve(t("fit.wav"), t("eq.wav"))
        loudness_normalize(t("eq.wav"), args.lufs, t("norm.wav"),
                           tp=args.true_peak, lra=args.lra)

        if args.vo:
            if not os.path.isfile(args.vo):
                print(f"ERROR: --vo file not found: {args.vo}")
                return 1
            duck_bed(t("norm.wav"), args.vo, t("stage5.wav"),
                     args.duck_ratio, args.duck_thresh,
                     args.duck_attack, args.duck_release)
        else:
            static_bed(t("norm.wav"), args.gain, t("stage5.wav"))

        apply_fades(t("stage5.wav"), args.fade_in, args.fade_out, t("faded.wav"))

        if args.vo and args.premix:
            premix(t("faded.wav"), args.vo, t("final.wav"))
            final_src = t("final.wav")
        else:
            final_src = t("faded.wav")

        made = export(final_src, out_wav, args.aac)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    sr, ch = probe_stream(out_wav)
    outdur = ffprobe_duration(out_wav)
    lufs = measure_lufs(out_wav)
    print("\nDone.")
    print(f"  file     : {os.path.abspath(out_wav)}")
    print(f"  format   : {sr} Hz / {ch} ch / "
          f"{outdur:.2f}s" if outdur else f"  format   : {sr} Hz / {ch} ch")
    if lufs is not None:
        print(f"  loudness : {lufs:.1f} LUFS integrated")
    for m in made[1:]:
        print(f"  also     : {os.path.abspath(m)}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="bed_prep.py",
        description="Turn Demucs-separated stems into a voiceover-ready music bed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--track", help="track name under the stems root, or a stem folder path")
    p.add_argument("--duration", type=float, default=135.0,
                   help="target bed length in seconds (trim or loop to fit)")
    p.add_argument("--stems-root", default=STEMS_ROOT,
                   help="root holding <TRACK>/{bass,drums,other,vocals}.wav")
    p.add_argument("--out-dir", default=OUTPUT_DIR, help="output folder for rendered beds")
    p.add_argument("--list", action="store_true", help="list available track folders and exit")

    p.add_argument("--lufs", type=float, default=-23.0,
                   help="integrated loudness target for the bed (EBU R128)")
    p.add_argument("--true-peak", type=float, default=-1.5, help="loudnorm true-peak ceiling (dBTP)")
    p.add_argument("--lra", type=float, default=11.0, help="loudnorm target loudness range")
    p.add_argument("--xfade", type=float, default=XFADE_SECONDS,
                   help="equal-power crossfade length at loop seams (seconds)")

    p.add_argument("--vo", help="voiceover WAV: enables sidechain ducking against the voice")
    p.add_argument("--premix", action="store_true",
                   help="with --vo, also sum ducked bed + VO into one premix output")
    p.add_argument("--gain", type=float, default=-12.0,
                   help="STATIC mode only: extra level trim in dB applied after loudnorm")
    p.add_argument("--duck-ratio", type=float, default=8.0, help="sidechain compression ratio")
    p.add_argument("--duck-thresh", type=float, default=0.05,
                   help="sidechain threshold (linear 0-1); lower = ducks more easily")
    p.add_argument("--duck-attack", type=float, default=20.0, help="sidechain attack (ms)")
    p.add_argument("--duck-release", type=float, default=300.0, help="sidechain release (ms)")

    p.add_argument("--fade-in", type=float, default=1.0, help="fade-in length (seconds)")
    p.add_argument("--fade-out", type=float, default=2.0, help="fade-out length (seconds)")
    p.add_argument("--aac", action="store_true", help="also write a 256k AAC .m4a alongside the WAV")
    return p


def main():
    args = build_parser().parse_args()
    if not (have("ffmpeg") and have("ffprobe")):
        print("ERROR: ffmpeg/ffprobe not found on PATH.")
        print("Install ffmpeg (https://www.gyan.dev/ffmpeg/builds/ on Windows) and retry.")
        sys.exit(1)
    if args.list:
        list_tracks(args.stems_root)
        return
    if not args.track:
        print("ERROR: --track is required (or use --list to see options).")
        sys.exit(2)
    sys.exit(build_bed(args))


if __name__ == "__main__":
    main()
