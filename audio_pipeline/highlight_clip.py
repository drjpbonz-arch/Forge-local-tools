#!/usr/bin/env python3
"""
highlight_clip.py - extract the most musically dynamic 6-10s clip from video files.

For each video found in INPUT_DIR, this analyzes the audio track, finds the
window with the highest "musical dynamism" (loudness energy + spectral flux,
i.e. how much is happening + how fast the sound is changing), and cuts that
span out as a short highlight clip into OUTPUT_DIR - ready-made highlight-reel
material.

Pairs with clean_and_split.py: drop a video in raw_jam_tracks and you get both
the separated stems AND a highlight clip.

Dependencies: numpy (already in the venv via demucs) + ffmpeg & ffprobe on PATH.
No librosa or extra ML deps required.

Usage:
    python highlight_clip.py                 # scans ./raw_jam_tracks -> ./highlight_clips
    python highlight_clip.py <in_dir> <out_dir>
"""

import os
import sys
import wave
import tempfile
import subprocess
import shutil
import numpy as np

INPUT_DIR = "./raw_jam_tracks"        # same drop folder as the stem splitter
OUTPUT_DIR = "./highlight_clips"
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")

CLIP_SECONDS = 8.0     # target clip length, clamped to [MIN_SECONDS, MAX_SECONDS]
MIN_SECONDS = 6.0
MAX_SECONDS = 10.0

ANALYZE_SR = 22050     # downsample audio for analysis (plenty for an energy/flux curve)
FRAME = 2048
HOP = 512
SMOOTH = 9             # moving-average window (frames) to avoid twitchy picks


def have(cmd):
    return shutil.which(cmd) is not None


def ffprobe_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", path],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except (ValueError, AttributeError):
        return None


def extract_audio(path, sr):
    """Decode the video's audio to a temp mono WAV at the analysis sample rate."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "wav", tmp.name],
        capture_output=True)
    return tmp.name


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    scale = {np.int8: 128.0, np.int16: 32768.0, np.int32: 2147483648.0}.get(dtype, 32768.0)
    return x / scale, sr


def dynamism_envelope(x, sr):
    """Per-frame score = 0.6*loudness(RMS) + 0.4*spectral-flux, normalized & smoothed."""
    if len(x) < FRAME:
        return np.array([0.0]), HOP / sr
    n_frames = 1 + (len(x) - FRAME) // HOP
    win = np.hanning(FRAME).astype(np.float32)
    rms = np.empty(n_frames, dtype=np.float32)
    flux = np.empty(n_frames, dtype=np.float32)
    prev_mag = None
    for i in range(n_frames):
        fr = x[i * HOP: i * HOP + FRAME] * win
        rms[i] = np.sqrt(np.mean(fr * fr) + 1e-12)
        mag = np.abs(np.fft.rfft(fr))
        if prev_mag is None:
            flux[i] = 0.0
        else:
            d = mag - prev_mag
            flux[i] = np.sum(d[d > 0])      # only rising energy = onsets / attacks
        prev_mag = mag

    def norm(a):
        a = a - a.min()
        m = a.max()
        return a / m if m > 0 else a

    score = 0.6 * norm(rms) + 0.4 * norm(flux)
    if SMOOTH > 1:
        ker = np.ones(SMOOTH, dtype=np.float32) / SMOOTH
        score = np.convolve(score, ker, mode="same")
    return score, HOP / sr


def best_window(score, frame_time, clip_s, dur):
    """Find the start time of the highest-scoring window of length clip_s."""
    clip_s = max(MIN_SECONDS, min(MAX_SECONDS, clip_s))
    if dur:
        clip_s = min(clip_s, dur)
    w = max(1, int(round(clip_s / frame_time)))
    if w >= len(score):
        return 0.0, clip_s
    csum = np.concatenate(([0.0], np.cumsum(score)))
    sums = csum[w:] - csum[:-w]              # window sums via prefix sums
    best = int(np.argmax(sums))
    start = best * frame_time
    if dur and start + clip_s > dur:
        start = max(0.0, dur - clip_s)
    return start, clip_s


def cut_clip(path, start, length, out_path):
    """Frame-accurate cut + re-encode (so the highlight starts exactly on the beat)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path],
        capture_output=True)


def make_highlight(path, out_dir=OUTPUT_DIR, clip_s=CLIP_SECONDS):
    os.makedirs(out_dir, exist_ok=True)
    wav = extract_audio(path, ANALYZE_SR)
    try:
        x, sr = read_wav(wav)
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass
    if x.size == 0:
        print(f"  ! no audio track in {os.path.basename(path)}; skipping")
        return None
    dur = ffprobe_duration(path) or (len(x) / sr)
    score, ft = dynamism_envelope(x, sr)
    start, length = best_window(score, ft, clip_s, dur)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{base}_highlight.mp4")
    cut_clip(path, start, length, out_path)
    print(f"  -> {os.path.basename(out_path)}  "
          f"[{start:.1f}s-{start + length:.1f}s of {dur:.1f}s total]")
    return out_path


def main():
    if not (have("ffmpeg") and have("ffprobe")):
        print("ERROR: ffmpeg/ffprobe not found on PATH.")
        print("Install ffmpeg (https://www.gyan.dev/ffmpeg/builds/ on Windows) and retry.")
        sys.exit(1)
    in_dir = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    os.makedirs(in_dir, exist_ok=True)
    vids = [f for f in os.listdir(in_dir) if f.lower().endswith(VIDEO_EXTS)]
    if not vids:
        print(f"No video files in {os.path.abspath(in_dir)} "
              f"(looking for {', '.join(VIDEO_EXTS)})")
        return
    for f in vids:
        print(f"Analyzing: {f}")
        make_highlight(os.path.join(in_dir, f), out_dir)
    print(f"\nDone. Highlight clips in: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
