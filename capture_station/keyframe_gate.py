#!/usr/bin/env python3
"""
keyframe_gate.py - lightweight motion-gated video keyframe extractor.

THE JOB: "cheap eyes, expensive brain." Watch a video's region of interest with
sub-1% of the compute of full processing; emit only the frames that matter.

Implements a hands-in/hands-out technique (from a kitchen-camera
calibration note): motion entering the ROI = HANDS_IN; sustained stillness =
HANDS_OUT, and THAT frame is the canonical "state of the plate." A meal becomes
a dozen clean plate-states instead of minutes of video. The same keyframes
assembled at 8-12 fps ARE the stop-motion clip - one capture, media + data.

Maps to Methods_and_Materials v2.0: plate-state keyframes automate the
"plated-stage photograph" of the paired capture protocol.

  TOOLBOX RULE 2 RECORD - searched 2026-08-04 (terms: keyframe; stop motion;
  time lapse; scene cut/detect/change; motion detect/gate/trigger; frame diff)
  against SCRIPT_REGISTRY_REBUILD_2026-07-31_v02 + additions: ZERO hits.
  Nearest: capture_squeeze.py samples frames at FIXED INTERVAL - different
  job (exhaustive extraction), shared doctrine (EXIF canonical, resumable).
  identify_batch.py / split_batch.py operate on STILLS. Nothing to extend -> new
  tool, same pipeline folder, registered same-day per Rule 5.

DOCTRINE: input video is READ-ONLY. Nothing written without --out. Health-
adjacent video stays local - no cloud processing. Fixture-tested before use.

Outputs (under --out):
  keyframes/state_NNN.jpg      canonical plate-states (HANDS_OUT moments)
  manifest.json                every state change w/ timestamp + frame index
  stopmotion.mp4               optional (--stopmotion, needs ffmpeg)
  motion_spans.json            active spans, for pass-2 gating (OCR/pose/VAD)

Usage:
  python keyframe_gate.py video.mp4 --probe                # report only
  python keyframe_gate.py video.mp4 --out C:\\scratch\\m1  # extract
  python keyframe_gate.py video.mp4 --out d --roi 0.2,0.2,0.8,0.9 --stopmotion
"""
import argparse, json, os, shutil, subprocess, sys
from datetime import datetime

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("ERROR: opencv-python + numpy required.")

# ---- tunables (calibrated on synthetic fixture 2026-08-04) ------------------
DIFF_THRESH = 12        # per-pixel gray delta that counts as change
MOTION_FRAC = 0.010     # fraction of ROI pixels changed => motion present
STILL_SECONDS = 1.5     # sustained stillness to fire HANDS_OUT
MIN_STATE_GAP = 2.0     # min seconds between emitted plate-states
SAMPLE_FPS = 6.0        # analysis rate; full video fps only for keyframe grab


def parse_roi(s):
    try:
        x0, y0, x1, y1 = (float(v) for v in s.split(","))
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        return x0, y0, x1, y1
    except Exception:
        sys.exit("ERROR: --roi must be x0,y0,x1,y1 as fractions, e.g. 0.2,0.2,0.8,0.9")


def run(video, out, roi, stopmotion, probe):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / SAMPLE_FPS))
    x0, y0, x1, y1 = (int(roi[0]*W), int(roi[1]*H), int(roi[2]*W), int(roi[3]*H))

    state, still_run, last_emit_t = "HANDS_OUT", 0.0, -1e9
    prev = None
    events, spans, keyframes = [], [], []
    span_start = None
    fi = -1

    # Walk the video at SAMPLE_FPS: diff each sampled ROI frame against the previous one,
    # driving a HANDS_IN / HANDS_OUT state machine that emits a keyframe when motion settles.
    while True:
        ok = cap.grab()
        if not ok:
            break
        fi += 1
        if fi % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = fi / fps
        g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (7, 7), 0)
        if prev is None:
            prev = g
            continue
        frac = float((cv2.absdiff(g, prev) > DIFF_THRESH).mean())
        prev = g
        moving = frac >= MOTION_FRAC

        # Motion opens a HANDS_IN span; sustained stillness closes it (HANDS_OUT) and
        # emits the settled frame as a canonical plate-state, throttled by MIN_STATE_GAP.
        if moving:
            still_run = 0.0
            if state == "HANDS_OUT":
                state = "HANDS_IN"; span_start = t
                events.append({"t": round(t, 2), "frame": fi, "event": "HANDS_IN"})
        else:
            still_run += step / fps
            if state == "HANDS_IN" and still_run >= STILL_SECONDS:
                state = "HANDS_OUT"
                spans.append({"start": round(span_start, 2), "end": round(t, 2)})
                events.append({"t": round(t, 2), "frame": fi, "event": "HANDS_OUT"})
                if t - last_emit_t >= MIN_STATE_GAP:
                    keyframes.append({"t": round(t, 2), "frame": fi, "img": frame.copy()})
                    last_emit_t = t
    cap.release()
    if state == "HANDS_IN" and span_start is not None:
        spans.append({"start": round(span_start, 2), "end": round(fi / fps, 2)})

    dur = n_total / fps if fps else 0
    active = sum(s["end"] - s["start"] for s in spans)
    print(f"video          : {video}")
    print(f"duration       : {dur:.1f}s @ {fps:.1f} fps | analysed every {step} frames")
    print(f"activity spans : {len(spans)} | active {active:.1f}s ({(active/dur*100 if dur else 0):.0f}%)")
    print(f"plate-states   : {len(keyframes)}")
    print(f"pass-2 saving  : full processing needed on {(active/dur*100 if dur else 0):.0f}% of timeline, not 100%")

    if probe or not out:
        print("\nPROBE ONLY - nothing written." if probe else "\nNo --out given - nothing written.")
        return

    # Persist results: keyframe JPEGs, a manifest of events + keyframes, and the motion spans.
    os.makedirs(os.path.join(out, "keyframes"), exist_ok=True)
    for i, k in enumerate(keyframes, 1):
        cv2.imwrite(os.path.join(out, "keyframes", f"state_{i:03d}.jpg"), k["img"],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"video": os.path.abspath(video), "run": datetime.now().isoformat(timespec="seconds"),
                   "roi": roi, "fps": fps, "events": events,
                   "keyframes": [{"t": k["t"], "frame": k["frame"],
                                  "file": f"keyframes/state_{i:03d}.jpg"}
                                 for i, k in enumerate(keyframes, 1)]}, fh, indent=2)
    with open(os.path.join(out, "motion_spans.json"), "w", encoding="utf-8") as fh:
        json.dump({"video": os.path.abspath(video), "spans": spans}, fh, indent=2)
    print(f"written        : {out}  (keyframes/ + manifest.json + motion_spans.json)")

    # Optional: assemble the keyframes into a stop-motion clip with ffmpeg when available.
    if stopmotion and keyframes:
        if not shutil.which("ffmpeg"):
            print("stopmotion     : SKIPPED - ffmpeg not on PATH (keyframes are on disk; assemble later)")
            return
        cmd = ["ffmpeg", "-y", "-framerate", "10",
               "-i", os.path.join(out, "keyframes", "state_%03d.jpg"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(out, "stopmotion.mp4")]
        r = subprocess.run(cmd, capture_output=True)
        print("stopmotion     :", "written" if r.returncode == 0 else f"FAILED rc={r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="output dir; without it nothing is written")
    ap.add_argument("--roi", default="0.15,0.15,0.85,0.90",
                    help="plate zone as fractions x0,y0,x1,y1")
    ap.add_argument("--stopmotion", action="store_true")
    ap.add_argument("--probe", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    run(a.video, a.out, parse_roi(a.roi), a.stopmotion, a.probe)


if __name__ == "__main__":
    main()
