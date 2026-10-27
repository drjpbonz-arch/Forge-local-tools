#!/usr/bin/env python3
"""
clean_and_split.py - batch AI stem separation for a drop folder of tracks.

Walks INPUT_DIR (./raw_jam_tracks) and runs Demucs (htdemucs) on every audio
or video file, writing the separated stems (bass / drums / other / vocals) to
OUTPUT_DIR (./separated_stems). Hardware is auto-detected: a CUDA GPU when one
is available, otherwise the CPU, so the run always completes. Any videos in the
drop folder additionally get a short highlight clip via highlight_clip.py.

Each track is processed in isolation: one file's failure falls back to a direct
shell invocation and, failing that, is reported and skipped without stopping the
rest of the batch.

Dependencies: torch + demucs, and (for the optional highlight step)
highlight_clip.py with ffmpeg/ffprobe on PATH.

Usage:
    python clean_and_split.py        # scans ./raw_jam_tracks -> ./separated_stems
"""

import os
import subprocess
import torch

# --- 1. DIRECTORY CONFIGURATION ---
INPUT_DIR = "./raw_jam_tracks"
OUTPUT_DIR = "./separated_stems"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print(f"🚀 INITIALIZING PRODUCTION-GRADE AI STEM SEPARATOR")
print(f"Looking for tracks in: {INPUT_DIR}")
print(f"Pristine AI stems will be saved to: {OUTPUT_DIR}")
print("=" * 50)

# --- 2. AUTOMATED HARDWARE DIAGNOSTIC ---
# Automatically probe the hardware to see if the NVIDIA CUDA bridge is online
if torch.cuda.is_available():
    hardware_device = "cuda"
    print("🚀 EXCELLENT: NVIDIA GPU Acceleration Detected. Processing will be lightning fast.")
else:
    hardware_device = "cpu"
    print("🐌 NOTICE: CUDA bridge offline. Falling back to CPU processing core to guarantee execution.")

print("=" * 50 + "\n")

# --- 3. AUTOMATION LOOP ---
track_count = 0

for file_name in os.listdir(INPUT_DIR):
    if file_name.lower().endswith(('.wav', '.mp3', '.flac', '.mp4')):
        track_count += 1
        full_input_path = os.path.join(INPUT_DIR, file_name)
        
        print("-" * 50)
        print(f"🧠 AI Demixing Track #{track_count}: {file_name}")
        print(f"Routing processing workload to: {hardware_device.upper()}")
        print("-" * 50)
        
        try:
            # Build the explicit execution array using the dynamically chosen hardware device
            command = [
                "demucs",
                "-d", hardware_device,
                "--out", OUTPUT_DIR,
                full_input_path
            ]
            
            # Execute the deep learning network process
            result = subprocess.run(command, check=True)
            
            if result.returncode == 0:
                print(f"✅ Neural network disassembly complete for: {file_name}\n")
            
        except Exception as error_message:
            # Fallback path 2: direct system call if subprocess pipeline environment gets restricted
            print("Redirecting execution pathway through local shell node...")
            fallback_command = f"python -m demucs -d {hardware_device} --out \"{OUTPUT_DIR}\" \"{full_input_path}\""
            exit_status = os.system(fallback_command)
            
            if exit_status == 0:
                print(f"✅ Finished disassembling via fallback node: {file_name}\n")
            else:
                print(f"❌ Both processing nodes failed. Error details: {error_message}\n")

# --- 4. HIGHLIGHT CLIPS (video files only) ---
# For any video that was processed, also cut a 6-10s highlight of the most
# musically dynamic moment. Fully optional: wrapped so it never affects stems.
try:
    import highlight_clip
    if highlight_clip.have("ffmpeg") and highlight_clip.have("ffprobe"):
        vids = [f for f in os.listdir(INPUT_DIR)
                if f.lower().endswith(highlight_clip.VIDEO_EXTS)]
        if vids:
            print("=" * 50)
            print(f"🎬 Generating highlight clips for {len(vids)} video file(s)...")
            for f in vids:
                try:
                    highlight_clip.make_highlight(os.path.join(INPUT_DIR, f),
                                                  "./highlight_clips")
                except Exception as e:
                    print(f"  ! highlight failed for {f}: {e}")
    else:
        print("(ffmpeg/ffprobe not on PATH - skipping highlight clips.)")
except Exception as e:
    print(f"(highlight feature unavailable: {e})")

print("=" * 50)
print(f"🏁 AI PIPELINE COMPLETE. Processed {track_count} tracks.")
print("=" * 50)