# Forge local tools

Nineteen single-file Python tools for local, private media and data work: capture-station helpers, an audio pipeline, media-organisation utilities and a blinding harness. Each file is self-contained and documented in its own header: what it reads, what it writes, what it never touches. Nothing here calls a cloud service; the tools run on your machine against your folders.

Python 3.10 or newer. Install a tool's third-party packages with pip as listed below; several also expect `ffmpeg`/`ffprobe` on the PATH.

## The tools

| folder | script | what it does | other uses | needs |
|---|---|---|---|---|
| capture_station | `keyframe_gate.py` | Motion-gated keyframe extraction — ROI frame-differencing, emit only stable-state frames | Sparse 'state changed' video sampling — security video, timelapse, process monitoring | numpy, opencv-python |
| capture_station | `crosscheck_hash_inventories.py` | Joins N hash-inventory CSVs to find cross-location duplicate content | Multi-drive/backup dedup audit; 'N inventories, one truth' joiner | standard library |
| capture_station | `match_inventory_to_tree.py` | Matches a hash-inventory CSV against live folder trees (size-first, then MD5), per-file timeout | 'Is this already backed up' checker for any large tree vs a manifest | standard library |
| capture_station | `prep_listing_photos.py` | Downscales/renames session photos into per-item folders for marketplace upload | Prep a folder of photos for e-commerce upload | pillow, pillow_heif |
| capture_station | `qwen_bench.py` | Benchmarks a local Ollama vision model vs ground-truth OCR — strict scoring, anti-hallucination prompting | Local-VLM OCR-accuracy benchmark harness | pillow, pillow_heif |
| capture_station | `pdf_ocr_to_docx.py` | Converts no-text-layer scanned PDFs to .docx via local VLM, keeps page images alongside text | OCR a scanned PDF with a local vision model | pymupdf, python-docx |
| capture_station | `stamp_colophon.py` | Stamps CRediT-taxonomy authorship credit into .docx, marks human/AI-collaborative work | AI-collaboration authorship-disclosure stamping | python-docx |
| capture_station | `make_aruco_placards.py` | Generates a print-ready, calibration-marked ArUco placard sheet (self-verifying print scale). FLAG: choose a corrected dictionary before use. | Generate your own printable fiducial markers with a built-in print-scale calibration check | numpy, opencv-python, pillow |
| blinding | `blinding_harness.py` | Blind-coding harness: strips metadata, randomizes order, seals a hashed code↔truth map, unblinds only after reads recorded | Blind measurement / reduce observer bias for any before/after image or data comparison | standard library |
| audio_pipeline | `highlight_clip.py` | Finds the most 'musically dynamic' 6–10s window (loudness + spectral flux) and cuts a highlight clip | Auto-highlight-reel clip extractor for raw video / jam footage | numpy |
| audio_pipeline | `clean_and_split.py` | Runs Demucs stem separation over a folder of tracks with CUDA auto-detect | Batch stem-separate a folder with Demucs | highlight_clip, torch |
| audio_pipeline | `bed_prep.py` | Turns Demucs stems into a voiceover-ready, loudness-normalized, ducked/static music bed. Diff vs bed_prep_oss_release — publish the newer. | Music-bed prep from stem-separated audio for video/podcast editing | numpy |
| audio_pipeline | `captions.py` | Local offline captioning via faster-whisper — word-level timestamps, broadcast SRT/VTT, 'never collapse pauses' | Local/private video captioning — no cloud, no telemetry | ctranslate2, faster-whisper |
| audio_pipeline | `batch_captions.py` | Manifest-first tree-walker / queue-runner wrapper around captions.py | Batch driver for a captioning pipeline | captions |
| audio_pipeline | `serve_audition.py` | Tiny stdlib static file server adding HTTP Range support + a POST-to-append-a-pick endpoint | 'Serve large media with Range support, stdlib only, no deps' reference | standard library |
| media_organization | `extract_capture_dates.py` | Recovers true capture dates from EXIF/QuickTime when filesystem timestamps were reset by an iCloud export | Fix bogus filesystem dates using embedded media metadata | pillow, pillow_heif |
| media_organization | `extract_thumbs.py` | Generates small preview thumbnails from a media catalog for manual classification | Cheap thumbnail generation for manual tagging | standard library |
| media_organization | `backup_media.py` | Hash-verified single-copy-only backup — finds files on exactly one device, copies + re-verifies. Device keys forge/ssd_porsche are load-bearing. | 'Back up only what isn't already backed up, verify by hash' | standard library |
| media_organization | `sanitize_manifest.py` | Copies only media rows out of a mixed manifest into the synced folder, leaves doc/PII rows local | 'Split a manifest so only the safe-to-sync rows sync' | standard library |

## Notes
- `backup_media.py` keeps a small table of device labels (for example `forge`, `ssd_porsche`) that it matches against a catalogue's `source_label` values; rename them in both places together if you adapt it.
- `make_aruco_placards.py` carries a boxed WARNING about the default ArUco dictionary; pick a higher-margin dictionary before real use.
- `keyframe_gate.py` is a "cheap eyes, expensive brain" motion gate: it emits only the stable frames of a region of interest.
- `captions.py` and `batch_captions.py` run faster-whisper locally; word-level timestamps are kept and pauses are never collapsed.
- `blinding_harness.py` strips every cue that could bias a paired before/after reading, seals the code-to-truth map, and unblinds only after the reads are recorded.

## Provenance
Public-release copies, commented and scrubbed of personal paths and names; the originals were never modified. `MANIFEST_SHA256.txt` lists each file's hash. Issues and pull requests are welcome; please keep each tool a single file.

## Licence
See `LICENSE` (MIT).
