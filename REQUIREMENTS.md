# Requirements by tool

Python 3.10+. Standard-library-only tools need nothing else.

- `capture_station/keyframe_gate.py`: numpy, opencv-python
- `capture_station/crosscheck_hash_inventories.py`: standard library
- `capture_station/match_inventory_to_tree.py`: standard library
- `capture_station/prep_listing_photos.py`: pillow, pillow_heif
- `capture_station/qwen_bench.py`: pillow, pillow_heif
- `capture_station/pdf_ocr_to_docx.py`: pypdfium2, pillow, python-docx
- `capture_station/stamp_colophon.py`: python-docx
- `capture_station/make_aruco_placards.py`: numpy, opencv-python, pillow
- `blinding/blinding_harness.py`: standard library
- `audio_pipeline/highlight_clip.py`: numpy
- `audio_pipeline/clean_and_split.py`: highlight_clip, torch
- `audio_pipeline/bed_prep.py`: numpy
- `audio_pipeline/captions.py`: ctranslate2, faster-whisper
- `audio_pipeline/batch_captions.py`: captions
- `audio_pipeline/serve_audition.py`: standard library
- `media_organization/extract_capture_dates.py`: pillow, pillow_heif
- `media_organization/extract_thumbs.py`: standard library
- `media_organization/backup_media.py`: standard library
- `media_organization/sanitize_manifest.py`: standard library

External programs some tools call: `ffmpeg`/`ffprobe` (audio_pipeline), `demucs` (clean_and_split.py), a local Ollama server (qwen_bench.py, pdf_ocr_to_docx.py).
