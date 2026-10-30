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

## Third-party licences

The code in this repository is MIT (see `LICENSE`). It bundles no third-party code: each tool imports packages you install yourself, and some call programs you install separately. Their licences, as each project declares them on PyPI (checked 27 and 30 OCT 2026):

| package | used by | licence |
|---|---|---|
| numpy | 4 tools | BSD-3-Clause, with 0BSD, MIT, Zlib and CC0-1.0 parts |
| opencv-python | keyframe_gate, make_aruco_placards | Apache-2.0 |
| Pillow | 5 tools | MIT-CMU |
| pillow-heif | prep_listing_photos, qwen_bench, extract_capture_dates | BSD-3-Clause (its binary wheels include native HEIF libraries under their own licences) |
| pypdfium2 | pdf_ocr_to_docx | Apache-2.0 or BSD-3-Clause (its wheels include PDFium and PDFium's dependencies under their own licences) |
| python-docx | pdf_ocr_to_docx, stamp_colophon | MIT |
| ctranslate2 | captions | MIT |
| faster-whisper | captions | MIT |
| torch | clean_and_split | Apache-2.0, BSD-2-Clause, BSD-3-Clause, BSL-1.0 and MIT parts |

External programs and models, each under its own terms, which you accept when you install them: `ffmpeg`/`ffprobe` (LGPL-2.1 or GPL, depending on the build you install), Demucs (MIT code; its pretrained weights carry their own terms, so check them before commercial use), Ollama (MIT) and whichever models you run in it, and the speech models faster-whisper downloads (see each model's card).
