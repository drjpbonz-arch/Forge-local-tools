#!/usr/bin/env python3
"""
qwen_bench.py - benchmark a LOCAL Ollama vision model against a confirmed inventory
ground truth (a corrections JSON) for OCR-style transcription of printed equipment text.
Reproduces a two-tier scoring scheme (strict / inclusive) so results compare to a
measured baseline (e.g. Tesseract 14%/50%, a frontier model 100% on a held-out set).

Runs on a local Windows box (Ollama at localhost:11434). No cloud calls.
Anti-hallucination prompt design: exact transcription, prefer UNREADABLE over guessing,
temperature 0, JSON schema forced.

Usage:
  python qwen_bench.py --model qwen2.5vl:7b --corrections <file.json> --frames <dir> [--n 28] [--all] [--seed 7]
Outputs: BENCH_<model>_<date>.md + .csv beside the corrections file.
HEIC frames are converted/downscaled (1600px) via pillow-heif.
"""
import json, os, sys, base64, argparse, random, re, csv, datetime, io
import urllib.request, urllib.error
from PIL import Image
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError:
    print("pillow-heif missing: pip install pillow-heif"); sys.exit(1)

# CLI. --corrections (ground-truth JSON) and --frames (source image dir) are REQUIRED and
# supplied at run time - no machine-specific default paths are baked into this tool.
AP = argparse.ArgumentParser()
AP.add_argument("--model", default="qwen2.5vl:7b")
AP.add_argument("--n", type=int, default=28)
AP.add_argument("--all", action="store_true")
AP.add_argument("--seed", type=int, default=7)
AP.add_argument("--corrections", required=True, help="path to the ground-truth corrections JSON")
AP.add_argument("--frames", required=True, help="directory of source frames referenced by the corrections file")
AP.add_argument("--host", default="http://localhost:11434")
A = AP.parse_args()

# Prompt is engineered against hallucination: transcribe verbatim, and say UNREADABLE
# rather than invent a character. Inventing text is scored as a critical failure.
PROMPT = ('Transcribe EXACTLY what is printed on the equipment in this photo - every character, digit, '
 'hyphen. Do not paraphrase, normalize, correct, or complete anything. Do not infer or guess any '
 'character you cannot clearly resolve. If a field is blurry, cut off, glare-obscured, or not visible, '
 'write UNREADABLE for that field instead of a plausible-looking guess - inventing text is a critical '
 'failure. Return ONLY JSON: {"manufacturer":"","model":"","serial":"","other_text":""}')
SCHEMA = {"type":"object","properties":{"manufacturer":{"type":"string"},"model":{"type":"string"},
          "serial":{"type":"string"},"other_text":{"type":"string"}},
          "required":["manufacturer","model","serial","other_text"]}

def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def img_b64(path):
    # Load, downscale to <=1600px, and base64-encode for the Ollama chat API.
    im = Image.open(path).convert("RGB")
    im.thumbnail((1600, 1600))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def _call(model, image, use_format=True):
    payload = {"model": model, "stream": False,
        "messages":[{"role":"user","content":PROMPT,"images":[image]}],
        "options":{"temperature":0}}
    if use_format: payload["format"] = SCHEMA
    req = urllib.request.Request(A.host+"/api/chat", json.dumps(payload).encode(),
                                 {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out["message"]["content"]

def ask(model, images):
    """ONE image per call - this Ollama build 400s on 3+ images (measured 19 AUG).
    Answers from up to 3 frames are merged; scoring checks the union."""
    merged = {"manufacturer":"","model":"","serial":"","other_text":""}
    raw_all = []
    for img in images:
        try:
            content = _call(model, img, True)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                content = _call(model, img, False)   # some builds 400 on format+images
            else:
                raise
        raw_all.append(content)
        try:
            ans = json.loads(content)
            for k in merged:
                v = str(ans.get(k,"") or "")
                if v and v.upper() != "UNREADABLE" and v not in merged[k]:
                    merged[k] = (merged[k] + " | " + v).strip(" |")
        except Exception:
            merged["other_text"] = (merged["other_text"] + " | " + content[:200]).strip(" |")
    merged["_raw"] = " ".join(raw_all)
    return merged

# --- Load the ground truth and keep only scorable records (confirmed, with make + model) ---
recs = json.load(open(A.corrections))
recs = recs if isinstance(recs, list) else recs["items"]
recs = [r for r in recs if r.get("confirmed") and r.get("manufacturer","").strip() and r.get("model_part_no","").strip()]  # scorable only
random.Random(A.seed).shuffle(recs)
if not A.all: recs = recs[:A.n]
print(f"model={A.model}  items={len(recs)}  frames dir={A.frames}")

# --- Score each item: gather up to 3 frames, query the model, compare to truth ---
rows=[]; strict=incl=0
for i, r in enumerate(recs, 1):
    frames = [f for f in r.get("frames","").split() if f]
    imgs=[]
    for fr in frames[:3]:
        for ext in (".HEIC",".heic",".JPG",".jpg",".JPEG",".PNG"):
            p = os.path.join(A.frames, fr+ext)
            if os.path.exists(p): imgs.append(img_b64(p)); break
    if not imgs:
        rows.append([r["id"], r["name"], "", "", "NO_FRAMES", ""]); continue
    try: ans = ask(A.model, imgs)
    except Exception as e:
        rows.append([r["id"], r["name"], "", "", "CALL_FAIL", str(e)[:80]]); continue
    gm, gp = norm(r.get("manufacturer")), norm(r.get("model_part_no"))
    am, ap = norm(ans.get("manufacturer")), norm(ans.get("model"))
    blob = norm(json.dumps(ans)) + norm(ans.get("_raw",""))
    s = bool(gm and gp and gm in blob and gp in blob)              # strict: both present anywhere in answer
    inc = bool((gm and gm in blob) or (gp and len(gp)>2 and gp in blob))  # inclusive: either
    strict += s; incl += inc
    rows.append([r["id"], r["name"], ans.get("manufacturer",""), ans.get("model",""),
                 "STRICT" if s else ("INCL" if inc else "MISS"), ";".join(frames[:3])])
    print(f"[{i}/{len(recs)}] {'STRICT' if s else ('INCL ' if inc else 'MISS ')}  {r['name'][:50]}")

# --- Write CSV + Markdown summary beside the corrections file ---
n=len(rows); d=datetime.date.today().isoformat(); tag=re.sub(r'[^a-z0-9]','_',A.model)
out=os.path.join(os.path.dirname(A.corrections), f"BENCH_{tag}_{d}")
with open(out+".csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["id","truth_name","ans_manufacturer","ans_model","verdict","frames"]); w.writerows(rows)
misses=sum(1 for r in rows if r[4]=="MISS"); fails=sum(1 for r in rows if r[4] in("CALL_FAIL","NO_FRAMES"))
md=(f"# BENCH {A.model} {d}\nitems={n} strict={strict} ({strict/max(n,1)*100:.0f}%) "
    f"inclusive={incl} ({incl/max(n,1)*100:.0f}%) miss={misses} fail={fails}\n"
    f"Baseline (Run 2, 28 held-out): Tesseract 14%/50% - frontier 100%.\n"
    f"Scoring identical fields (manufacturer+model strict / either inclusive), seed={A.seed}.\n")
open(out+".md","w",encoding="utf-8").write(md)
print("\n"+md+f"written: {out}.csv / .md")
