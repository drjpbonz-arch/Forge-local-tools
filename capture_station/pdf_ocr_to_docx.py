#!/usr/bin/env python3
"""
pdf_ocr_to_docx.py - convert no-text-layer (scanned) PDFs into .docx with BOTH
the OCR'd text AND the original page images, so figures/tables survive.

Runs on a local workstation (needs Ollama + a vision model, default qwen2.5vl:7b).
Local only - no cloud calls. Resume-safe: skips any PDF whose .docx already exists.

Per page: render at 200 DPI -> ask the VLM to transcribe -> write [page N] image
followed by its text. Figures are preserved as the page image itself, which is the
only honest way to keep a figure we cannot vectorize.

Usage:  python pdf_ocr_to_docx.py <in_dir> <out_dir> [--model qwen2.5vl:7b] [--dpi 200] [--max-pages 0]
"""
import os, sys, glob, io, json, base64, argparse, urllib.request, urllib.error, datetime, traceback

AP=argparse.ArgumentParser()
AP.add_argument("in_dir"); AP.add_argument("out_dir")
AP.add_argument("--model", default="qwen2.5vl:7b")
AP.add_argument("--dpi", type=int, default=200)
AP.add_argument("--max-pages", type=int, default=0, help="0 = all pages")
AP.add_argument("--host", default="http://localhost:11434")
A=AP.parse_args()

# Hard dependencies: PyMuPDF renders pages to images, python-docx builds the output.
try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF missing.  Fix:  pip install pymupdf python-docx  (the .bat does this for you)")
try:
    from docx import Document
    from docx.shared import Inches, Pt
except ImportError:
    sys.exit("python-docx missing.  Fix:  pip install python-docx")

# Transcription prompt: verbatim text in reading order, with no summarizing or "correcting".
PROMPT=("Transcribe ALL text on this scanned page of an academic paper, in reading order. "
        "Preserve paragraph breaks, headings, table contents and figure captions. "
        "Do NOT summarize, do NOT add commentary, do NOT correct apparent typos. "
        "If a region is illegible write [illegible]. If the page is a figure with no text, write [figure only]. "
        "Output plain text only.")

def ocr(img_bytes, model, host):
    # Send one rendered page image to the local vision model and return its transcription.
    b64=base64.b64encode(img_bytes).decode()
    body=json.dumps({"model":model,"stream":False,
        "messages":[{"role":"user","content":PROMPT,"images":[b64]}],
        "options":{"temperature":0}}).encode()
    req=urllib.request.Request(host+"/api/chat", body, {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]["content"].strip()

# --- Output dir + append-only run log (keeps progress visible and enables resume) ---
os.makedirs(A.out_dir, exist_ok=True)
log=open(os.path.join(A.out_dir,"_ocr_log.txt"),"a",encoding="utf-8")
def say(m):
    print(m, flush=True); log.write(m+"\n"); log.flush()

# --- Convert each PDF: skip ones already done, else OCR page by page ---
pdfs=sorted(glob.glob(os.path.join(A.in_dir,"*.pdf"))+glob.glob(os.path.join(A.in_dir,"*.PDF")))
say(f"=== OCR run {datetime.datetime.now():%Y-%m-%d %H:%M} | {len(pdfs)} pdfs | model={A.model} dpi={A.dpi} ===")
done=skip=fail=0
for n,p in enumerate(pdfs,1):
    stem=os.path.splitext(os.path.basename(p))[0]
    out=os.path.join(A.out_dir, stem+".docx")
    if os.path.exists(out) and os.path.getsize(out)>0:
        skip+=1; continue
    try:
        doc=fitz.open(p); npages=doc.page_count
        limit=npages if A.max_pages==0 else min(npages, A.max_pages)
        D=Document()
        h=D.add_heading(stem,0)
        D.add_paragraph(f"Source: {os.path.basename(p)} | {npages} pages | OCR: {A.model} @ {A.dpi} DPI | "
                        f"{datetime.datetime.now():%Y-%m-%d}").italic=True
        D.add_paragraph("Machine transcription of a scanned document. Page images retained below each "
                        "transcription so figures, tables and plates are preserved. Verify any quotation "
                        "against the page image before citing.").italic=True
        # Each page: render -> OCR (never fatal) -> write a heading, the text, then the page image.
        for i in range(limit):
            page=doc.load_page(i)
            pix=page.get_pixmap(dpi=A.dpi)
            img=pix.tobytes("png")
            try: text=ocr(img, A.model, A.host)
            except Exception as e: text=f"[OCR FAILED on this page: {e}]"
            D.add_page_break() if i else None
            D.add_heading(f"Page {i+1}", level=2)
            for para in [t for t in text.split("\n") if t.strip()]:
                D.add_paragraph(para)
            D.add_paragraph("— page image —").italic=True
            D.add_picture(io.BytesIO(img), width=Inches(6.0))
            say(f"  [{n}/{len(pdfs)}] {stem[:40]:40} page {i+1}/{limit}")
        # Save atomically: write to .part then replace, so an interrupted run never leaves a half file.
        tmp=out+".part"; D.save(tmp); os.replace(tmp,out); done+=1
        say(f"[{n}/{len(pdfs)}] OK   {stem[:50]}  ({limit} pages)")
    except Exception as e:
        fail+=1; say(f"[{n}/{len(pdfs)}] FAIL {stem[:50]} : {e}")
say(f"=== end: {done} converted, {skip} already done, {fail} failed ===")
