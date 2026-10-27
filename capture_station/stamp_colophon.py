#!/usr/bin/env python3
"""
stamp_colophon.py - stamp a collaborative-authorship credit into .docx files.
Adds (1) a visible colophon block on the final page and (2) embedded document
properties (author/comments) that survive conversion to PDF and travel with the file.

Uses the CRediT taxonomy (Contributor Roles Taxonomy), the academic standard that
separates intellectual contribution from typing volume, so a human-AI collaboration
is recorded honestly on the artifact itself.

Usage: stamp_colophon.py <file-or-folder> [--author "Name"] [--roles <roles.txt>]
  --author  name credited as the human author (default: a generic placeholder).
  --roles   path to a text file whose contents replace the default CRediT string.
Skips files already stamped (idempotent). Never modifies content above the colophon.
"""
import sys, os, glob, argparse, datetime

try:
    from docx import Document
    from docx.shared import Pt, RGBColor
except ImportError:
    sys.exit("python-docx missing. Fix: pip install python-docx (the .bat does this for you)")

# The human author name is supplied at runtime via --author; the source ships with
# a generic placeholder so no personal identity is baked into the code.
DEFAULT_AUTHOR = "A. Author"

def default_roles(author):
    """Build the default CRediT role statement for the given author name, separating
    human contribution (conception, adjudication, final authority) from the AI
    assistant's (research execution, original drafting)."""
    return (
        f"Conceptualization: {author}.  Methodology: {author} with an AI assistant.  "
        f"Investigation & Research Execution: AI assistant.  Validation & Adjudication: {author}.  "
        f"Writing - Original Draft: AI assistant.  Writing - Review, Editing & Final Authority: {author}."
    )

# Marker doubles as the visible heading and the idempotency sentinel (self-consistent).
MARKER = "COLLABORATIVE WORK PRODUCT"

def stamp(path, roles, author):
    # Idempotency: if the marker already appears near the end, treat as done.
    doc = Document(path)
    if any(MARKER in p.text for p in doc.paragraphs[-8:]):
        return "already-stamped"
    # Visible colophon block, appended to the final page.
    doc.add_paragraph()  # spacer
    line = doc.add_paragraph()
    r = line.add_run("_" * 60); r.font.size = Pt(8)
    head = doc.add_paragraph(); rh = head.add_run(MARKER)
    rh.bold = True; rh.font.size = Pt(8)
    body = doc.add_paragraph()
    rb = body.add_run(
        f"{roles}  Produced {datetime.date.today().isoformat()}. "
        "Human origination and final judgment; AI research, drafting and tooling. "
        "Provenance: hash manifests and session records held in project archives.")
    rb.font.size = Pt(7.5); rb.font.color.rgb = RGBColor(0x59,0x59,0x59)
    # Embedded properties survive into PDF exports and file-properties panes, so the
    # credit travels with the file even when the visible page is cropped or converted.
    cp = doc.core_properties
    cp.author = f"{author} (with an AI assistant)"
    cp.comments = f"CRediT: {roles}"
    if not cp.title: cp.title = os.path.splitext(os.path.basename(path))[0]
    doc.save(path)
    return "stamped"

# CLI: accept a single .docx or a folder to walk recursively.
A = argparse.ArgumentParser()
A.add_argument("target")
A.add_argument("--author", default=DEFAULT_AUTHOR, help="human author name to credit")
A.add_argument("--roles", default=None, help="path to a file overriding the default CRediT string")
args = A.parse_args()
# Roles precedence: an explicit --roles file wins, else the parameterized default.
roles = (open(args.roles, encoding="utf-8").read().strip()
         if args.roles and os.path.exists(args.roles) else default_roles(args.author))
# Resolve target to a list of .docx files, skipping Word lock files (~$...).
files = [args.target] if args.target.lower().endswith(".docx") else glob.glob(os.path.join(args.target, "**", "*.docx"), recursive=True)
files = [f for f in files if not os.path.basename(f).startswith("~$")]
ok = skip = fail = 0
for f in files:
    try:
        r = stamp(f, roles, args.author)
        ok += (r == "stamped"); skip += (r == "already-stamped")
        print(f"  {r:16} {os.path.basename(f)}")
    except Exception as e:
        fail += 1; print(f"  FAILED           {os.path.basename(f)}: {e}")
print(f"\nstamped {ok}, already-stamped {skip}, failed {fail}")
