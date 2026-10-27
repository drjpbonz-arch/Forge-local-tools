#!/usr/bin/env python3
"""
make_aruco_placards.py - printable ArUco placard sheet generator.

Generates printable fiducial-marker placards for use as scale/identity references
during photo capture. The dictionaries are open (BSD-licensed OpenCV) and marker
generation is a single call, and printing in-house is how we control the one thing
that actually causes measurement error: PRINTED SIZE. A reference card printed at
136.x mm against a 146 mm spec made every dimensional measurement over-read 6.6-7.4%.

So every sheet this tool emits carries a 100.0 mm CALIBRATION BAR. The rule
printed on the sheet: caliper the bar after printing; if it is not 100.0 mm,
the print dialog scaled it - reprint at 100% / Actual Size. Record the measured
bar value on the sheet in pen. A placard without a verified bar is not a
reference object.

DICTIONARY NOTE: this prototype uses DICT_ARUCO_ORIGINAL. See the prominent WARNING
at the getPredefinedDictionary() call in main() - that dictionary has a low
inter-marker Hamming margin and MUST be replaced with a higher-margin dictionary
before real use.

Usage:
  python make_aruco_placards.py --out ARUCO_PLACARDS.pdf
  python make_aruco_placards.py --out sheet.pdf --ids 0-11 --size-mm 60
"""
import argparse, sys
from datetime import date

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("ERROR: opencv-python + Pillow required.")

DPI = 300
MM = DPI / 25.4                      # px per mm
PAGE_W, PAGE_H = int(8.5 * DPI), int(11 * DPI)   # US Letter portrait


def font(sz):
    for c in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "C:\\Windows\\Fonts\\arialbd.ttf", "C:\\Windows\\Fonts\\arial.ttf"):
        try:
            return ImageFont.truetype(c, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def marker_img(dic, mid, size_px):
    m = cv2.aruco.generateImageMarker(dic, mid, size_px)
    return Image.fromarray(m).convert("L")


def calibration_bar(draw, x, y):
    """Draw the 100.0 mm bar with end ticks + label - the guard against a print
    dialog silently scaling the sheet and corrupting every measurement taken from it."""
    L = int(100 * MM)
    draw.rectangle([x, y, x + L, y + int(1.2 * MM)], fill=0)
    for tx in (x, x + L):
        draw.rectangle([tx - 2, y - int(2.5 * MM), tx + 2, y + int(3.5 * MM)], fill=0)
    draw.text((x, y + int(4.5 * MM)),
              "CALIBRATION BAR - must measure exactly 100.0 mm. If not, the print was scaled:",
              font=font(38), fill=0)
    draw.text((x, y + int(4.5 * MM) + 46),
              "reprint at 100% / Actual Size. Write the measured value + date here: ________  ________",
              font=font(38), fill=0)


# Render one US-Letter sheet: title, dictionary/lamination note, the calibration
# bar, then a grid of markers each with its own quiet zone, cut guide and id label.
def build_page(dic, ids, size_mm, title):
    page = Image.new("L", (PAGE_W, PAGE_H), 255)
    d = ImageDraw.Draw(page)
    margin = int(0.55 * DPI)
    d.text((margin, margin - 60), title, font=font(52), fill=0)
    d.text((margin, margin + 4),
           f"DICT_ARUCO_ORIGINAL | edge {size_mm:.0f} mm | laminate MATTE only - gloss glare kills detection",
           font=font(36), fill=0)
    calibration_bar(d, margin, margin + int(9 * MM))
    top = margin + int(24 * MM)

    s_px = int(size_mm * MM)
    quiet = int(max(6.0, size_mm * 0.12) * MM)         # white quiet zone
    cell = s_px + 2 * quiet
    label_h = 70
    cols = max(1, (PAGE_W - 2 * margin) // cell)
    x = y = 0
    for i, mid in enumerate(ids):
        cx = margin + (i % cols) * cell
        cy = top + (i // cols) * (cell + label_h)
        if cy + cell + label_h > PAGE_H - margin:
            print(f"  page full at id {mid}; increase pages or reduce ids")
            break
        page.paste(marker_img(dic, mid, s_px), (cx + quiet, cy + quiet))
        d.rectangle([cx + quiet - 3, cy + quiet - 3, cx + quiet + s_px + 3, cy + quiet + s_px + 3],
                    outline=200, width=1)               # faint cut guide
        d.text((cx + quiet, cy + quiet + s_px + 12),
               f"ID {mid}  |  {size_mm:.0f} mm  |  ORIGINAL", font=font(40), fill=0)
    d.text((margin, PAGE_H - margin + 30),
           f"ArUco placard sheet - generated {date.today().isoformat()} - make_aruco_placards.py",
           font=font(32), fill=100)
    return page


def parse_ids(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# Assemble a two-page PDF: a small-marker set and a large-marker set.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="0-9", help="e.g. 0-9 or 0,1,7")
    ap.add_argument("--size-mm", type=float, default=60.0)
    ap.add_argument("--big-ids", default="10-13", help="second page, large placards")
    ap.add_argument("--big-size-mm", type=float, default=100.0)
    a = ap.parse_args()

    # =========================================================================
    # WARNING - DEFECTIVE DICTIONARY CHOICE, MUST BE REPLACED BEFORE REAL USE
    # -------------------------------------------------------------------------
    # DICT_ARUCO_ORIGINAL has only a ~3-bit minimum inter-marker Hamming margin.
    # That low margin means a smudge, a glare hotspot, or a slight print/scan
    # defect can flip enough bits to MISREAD one marker id as another - a silent,
    # wrong identity, the worst failure mode for a reference/identity tag.
    # Do NOT ship this as-is: replace it with a higher-margin dictionary (e.g.
    # DICT_5X5_100 or DICT_6X6_250, which carry a much larger Hamming distance)
    # and regenerate every placard. This line is left UNCHANGED on purpose so the
    # defect stays visible and flagged, not silently "fixed".
    # =========================================================================
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    p1 = build_page(dic, parse_ids(a.ids), a.size_mm,
                    "ARUCO PLACARDS - small set")
    p2 = build_page(dic, parse_ids(a.big_ids), a.big_size_mm,
                    "ARUCO PLACARDS - large set")
    p1.save(a.out, "PDF", resolution=DPI, save_all=True, append_images=[p2])
    print(f"written: {a.out}  (2 pages @ {DPI} dpi)")
    print("PRINT AT 100% / ACTUAL SIZE. Caliper the 100.0 mm bar before first use.")


if __name__ == "__main__":
    main()
