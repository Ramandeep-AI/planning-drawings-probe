"""Stage 2: read the title block of each drawing page and extract three
structured fields with a confidence each: scale, drawing_type, floor_label.

The decisions below were taken after the page audit (Result 1) and after
measuring where the words sit on the 455 pages that carry a text layer.
Each one is stated so that it can be argued with.

D1  Source order. The PDF text layer is read first, for every page that has
    one. docTR (OCR on the rendered page) is used only when the page has no
    usable text layer, or when the text layer yields neither a scale nor a
    drawing type, which happens when the title block was exported as
    outlined glyphs rather than text.
D2  Title-block zone. Bottom 25 percent of the sheet, or right 30 percent.
    Measured on the text-layer pages: 74 percent of "1:NNN" strings sit in
    the bottom band and 35 percent in the right band; their union covers
    most title blocks in this register. OCR pages are read through two crops
    (bottom 30 percent, right 30 percent) so the zone is what the OCR sees;
    if the crops yield neither a scale nor a type, the whole sheet is read
    once more at reduced resolution, which catches large titles set in the
    body of the drawing and nothing smaller.
D3  Choosing between candidates. A scale on the same line as the word
    "scale" beats one that is not; a candidate inside the zone beats one
    outside it; a tie is broken by text size, because the title in a title
    block is set larger than the labels under individual drawings. Two
    distinct scales on one sheet (a 1:1250 location plan beside a 1:100
    plan is common) lower the confidence of whichever wins. The same rule
    picks the drawing type. A floor label is only looked for on pages
    predicted to be floor plans; elevations, sections and location plans
    get "none" at the drawing-type confidence.
D4  Confidence. A heuristic score in [0.05, 0.98] built from that evidence
    (zone, label, agreement), multiplied by the mean OCR word confidence on
    OCR pages. It is not calibrated by construction; stage 4 measures how
    far it is from calibrated and where to put the abstention threshold.
D5  Defaults. When a page has text but no keyword, the page is predicted
    "other" for drawing_type (0.30), "none" for floor_label (0.60) and
    "none" for scale (0.40), because absence of the words is weak evidence
    of absence of the thing. When neither the text layer nor OCR finds any
    words at all, no prediction is written and the blank counts as a miss.
D6  North arrow is not measured here; text cannot see an arrow. Annotated
    anyway for the vision stage.
D7  The document kind derived from the register's listed title (the file
    name) is never used. It would be a strong prior in a product, but the
    question here is whether the drawing itself can be read.

Only the three fields are ever written to the repo. Raw text and OCR
output stay in gitignored files because title blocks carry names and
addresses.

Run:  env/bin/python -m src.ocr_titleblock [--limit N] [--out path]
Reads:  data/processed/pages.csv, data/raw/<ref>/<file>.pdf,
        data/processed/pages/*.png (rendered by audit_pages.py)
Writes: data/processed/extractions.csv (reference,sheet,field,predicted,confidence,source)
        where sheet = two-digit document index + page number, e.g. 03-1
        data/processed/ocr/<page>.json and crops (gitignored)
"""
import argparse
import csv
import json
import os
import re
from collections import namedtuple
from pathlib import Path

import certifi
import pandas as pd
import pymupdf
from PIL import Image

# The python.org build of Python 3.13 ships without root certificates, so
# docTR's pretrained-model download fails with CERTIFICATE_VERIFY_FAILED.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

ROOT = Path(__file__).resolve().parents[1]
PAGES_CSV = ROOT / "data" / "processed" / "pages.csv"
PAGES = ROOT / "data" / "processed" / "pages"
RAW = ROOT / "data" / "raw"
OCR_DIR = ROOT / "data" / "processed" / "ocr"
OUT = ROOT / "data" / "processed" / "extractions.csv"

ZONE_BOTTOM = 0.75      # D2: y (0 top, 1 bottom) at or beyond this is the bottom band
ZONE_RIGHT = 0.70       # D2: x at or beyond this is the right band
CROP_BOTTOM = 0.70      # OCR crop: from this y to the bottom edge
CROP_RIGHT = 0.70       # OCR crop: from this x to the right edge

Line = namedtuple("Line", "text conf x y size")   # x,y = centre in page fractions; size = text height

SCALE_RE = re.compile(r"\b1\s*[:/]\s*(\d{1,4})\b")
NTS_RE = re.compile(r"\bn\.?t\.?s\.?\b|not\s+to\s+scale", re.I)
SCALE_LABEL_RE = re.compile(r"\bscale\b", re.I)
TYPE_RES = [   # (name, pattern, base score); "existing plans" alone is a weaker signal
    ("floor_plan", re.compile(r"floor\s*plans?|roof\s*plans?|(ground|first|second|third)\s*floor", re.I), 0.5),
    ("elevation", re.compile(r"elevations?", re.I), 0.5),
    ("section", re.compile(r"\bsections?\b", re.I), 0.5),
    ("location_plan", re.compile(r"location\s*plan", re.I), 0.5),
    ("site_plan", re.compile(r"site\s*plan|block\s*plan", re.I), 0.5),
    ("floor_plan", re.compile(r"\b(existing|proposed)\s+plans?\b|general\s+arrangement", re.I), 0.4),
]
FLOOR_RES = [
    ("ground", re.compile(r"(lower\s+)?ground\s*floor", re.I)),
    ("first", re.compile(r"first\s*floor|1st\s*floor", re.I)),
    ("second", re.compile(r"second\s*floor|2nd\s*floor", re.I)),
    ("roof", re.compile(r"roof\s*plan", re.I)),
]


def in_zone(x, y):
    return y >= ZONE_BOTTOM or x >= ZONE_RIGHT


def clip(c):
    return round(max(0.05, min(0.98, c)), 3)


# ---------------------------------------------------------------- sources
def text_layer_lines(pdf_path, page_no):
    """Lines of the PDF text layer with centre positions as page fractions."""
    doc = pymupdf.open(pdf_path)
    page = doc[page_no - 1]
    W, H = page.rect.width, page.rect.height
    groups = {}
    for x0, y0, x1, y1, word, block, line, _ in page.get_text("words"):
        g = groups.setdefault((block, line), [])
        g.append((word, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (y1 - y0) / H))
    doc.close()
    lines = []
    for ws in groups.values():
        lines.append(Line(" ".join(w[0] for w in ws), 1.0,
                          sum(w[1] for w in ws) / len(ws), sum(w[2] for w in ws) / len(ws),
                          max(w[3] for w in ws)))
    return lines


_predictor = None


def predictor():
    global _predictor
    if _predictor is None:
        import torch
        from doctr.models import ocr_predictor
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn",
                                   pretrained=True, assume_straight_pages=True).to(device)
    return _predictor


def ocr_lines(png, full_page=False):
    """docTR on two crops (bottom band, right band), or on the whole sheet
    at reduced resolution; positions mapped back to page fractions."""
    from doctr.io import DocumentFile
    img = Image.open(png).convert("RGB")
    w, h = img.size
    if full_page:
        s = min(1.0, 2000 / max(w, h))
        crops = {"full": (img.resize((int(w * s), int(h * s))), lambda x, y: (x, y))}
    else:
        crops = {
            "bottom": (img.crop((0, int(h * CROP_BOTTOM), w, h)), lambda x, y: (x, CROP_BOTTOM + y * (1 - CROP_BOTTOM))),
            "right": (img.crop((int(w * CROP_RIGHT), 0, w, h)), lambda x, y: (CROP_RIGHT + x * (1 - CROP_RIGHT), y)),
        }
    paths = []
    for name, (crop, _) in crops.items():
        p = OCR_DIR / f"{png.stem}_{name}.png"
        crop.save(p)
        paths.append(p)
    export = predictor()(DocumentFile.from_images([str(p) for p in paths])).export()
    (OCR_DIR / f"{png.stem}{'_full' if full_page else ''}.json").write_text(json.dumps(export))
    lines, seen = [], set()
    for page, (name, (_, remap)) in zip(export["pages"], crops.items()):
        for block in page["blocks"]:
            for line in block["lines"]:
                words = line["words"]
                if not words:
                    continue
                text = " ".join(w["value"] for w in words)
                conf = sum(float(w["confidence"]) for w in words) / len(words)
                (x0, y0), (x1, y1) = line["geometry"]
                x, y = remap((x0 + x1) / 2, (y0 + y1) / 2)
                size = (y1 - y0) * (1 - CROP_BOTTOM if name == "bottom" else 1)   # page fraction
                key = (text, round(x, 2), round(y, 2))
                if key in seen:          # the two crops overlap in the bottom-right corner
                    continue
                seen.add(key)
                lines.append(Line(text, conf, x, y, size))
    return lines


# ---------------------------------------------------------------- fields
def best(cands):
    """cands: list of (value, score, conf, size). Pick the top score, ties
    broken by text size; penalise disagreement between distinct values (D3)."""
    if not cands:
        return None
    cands = sorted(cands, key=lambda c: (c[1], c[3]), reverse=True)
    value, score, conf, _ = cands[0]
    distinct = {c[0] for c in cands}
    if len(distinct) > 1:
        rival = max((c[1] for c in cands if c[0] != value), default=0)
        score -= 0.15 if rival >= score - 0.1 else 0.05
    return value, clip(score * conf)


def extract_scale(lines):
    cands = []
    for ln in lines:
        labelled = bool(SCALE_LABEL_RE.search(ln.text))
        for m in SCALE_RE.finditer(ln.text):
            cands.append((f"1:{int(m.group(1))}", 0.55 + 0.2 * in_zone(ln.x, ln.y) + 0.2 * labelled,
                          ln.conf, ln.size))
        if NTS_RE.search(ln.text):
            cands.append(("nts", 0.55 + 0.2 * in_zone(ln.x, ln.y) + 0.2 * labelled, ln.conf, ln.size))
    return best(cands)


def extract_type(lines):
    cands = []
    for ln in lines:
        short = len(ln.text.split()) <= 6          # title-like line
        for name, rx, base in TYPE_RES:
            if rx.search(ln.text):
                cands.append((name, base + 0.3 * in_zone(ln.x, ln.y) + 0.1 * short, ln.conf, ln.size))
    return best(cands)


def extract_floor(lines):
    cands = []
    for ln in lines:
        short = len(ln.text.split()) <= 6
        for name, rx in FLOOR_RES:
            if rx.search(ln.text):
                cands.append((name, 0.5 + 0.3 * in_zone(ln.x, ln.y) + 0.1 * short, ln.conf, ln.size))
    zone_labels = {c[0] for c in cands if c[1] >= 0.8}
    if len(zone_labels) > 1:                       # two floors titled in the title block
        return "multiple", clip(0.6 * min(c[2] for c in cands))
    return best(cands)


def extract_fields(lines):
    """Three fields from the lines of one page; D5 defaults when words exist
    but no keyword does; floor label only on floor plans (D3)."""
    if not lines:
        return {}
    mean_conf = sum(l.conf for l in lines) / len(lines)
    dtype = extract_type(lines) or ("other", clip(0.30 * mean_conf))
    if dtype[0] == "floor_plan":
        floor = extract_floor(lines) or ("none", clip(0.60 * mean_conf))
    else:
        floor = ("none", clip(min(0.9, dtype[1])))
    return {
        "scale": extract_scale(lines) or ("none", clip(0.40 * mean_conf)),
        "drawing_type": dtype,
        "floor_label": floor,
    }


def found_anything(fields):
    return fields and (fields["scale"][0] != "none" or fields["drawing_type"][0] != "other")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages (smoke test)")
    ap.add_argument("--only-kind", help="only pages of this audit kind, e.g. scanned (smoke test)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    pages = pd.read_csv(PAGES_CSV)
    pages = pages[pages.kind != "unreadable"]
    if args.only_kind:
        pages = pages[pages.kind == args.only_kind]
    if args.limit:
        pages = pages.head(args.limit)

    rows, n_text, n_ocr, n_none = [], 0, 0, 0
    for p in pages.itertuples():
        pdf = RAW / p.reference / p.file
        png = PAGES / f"{p.reference}__{Path(p.file).stem}__p{p.page}.png"
        lines, source = [], "none"
        if p.text_chars >= 20 and pdf.exists():
            lines = text_layer_lines(pdf, int(p.page))
            source = "text_layer"
        fields = extract_fields(lines)
        if not found_anything(fields) and png.exists():          # D1 fallback
            ocr_fields = extract_fields(ocr_lines(png))
            if not found_anything(ocr_fields):                    # D2 last resort
                full_fields = extract_fields(ocr_lines(png, full_page=True))
                if found_anything(full_fields) or not ocr_fields:
                    ocr_fields = full_fields
            if found_anything(ocr_fields) or not fields:
                fields, source = ocr_fields, "doctr"
        if not fields:
            n_none += 1
        elif source == "doctr":
            n_ocr += 1
        else:
            n_text += 1
        sheet = f"{p.file[:2]}-{p.page}"          # document index + page, unique within an application
        for field, (value, conf) in fields.items():
            rows.append([p.reference, sheet, field, value, conf, source])
        print(f"{png.name}: {source:10s} " + ", ".join(f"{k}={v[0]}({v[1]})" for k, v in fields.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference", "sheet", "field", "predicted", "confidence", "source"])
        w.writerows(rows)
    print(f"\n{len(pages)} pages: {n_text} read from the text layer, {n_ocr} from OCR, "
          f"{n_none} with no words at all. Wrote {args.out} ({len(rows)} field predictions).")


if __name__ == "__main__":
    main()
