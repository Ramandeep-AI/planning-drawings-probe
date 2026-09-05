"""Stage 2: read the title block of each drawing page and extract a few
structured fields with a confidence each.

THIS IS YOURS TO FINISH. The plumbing (page loading, docTR setup, output
format) is here; the decisions marked TODO are yours and you should be
able to defend each one.

Fields extracted per page (only these are ever written to disk in the
repo; raw OCR text is never committed because title blocks carry names
and addresses):
    scale          e.g. "1:50", "1:100", "1:1250"
    drawing_type   one of: floor_plan, elevation, section, location_plan,
                   site_plan, other
    floor_label    ground | first | second | roof | none
    north_arrow    yes | no   (from the words "north" / "N" near an arrow;
                   a vision check is the better route, see TODO 5)

Run:  env/bin/python -m src.ocr_titleblock
Reads:  data/processed/pages.csv + data/processed/pages/*.png (from audit_pages.py)
Writes: data/processed/extractions.csv  (reference,page,field,predicted,confidence)
        data/processed/ocr/<page>.json   (gitignored; full OCR output for your eyes)
"""
import csv
import json
import os
import re
from pathlib import Path

import certifi
import pandas as pd
import torch
from PIL import Image

# The python.org build of Python 3.13 ships without root certificates, so
# docTR's pretrained-model download fails with CERTIFICATE_VERIFY_FAILED.
# Point the SSL layer at certifi's bundle instead of touching system settings.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

ROOT = Path(__file__).resolve().parents[1]
PAGES_CSV = ROOT / "data" / "processed" / "pages.csv"
PAGES = ROOT / "data" / "processed" / "pages"
OCR_DIR = ROOT / "data" / "processed" / "ocr"
OUT = ROOT / "data" / "processed" / "extractions.csv"

# TODO 1 - the title block is usually a band along the bottom or the right
# edge of the sheet. Decide the crop: bottom 25% of the page? right 30%?
# both, OCR'd separately? Look at 10 rendered pages first and pick a rule.
def title_block_crop(img):
    w, h = img.size
    return img.crop((0, int(h * 0.75), w, h))          # bottom 25%, placeholder


# TODO 2 - docTR predictor. Docs: https://mindee.github.io/doctr/using_doctr/using_models.html
# Decide det/reco architectures and whether to straighten pages (rotated
# dimension strings are common). Runs on MPS.
def build_predictor():
    from doctr.models import ocr_predictor
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pred = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True,
                         assume_straight_pages=True)
    return pred.to(device)


SCALE_RE = re.compile(r"1\s*[:/]\s*(\d{2,4})")
TYPE_WORDS = {
    "floor_plan": re.compile(r"floor\s*plan|ground\s*floor|first\s*floor|layout", re.I),
    "elevation": re.compile(r"elevation", re.I),
    "section": re.compile(r"section", re.I),
    "location_plan": re.compile(r"location\s*plan", re.I),
    "site_plan": re.compile(r"site\s*plan|block\s*plan", re.I),
}
FLOOR_RE = re.compile(r"\b(ground|first|second|roof)\b", re.I)


def words_with_conf(export):
    for page in export["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                for w in line["words"]:
                    yield w["value"], float(w["confidence"])


def extract_fields(words):
    """Turn OCR words into the four fields with a confidence each.
    TODO 3 - confidence aggregation. Placeholder: the confidence of the
    single word that matched. Better: min or mean over the matched span,
    and a penalty when several candidates disagree (two different scales
    on one sheet is common: location plan 1:1250 plus a 1:100 detail)."""
    text = " ".join(v for v, _ in words)
    fields = {}
    m = SCALE_RE.search(text)
    if m:
        conf = next((c for v, c in words if m.group(0).replace(" ", "") in v.replace(" ", "")), 0.5)
        fields["scale"] = (f"1:{m.group(1)}", conf)
    for name, rx in TYPE_WORDS.items():
        if rx.search(text):
            conf = max((c for v, c in words if rx.search(v)), default=0.5)
            fields["drawing_type"] = (name, conf); break
    fm = FLOOR_RE.search(text)
    if fm:
        conf = max((c for v, c in words if FLOOR_RE.search(v)), default=0.5)
        fields["floor_label"] = (fm.group(1).lower(), conf)
    # TODO 4 - north arrow from OCR alone is weak ("N" is one letter).
    # Decide: skip it here and detect the arrow glyph in the vision stage.
    return fields


def main():
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    pages = pd.read_csv(PAGES_CSV)
    pred = build_predictor()
    from doctr.io import DocumentFile
    rows = []
    for _, p in pages.iterrows():
        if p["kind"] == "unreadable":
            continue
        png = PAGES / f"{p['reference']}__{Path(p['file']).stem}__p{p['page']}.png"
        if not png.exists():
            continue
        crop = title_block_crop(Image.open(png).convert("RGB"))
        crop_path = OCR_DIR / (png.stem + "_titleblock.png")
        crop.save(crop_path)
        export = pred(DocumentFile.from_images([str(crop_path)])).export()
        (OCR_DIR / (png.stem + ".json")).write_text(json.dumps(export))
        words = list(words_with_conf(export))
        for field, (value, conf) in extract_fields(words).items():
            rows.append([p["reference"], p["page"], field, value, round(conf, 3)])
        print(f"{png.name}: {len(words)} words, fields {[r[2] for r in rows if r[0]==p['reference'] and r[1]==p['page']]}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["reference", "page", "field", "predicted", "confidence"]); w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} field predictions)")
    # TODO 5 - after the first run, read 10 of the JSON files by eye and
    # write down what the OCR gets wrong on drawings. That list is the
    # "failure cases" section of the README.


if __name__ == "__main__":
    main()
