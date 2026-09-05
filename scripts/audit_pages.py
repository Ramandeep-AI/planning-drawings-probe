"""Audit every page of every downloaded PDF: born-digital vector, scanned
raster, or mixed. This is the first real result of the study, and it
answers the question that decides which half of the literature applies.

Decision rule per page (documented so it can be argued with):
    text_chars   = characters in the page's text layer
    n_drawings   = vector drawing commands (paths) on the page
    image_cover  = fraction of the page area covered by embedded raster images
  born-digital  : image_cover < 0.5 and (text_chars >= 20 or n_drawings >= 50)
  scanned       : image_cover >= 0.5 and text_chars < 20 and n_drawings < 50
  mixed         : image_cover >= 0.5 but with a real text layer or vector
                  content (e.g. a scan with an OCR text layer or CAD export
                  placed as an image with vector annotations)
Anything else is 'unclear' and worth a look by eye.

Also renders each page to PNG at --dpi for the OCR and vision stages.

Run:  env/bin/python scripts/audit_pages.py [--dpi 200]
Reads:  data/raw/<reference>/*.pdf  (see fetch script / manifest)
Writes: data/processed/pages.csv, data/processed/pages/<ref>__<file>__p<N>.png,
        outputs/metrics/page_audit.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PAGES = ROOT / "data" / "processed" / "pages"
CSV = ROOT / "data" / "processed" / "pages.csv"
MET = ROOT / "outputs" / "metrics" / "page_audit.json"


def classify(text_chars, n_drawings, image_cover):
    has_content = text_chars >= 20 or n_drawings >= 50
    if image_cover < 0.5 and has_content:
        return "born-digital"
    if image_cover >= 0.5 and not has_content:
        return "scanned"
    if image_cover >= 0.5 and has_content:
        return "mixed"
    return "unclear"


def audit_page(page):
    text_chars = len(page.get_text("text").strip())
    n_drawings = len(page.get_drawings())
    area = page.rect.get_area() or 1.0
    covered = 0.0
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            covered += r.get_area()
    image_cover = min(covered / area, 1.0)
    return text_chars, n_drawings, round(image_cover, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    PAGES.mkdir(parents=True, exist_ok=True)
    MET.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pdfs = sorted(RAW.glob("*/*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs under {RAW} - run the fetch script or copy files in")
    for pdf in pdfs:
        ref = pdf.parent.name
        try:
            doc = pymupdf.open(pdf)
        except Exception as e:  # corrupt download etc.
            rows.append({"reference": ref, "file": pdf.name, "page": 0, "kind": "unreadable",
                         "text_chars": 0, "n_drawings": 0, "image_cover": 0,
                         "width_mm": 0, "height_mm": 0, "error": str(e)[:80]})
            continue
        for i, page in enumerate(doc, 1):
            t, d, c = audit_page(page)
            kind = classify(t, d, c)
            png = PAGES / f"{ref}__{pdf.stem}__p{i}.png"
            if not args.no_render and not png.exists():
                page.get_pixmap(dpi=args.dpi).save(png)
            rows.append({"reference": ref, "file": pdf.name, "page": i, "kind": kind,
                         "text_chars": t, "n_drawings": d, "image_cover": c,
                         "width_mm": round(page.rect.width / 72 * 25.4),
                         "height_mm": round(page.rect.height / 72 * 25.4), "error": ""})
        doc.close()

    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    counts = df["kind"].value_counts().to_dict()
    share = df["kind"].value_counts(normalize=True).round(3).to_dict()
    print(f"{len(df)} pages from {df['file'].nunique()} files across "
          f"{df['reference'].nunique()} applications")
    for k, v in counts.items():
        print(f"  {k:13s} {v:4d}  ({share[k]:.1%})")
    by_app = df.groupby("reference")["kind"].agg(lambda s: s.value_counts().idxmax())
    print(f"\napplications by dominant page kind: {by_app.value_counts().to_dict()}")
    MET.write_text(json.dumps({"pages": int(len(df)), "files": int(df['file'].nunique()),
                               "applications": int(df['reference'].nunique()),
                               "page_kind_counts": counts, "page_kind_share": share,
                               "applications_by_dominant_kind": by_app.value_counts().to_dict(),
                               "dpi": args.dpi}, indent=2))
    print(f"wrote {CSV} and {MET}")


if __name__ == "__main__":
    main()
