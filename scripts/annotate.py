"""Hand-annotate title-block fields, one page at a time, without ever seeing
a model output. This is the ground truth that every later metric is scored
against, so it is written by a person looking at the rendered page.

Sample rule (fixed before annotation starts, recorded in sample.txt):
    a seeded random sample of N applications from pages.csv, excluding any
    application with more than MAX_PAGES pages (one application carries an
    87-page document that is not a drawing set and would swallow an evening).

Per page you are asked for drawing_type, scale, floor_label and north_arrow;
for floor plans also room_count, and storeys on the ground-floor page.
Keys are single letters; Enter repeats the previous page's value where shown.
`s` skips the page (no row written), `q` quits. Rows are appended as you go,
so quitting loses nothing; rerun to continue where you stopped.

Run:  env/bin/python scripts/annotate.py                 # continue the sample
      env/bin/python scripts/annotate.py --ref 26_0025_FUL
      env/bin/python scripts/annotate.py --n 30 --seed 2026 --max-pages 30
Reads:  data/processed/pages.csv, data/processed/pages/*.png
Writes: data/annotations/sample.txt, data/annotations/annotations.csv (append)
"""
import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAGES_CSV = ROOT / "data" / "processed" / "pages.csv"
PAGES = ROOT / "data" / "processed" / "pages"
ANN_DIR = ROOT / "data" / "annotations"
ANN = ANN_DIR / "annotations.csv"
SAMPLE = ANN_DIR / "sample.txt"

TYPES = {"f": "floor_plan", "e": "elevation", "s": "section", "l": "location_plan",
         "p": "site_plan", "o": "other"}
FLOORS = {"g": "ground", "1": "first", "2": "second", "r": "roof", "-": "none", "m": "multiple"}
FIELDS = ["scale", "drawing_type", "floor_label", "north_arrow", "room_count", "storeys"]


def draw_sample(pages, n, seed, max_pages):
    counts = pages.groupby("reference").size()
    eligible = sorted(counts[counts <= max_pages].index)
    excluded = sorted(counts[counts > max_pages].index)
    chosen = sorted(random.Random(seed).sample(eligible, min(n, len(eligible))))
    SAMPLE.write_text(
        f"# annotation sample: {len(chosen)} of {len(eligible)} eligible applications, "
        f"seed {seed}, applications with more than {max_pages} pages excluded "
        f"({', '.join(excluded) or 'none'})\n" + "\n".join(chosen) + "\n")
    return chosen


def load_sample():
    return [l.strip() for l in SAMPLE.read_text().splitlines() if l.strip() and not l.startswith("#")]


def done_pages():
    if not ANN.exists():
        return set()
    df = pd.read_csv(ANN, dtype=str)
    return set(zip(df["reference"], df["page"].astype(str)))


def ask(prompt, valid=None, allow_blank=False, prev=None):
    """Read one answer. `valid` maps keys to values; None means free integer."""
    while True:
        hint = f" [{prev}]" if prev is not None else ""
        raw = input(f"  {prompt}{hint}: ").strip().lower()
        if raw in ("q", "s"):
            return raw
        if raw == "" and prev is not None:
            return prev
        if raw == "" and allow_blank:
            return ""
        if valid is None:
            if raw.isdigit():
                return raw
        elif raw in valid:
            return valid[raw]
        print("    ? " + (", ".join(f"{k}={v}" for k, v in valid.items()) if valid else "integer"))


def ask_scale():
    while True:
        raw = input("  scale (number after 1: e.g. 100 | n=nts | v=various | -=none): ").strip().lower()
        if raw in ("q", "s"):
            return raw
        if raw == "n":
            return "nts"
        if raw == "v":
            return "various"
        if raw == "-":
            return "none"
        if raw.isdigit():
            return f"1:{int(raw)}"
        print("    ? a number such as 50, 100, 1250, or n, v, -")


def write_rows(reference, page, values):
    new = not ANN.exists()
    with open(ANN, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["reference", "page", "field", "truth"])
        for field in FIELDS:
            if field in values and values[field] != "":
                w.writerow([reference, page, field, values[field]])


def annotate_page(reference, row, prev):
    png = PAGES / f"{reference}__{Path(row.file).stem}__p{row.page}.png"
    if not png.exists():
        print(f"  missing render {png.name}, skipped")
        return "s", prev
    if sys.platform == "darwin":
        subprocess.run(["open", str(png)], check=False)
    print(f"\n{reference} page {row.page} of {row.n_pages}  ({row.kind}, {png.name})")
    v = {}
    v["drawing_type"] = ask("type (f floor, e elev, s section, l location, p site, o other)", TYPES,
                            prev=prev.get("drawing_type"))
    if v["drawing_type"] in ("q", "s"):
        return v["drawing_type"], prev
    v["scale"] = ask_scale()
    if v["scale"] in ("q", "s"):
        return v["scale"], prev
    v["floor_label"] = ask("floor (g ground, 1 first, 2 second, r roof, m multiple, - none)", FLOORS,
                           prev="none" if v["drawing_type"] != "floor_plan" else None)
    if v["floor_label"] in ("q", "s"):
        return v["floor_label"], prev
    v["north_arrow"] = ask("north arrow (y/n)", {"y": "yes", "n": "no"})
    if v["north_arrow"] in ("q", "s"):
        return v["north_arrow"], prev
    if v["drawing_type"] == "floor_plan":
        v["room_count"] = ask("room count (integer, your stated rule)")
        if v["room_count"] in ("q", "s"):
            return v["room_count"], prev
        if v["floor_label"] == "ground":
            v["storeys"] = ask("storeys for the whole application (integer)", prev=prev.get("storeys"))
            if v["storeys"] in ("q", "s"):
                return v["storeys"], prev
    write_rows(reference, str(row.page), v)
    prev = {"drawing_type": v["drawing_type"], "storeys": v.get("storeys", prev.get("storeys"))}
    return "ok", prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-pages", type=int, default=30)
    ap.add_argument("--ref", help="annotate this one application instead of the sample")
    ap.add_argument("--resample", action="store_true", help="redraw sample.txt (only before annotating)")
    args = ap.parse_args()
    ANN_DIR.mkdir(exist_ok=True)
    pages = pd.read_csv(PAGES_CSV)
    pages = pages[pages.kind != "unreadable"]
    pages["n_pages"] = pages.groupby("reference")["page"].transform("size")

    if args.ref:
        refs = [args.ref]
    else:
        if args.resample or not SAMPLE.exists():
            if ANN.exists() and args.resample:
                raise SystemExit("annotations.csv exists; delete it first if you really want a new sample")
            refs = draw_sample(pages, args.n, args.seed, args.max_pages)
            print(f"sample of {len(refs)} applications written to {SAMPLE}")
        refs = load_sample()

    done = done_pages()
    todo = [(r, row) for r in refs for row in pages[pages.reference == r].itertuples()
            if (r, str(row.page)) not in done]
    total = sum((pages.reference == r).sum() for r in refs)
    print(f"{len(done)} pages annotated, {len(todo)} to go of {total} in the sample. "
          f"Keys: s skip page, q quit.")
    prev = {}
    for reference, row in todo:
        status, prev = annotate_page(reference, row, prev)
        if status == "q":
            break
    n = len(done_pages())
    print(f"\n{n} pages annotated in {ANN}")


if __name__ == "__main__":
    main()
