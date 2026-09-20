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
`x` skips the page (no row written), `q` quits. Rows are appended as you go,
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
import shutil
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
         "p": "site_plan", "o": "other",
         # full words also accepted, because l and 1 look alike in a terminal
         "floor": "floor_plan", "elev": "elevation", "elevation": "elevation", "section": "section",
         "loc": "location_plan", "location": "location_plan", "site": "site_plan", "other": "other"}
FLOORS = {"g": "ground", "1": "first", "2": "second", "r": "roof", "-": "none", "m": "multiple",
          "ground": "ground", "first": "first", "second": "second", "roof": "roof", "none": "none",
          "multiple": "multiple"}
PRIMARY_KEYS = {"f", "e", "s", "l", "p", "o", "g", "1", "2", "r", "-", "m"}
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


def sheet_id(file, page):
    """Sheet identifier unique within an application: the two-digit document
    index from the file name plus the page number, e.g. 03-1. Most files here
    are single-page PDFs, so the page number alone is not unique; the file
    name is not used because it carries the register-derived kind."""
    return f"{file[:2]}-{page}"


def done_pages():
    if not ANN.exists():
        return set()
    df = pd.read_csv(ANN, dtype=str)
    return set(zip(df["reference"], df["sheet"]))


def ask(prompt, valid=None, allow_blank=False, prev=None):
    """Read one answer. `valid` maps keys to values; None means free integer."""
    while True:
        hint = f" [{prev}]" if prev is not None else ""
        raw = input(f"  {prompt}{hint}: ").strip().lower()
        if raw in ("q", "x"):
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
        if valid:
            print("    ? " + ", ".join(f"{k}={v}" for k, v in valid.items() if k in PRIMARY_KEYS)
                  + "  (or type the word, e.g. location)")
        else:
            print("    ? integer")


def ask_scale():
    while True:
        raw = input("  scale (number after 1: e.g. 100 | n=nts | v=various | -=none): ").strip().lower()
        if raw in ("q", "x"):
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


def parse_scale(tok):
    if tok == "n":
        return "nts"
    if tok == "v":
        return "various"
    if tok == "-":
        return "none"
    if tok.isdigit():
        return f"1:{int(tok)}"
    return None


def parse_quick(tokens, prev):
    """One-line entry: TYPE SCALE ARROW for anything but a floor plan, or
    TYPE SCALE FLOOR ARROW ROOMS [STOREYS] for a floor plan. Example lines:
    'e 100 n', 'L v y', 'f 100 g y 5 2', 'f 50 1 y 4'. Returns the values
    dict, or an error string."""
    if len(tokens) < 3:
        return "need at least: type scale arrow"
    dtype = TYPES.get(tokens[0])
    if dtype is None:
        return f"unknown type '{tokens[0]}'"
    scale = parse_scale(tokens[1])
    if scale is None:
        return f"unknown scale '{tokens[1]}'"
    v = {"drawing_type": dtype, "scale": scale}
    if dtype != "floor_plan":
        arrow = {"y": "yes", "n": "no"}.get(tokens[2])
        if arrow is None or len(tokens) != 3:
            return "for this type the line is: type scale arrow"
        v["floor_label"], v["north_arrow"] = "none", arrow
        return v
    if len(tokens) < 5:
        return "for a floor plan the line is: type scale floor arrow rooms [storeys]"
    floor = FLOORS.get(tokens[2])
    arrow = {"y": "yes", "n": "no"}.get(tokens[3])
    if floor is None or arrow is None or not tokens[4].isdigit():
        return "for a floor plan the line is: type scale floor arrow rooms [storeys]"
    v["floor_label"], v["north_arrow"], v["room_count"] = floor, arrow, tokens[4]
    if floor == "ground":
        if len(tokens) >= 6 and tokens[5].isdigit():
            v["storeys"] = tokens[5]
        elif prev.get("storeys"):
            v["storeys"] = prev["storeys"]
        else:
            return "a ground-floor line needs storeys as the sixth item"
    elif len(tokens) > 5:
        return "storeys are only entered on a ground-floor sheet"
    return v


def write_rows(reference, sheet, values):
    new = not ANN.exists()
    with open(ANN, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["reference", "sheet", "field", "truth"])
        for field in FIELDS:
            if field in values and values[field] != "":
                w.writerow([reference, sheet, field, values[field]])


def annotate_page(reference, row, prev):
    png = PAGES / f"{reference}__{Path(row.file).stem}__p{row.page}.png"
    if not png.exists():
        print(f"  missing render for {reference} page {row.page}, skipped")
        return "s", prev
    # The file name carries the document kind derived from the register's
    # listed title. The annotator must judge the page itself, so the page is
    # shown under a neutral name and the kind is never printed.
    view = PAGES / "_current_page.png"
    shutil.copyfile(png, view)
    if sys.platform == "darwin":
        subprocess.run(["open", str(view)], check=False)
    print(f"\n{reference} sheet {row.seq} of {row.n_pages}")
    # Quick entry: a whole line such as 'e 100 n' or 'f 100 g y 5 2'.
    # A single key falls through to the step-by-step questions.
    while True:
        first = input("  quick line, or type key (f e s L p o): ").strip().lower()
        if first in ("q", "x"):
            return first, prev
        if " " not in first:
            break
        v = parse_quick(first.split(), prev)
        if isinstance(v, dict):
            write_rows(reference, row.sheet, v)
            prev = {"drawing_type": v["drawing_type"], "storeys": v.get("storeys", prev.get("storeys"))}
            return "ok", prev
        print(f"    ? {v}")
    v = {}
    v["drawing_type"] = TYPES.get(first) or ask(
        "type (f floor, e elev, s section, L location [letter L], p site, o other)", TYPES,
        prev=prev.get("drawing_type"))
    if v["drawing_type"] in ("q", "x"):
        return v["drawing_type"], prev
    v["scale"] = ask_scale()
    if v["scale"] in ("q", "x"):
        return v["scale"], prev
    v["floor_label"] = ask("floor (g ground, 1 first, 2 second, r roof, m multiple, - none)", FLOORS,
                           prev="none" if v["drawing_type"] != "floor_plan" else None)
    if v["floor_label"] in ("q", "x"):
        return v["floor_label"], prev
    v["north_arrow"] = ask("north arrow (y/n)", {"y": "yes", "n": "no"})
    if v["north_arrow"] in ("q", "x"):
        return v["north_arrow"], prev
    if v["drawing_type"] == "floor_plan":
        v["room_count"] = ask("room count (integer, rule in README)")
        if v["room_count"] in ("q", "x"):
            return v["room_count"], prev
        if v["floor_label"] == "ground":
            v["storeys"] = ask("storeys for the whole application (integer)", prev=prev.get("storeys"))
            if v["storeys"] in ("q", "x"):
                return v["storeys"], prev
    write_rows(reference, row.sheet, v)
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
    pages["seq"] = pages.groupby("reference").cumcount() + 1   # sheet number within the application
    pages["sheet"] = [sheet_id(f, p) for f, p in zip(pages["file"], pages["page"])]

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
            if (r, row.sheet) not in done]
    total = sum((pages.reference == r).sum() for r in refs)
    print(f"{len(done)} pages annotated, {len(todo)} to go of {total} in the sample. "
          f"Keys: x skip page, q quit.")
    prev = {}
    for reference, row in todo:
        status, prev = annotate_page(reference, row, prev)
        if status == "q":
            break
    n = len(done_pages())
    print(f"\n{n} pages annotated in {ANN}")


if __name__ == "__main__":
    main()
