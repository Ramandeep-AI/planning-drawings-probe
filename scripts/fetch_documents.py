"""Politely download the drawings for a short list of Exeter planning
applications, for a non-commercial feasibility study.

How Exeter publishes documents (verified 5 Sept 2026): the Idox register at
publicaccess.exeter.gov.uk does NOT serve documents (and its robots.txt
disallows crawling, so this script never touches it). Each application's
"External Documents" tab points to a council page,
    https://exeter.gov.uk/planning-services/permissions-and-applications/related-documents/?appref=YY%2FNNNN
which renders a category tree ("Application Details", "Drawings and Plans",
"Decision", ...) whose links open direct PDF URLs on
planningdocs.exeter.gov.uk. One HTML page per application plus one GET per
drawing is all this needs.

Manners built in: a named User-Agent with a contact address, one request at
a time, 3 to 5 seconds of jittered sleep between requests, retries only on
timeouts / 429 / 5xx with backoff, nothing re-downloaded that already
exists. Forty applications is roughly 25 minutes. Run it once, off-peak.
Before the first run, send the courtesy note in PLAN.md to Planning Services.

Rights: the downloaded PDFs stay in data/raw/ (gitignored) and are used
only for this study. See README, "Data and rights".

Personal data: document titles as listed on the register frequently contain
the applicant's name or the property address (e.g. "J SMITH PROPOSED
ELEVATIONS", "12 Some Road - Site Plan"). Listed titles are therefore never
written to the committed log or used as file names. Each document gets an
index and a kind derived from its title (proposed_elevation, location_plan,
...); the verbatim title is kept only in a gitignored side file so the same
document can be re-identified locally.

Usage:
    env/bin/python scripts/fetch_documents.py 26/1318/FUL 26/0954/FUL ...
    env/bin/python scripts/fetch_documents.py --from data/references_to_fetch.txt
    add --dry-run to list what would be downloaded (and what was skipped)
    without fetching any PDF; dry-run logs go to *_dryrun.csv (gitignored)
Writes: data/raw/<ref>/<NN>_<kind>.pdf            local only (gitignored)
        data/references.csv                        committed: reference,
            register_url, accessed_on, doc_index, doc_kind, category, bytes, status
        data/references_titles.csv                 gitignored: reference,
            doc_index, document_title_as_listed, file
"""
import argparse
import csv
import datetime as dt
import random
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LOG = ROOT / "data" / "references.csv"
TITLES = ROOT / "data" / "references_titles.csv"

UA = "PlanningDrawingsFeasibility/0.1 (non-commercial research; contact: m.singh.raman@gmail.com)"
DOCS_PAGE = "https://exeter.gov.uk/planning-services/permissions-and-applications/related-documents/?appref={ref}"
KEEP_CATEGORIES = {"Drawings and Plans", "Application Details"}
DRAWING_WORDS = re.compile(r"plan|elevation|section|location|site|block|floor|layout|existing|proposed|design|house|drawing|arrangement|detail|survey|\bgf\b|\bff\b", re.I)
EXCLUDE_WORDS = re.compile(r"application form|statement|design state|design (and|&) access|certificate|\bcil\b|checklist|heritage|ecolog|arboric|flood|photo|questionnaire|ownership|notice|fee|cover letter|planning cover|drawing register|drawing schedule|schedule of|consultation|response|comments?\b|letter|\bmap\s+\d{4}|report", re.I)
SLEEP = (3.0, 5.0)

# Derived document kinds. Order matters: the first matching base wins.
KIND_BASES = [
    ("location_plan", re.compile(r"location|\bos\b|ordnance", re.I)),
    ("block_plan", re.compile(r"block", re.I)),
    ("site_plan", re.compile(r"site\s*(plan|layout)|siteplan", re.I)),
    ("roof_plan", re.compile(r"roof\s*plan", re.I)),
    ("section", re.compile(r"section", re.I)),
    ("elevation", re.compile(r"elevation", re.I)),
    ("floor_plan", re.compile(r"floor|\bplans?\b|layout|arrangement|\bgf\b|\bff\b|ground|first|second", re.I)),
    ("details", re.compile(r"detail", re.I)),
    ("survey", re.compile(r"survey", re.I)),
]
PREFIX = re.compile(r"(existing|proposed)", re.I)
COMBINABLE = ("floor_plan", "elevation", "section")


def doc_kind(title):
    """Label a listed title without keeping any of its words. Unknown titles
    become 'drawing_other' rather than leaking through. A sheet whose title
    names two or more of plans, elevations and sections is 'combined'."""
    hits = [k for k, rx in KIND_BASES if rx.search(title)]
    base = hits[0] if hits else "drawing_other"
    if sum(k in COMBINABLE for k in hits) >= 2:
        base = "combined"
    if base in ("location_plan", "survey", "details"):
        return base
    prefixes = sorted({p.lower() for p in PREFIX.findall(title)})
    if prefixes == ["existing", "proposed"]:
        return f"existing_proposed_{base}"
    if len(prefixes) == 1:
        return f"{prefixes[0]}_{base}"
    return base


class Tree(HTMLParser):
    """Walk <div class="tree"> and pair each document link with its category."""

    def __init__(self):
        super().__init__()
        self.in_tree = False
        self.depth = 0
        self.category = None
        self.docs = []            # (category, title, url)
        self._pending_url = None
        self._pending_title = None
        self._in_cat_anchor = False
        self._cat_text = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div" and "tree" in (a.get("class") or ""):
            self.in_tree = True
        if not self.in_tree:
            return
        if tag == "ul":
            self.depth += 1
        if tag == "a":
            m = re.search(r"window\.open\('(https://planningdocs\.exeter\.gov\.uk/servlets/direct/[^']+)'", a.get("onclick") or "")
            if m:
                self._pending_url = m.group(1)
                self._pending_title = a.get("title") or ""
            elif self.depth == 1:
                self._in_cat_anchor = True
                self._cat_text = ""

    def handle_data(self, data):
        if self._in_cat_anchor:
            self._cat_text += data

    def handle_endtag(self, tag):
        if not self.in_tree:
            return
        if tag == "a":
            if self._in_cat_anchor:
                self.category = self._cat_text.strip()
                self._in_cat_anchor = False
            if self._pending_url:
                self.docs.append((self.category, self._pending_title.strip(), self._pending_url))
                self._pending_url = None
        if tag == "ul":
            self.depth -= 1
            if self.depth == 0:
                self.in_tree = False


def polite_get(session, url, stream=False, tries=3):
    delays = (10, 30, 90)
    for i in range(tries):
        try:
            r = session.get(url, timeout=60, stream=stream)
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            print(f"    retry after error: {e}"); time.sleep(delays[i]); continue
        if r.status_code in (429, 500, 502, 503, 504):
            wait = int(r.headers.get("Retry-After", delays[i]))
            print(f"    {r.status_code}, waiting {wait}s"); time.sleep(wait); continue
        return r
    return r


def is_drawing(category, title):
    """'Drawings and Plans' is the council's own classification, so everything
    there is kept unless it is clearly not a drawing (titles are often bare
    drawing numbers like '2603-P1_A'). 'Application Details' mixes drawings
    with forms and reports, so there a title must look like a drawing.
    'Superseded' is never fetched: only the current set is studied."""
    if category not in KEEP_CATEGORIES:
        return False
    if EXCLUDE_WORDS.search(title):
        return False
    if category == "Drawings and Plans":
        return True
    return bool(DRAWING_WORDS.search(title))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="*", help="application references like 26/1318/FUL")
    ap.add_argument("--from", dest="from_file", help="file with one reference per line")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-docs", type=int, default=20, help="per application (a householder application has 10 to 16 drawings)")
    args = ap.parse_args()
    refs = list(args.refs)
    if args.from_file:
        refs += [l.strip() for l in Path(args.from_file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not refs:
        sys.exit("give references on the command line or via --from")

    session = requests.Session()
    session.headers["User-Agent"] = UA
    suffix = "_dryrun" if args.dry_run else ""          # keep dry-run rows out of the committed log
    log_path = LOG.with_name(f"references{suffix}.csv")
    titles_path = TITLES.with_name(f"references_titles{suffix}.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log, new_titles = not log_path.exists(), not titles_path.exists()
    logf = open(log_path, "a", newline="")
    titf = open(titles_path, "a", newline="")
    w, wt = csv.writer(logf), csv.writer(titf)
    if new_log:
        w.writerow(["reference", "register_url", "accessed_on", "doc_index", "doc_kind", "category", "bytes", "status"])
    if new_titles:
        wt.writerow(["reference", "doc_index", "document_title_as_listed", "file"])
    today = dt.date.today().isoformat()

    for ref in refs:
        ref = ref.strip().upper()
        core = "/".join(ref.split("/")[:2])                    # the portal ignores the /TYPE suffix
        page_url = DOCS_PAGE.format(ref=core.replace("/", "%2F"))
        print(f"\n{ref}")
        r = polite_get(session, page_url)
        time.sleep(random.uniform(*SLEEP))
        if r.status_code != 200:
            print(f"  page {r.status_code}"); w.writerow([ref, page_url, today, 0, "", "", 0, f"page {r.status_code}"]); continue
        tree = Tree(); tree.feed(r.text)
        if not tree.docs:
            print("  no documents listed (unknown reference?)"); w.writerow([ref, page_url, today, 0, "", "", 0, "no documents"]); continue
        wanted = [(c, t, u) for c, t, u in tree.docs if is_drawing(c, t)][: args.max_docs]
        skipped = [(c, t) for c, t, u in tree.docs if not is_drawing(c, t)]
        print(f"  {len(tree.docs)} documents listed, {len(wanted)} look like drawings, {len(skipped)} skipped")
        if args.dry_run:
            for cat, title in skipped:
                print(f"  - skip  [{cat}] {title}")
        folder = RAW / ref.replace("/", "_")
        for idx, (cat, title, url) in enumerate(wanted, start=1):
            kind = doc_kind(title)
            fname = f"{idx:02d}_{kind}.pdf"
            dest = folder / fname
            wt.writerow([ref, idx, title, fname])
            if dest.exists():
                print(f"  = {fname} (already here)"); continue
            if args.dry_run:
                print(f"  ~ keep  [{cat}] {title}  ->  {fname}")
                w.writerow([ref, page_url, today, idx, kind, cat, 0, "dry-run"]); continue
            resp = polite_get(session, url, stream=True)
            time.sleep(random.uniform(*SLEEP))
            if resp.status_code == 403:
                sys.exit("  403 from planningdocs: check the User-Agent and stop for today")
            if resp.status_code == 401:
                print("  401 token rejected; re-run later so the page is re-parsed"); w.writerow([ref, page_url, today, idx, kind, cat, 0, "401"]); continue
            if resp.status_code != 200:
                print(f"  {resp.status_code} for document {idx}"); w.writerow([ref, page_url, today, idx, kind, cat, 0, str(resp.status_code)]); continue
            data = resp.content
            if not data.startswith(b"%PDF"):
                print(f"  not a PDF: document {idx}"); w.writerow([ref, page_url, today, idx, kind, cat, len(data), "not-pdf"]); continue
            folder.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(f"  + {fname} ({len(data)//1024} KB)")
            w.writerow([ref, page_url, today, idx, kind, cat, len(data), "ok"])
            logf.flush(); titf.flush()
    logf.close(); titf.close()
    print(f"\nlog: {log_path}\ntitles (gitignored): {titles_path}")


if __name__ == "__main__":
    main()
