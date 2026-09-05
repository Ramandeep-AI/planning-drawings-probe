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
exists. Fifty applications is roughly 25 minutes. Run it once, off-peak.
Before the first run, send the courtesy note in PLAN.md to Planning Services.

Rights: the downloaded PDFs stay in data/raw/ (gitignored) and are used
only for this study. See README, "Data and rights".

Usage:
    env/bin/python scripts/fetch_documents.py 26/1318/FUL 26/0954/FUL ...
    env/bin/python scripts/fetch_documents.py --from data/references_to_fetch.txt
    add --dry-run to list what would be downloaded without fetching PDFs
Writes: data/raw/<ref>/<title>.pdf and data/references.csv (no addresses,
no names: reference, register URL, access date, listed title, category).
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

UA = "PlanningDrawingsFeasibility/0.1 (non-commercial research; contact: m.singh.raman@gmail.com)"
DOCS_PAGE = "https://exeter.gov.uk/planning-services/permissions-and-applications/related-documents/?appref={ref}"
KEEP_CATEGORIES = {"Drawings and Plans", "Application Details"}
DRAWING_WORDS = re.compile(r"plan|elevation|section|location|site|block|floor|layout|existing|proposed|design|house|drawing", re.I)
EXCLUDE_WORDS = re.compile(r"application form|statement|certificate|cil|checklist|heritage|ecolog|arboric|flood|photo|questionnaire|ownership|notice|fee", re.I)
SLEEP = (3.0, 5.0)


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


def safe_name(title):
    title = re.sub(r"^\d{2}/\d{2}/\d{4}_", "", title)          # strip the date prefix
    title = re.sub(r"[^A-Za-z0-9._ -]+", "_", title).strip("_ ")
    return (title or "document")[:80]


def is_drawing(category, title):
    if category not in KEEP_CATEGORIES:
        return False
    if EXCLUDE_WORDS.search(title):
        return False
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
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new_log = not LOG.exists()
    logf = open(LOG, "a", newline="")
    w = csv.writer(logf)
    if new_log:
        w.writerow(["reference", "register_url", "accessed_on", "document_title_as_listed", "category", "file", "bytes", "status"])
    today = dt.date.today().isoformat()

    for ref in refs:
        ref = ref.strip().upper()
        core = "/".join(ref.split("/")[:2])                    # the portal ignores the /TYPE suffix
        page_url = DOCS_PAGE.format(ref=core.replace("/", "%2F"))
        print(f"\n{ref}")
        r = polite_get(session, page_url)
        time.sleep(random.uniform(*SLEEP))
        if r.status_code != 200:
            print(f"  page {r.status_code}"); w.writerow([ref, page_url, today, "", "", "", 0, f"page {r.status_code}"]); continue
        tree = Tree(); tree.feed(r.text)
        if not tree.docs:
            print("  no documents listed (unknown reference?)"); w.writerow([ref, page_url, today, "", "", "", 0, "no documents"]); continue
        wanted = [(c, t, u) for c, t, u in tree.docs if is_drawing(c, t)][: args.max_docs]
        skipped = len(tree.docs) - len(wanted)
        print(f"  {len(tree.docs)} documents listed, {len(wanted)} look like drawings, {skipped} skipped")
        folder = RAW / ref.replace("/", "_")
        for cat, title, url in wanted:
            fname = safe_name(title) + ".pdf"
            dest = folder / fname
            if dest.exists():
                print(f"  = {fname} (already here)"); continue
            if args.dry_run:
                print(f"  ~ {cat}: {title}"); w.writerow([ref, page_url, today, title, cat, fname, 0, "dry-run"]); continue
            resp = polite_get(session, url, stream=True)
            time.sleep(random.uniform(*SLEEP))
            if resp.status_code == 403:
                sys.exit("  403 from planningdocs: check the User-Agent and stop for today")
            if resp.status_code == 401:
                print("  401 token rejected; re-run later so the page is re-parsed"); w.writerow([ref, page_url, today, title, cat, fname, 0, "401"]); continue
            if resp.status_code != 200:
                print(f"  {resp.status_code} for {title}"); w.writerow([ref, page_url, today, title, cat, fname, 0, str(resp.status_code)]); continue
            data = resp.content
            if not data.startswith(b"%PDF"):
                print(f"  not a PDF: {title}"); w.writerow([ref, page_url, today, title, cat, fname, len(data), "not-pdf"]); continue
            folder.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(f"  + {fname} ({len(data)//1024} KB)")
            w.writerow([ref, page_url, today, title, cat, fname, len(data), "ok"])
            logf.flush()
    logf.close()
    print(f"\nlog: {LOG}")


if __name__ == "__main__":
    main()
