"""Pick candidate application references from the PlanIt API (metadata only).

PlanIt (planit.org.uk) aggregates UK planning application metadata and
returns Exeter references, proposal descriptions, application type and
state in one call. It holds no documents, so this is only for choosing
which 30 to 50 applications to look at. Householder extensions are the
sweet spot: they nearly always include existing and proposed floor plans,
elevations and a location plan.

Usage:
    env/bin/python scripts/find_references.py --start 2026-01-01 --end 2026-08-31 --n 60
Writes: data/candidates.csv (reference, app_type, app_state, start_date, description)
        data/references_to_fetch.txt (the chosen references, one per line, edit by hand)
No addresses or names are kept.
"""
import argparse
import csv
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
UA = "PlanningDrawingsFeasibility/0.1 (non-commercial research; contact: m.singh.raman@gmail.com)"
API = "https://www.planit.org.uk/api/applics/json"
HOUSEHOLDER = re.compile(r"extension|loft|dormer|garage|conversion|porch|outbuilding|annexe|roof|storey", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--n", type=int, default=60, help="how many to shortlist")
    ap.add_argument("--app-type", default="Full")
    args = ap.parse_args()

    s = requests.Session(); s.headers["User-Agent"] = UA
    records, page = [], 1
    while True:
        r = s.get(API, params={"auth": "Exeter", "start_date": args.start, "end_date": args.end,
                               "app_type": args.app_type, "pg_sz": 100, "page": page, "sort": "-start_date"}, timeout=60)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 30))); continue
        r.raise_for_status()
        j = r.json()
        batch = j.get("records", [])
        records += batch
        if len(batch) < 100 or page >= 10:
            break
        page += 1; time.sleep(3)
    print(f"{len(records)} applications from PlanIt")

    rows = []
    for rec in records:
        uid = rec.get("uid") or rec.get("name") or ""
        desc = (rec.get("description") or "").strip().replace("\n", " ")
        rows.append({"reference": uid, "app_type": rec.get("app_type", ""), "app_state": rec.get("app_state", ""),
                     "start_date": rec.get("start_date", ""), "description": desc[:160],
                     "householder_like": bool(HOUSEHOLDER.search(desc))})
    out = ROOT / "data" / "candidates.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    chosen = [r["reference"] for r in rows if r["householder_like"]][: args.n]
    (ROOT / "data" / "references_to_fetch.txt").write_text(
        "# one reference per line; edit freely before running fetch_documents.py\n" + "\n".join(chosen) + "\n")
    print(f"{sum(r['householder_like'] for r in rows)} look like householder applications; "
          f"{len(chosen)} written to data/references_to_fetch.txt; all {len(rows)} in {out}")


if __name__ == "__main__":
    main()
