# Planning drawings, honestly

A small feasibility study: how much structured building data can be
extracted from UK planning application drawings automatically, and how
often should the answer be "ask a human"?

Every floor-plan dataset in the literature is Finnish, Swiss, Asian or
clean vector CAD. UK planning application drawings are none of those
things. This study takes 30 to 50 real applications from one local
planning authority's public register and answers four questions in order:

1. **What are the documents actually like?** What fraction of pages are
   born-digital vector PDFs versus scanned raster images? That decides
   which half of the literature applies.
2. **Can the title block be read?** Scale, drawing type and floor label
   from OCR, with a confidence per field.
3. **Can a pretrained floor-plan model do anything useful on them?** A
   U-Net baseline for walls and rooms, scored against hand annotation.
4. **When should the system abstain?** A reliability diagram, Expected
   Calibration Error, and a selective-prediction curve with a threshold
   below which a field is routed to human review, because anyone pricing
   a decision off this output needs to know how often it is wrong.

The evaluation protocol was fixed before any model ran
(`scripts/evaluate_fields.py`, `scripts/calibration.py`), so the metrics
could not be chosen after seeing the results.

## Status

Started 5 September 2026. Results, failure cases and limitations are
written here as each stage completes.

### 1. What are the documents actually like? (9 September 2026)

44 householder-scale full applications from Exeter City Council's register,
January to August 2026, chosen from public metadata before any document was
seen (`data/references_to_fetch.txt`). The fetch collected 322 current
drawings (all listed drawings were retrieved, none refused) totalling
470 pages. `scripts/audit_pages.py` classified every page from its text
layer, vector drawing objects and raster image coverage
(`outputs/metrics/page_audit.json`, per page in `data/processed/pages.csv`):

| Page kind | Pages | Share | What it means |
|---|---|---|---|
| born-digital | 420 | 89.4% | vector content and a text layer, no dominant raster image |
| mixed | 35 | 7.4% | a raster image covers the page but a text layer or vector objects sit on it (a scan placed in a CAD sheet, or an OCR layer) |
| scanned | 15 | 3.2% | raster only, no text layer, no vector objects |

By application, 37 of 44 are born-digital throughout, 4 are mostly mixed
and 3 are mostly scanned. Sheets are A3 (214 pages), A4 (162), A1 (70),
A2 (4) and 20 non-standard sizes. 316 of the 322 files are single pages;
one file has 87. The median page carries about 940 characters of text and
about 430 vector drawing objects; 387 of 470 pages carry at least 200
characters, so the title block is readable from the text layer without OCR
on roughly four pages in five. Of the 15 scanned pages, 6 are location plans.

What this decides: the raster floor-plan literature (CubiCasa5K and its
successors) describes at most one page in ten here. For the rest, the
problem is parsing a vector PDF and its text layer, and OCR is the fallback
for a small tail rather than the main route. The next stage (title-block
fields with a confidence each) therefore reads the text layer first and
sends only image-dominant pages to docTR.

Caveat: "born-digital" means the page is not a raster image; it does not
mean the drawing's geometry is clean vector CAD. Some CAD exports embed a
raster of the plan under a vector title block; those fall in "mixed".

## Method and layout

| Stage | File | What it does |
|---|---|---|
| choose | `scripts/find_references.py` | shortlist householder applications from PlanIt metadata |
| fetch | `scripts/fetch_documents.py` | download the drawings for each reference, politely, once |
| audit | `scripts/audit_pages.py` | classify every page vector / scanned / mixed; render to PNG |
| annotate | `scripts/annotate.py` | hand truth per page for a seeded sample of applications, before any model output |
| read | `src/ocr_titleblock.py` | PDF text layer first, docTR only as fallback; scale, drawing type, floor label, with a confidence each |
| see | `src/segment_floorplan.py` | pretrained U-Net (ResNet-34) walls/rooms baseline on MPS |
| score | `scripts/evaluate_fields.py` | hit rate per field against hand annotations |
| trust | `scripts/calibration.py` | reliability diagram, ECE, abstention threshold |

Annotation format: `data/annotations/README.md`.

## Data and rights

This study analyses planning application drawings published on Exeter City
Council's online planning register, which the council makes available
under article 40 of the Town and Country Planning (Development Management
Procedure) (England) Order 2015. Copyright in each drawing remains with its
author or their employer, normally the architect or agent, not the council
and not this project. The council's notice states that plans, drawings and
material submitted are protected by the Copyright, Designs and Patents Act
1988 (section 47), may only be used for matters relating to the specific
application, and that further copies must not be made without the
copyright owner's permission.

Accordingly: **no drawing, page image, crop, thumbnail or full-resolution
mask is redistributed in this repository**, and none will be. The
downloaded documents are held privately for the duration of the study and
used only to answer the questions above, which is non-commercial research
within CDPA 1988 sections 29 and 29A. What is published is the code, the
list of application references with their register URLs (so anyone can
re-obtain the same documents under their own lawful basis), my own
field-level annotations, the extracted fields, and the metrics. Personal
data that appears on drawings (applicant, agent and architect names,
addresses, signatures) is stripped before anything leaves my machine;
applications are identified only by their public council reference.
Document titles as listed on the register are not recorded either, because
in practice they often contain the applicant's name or the property
address; each document is logged by an index and a kind derived from its
title (for example `proposed_elevation`), and the local files are named
the same way.

The segmentation baseline uses weights trained on CubiCasa5K, which is
licensed CC BY-NC-SA 4.0; that lineage is one more reason this study is
non-commercial. An organisation wishing to run this pipeline on planning
documents for a commercial purpose would need its own rights basis for
both the documents and the model. This section is a statement of practice,
not legal advice. Sources: CDPA 1988 s.29, s.29A, s.47; DMPO 2015 art. 40;
Exeter City Council, "Copyright, disclaimer and personal data notice",
https://exeter.gov.uk/planning-services/planning-applications/find-and-comment-on-a-planning-application/copyright-disclaimer-and-personal-data-notice/
(last updated by the council 10 July 2025, accessed 5 September 2026).

## Run

    python3.13 -m venv env && env/bin/pip install -r requirements.txt
    env/bin/python scripts/find_references.py --start 2026-01-01 --end 2026-08-31 --n 50
    env/bin/python scripts/fetch_documents.py --from data/references_to_fetch.txt --dry-run
    env/bin/python scripts/fetch_documents.py --from data/references_to_fetch.txt
    env/bin/python scripts/audit_pages.py
    env/bin/python scripts/annotate.py
    env/bin/python -m src.ocr_titleblock
    env/bin/python -m src.segment_floorplan
    env/bin/python scripts/evaluate_fields.py
    env/bin/python scripts/calibration.py --threshold 0.8

## Licence

Code: MIT. Model weights and source documents are not part of this
repository and carry their own terms (see above).
