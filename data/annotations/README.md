# Annotations

Ground truth is a plain CSV, one row per field per page, written by hand
while looking at each rendered page and before looking at any model output
for that page. No boxes or polygons are needed for this study; field-level
truth is enough to score every stage.

Helper: `env/bin/python scripts/annotate.py` opens each page of the sample
in turn and asks for the fields with single-key answers, appending rows as
it goes. It never shows a prediction.

Sample: `sample.txt` lists the applications annotated. It is a seeded random
sample of 30 of the 42 eligible applications, drawn before annotation
started. The two applications with more than 30 pages are excluded (one
carries an 87-page ground investigation report filed under drawings).
Every page of a sampled application is annotated unless it is not a
drawing at all (a product brochure or report text filed under drawings).
Those pages are listed in `skipped.csv`, have no annotation rows, and are
excluded from every rate.

File: `annotations.csv`

    reference,sheet,field,truth
    26_0025_FUL,02-1,scale,1:50
    26_0025_FUL,02-1,drawing_type,floor_plan
    26_0025_FUL,02-1,floor_label,ground
    26_0025_FUL,02-1,north_arrow,yes
    26_0025_FUL,02-1,room_count,5
    26_0025_FUL,02-1,storeys,2

References use the file-system form (`26_0025_FUL`), the same as
`data/processed/pages.csv` and `extractions.csv`. `sheet` is the two-digit
document index from the fetch log plus the page number within that
document (`02-1`), which is unique within an application; most documents
are single-page PDFs, so the page number alone would not be.

Fields and allowed values:
- scale: the scale stated in the title block, as printed, e.g. 1:50, 1:100,
  1:200, 1:500, 1:1250, 1:2500; `various` if the title block says "as shown",
  "various" or the sheet carries several drawings at different scales with no
  single title-block scale; `nts` if not to scale; `none` if no scale is stated
- drawing_type: floor_plan | elevation | section | location_plan | site_plan | other;
  the type named in the title block, and for a sheet with several drawings
  the one the title names first (a roof plan is a floor_plan with floor_label roof)
- floor_label: ground | first | second | roof | none | multiple;
  `multiple` when the sheet is titled with two or more floors
- north_arrow: yes | no
- room_count: integer, rooms on that floor plan page as a person would count
  them, by the rule stated in the main README; floor plans only
- storeys: integer, for the whole application, written on the ground-floor page only

Rules: annotate before looking at any model output for that page. No
names, addresses or free text are kept here; working notes live in a
private file outside the repo.
