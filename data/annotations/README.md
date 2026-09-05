# Annotations

Ground truth is a plain CSV, one row per field per page, written by hand
while looking at each rendered page. No boxes or polygons are needed for
this study; field-level truth is enough to score every stage.

File: `annotations.csv`

    reference,page,field,truth
    26/1318/FUL,3,scale,1:100
    26/1318/FUL,3,drawing_type,floor_plan
    26/1318/FUL,3,floor_label,ground
    26/1318/FUL,3,north_arrow,yes
    26/1318/FUL,3,room_count,5
    26/1318/FUL,3,storeys,2

Fields and allowed values:
- scale: as printed, e.g. 1:50, 1:100, 1:200, 1:500, 1:1250, 1:2500; `nts` if not to scale; `none` if absent
- drawing_type: floor_plan | elevation | section | location_plan | site_plan | other
- floor_label: ground | first | second | roof | none
- north_arrow: yes | no
- room_count: integer, rooms on that floor plan page as a person would count them (state your rule in README)
- storeys: integer, for the whole application, written on the ground-floor page

Rules: annotate before looking at any model output for that page. Annotate
at least 20 applications, ideally 30. Keep no names, addresses or free text
here; if you need notes, keep them in a private file outside the repo.
