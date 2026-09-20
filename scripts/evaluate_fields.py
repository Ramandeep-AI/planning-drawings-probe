"""Hit rate of extracted fields against hand annotations.

Joins the pipeline's extractions to the hand annotations and
reports, per field type, how often the extraction matched, how often it
was blank, and the confusion between the two. Also writes the merged
predictions file that scripts/calibration.py consumes.

Inputs:
    data/processed/extractions.csv   reference,sheet,field,predicted,confidence
    data/annotations/annotations.csv reference,sheet,field,truth
(sheet = two-digit document index + page number, e.g. 03-1)
Field names must match between the two files (see README field list).

Run:  env/bin/python scripts/evaluate_fields.py
Outputs: data/processed/predictions.csv, outputs/metrics/field_hit_rate.json
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "processed" / "extractions.csv"
ANN = ROOT / "data" / "annotations" / "annotations.csv"
OUT = ROOT / "data" / "processed" / "predictions.csv"
MET = ROOT / "outputs" / "metrics" / "field_hit_rate.json"


def norm(s):
    return str(s).strip().lower()


def main():
    ext = pd.read_csv(EXT, dtype=str, keep_default_na=False)
    ann = pd.read_csv(ANN, dtype=str, keep_default_na=False)
    key = ["reference", "sheet", "field"]
    # Only fields the extractor predicts are scored. Fields that are annotated
    # for a later stage (north_arrow, room_count, storeys) are reported as not
    # scored, never counted as misses.
    predicted_fields = sorted(ext["field"].unique())
    not_scored = ann[~ann["field"].isin(predicted_fields)]["field"].value_counts().to_dict()
    ann = ann[ann["field"].isin(predicted_fields)]
    merged = ann.merge(ext, on=key, how="left")
    merged["predicted"] = merged["predicted"].fillna("")
    # A sheet with no prediction at all is a blank: a miss at confidence 0.
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)
    if not_scored:
        print(f"annotated but not scored at this stage: {not_scored}")
    merged["hit"] = merged.apply(lambda r: norm(r.predicted) == norm(r.truth), axis=1)
    merged["blank"] = merged["predicted"].str.strip() == ""

    rep = (merged.groupby("field")
                 .agg(n=("hit", "size"), hit_rate=("hit", "mean"), blank_rate=("blank", "mean"))
                 .round(3))
    print(rep.to_string())
    print(f"\noverall hit rate {merged['hit'].mean():.3f} over {len(merged)} annotated fields "
          f"({merged['blank'].mean():.1%} blank)")

    by_source = None
    if "source" in merged.columns:
        merged["source"] = merged["source"].fillna("none").replace("", "none")
        by_source = (merged.groupby(["source", "field"])
                           .agg(n=("hit", "size"), hit_rate=("hit", "mean"))
                           .round(3))
        print("\nby source (text layer versus OCR):")
        print(by_source.to_string())

    merged[key + ["predicted", "truth", "confidence"]].to_csv(OUT, index=False)
    MET.parent.mkdir(parents=True, exist_ok=True)
    MET.write_text(json.dumps({
        "n_annotated": int(len(merged)),
        "fields_scored": predicted_fields,
        "annotated_not_scored": not_scored,
        "overall_hit_rate": round(float(merged["hit"].mean()), 4),
        "per_field": rep.reset_index().to_dict(orient="records"),
        "by_source": by_source.reset_index().to_dict(orient="records") if by_source is not None else None,
    }, indent=2))
    print(f"wrote {OUT} and {MET}")


if __name__ == "__main__":
    main()
