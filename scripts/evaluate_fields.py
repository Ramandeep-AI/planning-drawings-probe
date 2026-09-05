"""Hit rate of extracted fields against hand annotations.

Joins the pipeline's extractions to your ground-truth annotations and
reports, per field type, how often the extraction matched, how often it
was blank, and the confusion between the two. Also writes the merged
predictions file that scripts/calibration.py consumes.

Inputs:
    data/processed/extractions.csv   reference,page,field,predicted,confidence
    data/annotations/annotations.csv reference,page,field,truth
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
    key = ["reference", "page", "field"]
    merged = ann.merge(ext, on=key, how="left")
    merged["predicted"] = merged["predicted"].fillna("")
    merged["confidence"] = merged["confidence"].replace("", "0").astype(float)
    merged["hit"] = merged.apply(lambda r: norm(r.predicted) == norm(r.truth), axis=1)
    merged["blank"] = merged["predicted"].str.strip() == ""

    rep = (merged.groupby("field")
                 .agg(n=("hit", "size"), hit_rate=("hit", "mean"), blank_rate=("blank", "mean"))
                 .round(3))
    print(rep.to_string())
    print(f"\noverall hit rate {merged['hit'].mean():.3f} over {len(merged)} annotated fields "
          f"({merged['blank'].mean():.1%} blank)")

    merged[key + ["predicted", "truth", "confidence"]].to_csv(OUT, index=False)
    MET.parent.mkdir(parents=True, exist_ok=True)
    MET.write_text(json.dumps({
        "n_annotated": int(len(merged)),
        "overall_hit_rate": round(float(merged["hit"].mean()), 4),
        "per_field": rep.reset_index().to_dict(orient="records"),
    }, indent=2))
    print(f"wrote {OUT} and {MET}")


if __name__ == "__main__":
    main()
