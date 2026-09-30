"""Task 7A Part I — exclusive failure attribution for the natural-language L3 records.

Section 19 assigns exactly one bucket per Z-MiniVal240 record in this fixed priority order:

```text
1 PARSER_WRONG
2 REFERENCE_NO_PROPOSALS
3 REFERENCE_NO_ELIGIBLE
4 REFERENCE_NOT_COVERED_IOU50
5 REFERENCE_SELECTION_WRONG
6 REFERENCE_GEOMETRY_POOR          (selected reference IoU >= 0.50 but normalized centroid error > 0.05)
7 TARGET_FAIL_WITH_REFERENCE_OK    (reference IoU >= 0.50, centroid error <= 0.05, target IoU < 0.50)
8 TARGET_OK                        (adequate reference and target IoU >= 0.50)
```

Dominant bottleneck: `PARSER` if parser_fail / 240 > 0.05; else `REFERENCE` if reference_fail > target_fail;
else `L3_TARGET_DECODER` (ties go to `REFERENCE`). DSH reports no repair.

Writes `evaluation/task7a_failure_attribution.json`.

    python scripts/task7a_failure_attribution.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
OUT = EVAL / "task7a_failure_attribution.json"
PARSER_AUDIT = EVAL / "task7a_parser_l3_val.json"
NL_VAL = EVAL / "task7a_natural_language_val.json"
REF_QUALITY = EVAL / "task7a_predicted_reference_quality.json"
TOLERANCE = 1.0e-6
BUCKETS = ("PARSER_WRONG", "REFERENCE_NO_PROPOSALS", "REFERENCE_NO_ELIGIBLE",
           "REFERENCE_NOT_COVERED_IOU50", "REFERENCE_SELECTION_WRONG", "REFERENCE_GEOMETRY_POOR",
           "TARGET_FAIL_WITH_REFERENCE_OK", "TARGET_OK")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    parser_audit = json.loads(PARSER_AUDIT.read_text(encoding="utf-8"))
    nl_val = json.loads(NL_VAL.read_text(encoding="utf-8"))
    ref_quality = json.loads(REF_QUALITY.read_text(encoding="utf-8"))
    records = json.loads((PACK_ROOT / "z_mini_val_240.json").read_text(encoding="utf-8"))["records"]

    reference_by_id = {row["sample_id"]: row for row in ref_quality["rows"]}
    parser_by_id = {row["sample_id"]: row for row in parser_audit["rows"]}
    nl_by_id = {row["sample_id"]: row for row in nl_val["rows"]}

    buckets: dict[str, str] = {}
    details = []
    for record in records:
        sample_id = record["sample_id"]
        parser_row = parser_by_id.get(sample_id, {})
        reference_row = reference_by_id.get(sample_id, {})
        nl_row = nl_by_id.get(sample_id, {})
        parser_correct = bool(parser_row.get("correct", False))
        reference_iou = reference_row.get("reference_iou")
        centroid = reference_row.get("centroid_error")
        target_iou = float(nl_row.get("miou", 0.0))
        best_eligible = float(reference_row.get("best_eligible_coverage_at_0_5", 0.0))
        if not parser_correct:
            bucket = "PARSER_WRONG"
        elif reference_row.get("abstention_reason") == "no_proposals" \
                or reference_row.get("proposal_count", 0) == 0:
            bucket = "REFERENCE_NO_PROPOSALS"
        elif reference_row.get("abstention_reason") == "no_eligible_proposals" \
                or reference_row.get("eligible_count", 0) == 0:
            bucket = "REFERENCE_NO_ELIGIBLE"
        elif reference_iou is None:
            bucket = "REFERENCE_SELECTION_WRONG"
        elif reference_iou < 0.50:
            bucket = ("REFERENCE_SELECTION_WRONG" if best_eligible >= 0.50
                      else "REFERENCE_NOT_COVERED_IOU50")
        elif centroid is not None and centroid > 0.05:
            bucket = "REFERENCE_GEOMETRY_POOR"
        elif target_iou < 0.50:
            bucket = "TARGET_FAIL_WITH_REFERENCE_OK"
        else:
            bucket = "TARGET_OK"
        buckets[sample_id] = bucket
        details.append({"sample_id": sample_id, "direction": record["direction"],
                        "expected_program": record["program_id"],
                        "parsed_program": parser_row.get("parsed_program"),
                        "parser_correct": parser_correct,
                        "reference_iou": reference_iou, "centroid_error": centroid,
                        "best_eligible_coverage_at_0_5": best_eligible,
                        "target_iou": target_iou, "bucket": bucket})

    counts = Counter(details_row["bucket"] for details_row in details)
    total = len(details)
    parser_fail = counts.get("PARSER_WRONG", 0)
    reference_fail = sum(counts.get(name, 0) for name in
                         ("REFERENCE_NO_PROPOSALS", "REFERENCE_NO_ELIGIBLE",
                          "REFERENCE_NOT_COVERED_IOU50", "REFERENCE_SELECTION_WRONG",
                          "REFERENCE_GEOMETRY_POOR"))
    target_fail = counts.get("TARGET_FAIL_WITH_REFERENCE_OK", 0)
    if parser_fail / total > 0.05:
        bottleneck = "PARSER"
    elif reference_fail > target_fail:
        bottleneck = "REFERENCE"
    else:
        bottleneck = "L3_TARGET_DECODER"

    def summarise(subset: list[dict]) -> dict:
        subset_counts = Counter(row["bucket"] for row in subset)
        return {"records": len(subset),
                "buckets": {name: subset_counts.get(name, 0) for name in BUCKETS}}

    payload = {
        "_doc": ("Task 7A section 19. Exclusive failure attribution for the natural-language L3 chain on "
                 "Z-MiniVal240, assigned with the fixed section-19 priority. DSH reports the dominant "
                 "bottleneck and proposes no repair."),
        "task": "7A", "stage": "I-failure-attribution",
        "records": total,
        "bucket_order": list(BUCKETS),
        "counts": {name: counts.get(name, 0) for name in BUCKETS},
        "percentages": {name: counts.get(name, 0) / total for name in BUCKETS},
        "by_direction": {direction: summarise([row for row in details
                                               if row["direction"] == direction])
                         for direction in ("above", "below", "left", "right")},
        "summary": {"parser_fail": parser_fail, "reference_fail": reference_fail,
                    "target_fail": target_fail,
                    "parser_fail_rate": parser_fail / total,
                    "target_ok": counts.get("TARGET_OK", 0)},
        "dominant_bottleneck": bottleneck,
        "dominant_bottleneck_rule": ("parser_fail/240 > 0.05 -> PARSER; else reference_fail > target_fail "
                                     "-> REFERENCE; else L3_TARGET_DECODER (ties -> REFERENCE)"),
        "rows": details,
        "repair_proposed": False,
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[7a.attr] counts {payload['counts']} | parser_fail {parser_fail} reference_fail "
          f"{reference_fail} target_fail {target_fail} -> {bottleneck}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
