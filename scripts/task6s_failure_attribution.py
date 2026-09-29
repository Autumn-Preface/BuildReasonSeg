"""Task 6S Parts G — exclusive failure attribution and the dominant-bottleneck label.

`--stage attribution` (sections 20-21): assigns exactly one exclusive bucket per MiniVal240 record in the
predeclared priority order, reports counts/percentages plus the required breakdowns, and computes the
dominant-bottleneck label with the fixed rule. GT is used **only** in this offline evaluator.

Writes `evaluation/task6s_failure_attribution.json`.

    python scripts/task6s_failure_attribution.py --stage attribution
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6s_failure_attribution.json"
IOU_ADEQUATE = 0.50
CENTROID_ADEQUATE = 0.05
PARSER_FAIL_LIMIT = 0.05

KNOWN_BUCKETS = (
    "PARSER_WRONG",
    "REFERENCE_NO_PROPOSALS",
    "REFERENCE_NO_ELIGIBLE",
    "REFERENCE_NOT_COVERED_IOU50",
    "REFERENCE_SELECTION_WRONG",
    "REFERENCE_GEOMETRY_POOR",
    "TARGET_FAIL_WITH_REFERENCE_OK",
    "TARGET_OK",
)
REFERENCE_BUCKETS = (
    "REFERENCE_NO_PROPOSALS",
    "REFERENCE_NO_ELIGIBLE",
    "REFERENCE_NOT_COVERED_IOU50",
    "REFERENCE_SELECTION_WRONG",
    "REFERENCE_GEOMETRY_POOR",
)


def bucket_for(row: dict) -> str:
    """Exactly one bucket per record, in the fixed priority order of section 20."""

    target_ok = (row.get("target_iou") or 0.0) >= IOU_ADEQUATE
    if not row.get("parser_correct") or not row.get("program_supported"):
        return "PARSER_WRONG"
    if row.get("proposal_count") == 0:
        return "REFERENCE_NO_PROPOSALS"
    if row.get("eligible_reference_proposals") == 0:
        return "REFERENCE_NO_ELIGIBLE"
    if (row.get("best_eligible_reference_iou") or 0.0) < IOU_ADEQUATE:
        return "REFERENCE_NOT_COVERED_IOU50"
    if (row.get("selected_reference_iou") or 0.0) < IOU_ADEQUATE:
        return "REFERENCE_SELECTION_WRONG"
    if (row.get("selected_reference_centroid_error") or 1.0) > CENTROID_ADEQUATE:
        return "REFERENCE_GEOMETRY_POOR"
    if not target_ok:
        return "TARGET_FAIL_WITH_REFERENCE_OK"
    return "TARGET_OK"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("attribution",), default="attribution")
    args = parser.parse_args(argv)
    started = time.time()

    e2e_path = EVAL / "task6s_end_to_end_val.json"
    e2e = json.loads(e2e_path.read_text(encoding="utf-8"))
    rows = e2e.get("records") or []
    if not rows:
        write_json(OUT, {"_doc": "Task 6S section 20.", "task": "6S",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": "per-record end-to-end output is unavailable"})
        print("[6s.attribution] INVALID_EXPERIMENT (no per-record data)", flush=True)
        return 2

    buckets = [bucket_for(row) for row in rows]
    counts = Counter(buckets)
    total = len(buckets)
    parser_fail = counts["PARSER_WRONG"]
    reference_fail = sum(counts[name] for name in REFERENCE_BUCKETS)
    target_fail = counts["TARGET_FAIL_WITH_REFERENCE_OK"]

    if parser_fail / total > PARSER_FAIL_LIMIT:
        dominant = "PARSER"
    elif reference_fail > target_fail:
        dominant = "REFERENCE"
    else:
        dominant = "TARGET_DECODER_FIELD"

    def breakdown(key: str) -> dict:
        table: dict[str, Counter] = defaultdict(Counter)
        for row, bucket in zip(rows, buckets):
            table[str(row.get(key))][bucket] += 1
        return {name: {bucket: counter[bucket] for bucket in KNOWN_BUCKETS if counter[bucket]}
                for name, counter in sorted(table.items())}

    ordered = {name: counts.get(name, 0) for name in KNOWN_BUCKETS}
    payload = {
        "_doc": (
            "Task 6S sections 20-21. Exactly one exclusive bucket per MiniVal240 record in the "
            "predeclared priority order, the required breakdowns, and the dominant-bottleneck label "
            "computed with the fixed rule. GT is used only inside this offline evaluator, never in "
            "inference, and DSH does not choose how to repair the bottleneck."
        ),
        "task": "6S", "stage": "G-failure-attribution",
        "input": {"path": str(e2e_path), "records": total},
        "priority_order": list(KNOWN_BUCKETS),
        "thresholds": {"reference_iou_adequate": IOU_ADEQUATE,
                       "centroid_error_adequate": CENTROID_ADEQUATE,
                       "parser_fail_limit": PARSER_FAIL_LIMIT},
        "counts": ordered,
        "percentages": {name: ordered[name] / total for name in KNOWN_BUCKETS},
        "by_reference_family": breakdown("parsed_family"),
        "by_direction": breakdown("parsed_relation"),
        "by_border": breakdown("target_touches_border"),
        "by_expected_program": breakdown("expected_program"),
        "tiny_target_count": sum(1 for row in rows if row.get("target_tiny")),
        "tiny_target_buckets": dict(Counter(
            bucket for row, bucket in zip(rows, buckets) if row.get("target_tiny"))),
        "exclusive_bucket_check": {
            "one_bucket_per_record": len(buckets) == total,
            "bucket_values_known": sorted(set(buckets)) and set(buckets) <= set(KNOWN_BUCKETS),
            "no_record_lost": len(rows) == total,
        },
        "aggregation": {
            "parser_fail": parser_fail, "parser_fail_rate": parser_fail / total,
            "reference_fail": reference_fail, "target_fail": target_fail,
            "reference_fail_minus_target_fail": reference_fail - target_fail,
        },
        "dominant_bottleneck": dominant,
        "dominant_bottleneck_rule": (
            "if parser_fail/240 > 0.05 -> PARSER; else if reference_fail > target_fail -> REFERENCE; "
            "else TARGET_DECODER_FIELD; ties (reference_fail == target_fail) -> REFERENCE"
        ),
        "repair_choice": "WAIT_FOR_CHATGPT",
        "records_detail": [
            {"sample_id": row.get("sample_id"), "expected_program": row.get("expected_program"),
             "parsed_program": row.get("parsed_program"), "bucket": bucket,
             "target_iou": row.get("target_iou"),
             "selected_reference_iou": row.get("selected_reference_iou"),
             "best_eligible_reference_iou": row.get("best_eligible_reference_iou"),
             "selected_reference_centroid_error": row.get("selected_reference_centroid_error"),
             "proposal_count": row.get("proposal_count"),
             "eligible_reference_proposals": row.get("eligible_reference_proposals")}
            for row, bucket in zip(rows, buckets)
        ],
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[6s.attribution] parser_fail {parser_fail} ({parser_fail / total:.4f}) | reference_fail "
          f"{reference_fail} | target_fail {target_fail} -> {dominant}", flush=True)
    print("[6s.attribution] counts: " + ", ".join(f"{name}={ordered[name]}"
                                                  for name in KNOWN_BUCKETS), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
