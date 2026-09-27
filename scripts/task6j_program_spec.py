"""Task 6J section 2: build the canonical program vocabulary from the ACTUAL frozen query types.

No semantics are invented: the spec is derived from the v0.1.1 records' stored
``reasoning_steps`` operation patterns (asserted to be one pattern per query type and to match
the role-templated canonical form), and sample counts per split come from the dataset itself.
Writes `evaluation/task6j_program_spec.json`.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.structured_grounding import (  # noqa: E402
    EXPECTED_QUERY_TYPES,
    build_program_spec,
    canonical_program_template,
)

from task6j_common import EVAL, SPEC_OUT, write_json  # noqa: E402


def main() -> int:
    train = data_mod.read_records("train")
    val = data_mod.read_records("val")
    test = data_mod.read_records("test")

    # Templates must be consistent with EVERY split's stored operation patterns.
    programs = build_program_spec(train + val + test)

    type_counts: dict[str, dict[str, int]] = {}
    for split, records in (("train", train), ("val", val), ("test", test)):
        counts = Counter(record["query_type"] for record in records)
        for query_type in EXPECTED_QUERY_TYPES:
            type_counts.setdefault(query_type, {})[split] = int(counts.get(query_type, 0))

    for query_type, program in programs.items():
        program["sample_counts_by_split"] = type_counts[query_type]
        program["template"] = canonical_program_template(query_type)

    report = {
        "_doc": (
            "Task 6J section 2. Canonical program vocabulary built 1:1 from the frozen "
            "BuildSpatialReason v0.1.1 query types: each program id IS a frozen query type, the "
            "ordered operations are the dataset's stored reasoning-step patterns (verified to be "
            "unique per query type across train+val+test), and sample counts come from the dataset. "
            "The test split is listed for completeness and is never used for training or tuning."
        ),
        "task": "6J",
        "program_count": len(programs),
        "expected_query_types": list(EXPECTED_QUERY_TYPES),
        "programs": programs,
        "operation_vocabulary": [
            "argmin_centroid_x",
            "argmax_centroid_x",
            "argmin_centroid_y",
            "argmax_centroid_y",
            "argmax_area",
            "argmin_area",
            "filter_relation",
            "argmin_boundary_distance",
        ],
        "reference_roles": {"@1": "the single output of step 1 (the argument-step anchor)"},
        "note": (
            "filter_relation uses relation(subject, reference) in the frozen Task 3B convention; "
            "argmin_boundary_distance uses the frozen nearest_within semantics (boundary distance, "
            "non-border anchor, frozen margins)."
        ),
        "test_split_used_for_training": False,
    }
    write_json(SPEC_OUT, report)
    print(
        f"[task6j:spec] {len(programs)} canonical programs over "
        f"{sorted(programs)}; train samples {sum(c['sample_counts_by_split']['train'] for c in programs.values())}",
        flush=True,
    )
    print(f"[task6j:spec] wrote {SPEC_OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
