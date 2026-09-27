"""Task 6J sections 3-5: Stage J0 -- oracle program + oracle candidates.

The independent executor runs the oracle canonical program (the record's stored
``reasoning_steps``, with explicit instance ids) over the oracle candidate set (every visible
component of the frozen component map) and returns one candidate id. The executor never reads
the target component id, GT reasoning text or the target mask: GT is used only to score the
selected id afterwards. As an implementation-fidelity check, every sample is also re-executed by
the frozen generator function ``recompute_target_from_steps`` and the two must agree exactly.

J0 gate: exact target accuracy >= 0.98 on the fixed 120 validation records and paired target
selection >= 19/20 on the fixed 20 paired validation images. Failure is
`STRUCTURED_EXECUTOR_SEMANTICS_MISMATCH`.

Writes `evaluation/task6j_j0_oracle_executor.json`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "spatial_reasoning"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.structured_grounding import execute_program  # noqa: E402
from dataset_access import DEFAULT_DATASET_ROOT  # noqa: E402
from thresholds import load_config as load_relation_config  # noqa: E402

from annotator import recompute_target_from_steps  # noqa: E402
from component_quality import classify_image  # noqa: E402
from task6j_common import (  # noqa: E402
    EVAL,
    fixed_validation_material,
    gate_report,
    geometry_for,
    oracle_candidate_set,
    paired_sample_lists,
    record_accuracy_summary,
    write_json,
)

OUT = EVAL / "task6j_j0_oracle_executor.json"


def main() -> int:
    started = time.time()
    config = load_relation_config()
    val_samples, pairs = fixed_validation_material()
    a_samples, b_samples = paired_sample_lists(pairs)

    def run(sample):
        record = sample.raw
        program = record["reasoning_steps"]
        candidates = oracle_candidate_set(sample)
        geometry = geometry_for(sample)
        result = execute_program(program, candidates, config)
        # Fidelity check: the frozen generator's own recomputation must agree exactly.
        frozen = recompute_target_from_steps(
            geometry, program, config, classify_image(geometry, config),
            geometry.load_map(DEFAULT_DATASET_ROOT),
        )
        agree = (result.selected_id == frozen) or (result.abstained and frozen is None)
        target = int(record["target_component_id"])
        correct = (not result.abstained) and result.selected_id == target
        row = {
            "sample_id": str(sample.sample_id),
            "image_id": str(sample.image_id),
            "level": int(sample.level),
            "query_type": str(sample.query_type),
            "selected_id": result.selected_id,
            "target_id": target,
            "correct": bool(correct),
            "abstained": bool(result.abstained),
            "abstain_reason": result.reason,
            "frozen_recompute_agrees": bool(agree),
            "n_candidates": len(candidates.candidates),
        }
        return row

    val_rows = [run(sample) for sample in val_samples]
    paired_rows = [
        {"a": run(sample_a), "b": run(sample_b)}
        for sample_a, sample_b in zip(a_samples, b_samples)
    ]
    paired_pass = sum(
        1 for row in paired_rows if row["a"]["correct"] and row["b"]["correct"]
    )
    paired_summary = {
        "paired_total": len(paired_rows),
        "paired_selection_pass": paired_pass,
        "paired_selection_rate": float(paired_pass / max(len(paired_rows), 1)),
        "a_correct": sum(1 for row in paired_rows if row["a"]["correct"]),
        "b_correct": sum(1 for row in paired_rows if row["b"]["correct"]),
        "rows": paired_rows,
    }

    val_summary = record_accuracy_summary(val_rows)
    agree_count = sum(1 for row in val_rows if row["frozen_recompute_agrees"])
    abstentions: dict[str, int] = {}
    for row in val_rows:
        if row["abstained"]:
            abstentions[row["abstain_reason"]] = abstentions.get(row["abstain_reason"], 0) + 1

    gate_cfg = {"exact_accuracy_min": 0.98, "paired_selection_min": 19}
    checks = {
        "exact_accuracy_ge": float(val_summary["exact_accuracy"]) >= 0.98,
        "paired_selection_ge": paired_pass >= 19,
        "frozen_recompute_full_agreement": agree_count == len(val_rows),
    }
    gate = gate_report(checks, gate_cfg)

    report = {
        "_doc": (
            "Task 6J sections 3-5. Stage J0: oracle program (the stored reasoning_steps) executed "
            "by the independent executor over oracle candidates (every visible building component "
            "of the frozen component map, geometry = mask/centroid/bbox/area + border flag for "
            "diagnostics only). The executor reads no target id, no GT reasoning and no target "
            "mask; GT is post-selection scoring only. Every sample is also recomputed by the "
            "frozen generator function and must agree exactly."
        ),
        "task": "6J",
        "stage": "J0",
        "relation_config_version": config.version,
        "relation_config_path": config.source_path,
        "val": val_summary,
        "paired": paired_summary,
        "abstentions": abstentions,
        "frozen_recompute_agreement": {
            "agreeing": agree_count,
            "total": len(val_rows),
            "all_agree": agree_count == len(val_rows),
        },
        "gate": gate,
        "verdict": "J0_PASS" if gate["passed"] else "STRUCTURED_EXECUTOR_SEMANTICS_MISMATCH",
        "no_gt_leakage": {
            "executor_inputs": "canonical program + candidate geometry (mask/centroid/bbox/area)",
            "gt_used_for": ["post-selection scoring only"],
        },
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6j.j0] exact accuracy {val_summary['exact_accuracy']:.4f} "
        f"({val_summary['exact_correct']}/{val_summary['count']}), abstentions "
        f"{val_summary['abstained']} {abstentions}, paired {paired_pass}/"
        f"{len(paired_rows)}, frozen-recompute agreement {agree_count}/{len(val_rows)}, "
        f"gate {gate['passed']}",
        flush=True,
    )
    print(f"[task6j.j0] by-level { {k: round(v['accuracy'], 4) for k, v in val_summary['by_level'].items()} }", flush=True)
    print(f"[task6j.j0] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0 if gate["passed"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
