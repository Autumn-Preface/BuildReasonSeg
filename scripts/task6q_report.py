"""Task 6Q Parts F-G — gates and the single allowed verdict.

Predeclared gates (fixed, never altered):

* section 11 proposal coverage — eligible reference coverage@0.50 >= 0.70 overall, >= 0.75 for largest,
  >= 0.60 for smallest;
* section 12 reference resolver adequacy — selected-reference mIoU >= 0.35, median normalized centroid
  error <= 0.05, p90 <= 0.12, abstention rate <= 0.10;
* section 13 downstream chain retention — section 12 passes, target mIoU >= 0.3009776246035379,
  PairedVal >= 10/20, own-cross margin >= 0.20.

Writes `evaluation/task6q_verdict.json`.

    python scripts/task6q_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6q_verdict.json"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "PROPOSAL_CHECKPOINT_UNAVAILABLE",
    "REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT",
    "DETERMINISTIC_REFERENCE_SELECTOR_INSUFFICIENT",
    "REFERENCE_ERROR_PROPAGATION_SEVERE",
    "PROPOSAL_REFERENCE_CHAIN_FEASIBLE",
)


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    resolver = load("task6q_reference_resolver_val.json")
    failure = load("task6q_reference_failure_attribution.json")
    target = load("task6q_target_val.json")
    paired = load("task6q_target_paired_val.json")
    manifest = load("task6q_proposal_cache_manifest.json")
    required = {
        "task6q_reference_resolver_val.json": resolver,
        "task6q_reference_failure_attribution.json": failure,
        "task6q_target_val.json": target,
        "task6q_target_paired_val.json": paired,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 6Q section 14 verdict.", "task": "6Q",
                         "verdict": "INVALID_EXPERIMENT", "reason": f"missing artifacts: {missing}"})
        print(f"[6q.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2
    if resolver.get("verdict") == "PROPOSAL_CHECKPOINT_UNAVAILABLE":
        write_json(OUT, {"_doc": "Task 6Q section 14 verdict.", "task": "6Q",
                         "verdict": "PROPOSAL_CHECKPOINT_UNAVAILABLE"})
        print("[6q.report] PROPOSAL_CHECKPOINT_UNAVAILABLE", flush=True)
        return 2

    coverage_gate = resolver["coverage_gate"]
    resolver_gate = resolver["resolver_gate"]
    chain_measured = target["gate"]
    paired_gate = paired["gate"]

    section_13 = {
        "section_12_passed": {"required": True, "measured": resolver_gate["passed"],
                              "passed": bool(resolver_gate["passed"])},
        "target_miou": {"required": 0.3009776246035379,
                        "measured": chain_measured["measured_target_miou"],
                        "passed": chain_measured["measured_target_miou"] >= 0.3009776246035379},
        "paired_pass": {"required": 10, "measured": paired["passed"],
                        "passed": paired["passed"] >= 10},
        "own_cross_margin": {"required": 0.20, "measured": paired["own_cross_margin"],
                             "passed": paired["own_cross_margin"] >= 0.20},
    }
    section_13["passed"] = all(entry["passed"] for entry in section_13.values())

    # section 14 priority order, applied literally
    if not resolver["checkpoint"]["matches"]:
        verdict = "PROPOSAL_CHECKPOINT_UNAVAILABLE"
        reason = "the frozen proposal checkpoint is unavailable or does not match its SHA256"
    elif not coverage_gate["passed"]:
        verdict = "REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT"
        reason = "section 11 proposal coverage gates fail"
    elif not resolver_gate["passed"]:
        verdict = "DETERMINISTIC_REFERENCE_SELECTOR_INSUFFICIENT"
        reason = "section 11 passes but section 12 reference-resolver adequacy fails"
    elif not section_13["passed"]:
        verdict = "REFERENCE_ERROR_PROPAGATION_SEVERE"
        reason = "section 12 passes but section 13 downstream chain retention fails"
    else:
        verdict = "PROPOSAL_REFERENCE_CHAIN_FEASIBLE"
        reason = "sections 11, 12 and 13 all pass"

    payload = {
        "_doc": (
            "Task 6Q sections 11-14. Frozen-proposal reference resolver gates and the single verdict. "
            "Diagnostic/selection task only: no training, no threshold tuning, no test split. DSH "
            "reports measurements and claims no novelty for the resolver."
        ),
        "task": "6Q",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "priority_order_applied": list(ALLOWED_VERDICTS),
        "criteria": {"11_proposal_coverage": coverage_gate,
                     "12_reference_resolver_adequacy": resolver_gate,
                     "13_downstream_chain_retention": section_13},
        "checkpoint": resolver["checkpoint"],
        "proposal_config": resolver["proposal_config"],
        "pack": resolver["pack"],
        "coverage": resolver["coverage"],
        "selected_reference_quality": resolver["selected_reference_quality"],
        "failure_categories": resolver["failure_categories"],
        "failure_categories_by_family": resolver["failure_categories_by_family"],
        "target_chain": {
            "overall": target["target"]["overall"],
            "abstentions": target["target"]["abstentions"],
            "abstention_rate": target["target"]["abstention_rate"],
            "per_relation": target["target"]["per_relation"],
            "per_reference_family": target["target"]["per_reference_family"],
            "comparison": target["comparison"],
        },
        "paired_chain": {
            "passed": paired["passed"], "pairs": paired["pairs"],
            "mean_own_iou": paired["mean_own_iou"], "mean_cross_iou": paired["mean_cross_iou"],
            "own_cross_margin": paired["own_cross_margin"],
            "pairs_with_reference_abstention": paired["pairs_with_reference_abstention"],
            "comparison": paired["comparison"],
        },
        "proposal_cache": None if manifest is None else {
            "cache_entries": manifest["cache_entries"], "tiles": manifest["tiles"],
            "proposal_counts": manifest["proposal_counts"],
            "abstained_family_tiles": manifest["abstained_family_tiles"],
            "cache_sha256": manifest["cache_sha256"],
        },
        "test_split_used": False,
        "training_performed": False,
        "threshold_tuning": False,
        "no_architecture_decision_by_dsh": True,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(
        f"[6q.report] coverage@0.50 overall {coverage_gate['measured_overall']:.4f} "
        f"largest {coverage_gate['measured_largest']:.4f} smallest "
        f"{coverage_gate['measured_smallest']:.4f} | resolver mIoU "
        f"{resolver_gate['measured_miou']:.6f} centroid p90 {resolver_gate['measured_p90_centroid']:.6f} "
        f"| target mIoU {chain_measured['measured_target_miou']:.6f} paired {paired['passed']}/20 "
        f"margin {paired['own_cross_margin']:+.6f} -> {verdict}",
        flush=True,
    )
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
