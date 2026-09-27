"""Task 6H.1 section 22: resolve exactly one verdict from the recorded artifacts.

Reads only tracked artifacts; it never re-runs a model. Writes
`evaluation/task6h1_verdict.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6h1_common import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6h1_verdict.json"

VERDICTS = (
    "BOUNDED_COUNTERFACTUAL_FIX_FOUND",
    "BOUNDED_COUNTERFACTUAL_PARTIAL",
    "BOUNDED_COUNTERFACTUAL_GROUNDING_FAILED",
    "BOUNDED_POINT_OBJECTIVE_FAILED",
    "INVALID_EXPERIMENT",
)


def _load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    setup = _load("task6h1_objective_setup.json")
    h0r = _load("task6h1_h0r_overfit.json")
    h1r = _load("task6h1_h1r_training.json")
    segmentation = _load("task6h1_segmentation_eval.json")

    invalid_reasons = []
    if setup is None:
        invalid_reasons.append("objective setup artifact missing")
    elif not setup.get("passed"):
        invalid_reasons.append("objective setup smoke did not pass")
    if setup is not None:
        if not setup.get("architecture", {}).get("unchanged_from_task6g"):
            invalid_reasons.append("the Task 6G/6H architecture dimensions changed")
        if not setup.get("causal_placement", {}).get("box_hidden_bit_identical"):
            invalid_reasons.append("the causal-placement bit-identity test did not pass")
        retired = setup.get("retired_terms", {})
        if retired.get("bce_dice_requires_grad") or retired.get("logit_ranking_requires_grad"):
            invalid_reasons.append("a retired objective term still carries gradient")
        smoke = setup.get("one_pair_step", {})
        if not smoke.get("sam2_bit_identical"):
            invalid_reasons.append("SAM2 parameters moved during a pair step (freeze violation)")
        if not smoke.get("shared_feature_bit_identical"):
            invalid_reasons.append("the shared SAM2 feature changed during a pair step")
    if segmentation is not None and not segmentation.get("sam_frozen", False):
        invalid_reasons.append("SAM2 parameters were not frozen during H2-R")

    evidence = {"objective_setup": None, "H0R": None, "H1R": None, "H2R": None}
    if setup is not None:
        evidence["objective_setup"] = {
            "passed": setup["passed"],
            "architecture": setup["architecture"],
            "target_cells": setup["target_cells"],
            "objective": setup["objective"],
            "retired_terms": setup["retired_terms"],
            "gradients": setup["gradients"],
            "one_pair_step": {
                "mean_abs_logit": setup["one_pair_step"]["mean_abs_logit"],
                "pair_preference_pass": setup["one_pair_step"]["pair_preference_pass"],
                "sam2_bit_identical": setup["one_pair_step"]["sam2_bit_identical"],
            },
        }
    if h0r is not None:
        evidence["H0R"] = {
            "verdict": h0r["verdict"],
            "point_inside_count": h0r["final"]["records"]["point_inside_count"],
            "record_count": h0r["final"]["records"]["count"],
            "target_cell_top1_count": h0r["final"]["records"]["target_cell_top1_count"],
            "target_cell_top5_count": h0r["final"]["records"]["target_cell_top5_count"],
            "paired_point_selection_pass": h0r["final"]["pair"]["paired_point_selection_pass"],
            "pair_ranking_pass": h0r["final"]["pair"]["pair_ranking_pass"],
            "pair_count": h0r["final"]["pair"]["pair_count"],
            "mean_own_mass": h0r["final"]["pair"]["mean_own_mass"],
            "mean_cross_mass": h0r["final"]["pair"]["mean_cross_mass"],
            "mean_abs_logit": h0r["final"]["mean_abs_logit"],
            "gate": h0r["final"]["gate"],
        }
    if h1r is not None:
        evidence["H1R"] = {
            "verdict": h1r["verdict"],
            "selected_epoch": h1r["selection"]["selected_epoch"],
            "paired_point_selection_pass": h1r["final"]["validation_pairs"][
                "paired_point_selection_pass"
            ],
            "paired_point_total": h1r["final"]["validation_pairs"]["pair_count"],
            "pair_ranking_pass": h1r["final"]["validation_pairs"]["pair_ranking_pass"],
            "mean_own_mass": h1r["final"]["validation_pairs"]["mean_own_mass"],
            "mean_cross_mass": h1r["final"]["validation_pairs"]["mean_cross_mass"],
            "point_inside_target_rate": h1r["final"]["val"]["point_inside_target_rate"],
            "target_cell_top5_rate": h1r["final"]["val"]["target_cell_top5_rate"],
            "gate": h1r["final"]["gate"],
        }
    if segmentation is not None:
        paired = segmentation["segmentation"]["paired"]
        evidence["H2R"] = {
            "strict_end_to_end_miou": segmentation["segmentation"]["strict_end_to_end_miou"],
            "mean_dice": segmentation["segmentation"]["mean_dice"],
            "mask_paired_pass": paired["mask_paired_pass"],
            "paired_total": paired["paired_total"],
            "mean_own_minus_cross_margin": paired["mean_own_minus_cross_margin"],
            "fix_checks": segmentation["fix_checks"],
        }

    if invalid_reasons:
        verdict, reason = "INVALID_EXPERIMENT", "; ".join(invalid_reasons)
    elif h0r is None or h0r["verdict"] != "H0R_PASS":
        pair = (h0r or {}).get("final", {}).get("pair", {})
        records = (h0r or {}).get("final", {}).get("records", {})
        verdict, reason = (
            "BOUNDED_POINT_OBJECTIVE_FAILED",
            "H0-R cannot pass after the focused audit: inside-own "
            f"{records.get('point_inside_count')}/{records.get('count')}, paired point "
            f"{pair.get('paired_point_selection_pass')}/{pair.get('pair_count')}, |logit| "
            f"{(h0r or {}).get('final', {}).get('mean_abs_logit')}",
        )
    elif h1r is None or not h1r["final"]["gate"]["passed"]:
        verdict, reason = (
            "BOUNDED_COUNTERFACTUAL_GROUNDING_FAILED",
            "H0-R passes, but the corrected objective does not generalize: the H1-R gate failed on "
            "the selected epoch",
        )
    elif segmentation is not None and all(segmentation["fix_checks"].values()):
        verdict, reason = (
            "BOUNDED_COUNTERFACTUAL_FIX_FOUND",
            "H0-R, the H1-R point/pair gate, the H2-R mask gate and the own-minus-cross mask margin "
            "all pass with no ground-truth geometry in any prompt",
        )
    else:
        verdict, reason = (
            "BOUNDED_COUNTERFACTUAL_PARTIAL",
            "the corrected objective yields real localization improvement, but the full H2-R gate "
            "is not met or H2-R was not run",
        )

    if verdict not in VERDICTS:
        raise RuntimeError(f"unknown verdict {verdict!r}")
    report = {
        "_doc": (
            "Task 6H.1 section 22. Exactly one verdict, resolved from the recorded artifacts. The "
            "Task 6H machine verdict and numbers are preserved; this task's interpretation narrows "
            "the Task 6H causal claim (section 3)."
        ),
        "task": "6H.1",
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
    }
    write_json(OUT, report)
    print(f"[task6h1:verdict] {verdict}: {reason}", flush=True)
    print(f"[task6h1:verdict] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
