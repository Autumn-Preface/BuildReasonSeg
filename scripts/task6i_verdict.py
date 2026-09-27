"""Task 6I section 13: resolve exactly one verdict from the recorded artifacts.

Reads only tracked artifacts; it never re-runs a model. Writes
`evaluation/task6i_verdict.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6i_common import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6i_verdict.json"

VERDICTS = (
    "VISUAL_QUERY_REFINEMENT_FIX_FOUND",
    "VISUAL_QUERY_REFINEMENT_PARTIAL",
    "VISUAL_QUERY_REFINEMENT_NO_GENERALIZATION",
    "VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT",
    "INVALID_EXPERIMENT",
)


def _load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    setup = _load("task6i_architecture_setup.json")
    i0 = _load("task6i_i0_overfit.json")
    i1 = _load("task6i_i1_training.json")
    segmentation = _load("task6i_segmentation_eval.json")

    invalid_reasons = []
    if setup is None:
        invalid_reasons.append("architecture setup artifact missing")
    elif not setup.get("passed"):
        invalid_reasons.append("architecture setup smoke did not pass")
    if setup is not None:
        arch = setup.get("architecture", {})
        if not arch.get("coarse", {}).get("coarse_exact"):
            invalid_reasons.append("the coarse F64 feature is not exactly [B,256,64,64]")
        if not arch.get("coarse", {}).get("fine_exact"):
            invalid_reasons.append("the fine F256 feature is not exactly [B,32,256,256]")
        if not arch.get("single_cross_attention_layer"):
            invalid_reasons.append("the block does not have exactly one cross-attention layer")
        if not setup.get("causal_placement", {}).get("box_hidden_bit_identical"):
            invalid_reasons.append("the causal-placement bit-identity test did not pass")
        retired = setup.get("retired_terms", {})
        if retired.get("bce_dice_requires_grad") or retired.get("logit_ranking_requires_grad"):
            invalid_reasons.append("a retired objective term still carries gradient")
        smoke = setup.get("one_pair_step", {})
        if not smoke.get("sam2_bit_identical") or smoke.get("sam2_any_grad"):
            invalid_reasons.append("SAM2 parameters moved during a pair step (freeze violation)")
        if not smoke.get("shared_features_bit_identical"):
            invalid_reasons.append("the shared SAM2 features changed during a pair step")
        if not smoke.get("frozen_dense_head_bit_identical"):
            invalid_reasons.append("the frozen Task 6G head moved during a pair step")
    if segmentation is not None and not segmentation.get("sam_frozen", False):
        invalid_reasons.append("SAM2 parameters were not frozen during I2")

    evidence = {"architecture_setup": None, "I0": None, "I1": None, "I2": None}
    if setup is not None:
        evidence["architecture_setup"] = {
            "passed": setup["passed"],
            "architecture": setup["architecture"],
            "causal_placement": setup["causal_placement"],
            "objective": setup["objective"],
            "retired_terms": setup["retired_terms"],
            "gradients": setup["gradients"],
            "one_pair_step": {
                "mean_abs_logit": setup["one_pair_step"]["mean_abs_logit"],
                "pair_preference_pass": setup["one_pair_step"]["pair_preference_pass"],
                "sam2_bit_identical": setup["one_pair_step"]["sam2_bit_identical"],
            },
        }
    if i0 is not None:
        evidence["I0"] = {
            "verdict": i0["verdict"],
            "point_inside_count": i0["final"]["records"]["point_inside_count"],
            "record_count": i0["final"]["records"]["count"],
            "target_cell_top1_count": i0["final"]["records"]["target_cell_top1_count"],
            "target_cell_top5_count": i0["final"]["records"]["target_cell_top5_count"],
            "paired_point_selection_pass": i0["final"]["pair"]["paired_point_selection_pass"],
            "pair_ranking_pass": i0["final"]["pair"]["pair_ranking_pass"],
            "pair_count": i0["final"]["pair"]["pair_count"],
            "mean_own_mass": i0["final"]["pair"]["mean_own_mass"],
            "mean_cross_mass": i0["final"]["pair"]["mean_cross_mass"],
            "mean_normalized_error": i0["final"]["records"]["mean_normalized_point_error"],
            "mean_abs_logit": i0["final"]["mean_abs_logit"],
            "attention": i0["final"]["attention"],
            "gate": i0["final"]["gate"],
        }
    if i1 is not None:
        evidence["I1"] = {
            "verdict": i1["verdict"],
            "selected_epoch": i1["selection"]["selected_epoch"],
            "paired_point_selection_pass": i1["final"]["validation_pairs"][
                "paired_point_selection_pass"
            ],
            "paired_point_total": i1["final"]["validation_pairs"]["pair_count"],
            "pair_ranking_pass": i1["final"]["validation_pairs"]["pair_ranking_pass"],
            "mean_own_mass": i1["final"]["validation_pairs"]["mean_own_mass"],
            "mean_cross_mass": i1["final"]["validation_pairs"]["mean_cross_mass"],
            "point_inside_target_rate": i1["final"]["val"]["point_inside_target_rate"],
            "target_cell_top5_rate": i1["final"]["val"]["target_cell_top5_rate"],
            "attention": i1["final"]["attention"],
            "gate": i1["final"]["gate"],
        }
    if segmentation is not None:
        paired = segmentation["segmentation"]["paired"]
        evidence["I2"] = {
            "strict_end_to_end_miou": segmentation["segmentation"]["strict_end_to_end_miou"],
            "mean_dice": segmentation["segmentation"]["mean_dice"],
            "mask_paired_pass": paired["mask_paired_pass"],
            "paired_total": paired["paired_total"],
            "mean_own_minus_cross_margin": paired["mean_own_minus_cross_margin"],
            "fix_checks": segmentation["fix_checks"],
        }

    if invalid_reasons:
        verdict, reason = "INVALID_EXPERIMENT", "; ".join(invalid_reasons)
    elif i0 is None or i0["verdict"] != "I0_PASS":
        records = (i0 or {}).get("final", {}).get("records", {})
        pair = (i0 or {}).get("final", {}).get("pair", {})
        verdict, reason = (
            "VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT",
            "I0 cannot pass after the focused audit: inside-own "
            f"{records.get('point_inside_count')}/{records.get('count')}, paired point "
            f"{pair.get('paired_point_selection_pass')}/{pair.get('pair_count')}, ranking "
            f"{pair.get('pair_ranking_pass')}/{pair.get('pair_count')}",
        )
    elif i1 is None or not i1["final"]["gate"]["passed"]:
        verdict, reason = (
            "VISUAL_QUERY_REFINEMENT_NO_GENERALIZATION",
            "I0 passes, but the refined path does not generalize: the I1 gate failed on the "
            "selected epoch",
        )
    elif segmentation is not None and all(segmentation["fix_checks"].values()):
        verdict, reason = (
            "VISUAL_QUERY_REFINEMENT_FIX_FOUND",
            "I0, the I1 point/pair gate, the I2 mask gate and the own-minus-cross mask margin "
            "all pass with no ground-truth geometry in any prompt",
        )
    else:
        verdict, reason = (
            "VISUAL_QUERY_REFINEMENT_PARTIAL",
            "the refinement block yields real target-specific localization improvement, but the "
            "full I2 gate is not met or I2 was not run",
        )

    if verdict not in VERDICTS:
        raise RuntimeError(f"unknown verdict {verdict!r}")
    report = {
        "_doc": (
            "Task 6I section 13. Exactly one verdict, resolved from the recorded artifacts. The "
            "frozen Task 6H.1 objective and the Task 6H/6H.1 numbers are preserved for comparison."
        ),
        "task": "6I",
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
    }
    write_json(OUT, report)
    print(f"[task6i:verdict] {verdict}: {reason}", flush=True)
    print(f"[task6i:verdict] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
