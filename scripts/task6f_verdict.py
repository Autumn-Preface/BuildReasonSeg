"""Task 6F section 17: resolve exactly one verdict from the recorded artifacts.

Reads only tracked artifacts; it never re-runs a model. Writes
`evaluation/task6f_verdict.json` with the verdict and the evidence for each section-17 condition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6f_common import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6f_verdict.json"

VERDICTS = (
    "TARGET_AWARE_QUERY_FIX_FOUND",
    "TARGET_AWARE_QUERY_PARTIAL",
    "TARGET_AWARE_QUERY_FAILED",
    "TARGET_QUERY_IMPLEMENTATION_FAILED",
    "INVALID_EXPERIMENT",
)


def _load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    token_setup = _load("task6f_token_setup.json")
    f0 = _load("task6f_f0_overfit.json")
    f1 = _load("task6f_f1_training.json")
    geometry = _load("task6f_geometry_eval.json")
    segmentation = _load("task6f_segmentation_eval.json")

    invalid_reasons = []
    if token_setup is None:
        invalid_reasons.append("token setup artifact missing")
    elif not token_setup.get("passed"):
        invalid_reasons.append("token setup smoke did not pass")
    if token_setup is not None:
        if not token_setup.get("causal_placement", {}).get("box_hidden_bit_identical"):
            invalid_reasons.append("the causal-placement bit-identity test did not pass")
        if not token_setup.get("vocabulary", {}).get("no_loc_tokens_added"):
            invalid_reasons.append("Task 6E location tokens leaked into the Task 6F vocabulary")
    if segmentation is not None and not segmentation.get("sam_frozen", False):
        invalid_reasons.append("SAM2 parameters were not frozen during F2")

    evidence = {
        "token_setup": (
            None
            if token_setup is None
            else {
                "passed": token_setup["passed"],
                "box_token_id": token_setup["vocabulary"]["box_token_id"],
                "causal_placement": token_setup["causal_placement"],
                "structure": token_setup["structure"],
                "trainables": token_setup["trainables"],
            }
        ),
        "F0": None,
        "F1": None,
        "F2": None,
    }
    if f0 is not None:
        evidence["F0"] = {
            "verdict": f0["verdict"],
            "train_box_iou": f0["final"]["train_box_iou"],
            "geometry_paired_pass": f0["final"]["paired"]["geometry_paired_pass"],
            "paired_total": f0["final"]["paired"]["paired_total"],
            "pairs_with_non_identical_boxes": f0["final"]["paired"][
                "pairs_with_non_identical_boxes"
            ],
            "mean_same_image_predicted_box_l1": f0["final"]["paired"][
                "mean_same_image_predicted_box_l1"
            ],
            "gate": f0["final"]["gate"],
        }
    if f1 is not None:
        evidence["F1"] = {
            "verdict": f1["verdict"],
            "selected_epoch": f1["selection"]["selected_epoch"],
            "stop_reason": f1["selection"]["stop_reason"],
            "train_box_iou_mean": f1["final"]["train_box_iou_mean"],
            "mean_val_box_iou": f1["final"]["metrics"]["mean_box_iou"],
            "center_inside_rate": f1["final"]["metrics"]["center_inside_rate"],
            "geometry_paired_pass": f1["final"]["paired"]["geometry_paired_pass"],
            "paired_total": f1["final"]["paired"]["paired_total"],
            "mean_same_image_predicted_box_l1": f1["final"]["paired"][
                "mean_same_image_predicted_box_l1"
            ],
            "gate": f1["final"]["gate"],
        }
    if segmentation is not None:
        paired = segmentation["segmentation"]["paired"]
        evidence["F2"] = {
            "strict_end_to_end_miou": segmentation["segmentation"]["strict_end_to_end_miou"],
            "mean_dice": segmentation["segmentation"]["mean_dice"],
            "mask_paired_pass": paired["mask_paired_pass"],
            "paired_total": paired["paired_total"],
            "mean_own_minus_cross_margin": paired["mean_own_minus_cross_margin"],
            "fix_checks": segmentation["fix_checks"],
        }

    verdict = None
    reason = ""
    if invalid_reasons:
        verdict, reason = "INVALID_EXPERIMENT", "; ".join(invalid_reasons)
    elif f0 is None or f0["verdict"] != "F0_PASS":
        verdict, reason = (
            "TARGET_QUERY_IMPLEMENTATION_FAILED",
            "F0 did not pass the 20-record implementation gate after the implementation audit",
        )
    elif f1 is None or not f1["final"]["gate"]["passed"]:
        verdict, reason = (
            "TARGET_AWARE_QUERY_FAILED",
            "the implementation is valid and F0 passes, but 480-record query grounding does not "
            "generalize: the F1 geometry gate failed on the selected epoch",
        )
    elif segmentation is not None and all(segmentation["fix_checks"].values()):
        verdict, reason = (
            "TARGET_AWARE_QUERY_FIX_FOUND",
            "F0, the F1 geometry gate, the F2 mask gate and the own-minus-cross margin all pass "
            "with no ground-truth geometry in any prompt",
        )
    elif f1["final"]["gate"]["passed"]:
        verdict, reason = (
            "TARGET_AWARE_QUERY_PARTIAL",
            "the query becomes target-specific and geometry generalizes (F1 gate passes), but the "
            "full F2 mask gate is not met or F2 was not run",
        )
    else:
        verdict, reason = ("TARGET_AWARE_QUERY_FAILED", "the F1 geometry gate failed")

    if verdict not in VERDICTS:
        raise RuntimeError(f"unknown verdict {verdict!r}")
    report = {
        "_doc": (
            "Task 6F section 17. Exactly one verdict, resolved from the recorded artifacts. "
            "Task 6E's coordinate-token pathway is retired and is not part of this verdict."
        ),
        "task": "6F",
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
    }
    write_json(OUT, report)
    print(f"[task6f:verdict] {verdict}: {reason}", flush=True)
    print(f"[task6f:verdict] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
