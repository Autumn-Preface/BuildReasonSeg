"""Task 6G section 18: resolve exactly one verdict from the recorded artifacts.

Reads only tracked artifacts; it never re-runs a model. Writes
`evaluation/task6g_verdict.json` with the verdict and the evidence for each section-18 condition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6g_common import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6g_verdict.json"

VERDICTS = (
    "DENSE_SPATIAL_GROUNDING_FIX_FOUND",
    "DENSE_SPATIAL_GROUNDING_PARTIAL",
    "DENSE_SPATIAL_GROUNDING_FAILED",
    "DENSE_GROUNDING_IMPLEMENTATION_FAILED",
    "DENSE_GRID_POINT_PATH_INADEQUATE",
    "INVALID_EXPERIMENT",
)


def _load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    oracle = _load("task6g_grid_oracle.json")
    token_setup = _load("task6g_token_setup.json")
    g0 = _load("task6g_g0_overfit.json")
    g1 = _load("task6g_g1_training.json")
    spatial = _load("task6g_spatial_eval.json")
    segmentation = _load("task6g_segmentation_eval.json")

    invalid_reasons = []
    if token_setup is None:
        invalid_reasons.append("token setup artifact missing")
    elif not token_setup.get("passed"):
        invalid_reasons.append("token setup smoke did not pass")
    if token_setup is not None:
        if not token_setup.get("causal_placement", {}).get("box_hidden_bit_identical"):
            invalid_reasons.append("the causal-placement bit-identity test did not pass")
        if not token_setup.get("one_step_smoke", {}).get("sam2_bit_identical"):
            invalid_reasons.append("SAM2 parameters moved during a training step (backbone-freeze violation)")
        if not token_setup.get("vocabulary", {}).get("no_loc_tokens_added"):
            invalid_reasons.append("Task 6E location tokens leaked into the Task 6G vocabulary")
    if segmentation is not None and not segmentation.get("sam_frozen", False):
        invalid_reasons.append("SAM2 parameters were not frozen during G2")

    selection = (oracle or {}).get("selection", {})
    evidence = {
        "grid_oracle": {
            "selected_grid": selection.get("selected_grid"),
            "qualifying_grids": selection.get("qualifying_grids"),
            "miou_threshold": selection.get("miou_threshold"),
            "per_grid": (oracle or {}).get("candidates") and {
                str(grid): {
                    "strict_miou": entry["strict_miou"],
                    "paired_pass": entry["paired_pass"],
                    "paired_total": entry["paired_total"],
                    "mean_displacement_512px": entry["mean_displacement_512px"],
                }
                for grid, entry in sorted(
                    ((int(key), value) for key, value in (oracle or {}).get("candidates", {}).items())
                )
            },
        },
        "G0": None,
        "G1": None,
        "G2": None,
    }
    if g0 is not None:
        evidence["G0"] = {
            "verdict": g0["verdict"],
            "point_inside_rate": g0["final"]["metrics"]["point_inside_target_rate"],
            "mean_heatmap_dice": g0["final"]["metrics"]["mean_heatmap_dice"],
            "paired_point_selection_pass": g0["final"]["paired"]["paired_point_selection_pass"],
            "pairs_with_distinct_points": g0["final"]["paired"]["pairs_with_distinct_points"],
            "mean_same_image_point_distance": g0["final"]["paired"]["mean_same_image_point_distance"],
            "gate": g0["final"]["gate"],
        }
    if g1 is not None:
        evidence["G1"] = {
            "verdict": g1["verdict"],
            "selected_epoch": g1["selection"]["selected_epoch"],
            "stop_reason": g1["selection"]["stop_reason"],
            "point_inside_target_rate": g1["final"]["metrics"]["point_inside_target_rate"],
            "mean_heatmap_dice": g1["final"]["metrics"]["mean_heatmap_dice"],
            "mean_512px_point_error": g1["final"]["metrics"]["mean_512px_point_error"],
            "paired_point_selection_pass": g1["final"]["paired"]["paired_point_selection_pass"],
            "paired_total": g1["final"]["paired"]["paired_total"],
            "gate": g1["final"]["gate"],
        }
    if segmentation is not None:
        paired = segmentation["segmentation"]["paired"]
        evidence["G2"] = {
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
    elif selection.get("selected_grid") is None:
        verdict, reason = (
            "DENSE_GRID_POINT_PATH_INADEQUATE",
            "no grid keeps the snapped-point oracle at paired >= 18/20 within the mIoU tolerance",
        )
    elif g0 is None or g0["verdict"] != "G0_PASS":
        verdict, reason = (
            "DENSE_GROUNDING_IMPLEMENTATION_FAILED",
            "G0 did not pass the 20-record implementation gate after the implementation audit",
        )
    elif g1 is None or not g1["final"]["gate"]["passed"]:
        verdict, reason = (
            "DENSE_SPATIAL_GROUNDING_FAILED",
            "the implementation is valid and G0 passes, but dense grounding does not generalize: "
            "the G1 localization gate failed on the selected epoch",
        )
    elif segmentation is not None and all(segmentation["fix_checks"].values()):
        verdict, reason = (
            "DENSE_SPATIAL_GROUNDING_FIX_FOUND",
            "G0, the G1 localization gate, the G2 mask gate and the own-minus-cross margin all "
            "pass with no ground-truth geometry in any prompt",
        )
    elif g1["final"]["gate"]["passed"]:
        verdict, reason = (
            "DENSE_SPATIAL_GROUNDING_PARTIAL",
            "the heatmap/point becomes target-specific and materially improves localization "
            "(G1 gate passes), but the full G2 mask gate is not met or G2 was not run",
        )
    else:
        verdict, reason = ("DENSE_SPATIAL_GROUNDING_FAILED", "the G1 localization gate failed")

    if verdict not in VERDICTS:
        raise RuntimeError(f"unknown verdict {verdict!r}")
    report = {
        "_doc": (
            "Task 6G section 18. Exactly one verdict, resolved from the recorded artifacts. The "
            "dot-product query-to-map fusion is not claimed as final novelty (section 19)."
        ),
        "task": "6G",
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
    }
    write_json(OUT, report)
    print(f"[task6g:verdict] {verdict}: {reason}", flush=True)
    print(f"[task6g:verdict] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
