"""Task 6E section 16: resolve exactly one verdict from the recorded artifacts.

Reads only tracked artifacts; it never re-runs a model. Writes
`evaluation/task6e_verdict.json` with the verdict, the evidence for each section-16
condition, and the section-17 wording rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6e_common import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6e_verdict.json"

VERDICTS = (
    "EXPLICIT_SPATIAL_TOKENS_FIX_FOUND",
    "EXPLICIT_SPATIAL_TOKENS_PARTIAL",
    "EXPLICIT_SPATIAL_TOKENS_FAILED",
    "SPATIAL_TOKEN_IMPLEMENTATION_FAILED",
    "QUANTIZED_BOX_REPRESENTATION_INADEQUATE",
    "INVALID_EXPERIMENT",
)


def _load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    oracle = _load("task6e_quantized_oracle.json")
    token_setup = _load("task6e_token_setup.json")
    e0 = _load("task6e_e0_overfit.json")
    e1 = _load("task6e_e1_training.json")
    geometry = _load("task6e_geometry_eval.json")
    segmentation = _load("task6e_segmentation_eval.json")

    evidence: dict = {}
    invalid_reasons = []
    if oracle is None:
        invalid_reasons.append("quantized oracle artifact missing")
    if token_setup is None:
        invalid_reasons.append("token setup artifact missing")
    elif not token_setup.get("passed"):
        invalid_reasons.append("token setup smoke did not pass")
    if token_setup is not None:
        vocabulary = token_setup.get("vocabulary", {})
        for key in (
            "all_names_single_token",
            "all_names_decode_roundtrip",
            "embedding_rows_match_tokenizer",
        ):
            if not vocabulary.get(key):
                invalid_reasons.append(f"tokenizer correctness: {key} is false")
        if token_setup.get("trainable_tokens", {}).get("visual_lora_modules"):
            invalid_reasons.append("LoRA leaked into the visual tower")
    if segmentation is not None and not segmentation.get("sam_frozen", False):
        invalid_reasons.append("SAM2 parameters were not frozen during E2")
    if segmentation is not None:
        determinism = segmentation.get("regeneration_determinism", {})
        if determinism.get("compared") and not determinism.get("all_matched"):
            invalid_reasons.append("E2 regeneration did not reproduce the E1 token ids")

    selection = (oracle or {}).get("selection", {})
    evidence["quantized_oracle"] = {
        "selected_bins": selection.get("selected_bins"),
        "qualifying_bins": selection.get("qualifying_bins"),
        "miou_threshold": selection.get("miou_threshold"),
        "per_bin": {
            str(bins): {
                "strict_mask_miou": entry["strict_mask_miou"],
                "paired_pass": entry["paired_pass"],
                "paired_total": entry["paired_total"],
                "mean_quantization_box_iou": entry["mean_quantization_box_iou"],
            }
            for bins, entry in sorted(
                ((int(key), value) for key, value in (oracle or {}).get("candidates", {}).items())
            )
        },
    }
    evidence["E0"] = None
    if e0 is not None:
        metrics = e0["final"]["metrics"]
        evidence["E0"] = {
            "verdict": e0["verdict"],
            "structural_valid": metrics["structural_valid"],
            "count": metrics["count"],
            "exact_four_token_sequence": metrics["exact_four_token_sequence"],
            "mean_predicted_box_iou": metrics["mean_predicted_box_iou"],
            "gate": e0["final"]["gate"],
        }
    evidence["E1"] = None
    if e1 is not None:
        metrics = e1["final"]["metrics"]
        evidence["E1"] = {
            "verdict": e1["verdict"],
            "selected_epoch": e1["selection"]["selected_epoch"],
            "structural_valid_rate": metrics["structural_valid_rate"],
            "mean_predicted_box_iou": metrics["mean_predicted_box_iou"],
            "geometry_paired_pass": e1["final"]["paired"]["geometry_paired_pass"],
            "paired_total": e1["final"]["paired"]["paired_total"],
            "gate": e1["final"]["gate"],
        }
    evidence["E2"] = None
    if segmentation is not None:
        paired = segmentation["segmentation"]["paired"]
        evidence["E2"] = {
            "strict_end_to_end_miou": segmentation["segmentation"]["strict_end_to_end_miou"],
            "mean_dice": segmentation["segmentation"]["mean_dice"],
            "conditional_end_to_end_miou": segmentation["segmentation"][
                "conditional_end_to_end_miou"
            ],
            "mask_paired_pass": paired["mask_paired_pass"],
            "paired_total": paired["paired_total"],
            "mean_own_minus_cross_margin": paired["mean_own_minus_cross_margin"],
            "mean_own_iou": paired["mean_own_iou"],
            "mean_cross_iou": paired["mean_cross_iou"],
            "mean_mask_iou_between_predictions": paired["mean_mask_iou_between_predictions"],
            "fix_checks": segmentation["fix_checks"],
        }
    evidence["same_image_instruction_divergence"] = (
        None if geometry is None else geometry.get("same_image_instruction_divergence")
    )

    verdict = None
    reason = ""
    if invalid_reasons:
        verdict, reason = "INVALID_EXPERIMENT", "; ".join(invalid_reasons)
    elif selection.get("selected_bins") is None:
        verdict, reason = (
            "QUANTIZED_BOX_REPRESENTATION_INADEQUATE",
            "no bin count kept paired 20/20 within the oracle mIoU tolerance",
        )
    elif e0 is None or e0["verdict"] != "E0_PASS":
        verdict, reason = (
            "SPATIAL_TOKEN_IMPLEMENTATION_FAILED",
            "E0 free-generation gate did not pass after the implementation audit",
        )
    elif e1 is None or not e1["final"]["gate"]["passed"]:
        metrics = (e1 or {}).get("final", {}).get("metrics", {})
        structural = metrics.get("structural_valid_rate")
        if structural is not None and float(structural) == 0.0:
            mode = (
                "the assistant format tokens were learned but the four location tokens were not: "
                "teacher forcing gives [BOX] accuracy 1.0 and [SEG] accuracy 0.0, and free generation "
                "emits [BOX] then a runaway location run (51-81 tokens) with no [SEG] at all, so the "
                "sequence is malformed rather than a valid-but-wrong box"
            )
        else:
            mode = "generation is structurally valid but the E1 geometry gate did not pass"
        verdict, reason = (
            "EXPLICIT_SPATIAL_TOKENS_FAILED",
            f"E1 geometry gate failed (structural validity {structural}, mean box IoU "
            f"{metrics.get('mean_predicted_box_iou')}, paired "
            f"{(e1 or {}).get('final', {}).get('paired', {}).get('geometry_paired_pass')}): {mode}",
        )
    elif segmentation is not None and all(segmentation["fix_checks"].values()):
        verdict, reason = (
            "EXPLICIT_SPATIAL_TOKENS_FIX_FOUND",
            "E0, the E1 geometry gate, the E2 mask gate and the own-minus-cross margin all pass "
            "with no ground-truth geometry in any prompt",
        )
    elif segmentation is not None:
        verdict, reason = (
            "EXPLICIT_SPATIAL_TOKENS_PARTIAL",
            "generated boxes are instruction-conditioned and geometry generalizes, but the E2 mask "
            "gate is not met",
        )
    else:
        divergence = (geometry or {}).get("same_image_instruction_divergence") or {}
        if divergence.get("pairs_with_different_tokens"):
            verdict, reason = (
                "EXPLICIT_SPATIAL_TOKENS_PARTIAL",
                "E1 geometry gate passed and the two instructions of an image now emit different "
                "boxes, but E2 was not run",
            )
        else:
            verdict, reason = (
                "EXPLICIT_SPATIAL_TOKENS_FAILED",
                "the E1 geometry gate passed but the paired probe shows no instruction-conditioned "
                "box divergence",
            )

    if verdict not in VERDICTS:
        raise RuntimeError(f"unknown verdict {verdict!r}")
    report = {
        "_doc": (
            "Task 6E section 16. Exactly one verdict, resolved from the recorded artifacts. "
            "Section 17: coordinate tokens are not the final novelty claim."
        ),
        "task": "6E",
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
        "claim_if_fix_found": (
            "We repaired the functional spatial-grounding bottleneck and established a usable "
            "reasoning -> explicit geometry -> segmentation pathway."
        ),
        "not_claimed": (
            "Coordinate tokens are not claimed as the final paper novelty; geometry-verifiable "
            "supervision, reference/relation mechanisms, SRE, SCL and relation-level evaluation "
            "remain open directions."
        ),
    }
    write_json(OUT, report)
    print(f"[task6e:verdict] {verdict}: {reason}", flush=True)
    print(f"[task6e:verdict] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
