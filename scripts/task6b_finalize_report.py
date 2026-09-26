#!/usr/bin/env python
"""Task 6B: consolidate the recorded runs into the final artifact set.

    python scripts/task6b_finalize_report.py

Three Phase-B runs are on record and Task 6B section 12 requires the original and
the adjusted results to be kept:

* **original recipe (headline)** -- `configs/mvp/task6b_2b_minitrain.yaml`, the
  section 12 initial objective. Its canonical `evaluation/task6b_*.json` files are
  written by `scripts/task6b_train.py` itself.
* **original recipe, first execution** -- the same recipe run before the
  teacher-forced token-accuracy indexing bug was fixed. Re-measured with the fixed
  code in `evaluation/task6b_revalidation_original_recipe.json`, which also acts as
  the reproducibility check for the headline run.
* **adjusted recipe** -- `configs/mvp/task6b_2b_minitrain_adjusted.yaml`, the single
  permitted bounded adjustment, which did **not** improve the result. Kept in full as
  `evaluation/task6b_*_adjusted_recipe.json`.

This script attaches the recipe-adjustment and reproducibility blocks to the canonical
artifacts and verifies that the adjusted run is archived alongside them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"

#: The canonical files describe the headline (original-recipe) run. The adjusted run
#: keeps its own explicit names; the headline run does not need duplicates, because
#: the canonical files already are it.
HEADLINE = (
    "task6b_baseline.json",
    "task6b_training_report.json",
    "task6b_validation.json",
    "task6b_checkpoint_manifest.json",
)
ADJUSTED = tuple(name.replace(".json", "_adjusted_recipe.json") for name in HEADLINE)

ADJUSTMENT = {
    "allowed_by": (
        "handoff/TO_DSH.md section 12: one bounded recipe adjustment allowed after the first "
        "complete documented Phase-B epoch if one objective clearly fails"
    ),
    "headline_recipe_config": "configs/mvp/task6b_2b_minitrain.yaml",
    "adjusted_recipe_config": "configs/mvp/task6b_2b_minitrain_adjusted.yaml",
    "what_changed": [
        {
            "loss_weights": {
                "from": {"lm_ce": 2.0, "mask_bce": 2.0, "mask_dice": 1.0},
                "to": {"lm_ce": 1.5, "mask_bce": 4.0, "mask_dice": 3.0},
            }
        },
        {"phase_b_decoder_lr": {"from": 3.0e-4, "to": 1.0e-3}},
    ],
    "why": (
        "After five documented Phase-B epochs with the section 12 initial objective (the decision to "
        "adjust was taken after the first full epoch was on record), the language objective was solved "
        "(val lm_ce 3.4576 -> 0.0020, valid [SEG] emission 120/120, operation-chain accuracy 1.00, "
        "generated reasoning exact match 1.00) while the segmentation objective clearly failed: "
        "training mask Dice loss stayed flat at ~0.65-1.00 in every epoch (an all-background prediction "
        "costs ~0.02 BCE and ~1.00 Dice loss on this data), strict end-to-end val mIoU stalled at "
        "0.1102, and the paired instruction probe scored 0/20 because two different instructions on one "
        "image produced near-identical masks (mean own-target IoU 0.13885 vs mean cross-target IoU "
        "0.13873). The adjustment adopted the mask-dominant weighting Task 6A demonstrated on this exact "
        "stack and raised the mask-pathway learning rate, on the reading that the failed objective was "
        "under-trained rather than mis-specified."
    ),
    "outcome": (
        "The adjustment did NOT help. Strict end-to-end val mIoU fell from 0.11018 to 0.09498 and "
        "the paired probe stayed at 0/20, so the failure is not explained by loss weighting or by "
        "the mask-pathway learning rate. It is explained by the projection collapsing to a "
        "near-constant prompt (evaluation/task6b_prompt_diagnosis_*.json). Recorded honestly rather "
        "than reverted silently."
    ),
    "unchanged": [
        "seed 20260926",
        "section 9 subsets (train 480 / val 120 / paired 20 pairs)",
        "architecture and frozen/trainable split (ADR-013)",
        "Phase A recipe",
        "5 Phase-B epochs and the 2400-step cap",
        "early-stop patience 2",
        "the best_joint selection rule declared before Phase B",
    ],
    "not_a_search": "Exactly one adjustment was made. No other configuration was tried.",
    "files": {
        "headline_training_report": "evaluation/task6b_training_report.json",
        "headline_validation": "evaluation/task6b_validation.json",
        "headline_first_execution_remeasured": "evaluation/task6b_revalidation_original_recipe.json",
        "adjusted_training_report": "evaluation/task6b_training_report_adjusted_recipe.json",
        "adjusted_validation": "evaluation/task6b_validation_adjusted_recipe.json",
        "prompt_diagnosis_headline": "evaluation/task6b_prompt_diagnosis_original_recipe.json",
        "prompt_diagnosis_adjusted": "evaluation/task6b_prompt_diagnosis_adjusted_recipe.json",
    },
}

SELECTION_KEYS = (
    "valid_seg_emission_rate",
    "strict_end_to_end_miou",
    "conditional_miou",
    "teacher_forced_miou",
    "operation_chain_accuracy",
    "lm_ce",
)


def _load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _epoch_table(report: dict | None) -> list[dict]:
    if not report:
        return []
    return [
        {
            "epoch": entry.get("epoch"),
            "total_phase_b_steps": entry.get("total_phase_b_steps"),
            "seconds": entry.get("seconds"),
            "peak_vram": entry.get("peak_vram"),
            "selection": entry.get("selection"),
            "nan_seen": entry.get("nan_seen"),
        }
        for entry in report.get("phases", {}).get("B_epochs", []) or []
    ]


def _reproducibility(headline_val: dict, revalidation: dict | None) -> dict:
    """Did the headline re-run reproduce the first execution of the same recipe?"""

    if not revalidation:
        return {"checked": False}
    headline = headline_val.get("final_selection") or {}
    first = revalidation.get("selection") or {}
    deltas = {
        key: (
            None
            if headline.get(key) is None or first.get(key) is None
            else abs(float(headline[key]) - float(first[key]))
        )
        for key in SELECTION_KEYS
    }
    return {
        "checked": True,
        "first_execution_remeasured_report": "evaluation/task6b_revalidation_original_recipe.json",
        "first_execution_selection": first,
        "headline_selection": headline,
        "absolute_deltas": deltas,
        "max_absolute_delta": max((value for value in deltas.values() if value is not None), default=None),
        "headline_metric_reproduced_within_1e-4": bool(
            deltas["strict_end_to_end_miou"] is not None and deltas["strict_end_to_end_miou"] < 1e-4
        ),
        "all_selection_metrics_reproduced_within_1e-3": all(
            value is not None and value < 1e-3 for value in deltas.values()
        ),
        "bit_identical": all(value is not None and value == 0.0 for value in deltas.values()),
        "note": (
            "The fresh baseline and Phase A reproduced bit-identically across processes (same LM CE, "
            "same mIoU, same per-step losses), but Phase B did not: `training.deterministic: true` in the "
            "config is never consumed, because `buildreasonseg_mvp/runtime.py` calls `set_seed` and never "
            "`torch.use_deterministic_algorithms` / `torch.backends.cudnn.deterministic`. Sampled CUDA "
            "kernels in the SAM2 path are therefore not reproducible step-for-step. The divergence stays "
            "small: strict end-to-end mIoU differs by 7.8e-06 (0.11018644 vs 0.11017864) and the "
            "operation-chain accuracy and emission rate are identical. Per-epoch mIoU did differ more "
            "than that early on (epoch 1: 0.1008 vs 0.0932), which is why the section 17 +0.10 criterion "
            "should be read as marginal rather than comfortable. The first execution could not keep its "
            "checkpoint bytes -- scripts/task6b_train.py uses a module-level CHECKPOINT_DIR rather than "
            "`paths.checkpoints`, so the adjusted run overwrote artifacts/checkpoints/task6b/. The "
            "recorded sha256 values are archived in "
            "evaluation/task6b_checkpoint_manifest_adjusted_recipe.json and the canonical "
            "evaluation/task6b_checkpoint_manifest.json now describes the re-run."
        ),
    }


def main() -> int:
    missing = [name for name in HEADLINE if not (EVAL / name).is_file()]
    if missing:
        print(f"[finalize] missing headline inputs: {missing}")
        return 2

    for name, adjusted in zip(HEADLINE, ADJUSTED):
        if not (EVAL / adjusted).is_file():
            print(f"[finalize] WARNING: {adjusted} absent; adjusted run not archived")

    training = _load("task6b_training_report.json") or {}
    adjusted_training = _load("task6b_training_report_adjusted_recipe.json") or {}
    headline_val = _load("task6b_validation.json") or {}
    adjusted_val = _load("task6b_validation_adjusted_recipe.json") or {}
    revalidation = _load("task6b_revalidation_original_recipe.json")

    headline_selection = headline_val.get("final_selection") or {}
    adjusted_selection = adjusted_val.get("final_selection") or {}
    headline_paired = (headline_val.get("paired_probe") or {}).get("passed")
    adjusted_paired = (adjusted_val.get("paired_probe") or {}).get("passed")

    comparison = [
        {
            "metric": key,
            "original_recipe": headline_selection.get(key),
            "adjusted_recipe": adjusted_selection.get(key),
        }
        for key in SELECTION_KEYS
    ]
    comparison.append(
        {
            "metric": "paired_instruction_dependence",
            "original_recipe": f"{headline_paired}/20" if headline_paired is not None else None,
            "adjusted_recipe": f"{adjusted_paired}/20" if adjusted_paired is not None else None,
        }
    )

    training["recipe_adjustment"] = {
        "_doc": (
            "Recorded original versus adjusted Phase-B recipe (Task 6B section 12). Both runs start "
            "from the same clean base, the same subsets and the same seed. The canonical artifacts "
            "describe the ORIGINAL (pre-declared section 12) recipe."
        ),
        **ADJUSTMENT,
        "original_recipe_run": {
            "config_path": "configs/mvp/task6b_2b_minitrain.yaml",
            "phase_b_total_steps": training.get("phase_b_total_steps"),
            "best_joint_selection": training.get("best_joint_selection"),
            "epochs": _epoch_table(training),
            "paired_probe_passed": headline_paired,
        },
        "adjusted_recipe_run": {
            "config_path": "configs/mvp/task6b_2b_minitrain_adjusted.yaml",
            "phase_b_total_steps": adjusted_training.get("phase_b_total_steps"),
            "best_joint_selection": adjusted_training.get("best_joint_selection"),
            "epochs": _epoch_table(adjusted_training),
            "paired_probe_passed": adjusted_paired,
        },
        "comparison": comparison,
        "reproducibility_of_the_headline_recipe": _reproducibility(headline_val, revalidation),
    }
    write_json(EVAL / "task6b_training_report.json", training)

    headline_val["recipe"] = {
        "headline": "original section 12 recipe",
        "config_path": "configs/mvp/task6b_2b_minitrain.yaml",
        "adjusted_recipe_config_path": "configs/mvp/task6b_2b_minitrain_adjusted.yaml",
        "adjustment_did_help": False,
        "reproducibility": _reproducibility(headline_val, revalidation),
    }
    write_json(EVAL / "task6b_validation.json", headline_val)

    adjusted_val.setdefault("recipe", {})
    adjusted_val["recipe"].update(
        {
            "headline": "adjusted recipe (the single permitted bounded adjustment)",
            "config_path": "configs/mvp/task6b_2b_minitrain_adjusted.yaml",
            "adjustment_did_help": False,
        }
    )
    write_json(EVAL / "task6b_validation_adjusted_recipe.json", adjusted_val)

    print("[finalize] canonical artifacts describe the original (headline) recipe")
    for row in comparison:
        print(f"[finalize] {row['metric']:<34} original={row['original_recipe']} adjusted={row['adjusted_recipe']}")
    reproduced = training["recipe_adjustment"]["reproducibility_of_the_headline_recipe"]
    print(f"[finalize] headline metric reproduced within 1e-4: "
          f"{reproduced.get('headline_metric_reproduced_within_1e-4')} "
          f"(max delta {reproduced.get('max_absolute_delta')}, bit-identical {reproduced.get('bit_identical')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
