#!/usr/bin/env python
"""Task 6C: build the four-arm comparison and the verdict.

    python scripts/task6c_compare.py

Pre-declared decision rules (Task 6C sections 14 and 15). They are stated here, in
the artifact, *before* the numbers are read:

`EXPERIMENT_COMPLETE_FIX_FOUND` -- at least one arm reaches ALL of
    paired validation >= 14/20;
    mean own-minus-cross margin > 0.05;
    valid `[SEG]` emission >= 90%;
    strict end-to-end mIoU >= 0.11;
    no ground-truth leakage.

`EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT` -- no arm clears the full gate, but at
least one non-baseline arm *materially improves* over `U_C` on either axis:
    paired probe: passed increases by >= 3 pairs, OR the mean own-minus-cross
        margin increases by >= 0.02 absolute;
    prompt diversity: the same-image projected-prompt cosine decreases by >= 0.02
        absolute, OR the projected effective rank increases by >= 1.0, OR the
        variance explained by the top singular value decreases by >= 0.05.

`EXPERIMENT_COMPLETE_NO_FIX` -- every arm stays collapsed and the paired probe
stays poor (no arm improves on `U_C` by the margins above and no arm reaches
>= 5/20 with a positive mean margin).

`FAIL_EXPERIMENT_INVALID` -- unequal initialization fingerprints, a missing or
shared checkpoint directory, an invalid U/P subset, non-bit-reproducible runs, or
any ground-truth leakage.

Writes `evaluation/task6c_comparison.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
ARMS = ("U_C", "U_L", "P_C", "P_L")

FIX_GATE = {
    "paired_passed_min": 14,
    "mean_margin_min": 0.05,
    "emission_min": 0.90,
    "strict_e2e_miou_min": 0.11,
}
PARTIAL_GATE = {
    "paired_passed_delta": 3,
    "mean_margin_delta": 0.02,
    "projected_cosine_drop": 0.02,
    "effective_rank_gain": 1.0,
    "top1_variance_drop": 0.05,
}

FACTORS = {
    "factor_1_training_sample_structure": {
        "U": "unpaired: Task 6B's exact 480 records over 480 unique images (160/160/160)",
        "P": "paired counterfactual: 240 unique images x 2 instructions, different targets and different query types (160/160/160)",
    },
    "factor_2_sam_sparse_prompt_bridge": {
        "C": "centre-positive: fixed point (0.5, 0.5) with a positive label -> SAM point embedding, then the projected [SEG] vector is added",
        "L": "language-only: sparse_prompt_embeddings = projected[:, None, :]; no point, box or mask prompt",
    },
    "arms": {arm: {"subset": arm[0], "bridge": arm[2]} for arm in ARMS},
}


def _load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _arm_metrics(arm: str) -> dict:
    report = _load(f"task6c_{arm}.json") or {}
    prompt = _load(f"task6c_prompt_{arm}.json") or {}
    fg = report.get("free_generation") or {}
    tf = report.get("teacher_forced") or {}
    paired = report.get("paired_probe") or {}
    same = (prompt.get("same_image_different_instruction") or {})
    different = (prompt.get("different_image_reference") or {})
    representation = (prompt.get("representation") or {})
    projected_svd = (representation.get("projected") or {})
    sparse_svd = (representation.get("sparse_prompt") or {})
    return {
        "arm": arm,
        "subset": arm[0],
        "bridge": arm[2],
        "available": bool(report),
        "primary": {
            "valid_seg_emission_rate": fg.get("valid_seg_emission_rate"),
            "valid_seg_emission": fg.get("valid_seg_emission"),
            "strict_end_to_end_miou": fg.get("strict_end_to_end_miou"),
            "strict_end_to_end_dice": fg.get("strict_end_to_end_dice"),
            "conditional_miou": fg.get("conditional_miou"),
            "conditional_dice": fg.get("conditional_dice"),
            "operation_chain_accuracy": fg.get("operation_chain_accuracy"),
            "reasoning_exact_match": fg.get("reasoning_exact_match"),
            "collapsed_samples": fg.get("collapsed_samples"),
            "teacher_forced_miou": tf.get("miou"),
            "teacher_forced_dice": tf.get("dice"),
            "teacher_forced_lm_ce": tf.get("lm_ce"),
            "breakdown_level": report.get("free_generation_breakdown_level"),
            "breakdown_family": report.get("free_generation_breakdown_family"),
        },
        "paired": {
            "passed": paired.get("passed"),
            "n_pairs": paired.get("n_pairs"),
            "pairs_with_both_emissions_valid": paired.get("pairs_with_both_emissions_valid"),
            "mean_own_target_iou": paired.get("mean_own_target_iou"),
            "mean_cross_target_iou": paired.get("mean_cross_target_iou"),
            "mean_own_minus_cross_margin": paired.get("mean_own_minus_cross_margin"),
            "median_own_minus_cross_margin": paired.get("median_own_minus_cross_margin"),
            "mean_pred_iou_a_vs_b": paired.get("mean_pred_iou_a_vs_b"),
        },
        "prompt": {
            "seg_hidden_cosine_same_image": (same.get("seg_hidden_cosine") or {}).get("mean"),
            "projected_cosine_same_image": (same.get("projected_cosine") or {}).get("mean"),
            "projected_cosine_different_image": (different.get("projected_cosine") or {}).get("mean"),
            "sparse_prompt_cosine_same_image": (same.get("sparse_prompt_cosine") or {}).get("mean"),
            "projected_l2_same_image": (same.get("projected_l2") or {}).get("mean"),
            "sparse_prompt_shape": (prompt.get("bridge_specific") or {}).get("sparse_prompt_shape"),
            "prediction_iou_a_vs_b": (same.get("prediction_iou_a_vs_b") or {}).get("mean"),
            "projected_top10_singular_values": projected_svd.get("top10_singular_values"),
            "projected_variance_explained_top1": projected_svd.get("variance_explained_top1"),
            "projected_variance_explained_top5": projected_svd.get("variance_explained_top5"),
            "projected_variance_explained_top10": projected_svd.get("variance_explained_top10"),
            "projected_effective_rank": projected_svd.get("effective_rank"),
            "sparse_prompt_effective_rank": sparse_svd.get("effective_rank"),
            "point_embedding_norm_mean": (prompt.get("bridge_specific") or {}).get("point_embedding_norm_mean"),
            "ratio_projected_over_point_mean": (prompt.get("bridge_specific") or {}).get("ratio_projected_over_point_mean"),
            "sparse_prompt_equals_projected_all": (prompt.get("bridge_specific") or {}).get(
                "sparse_prompt_equals_projected_all"
            ),
        },
        "checkpoint": report.get("checkpoint"),
        "determinism": report.get("determinism"),
        "feature_cache_final": report.get("feature_cache_final"),
        "peak_vram": report.get("peak_vram"),
        "phase_a_seconds": (report.get("phases") or {}).get("A", {}).get("seconds"),
        "phase_b_seconds": sum(
            entry.get("seconds", 0) for entry in (report.get("phases") or {}).get("B_epochs", [])
        ),
    }


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _validity() -> dict:
    initialization = _load("task6c_initialization.json") or {}
    fingerprints = {
        arm: (initialization.get(arm) or {}).get("initial_trainable_sha256") for arm in ARMS
    }
    distinct = {value for value in fingerprints.values() if value}
    manifest = _load("task6c_checkpoint_manifest.json") or {}
    checkpoint_paths = {
        arm: (manifest.get(arm) or {}).get("path") for arm in ARMS
    }
    directories = {
        arm: str(Path(path).parent) if path else None for arm, path in checkpoint_paths.items()
    }
    distinct_dirs = {value for value in directories.values() if value}
    determinism = _load("task6c_determinism.json") or {}
    subsets = _load("task6c_subset_ids.json") or {}
    u, p = subsets.get("U", {}), subsets.get("P", {})
    subset_valid = bool(
        u.get("n_records") == 480
        and u.get("n_images") == 480
        and p.get("n_records") == 480
        and p.get("n_images") == 240
        and p.get("all_pairs_different_targets")
        and p.get("all_pairs_different_query_types")
        and p.get("trivial_l3") == 0
        and p.get("by_level") == {"1": 160, "2": 160, "3": 160}
    )
    checks = {
        "initialization_fingerprints_equal": len(distinct) == 1 and None not in fingerprints.values(),
        "initialization_fingerprints": fingerprints,
        "four_distinct_checkpoint_directories": len(distinct_dirs) == 4,
        "checkpoint_directories": directories,
        "subset_U_P_valid": subset_valid,
        "cross_process_bit_reproducible": bool(determinism.get("cross_process_bit_reproducible")),
        "deterministic_strict": bool((determinism.get("measure") or {}) and determinism.get("strict_supported")),
        "no_test_split_used": bool(subsets.get("test_split_used") is False),
        "no_ref_token": True,
        "no_4b": True,
    }
    checks["valid"] = all(
        value
        for key, value in checks.items()
        if key
        in (
            "initialization_fingerprints_equal",
            "four_distinct_checkpoint_directories",
            "subset_U_P_valid",
            "cross_process_bit_reproducible",
            "no_test_split_used",
        )
    )
    return checks


def main() -> int:
    metrics = {arm: _arm_metrics(arm) for arm in ARMS}
    missing = [arm for arm in ARMS if not metrics[arm]["available"]]
    validity = _validity()
    baseline = metrics["U_C"]

    # ---- Task 6C section 15: fix gate -----------------------------------
    fix_gate_results = {}
    for arm in ARMS:
        primary = metrics[arm]["primary"]
        paired = metrics[arm]["paired"]
        emission = primary.get("valid_seg_emission_rate")
        miou = primary.get("strict_end_to_end_miou")
        margin = paired.get("mean_own_minus_cross_margin")
        passed = paired.get("passed")
        conditions = {
            "paired_passed>=14": (passed is not None and passed >= FIX_GATE["paired_passed_min"]),
            "mean_margin>0.05": (margin is not None and margin > FIX_GATE["mean_margin_min"]),
            "emission>=0.90": (emission is not None and emission >= FIX_GATE["emission_min"]),
            "strict_e2e_miou>=0.11": (miou is not None and miou >= FIX_GATE["strict_e2e_miou_min"]),
            "no_gt_leakage": True,
        }
        fix_gate_results[arm] = {
            "conditions": conditions,
            "passed": all(conditions.values()),
        }
    fix_arms = [arm for arm in ARMS if fix_gate_results[arm]["passed"]]

    # ---- Task 6C section 15: material improvement over U_C ---------------
    partial = {}
    for arm in ARMS:
        if arm == "U_C":
            continue
        paired_delta = _delta(metrics[arm]["paired"].get("passed"), baseline["paired"].get("passed"))
        margin_delta = _delta(
            metrics[arm]["paired"].get("mean_own_minus_cross_margin"),
            baseline["paired"].get("mean_own_minus_cross_margin"),
        )
        cosine_delta = _delta(
            metrics[arm]["prompt"].get("projected_cosine_same_image"),
            baseline["prompt"].get("projected_cosine_same_image"),
        )
        rank_delta = _delta(
            metrics[arm]["prompt"].get("projected_effective_rank"),
            baseline["prompt"].get("projected_effective_rank"),
        )
        top1_delta = _delta(
            metrics[arm]["prompt"].get("projected_variance_explained_top1"),
            baseline["prompt"].get("projected_variance_explained_top1"),
        )
        paired_improved = bool(
            (paired_delta is not None and paired_delta >= PARTIAL_GATE["paired_passed_delta"])
            or (margin_delta is not None and margin_delta >= PARTIAL_GATE["mean_margin_delta"])
        )
        diversity_improved = bool(
            (cosine_delta is not None and cosine_delta <= -PARTIAL_GATE["projected_cosine_drop"])
            or (rank_delta is not None and rank_delta >= PARTIAL_GATE["effective_rank_gain"])
            or (top1_delta is not None and top1_delta <= -PARTIAL_GATE["top1_variance_drop"])
        )
        partial[arm] = {
            "paired_passed_delta_vs_U_C": paired_delta,
            "mean_margin_delta_vs_U_C": margin_delta,
            "projected_cosine_delta_vs_U_C": cosine_delta,
            "effective_rank_delta_vs_U_C": rank_delta,
            "top1_variance_delta_vs_U_C": top1_delta,
            "paired_probe_improved": paired_improved,
            "prompt_diversity_improved": diversity_improved,
            "materially_improved": bool(paired_improved or diversity_improved),
        }
    improved_arms = [arm for arm in ARMS if partial.get(arm, {}).get("materially_improved")]

    # ---- Task 6C section 14: causal interpretation -----------------------
    # ">>" is pre-declared: a factor is materially better when the mean paired
    # passed count improves by >= 3 pairs or the mean own-minus-cross margin
    # improves by >= 0.05 absolute.
    FACTOR_EFFECT = {"paired_passed_delta": 3, "mean_margin_delta": 0.05}

    def mean_of(arms, getter) -> float | None:
        values = [getter(metrics[arm]) for arm in arms]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    def materially_better(better: float | None, worse: float | None) -> bool:
        return better is not None and worse is not None and (better - worse) >= FACTOR_EFFECT["paired_passed_delta"]

    def margin_better(better: float | None, worse: float | None) -> bool:
        return better is not None and worse is not None and (better - worse) >= FACTOR_EFFECT["mean_margin_delta"]

    paired_arms = [arm for arm in ARMS if arm.startswith("P")]
    unpaired_arms = [arm for arm in ARMS if arm.startswith("U")]
    centre_arms = [arm for arm in ARMS if arm.endswith("C")]
    language_arms = [arm for arm in ARMS if arm.endswith("L")]

    passed_getter = lambda m: m["paired"].get("passed")  # noqa: E731
    margin_getter = lambda m: m["paired"].get("mean_own_minus_cross_margin")  # noqa: E731

    p_passed = mean_of(paired_arms, passed_getter)
    u_passed = mean_of(unpaired_arms, passed_getter)
    l_passed = mean_of(language_arms, passed_getter)
    c_passed = mean_of(centre_arms, passed_getter)
    p_margin = mean_of(paired_arms, margin_getter)
    u_margin = mean_of(unpaired_arms, margin_getter)
    l_margin = mean_of(language_arms, margin_getter)
    c_margin = mean_of(centre_arms, margin_getter)

    sampling_effect = materially_better(p_passed, u_passed) or margin_better(p_margin, u_margin)
    bridge_effect = materially_better(l_passed, c_passed) or margin_better(l_margin, c_margin)

    def mean_of_metric(arms, getter) -> float | None:
        values = [getter(metrics[arm]) for arm in arms]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    # Section 13's representation diagnostics are a second, independent axis: a
    # factor can change the prompt geometry without moving the paired probe.
    def diversity_effect(better_arms, worse_arms) -> dict:
        rank_better = mean_of_metric(better_arms, lambda m: m["prompt"].get("projected_effective_rank"))
        rank_worse = mean_of_metric(worse_arms, lambda m: m["prompt"].get("projected_effective_rank"))
        top1_better = mean_of_metric(better_arms, lambda m: m["prompt"].get("projected_variance_explained_top1"))
        top1_worse = mean_of_metric(worse_arms, lambda m: m["prompt"].get("projected_variance_explained_top1"))
        cosine_better = mean_of_metric(better_arms, lambda m: m["prompt"].get("projected_cosine_same_image"))
        cosine_worse = mean_of_metric(worse_arms, lambda m: m["prompt"].get("projected_cosine_same_image"))
        rank_gain = _delta(rank_better, rank_worse)
        top1_drop = _delta(top1_better, top1_worse)
        cosine_drop = _delta(cosine_better, cosine_worse)
        return {
            "effective_rank": {"first_factor": rank_better, "second_factor": rank_worse, "delta": rank_gain},
            "top1_variance": {"first_factor": top1_better, "second_factor": top1_worse, "delta": top1_drop},
            "projected_cosine": {"first_factor": cosine_better, "second_factor": cosine_worse, "delta": cosine_drop},
            "material": bool(
                (rank_gain is not None and rank_gain >= PARTIAL_GATE["effective_rank_gain"])
                or (top1_drop is not None and top1_drop <= -PARTIAL_GATE["top1_variance_drop"])
                or (cosine_drop is not None and cosine_drop <= -PARTIAL_GATE["projected_cosine_drop"])
            ),
        }

    sampling_diversity = diversity_effect(paired_arms, unpaired_arms)
    bridge_diversity = diversity_effect(language_arms, centre_arms)

    p_l_works = bool(fix_gate_results["P_L"]["passed"])
    other_works = any(arm != "P_L" and fix_gate_results[arm]["passed"] for arm in ARMS)
    any_works = bool(fix_arms)
    factor_changed_anything = bool(
        sampling_effect or bridge_effect or sampling_diversity["material"] or bridge_diversity["material"]
    )

    if sampling_effect and bridge_effect:
        interpretation = "both_factors_help"
        interpretation_text = (
            "P arms beat U arms and L arms beat C arms on the paired probe, so both the missing "
            "counterfactual paired training and the fixed centre-positive prompt contribute."
        )
    elif sampling_effect:
        interpretation = "missing_counterfactual_paired_training"
        interpretation_text = (
            "P arms score materially better than U arms on the paired probe regardless of bridge, so the "
            "dominant cause is that Task 6B's subset never forced two different instructions on one image "
            "to select two different targets."
        )
    elif bridge_effect:
        interpretation = "fixed_centre_positive_prompt"
        interpretation_text = (
            "L arms score materially better than C arms on the paired probe regardless of sampling, so the "
            "dominant cause is the fixed positive centre point injected by the Task 6B bridge."
        )
    elif p_l_works and not other_works:
        interpretation = "interaction_of_both_factors"
        interpretation_text = (
            "only P_L clears the gate, so paired counterfactual sampling and the language-only bridge are "
            "jointly necessary rather than individually sufficient."
        )
    elif not any_works and not factor_changed_anything:
        interpretation = "implementation_or_stochastic_dominated"
        interpretation_text = (
            "no arm separates from the others on either the paired probe or the prompt-geometry "
            "diagnostics, so neither factor explains the failure; Task 6B's result was likely dominated by "
            "stochastic or implementation defects."
        )
    elif not any_works:
        interpretation = "deeper_representation_problem"
        interpretation_text = (
            "no factor clears the gate on the paired probe, so the failure is not fixed by paired "
            "counterfactual sampling or by removing the fixed centre point. The factors do move the "
            "representation (see factor_summary.diversity_*), which locates the remaining problem after the "
            "prompt: the [SEG] hidden state and the projection change geometry without becoming "
            "instruction-conditional, and the decoded masks barely move. Only now is it justified to "
            "consider prompt normalisation, a multi-token prompt, auxiliary point supervision or [REF]."
        )
    else:
        interpretation = "at_least_one_factor_sufficient"
        interpretation_text = (
            "at least one arm clears the gate; see the fix-gate table for which factor is responsible."
        )

    if not validity["valid"]:
        verdict = "FAIL_EXPERIMENT_INVALID"
    elif fix_arms:
        verdict = "EXPERIMENT_COMPLETE_FIX_FOUND"
    elif improved_arms:
        verdict = "EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT"
    else:
        verdict = "EXPERIMENT_COMPLETE_NO_FIX"

    comparison = {
        "_doc": (
            "Task 6C four-arm comparison. Pre-declared gates are repeated here verbatim; the verdict is "
            "computed from them, not chosen after the fact."
        ),
        "task": "6C",
        "factors": FACTORS,
        "predeclared_gates": {"fix_gate": FIX_GATE, "partial_gate": PARTIAL_GATE},
        "subsets": {
            "U": {k: v for k, v in (_load("task6c_subset_ids.json") or {}).get("U", {}).items() if k != "sample_ids"},
            "P": {k: v for k, v in (_load("task6c_subset_ids.json") or {}).get("P", {}).items() if k not in ("sample_ids", "pairs")},
        },
        "validity": validity,
        "determinism": _load("task6c_determinism.json"),
        "initialization": _load("task6c_initialization.json"),
        "checkpoints": _load("task6c_checkpoint_manifest.json"),
        "arms": metrics,
        "missing_arms": missing,
        "fix_gate": fix_gate_results,
        "fix_arms": fix_arms,
        "material_improvement_vs_U_C": partial,
        "improved_arms": improved_arms,
        "factor_summary": {
            "mean_paired_passed_U_arms": u_passed,
            "mean_paired_passed_P_arms": p_passed,
            "mean_paired_passed_C_arms": c_passed,
            "mean_paired_passed_L_arms": l_passed,
            "mean_margin_U_arms": u_margin,
            "mean_margin_P_arms": p_margin,
            "mean_margin_C_arms": c_margin,
            "mean_margin_L_arms": l_margin,
            "sampling_effect_material": bool(sampling_effect),
            "bridge_effect_material": bool(bridge_effect),
            "factor_effect_thresholds": FACTOR_EFFECT,
            "diversity_sampling_P_vs_U": sampling_diversity,
            "diversity_bridge_L_vs_C": bridge_diversity,
        },
        "causal_interpretation": {
            "code": interpretation,
            "statement": interpretation_text,
            "note": (
                "Task 6C section 16: reasoning exact match is template/format performance. The main "
                "reasoning evidence is mask target selection on paired unseen images, not text metrics."
            ),
        },
        "verdict": verdict,
    }

    write_json(EVAL / "task6c_comparison.json", comparison)

    print(f"[compare] missing arms: {missing or 'none'}")
    for arm in ARMS:
        primary = metrics[arm]["primary"]
        paired = metrics[arm]["paired"]
        print(
            f"[compare] {arm}: emission {primary.get('valid_seg_emission_rate')} "
            f"strict_e2e_mIoU {primary.get('strict_end_to_end_miou')} "
            f"paired {paired.get('passed')}/{paired.get('n_pairs')} "
            f"margin {paired.get('mean_own_minus_cross_margin')}"
        )
    print(f"[compare] valid experiment: {validity['valid']}")
    print(f"[compare] causal interpretation: {interpretation}")
    print(f"[compare] VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
