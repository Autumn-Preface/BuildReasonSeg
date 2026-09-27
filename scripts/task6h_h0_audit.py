"""Task 6H section 13: focused H0 audit (run only when H0 fails).

Checks, without adding any architecture module:

* whether the counterfactual loss reaches **every** trainable group (query projection, Qwen LoRA,
  the `[BOX]`/`[SEG]` token rows, the 1x1 visual-key projection);
* the sign of the pair loss on the real logits (raising the own score must lower `L_cf`, raising the
  cross score must raise it);
* the [BOX] query geometry change: A/B query-hidden distance and projected-query distance **before**
  (clean initialization) vs **after** H0 training;
* whether `L_cf` decreased over the recorded H0 history;
* the shared visual feature's bit identity and the pair ordering / target-mask identity from the
  manifest.

Writes `evaluation/task6h_h0_audit.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.counterfactual import counterfactual_loss, region_score, region_scores  # noqa: E402
from buildreasonseg_mvp.dense_eval import forward_heatmap  # noqa: E402
from buildreasonseg_mvp.dense_grounding import (  # noqa: E402
    argmax_point_from_logits,
    feature_tensor_for_grid,
    point_inside_mask,
)
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6h_common import (  # noqa: E402
    EVAL,
    PAIR_MANIFEST,
    canonical_train_pairs,
    h0_pairs,
    load_pair_config,
    write_json,
)

H0_JSON = EVAL / "task6h_h0_overfit.json"
OUT = EVAL / "task6h_h0_audit.json"


def _gradient_groups(runtime, loss) -> dict:
    """Gradient norm of a loss into each trainable group (section 13)."""

    runtime.model.zero_grad(set_to_none=True)
    loss.backward()
    groups = {
        "dense_head.query_proj": 0.0,
        "dense_head.query_norm": 0.0,
        "dense_head.visual_proj": 0.0,
        "qwen_lora": 0.0,
        "token_rows": 0.0,
        "other": 0.0,
    }
    counts = {key: 0 for key in groups}
    for name, parameter in runtime.model.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm())
        if name.startswith("dense_head.query_proj"):
            key = "dense_head.query_proj"
        elif name.startswith("dense_head.query_norm"):
            key = "dense_head.query_norm"
        elif name.startswith("dense_head.visual_proj"):
            key = "dense_head.visual_proj"
        elif "lora_" in name:
            key = "qwen_lora"
        elif "trainable_tokens" in name or "token_row" in name or "token_holder" in name:
            key = "token_rows"
        else:
            key = "other"
        groups[key] = (groups[key] ** 2 + norm**2) ** 0.5
        counts[key] += 1
    runtime.model.zero_grad(set_to_none=True)
    return {"norms": groups, "tensor_counts": counts}


def _counterfactual_gradients(runtime, sample_a, sample_b, features, margin: float) -> dict:
    """Gradient of `L_cf` alone into the model's trainable groups, from a fresh forward graph."""

    from buildreasonseg_mvp.box_query import box_query_forward, build_box_query_batch
    from buildreasonseg_mvp.dense_grounding import downsample_target_mask

    image = sample_a.image_rgb()
    runtime.set_visual_cache_key(str(sample_a.image_id))
    batch_a = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample_a.instruction_zh,
        sample_a.reasoning_zh, int(runtime.box_setup.box_token_id),
        int(runtime.model.seg_token_id), append_eos=True,
    ).to(runtime.device)
    batch_b = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample_b.instruction_zh,
        sample_b.reasoning_zh, int(runtime.box_setup.box_token_id),
        int(runtime.model.seg_token_id), append_eos=True,
    ).to(runtime.device)
    grid = int(runtime.dense_grid)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        spatial = feature_tensor_for_grid(features, grid).to(runtime.device)
        _la, hidden_a, _sa = box_query_forward(runtime.model.qwen, batch_a)
        heatmap_a = runtime.model.dense_head(hidden_a, spatial).unsqueeze(1)
        _lb, hidden_b, _sb = box_query_forward(runtime.model.qwen, batch_b)
        heatmap_b = runtime.model.dense_head(hidden_b, spatial).unsqueeze(1)
        target_a = downsample_target_mask(sample_a.target_mask(), grid).to(runtime.device)[None, None]
        target_b = downsample_target_mask(sample_b.target_mask(), grid).to(runtime.device)[None, None]
        scores = region_scores(heatmap_a[0, 0], heatmap_b[0, 0], target_a[0, 0], target_b[0, 0])
        loss = counterfactual_loss(scores, margin)["loss"]
    return _gradient_groups(runtime, loss)


def _pair_geometry(runtime, triples) -> dict:
    """A/B query geometry and counterfactual quantities over the H0 pairs.

    Beyond the section 13 checklist this also measures whether a growing own-minus-cross *logit*
    margin comes with a sharper peak (localization) or merely with a larger logit scale (the
    degenerate way to satisfy a ranking loss on unbounded logits): mean peakiness, mean argmax
    probability, probability-space margin and the point-inside rate are recorded alongside.
    """

    hidden_distances, projected_distances, margins, cf_losses = [], [], [], []
    peakiness, probability_margins, inside, logit_scale = [], [], [], []
    for pair, record_a, record_b in triples:
        sample_a = data_mod.to_sample(record_a)
        sample_b = data_mod.to_sample(record_b)
        image = sample_a.image_rgb()
        features, _cached = runtime.features_for(sample_a, image)
        raw_a = forward_heatmap(runtime, sample_a, image=image, features=features)
        raw_b = forward_heatmap(runtime, sample_b, image=image, features=features)
        hidden_distances.append(float((raw_a["box_hidden"] - raw_b["box_hidden"]).norm()))
        projected_distances.append(
            float((raw_a["query_projected"] - raw_b["query_projected"]).norm())
        )
        scores = region_scores(raw_a["logits"], raw_b["logits"], raw_a["soft_target"], raw_b["soft_target"])
        margins.append(float(scores["mean_margin"].detach()))
        cf_losses.append(float(counterfactual_loss(scores)["loss"].detach()))
        for raw, sample in ((raw_a, sample_a), (raw_b, sample_b)):
            extraction = argmax_point_from_logits(raw["logits"], int(raw["grid"]))
            peakiness.append(float(extraction["peakiness"]))
            inside.append(1.0 if point_inside_mask(sample.target_mask(), extraction["point"]) else 0.0)
            logit_scale.append(float(raw["logits"].abs().mean()))
        probability_margins.append(
            0.5
            * (
                scores["raw"]["probability_margin_a"] + scores["raw"]["probability_margin_b"]
            )
        )
    count = max(len(margins), 1)
    return {
        "pairs": len(margins),
        "mean_ab_hidden_distance": float(sum(hidden_distances) / count),
        "mean_ab_projected_query_distance": float(sum(projected_distances) / count),
        "mean_own_minus_cross_margin": float(sum(margins) / count),
        "mean_probability_margin": float(sum(probability_margins) / count),
        "mean_cf_loss": float(sum(cf_losses) / count),
        "mean_argmax_peakiness": float(sum(peakiness) / max(len(peakiness), 1)),
        "mean_abs_logit": float(sum(logit_scale) / max(len(logit_scale), 1)),
        "point_inside_rate": float(sum(inside) / max(len(inside), 1)),
    }


def main() -> int:
    cfg, payload = load_pair_config()
    grid = int(cfg["dense_grounding"]["grid"])
    triples = h0_pairs(canonical_train_pairs(payload), int(cfg["h0"]["pairs"]))
    h0 = json.loads(H0_JSON.read_text(encoding="utf-8"))
    margin = float(cfg["counterfactual"]["margin"])

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    first = data_mod.to_sample(triples[0][1])
    image = first.image_rgb()
    features, _cached = runtime.features_for(first, image)
    in_channels = int(feature_tensor_for_grid(features, grid).shape[1])
    runtime.install_dense_head(in_channels=in_channels)
    runtime.freeze_for_counterfactual()

    before = _pair_geometry(runtime, triples)

    # pair-loss sign on the real logits: own-score up must lower L_cf, cross-score up must raise it.
    # This uses a *separate* leaf-logit graph so the later model-parameter gradient measurement can
    # build its own forward graph.
    sample_a = data_mod.to_sample(triples[0][1])
    sample_b = data_mod.to_sample(triples[0][2])
    raw_a = forward_heatmap(runtime, sample_a, image=image, features=features)
    raw_b = forward_heatmap(runtime, sample_b, image=image, features=features)
    logits_a = raw_a["logits"].clone().requires_grad_(True)
    logits_b = raw_b["logits"].clone().requires_grad_(True)
    target_a, target_b = raw_a["soft_target"], raw_b["soft_target"]
    scores = region_scores(logits_a, logits_b, target_a, target_b)
    cf = counterfactual_loss(scores, margin)
    cf["loss"].backward()
    def _grad(tensor):
        return float(tensor.grad) if tensor.grad is not None else None

    sign_check = {
        "d_cf_d_s_aa": _grad(scores["s_aa"]),
        "d_cf_d_s_ab": _grad(scores["s_ab"]),
        "d_cf_d_s_bb": _grad(scores["s_bb"]),
        "d_cf_d_s_ba": _grad(scores["s_ba"]),
        "own_up_lowers_loss": None,
        "cross_up_raises_loss": None,
    }
    eps = 0.5
    base = float(cf["loss"].detach())
    for label, region in (("own", target_a), ("cross", target_b)):
        perturbed_a = raw_a["logits"] + eps * (region / torch.clamp(region.sum(), min=1e-6))
        perturbed_scores = region_scores(perturbed_a, raw_b["logits"], target_a, target_b)
        perturbed = float(counterfactual_loss(perturbed_scores, margin)["loss"].detach())
        if label == "own":
            sign_check["own_up_lowers_loss"] = bool(perturbed < base)
            sign_check["own_up_loss_delta"] = perturbed - base
        else:
            sign_check["cross_up_raises_loss"] = bool(perturbed > base)
            sign_check["cross_up_loss_delta"] = perturbed - base

    # gradient of L_cf alone into the model's parameter groups (a fresh forward graph)
    gradient_groups = _counterfactual_gradients(runtime, sample_a, sample_b, features, margin)

    # shared visual feature bit identity (two independent encodes of the same image)
    features_again, _cached = runtime.features_for(first, image)
    feature_delta = float(
        (
            feature_tensor_for_grid(features, grid).detach()
            - feature_tensor_for_grid(features_again, grid).detach()
        )
        .abs()
        .max()
    )

    load_report = load_checkpoint(Path(h0["checkpoint"]["path"]), runtime.model)
    after = _pair_geometry(runtime, triples)

    manifest = json.loads(PAIR_MANIFEST.read_text(encoding="utf-8"))
    ordering = {
        "pair_count": manifest["pair_count"],
        "hash": manifest["pair_identity_sha256"],
        "overlapping_pairs": manifest["overlap_summary"]["pairs_with_overlapping_targets"],
        "h0_pair_ids_match_manifest_prefix": [
            f"{pair.a}|{pair.b}" for pair, _a, _b in triples
        ]
        == [f"{row['a']}|{row['b']}" for row in manifest["pairs"][: len(triples)]],
        "target_masks_differ_for_all_h0_pairs": all(
            not row["mask_overlap"]["same_mask"] for row in manifest["pairs"][: len(triples)]
        ),
    }

    cf_history = [
        entry["losses"]["cf"] for entry in h0["history"] if "losses" in entry
    ]
    report = {
        "_doc": (
            "Task 6H section 13. Focused H0 audit: does the counterfactual loss reach every "
            "trainable group, does it have the correct sign, did the [BOX] query geometry and the "
            "region margins move between clean initialization and the H0 checkpoint, and is the "
            "pair construction sound. No architecture module was added."
        ),
        "task": "6H",
        "h0_verdict": h0["verdict"],
        "h0_final": {
            "pair_ranking_pass": h0["final"]["pair"]["pair_ranking_pass"],
            "pair_count": h0["final"]["pair"]["pair_count"],
            "point_inside_own": h0["final"]["point_inside_own"],
            "record_count": h0["final"]["record_count"],
            "mean_own_minus_cross_margin": h0["final"]["pair"]["mean_own_minus_cross_margin"],
            "heatmap_dice": h0["final"]["heatmap_dice_diagnostic"],
        },
        "before_training": before,
        "after_training": after,
        "margin_change": {
            "before": before["mean_own_minus_cross_margin"],
            "after": after["mean_own_minus_cross_margin"],
            "delta": after["mean_own_minus_cross_margin"] - before["mean_own_minus_cross_margin"],
            "probability_margin_before": before["mean_probability_margin"],
            "probability_margin_after": after["mean_probability_margin"],
            "mean_abs_logit_before": before["mean_abs_logit"],
            "mean_abs_logit_after": after["mean_abs_logit"],
            "mean_argmax_peakiness_before": before["mean_argmax_peakiness"],
            "mean_argmax_peakiness_after": after["mean_argmax_peakiness"],
            "point_inside_rate_before": before["point_inside_rate"],
            "point_inside_rate_after": after["point_inside_rate"],
            "interpretation": (
                "the margin is measured in logit units (section 7); a margin that grows while the "
                "argmax probability, the point-inside rate and the heatmap Dice stay flat means the "
                "ranking objective was satisfied by rescaling the logit field rather than by "
                "sharpening or relocating the peak"
            ),
        },
        "cf_loss_change": {
            "before": before["mean_cf_loss"],
            "after": after["mean_cf_loss"],
            "delta": after["mean_cf_loss"] - before["mean_cf_loss"],
            "decreased": bool(after["mean_cf_loss"] < before["mean_cf_loss"]),
            "history_first": cf_history[0] if cf_history else None,
            "history_last": cf_history[-1] if cf_history else None,
        },
        "pair_loss_sign": sign_check,
        "gradient_groups_of_cf_loss": gradient_groups,
        "shared_feature_max_abs_delta": feature_delta,
        "shared_feature_bit_identical": bool(feature_delta == 0.0),
        "pair_ordering": ordering,
        "checkpoint_load": load_report,
    }
    write_json(OUT, report)
    print(
        f"[task6h:audit] cf loss {before['mean_cf_loss']:.4f} -> {after['mean_cf_loss']:.4f}; "
        f"margin {before['mean_own_minus_cross_margin']:+.4f} -> "
        f"{after['mean_own_minus_cross_margin']:+.4f}; sign own/cross "
        f"{sign_check['own_up_lowers_loss']}/{sign_check['cross_up_raises_loss']}; grad groups "
        f"{gradient_groups['norms']}",
        flush=True,
    )
    print(f"[task6h:audit] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
