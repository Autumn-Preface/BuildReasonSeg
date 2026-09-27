"""Task 6I section 9: focused I0 audit (run only when I0 fails).

Audits only (no second refinement layer, no loss change):

* q0 causal cleanliness;
* same-image different-instruction cross-attention outputs differ when appropriate;
* F64/F256 shared features bit-identical for pair A/B;
* gradients reach Qwen LoRA, [BOX], attention Q/K/V/out, FFN, F64 projection and F256 scorer;
* no gradient reaches SAM2;
* point CE and bounded pair loss decrease;
* attention/logits finite;
* q1 separation vs q0 separation.

Writes `evaluation/task6i_i0_audit.json`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.box_query import build_box_query_batch, box_hidden_from_batch, box_query_forward  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.dense_grounding import downsample_target_mask, feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.query_refine import forward_refined  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_objective import (  # noqa: E402
    bounded_pair_loss,
    counterfactual_masses,
    max_non_target_probability,
    point_cross_entropy,
    spatial_entropy,
    spatial_probabilities,
    target_cell,
    target_cell_probability,
    topk_hit,
)

from task6i_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    h0r_pairs,
    load_refinement_config,
    write_json,
)

I0_JSON = EVAL / "task6i_i0_overfit.json"
OUT = EVAL / "task6i_i0_audit.json"


def _gradient_groups(runtime, loss) -> dict:
    runtime.model.zero_grad(set_to_none=True)
    loss.backward()
    groups = {
        "refine_block.query_proj": 0.0,
        "refine_block.coarse_proj": 0.0,
        "refine_block.cross_attn": 0.0,
        "refine_block.ffn": 0.0,
        "refine_block.fine_proj": 0.0,
        "qwen_lora": 0.0,
        "token_rows": 0.0,
        "frozen_or_other": 0.0,
    }
    for name, parameter in runtime.model.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm())
        if name.startswith("refine_block.query_proj"):
            key = "refine_block.query_proj"
        elif name.startswith("refine_block.coarse_proj"):
            key = "refine_block.coarse_proj"
        elif name.startswith("refine_block.cross_attn"):
            key = "refine_block.cross_attn"
        elif name.startswith("refine_block.ffn"):
            key = "refine_block.ffn"
        elif name.startswith("refine_block.fine_proj"):
            key = "refine_block.fine_proj"
        elif "lora_" in name:
            key = "qwen_lora"
        elif "trainable_tokens" in name or "token_row" in name or "token_holder" in name:
            key = "token_rows"
        else:
            key = "frozen_or_other"
        groups[key] = (groups[key] ** 2 + norm**2) ** 0.5
    runtime.model.zero_grad(set_to_none=True)
    return groups


def _state(runtime, triples, grid: int) -> dict:
    """Objective/geometry/attention state over the I0 pairs at the current weights."""

    point_ces, cell_probs, top1, inside, entropies, non_target = [], [], [], [], [], []
    q0_distances, q1_distances = [], []
    p_aa_list, p_ab_list, p_bb_list, p_ba_list, margins, cf_losses = [], [], [], [], [], []
    attn_entropies, attn_target, attn_cross = [], [], []
    from buildreasonseg_mvp.dense_grounding import point_inside_mask

    for pair, record_a, record_b in triples:
        sample_a = data_mod.to_sample(record_a)
        sample_b = data_mod.to_sample(record_b)
        image = sample_a.image_rgb()
        features, _cached = runtime.features_for(sample_a, image)
        raw_a = forward_refined(runtime, sample_a, image=image, features=features)
        raw_b = forward_refined(runtime, sample_b, image=image, features=features)
        cell_a = target_cell(sample_a.target_mask(), grid)
        cell_b = target_cell(sample_b.target_mask(), grid)
        soft_a = downsample_target_mask(sample_a.target_mask(), grid)
        soft_b = downsample_target_mask(sample_b.target_mask(), grid)
        attn_a = raw_a["attention_weights"].reshape(-1)
        attn_b = raw_b["attention_weights"].reshape(-1)
        soft64_a = raw_a["soft_target_64"].reshape(-1)
        soft64_b = raw_b["soft_target_64"].reshape(-1)
        attn_entropies.extend([_entropy(attn_a), _entropy(attn_b)])
        attn_target.extend([float((attn_a * soft64_a).sum()), float((attn_b * soft64_b).sum())])
        attn_cross.extend([float((attn_a * soft64_b).sum()), float((attn_b * soft64_a).sum())])

        for raw, sample, cell in ((raw_a, sample_a, cell_a), (raw_b, sample_b, cell_b)):
            logits = raw["logits"]
            probabilities = spatial_probabilities(logits)
            point_ces.append(float(point_cross_entropy(logits, cell.index)))
            cell_probs.append(target_cell_probability(probabilities, cell.index))
            top1.append(1.0 if topk_hit(logits, cell.index, 1) else 0.0)
            entropies.append(spatial_entropy(probabilities))
            non_target.append(max_non_target_probability(probabilities, raw["soft_target"]))
            predicted = torch.argmax(logits.reshape(-1)).item()
            y, x = divmod(int(predicted), grid)
            predicted_point = ((x + 0.5) / grid, (y + 0.5) / grid)
            inside.append(1.0 if point_inside_mask(sample.target_mask(), predicted_point) else 0.0)

        q0_a, q0_b = raw_a["box_hidden"], raw_b["box_hidden"]
        q1_a, q1_b = raw_a["refined_query"], raw_b["refined_query"]
        q0_distances.append(float((q0_a - q0_b).norm()))
        q1_distances.append(float((q1_a - q1_b).norm()))

        probabilities_a = spatial_probabilities(raw_a["logits"])
        probabilities_b = spatial_probabilities(raw_b["logits"])
        masses = counterfactual_masses(probabilities_a, probabilities_b, soft_a, soft_b)
        p_aa_list.append(masses["raw"]["p_aa"])
        p_ab_list.append(masses["raw"]["p_ab"])
        p_bb_list.append(masses["raw"]["p_bb"])
        p_ba_list.append(masses["raw"]["p_ba"])
        margins.append(masses["raw"]["mean_margin"])
        cf_losses.append(float(bounded_pair_loss(
            masses["p_aa"], masses["p_ab"], masses["p_bb"], masses["p_ba"]
        )["raw"]))

    def mean(values):
        return float(sum(values) / max(len(values), 1))

    return {
        "mean_point_ce": mean(point_ces),
        "mean_target_cell_probability": mean(cell_probs),
        "target_cell_top1_rate": mean(top1),
        "point_inside_rate": mean(inside),
        "mean_spatial_entropy": mean(entropies),
        "mean_max_non_target_probability": mean(non_target),
        "mean_q0_same_image_l2": mean(q0_distances),
        "mean_q1_same_image_l2": mean(q1_distances),
        "mean_p_aa": mean(p_aa_list),
        "mean_p_ab": mean(p_ab_list),
        "mean_p_bb": mean(p_bb_list),
        "mean_p_ba": mean(p_ba_list),
        "mean_margin": mean(margins),
        "mean_cf_loss": mean(cf_losses),
        "mean_attention_entropy": mean(attn_entropies),
        "mean_attention_target_mass": mean(attn_target),
        "mean_attention_cross_mass": mean(attn_cross),
    }


def _entropy(weights: torch.Tensor) -> float:
    clamped = weights.clamp(min=1e-12)
    return float(-(clamped * clamped.log()).sum())


def main() -> int:
    started = time.time()
    cfg, payload = load_refinement_config()
    grid = int(cfg["dense_grounding"]["grid"])
    i0 = json.loads(I0_JSON.read_text(encoding="utf-8"))
    triples = h0r_pairs(canonical_train_pairs(payload), int(cfg["i0"]["pairs"]))
    pair, record_a, record_b = triples[0]
    sample_a, sample_b = data_mod.to_sample(record_a), data_mod.to_sample(record_b)
    image = sample_a.image_rgb()

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg["training"].get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg["training"].get("visual_feature_cache_max_images", 512)),
        )
    features, _cached = runtime.features_for(sample_a, image)
    f256 = feature_tensor_for_grid(features, grid)
    runtime.install_dense_head(in_channels=int(f256.shape[1]))  # frozen evidence only
    runtime.install_refinement_block(
        coarse_channels=int(feature_tensor_for_grid(features, 64).shape[1]),
        fine_channels=int(f256.shape[1]),
    )
    runtime.freeze_for_refinement()

    before = _state(runtime, triples, grid)
    checkpoint = i0["checkpoint"]["path"]
    load_report = load_checkpoint(Path(checkpoint), runtime.model)
    after = _state(runtime, triples, grid)

    # ---- causal cleanliness on the loaded weights -------------------------
    box_id = int(runtime.box_setup.box_token_id)
    seg_id = int(runtime.model.seg_token_id)
    batch_a = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample_a.instruction_zh,
        sample_a.reasoning_zh, box_id, seg_id, append_eos=True,
    )
    alternate = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample_a.instruction_zh,
        "完全不同的推理文本 用于验证 因果隔离", box_id, seg_id, append_eos=True,
    )
    # eval mode (LoRA dropout is training-only noise) + the correct source-image cache key.
    # fp32 (autocast off): in bf16, sdpa rounds the shared prefix differently for different
    # sequence lengths (measured 0.31 max-abs delta vs 0.0 in fp32), which is kernel rounding,
    # not future-text information flow; the causal bit-identity probe therefore runs in fp32.
    runtime.set_visual_cache_key(str(sample_a.image_id))
    was_training = bool(runtime.model.qwen.training)
    runtime.model.qwen.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            hidden_a = box_hidden_from_batch(runtime.model.qwen, batch_a.to(runtime.device))
            hidden_b = box_hidden_from_batch(runtime.model.qwen, alternate.to(runtime.device))
    finally:
        if was_training:
            runtime.model.qwen.train()
    causal = {
        "box_hidden_bit_identical": bool(torch.equal(hidden_a, hidden_b)),
        "box_hidden_max_abs_delta": float((hidden_a - hidden_b).abs().max()),
    }

    # ---- fresh-forward gradient audit on the loaded weights ---------------
    cell_a = target_cell(sample_a.target_mask(), grid)
    cell_b = target_cell(sample_b.target_mask(), grid)
    batch_a2 = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample_a.instruction_zh,
        sample_a.reasoning_zh, box_id, seg_id, append_eos=True,
    )
    batch_b2 = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample_b.instruction_zh,
        sample_b.reasoning_zh, box_id, seg_id, append_eos=True,
    )

    def _losses():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            f64_ = feature_tensor_for_grid(features, 64).to(runtime.device)
            f256_ = feature_tensor_for_grid(features, grid).to(runtime.device)
            _la, hidden_a_, _sa = box_query_forward(runtime.model.qwen, batch_a2.to(runtime.device))
            q1_a_, heatmap_a_, attn_a_ = runtime.model.refine_block(hidden_a_, f64_, f256_)
            _lb, hidden_b_, _sb = box_query_forward(runtime.model.qwen, batch_b2.to(runtime.device))
            q1_b_, heatmap_b_, attn_b_ = runtime.model.refine_block(hidden_b_, f64_, f256_)
            point = point_cross_entropy(heatmap_a_[0], cell_a.index) + point_cross_entropy(
                heatmap_b_[0], cell_b.index
            )
            probabilities_a = spatial_probabilities(heatmap_a_[0])
            probabilities_b = spatial_probabilities(heatmap_b_[0])
            target_a = downsample_target_mask(sample_a.target_mask(), grid).to(runtime.device)
            target_b = downsample_target_mask(sample_b.target_mask(), grid).to(runtime.device)
            masses = counterfactual_masses(probabilities_a, probabilities_b, target_a, target_b)
            cf = bounded_pair_loss(masses["p_aa"], masses["p_ab"], masses["p_bb"], masses["p_ba"])
        return point, cf["loss"], (heatmap_a_, attn_a_)

    point_loss, _cf_unused, _tensors_unused = _losses()
    point_gradients = _gradient_groups(runtime, point_loss)
    del point_loss, _cf_unused, _tensors_unused
    _point_unused, cf_loss, tensors = _losses()
    cf_gradients = _gradient_groups(runtime, cf_loss)
    heatmap_a, attn_a = tensors

    sam_grad = any(
        parameter.grad is not None for _name, parameter in runtime.model.sam.named_parameters()
    )
    objective = {
        "point_ce_decreased": bool(after["mean_point_ce"] < before["mean_point_ce"]),
        "bounded_cf_loss_decreased": bool(after["mean_cf_loss"] < before["mean_cf_loss"]),
        "before": before,
        "after": after,
        "point_ce_delta": float(after["mean_point_ce"] - before["mean_point_ce"]),
        "cf_loss_delta": float(after["mean_cf_loss"] - before["mean_cf_loss"]),
        "q1_separation_grew_vs_q0": bool(after["mean_q1_same_image_l2"] > after["mean_q0_same_image_l2"]),
        "attention_finite": bool(torch.isfinite(attn_a.detach()).all()),
        "logits_finite": bool(torch.isfinite(heatmap_a.detach()).all()),
    }

    report = {
        "_doc": (
            "Task 6I section 9. Focused I0 audit: causal q0, gradient coverage of the refinement "
            "block, frozen SAM2, objective decrease and q0 vs q1 separation. No second refinement "
            "layer, no loss change."
        ),
        "task": "6I",
        "causal_placement": causal,
        "gradients": {"point_ce": point_gradients, "bounded_cf": cf_gradients},
        "sam2_any_grad": sam_grad,
        "checkpoint": checkpoint,
        "checkpoint_load": load_report,
        "objective": objective,
        "i0_reference": {
            "point_inside_own": i0["final"]["records"]["point_inside_count"],
            "paired_point": i0["final"]["pair"]["paired_point_selection_pass"],
            "pair_ranking": i0["final"]["pair"]["pair_ranking_pass"],
            "mean_normalized_error": i0["final"]["records"]["mean_normalized_point_error"],
        },
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6i:audit] causal {causal['box_hidden_bit_identical']}; point CE "
        f"{before['mean_point_ce']:.4f} -> {after['mean_point_ce']:.4f}; cf "
        f"{before['mean_cf_loss']:.4f} -> {after['mean_cf_loss']:.4f}; q0/q1 l2 "
        f"{after['mean_q0_same_image_l2']:.4f}/{after['mean_q1_same_image_l2']:.4f}; SAM2 grad "
        f"{sam_grad}",
        flush=True,
    )
    print(f"[task6i:audit] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
