"""Task 6I sections 3-7: refinement-block setup, gradient audit and one-pair smoke.

Verifies on the real model that:

* the coarse feature is exactly the frozen SAM2 64x64 main image embedding `[B,256,64,64]` and the
  fine feature is exactly the 256x256 / 32-channel high-res level `[B,32,256,256]` (by size, never
  by list order), and the `[BOX]` query is still causally clean;
* the block contains exactly ONE cross-attention layer (embed 256, 4 heads), the q0->q1 and FFN
  residuals, and the high-res scorer consumes q1 (bit-identical to the combined forward);
* the frozen Task 6H.1 point-cell CE and bounded pair mass are the only gradient-carrying scalar
  terms; the retired Task 6G BCE+Dice and Task 6H raw-logit ranking stay detached (zero gradient);
* gradients reach the block's query projection, coarse projection, attention Q/K/V/out, FFN, fine
  scorer, the Qwen LoRA and the token rows — and nothing reaches SAM2, the frozen Task 6G head, the
  Task 6F box head or the Task 6D grounding head;
* one pair step moves the `[BOX]`/`[SEG]` rows and every refinement-block parameter, leaves the
  frozen Task 6G head and SAM2 plus the shared features **bit-identical**, and leaves ordinary
  embedding rows unchanged;
* the 1x4096 attention weights are normalized, finite, and produce the section 7 diagnostics with
  GT used for scoring only.

Writes `evaluation/task6i_architecture_setup.json`.
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
from buildreasonseg_mvp.counterfactual import region_scores  # noqa: E402
from buildreasonseg_mvp.dense_grounding import (  # noqa: E402
    downsample_target_mask,
    feature_level_report,
    feature_tensor_for_grid,
    heatmap_loss,
)
from buildreasonseg_mvp.qwen_seg import (  # noqa: E402
    base_embedding_weight,
    effective_token_rows,
    token_adapter_state,
)
from buildreasonseg_mvp.query_refine import (  # noqa: E402
    COARSE_CHANNELS,
    COARSE_GRID,
    FINE_CHANNELS,
    FINE_GRID,
    attention_diagnostics,
)
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_objective import (  # noqa: E402
    bounded_pair_loss,
    counterfactual_masses,
    point_cross_entropy,
    spatial_entropy,
    spatial_probabilities,
    target_cell,
    topk_hit,
)

from task6i_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    h0r_pairs,
    load_refinement_config,
    optimizer_and_scheduler,
    optimizer_coverage,
    write_json,
)

OUT = EVAL / "task6i_architecture_setup.json"
ORDINARY_ROWS_PROBED = 16


def _gradient_groups(runtime, loss) -> dict:
    runtime.model.zero_grad(set_to_none=True)
    loss.backward()
    groups = {
        "refine_block.query_proj": 0.0,
        "refine_block.coarse_proj": 0.0,
        "refine_block.cross_attn": 0.0,
        "refine_block.ffn": 0.0,
        "refine_block.fine_proj": 0.0,
        "refine_block_norms_bias": 0.0,
        "qwen_lora": 0.0,
        "token_rows": 0.0,
        "frozen_or_other": 0.0,
    }
    counts = {key: 0 for key in groups}
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
        elif name.startswith("refine_block."):
            key = "refine_block_norms_bias"
        elif "lora_" in name:
            key = "qwen_lora"
        elif "trainable_tokens" in name or "token_row" in name or "token_holder" in name:
            key = "token_rows"
        else:
            key = "frozen_or_other"
        groups[key] = (groups[key] ** 2 + norm**2) ** 0.5
        counts[key] += 1
    runtime.model.zero_grad(set_to_none=True)
    return {"norms": groups, "tensor_counts": counts}


def main() -> int:
    started = time.time()
    cfg, payload = load_refinement_config()
    grid = int(cfg["dense_grounding"]["grid"])
    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg["training"].get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg["training"].get("visual_feature_cache_max_images", 512)),
        )

    pair, record_a, record_b = h0r_pairs(canonical_train_pairs(payload), 10)[0]
    sample_a, sample_b = data_mod.to_sample(record_a), data_mod.to_sample(record_b)
    image = sample_a.image_rgb()
    features, _cached = runtime.features_for(sample_a, image)
    level_report = feature_level_report(features)

    f64 = feature_tensor_for_grid(features, COARSE_GRID)
    f256 = feature_tensor_for_grid(features, FINE_GRID)
    shapes = {
        "coarse_shape": list(tuple(f64.shape)),
        "fine_shape": list(tuple(f256.shape)),
        "coarse_exact": bool(tuple(f64.shape) == (1, COARSE_CHANNELS, COARSE_GRID, COARSE_GRID)),
        "fine_exact": bool(tuple(f256.shape) == (1, FINE_CHANNELS, FINE_GRID, FINE_GRID)),
        "coarse_is_main_embedding": bool(tuple(f64.shape) == tuple(features.image_embeddings.shape)),
        "expected_coarse": [1, COARSE_CHANNELS, COARSE_GRID, COARSE_GRID],
        "expected_fine": [1, FINE_CHANNELS, FINE_GRID, FINE_GRID],
    }

    # The Task 6G head stays attached as frozen evidence only.
    old_head_report = runtime.install_dense_head(in_channels=int(f256.shape[1]))
    block_report = runtime.install_refinement_block(
        coarse_channels=int(f64.shape[1]), fine_channels=int(f256.shape[1])
    )
    trainables = runtime.freeze_for_refinement()

    architecture = {
        "grid": grid,
        "coarse": shapes,
        "level_report": level_report,
        "block": block_report,
        "single_cross_attention_layer": bool(int(block_report["cross_attention_layers"]) == 1),
        "attention_geometry": {
            "embed_dim": int(block_report["embed_dim"]),
            "num_heads": int(block_report["num_heads"]),
            "matches_spec": bool(
                int(block_report["embed_dim"]) == 256 and int(block_report["num_heads"]) == 4
            ),
        },
        "frozen_task6g_head": old_head_report,
        "frozen_task6g_head_parameters": int(old_head_report["parameters"]),
        "task6g_reference": {"grid": 256, "level_shape": [1, 32, 256, 256], "head_parameters": 270593},
    }

    # ---- target point-cell convention ------------------------------------
    cell_a = target_cell(sample_a.target_mask(), grid)
    cell_b = target_cell(sample_b.target_mask(), grid)
    target_cells = {
        "a": cell_a.as_dict(),
        "b": cell_b.as_dict(),
        "different_indices": bool(cell_a.index != cell_b.index),
        "index_convention": "index = y_cell * grid + x_cell",
        "index_within_range": bool(0 <= cell_a.index < grid * grid and 0 <= cell_b.index < grid * grid),
    }

    box_id = int(runtime.box_setup.box_token_id)
    seg_id = int(runtime.model.seg_token_id)

    def _batch(sample):
        return build_box_query_batch(
            runtime.processor, runtime.tokenizer, image, sample.instruction_zh,
            sample.reasoning_zh, box_id, seg_id, append_eos=True,
        )

    batch_a, batch_b = _batch(sample_a), _batch(sample_b)

    # ---- causal bit-identity ---------------------------------------------
    alternate = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample_a.instruction_zh,
        "完全不同的推理文本 用于验证 因果隔离", box_id, seg_id, append_eos=True,
    )
    # Runs in eval mode (LoRA dropout p=0.05 is training-only noise) and in fp32 (bf16 sdpa
    # rounds the shared prefix differently for different sequence lengths; fp32 makes the causal
    # bit-identity probe exact). The visual cache key is the correct source image.
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
        "future_text_changed": bool(
            batch_a.total_length != alternate.total_length
            or not torch.equal(batch_a.input_ids[0], alternate.input_ids[0])
        ),
        "box_hidden_bit_identical": bool(torch.equal(hidden_a, hidden_b)),
        "box_hidden_max_abs_delta": float((hidden_a - hidden_b).abs().max()),
    }

    # ---- objective gradients (fresh forward graph per loss) ---------------
    def _objective_losses():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            f64_ = feature_tensor_for_grid(features, COARSE_GRID).to(runtime.device)
            f256_ = feature_tensor_for_grid(features, FINE_GRID).to(runtime.device)
            _la, hidden_a_, _sa = box_query_forward(runtime.model.qwen, batch_a.to(runtime.device))
            q1_a_, heatmap_a_, attn_a_ = runtime.model.refine_block(hidden_a_, f64_, f256_)
            _lb, hidden_b_, _sb = box_query_forward(runtime.model.qwen, batch_b.to(runtime.device))
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
        return point, cf["loss"], masses, (
            heatmap_a_, heatmap_b_, target_a, target_b, attn_a_, attn_b_, q1_a_, q1_b_
        )

    point_loss, _cf_unused, _masses_unused, _tensors_unused = _objective_losses()
    point_gradients = _gradient_groups(runtime, point_loss)
    del point_loss, _cf_unused, _masses_unused, _tensors_unused
    _point_unused, cf_loss, masses, tensors = _objective_losses()
    cf_gradients = _gradient_groups(runtime, cf_loss)

    heatmap_a, heatmap_b, target_a, target_b, attn_a, attn_b, q1_a, q1_b = tensors
    with torch.no_grad():
        legacy_heat = heatmap_loss(heatmap_a.detach().unsqueeze(1), target_a[None, None])
        legacy_ranking = region_scores(heatmap_a.detach()[0], heatmap_b.detach()[0], target_a, target_b)
        soft64_a = downsample_target_mask(sample_a.target_mask(), COARSE_GRID)
        soft64_b = downsample_target_mask(sample_b.target_mask(), COARSE_GRID)
        attn_diag_a = attention_diagnostics(attn_a.detach()[0], soft64_a)
        attn_diag_b = attention_diagnostics(attn_b.detach()[0], soft64_b)
    # scorer identity: the deployed heatmap is exactly score(refined q1) in the SAME forward context
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        f64_check = feature_tensor_for_grid(features, COARSE_GRID).to(runtime.device)
        f256_check = feature_tensor_for_grid(features, FINE_GRID).to(runtime.device)
        q1_check, heatmap_check, _attn_check = runtime.model.refine_block(hidden_a, f64_check, f256_check)
        scorer_check = runtime.model.refine_block.score(q1_check, f256_check)
    scorer_matches_forward = bool(torch.equal(heatmap_check.detach(), scorer_check.detach()))
    legacy = {
        "bce_dice_requires_grad": bool(legacy_heat["heatmap"].requires_grad),
        "logit_ranking_requires_grad": bool(legacy_ranking["mean_margin"].requires_grad),
        "bce_dice_value": legacy_heat["heatmap_raw"],
        "logit_ranking_margin": legacy_ranking["raw"]["mean_margin"],
        "gradient_contribution": 0.0,
        "note": "computed detached under torch.no_grad(); they are logged only (Task 6H.1 sections 8-9)",
    }

    probabilities_a = spatial_probabilities(heatmap_a.detach()[0])
    probabilities_b = spatial_probabilities(heatmap_b.detach()[0])
    objective = {
        "point_ce_a": float(point_cross_entropy(heatmap_a.detach()[0], cell_a.index)),
        "point_ce_b": float(point_cross_entropy(heatmap_b.detach()[0], cell_b.index)),
        "softmax_sums_to_one": bool(
            abs(float(probabilities_a.sum()) - 1.0) < 1e-4 and abs(float(probabilities_b.sum()) - 1.0) < 1e-4
        ),
        "masses": masses["raw"],
        "pair_preference_pass": masses["pair_preference_pass"],
        "cf_loss": float(cf_loss.detach()),
        "spatial_entropy_a": spatial_entropy(probabilities_a),
        "target_cell_top1_a": topk_hit(heatmap_a.detach()[0], cell_a.index, 1),
        "mean_abs_logit_a": float(heatmap_a.detach().abs().mean()),
        "q1_norm_a": float(q1_a.norm()),
        "attention": {
            "a": attn_diag_a,
            "b": attn_diag_b,
            "rows_sum_a": float(attn_a.detach()[0].sum()),
            "rows_sum_b": float(attn_b.detach()[0].sum()),
            "finite": bool(
                torch.isfinite(attn_a.detach()).all() and torch.isfinite(attn_b.detach()).all()
            ),
            "scorer_matches_combined_forward": scorer_matches_forward,
        },
    }

    # ---- one pair optimizer step ------------------------------------------
    trainable_ids = [int(value) for value in runtime.reports["lora"]["trainable_token_ids"]]
    rows_before = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder).clone()
    refine_before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.refine_block.named_parameters()
    }
    dense_before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.dense_head.named_parameters()
    }
    sam_before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.sam.named_parameters()
    }
    f64_before = f64.detach().clone()
    f256_before = f256.detach().clone()
    ordinary_ids = [
        token_id for token_id in range(len(runtime.tokenizer)) if token_id not in set(trainable_ids)
    ][:ORDINARY_ROWS_PROBED]
    ordinary_before = base_embedding_weight(runtime.model.qwen)[ordinary_ids].clone()
    base_before = base_embedding_weight(runtime.model.qwen).clone()

    optimizer, scheduler, groups = optimizer_and_scheduler(runtime, total_steps=1, warmup_steps=0)
    step = runtime.refinement_pair_train_step(
        batch_a, batch_b, sample_a.target_mask(), sample_b.target_mask(), features, optimizer=optimizer
    )

    rows_after = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder)
    row_delta = (rows_after - rows_before).abs().amax(dim=-1)
    position = {token_id: index for index, token_id in enumerate(trainable_ids)}
    refine_delta = {
        name: float((parameter.detach() - refine_before[name]).abs().max())
        for name, parameter in runtime.model.refine_block.named_parameters()
    }
    dense_delta = {
        name: float((parameter.detach() - dense_before[name]).abs().max())
        for name, parameter in runtime.model.dense_head.named_parameters()
    }
    adapter = token_adapter_state(runtime.model.qwen)
    box_row_grad = seg_row_grad = None
    if adapter is not None and adapter["rows"].grad is not None:
        index = {token_id: i for i, token_id in enumerate(adapter["token_indices"])}
        box_row_grad = float(adapter["rows"].grad[index[box_id]].detach().float().norm())
        seg_row_grad = float(adapter["rows"].grad[index[seg_id]].detach().float().norm())
    sam_max_delta = max(
        float((parameter.detach() - sam_before[name]).abs().max())
        for name, parameter in runtime.model.sam.named_parameters()
    )
    sam_any_grad = any(
        parameter.grad is not None for _name, parameter in runtime.model.sam.named_parameters()
    )
    dense_any_grad = any(
        parameter.grad is not None for _name, parameter in runtime.model.dense_head.named_parameters()
    )
    f64_max_delta = float(
        (feature_tensor_for_grid(runtime.features_for(sample_a, image)[0], COARSE_GRID).detach() - f64_before)
        .abs()
        .max()
    )
    f256_max_delta = float(
        (feature_tensor_for_grid(runtime.features_for(sample_a, image)[0], FINE_GRID).detach() - f256_before)
        .abs()
        .max()
    )
    ordinary_delta = float(
        (base_embedding_weight(runtime.model.qwen)[ordinary_ids] - ordinary_before).abs().amax().item()
    )

    report = {
        "_doc": (
            "Task 6I sections 3-7. Exact coarse/fine feature shapes, the single-cross-attention "
            "block structure, the q1-based scorer, the frozen Task 6H.1 objective's gradient "
            "coverage, the zero-gradient status of the retired Task 6G BCE+Dice and Task 6H logit "
            "ranking, section 7 attention diagnostics, and a one-pair optimizer-step smoke on a "
            "real canonical pair. Every claim comes from a real forward/backward."
        ),
        "task": "6I",
        "pair": {
            "image_id": pair.image_id,
            "a": pair.a,
            "b": pair.b,
            "target_a": pair.target_a,
            "target_b": pair.target_b,
        },
        "architecture": architecture,
        "trainables": trainables,
        "optimizer_groups": optimizer_coverage(groups),
        "causal_placement": causal,
        "target_cells": target_cells,
        "objective": objective,
        "retired_terms": legacy,
        "gradients": {
            "point_ce": point_gradients,
            "bounded_cf": cf_gradients,
        },
        "one_pair_step": {
            "losses": step["losses"],
            "weights": step["weights"],
            "region_masses": step["region_masses"],
            "pair_preference_pass": step["pair_preference_pass"],
            "mean_abs_logit": step["mean_abs_logit"],
            "logit_std": step["logit_std"],
            "attention": step["attention"],
            "diagnostics_require_grad": step["diagnostics"].get("legacy_terms_require_grad"),
            "grad_clip_total_norm": step["grad_clip_total_norm"],
            "box_row_delta": float(row_delta[position[box_id]]),
            "seg_row_delta": float(row_delta[position[seg_id]]),
            "box_row_grad_norm": box_row_grad,
            "seg_row_grad_norm": seg_row_grad,
            "refine_block_param_max_delta": refine_delta,
            "frozen_dense_head_param_max_delta": dense_delta,
            "frozen_dense_head_bit_identical": all(value == 0.0 for value in dense_delta.values()),
            "frozen_dense_head_any_grad": dense_any_grad,
            "ordinary_rows_probed": len(ordinary_ids),
            "ordinary_rows_max_abs_delta": ordinary_delta,
            "ordinary_rows_unchanged_exactly": bool(ordinary_delta == 0.0),
            "base_embedding_tensor_bit_identical": bool(
                torch.equal(base_before, base_embedding_weight(runtime.model.qwen))
            ),
            "sam2_max_abs_delta": sam_max_delta,
            "sam2_bit_identical": bool(sam_max_delta == 0.0),
            "sam2_any_grad": sam_any_grad,
            "shared_f64_max_abs_delta": f64_max_delta,
            "shared_f256_max_abs_delta": f256_max_delta,
            "shared_features_bit_identical": bool(f64_max_delta == 0.0 and f256_max_delta == 0.0),
            "visual_tower_lora_modules": runtime.reports["lora"]["visual_lora_modules"],
        },
        "seconds": round(time.time() - started, 2),
    }
    smoke = report["one_pair_step"]
    report["passed"] = bool(
        shapes["coarse_exact"]
        and shapes["fine_exact"]
        and shapes["coarse_is_main_embedding"]
        and architecture["single_cross_attention_layer"]
        and architecture["attention_geometry"]["matches_spec"]
        and causal["box_hidden_bit_identical"]
        and target_cells["different_indices"]
        and target_cells["index_within_range"]
        and objective["softmax_sums_to_one"]
        and objective["attention"]["scorer_matches_combined_forward"]
        and abs(objective["attention"]["rows_sum_a"] - 1.0) < 1e-3
        and abs(objective["attention"]["rows_sum_b"] - 1.0) < 1e-3
        and objective["attention"]["finite"]
        and not legacy["bce_dice_requires_grad"]
        and not legacy["logit_ranking_requires_grad"]
        and all(
            value > 0.0
            for key, value in point_gradients["norms"].items()
            if key != "frozen_or_other"
        )
        and point_gradients["norms"]["frozen_or_other"] == 0.0
        and all(
            value > 0.0
            for key, value in cf_gradients["norms"].items()
            if key != "frozen_or_other"
        )
        and cf_gradients["norms"]["frozen_or_other"] == 0.0
        and smoke["box_row_delta"] > 0.0
        and smoke["seg_row_delta"] > 0.0
        and all(value > 0.0 for value in refine_delta.values())
        and smoke["frozen_dense_head_bit_identical"]
        and not smoke["frozen_dense_head_any_grad"]
        and smoke["ordinary_rows_unchanged_exactly"]
        and smoke["base_embedding_tensor_bit_identical"]
        and smoke["sam2_bit_identical"]
        and not smoke["sam2_any_grad"]
        and smoke["shared_features_bit_identical"]
        and not smoke["visual_tower_lora_modules"]
        and smoke["diagnostics_require_grad"] is False
    )
    write_json(OUT, report)
    print(
        f"[task6i:setup] coarse {shapes['coarse_exact']} {shapes['coarse_shape']} fine "
        f"{shapes['fine_exact']} {shapes['fine_shape']}; layers "
        f"{block_report['cross_attention_layers']}; causal {causal['box_hidden_bit_identical']}; "
        f"point CE grads {point_gradients['norms']}; cf grads {cf_gradients['norms']}; "
        f"legacy requires_grad {legacy['bce_dice_requires_grad']}/{legacy['logit_ranking_requires_grad']}; "
        f"SAM2 delta {sam_max_delta}; passed {report['passed']}",
        flush=True,
    )
    print(f"[task6i:setup] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if report["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
