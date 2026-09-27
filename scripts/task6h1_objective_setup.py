"""Task 6H.1 sections 4-11: objective setup, gradient audit and one-pair smoke.

Verifies on the real model that:

* the architecture is unchanged from Task 6G/6H (grid 256, level `[1,32,256,256]`, head 270 593
  parameters) and the `[BOX]` query is still causally clean;
* the target point-cell convention matches the frozen interior point and the flat index `y*grid+x`;
* the point cross-entropy and the bounded pair mass are the **only** gradient-carrying terms:
  Task 6G's BCE+Dice and Task 6H's raw-logit ranking are computed detached (`requires_grad=False`),
  so they contribute exactly zero gradient (sections 8-9);
* the point-CE and bounded-`L_cf` gradients reach the query projection, the query norm, the visual
  1×1 projection, the Qwen LoRA and the token rows;
* one pair step moves the `[BOX]`/`[SEG]` rows and every head parameter, leaves 16 ordinary rows and
  the base table exactly unchanged, and leaves SAM2 plus the shared feature **bit-identical**.

Writes `evaluation/task6h1_objective_setup.json`.
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
from buildreasonseg_mvp.dense_eval import forward_heatmap  # noqa: E402
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

from task6h1_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    h0r_pairs,
    load_point_config,
    optimizer_and_scheduler,
    optimizer_coverage,
    write_json,
)

OUT = EVAL / "task6h1_objective_setup.json"
ORDINARY_ROWS_PROBED = 16


def _gradient_groups(runtime, loss) -> dict:
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


def main() -> int:
    started = time.time()
    cfg, payload = load_point_config()
    grid = int(cfg["dense_grounding"]["grid"])
    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)

    pair, record_a, record_b = h0r_pairs(canonical_train_pairs(payload), 10)[0]
    sample_a, sample_b = data_mod.to_sample(record_a), data_mod.to_sample(record_b)
    image = sample_a.image_rgb()
    features, _cached = runtime.features_for(sample_a, image)
    level_report = feature_level_report(features)
    spatial_feature = feature_tensor_for_grid(features, grid)
    in_channels = int(spatial_feature.shape[1])
    head_report = runtime.install_dense_head(in_channels=in_channels)
    trainables = runtime.freeze_for_point_objective()

    architecture = {
        "grid": grid,
        "selected_level_shape": list(tuple(spatial_feature.shape)),
        "in_channels": in_channels,
        "head": head_report,
        "task6g_reference": {"grid": 256, "level_shape": [1, 32, 256, 256], "head_parameters": 270593},
        "unchanged_from_task6g": bool(
            grid == 256
            and list(tuple(spatial_feature.shape)) == [1, 32, 256, 256]
            and int(head_report["parameters"]) == 270593
        ),
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
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=bool(cfg["training"].get("bf16_autocast", True))
    ):
        hidden_a = box_hidden_from_batch(runtime.model.qwen, batch_a.to(runtime.device))
        hidden_b = box_hidden_from_batch(runtime.model.qwen, alternate.to(runtime.device))
    causal = {
        "future_text_changed": bool(
            batch_a.total_length != alternate.total_length
            or not torch.equal(batch_a.input_ids[0], alternate.input_ids[0])
        ),
        "box_hidden_bit_identical": bool(torch.equal(hidden_a, hidden_b)),
        "box_hidden_max_abs_delta": float((hidden_a - hidden_b).abs().max()),
    }

    # ---- objective gradients (fresh forward graph) ------------------------
    def _objective_losses():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            spatial = feature_tensor_for_grid(features, grid).to(runtime.device)
            _la, hidden_a_, _sa = box_query_forward(runtime.model.qwen, batch_a.to(runtime.device))
            heatmap_a = runtime.model.dense_head(hidden_a_, spatial)
            _lb, hidden_b_, _sb = box_query_forward(runtime.model.qwen, batch_b.to(runtime.device))
            heatmap_b = runtime.model.dense_head(hidden_b_, spatial)
            point = point_cross_entropy(heatmap_a[0], cell_a.index) + point_cross_entropy(
                heatmap_b[0], cell_b.index
            )
            probabilities_a = spatial_probabilities(heatmap_a[0])
            probabilities_b = spatial_probabilities(heatmap_b[0])
            target_a = downsample_target_mask(sample_a.target_mask(), grid).to(runtime.device)
            target_b = downsample_target_mask(sample_b.target_mask(), grid).to(runtime.device)
            masses = counterfactual_masses(probabilities_a, probabilities_b, target_a, target_b)
            cf = bounded_pair_loss(masses["p_aa"], masses["p_ab"], masses["p_bb"], masses["p_ba"])
        return point, cf["loss"], masses, (heatmap_a, heatmap_b, target_a, target_b)

    # each loss is measured on its own fresh forward graph (a single graph cannot be backwarded twice)
    point_loss, _cf_loss_unused, _masses_unused, _tensors_unused = _objective_losses()
    point_gradients = _gradient_groups(runtime, point_loss)
    del point_loss, _cf_loss_unused, _masses_unused, _tensors_unused
    _point_unused, cf_loss, masses, tensors = _objective_losses()
    cf_gradients = _gradient_groups(runtime, cf_loss)

    heatmap_a, heatmap_b, target_a, target_b = tensors
    with torch.no_grad():
        legacy_heat = heatmap_loss(heatmap_a.detach().unsqueeze(1), target_a[None, None])
        legacy_ranking = region_scores(heatmap_a.detach()[0], heatmap_b.detach()[0], target_a, target_b)
    legacy = {
        "bce_dice_requires_grad": bool(legacy_heat["heatmap"].requires_grad),
        "logit_ranking_requires_grad": bool(legacy_ranking["mean_margin"].requires_grad),
        "bce_dice_value": legacy_heat["heatmap_raw"],
        "logit_ranking_margin": legacy_ranking["raw"]["mean_margin"],
        "gradient_contribution": 0.0,
        "note": "computed detached under torch.no_grad(); they are logged only (sections 8-9)",
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
    }

    # ---- one pair optimizer step ------------------------------------------
    trainable_ids = [int(value) for value in runtime.reports["lora"]["trainable_token_ids"]]
    rows_before = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder).clone()
    head_before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.dense_head.named_parameters()
    }
    sam_before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.sam.named_parameters()
    }
    feature_before = feature_tensor_for_grid(features, grid).detach().clone()
    ordinary_ids = [
        token_id for token_id in range(len(runtime.tokenizer)) if token_id not in set(trainable_ids)
    ][:ORDINARY_ROWS_PROBED]
    ordinary_before = base_embedding_weight(runtime.model.qwen)[ordinary_ids].clone()
    base_before = base_embedding_weight(runtime.model.qwen).clone()

    optimizer, scheduler, groups = optimizer_and_scheduler(runtime, total_steps=1, warmup_steps=0)
    step = runtime.bounded_pair_train_step(
        batch_a, batch_b, sample_a.target_mask(), sample_b.target_mask(), features, optimizer=optimizer
    )

    rows_after = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder)
    row_delta = (rows_after - rows_before).abs().amax(dim=-1)
    position = {token_id: index for index, token_id in enumerate(trainable_ids)}
    head_delta = {
        name: float((parameter.detach() - head_before[name]).abs().max())
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
    feature_max_delta = float(
        (
            feature_tensor_for_grid(runtime.features_for(sample_a, image)[0], grid).detach()
            - feature_before
        )
        .abs()
        .max()
    )
    ordinary_delta = float(
        (base_embedding_weight(runtime.model.qwen)[ordinary_ids] - ordinary_before).abs().amax().item()
    )

    report = {
        "_doc": (
            "Task 6H.1 sections 4-11. Architecture-identity check, target-cell convention, the "
            "training objective's gradient coverage, the zero-gradient status of the retired Task 6G "
            "BCE+Dice and Task 6H logit ranking, and a one-pair optimizer-step smoke on a real "
            "canonical pair. Every claim comes from a real forward/backward."
        ),
        "task": "6H.1",
        "pair": {
            "image_id": pair.image_id,
            "a": pair.a,
            "b": pair.b,
            "target_a": pair.target_a,
            "target_b": pair.target_b,
        },
        "architecture": architecture,
        "feature_levels": level_report,
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
            "diagnostics_require_grad": step["diagnostics"].get("legacy_terms_require_grad"),
            "grad_clip_total_norm": step["grad_clip_total_norm"],
            "box_row_delta": float(row_delta[position[box_id]]),
            "seg_row_delta": float(row_delta[position[seg_id]]),
            "box_row_grad_norm": box_row_grad,
            "seg_row_grad_norm": seg_row_grad,
            "dense_head_param_max_delta": head_delta,
            "ordinary_rows_probed": len(ordinary_ids),
            "ordinary_rows_max_abs_delta": ordinary_delta,
            "ordinary_rows_unchanged_exactly": bool(ordinary_delta == 0.0),
            "base_embedding_tensor_bit_identical": bool(
                torch.equal(base_before, base_embedding_weight(runtime.model.qwen))
            ),
            "sam2_max_abs_delta": sam_max_delta,
            "sam2_bit_identical": bool(sam_max_delta == 0.0),
            "shared_feature_max_abs_delta": feature_max_delta,
            "shared_feature_bit_identical": bool(feature_max_delta == 0.0),
            "visual_tower_lora_modules": runtime.reports["lora"]["visual_lora_modules"],
        },
        "seconds": round(time.time() - started, 2),
    }
    smoke = report["one_pair_step"]
    report["passed"] = bool(
        architecture["unchanged_from_task6g"]
        and causal["box_hidden_bit_identical"]
        and target_cells["different_indices"]
        and target_cells["index_within_range"]
        and objective["softmax_sums_to_one"]
        and not legacy["bce_dice_requires_grad"]
        and not legacy["logit_ranking_requires_grad"]
        and all(value > 0.0 for value in point_gradients["norms"].values() if value is not None)
        and all(value > 0.0 for value in cf_gradients["norms"].values() if value is not None)
        and smoke["box_row_delta"] > 0.0
        and smoke["seg_row_delta"] > 0.0
        and all(value > 0.0 for value in head_delta.values())
        and smoke["ordinary_rows_unchanged_exactly"]
        and smoke["base_embedding_tensor_bit_identical"]
        and smoke["sam2_bit_identical"]
        and smoke["shared_feature_bit_identical"]
        and not smoke["visual_tower_lora_modules"]
        and smoke["diagnostics_require_grad"] is False
    )
    write_json(OUT, report)
    print(
        f"[task6h1:setup] arch unchanged {architecture['unchanged_from_task6g']}; target cells "
        f"{cell_a.index}/{cell_b.index}; point CE grads {point_gradients['norms']}; cf grads "
        f"{cf_gradients['norms']}; legacy requires_grad "
        f"{legacy['bce_dice_requires_grad']}/{legacy['logit_ranking_requires_grad']}; SAM2 delta "
        f"{sam_max_delta}; passed {report['passed']}",
        flush=True,
    )
    print(f"[task6h1:setup] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if report["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
