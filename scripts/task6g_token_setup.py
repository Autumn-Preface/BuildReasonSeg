"""Task 6G sections 2, 5-10: dense grounding setup and one-step gradient smoke.

Builds the runtime with the reused `[BOX]` query, installs the `DenseSpatialGroundingHead` on the
oracle-selected SAM2 feature level, and proves on a real forward/backward that:

* every SAM2 spatial level's shape is measured (64 = main embedding; 128/256 = high-res levels
  picked by spatial size), and the head operates on the selected one;
* the head emits a `[H,W]` heatmap whose GT soft target (instruction-selected mask, area
  downsampled) has mass > 0;
* one optimizer step moves the `[BOX]` row, the `[SEG]` row and the head parameters with non-zero
  gradient, leaves 16 ordinary rows and the base embedding tensor exactly unchanged, and leaves
  every SAM2 parameter bit-identical (backbone-freeze proof, section 6);
* the reused `[BOX]` query stays causally clean: replacing the future reasoning tail leaves its
  hidden bit-identical (section 4);
* the visual tower carries zero LoRA.

Writes `evaluation/task6g_token_setup.json`.
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
from buildreasonseg_mvp.box_query import build_box_query_batch, box_hidden_from_batch  # noqa: E402
from buildreasonseg_mvp.dense_grounding import (  # noqa: E402
    downsample_target_mask,
    feature_level_report,
    feature_tensor_for_grid,
)
from buildreasonseg_mvp.qwen_seg import (  # noqa: E402
    base_embedding_weight,
    effective_token_rows,
    token_adapter_state,
)
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6g_common import (  # noqa: E402
    EVAL,
    load_dense_config,
    optimizer_and_scheduler,
    optimizer_coverage,
    write_json,
)

OUT = EVAL / "task6g_token_setup.json"
ORDINARY_ROWS_PROBED = 16


def main() -> int:
    started = time.time()
    cfg, _payload = load_dense_config()
    grid = int(cfg["dense_grounding"]["grid"])
    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)

    sample = data_mod.to_sample(data_mod.read_records("train")[0])
    image = sample.image_rgb()
    features, cached = runtime.features_for(sample, image)
    level_report = feature_level_report(features)
    spatial_feature = feature_tensor_for_grid(features, grid)
    in_channels = int(spatial_feature.shape[1])

    head_report = runtime.install_dense_head(in_channels=in_channels)
    trainables = runtime.freeze_for_dense_grounding()

    box_id = int(runtime.box_setup.box_token_id)
    seg_id = int(runtime.model.seg_token_id)
    batch = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample.instruction_zh,
        sample.reasoning_zh, box_id, seg_id, append_eos=True,
    )

    # ---- heatmap shape + supervision checks --------------------------------
    runtime.set_visual_cache_key(str(sample.image_id))
    from buildreasonseg_mvp.box_query import box_query_forward

    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=bool(cfg["training"].get("bf16_autocast", True))
    ):
        _logits, box_hidden, _seg_hidden = box_query_forward(runtime.model.qwen, batch.to(runtime.device))
        heatmap = runtime.model.dense_head(box_hidden, spatial_feature.to(runtime.device))
    soft_target = downsample_target_mask(sample.target_mask(), grid)
    supervision = {
        "heatmap_shape": list(tuple(heatmap.shape)),
        "expected_shape": [1, grid, grid],
        "soft_target_mass": float(soft_target.sum()),
        "soft_target_mass_positive": bool(float(soft_target.sum()) > 0.0),
        "soft_target_range": [float(soft_target.min()), float(soft_target.max())],
    }

    # ---- causal bit-identity (reused [BOX] query, section 4) ------------------
    alternate = build_box_query_batch(
        runtime.processor, runtime.tokenizer, image, sample.instruction_zh,
        "完全不同的推理文本 用于验证 因果隔离", box_id, seg_id, append_eos=True,
    )
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=bool(cfg["training"].get("bf16_autocast", True))
    ):
        hidden_a = box_hidden_from_batch(runtime.model.qwen, batch.to(runtime.device))
        hidden_b = box_hidden_from_batch(runtime.model.qwen, alternate.to(runtime.device))
    causal = {
        "future_text_changed": bool(
            batch.total_length != alternate.total_length
            or not torch.equal(batch.input_ids[0], alternate.input_ids[0])
        ),
        "box_hidden_bit_identical": bool(torch.equal(hidden_a, hidden_b)),
        "box_hidden_max_abs_delta": float((hidden_a - hidden_b).abs().max()),
    }

    # ---- one optimizer step ---------------------------------------------------
    trainable_ids = [int(value) for value in runtime.reports["lora"]["trainable_token_ids"]]
    rows_before = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder).clone()
    head_before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.dense_head.named_parameters()
    }
    sam_before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.sam.named_parameters()
    }
    ordinary_ids = [
        token_id for token_id in range(len(runtime.tokenizer)) if token_id not in set(trainable_ids)
    ][:ORDINARY_ROWS_PROBED]
    ordinary_before = base_embedding_weight(runtime.model.qwen)[ordinary_ids].clone()
    base_before = base_embedding_weight(runtime.model.qwen).clone()

    optimizer, scheduler, groups = optimizer_and_scheduler(runtime, total_steps=1, warmup_steps=0)
    step = runtime.dense_train_step(batch, sample.target_mask(), features, optimizer=optimizer)

    rows_after = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder)
    row_delta = (rows_after - rows_before).abs().amax(dim=-1)
    position = {token_id: index for index, token_id in enumerate(trainable_ids)}
    head_delta = {
        name: float((parameter.detach() - head_before[name]).abs().max())
        for name, parameter in runtime.model.dense_head.named_parameters()
    }
    head_grad = {
        name: None if parameter.grad is None else float(parameter.grad.detach().float().norm())
        for name, parameter in runtime.model.dense_head.named_parameters()
    }
    adapter = token_adapter_state(runtime.model.qwen)
    box_row_grad = None
    seg_row_grad = None
    if adapter is not None:
        rows = adapter["rows"]
        index = {token_id: i for i, token_id in enumerate(adapter["token_indices"])}
        if rows.grad is not None:
            box_row_grad = float(rows.grad[index[box_id]].detach().float().norm())
            seg_row_grad = float(rows.grad[index[seg_id]].detach().float().norm())
    sam_max_delta = max(
        float((parameter.detach() - sam_before[name]).abs().max())
        for name, parameter in runtime.model.sam.named_parameters()
    )
    ordinary_delta = float(
        (base_embedding_weight(runtime.model.qwen)[ordinary_ids] - ordinary_before).abs().amax().item()
    )

    report = {
        "_doc": (
            "Task 6G sections 2, 4-10. Feature-level measurement, dense-head setup and one-step "
            "gradient smoke. Every claim comes from a real forward/backward on one training "
            "sample; nothing here is assumed."
        ),
        "task": "6G",
        "grid": grid,
        "oracle_selection": cfg["_grid_oracle_selection"],
        "config": "configs/mvp/task6g_dense_grounding.yaml",
        "vocabulary": {
            **runtime.box_setup.as_dict(),
            "seg_token_id": seg_id,
            "no_loc_tokens_added": not any(
                name.startswith("<loc_") for name in runtime.tokenizer.get_added_vocab()
            ),
            "added_vocabulary": sorted(runtime.tokenizer.get_added_vocab()),
        },
        "feature_levels": level_report,
        "selected_level": {
            "grid": grid,
            "shape": list(tuple(spatial_feature.shape)),
            "in_channels": in_channels,
            "source": "image_embeddings" if grid == 64 else "high_res_features (by spatial size)",
        },
        "head": head_report,
        "trainables": trainables,
        "optimizer_groups": optimizer_coverage(groups),
        "supervision": supervision,
        "causal_placement": causal,
        "one_step_smoke": {
            "sample_id": str(sample.sample_id),
            "losses": step["losses"],
            "weights": step["weights"],
            "heatmap_dice_quality": step["heatmap_dice_quality"],
            "grad_clip_total_norm": step["grad_clip_total_norm"],
            "box_row_delta": float(row_delta[position[box_id]]),
            "seg_row_delta": float(row_delta[position[seg_id]]),
            "box_row_grad_norm": box_row_grad,
            "seg_row_grad_norm": seg_row_grad,
            "dense_head_param_max_delta": head_delta,
            "dense_head_param_grad_norms": head_grad,
            "ordinary_rows_probed": len(ordinary_ids),
            "ordinary_rows_max_abs_delta": ordinary_delta,
            "ordinary_rows_unchanged_exactly": bool(ordinary_delta == 0.0),
            "base_embedding_tensor_bit_identical": bool(
                torch.equal(base_before, base_embedding_weight(runtime.model.qwen))
            ),
            "sam2_max_abs_delta": sam_max_delta,
            "sam2_bit_identical": bool(sam_max_delta == 0.0),
            "visual_tower_lora_modules": runtime.reports["lora"]["visual_lora_modules"],
        },
        "seconds": round(time.time() - started, 2),
    }
    smoke = report["one_step_smoke"]
    report["passed"] = bool(
        supervision["expected_shape"] == supervision["heatmap_shape"]
        and supervision["soft_target_mass_positive"]
        and causal["box_hidden_bit_identical"]
        and smoke["box_row_delta"] > 0.0
        and smoke["seg_row_delta"] > 0.0
        and (box_row_grad or 0.0) > 0.0
        and (seg_row_grad or 0.0) > 0.0
        and all(value > 0.0 for value in head_delta.values())
        and all((value or 0.0) > 0.0 for value in head_grad.values())
        and smoke["ordinary_rows_unchanged_exactly"]
        and smoke["base_embedding_tensor_bit_identical"]
        and smoke["sam2_bit_identical"]
        and not smoke["visual_tower_lora_modules"]
    )
    write_json(OUT, report)
    print(
        f"[task6g:token] grid {grid} level {tuple(spatial_feature.shape)}; causal bit-identical "
        f"{causal['box_hidden_bit_identical']}; box/seg row deltas {smoke['box_row_delta']:.2e}/"
        f"{smoke['seg_row_delta']:.2e}; ordinary max delta {ordinary_delta}; SAM2 max delta "
        f"{sam_max_delta}; head moved {all(v > 0 for v in head_delta.values())}; passed "
        f"{report['passed']}",
        flush=True,
    )
    print(f"[task6g:token] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if report["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
