"""Task 6H sections 5-9: setup verification and one-pair gradient smoke.

Confirms the Task 6G architecture is unchanged (256x256 / 32 channels, same head dimensions), then
runs one **pair** optimizer step on a real canonical pair and records:

* both instructions' region scores before/after, the own-minus-cross margins and `L_cf`;
* that the one step moved the `[BOX]` row, the `[SEG]` row and every dense-head parameter;
* that the shared frozen SAM2 feature and every SAM2 parameter stayed **bit-identical**;
* that 16 ordinary rows and the base embedding stayed exactly unchanged;
* that the `[BOX]` query stayed causally clean (bit-identical hidden under future-text mutation);
* optimizer coverage and the pair-step loss weights.

Writes `evaluation/task6h_token_setup.json`.
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
from buildreasonseg_mvp.counterfactual import region_scores  # noqa: E402
from buildreasonseg_mvp.dense_eval import forward_heatmap  # noqa: E402
from buildreasonseg_mvp.dense_grounding import (  # noqa: E402
    feature_level_report,
    feature_tensor_for_grid,
)
from buildreasonseg_mvp.qwen_seg import (  # noqa: E402
    base_embedding_weight,
    effective_token_rows,
    token_adapter_state,
)
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6h_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    h0_pairs,
    load_pair_config,
    optimizer_and_scheduler,
    optimizer_coverage,
    write_json,
)

OUT = EVAL / "task6h_token_setup.json"
ORDINARY_ROWS_PROBED = 16


def main() -> int:
    started = time.time()
    cfg, payload = load_pair_config()
    grid = int(cfg["dense_grounding"]["grid"])
    margin = float(cfg["counterfactual"]["margin"])
    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)

    pair, record_a, record_b = h0_pairs(canonical_train_pairs(payload), 10)[0]
    sample_a, sample_b = data_mod.to_sample(record_a), data_mod.to_sample(record_b)
    image = sample_a.image_rgb()
    features, _cached = runtime.features_for(sample_a, image)
    level_report = feature_level_report(features)
    spatial_feature = feature_tensor_for_grid(features, grid)
    in_channels = int(spatial_feature.shape[1])
    head_report = runtime.install_dense_head(in_channels=in_channels)
    trainables = runtime.freeze_for_counterfactual()

    box_id = int(runtime.box_setup.box_token_id)
    seg_id = int(runtime.model.seg_token_id)

    def _batch(sample):
        return build_box_query_batch(
            runtime.processor, runtime.tokenizer, image, sample.instruction_zh,
            sample.reasoning_zh, box_id, seg_id, append_eos=True,
        )

    batch_a, batch_b = _batch(sample_a), _batch(sample_b)

    # ---- architecture identity (Task 6G frozen) ---------------------------
    architecture = {
        "grid": grid,
        "selected_level_shape": list(tuple(spatial_feature.shape)),
        "in_channels": in_channels,
        "head": head_report,
        "task6g_reference": {
            "grid": 256,
            "selected_level_shape": [1, 32, 256, 256],
            "head_parameters": 270593,
            "source": "evaluation/task6g_token_setup.json",
        },
        "unchanged_from_task6g": bool(
            grid == 256
            and list(tuple(spatial_feature.shape)) == [1, 32, 256, 256]
            and int(head_report["parameters"]) == 270593
        ),
    }

    # ---- region scores before the step -----------------------------------
    raw_a = forward_heatmap(runtime, sample_a, image=image, features=features)
    raw_b = forward_heatmap(runtime, sample_b, image=image, features=features)
    scores_before = region_scores(raw_a["logits"], raw_b["logits"], raw_a["soft_target"], raw_b["soft_target"])

    # ---- causal bit-identity (query unchanged) ---------------------------
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
        "ab_hidden_distance": float((hidden_a - hidden_b).norm()),
    }

    # ---- one PAIR optimizer step -----------------------------------------
    trainable_ids = [int(value) for value in runtime.reports["lora"]["trainable_token_ids"]]
    rows_before = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder).clone()
    head_before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.dense_head.named_parameters()
    }
    sam_before = {
        name: parameter.detach().clone() for name, parameter in runtime.model.sam.named_parameters()
    }
    feature_before = spatial_feature.detach().clone()
    ordinary_ids = [
        token_id for token_id in range(len(runtime.tokenizer)) if token_id not in set(trainable_ids)
    ][:ORDINARY_ROWS_PROBED]
    ordinary_before = base_embedding_weight(runtime.model.qwen)[ordinary_ids].clone()
    base_before = base_embedding_weight(runtime.model.qwen).clone()

    optimizer, scheduler, groups = optimizer_and_scheduler(runtime, total_steps=1, warmup_steps=0)
    step = runtime.pair_train_step(
        batch_a, batch_b, sample_a.target_mask(), sample_b.target_mask(), features, optimizer=optimizer
    )

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
            feature_tensor_for_grid(
                runtime.features_for(sample_a, image)[0], grid
            ).detach()
            - feature_before
        ).abs().max()
    )
    ordinary_delta = float(
        (base_embedding_weight(runtime.model.qwen)[ordinary_ids] - ordinary_before).abs().amax().item()
    )

    # ---- region scores / margins after the step --------------------------
    raw_a_after = forward_heatmap(runtime, sample_a, image=image, features=features)
    raw_b_after = forward_heatmap(runtime, sample_b, image=image, features=features)
    scores_after = region_scores(
        raw_a_after["logits"], raw_b_after["logits"], raw_a_after["soft_target"], raw_b_after["soft_target"]
    )

    report = {
        "_doc": (
            "Task 6H sections 5-9. Architecture-identity check, causal-placement re-verification and "
            "one-pair optimizer-step smoke on a real canonical pair. Every claim comes from a real "
            "forward/backward; nothing here is assumed."
        ),
        "task": "6H",
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
        "region_scores_before": scores_before["raw"],
        "region_scores_after": scores_after["raw"],
        "one_pair_step": {
            "losses": step["losses"],
            "weights": step["weights"],
            "region_scores": step["region_scores"],
            "pair_ranking_pass": step["pair_ranking_pass"],
            "strict_margin_pass": step["strict_margin_pass"],
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
            "shared_feature_max_abs_delta": feature_max_delta,
            "shared_feature_bit_identical": bool(feature_max_delta == 0.0),
            "visual_tower_lora_modules": runtime.reports["lora"]["visual_lora_modules"],
        },
        "margin": margin,
        "seconds": round(time.time() - started, 2),
    }
    smoke = report["one_pair_step"]
    report["passed"] = bool(
        architecture["unchanged_from_task6g"]
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
        and smoke["shared_feature_bit_identical"]
        and not smoke["visual_tower_lora_modules"]
    )
    write_json(OUT, report)
    print(
        f"[task6h:token] arch unchanged {architecture['unchanged_from_task6g']}; causal "
        f"{causal['box_hidden_bit_identical']}; box/seg row deltas {smoke['box_row_delta']:.2e}/"
        f"{smoke['seg_row_delta']:.2e}; SAM2 max delta {sam_max_delta}; feature delta "
        f"{feature_max_delta}; margins before "
        f"{scores_before['raw']['margin_a']:.3f}/{scores_before['raw']['margin_b']:.3f} -> after "
        f"{scores_after['raw']['margin_a']:.3f}/{scores_after['raw']['margin_b']:.3f}; passed "
        f"{report['passed']}",
        flush=True,
    )
    print(f"[task6h:token] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if report["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
