"""Task 6F sections 3-4 and 8: token setup, causal placement and one-step gradient smoke.

Adds exactly one `[BOX]` query token on top of the existing `[SEG]`, attaches the
`TargetAwareBoxHead`, and then *proves* on the real model that:

* `[BOX]` is one token, unique from `[SEG]`, and round-trips through decode;
* the query is inserted (never a prediction target) at exactly the position after the
  assistant-generation prefix, before every target token;
* the `[BOX]` hidden is bit-identical when the future reasoning text is changed — the direct
  causal-position proof that it cannot attend to future tokens (sections 4 and 9);
* one optimizer step moves the `[BOX]` row, the `[SEG]` row and the box head, leaves 16
  ordinary rows and the base embedding tensor exactly unchanged, and gives the box head and the
  query row non-zero gradient (section 8);
* the `[SEG]` output row still receives gradient (it must remain emittable for section 14);
* the visual tower carries zero LoRA and SAM2 is fully frozen.

Writes `evaluation/task6f_token_setup.json`.
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
from buildreasonseg_mvp.box_query import build_box_query_batch  # noqa: E402
from buildreasonseg_mvp.qwen_seg import (  # noqa: E402
    base_embedding_weight,
    effective_token_rows,
    output_row_gradients,
    token_adapter_state,
)
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6f_common import (  # noqa: E402
    EVAL,
    load_box_config,
    optimizer_and_scheduler,
    optimizer_coverage,
    write_json,
)

OUT = EVAL / "task6f_token_setup.json"
ORDINARY_ROWS_PROBED = 16


def main() -> int:
    started = time.time()
    cfg, _payload = load_box_config()
    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    head_report = runtime.install_box_head(
        hidden_dim=int(cfg["box_query"]["head_hidden_dim"]),
        mid_dim=int(cfg["box_query"]["head_mid_dim"]),
    )
    trainables = runtime.freeze_for_box_query()

    tokenizer = runtime.tokenizer
    box_id = int(runtime.box_setup.box_token_id)
    seg_id = int(runtime.model.seg_token_id)

    sample = data_mod.to_sample(data_mod.read_records("train")[0])
    image = sample.image_rgb()
    batch = build_box_query_batch(
        runtime.processor,
        tokenizer,
        image,
        sample.instruction_zh,
        sample.reasoning_zh,
        box_id,
        seg_id,
        append_eos=True,
    )

    # ---- structural checks (section 4) -----------------------------------
    input_ids = batch.input_ids[0]
    box_positions = (input_ids == box_id).nonzero(as_tuple=False).flatten().tolist()
    seg_positions = (input_ids == seg_id).nonzero(as_tuple=False).flatten().tolist()
    structure = {
        "box_token_count": len(box_positions),
        "box_position": int(batch.box_position),
        "prompt_length": int(batch.prompt_length),
        "box_immediately_after_prompt": bool(
            len(box_positions) == 1 and box_positions[0] == batch.prompt_length
        ),
        "box_before_all_target_tokens": bool(
            box_positions and box_positions[0] < min(seg_positions) if seg_positions else False
        ),
        "box_not_a_prediction_target": bool(int(batch.labels[0, batch.prompt_length - 1]) == -100),
        "labels_start_at_box_position": bool(int(batch.labels[0, batch.box_position]) == int(input_ids[batch.box_position + 1])),
        "seg_token_count": len(seg_positions),
    }

    # ---- causal bit-identity test: the query hidden must not see future tokens ----
    alternate = build_box_query_batch(
        runtime.processor,
        tokenizer,
        image,
        sample.instruction_zh,
        "完全不同的推理文本 用于验证 因果隔离",
        box_id,
        seg_id,
        append_eos=True,
    )
    runtime.set_visual_cache_key(str(sample.image_id))
    from buildreasonseg_mvp.box_query import box_hidden_from_batch

    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=bool(cfg["training"].get("bf16_autocast", True))
    ):
        hidden_a = box_hidden_from_batch(runtime.model.qwen, batch.to(runtime.device))
        hidden_b = box_hidden_from_batch(runtime.model.qwen, alternate.to(runtime.device))
    causal = {
        "future_text_changed": batch.total_length != alternate.total_length
        or not bool(torch.equal(input_ids, alternate.input_ids[0])),
        "box_hidden_bit_identical": bool(torch.equal(hidden_a, hidden_b)),
        "box_hidden_max_abs_delta": float((hidden_a - hidden_b).abs().max()),
    }

    # ---- one optimizer step ----------------------------------------------
    trainable_ids = [int(value) for value in runtime.reports["lora"]["trainable_token_ids"]]
    rows_before = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder).clone()
    head_before = {name: parameter.detach().clone() for name, parameter in runtime.model.box_head.named_parameters()}
    ordinary_ids = [
        token_id for token_id in range(len(tokenizer)) if token_id not in set(trainable_ids)
    ][:ORDINARY_ROWS_PROBED]
    ordinary_before = base_embedding_weight(runtime.model.qwen)[ordinary_ids].clone()
    base_before = base_embedding_weight(runtime.model.qwen).clone()

    optimizer, scheduler, groups = optimizer_and_scheduler(
        runtime, total_steps=1, warmup_steps=0
    )
    step = runtime.box_query_train_step(batch, sample.target_mask(), optimizer=optimizer)

    rows_after = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder)
    row_delta = (rows_after - rows_before).abs().amax(dim=-1)
    position = {token_id: index for index, token_id in enumerate(trainable_ids)}
    head_delta = {
        name: float((parameter.detach() - head_before[name]).abs().max())
        for name, parameter in runtime.model.box_head.named_parameters()
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
    head_grad = {
        name: None if parameter.grad is None else float(parameter.grad.detach().float().norm())
        for name, parameter in runtime.model.box_head.named_parameters()
    }
    out_grad = output_row_gradients(runtime.model.qwen, [seg_id])

    ordinary_after = base_embedding_weight(runtime.model.qwen)[ordinary_ids]
    base_after = base_embedding_weight(runtime.model.qwen)
    ordinary_delta = float((ordinary_after - ordinary_before).abs().amax().item())

    report = {
        "_doc": (
            "Task 6F sections 3-4 and 8. Token setup, causal placement proof and one-step "
            "trainable-row/head smoke. Every claim comes from a real forward/backward on one "
            "training sample; nothing here is assumed."
        ),
        "task": "6F",
        "config": str(task6f_common_config_relative()),
        "vocabulary": {
            **runtime.box_setup.as_dict(),
            "seg_token_id": seg_id,
            "unique_from_seg": bool(box_id != seg_id),
            "no_loc_tokens_added": not any(name.startswith("<loc_") for name in tokenizer.get_added_vocab()),
            "added_vocabulary": sorted(tokenizer.get_added_vocab()),
        },
        "head": head_report,
        "trainables": trainables,
        "optimizer_groups": optimizer_coverage(groups),
        "structure": structure,
        "causal_placement": causal,
        "one_step_smoke": {
            "sample_id": str(sample.sample_id),
            "losses": step["losses"],
            "weights": step["weights"],
            "predicted_box": step["predicted_box"],
            "gt_box": step["gt_box"],
            "grad_clip_total_norm": step["grad_clip_total_norm"],
            "box_row_delta": float(row_delta[position[box_id]]),
            "seg_row_delta": float(row_delta[position[seg_id]]),
            "box_row_grad_norm": box_row_grad,
            "seg_row_grad_norm": seg_row_grad,
            "box_head_param_max_delta": head_delta,
            "box_head_param_grad_norms": head_grad,
            "ordinary_rows_probed": len(ordinary_ids),
            "ordinary_rows_probed_ids": ordinary_ids,
            "ordinary_rows_max_abs_delta": ordinary_delta,
            "ordinary_rows_unchanged_exactly": bool(ordinary_delta == 0.0),
            "base_embedding_tensor_bit_identical": bool(torch.equal(base_before, base_after)),
            "seg_output_row_gradient_probe": out_grad,
            "visual_tower_lora_modules": runtime.reports["lora"]["visual_lora_modules"],
            "sam_frozen": bool(not any(p.requires_grad for p in runtime.model.sam.parameters())),
        },
        "seconds": round(time.time() - started, 2),
    }
    smoke = report["one_step_smoke"]
    report["passed"] = bool(
        report["vocabulary"]["single_token_roundtrip"]
        and report["vocabulary"]["unique_from_seg"]
        and report["vocabulary"]["no_loc_tokens_added"]
        and structure["box_token_count"] == 1
        and structure["box_immediately_after_prompt"]
        and structure["box_before_all_target_tokens"]
        and structure["box_not_a_prediction_target"]
        and causal["box_hidden_bit_identical"]
        and smoke["box_row_delta"] > 0.0
        and smoke["seg_row_delta"] > 0.0
        and (box_row_grad or 0.0) > 0.0
        and (seg_row_grad or 0.0) > 0.0
        and all(value > 0.0 for value in head_delta.values())
        and all((value or 0.0) > 0.0 for value in head_grad.values())
        and smoke["ordinary_rows_unchanged_exactly"]
        and smoke["base_embedding_tensor_bit_identical"]
        and not smoke["visual_tower_lora_modules"]
        and smoke["sam_frozen"]
    )
    write_json(OUT, report)
    print(
        f"[task6f:token] box id {box_id} seg id {seg_id}; causal bit-identical "
        f"{causal['box_hidden_bit_identical']}; box/seg row deltas {smoke['box_row_delta']:.2e}/"
        f"{smoke['seg_row_delta']:.2e}; ordinary max delta {ordinary_delta}; head moved "
        f"{all(value > 0 for value in head_delta.values())}; passed {report['passed']}",
        flush=True,
    )
    print(f"[task6f:token] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if report["passed"] else 5


def task6f_common_config_relative() -> str:
    from task6f_common import CONFIG

    return str(CONFIG.relative_to(REPO_ROOT)).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
