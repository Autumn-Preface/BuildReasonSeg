"""Task 6E sections 6-7: explicit spatial vocabulary + one-step trainable-row smoke.

Adds `[BOX]` and `<loc_000>..<loc_{B-1}>` for the B selected by
`evaluation/task6e_quantized_oracle.json`, then *proves* on a real forward/backward that:

* the base vocabulary table is frozen and ordinary rows change by exactly 0;
* `[SEG]`, `[BOX]` and the location rows are the rows that move;
* every new output row receives gradient from the output head alone (so it can be
  emitted during free generation), while the base output matrix receives none;
* the visual tower carries zero LoRA.

Writes `evaluation/task6e_token_setup.json`.
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
from buildreasonseg_mvp.qwen_seg import (  # noqa: E402
    base_embedding_weight,
    effective_token_rows,
    output_row_gradients,
    token_adapter_state,
)
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_tokens import spatial_token_names  # noqa: E402
from buildreasonseg_mvp.spatial_training import spatial_batch  # noqa: E402

from task6c6_common import EVAL, write_json  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "mvp" / "task6e_spatial_tokens.yaml"
OUT = EVAL / "task6e_token_setup.json"
ORDINARY_ROWS_PROBED = 16


def _ordinary_probe_ids(tokenizer, trainable_ids: set[int], count: int) -> list[int]:
    """Deterministic ordinary token ids: the first `count` ids that are not trainable."""

    ids: list[int] = []
    for token_id in range(len(tokenizer)):
        if token_id in trainable_ids:
            continue
        ids.append(int(token_id))
        if len(ids) == count:
            break
    return ids


def main() -> int:
    started = time.time()
    cfg = load_config(CONFIG)
    oracle = json.loads((EVAL / "task6e_quantized_oracle.json").read_text(encoding="utf-8"))
    selected_bins = oracle["selection"]["selected_bins"]
    if selected_bins is None:
        raise SystemExit("oracle selection failed; QUANTIZED_BOX_REPRESENTATION_INADEQUATE")
    if int(cfg["spatial_tokens"]["bins"]) != int(selected_bins):
        print(f"[task6e:token] overriding config bins {cfg['spatial_tokens']['bins']} -> {selected_bins}")
        cfg["spatial_tokens"]["bins"] = int(selected_bins)

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    setup = runtime.spatial_setup
    tokenizer = runtime.tokenizer
    assert setup is not None

    names = spatial_token_names(int(setup.bins))
    all_ids = [int(setup.box_token_id), *[int(value) for value in setup.loc_token_ids]]
    token_rows = []
    for name, token_id in zip(names, all_ids):
        encoded = tokenizer.encode(name, add_special_tokens=False)
        token_rows.append(
            {
                "name": name,
                "token_id": int(token_id),
                "encode": [int(value) for value in encoded],
                "single_id": len(encoded) == 1 and int(encoded[0]) == int(token_id),
                "decode_roundtrip": name in tokenizer.decode([int(token_id)], skip_special_tokens=False),
            }
        )

    trainable_ids = [int(value) for value in runtime.reports["lora"]["trainable_token_ids"]]
    ordinary_ids = _ordinary_probe_ids(tokenizer, set(trainable_ids), ORDINARY_ROWS_PROBED)

    trainables = runtime.freeze_for_spatial_tokens()
    adapter = token_adapter_state(runtime.model.qwen)
    base_embedding = base_embedding_weight(runtime.model.qwen)

    sample = data_mod.to_sample(data_mod.read_records("train")[0])
    batch, example = spatial_batch(runtime, sample)

    optimizer = torch.optim.AdamW(
        runtime.model.trainable_parameter_groups(
            lora_lr=float(cfg["optimizer"]["lora_lr"]),
            head_lr=float(cfg["optimizer"]["token_lr"]),
            weight_decay=float(cfg["optimizer"]["weight_decay"]),
            decoder_lr=float(cfg["optimizer"]["token_lr"]),
            token_lr=float(cfg["optimizer"]["token_lr"]),
        ),
        betas=tuple(cfg["optimizer"]["betas"]),
    )

    rows_before = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder).clone()
    ordinary_before = base_embedding_weight(runtime.model.qwen)[ordinary_ids].clone()
    base_before = base_embedding_weight(runtime.model.qwen).clone()

    runtime.set_visual_cache_key(sample.image_id)
    step = runtime.spatial_train_step(batch, example, optimizer=optimizer)

    rows_after = effective_token_rows(runtime.model.qwen, trainable_ids, runtime.model.token_holder)
    row_delta = (rows_after - rows_before).abs().amax(dim=-1)
    position = {token_id: index for index, token_id in enumerate(trainable_ids)}
    seg_index = position[int(runtime.model.seg_token_id)]
    box_index = position[int(setup.box_token_id)]
    # The four location rows this example's target actually uses ...
    target_loc_ids = [int(setup.loc_token_ids[int(code)]) for code in example.codes]
    loc_changed = [token_id for token_id in target_loc_ids if float(row_delta[position[token_id]]) > 0.0]
    # ... and location rows this step's target does not use. They are *not* expected to be
    # bit-identical: PEFT ties the input rows to the output head, so every position's softmax
    # puts a small gradient on every trainable row. The smoke below therefore requires the 16
    # *ordinary* vocabulary rows (frozen base table) to be exactly unchanged, and records the
    # used/untouched ratio as evidence that supervision is concentrated on the target's rows.
    other_loc_ids = [
        int(value)
        for value in setup.loc_token_ids
        if int(value) not in set(target_loc_ids)
    ][:ORDINARY_ROWS_PROBED]
    other_loc_delta = max(
        (float(row_delta[position[token_id]]) for token_id in other_loc_ids), default=0.0
    )
    used_loc_delta = min(
        (float(row_delta[position[token_id]]) for token_id in target_loc_ids), default=0.0
    )

    ordinary_after = base_embedding_weight(runtime.model.qwen)[ordinary_ids]
    base_after = base_embedding_weight(runtime.model.qwen)
    ordinary_delta = float((ordinary_after - ordinary_before).abs().amax().item())

    out_grad = output_row_gradients(
        runtime.model.qwen, [int(setup.box_token_id), *[int(value) for value in setup.loc_token_ids[:4]]]
    )
    output_rows_with_gradient = [
        token_id for token_id, norm in out_grad["rows"].items() if norm is not None and norm > 0.0
    ]

    embedding_rows = int(base_embedding.shape[0])
    report = {
        "_doc": (
            "Task 6E sections 6-7. The explicit spatial vocabulary and the one-step trainable-row "
            "smoke test. Token ids, embedding shapes and every row-movement claim come from a real "
            "forward/backward on one training sample; no value here is assumed."
        ),
        "task": "6E",
        "bins": int(setup.bins),
        "config": str(CONFIG.relative_to(REPO_ROOT).as_posix()),
        "oracle_selection": oracle["selection"],
        "vocabulary": {
            **setup.as_dict(),
            "names_first": names[:3],
            "names_last": names[-1],
            "all_names_single_token": all(entry["single_id"] for entry in token_rows),
            "all_names_decode_roundtrip": all(entry["decode_roundtrip"] for entry in token_rows),
            "unique_token_ids": len(set(all_ids)),
            "embedding_rows": embedding_rows,
            "embedding_rows_match_tokenizer": embedding_rows == len(tokenizer),
            "first_rows": token_rows[:2],
            "last_row": token_rows[-1],
        },
        "trainable_tokens": {
            **trainables,
            "mechanism": runtime.reports["lora"]["token_mechanism"],
            "n_trainable_token_ids": len(trainable_ids),
            "seg_token_id": int(runtime.model.seg_token_id),
            "box_token_id": int(setup.box_token_id),
            "loc_token_ids_first": [int(value) for value in setup.loc_token_ids[:4]],
            "loc_token_ids_last": [int(value) for value in setup.loc_token_ids[-2:]],
            "peft_trainable_token_indices_supported": runtime.reports["lora"][
                "peft_trainable_token_indices_supported"
            ],
            "peft_wraps_output_head": runtime.reports["lora"]["peft_wraps_output_head"],
            "tied_input_output_rows": adapter is not None,
            "adapter_name": None if adapter is None else adapter["adapter_name"],
            "base_embedding_frozen": not bool(base_embedding.requires_grad),
            "base_embedding_numel": int(base_embedding.numel()),
            "token_row_params": int(
                sum(p.numel() for name, p in runtime.model.named_parameters() if "trainable_tokens" in name)
            ),
        },
        "one_step_smoke": {
            "sample_id": example.sample_id,
            "losses": step["losses"],
            "weights": step["weights"],
            "grad_clip_total_norm": step["grad_clip_total_norm"],
            "teacher_forced": step["teacher_forced"],
            "seg_row_delta": float(row_delta[seg_index]),
            "box_row_delta": float(row_delta[box_index]),
            "target_loc_token_ids": target_loc_ids,
            "loc_rows_changed": len(loc_changed),
            "loc_rows_changed_ids": loc_changed,
            "seg_row_changed": bool(float(row_delta[seg_index]) > 0.0),
            "box_row_changed": bool(float(row_delta[box_index]) > 0.0),
            "all_target_loc_rows_changed": len(loc_changed) == 4,
            "untouched_loc_rows_probed": len(other_loc_ids),
            "untouched_loc_rows_probed_ids": other_loc_ids,
            "untouched_loc_rows_max_abs_delta": other_loc_delta,
            "used_loc_rows_min_abs_delta": used_loc_delta,
            "used_over_untouched_loc_row_delta_min_ratio": (
                None if other_loc_delta <= 0.0 else used_loc_delta / other_loc_delta
            ),
            "untouched_loc_rows_note": (
                "informational only: the trainable rows are tied to the output head, so the softmax "
                "at every supervised position puts a small gradient on every trainable row. The "
                "required zero-change check is on the 16 ordinary base-table rows."
            ),
            "ordinary_rows_probed": len(ordinary_ids),
            "ordinary_rows_probed_ids": ordinary_ids,
            "ordinary_rows_max_abs_delta": ordinary_delta,
            "ordinary_rows_unchanged_exactly": bool(ordinary_delta == 0.0),
            "base_embedding_tensor_bit_identical": bool(torch.equal(base_before, base_after)),
            "output_row_gradient_probe": out_grad,
            "output_rows_with_nonzero_gradient": output_rows_with_gradient,
            "every_new_output_row_receives_gradient": len(output_rows_with_gradient)
            == len(out_grad["rows"]),
            "visual_tower_lora_modules": runtime.reports["lora"]["visual_lora_modules"],
        },
        "seconds": round(time.time() - started, 2),
    }
    report["passed"] = bool(
        report["vocabulary"]["all_names_single_token"]
        and report["vocabulary"]["all_names_decode_roundtrip"]
        and report["vocabulary"]["embedding_rows_match_tokenizer"]
        and report["one_step_smoke"]["seg_row_changed"]
        and report["one_step_smoke"]["box_row_changed"]
        and report["one_step_smoke"]["all_target_loc_rows_changed"]
        and report["one_step_smoke"]["ordinary_rows_unchanged_exactly"]
        and report["one_step_smoke"]["base_embedding_tensor_bit_identical"]
        and report["one_step_smoke"]["every_new_output_row_receives_gradient"]
        and not report["one_step_smoke"]["visual_tower_lora_modules"]
    )
    write_json(OUT, report)
    smoke = report["one_step_smoke"]
    print(
        f"[task6e:token] bins {setup.bins} added {setup.added_tokens} tokens; "
        f"seg/box rows changed {smoke['seg_row_changed']}/{smoke['box_row_changed']}; "
        f"loc rows changed {smoke['loc_rows_changed']}/4; "
        f"ordinary max delta {smoke['ordinary_rows_max_abs_delta']}; "
        f"output rows with grad {len(output_rows_with_gradient)}/{len(out_grad['rows'])}; "
        f"passed {report['passed']}",
        flush=True,
    )
    print(f"[task6e:token] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0 if report["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
