"""Task 6F tests: target-aware `[BOX]` query grounding (section 21's 22 required checks).

Model-level facts (real forward/backward) are asserted against the recorded measurement in
`evaluation/task6f_token_setup.json`; everything else is CPU-only or static so this file never
builds a second runtime and never competes for VRAM. Task 6E's coordinate-token machinery must
stay retired: no `<loc_*>` vocabulary, no location CE, no location-run parsing in any 6F path.

Run with pytest, or directly::

    python tests/test_task6f_box_query.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp.box_query import (  # noqa: E402
    BOX_QUERY_TOKEN,
    BOX_LOSS_WEIGHT,
    REASONING_LOSS_WEIGHT,
    TargetAwareBoxHead,
    box_l1,
    box_values_iou,
    build_box_query_batch,
    center_inside_mask,
)
from buildreasonseg_mvp.grounding import GEOMETRY_BOX, target_geometry  # noqa: E402
from buildreasonseg_mvp.qwen_seg import TeacherForcedBatch  # noqa: E402
from buildreasonseg_mvp.spatial_tokens import loc_name  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
QWEN_HUB = REPO_ROOT / "local_cache" / "huggingface" / "hub"


# ------------------------------------------------------------------ 1-2 vocabulary / query


@pytest.fixture(scope="module")
def qwen_tokenizer():
    if not QWEN_HUB.is_dir() or not any(QWEN_HUB.glob("models--Qwen--Qwen3-VL-2B-Instruct")):
        pytest.skip("Qwen tokenizer snapshot not present under local_cache/")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        cache_dir=str(QWEN_HUB),
        local_files_only=True,
    )


def test_box_token_is_one_token_and_unique_from_seg(qwen_tokenizer):
    tokenizer = qwen_tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": [BOX_QUERY_TOKEN, "[SEG]"]})
    box_ids = tokenizer.encode(BOX_QUERY_TOKEN, add_special_tokens=False)
    seg_ids = tokenizer.encode("[SEG]", add_special_tokens=False)
    assert len(box_ids) == 1, "[BOX] must be exactly one token"
    assert len(seg_ids) == 1
    assert box_ids[0] != seg_ids[0]
    assert BOX_QUERY_TOKEN in tokenizer.decode([box_ids[0]], skip_special_tokens=False)


def _fake_processor(prompt_ids: torch.Tensor, prompt_length: int, grid=None):
    class _Processor:
        image_processor = type("IP", (), {"merge_size": 2})()

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                                return_dict=True, return_tensors=None):
            assert tokenize and add_generation_prompt
            return {
                "input_ids": prompt_ids,
                "attention_mask": torch.ones_like(prompt_ids),
                "pixel_values": torch.zeros(1, 3, 4, 4),
                "image_grid_thw": grid,
                "mm_token_type_ids": torch.zeros(1, prompt_length, dtype=torch.long),
            }

    return _Processor()


def _fake_tokenizer(eos: int = 2):
    class _Tokenizer:
        eos_token_id = eos

        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            ids = [10 + (ord(character) % 50) for character in text]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    return _Tokenizer()


def _training_batch(box_id: int = 99, seg_id: int = 100, reasoning: str = "abc", prompt_length: int = 6):
    processor = _fake_processor(torch.arange(20, 20 + prompt_length)[None, :], prompt_length)
    tokenizer = _fake_tokenizer()
    return build_box_query_batch(
        processor, tokenizer, None, "instruction", reasoning, box_id, seg_id, append_eos=True
    )


def test_box_is_inserted_as_a_fixed_query_not_predicted():
    batch = _training_batch()
    input_ids = batch.input_ids[0]
    box_positions = (input_ids == 99).nonzero(as_tuple=False).flatten().tolist()
    assert box_positions == [batch.prompt_length], "the query sits immediately after the prompt"
    # the position that would predict [BOX] is unsupervised
    assert int(batch.labels[0, batch.prompt_length - 1]) == -100
    # supervision starts AT the query position (predicting the first reasoning token)
    assert int(batch.labels[0, batch.box_position]) == int(input_ids[batch.box_position + 1])
    # every supervised position obeys the causal label rule
    supervised = (batch.labels[0] != -100).nonzero(as_tuple=False).flatten().tolist()
    assert supervised == list(range(batch.prompt_length, batch.total_length - 1))
    for position in supervised:
        assert int(batch.labels[0, position]) == int(input_ids[position + 1])


def test_box_hidden_cannot_attend_to_future_tokens_positionally():
    batch = _training_batch(reasoning="abcdefgh")
    alternate = _training_batch(reasoning="zzzzzzzz")
    input_ids = batch.input_ids[0]
    alt_ids = alternate.input_ids[0]
    assert int((input_ids == 99).nonzero(as_tuple=False).flatten()[0]) < int(
        (input_ids == 100).nonzero(as_tuple=False).flatten()[0]
    )
    # the query and the prefix are identical; everything after it differs
    assert torch.equal(input_ids[: batch.box_position + 1], alt_ids[: alternate.box_position + 1])
    assert not torch.equal(input_ids[batch.box_position + 1 :], alt_ids[alternate.box_position + 1 :])
    # causal principle: position p's output depends only on positions <= p
    assert batch.box_position == alternate.box_position
    assert batch.box_position < int((input_ids == 100).nonzero(as_tuple=False).flatten()[0])


def test_causal_attention_principle_on_a_tiny_layer():
    """A one-layer causal attention model: mutating future tokens leaves the query output unchanged."""

    dim = 8
    query_position = 2
    embedding = nn.Embedding(40, dim)
    attention = nn.MultiheadAttention(dim, num_heads=2, batch_first=True)

    def forward(ids):
        x = embedding(ids)  # [1, T, D]
        length = ids.shape[1]
        mask = torch.triu(torch.full((length, length), float("-inf")), diagonal=1)
        output, _ = attention(x, x, x, attn_mask=mask)
        return output[0, query_position]

    base = torch.tensor([[5, 6, 7, 8, 9, 10]])
    changed = torch.tensor([[5, 6, 7, 30, 31, 32]])
    assert torch.equal(forward(base), forward(changed))


def test_query_batch_contains_no_target_tokens():
    from buildreasonseg_mvp.box_query import build_query_batch

    class _Runtime:
        box_setup = type("S", (), {"box_token_id": 99})()
        processor = _fake_processor(torch.arange(20, 26)[None, :], 6)
        tokenizer = _fake_tokenizer()

    batch = build_query_batch(_Runtime(), None, "instruction")
    assert int((batch.input_ids[0] == 99).sum()) == 1
    assert int((batch.input_ids[0] == 100).sum()) == 0
    assert int((batch.labels[0] != -100).sum()) == 0
    assert batch.box_position == batch.prompt_length == 6
    assert batch.seg_position == -1


# ------------------------------------------------------------------ 3-6 geometry / supervision


def test_box_target_uses_the_frozen_task6d_convention():
    import numpy as np

    mask = np.zeros((16, 16), dtype=bool)
    mask[4:9, 3:11] = True
    box = target_geometry(mask, GEOMETRY_BOX)
    assert box == (3 / 16, 4 / 16, 11 / 16, 9 / 16)
    assert box[0] <= box[2] and box[1] <= box[3]


def test_query_head_output_is_canonical_and_in_unit_cube():
    torch.manual_seed(0)
    head = TargetAwareBoxHead(hidden_dim=16, mid_dim=8)
    output = head(torch.randn(32, 16))
    assert output.shape == (32, 4)
    assert bool((output >= 0.0).all()) and bool((output <= 1.0).all())
    assert bool((output[:, 0] <= output[:, 2]).all())
    assert bool((output[:, 1] <= output[:, 3]).all())
    # determinism
    torch.manual_seed(0)
    again = TargetAwareBoxHead(hidden_dim=16, mid_dim=8)(torch.randn(32, 16))
    assert torch.equal(output, again)


def test_gt_box_is_supervision_only():
    import inspect

    from buildreasonseg_mvp import box_query

    train_step = inspect.getsource(box_query.box_query_loss)
    assert "target_geometry" not in train_step  # the GT box is passed in, not derived inside
    predict = inspect.getsource(box_query.predict_box_for_sample)
    assert "target_mask" not in predict and "target_geometry" not in predict
    assert "sample.instruction_zh" in predict
    forward = inspect.getsource(box_query.box_hidden_from_batch)
    assert "labels" not in forward


def test_inference_needs_only_image_instruction_and_constant_box():
    import inspect

    from buildreasonseg_mvp import box_query

    signature = inspect.signature(box_query.predict_box_for_sample)
    parameters = set(signature.parameters)
    assert not (parameters & {"mask", "gt", "box", "geometry"})
    generation = inspect.getsource(box_query.generate_with_box_prefix)
    assert "instruction" in generation and "box_token_id" in generation
    assert "target_mask" not in generation and "target_geometry" not in generation


# ------------------------------------------------------------------ 7 loss / head


def test_loss_weights_are_fixed_constants():
    assert REASONING_LOSS_WEIGHT == 1.0
    assert BOX_LOSS_WEIGHT == 5.0


def test_box_iou_and_l1_helpers():
    assert box_values_iou((0, 0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75)) == pytest.approx(0.142857, abs=1e-4)
    assert box_l1((0, 0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75)) == pytest.approx(0.25, abs=1e-9)
    assert box_values_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0


def test_center_inside():
    import numpy as np

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    assert center_inside_mask(mask, (0.0, 0.0, 0.4, 0.4)) is True  # centre rounds to (2,2): inside
    assert center_inside_mask(mask, (0.0, 0.0, 0.2, 0.2)) is False  # centre rounds to (1,1): outside
    assert center_inside_mask(mask, (0.8, 0.8, 1.0, 1.0)) is False  # centre (8,8): outside


# ------------------------------------------------------------------ recorded model-level smoke


def test_token_setup_artifact_records_the_section_8_requirements():
    setup = json.loads((EVAL / "task6f_token_setup.json").read_text(encoding="utf-8"))
    assert setup["passed"] is True
    vocabulary = setup["vocabulary"]
    assert vocabulary["single_token_roundtrip"] is True
    assert vocabulary["unique_from_seg"] is True
    assert vocabulary["no_loc_tokens_added"] is True
    assert vocabulary["added_tokens"] == 1
    structure = setup["structure"]
    assert structure["box_token_count"] == 1
    assert structure["box_immediately_after_prompt"] is True
    assert structure["box_before_all_target_tokens"] is True
    assert structure["box_not_a_prediction_target"] is True
    assert setup["causal_placement"]["box_hidden_bit_identical"] is True
    smoke = setup["one_step_smoke"]
    assert smoke["box_row_delta"] > 0.0
    assert smoke["seg_row_delta"] > 0.0
    assert (smoke["box_row_grad_norm"] or 0.0) > 0.0
    assert (smoke["seg_row_grad_norm"] or 0.0) > 0.0
    assert all(value > 0.0 for value in smoke["box_head_param_max_delta"].values())
    assert smoke["ordinary_rows_probed"] >= 16
    assert smoke["ordinary_rows_unchanged_exactly"] is True
    assert smoke["base_embedding_tensor_bit_identical"] is True
    assert smoke["visual_tower_lora_modules"] == []
    assert smoke["sam_frozen"] is True
    groups = setup["optimizer_groups"]
    by_name = {group["name"]: group for group in groups}
    assert by_name["token"]["tensors"] == 1  # one trainable-token parameter (both rows)
    assert by_name["decoder"]["tensors"] == 6  # the box head's six parameters
    assert by_name["decoder"]["parameters"] == 1_055_236  # LayerNorm + 2048x512 + 512x4
    assert "lora" in by_name


# ------------------------------------------------------------------ static hygiene


FORBIDDEN = ("[REF]", "SpatialRelationEncoder", "spatial_consistency", "Qwen3-VL-4B", "4B-Instruct")

FORBIDDEN_6F_SOURCES = (
    "buildreasonseg_mvp/box_query.py",
    "buildreasonseg_mvp/box_query_eval.py",
    "scripts/task6f_common.py",
    "scripts/task6f_token_setup.py",
    "scripts/task6f_f0.py",
    "scripts/task6f_f1.py",
    "scripts/task6f_f2.py",
    "scripts/task6f_error_analysis.py",
    "scripts/task6f_verdict.py",
)


def test_task6f_sources_contain_no_forbidden_components():
    for relative in FORBIDDEN_6F_SOURCES:
        path = REPO_ROOT / relative
        text = path.read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
        for marker in FORBIDDEN:
            assert marker not in code, f"{relative} must not use {marker}"
        assert 'read_records("test")' not in code
        assert "test_split" not in code


def _strip_docstrings(text: str) -> str:
    """Remove triple-quoted blocks so *documentation* about what is NOT used is not mistaken
    for code that uses it."""

    import re

    stripped = re.sub(r'""".*?"""', '""" """', text, flags=re.DOTALL)
    return re.sub(r"'''.*?'''", "''' '''", stripped, flags=re.DOTALL)


def test_task6e_location_token_machinery_is_retired_in_task6f():
    for relative in FORBIDDEN_6F_SOURCES:
        code = _strip_docstrings((REPO_ROOT / relative).read_text(encoding="utf-8"))
        code = "\n".join(line for line in code.splitlines() if not line.strip().startswith("#"))
        for marker in (
            "add_spatial_tokens",
            "location_token_ids",
            "spatial_loss",
            "spatial_train_step",
            "parse_generated_ids",
            "QuantizedBoxCodec",
            "location CE",
        ):
            assert marker not in code, f"{relative} must not reuse Task 6E coordinate machinery"


def test_no_loc_tokens_in_the_task6f_config():
    import yaml

    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "mvp" / "task6f_box_query.yaml").read_text(encoding="utf-8")
    )
    assert "spatial_tokens" not in cfg
    assert cfg["box_query"]["enabled"] is True
    assert cfg["task"] == "6F"
    assert cfg["training"]["deterministic"] is True
    assert cfg["training"]["deterministic_strict"] is True
    assert cfg["inference"]["do_sample"] is False
    assert float(cfg["box_query"]["box_loss_weight"]) == BOX_LOSS_WEIGHT
    assert float(cfg["box_query"]["reasoning_loss_weight"]) == REASONING_LOSS_WEIGHT


def test_scheduler_horizon_uses_the_actual_max_step_budget():
    scripts = REPO_ROOT / "scripts"
    f0_source = (scripts / "task6f_f0.py").read_text(encoding="utf-8")
    assert "total_steps=max_steps" in f0_source
    f1_source = (scripts / "task6f_f1.py").read_text(encoding="utf-8")
    assert "total_steps = max(1, epochs * steps_per_epoch)" in f1_source
    assert '"scheduler_horizon": total_steps' in f1_source

    from buildreasonseg_mvp.runtime import MvpRuntime

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.3)
    scheduler = MvpRuntime.build_scheduler_for(
        None, optimizer, max_steps=4, schedule_cfg={"lr_schedule": "cosine", "warmup_steps": 0}
    )
    lrs = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))
    assert lrs[0] > lrs[1] > lrs[2] > 0.0, f"no step before the end may train at LR 0: {lrs}"
    assert lrs[3] == pytest.approx(0.0, abs=1e-12)


def test_f1_selection_rule_is_paired_then_iou_then_center():
    source = (REPO_ROOT / "scripts" / "task6f_f1.py").read_text(encoding="utf-8")
    assert "geometry paired /20" in source
    assert "best = max(report[\"epochs\"], key=_selection_key)" in source
    assert "early_stop_after_epoch" in source and "early_stop_patience_epochs" in source
    key_source = (REPO_ROOT / "scripts" / "task6f_common.py").read_text(encoding="utf-8")
    assert '"geometry_paired_ge"' in key_source and '"val_box_iou_ge"' in key_source and '"center_inside_ge"' in key_source


def test_old_seg_grounding_head_is_unused_in_the_6f_path():
    from buildreasonseg_mvp.runtime import MvpRuntime

    import inspect

    freeze = inspect.getsource(MvpRuntime.freeze_for_box_query)
    assert "grounding_head" in freeze and "requires_grad_(False)" in freeze
    step = inspect.getsource(MvpRuntime.box_query_train_step)
    assert "forward_grounded" not in step
    assert "decode_mask" not in step
    assert "self.model.sam" not in step


def test_checkpointing_carries_the_box_head():
    source = (REPO_ROOT / "buildreasonseg_mvp" / "checkpointing.py").read_text(encoding="utf-8")
    assert '"box_head"' in source
    assert "box_head_loaded" in source


def test_json_artifacts_are_written_deterministically():
    from buildreasonseg_mvp.checkpointing import write_json

    path = EVAL / "task6f_verdict_probe.json"
    payload = {"b": 1, "a": [1, 2, 3], "nested": {"z": None}}
    write_json(path, payload)
    first = path.read_text(encoding="utf-8")
    write_json(path, payload)
    assert first == path.read_text(encoding="utf-8")
    path.unlink()


def main() -> int:
    import inspect

    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value) and not inspect.signature(value).parameters
    ]
    failures = 0
    for function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {function.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} task6f box-query checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
