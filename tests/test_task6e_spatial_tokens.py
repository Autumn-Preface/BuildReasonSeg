"""Task 6E tests: explicit spatial tokens (quantizer, vocabulary, targets, parser, gates).

Section 21 of the Task 6E spec lists 22 required checks. Everything that needs the 2B
model is asserted against the **recorded measurement** in
`evaluation/task6e_token_setup.json` (a real one-step forward/backward) plus a small
CPU-level unit test of the same mechanism, so this file never builds a second runtime and
never competes with the Task 6A fixtures for VRAM. Tokenizer-only tests load the tokenizer
(no weights) and skip cleanly when the snapshot is absent.

Run with pytest, or directly::

    python tests/test_task6e_spatial_tokens.py
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

from buildreasonseg_mvp.qwen_seg import (  # noqa: E402
    SegOutputRowLinear,
    TrainableTokenRow,
    install_trainable_output_rows,
    install_trainable_token_rows,
    is_token_parameter,
    token_holder_parameters,
)
from buildreasonseg_mvp.spatial_tokens import (  # noqa: E402
    BOX_TOKEN,
    QuantizedBoxCodec,
    SpatialTokenSetup,
    box_iou_values,
    build_target_ids,
    loc_name,
    parse_generated_ids,
    spatial_token_names,
)
from buildreasonseg_mvp.spatial_training import (  # noqa: E402
    ASSISTANT_LOSS_WEIGHT,
    LOCATION_LOSS_WEIGHT,
    SpatialExample,
    spatial_loss,
    target_label_positions,
    verify_label_positions,
)

EVAL = REPO_ROOT / "evaluation"
QWEN_HUB = REPO_ROOT / "local_cache" / "huggingface" / "hub"
SELECTED_BINS = 256


# --------------------------------------------------------------------- fixtures


class _StubBatch:
    def __init__(self, input_ids: torch.Tensor, prompt_length: int):
        self.input_ids = input_ids
        self.labels = torch.full_like(input_ids, -100)
        total = int(input_ids.shape[1])
        self.labels[0, prompt_length - 1 : total - 1] = input_ids[0, prompt_length:total]
        self.prompt_length = prompt_length
        self.total_length = total


class _StubTokenizer:
    """Deterministic 1:1 id mapping; enough for `build_target_ids`."""

    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [10 + (ord(character) % 50) for character in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _setup(bins: int = SELECTED_BINS) -> SpatialTokenSetup:
    return SpatialTokenSetup(
        bins=bins,
        box_token_id=900000,
        loc_token_ids=[900001 + index for index in range(bins)],
        vocab_before=900000,
        vocab_after=900000 + 1 + bins,
        added_tokens=1 + bins,
        embedding_shape_before=(900000, 8),
        embedding_shape_after=(900000 + 1 + bins, 8),
        tied_embeddings=True,
        single_token_roundtrip=True,
        label_names=spatial_token_names(bins),
    )


def _example(setup: SpatialTokenSetup, codes, prompt_length: int = 7, reasoning: str = "abcd"):
    tokenizer = _StubTokenizer()
    target_ids, spans = build_target_ids(tokenizer, reasoning, list(codes), setup, 99, append_eos=True)
    input_ids = torch.cat([torch.arange(10, 10 + prompt_length)[None, :], target_ids], dim=1)
    batch = _StubBatch(input_ids, prompt_length)
    example = SpatialExample(
        sample_id="stub",
        image_id="stub",
        level=1,
        query_family="size_extreme",
        reasoning=reasoning,
        codes=[int(value) for value in codes],
        box=list(setup_codes_box(codes)),
        gt_box=[0.1, 0.2, 0.5, 0.6],
        target_ids=[int(value) for value in target_ids[0].tolist()],
        spans={key: [int(v) for v in value] for key, value in spans.items()},
        label_positions=target_label_positions(prompt_length, spans),
        prompt_length=prompt_length,
        total_length=int(input_ids.shape[1]),
    )
    return batch, example


def setup_codes_box(codes):
    return QuantizedBoxCodec(SELECTED_BINS).decode(list(codes))


# ------------------------------------------------------------------ 1-3 quantizer


@pytest.mark.parametrize("bins", [32, 64, 128, 256])
def test_enclosing_quantizer_is_deterministic(bins):
    codec = QuantizedBoxCodec(bins)
    boxes = [
        (0.0, 0.0, 1.0, 1.0),
        (0.1, 0.2, 0.5, 0.6),
        (0.5001, 0.5001, 0.5002, 0.5002),
        (0.999, 0.001, 1.0, 0.002),
        (0.25, 0.25, 0.2500001, 0.75),
    ]
    for box in boxes:
        first = codec.encode(box)
        for _ in range(5):
            assert codec.encode(box) == first


@pytest.mark.parametrize("bins", [32, 64, 128, 256])
def test_quantizer_dequantizer_is_canonical(bins):
    codec = QuantizedBoxCodec(bins)
    for codes in ([0, 0, 1, 1], [bins - 1, bins - 1, bins - 1, bins - 1], [3, 6, 16, 19]):
        box = codec.decode(codes)
        if codec.is_valid(list(codes)):
            assert codec.encode(box) == list(codes), "canonical codes must round-trip"
        for value in box:
            assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("bins", [32, 64, 128, 256])
def test_non_empty_target_stays_non_empty(bins):
    codec = QuantizedBoxCodec(bins)
    tiny = 1.0 / (4 * bins)
    boxes = [
        (0.5, 0.5, 0.5 + tiny, 0.5 + tiny),
        (0.0, 0.0, tiny, tiny),
        (1.0 - tiny, 1.0 - tiny, 1.0, 1.0),
        (0.999999, 0.999999, 1.0, 1.0),
        (0.3, 0.3, 0.3000001, 0.3000001),
    ]
    for box in boxes:
        codes = codec.encode(box)
        assert codec.is_valid(codes), f"{box} quantized to a degenerate {codes}"
        x1, y1, x2, y2 = codec.decode(codes)
        assert x2 > x1 and y2 > y1
        assert 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0


def test_enclosing_rule_prefers_the_outer_box():
    codec = QuantizedBoxCodec(256)
    box = (0.1, 0.2, 0.5, 0.6)
    codes = codec.encode(box)
    scale = float(255)
    assert codes[0] == int(torch.floor(torch.tensor(box[0] * scale)).item())
    assert codes[1] == int(torch.floor(torch.tensor(box[1] * scale)).item())
    assert codes[2] == int(torch.ceil(torch.tensor(box[2] * scale)).item())
    assert codes[3] == int(torch.ceil(torch.tensor(box[3] * scale)).item())
    assert box_iou_values(codec.decode(codes), box) > 0.9


# ------------------------------------------------------------------ 4 oracle rule


def test_selected_bin_count_obeys_the_oracle_rule():
    oracle = json.loads((EVAL / "task6e_quantized_oracle.json").read_text(encoding="utf-8"))
    selection = oracle["selection"]
    threshold = selection["miou_threshold"]
    assert abs(threshold - (0.7506 - 0.03)) < 1e-9, "section 5: threshold is oracle minus 0.03"
    qualifying = []
    for bins, entry in sorted(((int(k), v) for k, v in oracle["candidates"].items())):
        assert entry["paired_pass"] <= entry["paired_total"]
        if entry["paired_pass"] == entry["paired_total"] and entry["strict_mask_miou"] >= threshold:
            qualifying.append(bins)
    assert selection["qualifying_bins"] == qualifying
    if qualifying:
        assert selection["selected_bins"] == min(qualifying), "smallest qualifying B wins"
        assert selection["selected_bins"] == SELECTED_BINS
        assert selection["verdict"] == f"PROCEED_WITH_B{SELECTED_BINS}"
    else:
        assert selection["selected_bins"] is None
        assert selection["verdict"] == "QUANTIZED_BOX_REPRESENTATION_INADEQUATE"


def test_config_bin_count_matches_the_oracle_selection():
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "mvp" / "task6e_spatial_tokens.yaml").read_text(encoding="utf-8"))
    assert int(cfg["spatial_tokens"]["bins"]) == SELECTED_BINS
    assert float(cfg["spatial_tokens"]["assistant_loss_weight"]) == ASSISTANT_LOSS_WEIGHT
    assert float(cfg["spatial_tokens"]["location_loss_weight"]) == LOCATION_LOSS_WEIGHT
    assert cfg["task"] == "6E"
    assert cfg["training"]["deterministic"] is True
    assert cfg["training"]["deterministic_strict"] is True
    assert cfg["inference"]["do_sample"] is False


# ------------------------------------------------------------- 5-7 real tokenizer


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


def test_spatial_tokens_are_single_unique_tokens(qwen_tokenizer):
    tokenizer = qwen_tokenizer
    names = spatial_token_names(SELECTED_BINS)
    added = tokenizer.add_special_tokens({"additional_special_tokens": names})
    assert added == len(names)
    ids = [int(tokenizer.convert_tokens_to_ids(name)) for name in names]
    assert len(set(ids)) == len(names), "token ids must be unique"
    for name, token_id in zip(names, ids):
        encoded = tokenizer.encode(name, add_special_tokens=False)
        assert encoded == [token_id], f"{name} is not exactly one token"
        assert name in tokenizer.decode([token_id], skip_special_tokens=False)


def test_location_token_names_are_shared_and_indexed():
    names = spatial_token_names(256)
    assert names[0] == BOX_TOKEN
    assert names[1] == loc_name(0, 256) == "<loc_000>"
    assert names[-1] == loc_name(255, 256) == "<loc_255>"
    assert len(names) == 257
    assert len(set(names)) == 257


# ------------------------------------------- 8-11 trainable rows (mechanism, CPU)


class _TinyModel(nn.Module):
    """Tiny tied embedding/head pair that mimics the Qwen + PEFT token adapter layout."""

    def __init__(self, vocab: int = 24, dim: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embedding.weight  # tied, like Qwen3-VL

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


def test_multiple_trainable_rows_leave_the_base_table_frozen():
    torch.manual_seed(0)
    model = _TinyModel()
    ids = [3, 4, 5]
    base_before = model.embedding.weight.detach().clone()
    holder = install_trainable_token_rows(model, ids)
    assert isinstance(holder, TrainableTokenRow)
    assert holder.token_ids == ids
    assert len(token_holder_parameters(holder)) == 1
    # every requested row is trainable; the base table is not
    assert holder.rows.requires_grad
    assert not model.embedding.weight.requires_grad


def test_new_token_rows_receive_gradient_and_can_be_emitted():
    torch.manual_seed(0)
    model = _TinyModel()
    ids = [3, 4]
    holder = install_trainable_token_rows(model, ids)
    wrapper = install_trainable_output_rows(model, ids)
    assert isinstance(wrapper, SegOutputRowLinear)
    assert wrapper.token_ids == ids

    # (a) the input side replaces exactly the new rows
    out = model.embedding(torch.tensor([[3, 4, 7]]))
    assert torch.allclose(out[0, 0].detach(), holder.rows[0].detach(), atol=1e-6)
    assert torch.allclose(out[0, 2].detach(), model.embedding.weight[7].detach(), atol=1e-6)

    # (b) the output side is trainable for every new token, ordinary columns untouched
    hidden = torch.randn(2, 6)
    logits = wrapper(hidden)
    assert logits.shape == (2, model.embedding.weight.shape[0])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3, 4]))
    loss.backward()
    assert wrapper.delta.grad is not None
    for row in range(len(ids)):
        assert float(wrapper.delta.grad[row].abs().sum()) > 0.0, "every new output row needs gradient"
    base_parameter = getattr(model.head, "base").weight
    assert base_parameter.grad is None, "the base output matrix must stay frozen"

    # (c) one real step through embedding -> head: new rows move, an ordinary row does not
    wrapper.delta.grad = None
    holder.rows.grad = None
    logits = wrapper(model.embedding(torch.tensor([[3, 4, 9]])))
    ordinary_before = model.embedding.weight[9].detach().clone()
    new_rows_before = holder.rows.detach().clone()
    torch.nn.functional.cross_entropy(logits.reshape(-1, 24), torch.tensor([3, 4, 9])).backward()
    assert holder.rows.grad is not None and float(holder.rows.grad.abs().sum()) > 0.0
    torch.optim.AdamW(
        [
            {"params": [holder.rows], "lr": 0.1, "weight_decay": 0.0},
            {"params": [wrapper.delta], "lr": 0.1, "weight_decay": 0.0},
        ]
    ).step()
    assert torch.equal(ordinary_before, model.embedding.weight[9].detach())
    assert float((holder.rows.detach() - new_rows_before).abs().max()) > 0.0
    assert torch.isfinite(holder.rows).all() and torch.isfinite(wrapper.delta).all()


def test_token_parameter_marker_covers_the_new_rows():
    assert is_token_parameter("base_model.model.model.embed_tokens.trainable_tokens_delta.default")
    assert is_token_parameter("qwen.token_holder.rows")
    assert not is_token_parameter("base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight")


# ------------------------------------------- 12-13 target structure / off-by-one


def test_target_format_has_one_box_four_loc_and_one_seg_in_order():
    setup = _setup()
    target_ids, spans = build_target_ids(_StubTokenizer(), "abc", [1, 2, 3, 4], setup, 99)
    flat = [int(value) for value in target_ids[0].tolist()]
    assert spans["box"] == [len(spans["reasoning"])]
    assert spans["loc"] == [spans["box"][0] + 1 + offset for offset in range(4)]
    assert spans["seg"] == [spans["loc"][-1] + 1]
    assert spans["eos"] == [spans["seg"][0] + 1]
    assert flat.count(setup.box_token_id) == 1
    assert flat.count(99) == 1
    assert sum(1 for value in flat if value in set(setup.loc_token_ids)) == 4
    assert flat.index(setup.box_token_id) < flat.index(setup.loc_token_ids[1]) < flat.index(99)
    parsed = parse_generated_ids(flat[len(spans["reasoning"]) :], setup, QuantizedBoxCodec(setup.bins), 99)
    assert parsed.structural_valid and parsed.exact_four_token_sequence


def test_causal_location_label_positions_are_off_by_one_correct():
    setup = _setup()
    batch, example = _example(setup, [3, 6, 16, 19], prompt_length=7)
    # section 8: logits at position i predict token i + 1
    for name, positions in example.label_positions.items():
        for position, offset in zip(positions, example.spans[name]):
            assert int(batch.labels[0, position]) == int(batch.input_ids[0, position + 1])
            assert int(batch.labels[0, position]) == int(example.target_ids[offset])
    loc_positions = example.label_positions["loc"]
    assert len(loc_positions) == 4
    assert loc_positions == [example.prompt_length + offset - 1 for offset in example.spans["loc"]]
    # the first reasoning token is predicted by the last prompt token
    assert example.label_positions["reasoning"][0] == example.prompt_length - 1
    verify_label_positions(batch, example, setup)


def test_verify_label_positions_rejects_a_shifted_layout():
    setup = _setup()
    batch, example = _example(setup, [3, 6, 16, 19])
    example.label_positions = {
        key: [position + 1 for position in value] for key, value in example.label_positions.items()
    }
    with pytest.raises(RuntimeError):
        verify_label_positions(batch, example, setup)


# ------------------------------------------------------------- 9 loss weighting


def test_loss_is_assistant_plus_five_times_location():
    setup = _setup()
    batch, example = _example(setup, [3, 6, 16, 19])
    vocab = setup.vocab_after
    torch.manual_seed(1)
    logits = torch.randn(1, example.total_length, vocab, requires_grad=True)
    losses = spatial_loss(logits, batch, example, setup)
    expected_assistant = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab).float(), batch.labels.reshape(-1), ignore_index=-100
    )
    loc_positions = example.label_positions["loc"]
    expected_location = torch.nn.functional.cross_entropy(
        logits[0, loc_positions, :].float(), batch.labels[0, loc_positions]
    )
    assert torch.allclose(losses["assistant_ce"], expected_assistant)
    assert torch.allclose(losses["location_ce"], expected_location)
    assert torch.allclose(
        losses["total"], ASSISTANT_LOSS_WEIGHT * expected_assistant + LOCATION_LOSS_WEIGHT * expected_location
    )
    assert losses["assistant_weight"] == 1.0 and losses["location_weight"] == 5.0


def test_loss_rejects_a_location_layout_that_does_not_hold_loc_ids():
    setup = _setup()
    batch, example = _example(setup, [3, 6, 16, 19])
    example.label_positions["loc"] = [example.label_positions["loc"][0] + 1] + example.label_positions["loc"][1:]
    with pytest.raises(RuntimeError):
        spatial_loss(torch.randn(1, example.total_length, setup.vocab_after), batch, example, setup)


# ------------------------------------------------------------- 14 parser


def test_parser_operates_on_token_ids_and_rejects_malformed_output():
    setup = _setup()
    codec = QuantizedBoxCodec(setup.bins)
    seg_id = 99
    good = [setup.loc_token_ids[3], setup.box_token_id, *[setup.loc_token_ids[c] for c in (1, 2, 3, 4)], seg_id]
    parsed = parse_generated_ids(good, setup, codec, seg_id)
    assert parsed.structural_valid and parsed.codes == [1, 2, 3, 4]

    # a decoded *string* of the same tokens must not be accepted as token ids
    as_text_ids = [ord(character) % 100 for character in "[BOX] <loc_001> <loc_002> <loc_003> <loc_004> [SEG]"]
    assert not parse_generated_ids(as_text_ids, setup, codec, seg_id).structural_valid

    cases = {
        "no box": [setup.loc_token_ids[1], setup.loc_token_ids[2], seg_id],
        "two boxes": [setup.box_token_id, setup.box_token_id, *[setup.loc_token_ids[c] for c in (1, 2, 3, 4)], seg_id],
        "three loc": [setup.box_token_id, *[setup.loc_token_ids[c] for c in (1, 2, 3)], seg_id],
        "five loc": [setup.box_token_id, *[setup.loc_token_ids[c] for c in (1, 2, 3, 4, 5)], seg_id],
        "gap after box": [
            setup.box_token_id,
            77,
            *[setup.loc_token_ids[c] for c in (1, 2, 3, 4)],
            seg_id,
        ],
        "no seg": [setup.box_token_id, *[setup.loc_token_ids[c] for c in (1, 2, 3, 4)]],
        "two seg": [
            setup.box_token_id,
            *[setup.loc_token_ids[c] for c in (1, 2, 3, 4)],
            seg_id,
            seg_id,
        ],
        "seg before box": [seg_id, setup.box_token_id, *[setup.loc_token_ids[c] for c in (1, 2, 3, 4)]],
        "degenerate box": [setup.box_token_id, *[setup.loc_token_ids[c] for c in (4, 4, 4, 4)], seg_id],
    }
    for name, ids in cases.items():
        parsed = parse_generated_ids(ids, setup, codec, seg_id)
        assert not parsed.structural_valid, f"{name} must be rejected"
        assert parsed.failure, f"{name} must record why"


# ------------------------------------------- 15 no ground truth in the prompt path


def test_inference_path_accepts_no_ground_truth():
    import inspect

    from buildreasonseg_mvp import spatial_inference

    forbidden = {"mask", "target_mask", "gt", "gt_box", "geometry", "gt_geometry", "box"}
    for name in (
        "generate_spatial_tokens",
        "parse_box_tokens",
        "predict_box",
        "predict_mask_from_generated_box",
    ):
        signature = inspect.signature(getattr(spatial_inference, name))
        parameters = set(signature.parameters)
        assert not (parameters & forbidden), f"{name} must not take ground truth: {parameters}"
    source = inspect.getsource(spatial_inference.generate_spatial_tokens)
    assert "target_mask" not in source and "target_geometry" not in source
    # the box prompt comes from the generated tokens only
    mask_source = inspect.getsource(spatial_inference.predict_mask_from_generated_box)
    assert "prediction[\"box\"]" in mask_source and "target_geometry" not in mask_source


def test_training_supervision_is_the_only_target_geometry_consumer():
    import inspect

    from buildreasonseg_mvp import spatial_training

    source = inspect.getsource(spatial_training)
    assert "target_geometry" in source, "the training target must come from the GT mask"
    generation = inspect.getsource(spatial_training.spatial_batch)
    assert "instruction_zh" in generation and "target_ids=torch.tensor" in generation


# ------------------------------------------- 16 scheduler horizon


def test_scheduler_horizon_uses_the_true_total_step_count():
    scripts = REPO_ROOT / "scripts"
    e1_source = (scripts / "task6e_e1.py").read_text(encoding="utf-8")
    assert "total_steps = max(1, epochs * steps_per_epoch)" in e1_source
    assert "total_steps=total_steps" in e1_source
    assert '"scheduler_horizon": total_steps' in e1_source
    e0_source = (scripts / "task6e_e0.py").read_text(encoding="utf-8")
    assert "total_steps = max(1, max_steps)" in e0_source
    assert '"scheduler_horizon": total_steps' in e0_source
    common_source = (scripts / "task6e_common.py").read_text(encoding="utf-8")
    assert "max_steps=max(1, int(total_steps))" in common_source

    # numeric behaviour of the real scheduler: cosine reaches 0 only at the final step
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
    # the terminal factor is reached only on the final step of the real budget
    assert lrs[0] > lrs[1] > lrs[2] > 0.0, f"no step before the end may train at LR 0: {lrs}"
    assert lrs[3] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------- 17-18 frozen SAM / unused head


def test_spatial_training_freezes_sam_projection_and_the_old_head():
    import inspect

    from buildreasonseg_mvp.runtime import MvpRuntime

    freeze = inspect.getsource(MvpRuntime.freeze_for_spatial_tokens)
    assert 'set_phase_trainables(self.model, "A")' in freeze
    assert "grounding_head" in freeze and "requires_grad_(False)" in freeze
    step = inspect.getsource(MvpRuntime.spatial_train_step)
    assert "forward_qwen" in step
    assert "self.model(" not in step and "forward_grounded" not in step
    assert "sam" not in step.replace("spatial_", ""), "E steps must not touch SAM2"


def test_records_show_sam_frozen_and_head_unused():
    setup = json.loads((EVAL / "task6e_token_setup.json").read_text(encoding="utf-8"))
    trainables = setup["trainable_tokens"]
    assert trainables["sam_frozen"] is True
    assert trainables["projection_frozen"] is True
    assert trainables["base_embedding_frozen"] is True


# ------------------------------------------- 8-11 recorded one-step smoke


def test_one_step_smoke_records_every_required_row_behaviour():
    setup = json.loads((EVAL / "task6e_token_setup.json").read_text(encoding="utf-8"))
    assert setup["passed"] is True
    vocabulary = setup["vocabulary"]
    assert vocabulary["all_names_single_token"] and vocabulary["all_names_decode_roundtrip"]
    assert vocabulary["unique_token_ids"] == 257
    assert vocabulary["added_tokens"] == 257
    assert vocabulary["embedding_rows_match_tokenizer"] is True
    smoke = setup["one_step_smoke"]
    assert smoke["seg_row_changed"] is True
    assert smoke["box_row_changed"] is True
    assert smoke["all_target_loc_rows_changed"] is True
    assert smoke["loc_rows_changed"] == 4
    assert smoke["ordinary_rows_probed"] >= 16
    assert smoke["ordinary_rows_unchanged_exactly"] is True
    assert smoke["base_embedding_tensor_bit_identical"] is True
    assert smoke["every_new_output_row_receives_gradient"] is True
    assert smoke["visual_tower_lora_modules"] == []
    assert setup["trainable_tokens"]["n_trainable_token_ids"] == 258
    assert setup["trainable_tokens"]["mechanism"] == "peft_trainable_token_indices"


# ------------------------------------------- 19-21 hygiene / determinism


FORBIDDEN = ("[REF]", "SpatialRelationEncoder", "spatial_consistency", "Qwen3-VL-4B", "4B-Instruct")


def test_task6e_sources_contain_no_forbidden_components():
    paths = [
        REPO_ROOT / "buildreasonseg_mvp" / "spatial_tokens.py",
        REPO_ROOT / "buildreasonseg_mvp" / "spatial_training.py",
        REPO_ROOT / "buildreasonseg_mvp" / "spatial_eval.py",
        REPO_ROOT / "buildreasonseg_mvp" / "spatial_inference.py",
        REPO_ROOT / "scripts" / "task6e_common.py",
        REPO_ROOT / "scripts" / "task6e_e0.py",
        REPO_ROOT / "scripts" / "task6e_e1.py",
        REPO_ROOT / "scripts" / "task6e_e2.py",
        REPO_ROOT / "scripts" / "task6e_token_setup.py",
        REPO_ROOT / "scripts" / "task6e_quantized_oracle.py",
        REPO_ROOT / "scripts" / "task6e_error_analysis.py",
        REPO_ROOT / "scripts" / "task6e_verdict.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
        for marker in FORBIDDEN:
            assert marker not in code, f"{path.name} must not use {marker}"
        assert 'read_records("test")' not in code
        assert "test_split" not in code


def test_no_test_split_in_the_task6e_artifacts():
    for name in ("task6e_quantized_oracle.json", "task6e_token_setup.json"):
        payload = json.loads((EVAL / name).read_text(encoding="utf-8"))
        text = json.dumps(payload)
        assert "buildsr_test" not in text, f"{name} must not contain test-split samples"


def test_json_artifacts_are_written_deterministically():
    from buildreasonseg_mvp.checkpointing import write_json

    path = EVAL / "task6e_verdict_probe.json"
    payload = {"b": 1, "a": [1, 2, 3], "nested": {"z": None}}
    write_json(path, payload)
    first = path.read_text(encoding="utf-8")
    write_json(path, payload)
    assert first == path.read_text(encoding="utf-8")
    path.unlink()


# ------------------------------------------- 22 Task 6D.1 wording / boolean


def test_decodability_boolean_now_states_what_the_report_concludes():
    summary = json.loads((EVAL / "task6d1_decodability_summary.json").read_text(encoding="utf-8"))
    inputs = summary["inputs"]
    assert "twenty_sample_overfit_is_diagnostic" not in inputs
    assert inputs["twenty_sample_overfit_is_evidence_of_decodability"] is False
    assert inputs["shuffled_control_also_overfits_twenty"] is True
    assert inputs["twenty_sample_overfit_final_loss"] is not None
    assert "NOT evidence" in inputs["twenty_sample_overfit_semantics"]
    # measured probe numbers are untouched
    assert inputs["probe_raw_mlp_train_480_box_iou"] == pytest.approx(0.035506073385477066)
    assert inputs["probe_raw_mlp_val_box_iou"] == pytest.approx(0.007475934457033873)


def test_representation_stats_generator_emits_the_corrected_fields():
    source = (REPO_ROOT / "scripts" / "task6d1_representation_stats.py").read_text(encoding="utf-8")
    assert "twenty_sample_overfit_is_diagnostic" not in source
    assert "twenty_sample_overfit_is_evidence_of_decodability" in source
    assert "twenty_sample_overfit_semantics" in source
    assert "overfit_blocks_implementation_bug_claim" in source


def test_project_state_no_longer_asserts_the_stale_claim_as_current():
    state = (REPO_ROOT / "handoff" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    stale = "`[SEG]` hidden state does not carry the target's location"
    assert stale not in state
    assert "practically decodable" in state
    assert "not an information-theoretic absence claim" in state


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
    print(f"\n{len(tests) - failures}/{len(tests)} task6e spatial-token checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
