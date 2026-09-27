"""Task 6I tests: Visual Query Refinement Block.

Block-level facts are asserted with real CPU forwards; model-level facts are asserted against
`sources` (the candidate forward, the freeze policy and the config) and against
`evaluation/task6i_architecture_setup.json` when it exists. The frozen Task 6H.1 objective is
imported unchanged, so its own test suite stays the source of truth for the scalar loss.

Run with pytest, or directly::

    python tests/test_task6i_visual_query_refinement.py
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp.dense_grounding import downsample_target_mask  # noqa: E402
from buildreasonseg_mvp.query_refine import (  # noqa: E402
    COARSE_CHANNELS,
    COARSE_GRID,
    FINE_CHANNELS,
    FINE_GRID,
    QUERY_DIM,
    VisualQueryRefinementBlock,
    attention_diagnostics,
)
from buildreasonseg_mvp.spatial_objective import (  # noqa: E402
    BOUNDED_CF_EPS,
    POINT_CF_WEIGHT,
    POINT_CLASSIFICATION_WEIGHT,
    POINT_REASONING_WEIGHT,
    bounded_pair_loss,
    counterfactual_masses,
    point_cross_entropy,
    region_mass,
    spatial_probabilities,
)

EVAL = REPO_ROOT / "evaluation"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6i_visual_query_refinement.yaml"
FORBIDDEN_SOURCES = (
    "buildreasonseg_mvp/query_refine.py",
    "scripts/task6i_i0.py",
    "scripts/task6i_i1.py",
    "scripts/task6i_i2.py",
    "scripts/task6i_architecture_setup.py",
    "scripts/task6i_i0_audit.py",
)


def _code(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _config() -> dict:
    import yaml

    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _block() -> VisualQueryRefinementBlock:
    return VisualQueryRefinementBlock()


def _runtime_step_body() -> str:
    """The Task 6I pair-step method body with the docstring stripped."""

    code = _code("buildreasonseg_mvp/runtime.py")
    body = code.split("def refinement_pair_train_step")[1].split("# -- Task 6H:")[0]
    first = body.find('"""')
    if first >= 0:
        second = body.find('"""', first + 3)
        if second >= 0:
            body = body[:first] + body[second + 3:]
    return body


def _f64(batch: int = 1, channels: int = COARSE_CHANNELS, grid: int = COARSE_GRID) -> torch.Tensor:
    return torch.randn(batch, channels, grid, grid)


def _f256(batch: int = 1, channels: int = FINE_CHANNELS, grid: int = FINE_GRID) -> torch.Tensor:
    return torch.randn(batch, channels, grid, grid)


# ------------------------------------------------------------------ 1-6 block structure


def test_block_has_exactly_one_cross_attention_layer():
    block = _block()
    assert block.cross_attention_layer_count == 1
    multihead = [m for m in block.modules() if isinstance(m, torch.nn.MultiheadAttention)]
    assert len(multihead) == 1
    # no self-attention stack anywhere else
    assert sum(1 for m in block.modules() if isinstance(m, torch.nn.TransformerEncoderLayer)) == 0


def test_block_attention_geometry_matches_spec():
    block = _block()
    assert block.embed_dim == 256
    assert block.num_heads == 4
    assert block.query_dim == 2048
    assert block.cross_attn.embed_dim == 256
    assert block.cross_attn.num_heads == 4
    assert block.ff_dim == 512


def test_block_validates_coarse_feature_shape():
    block = _block()
    with pytest.raises(RuntimeError):
        block.refine(torch.randn(1, QUERY_DIM), _f64(channels=128))
    with pytest.raises(RuntimeError):
        block.refine(torch.randn(1, QUERY_DIM), torch.randn(1, COARSE_CHANNELS, 32, 32))


def test_block_validates_fine_feature_shape():
    block = _block()
    with pytest.raises(RuntimeError):
        block.score(torch.randn(1, 256), _f256(channels=64))
    with pytest.raises(RuntimeError):
        block.score(torch.randn(1, 256), torch.randn(1, FINE_CHANNELS, 128, 128))


def test_block_validates_query_shape():
    block = _block()
    with pytest.raises(RuntimeError):
        block.refine(torch.randn(1, 512), _f64())


def test_forward_shapes_and_attention_normalization():
    torch.manual_seed(0)
    block = _block()
    q0 = torch.randn(2, QUERY_DIM)
    q1, heatmap, attn = block(q0, _f64(2), _f256(2))
    assert tuple(q1.shape) == (2, 256)
    assert tuple(heatmap.shape) == (2, FINE_GRID, FINE_GRID)
    assert tuple(attn.shape) == (2, COARSE_GRID * COARSE_GRID)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-4)
    assert torch.isfinite(heatmap).all() and torch.isfinite(attn).all()


def test_q0_to_q1_residual_is_exact():
    block = _block()
    q0 = torch.randn(1, QUERY_DIM)

    class _ZeroAttn(torch.nn.Module):
        def forward(self, query, key, value, **kwargs):
            return torch.zeros_like(query), torch.zeros(query.shape[0], query.shape[1], key.shape[1])

    block.cross_attn = _ZeroAttn()
    block.ffn[0].weight.data.zero_()
    block.ffn[0].bias.data.zero_()
    block.ffn[2].weight.data.zero_()
    block.ffn[2].bias.data.zero_()
    q_projected = block.query_proj(block.query_norm(q0.float())).unsqueeze(1)
    q1, _attn = block.refine(q0, _f64())
    assert torch.equal(q1, q_projected[:, 0, :]), "q_ref must equal q when attention and FFN are zero"


def test_ffn_residual_is_exact():
    block = _block()
    q0 = torch.randn(1, QUERY_DIM)

    class _IdentityAttn(torch.nn.Module):
        def forward(self, query, key, value, **kwargs):
            return torch.zeros_like(query), torch.zeros(query.shape[0], query.shape[1], key.shape[1])

    block.cross_attn = _IdentityAttn()
    block.ffn[0].weight.data.zero_()
    block.ffn[0].bias.data.zero_()
    block.ffn[2].weight.data.zero_()
    block.ffn[2].bias.data.zero_()
    q_projected = block.query_proj(block.query_norm(q0.float())).unsqueeze(1)
    q1, _attn = block.refine(q0, _f64())
    assert torch.equal(q1, q_projected[:, 0, :]), "q1 must equal q_ref when the FFN is zeroed"


def test_high_res_scorer_uses_q1():
    block = _block()
    f256 = _f256(2)
    q1_a = torch.randn(2, 256)
    q1_b = q1_a + 0.7
    assert tuple(block.score(q1_a, f256).shape) == (2, FINE_GRID, FINE_GRID)
    assert not torch.equal(block.score(q1_a, f256), block.score(q1_b, f256))
    # the combined forward's heatmap is exactly score(refined q1): the scorer never sees q0
    q0 = torch.randn(2, QUERY_DIM)
    q1, heatmap, _attn = block(q0, _f64(2), f256)
    assert torch.equal(heatmap, block.score(q1, f256))


# ------------------------------------------------------------------ 7 attention diagnostics


def test_attention_diagnostics_match_a_peaked_attention_map():
    weights = torch.zeros(COARSE_GRID * COARSE_GRID)
    weights[5] = 0.9
    weights[7] = 0.1
    soft = torch.zeros(COARSE_GRID, COARSE_GRID)
    soft.reshape(-1)[5] = 1.0
    diag = attention_diagnostics(weights, soft)
    assert diag["attn_top1_index"] == 5
    assert diag["attn_top1_cell"] == [5 % COARSE_GRID, 5 // COARSE_GRID]
    assert diag["attn_top10_mass"] == pytest.approx(1.0, abs=1e-6)
    assert diag["attn_target_mass"] == pytest.approx(0.9, abs=1e-6)
    assert diag["attn_entropy"] < 0.4


def test_attention_diagnostics_reject_wrong_shape():
    with pytest.raises(RuntimeError):
        attention_diagnostics(torch.zeros(100), torch.zeros(64, 64))


# ------------------------------------------------------------------ 8-11 frozen objective


def test_frozen_6h1_objective_is_imported_unchanged():
    # the Task 6I scalar objective re-imports the Task 6H.1 functions and constants
    assert POINT_REASONING_WEIGHT == 0.5
    assert POINT_CLASSIFICATION_WEIGHT == 1.0
    assert POINT_CF_WEIGHT == 1.0
    assert BOUNDED_CF_EPS == 1e-8
    signature = inspect.signature(bounded_pair_loss)
    assert "margin" not in signature.parameters
    # and the frozen machinery still behaves as Task 6H.1 defines it
    logits_a = torch.zeros(4, 4)
    logits_a[0, 0] = 20.0
    mask_a = torch.zeros(4, 4)
    mask_a[0, 0] = 1.0
    assert float(region_mass(spatial_probabilities(logits_a), mask_a)) > 0.0
    assert float(point_cross_entropy(logits_a, 0)) < 1e-3
    masses = counterfactual_masses(
        spatial_probabilities(logits_a), spatial_probabilities(torch.zeros(4, 4)),
        mask_a, torch.zeros(4, 4),
    )
    assert masses["raw"]["bounded"] is True


def test_legacy_terms_stay_detached_in_the_refinement_step():
    step = _runtime_step_body()
    assert "torch.no_grad()" in step
    assert "heatmap_loss" in step
    assert "region_scores" in step
    assert "legacy_terms_require_grad" in step
    assert "dense_head" not in step  # the old Task 6G head is absent from the candidate forward
    assert "box_head" not in step
    # weights come from the frozen constants, not new literals
    assert "POINT_REASONING_WEIGHT" in step
    assert "POINT_CLASSIFICATION_WEIGHT" in step
    assert "POINT_CF_WEIGHT" in step


def test_one_pair_is_exactly_one_optimizer_step():
    step = _runtime_step_body()
    assert step.count("optimizer.step()") == 1
    assert step.count("zero_grad") == 1
    assert step.count("total.backward()") == 1


def test_scheduler_horizon_counts_pair_steps():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from task6i_common import pair_step_budget

    cfg = _config()
    budget = pair_step_budget(cfg)
    assert budget["pairs_per_epoch"] == 240
    assert budget["total_pair_steps"] == 240 * 8
    assert budget["scheduler_horizon"] == 240 * 8
    assert budget["max_pair_steps"] == 1920


def test_pair_a_and_b_share_one_frozen_sam_feature_set():
    step = _runtime_step_body()
    assert step.count("feature_tensor_for_grid(") == 2  # F64 + F256, each fetched exactly once
    f64_line = [line for line in step.splitlines() if "COARSE_GRID).to" in line][0]
    f256_line = [line for line in step.splitlines() if "grid).to" in line][0]
    assert step.index(f64_line) < step.index("box_query_forward(self.model.qwen, batch_a)")
    assert step.index(f256_line) < step.index("box_query_forward(self.model.qwen, batch_a)")
    # both sides consume the same tensors
    assert "refine_block(hidden_a, f64, f256)" in step
    assert "refine_block(hidden_b, f64, f256)" in step


# ------------------------------------------------------------------ 15-16 frozen backbone


def test_sam2_and_visual_tower_stay_frozen():
    freeze = _code("buildreasonseg_mvp/runtime.py").split("def freeze_for_refinement")[1].split("def refinement_pair_train_step")[0]
    assert "set_phase_trainables(self.model, \"A\")" in freeze
    for marker in ("sam.", "visual", "vision_tower", "vision_model"):
        assert f"requires_grad_(True)" not in [
            line for line in freeze.splitlines() if marker in line
        ]
    assert "refine_block.parameters()" in freeze
    assert "requires_grad_(True)" in freeze  # only the block is explicitly unfrozen
    step = _runtime_step_body()
    assert "model.sam" not in step


def test_attention_diagnostics_never_inject_gt_into_inference():
    forward = _code("buildreasonseg_mvp/query_refine.py").split("def forward_refined")[1].split("def refined_record_from_raw")[0]
    assert "build_query_batch" in forward
    assert "target_mask = sample.target_mask()" in forward
    # the soft targets are computed strictly after the model call
    assert forward.index("target_mask = sample.target_mask()") > forward.index("refine_block(q0, f64, f256)")
    # the block signature takes no mask / ground-truth argument
    signature = inspect.signature(VisualQueryRefinementBlock.forward)
    assert {"q0", "f64", "f256"} == set(signature.parameters) - {"self"}
    signature = inspect.signature(VisualQueryRefinementBlock.refine)
    assert set(signature.parameters) - {"self"} == {"q0", "f64"}


def test_old_task6g_head_and_task6f_box_head_absent_from_candidate():
    for relative in ("buildreasonseg_mvp/query_refine.py",):
        code = _code(relative)
        for function in ("def forward_refined", "def refine"):
            body = code.split(function)[1].split("\ndef ")[0]
            assert "dense_head" not in body, f"{relative}:{function} must not use the old head"
            assert "box_head" not in body, f"{relative}:{function} must not use the box head"
    step = _runtime_step_body()
    assert "dense_head" not in step
    assert "box_head" not in step


def test_task6e_loc_path_absent():
    config = _config()
    assert "spatial_tokens" not in config
    assert "<loc_" not in CONFIG.read_text(encoding="utf-8")
    assert config["box_query"]["box_token"] == "[BOX]"
    assert config["visual_query_refinement"]["enabled"] is True


def test_no_test_split_is_used():
    for relative in FORBIDDEN_SOURCES:
        code = _code(relative)
        assert 'read_records("test")' not in code, relative
        assert "read_records('test')" not in code, relative


def test_i2_uses_the_predicted_point_only():
    code = _code("scripts/task6i_i2.py")
    assert "g2_report" in code
    assert "target_geometry" not in code
    assert "distance_transform_point" not in code
    assert "refined_point_record" in code


def test_no_forbidden_extensions():
    config = _config()
    # no config keys enable any of the forbidden extensions (comments may name them)
    for marker in ("sre", "scl", "ref"):
        assert marker not in config, f"forbidden config key {marker!r} appeared"
    config_text = CONFIG.read_text(encoding="utf-8")
    assert '"[REF]"' not in config_text  # no new [REF] special token is configured
    assert config["box_query"]["box_token"] == "[BOX]"
    assert config["models"]["qwen_model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert config["lora"]["text_only"] is True
    # the refinement module itself adds no new language special tokens
    refine = _code("buildreasonseg_mvp/query_refine.py")
    assert "add_special_tokens" not in refine
    # (the exact one-layer count is asserted structurally in
    #  test_block_has_exactly_one_cross_attention_layer)


def test_strict_determinism_is_configured():
    config = _config()
    assert config["training"]["deterministic"] is True
    assert config["training"]["deterministic_strict"] is True
    assert config["training"]["bf16_autocast"] is True


def test_i0_and_i1_gate_constants_match_the_spec():
    config = _config()
    i0 = config["i0"]["gate"]
    assert i0["point_inside_own_min"] == 18
    assert i0["paired_point_selection_min"] == 9
    assert i0["pair_ranking_min"] == 9
    assert i0["max_mean_normalized_error"] == 0.08
    assert config["i0"]["max_pair_steps"] == 1500
    i1 = config["i1"]["gate"]
    assert i1["paired_point_selection_min"] == 14
    assert i1["pair_ranking_min"] == 14
    assert i1["point_inside_target_min"] == 0.60
    assert config["i1"]["epochs"] == 8


# ------------------------------------------------------------------ 22-24 plumbing


def test_checkpoint_roundtrip_carries_the_refinement_block():
    import tempfile

    from buildreasonseg_mvp.checkpointing import load_checkpoint, save_checkpoint

    class _FakeQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))

    class _FakeDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 3))

    class _FakeModel(torch.nn.Module):
        def __init__(self, seed: int = 0):
            super().__init__()
            torch.manual_seed(seed)
            self.qwen = _FakeQwen()
            self.projection = _FakeDecoder()
            self.sam = torch.nn.Module()
            self.sam.sam_mask_decoder = _FakeDecoder()
            self.token_holder = None
            self.grounding_head = None
            self.box_head = None
            self.dense_head = None
            self.refine_block = VisualQueryRefinementBlock()
            self.geometry_kind = None

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        source = _FakeModel(seed=0)
        info = save_checkpoint(path, source, step=3, metrics={"a": 1})
        assert info["step"] == 3
        target = _FakeModel(seed=1)
        report = load_checkpoint(path, target)
        assert report["refine_block_loaded"] is True
        for (name_s, p_s), (name_t, p_t) in zip(
            source.refine_block.named_parameters(), target.refine_block.named_parameters()
        ):
            assert name_s == name_t
            assert torch.equal(p_s.detach().cpu(), p_t.detach().cpu())


def test_trainable_parameter_groups_include_the_refinement_block():
    code = _code("buildreasonseg_mvp/model.py")
    assert 'name.startswith("refine_block.")' in code
    groups = code.split("def trainable_parameter_groups")[1]
    decoder_block = groups.split("decoder.append(parameter)")[0]
    assert 'name.startswith("refine_block.")' in decoder_block
    # the block is in the decoder (freshly-initialised) bucket, not the adapter bucket


def test_block_determinism_on_cpu():
    block = _block()
    torch.manual_seed(7)
    q0 = torch.randn(1, QUERY_DIM)
    q1_a, heatmap_a, attn_a = block(q0, _f64(), _f256())
    torch.manual_seed(7)
    q0_b = torch.randn(1, QUERY_DIM)
    q1_b, heatmap_b, attn_b = block(q0_b, _f64(), _f256())
    assert torch.equal(q0, q0_b)
    assert torch.equal(q1_a, q1_b)
    assert torch.equal(heatmap_a, heatmap_b)
    assert torch.equal(attn_a, attn_b)


def test_architecture_setup_artifact_passes_when_present():
    path = EVAL / "task6i_architecture_setup.json"
    if not path.exists():
        pytest.skip("evaluation/task6i_architecture_setup.json not produced yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["task"] == "6I"
    assert report["passed"] is True
    assert report["architecture"]["coarse"]["coarse_exact"] is True
    assert report["architecture"]["coarse"]["fine_exact"] is True
    assert report["architecture"]["single_cross_attention_layer"] is True
    assert report["one_pair_step"]["sam2_bit_identical"] is True
    assert report["one_pair_step"]["frozen_dense_head_bit_identical"] is True
    assert report["one_pair_step"]["diagnostics_require_grad"] is False


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
    print(f"\n{len(tests) - failures}/{len(tests)} task6i visual-query-refinement checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
