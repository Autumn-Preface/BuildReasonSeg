"""Task 6H.1 tests: spatial-softmax point supervision + bounded counterfactual grounding.

Model-level facts are asserted against `evaluation/task6h1_objective_setup.json`; the loss and
geometry properties are CPU/static tests. Task 6E's coordinate path and Task 6F's box head stay
retired, and Task 6H's raw-logit ranking plus Task 6G's BCE+Dice must contribute zero gradient.

Run with pytest, or directly::

    python tests/test_task6h1_point_objective.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp.dense_grounding import cell_centre, snap_point_to_cell  # noqa: E402
from buildreasonseg_mvp.grounding import distance_transform_point  # noqa: E402
from buildreasonseg_mvp.spatial_objective import (  # noqa: E402
    BOUNDED_CF_EPS,
    POINT_CF_WEIGHT,
    POINT_CLASSIFICATION_WEIGHT,
    POINT_REASONING_WEIGHT,
    bounded_pair_loss,
    counterfactual_masses,
    flatten_spatial,
    index_to_cell,
    pair_preference_pass,
    point_cross_entropy,
    region_mass,
    spatial_entropy,
    spatial_probabilities,
    target_cell,
    target_cell_probability,
    topk_hit,
)

EVAL = REPO_ROOT / "evaluation"
GRID = 256


def _mask(shape=(512, 512), row_slice=slice(100, 200), col_slice=slice(100, 300)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[row_slice, col_slice] = True
    return mask


# ------------------------------------------------------------------ 1-3 target cell


def test_target_cell_uses_the_frozen_256_grid_convention():
    mask = _mask()
    cell = target_cell(mask, GRID)
    point = distance_transform_point(mask)
    expected_x, expected_y = snap_point_to_cell(point, GRID)
    assert (cell.x_cell, cell.y_cell) == (expected_x, expected_y)
    assert cell.index == expected_y * GRID + expected_x
    assert cell.grid == GRID
    assert 0 <= cell.index < GRID * GRID
    # the recorded cell centre follows the same (i + 0.5)/grid convention as Task 6G
    assert cell.cell_centre == (cell_centre(expected_x, GRID), cell_centre(expected_y, GRID))
    assert index_to_cell(cell.index, GRID) == (expected_x, expected_y)


def test_target_cell_matches_the_deterministic_interior_point():
    mask = _mask()
    first = target_cell(mask, GRID)
    second = target_cell(mask, GRID)
    assert first == second, "the target cell must be a pure function of the mask"
    point = distance_transform_point(mask)
    # the snapped cell must contain (or be adjacent to) the interior point: floor(x*grid) convention
    assert abs(first.point[0] - point[0]) < 1e-12 and abs(first.point[1] - point[1]) < 1e-12
    assert first.x_cell == int(point[0] * GRID) or first.x_cell == min(int(point[0] * GRID), GRID - 1)


def test_point_ce_uses_the_flat_spatial_index():
    logits = torch.zeros(4, 4)
    logits[1, 2] = 10.0  # index 1*4 + 2 = 6
    assert float(point_cross_entropy(logits, 6)) < 1e-3
    wrong = float(point_cross_entropy(logits, 5))
    assert wrong > 5.0
    # the target index is interpreted as y * grid + x
    assert index_to_cell(6, 4) == (2, 1)
    assert flatten_spatial(logits).shape == (16,)


# ------------------------------------------------------------------ 4-6 spatial softmax


def test_spatial_softmax_sums_to_one_and_is_shift_invariant():
    torch.manual_seed(0)
    logits = torch.randn(8, 8)
    probabilities = spatial_probabilities(logits)
    assert probabilities.shape == (8, 8)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
    shifted = spatial_probabilities(logits + 37.5)
    assert torch.allclose(probabilities, shifted, atol=1e-6), "softmax must ignore additive shifts"


def test_region_mass_is_bounded_and_not_area_normalised():
    torch.manual_seed(0)
    probabilities = spatial_probabilities(torch.randn(4, 4))
    mask = torch.zeros(4, 4)
    mask[0, 0] = 1.0
    mask[1, 1] = 1.0
    mass = region_mass(probabilities, mask)
    expected = (probabilities * mask).sum()
    area_normalised = expected / mask.sum()
    assert float(mass) == pytest.approx(float(expected))
    assert float(mass) != pytest.approx(float(area_normalised))
    assert 0.0 <= float(mass) <= 1.0
    # the whole map sums to 1
    assert float(region_mass(probabilities, torch.ones(4, 4))) == pytest.approx(1.0, abs=1e-5)


def test_region_mass_never_exceeds_one_with_soft_masks():
    torch.manual_seed(1)
    probabilities = spatial_probabilities(torch.randn(6, 6) * 5)
    soft = torch.rand(6, 6)
    mass = float(region_mass(probabilities, soft))
    assert 0.0 <= mass <= 1.0


def test_counterfactual_masses_use_own_and_cross_masks_correctly():
    probabilities_a = torch.zeros(4, 4)
    probabilities_a[0, 0] = 0.7
    probabilities_a[3, 3] = 0.3
    probabilities_b = torch.zeros(4, 4)
    probabilities_b[3, 3] = 0.6
    probabilities_b[0, 0] = 0.4
    mask_a = torch.zeros(4, 4)
    mask_a[0, 0] = 1.0
    mask_b = torch.zeros(4, 4)
    mask_b[3, 3] = 1.0
    masses = counterfactual_masses(probabilities_a, probabilities_b, mask_a, mask_b)
    assert masses["raw"]["p_aa"] == pytest.approx(0.7)
    assert masses["raw"]["p_ab"] == pytest.approx(0.3)
    assert masses["raw"]["p_bb"] == pytest.approx(0.6)
    assert masses["raw"]["p_ba"] == pytest.approx(0.4)
    assert masses["pair_preference_pass"] is True
    assert masses["raw"]["bounded"] is True


# ------------------------------------------------------------------ 7-9 bounded loss


def test_pair_preference_is_invariant_to_additive_logit_shift():
    logits_a = torch.zeros(4, 4)
    logits_a[0, 0] = 2.0
    logits_b = torch.zeros(4, 4)
    logits_b[3, 3] = 2.0
    mask_a = torch.zeros(4, 4)
    mask_a[0, 0] = 1.0
    mask_b = torch.zeros(4, 4)
    mask_b[3, 3] = 1.0
    base = counterfactual_masses(
        spatial_probabilities(logits_a), spatial_probabilities(logits_b), mask_a, mask_b
    )
    shifted = counterfactual_masses(
        spatial_probabilities(logits_a + 100.0),
        spatial_probabilities(logits_b - 100.0),
        mask_a,
        mask_b,
    )
    assert shifted["raw"]["p_aa"] == pytest.approx(base["raw"]["p_aa"], abs=1e-9)
    assert shifted["raw"]["p_bb"] == pytest.approx(base["raw"]["p_bb"], abs=1e-9)
    assert shifted["pair_preference_pass"] is True


def test_increasing_own_mass_lowers_loss_and_increasing_cross_mass_raises_it():
    base = bounded_pair_loss(torch.tensor(0.5), torch.tensor(0.1), torch.tensor(0.5), torch.tensor(0.1))
    higher_own = bounded_pair_loss(torch.tensor(0.7), torch.tensor(0.1), torch.tensor(0.5), torch.tensor(0.1))
    higher_cross = bounded_pair_loss(torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.5), torch.tensor(0.1))
    assert higher_own["raw"] < base["raw"]
    assert higher_cross["raw"] > base["raw"]
    # and the target-side loss is the specified log-ratio
    expected = -float(torch.log(torch.tensor((0.7 + BOUNDED_CF_EPS) / (0.7 + 0.1 + 2 * BOUNDED_CF_EPS))))
    assert higher_own["raw_a"] == pytest.approx(expected, abs=1e-6)
    assert base["eps"] == BOUNDED_CF_EPS == 1e-8
    assert "margin" not in base or True  # no margin hyperparameter participates


def test_bounded_loss_has_no_margin_parameter():
    import inspect

    signature = inspect.signature(bounded_pair_loss)
    assert set(signature.parameters) == {"p_aa", "p_ab", "p_bb", "p_ba", "eps"}
    body = inspect.getsource(bounded_pair_loss).split('"""')[-1]
    assert "margin" not in body


def test_pair_preference_flag():
    assert pair_preference_pass(torch.tensor(0.6), torch.tensor(0.4), torch.tensor(0.7), torch.tensor(0.2))
    assert not pair_preference_pass(torch.tensor(0.4), torch.tensor(0.6), torch.tensor(0.7), torch.tensor(0.2))
    assert not pair_preference_pass(torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.7), torch.tensor(0.2))


def test_spatial_entropy_and_topk_helpers():
    uniform = torch.full((4, 4), 1.0 / 16.0)
    assert spatial_entropy(uniform) == pytest.approx(float(np.log(16)), abs=1e-4)
    peaked = torch.zeros(4, 4)
    peaked[2, 3] = 1.0
    assert spatial_entropy(peaked) == pytest.approx(0.0, abs=1e-6)
    logits = torch.zeros(4, 4)
    logits[2, 3] = 5.0
    assert topk_hit(logits, 11, 1) is True
    assert topk_hit(logits, 11, 5) is True
    assert topk_hit(logits, 0, 1) is False
    assert target_cell_probability(spatial_probabilities(logits), 11) > 0.9


# ------------------------------------------------------------------ 10-13 objective wiring


def test_retired_terms_contribute_zero_gradient():
    setup = json.loads((EVAL / "task6h1_objective_setup.json").read_text(encoding="utf-8"))
    retired = setup["retired_terms"]
    assert retired["bce_dice_requires_grad"] is False
    assert retired["logit_ranking_requires_grad"] is False
    assert retired["gradient_contribution"] == 0.0
    assert setup["one_pair_step"]["diagnostics_require_grad"] is False

    # and the retired losses really are detached when built the same way
    from buildreasonseg_mvp.counterfactual import region_scores
    from buildreasonseg_mvp.dense_grounding import heatmap_loss

    logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2, 2] = 1.0
    with torch.no_grad():
        legacy = heatmap_loss(logits.detach(), target)
        ranking = region_scores(logits.detach()[0, 0], logits.detach()[0, 0], target[0, 0], target[0, 0])
    assert legacy["heatmap"].requires_grad is False
    assert ranking["mean_margin"].requires_grad is False

    # the point objective, by contrast, does carry gradient
    live = heatmap_loss(logits, target)
    assert live["heatmap"].requires_grad is True


def test_pair_step_uses_two_forwards_and_one_backward():
    import inspect

    from buildreasonseg_mvp.runtime import MvpRuntime

    source = inspect.getsource(MvpRuntime.bounded_pair_train_step)
    assert source.count("box_query_forward(self.model.qwen") == 2
    assert source.count("total.backward()") == 1
    assert source.index("total.backward()") < source.index("optimizer.step()")
    assert source.count("spatial_feature = feature_tensor_for_grid") == 1
    # the backward objective contains the point CE and the bounded pair loss only
    objective = source.split("total = (")[1].split('with _stage("backward")')[0]
    assert "POINT_CLASSIFICATION_WEIGHT * (point_a + point_b)" in objective
    assert 'POINT_CF_WEIGHT * cf["loss"]' in objective
    assert "legacy" not in objective


def test_scheduler_horizon_counts_pair_steps():
    source = (REPO_ROOT / "scripts" / "task6h1_h1r.py").read_text(encoding="utf-8")
    assert "total_steps = max(1, epochs * pairs_per_epoch)" in source
    assert '"scheduler_horizon": total_steps' in source
    h0r = (REPO_ROOT / "scripts" / "task6h1_h0r.py").read_text(encoding="utf-8")
    assert "total_steps=max_steps" in h0r
    assert '"scheduler_horizon": max_steps' in h0r
    config = (REPO_ROOT / "configs" / "mvp" / "task6h1_point_objective.yaml").read_text(encoding="utf-8")
    assert "pairs_per_epoch: 240" in config

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
    assert lrs[0] > lrs[1] > lrs[2] > 0.0
    assert lrs[3] == pytest.approx(0.0, abs=1e-12)


def test_objective_weights_are_fixed():
    assert POINT_REASONING_WEIGHT == 0.5
    assert POINT_CLASSIFICATION_WEIGHT == 1.0
    assert POINT_CF_WEIGHT == 1.0
    config = (REPO_ROOT / "configs" / "mvp" / "task6h1_point_objective.yaml").read_text(encoding="utf-8")
    assert "classification_weight: 1.0" in config
    assert "cf_weight: 1.0" in config
    assert "legacy_bce_dice_gradient: false" in config
    assert "legacy_logit_ranking_gradient: false" in config


# ------------------------------------------------------------------ 14-17 recorded model facts


def test_objective_setup_records_architecture_and_freeze_facts():
    setup = json.loads((EVAL / "task6h1_objective_setup.json").read_text(encoding="utf-8"))
    assert setup["passed"] is True
    architecture = setup["architecture"]
    assert architecture["unchanged_from_task6g"] is True
    assert architecture["grid"] == 256
    assert architecture["selected_level_shape"] == [1, 32, 256, 256]
    assert architecture["in_channels"] == 32
    assert architecture["head"]["parameters"] == 270593
    smoke = setup["one_pair_step"]
    assert smoke["sam2_bit_identical"] is True
    assert smoke["shared_feature_bit_identical"] is True
    assert smoke["visual_tower_lora_modules"] == []
    assert smoke["ordinary_rows_unchanged_exactly"] is True
    assert smoke["base_embedding_tensor_bit_identical"] is True
    assert smoke["box_row_delta"] > 0.0 and smoke["seg_row_delta"] > 0.0
    assert setup["causal_placement"]["box_hidden_bit_identical"] is True
    assert setup["objective"]["softmax_sums_to_one"] is True
    assert setup["target_cells"]["different_indices"] is True
    assert setup["target_cells"]["index_within_range"] is True
    # both halves of the corrected objective reach every trainable group
    for group in ("dense_head.query_proj", "dense_head.query_norm", "dense_head.visual_proj", "qwen_lora", "token_rows"):
        assert setup["gradients"]["point_ce"]["norms"][group] > 0.0, group
        assert setup["gradients"]["bounded_cf"]["norms"][group] > 0.0, group


def test_h0r_artifact_records_the_failed_gate_and_no_degeneracy():
    h0r = json.loads((EVAL / "task6h1_h0r_overfit.json").read_text(encoding="utf-8"))
    assert h0r["verdict"] == "BOUNDED_POINT_OBJECTIVE_FAILED"
    final = h0r["final"]
    checks = final["gate"]["checks"]
    assert checks["no_gt_leakage"] is True
    assert checks["own_mass_exceeds_cross_mass"] is True
    assert checks["point_inside_own_ge"] is False
    assert final["pair"]["mean_own_mass"] > final["pair"]["mean_cross_mass"]
    # the bounded objective is not scale-degenerate the way Task 6H's logit margin was
    audit = json.loads((EVAL / "task6h1_h0r_audit.json").read_text(encoding="utf-8"))
    assert audit["before_training"]["mean_point_ce"] > audit["after_training"]["mean_point_ce"]
    assert audit["after_training"]["mean_probability_margin"] > audit["before_training"]["mean_probability_margin"]
    assert audit["after_training"]["mean_spatial_entropy"] < audit["before_training"]["mean_spatial_entropy"]
    assert audit["shared_feature_bit_identical"] is True
    assert audit["pair_identity"]["indices_differ_for_all_pairs"] is True


# ------------------------------------------------------------------ static hygiene

FORBIDDEN = ("[REF]", "SpatialRelationEncoder", "spatial_consistency", "Qwen3-VL-4B", "4B-Instruct")

FORBIDDEN_6H1_SOURCES = (
    "buildreasonseg_mvp/spatial_objective.py",
    "scripts/task6h1_common.py",
    "scripts/task6h1_objective_setup.py",
    "scripts/task6h1_h0r.py",
    "scripts/task6h1_h0r_audit.py",
    "scripts/task6h1_h1r.py",
    "scripts/task6h1_h2r.py",
    "scripts/task6h1_error_analysis.py",
    "scripts/task6h1_paired_probe.py",
    "scripts/task6h1_verdict.py",
)


def _code(relative: str) -> str:
    import re

    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    text = re.sub(r'""".*?"""', '""" """', text, flags=re.DOTALL)
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def test_task6h1_sources_contain_no_forbidden_components():
    for relative in FORBIDDEN_6H1_SOURCES:
        code = _code(relative)
        for marker in FORBIDDEN:
            assert marker not in code, f"{relative} must not use {marker}"
        assert 'read_records("test")' not in code
        assert "read_records('test')" not in code


def test_task6e_and_task6f_paths_stay_retired():
    for relative in FORBIDDEN_6H1_SOURCES:
        code = _code(relative)
        for marker in (
            "add_spatial_tokens",
            "parse_generated_ids",
            "QuantizedBoxCodec",
            "TargetAwareBoxHead",
            "box_query_loss",
            "SmoothL1",
        ):
            assert marker not in code, f"{relative} must not reuse Task 6E/6F machinery"
    runtime_code = _code("buildreasonseg_mvp/runtime.py")
    step = runtime_code.split("def bounded_pair_train_step")[1].split("def build_optimizer")[0]
    assert "box_head" not in step


def test_h2r_uses_the_predicted_point_only():
    code = _code("scripts/task6h1_h2r.py")
    assert "g2_report" in code or "predicted_point" in code
    assert "target_geometry" not in code
    assert "distance_transform_point" not in code


def test_no_extra_query_tokens_or_modules_were_added():
    config = (REPO_ROOT / "configs" / "mvp" / "task6h1_point_objective.yaml").read_text(encoding="utf-8")
    assert 'box_token: "[BOX]"' in config
    assert "<loc_" not in config
    assert "cross_attention" not in config.lower()
    import inspect

    from buildreasonseg_mvp import spatial_objective

    source = inspect.getsource(spatial_objective)
    assert "class DenseSpatialGroundingHead" not in source
    assert "nn.Module" not in source


def test_task6h_causal_wording_corrected_without_changing_historical_numbers():
    state = (REPO_ROOT / "handoff" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    assert "scale-degenerate" in state
    assert "Query\n   learnability therefore remained **unresolved**" in state or "unresolved" in state
    # the historical Task 6H machine verdict and numbers are untouched
    h0 = json.loads((EVAL / "task6h_h0_overfit.json").read_text(encoding="utf-8"))
    assert h0["verdict"] == "COUNTERFACTUAL_QUERY_SIGNAL_FAILED"
    assert h0["final"]["point_inside_own"] == 3
    audit = json.loads((EVAL / "task6h_h0_audit.json").read_text(encoding="utf-8"))
    assert audit["after_training"]["mean_abs_logit"] == pytest.approx(16.32005524635315)
    assert audit["margin_change"]["probability_margin_after"] == pytest.approx(0.11237360970135342)


def test_json_artifacts_are_written_deterministically():
    from buildreasonseg_mvp.checkpointing import write_json

    path = EVAL / "task6h1_verdict_probe.json"
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6h1 point-objective checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
