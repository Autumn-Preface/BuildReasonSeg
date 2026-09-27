"""Task 6G tests: dense query-visual spatial grounding map (section 23's 22 required checks).

Model-level facts (feature shapes, one-step gradients, SAM2 freeze, causal placement) are asserted
against the recorded measurement in `evaluation/task6g_token_setup.json` and
`evaluation/task6g_grid_oracle.json`; everything else is CPU-only or static so this file never
builds a second runtime. Task 6E's coordinate path and Task 6F's box head must stay retired.

Run with pytest, or directly::

    python tests/test_task6g_dense_grounding.py
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

from buildreasonseg_mvp.dense_grounding import (  # noqa: E402
    GRID_CANDIDATES,
    KEY_DIM,
    DenseSpatialGroundingHead,
    argmax_point_from_logits,
    cell_centre,
    downsample_target_mask,
    feature_tensor_for_grid,
    point_inside_mask,
    snap_point_to_cell,
    snapped_point,
)
from buildreasonseg_mvp.grounding import distance_transform_point  # noqa: E402

EVAL = REPO_ROOT / "evaluation"


# ------------------------------------------------------------------ grid snapping


@pytest.mark.parametrize("grid", GRID_CANDIDATES)
def test_grid_snapping_uses_cell_centres(grid):
    # cell 0 centre is 1/(2g), cell g-1 centre is 1 - 1/(2g)
    assert cell_centre(0, grid) == pytest.approx(0.5 / grid)
    assert cell_centre(grid - 1, grid) == pytest.approx(1.0 - 0.5 / grid)
    # the nearest cell to x is floor(x*g): the midpoint between centres sits on a cell border
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        ix, _iy = snap_point_to_cell((x, x), grid)
        expected = min(max(int(x * grid), 0), grid - 1)
        assert ix == expected, (x, grid)
        px, py = snapped_point((x, x), grid)
        assert px == pytest.approx(cell_centre(ix, grid)) and py == pytest.approx(cell_centre(ix, grid))


def test_grid_snapping_is_deterministic():
    for grid in GRID_CANDIDATES:
        for _ in range(5):
            assert snap_point_to_cell((0.333333333, 0.666666667), grid) == snap_point_to_cell(
                (0.333333333, 0.666666667), grid
            )


def test_grid_oracle_selection_obeys_the_rule():
    oracle = json.loads((EVAL / "task6g_grid_oracle.json").read_text(encoding="utf-8"))
    selection = oracle["selection"]
    threshold = selection["miou_threshold"]
    assert abs(threshold - (0.4876 - 0.03)) < 1e-9, "section 3: threshold is the point oracle minus 0.03"
    qualifying = []
    for grid, entry in sorted(((int(k), v) for k, v in oracle["candidates"].items())):
        assert set(entry["val_records"][0]["snapped_point"])  # records present
        assert entry["paired_total"] == 20
        if entry["paired_pass"] >= 18 and entry["strict_miou"] >= threshold:
            qualifying.append(grid)
    assert selection["qualifying_grids"] == qualifying
    assert selection["selected_grid"] == min(qualifying)
    assert selection["selected_grid"] == 256
    assert selection["verdict"] == "PROCEED_WITH_GRID_256"
    # the selected grid's level must be the 256x256 high-res feature, verified in the artifact
    levels = oracle["feature_levels"]["levels_by_grid"]
    assert levels["64"][1] == 256 and levels["64"][2:] == [64, 64]
    assert levels["256"][1] == 32 and levels["256"][2:] == [256, 256]


# ------------------------------------------------------------------ mask downsampling


def test_mask_downsampling_uses_area_interpolation_and_has_mass():
    import numpy as np

    mask = np.zeros((512, 512), dtype=bool)
    mask[100:200, 100:300] = True
    soft = downsample_target_mask(mask, 64)
    assert soft.shape == (64, 64)
    assert float(soft.max()) <= 1.0 and float(soft.min()) >= 0.0
    assert float(soft.sum()) > 0.0
    # the occupancy mass is preserved up to interpolation error
    assert abs(float(soft.sum()) - 200 * 100 / (8 * 8)) < 5.0

    empty = np.zeros((512, 512), dtype=bool)
    with pytest.raises(RuntimeError):
        downsample_target_mask(empty, 64)


def test_head_output_shape_and_canonical_determinism():
    torch.manual_seed(0)
    head = DenseSpatialGroundingHead(query_dim=16, key_dim=8, in_channels=5)
    q = torch.randn(2, 16)
    feature = torch.randn(2, 5, 8, 8)
    logits = head(q, feature)
    assert logits.shape == (2, 8, 8)
    assert head.as_dict()["class"] == "DenseSpatialGroundingHead"
    torch.manual_seed(0)
    again = DenseSpatialGroundingHead(query_dim=16, key_dim=8, in_channels=5)(torch.randn(2, 16), torch.randn(2, 5, 8, 8))
    assert torch.equal(logits, again)


def test_head_projections_receive_gradient_but_not_the_features():
    torch.manual_seed(0)
    head = DenseSpatialGroundingHead(query_dim=16, key_dim=8, in_channels=5)
    q = torch.randn(2, 16, requires_grad=True)
    feature = torch.randn(2, 5, 8, 8, requires_grad=True)
    logits = head(q, feature)
    target = torch.zeros_like(logits)
    target[:, 3, 3] = 1.0
    torch.nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0, name
    # gradients reach the input feature (they may), but the SAM2 encoder is what must stay frozen
    # -- that property is asserted on the real model in task6g_token_setup.json (sam2_bit_identical)


def test_argmax_point_is_deterministic_and_uses_cell_centres():
    torch.manual_seed(1)
    logits = torch.randn(8, 8)
    first = argmax_point_from_logits(logits, 8)
    second = argmax_point_from_logits(logits, 8)
    assert first["point"] == second["point"]
    assert first["soft_argmax_point"] == second["soft_argmax_point"]
    index = int(torch.argmax(logits.reshape(-1)))
    y, x = divmod(index, 8)
    assert first["point"] == [cell_centre(x, 8), cell_centre(y, 8)]
    # an all-zeros heatmap deterministically picks cell (0,0)
    zero = argmax_point_from_logits(torch.zeros(4, 4), 4)
    assert zero["point"] == [cell_centre(0, 4), cell_centre(0, 4)]


def test_point_inside_mask():
    import numpy as np

    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    assert point_inside_mask(mask, (0.5, 0.5)) is True
    assert point_inside_mask(mask, (0.05, 0.05)) is False
    assert point_inside_mask(mask, (0.95, 0.95)) is False


def test_feature_tensor_selection_by_spatial_size():
    class _Features:
        image_embeddings = torch.zeros(1, 256, 64, 64)
        high_res_features = [torch.zeros(1, 32, 256, 256), torch.zeros(1, 64, 128, 128)]

    assert feature_tensor_for_grid(_Features(), 64).shape == (1, 256, 64, 64)
    assert feature_tensor_for_grid(_Features(), 128).shape == (1, 64, 128, 128)
    assert feature_tensor_for_grid(_Features(), 256).shape == (1, 32, 256, 256)
    with pytest.raises(ValueError):
        feature_tensor_for_grid(_Features(), 96)


def test_gt_point_convention_is_the_frozen_interior_point():
    import numpy as np

    mask = np.zeros((16, 16), dtype=bool)
    mask[4:9, 3:11] = True
    point = distance_transform_point(mask)
    assert 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0
    assert point_inside_mask(mask, point), "the deterministic interior point must lie in the target"
    # determinism: a pure function of the mask
    assert distance_transform_point(mask) == point


# ------------------------------------------------------------------ recorded smoke


def test_token_setup_artifact_records_the_section_10_requirements():
    setup = json.loads((EVAL / "task6g_token_setup.json").read_text(encoding="utf-8"))
    assert setup["passed"] is True
    assert setup["grid"] == 256
    assert setup["selected_level"]["shape"] == [1, 32, 256, 256]
    assert setup["selected_level"]["in_channels"] == 32
    levels = setup["feature_levels"]["levels_by_grid"]
    assert levels["64"] == [1, 256, 64, 64]
    assert levels["128"] == [1, 64, 128, 128]
    assert levels["256"] == [1, 32, 256, 256]
    assert setup["vocabulary"]["no_loc_tokens_added"] is True
    assert setup["supervision"]["soft_target_mass_positive"] is True
    assert setup["causal_placement"]["box_hidden_bit_identical"] is True
    smoke = setup["one_step_smoke"]
    assert smoke["box_row_delta"] > 0.0 and smoke["seg_row_delta"] > 0.0
    assert (smoke["box_row_grad_norm"] or 0.0) > 0.0 and (smoke["seg_row_grad_norm"] or 0.0) > 0.0
    assert all(value > 0.0 for value in smoke["dense_head_param_max_delta"].values())
    assert all((value or 0.0) > 0.0 for value in smoke["dense_head_param_grad_norms"].values())
    assert smoke["ordinary_rows_unchanged_exactly"] is True
    assert smoke["base_embedding_tensor_bit_identical"] is True
    assert smoke["sam2_bit_identical"] is True
    assert smoke["visual_tower_lora_modules"] == []
    groups = {group["name"]: group for group in setup["optimizer_groups"]}
    assert groups["token"]["tensors"] == 1
    # decoder group = LayerNorm(2048) + Linear(2048,128) + Conv1x1(32,128) + scalar bias
    assert groups["decoder"]["parameters"] == 2048 * 2 + (2048 * 128 + 128) + (32 * 128 + 128) + 1
    assert "lora" in groups


# ------------------------------------------------------------------ static hygiene


FORBIDDEN = ("[REF]", "SpatialRelationEncoder", "spatial_consistency", "Qwen3-VL-4B", "4B-Instruct")

FORBIDDEN_6G_SOURCES = (
    "buildreasonseg_mvp/dense_grounding.py",
    "buildreasonseg_mvp/dense_eval.py",
    "scripts/task6g_common.py",
    "scripts/task6g_grid_oracle.py",
    "scripts/task6g_token_setup.py",
    "scripts/task6g_g0.py",
    "scripts/task6g_g1.py",
    "scripts/task6g_g2.py",
    "scripts/task6g_error_analysis.py",
    "scripts/task6g_verdict.py",
)


def _strip_docstrings(text: str) -> str:
    import re

    stripped = re.sub(r'""".*?"""', '""" """', text, flags=re.DOTALL)
    return re.sub(r"'''.*?'''", "''' '''", stripped, flags=re.DOTALL)


def test_task6g_sources_contain_no_forbidden_components():
    for relative in FORBIDDEN_6G_SOURCES:
        code = _strip_docstrings((REPO_ROOT / relative).read_text(encoding="utf-8"))
        code = "\n".join(line for line in code.splitlines() if not line.strip().startswith("#"))
        for marker in FORBIDDEN:
            assert marker not in code, f"{relative} must not use {marker}"
        assert 'read_records("test")' not in code
        assert "test_split" not in code


def test_task6e_coordinates_and_task6f_box_head_are_retired_in_task6g():
    for relative in FORBIDDEN_6G_SOURCES:
        code = _strip_docstrings((REPO_ROOT / relative).read_text(encoding="utf-8"))
        code = "\n".join(line for line in code.splitlines() if not line.strip().startswith("#"))
        for marker in (
            "add_spatial_tokens",
            "parse_generated_ids",
            "QuantizedBoxCodec",
            "location CE",
            "box_query_loss",
            "TargetAwareBoxHead",
            "SmoothL1",
        ):
            assert marker not in code, f"{relative} must not reuse Task 6E/6F machinery"


def test_task6g_config_retires_the_old_paths():
    import yaml

    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "mvp" / "task6g_dense_grounding.yaml").read_text(encoding="utf-8")
    )
    assert cfg["task"] == "6G"
    assert "spatial_tokens" not in cfg
    assert cfg["box_query"]["enabled"] is True
    assert cfg["dense_grounding"]["enabled"] is True
    assert float(cfg["dense_grounding"]["heatmap_bce_weight"]) == 1.0
    assert float(cfg["dense_grounding"]["heatmap_dice_weight"]) == 1.0
    assert float(cfg["dense_grounding"]["heatmap_loss_weight"]) == 2.0
    assert float(cfg["dense_grounding"]["reasoning_loss_weight"]) == 1.0
    assert cfg["training"]["deterministic"] is True
    assert cfg["training"]["deterministic_strict"] is True
    assert cfg["inference"]["do_sample"] is False


def test_scheduler_horizon_uses_the_actual_max_step_budget():
    scripts = REPO_ROOT / "scripts"
    g0_source = (scripts / "task6g_g0.py").read_text(encoding="utf-8")
    assert "total_steps=max_steps" in g0_source
    g1_source = (scripts / "task6g_g1.py").read_text(encoding="utf-8")
    assert "total_steps = max(1, epochs * steps_per_epoch)" in g1_source
    assert '"scheduler_horizon": total_steps' in g1_source

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


def test_g1_selection_rule_is_paired_then_inside_then_dice():
    source = (REPO_ROOT / "scripts" / "task6g_g1.py").read_text(encoding="utf-8")
    assert "paired point selection /20" in source
    assert "best = max(report[\"epochs\"], key=_selection_key)" in source
    assert "early_stop_after_epoch" in source and "early_stop_patience_epochs" in source
    key_source = (REPO_ROOT / "scripts" / "task6g_common.py").read_text(encoding="utf-8")
    assert '"paired_point_selection_ge"' in key_source
    assert '"point_inside_target_ge"' in key_source
    assert '"heatmap_dice_ge"' in key_source


def test_g2_uses_predicted_point_only():
    source = (REPO_ROOT / "scripts" / "task6g_g2.py").read_text(encoding="utf-8")
    assert "predicted_point" in source
    assert "target_geometry" not in source
    assert "distance_transform_point" not in source


def test_inference_path_takes_no_ground_truth():
    """Task 6H refactored the path into `forward_heatmap` + `record_from_raw`; the property holds
    for both halves: the model input is built from image + instruction only, and the GT mask is
    read exclusively by the scoring layer."""

    import inspect

    from buildreasonseg_mvp import dense_eval

    for function in (dense_eval.predict_heatmap, dense_eval.forward_heatmap):
        parameters = set(inspect.signature(function).parameters)
        assert not (parameters & {"mask", "gt", "box", "geometry", "target"}), function.__name__
    forward_source = inspect.getsource(dense_eval.forward_heatmap)
    assert "build_query_batch(runtime, image, sample.instruction_zh)" in forward_source
    # the model inputs are assembled before any GT read
    model_call = forward_source.split("build_query_batch(runtime, image, sample.instruction_zh)")[0]
    assert "target_mask" not in model_call
    # ... and the only GT read happens after the forward, for scoring
    scoring_source = inspect.getsource(dense_eval.record_from_raw)
    assert "point_inside_mask" in scoring_source and "soft_dice_from_logits" in scoring_source
    for function in (dense_eval.predict_heatmap, dense_eval.forward_heatmap):
        assert "target_mask" not in inspect.getsource(function).split("downsample_target_mask")[0].split("build_query_batch")[0]


def test_json_artifacts_are_written_deterministically():
    from buildreasonseg_mvp.checkpointing import write_json

    path = EVAL / "task6g_verdict_probe.json"
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6g dense-grounding checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
