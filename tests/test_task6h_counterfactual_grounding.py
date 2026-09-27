"""Task 6H tests: counterfactual pair construction, region-ranking loss and pair-step semantics.

Model-level facts (pair step, SAM2 freeze, shared feature identity, causal placement) are asserted
against the recorded measurement in `evaluation/task6h_token_setup.json`; the pair manifest is
asserted against `evaluation/task6h_pair_manifest.json`; everything else is CPU-only or static.
Task 6E's coordinate path and Task 6F's box head must stay retired.

Run with pytest, or directly::

    python tests/test_task6h_counterfactual_grounding.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp.counterfactual import (  # noqa: E402
    CF_MARGIN,
    PAIR_CF_WEIGHT,
    PAIR_HEATMAP_WEIGHT,
    PAIR_REASONING_WEIGHT,
    build_canonical_pairs,
    counterfactual_loss,
    pair_manifest,
    pair_ranking_summary,
    pair_step_loss,
    region_score,
    region_scores,
)

EVAL = REPO_ROOT / "evaluation"


# ------------------------------------------------------------------ pair construction


def _synthetic_payload(pairs):
    return {"P": {"pairs": pairs}}


def _records(image_id: str, target: int, sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "image_id": image_id,
        "target_component_id": target,
        "level": 1,
        "query_type": "largest",
    }


def test_canonical_pair_builder_accepts_valid_pairs_and_orders_them():
    payload = _synthetic_payload(
        [
            {"image_id": "img1", "a": "s1a", "b": "s1b"},
            {"image_id": "img2", "a": "s2a", "b": "s2b"},
        ]
    )
    records = {
        "s1a": _records("img1", 3, "s1a"),
        "s1b": _records("img1", 7, "s1b"),
        "s2a": _records("img2", 1, "s2a"),
        "s2b": _records("img2", 2, "s2b"),
    }
    triples = build_canonical_pairs(payload, records)
    assert [pair.a for pair, _a, _b in triples] == ["s1a", "s2a"]
    assert [pair.index for pair, _a, _b in triples] == [0, 1]
    assert triples[0][0].image_id == "img1" and triples[0][0].target_a != triples[0][0].target_b


def test_canonical_pair_builder_rejects_wrong_image():
    payload = _synthetic_payload([{"image_id": "img1", "a": "s1a", "b": "s1b"}])
    records = {"s1a": _records("img1", 3, "s1a"), "s1b": _records("img2", 7, "s1b")}
    with pytest.raises(RuntimeError):
        build_canonical_pairs(payload, records)


def test_canonical_pair_builder_rejects_identical_targets():
    payload = _synthetic_payload([{"image_id": "img1", "a": "s1a", "b": "s1b"}])
    records = {"s1a": _records("img1", 5, "s1a"), "s1b": _records("img1", 5, "s1b")}
    with pytest.raises(RuntimeError):
        build_canonical_pairs(payload, records)


def test_canonical_pair_builder_rejects_reused_samples():
    payload = _synthetic_payload(
        [{"image_id": "img1", "a": "s1a", "b": "s1b"}, {"image_id": "img1", "a": "s1a", "b": "s1c"}]
    )
    records = {
        "s1a": _records("img1", 3, "s1a"),
        "s1b": _records("img1", 7, "s1b"),
        "s1c": _records("img1", 9, "s1c"),
    }
    with pytest.raises(RuntimeError):
        build_canonical_pairs(payload, records)


def test_recorded_pair_manifest_has_240_canonical_pairs():
    manifest = json.loads((EVAL / "task6h_pair_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pair_count"] == 240
    assert manifest["image_count"] == 240
    assert manifest["test_split_used"] is False
    assertions = manifest["assertions"]
    assert assertions["same_source_image"] is True
    assert assertions["different_sample_ids"] is True
    assert assertions["different_target_component_ids"] is True
    assert assertions["target_masks_differ"] is True
    assert assertions["deterministic_order"] is True
    # every pair row carries a mask-overlap diagnostic and no pair is discarded
    rows = manifest["pairs"]
    assert len(rows) == 240
    assert all("mask_overlap" in row for row in rows)
    assert all(not row["mask_overlap"]["same_mask"] for row in rows)
    assert manifest["overlap_summary"]["pairs_with_overlapping_targets"] == 0


def test_pair_ordering_is_deterministic_and_hashed():
    payload = _synthetic_payload([{"image_id": "img1", "a": "s1a", "b": "s1b"}])
    records = {"s1a": _records("img1", 3, "s1a"), "s1b": _records("img1", 7, "s1b")}
    first = pair_manifest(build_canonical_pairs(payload, records), 256)
    second = pair_manifest(build_canonical_pairs(payload, records), 256)
    assert first["pair_identity_sha256"] == second["pair_identity_sha256"]
    assert len(first["pair_identity_sha256"]) == 64


# ------------------------------------------------------------------ region scores / loss


def test_region_score_is_the_mask_mean_logit():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert float(region_score(logits, mask)) == pytest.approx(1.0)
    half = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert float(region_score(logits, half)) == pytest.approx(1.5)
    empty = torch.zeros(2, 2)
    assert float(region_score(logits, empty)) == 0.0  # clamped denominator


def test_own_and_cross_scores_use_the_correct_masks():
    logits_a = torch.zeros(4, 4)
    logits_a[0, 0] = 5.0
    logits_b = torch.zeros(4, 4)
    logits_b[3, 3] = 5.0
    mask_a = torch.zeros(4, 4)
    mask_a[0, 0] = 1.0
    mask_b = torch.zeros(4, 4)
    mask_b[3, 3] = 1.0
    scores = region_scores(logits_a, logits_b, mask_a, mask_b)
    assert float(scores["s_aa"]) == pytest.approx(5.0)
    assert float(scores["s_ab"]) == pytest.approx(0.0)
    assert float(scores["s_bb"]) == pytest.approx(5.0)
    assert float(scores["s_ba"]) == pytest.approx(0.0)
    assert float(scores["mean_margin"]) == pytest.approx(5.0)


def test_pair_loss_sign_and_margin():
    """Raising the own score lowers L_cf; raising the cross score raises it."""

    def scores_for(margin_a: float, margin_b: float) -> dict:
        s_aa = torch.tensor(margin_a)
        s_ab = torch.tensor(0.0)
        s_bb = torch.tensor(margin_b)
        s_ba = torch.tensor(0.0)
        return {
            "s_aa": s_aa,
            "s_ab": s_ab,
            "s_bb": s_bb,
            "s_ba": s_ba,
            "margin_a": s_aa - s_ab,
            "margin_b": s_bb - s_ba,
            "mean_margin": 0.5 * ((s_aa - s_ab) + (s_bb - s_ba)),
            "raw": {},
        }

    good = counterfactual_loss(scores_for(2.0, 2.0))
    bad = counterfactual_loss(scores_for(0.0, 0.0))
    worse = counterfactual_loss(scores_for(-1.0, -1.0))
    assert good["raw"] < bad["raw"] < worse["raw"], "L_cf must fall as the own-vs-cross margin grows"
    assert good["pair_ranking_pass"] is True and good["strict_margin_pass"] is True
    assert bad["pair_ranking_pass"] is False  # margin exactly 0 is not a pass
    assert worse["pair_ranking_pass"] is False
    assert good["margin"] == CF_MARGIN == 1.0

    # lowering the own score alone increases the loss
    lowered = counterfactual_loss(scores_for(1.0, 2.0))
    assert lowered["raw"] > good["raw"]
    # raising the cross score alone (equivalent to lowering this side's margin) increases it
    raised_cross = counterfactual_loss(scores_for(2.0, 1.0))
    assert raised_cross["raw"] > good["raw"]


def test_loss_decreases_with_own_score_and_increases_with_cross_score():
    base_logits_a = torch.zeros(4, 4)
    base_logits_b = torch.zeros(4, 4)
    mask_a = torch.zeros(4, 4)
    mask_a[0, 0] = 1.0
    mask_b = torch.zeros(4, 4)
    mask_b[3, 3] = 1.0
    base = float(
        counterfactual_loss(region_scores(base_logits_a, base_logits_b, mask_a, mask_b))["loss"]
    )
    own_up = base_logits_a.clone()
    own_up[0, 0] = 3.0
    assert float(counterfactual_loss(region_scores(own_up, base_logits_b, mask_a, mask_b))["loss"]) < base
    cross_up = base_logits_a.clone()
    cross_up[3, 3] = 3.0  # B's target region rises on A's heatmap -> A's margin falls
    assert float(counterfactual_loss(region_scores(cross_up, base_logits_b, mask_a, mask_b))["loss"]) > base


def test_pair_step_loss_weights_are_fixed():
    assert PAIR_REASONING_WEIGHT == 0.5
    assert PAIR_HEATMAP_WEIGHT == 1.0
    assert PAIR_CF_WEIGHT == 2.0
    scores = {
        "s_aa": torch.tensor(2.0),
        "s_ab": torch.tensor(0.0),
        "s_bb": torch.tensor(2.0),
        "s_ba": torch.tensor(0.0),
        "margin_a": torch.tensor(2.0),
        "margin_b": torch.tensor(2.0),
        "mean_margin": torch.tensor(2.0),
        "raw": {},
    }
    result = pair_step_loss(
        torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.5), scores
    )
    # 0.5 * mean(reasoning) + 1.0 * sum(heatmaps) + 2.0 * L_cf
    expected = 0.5 * 1.0 + 1.0 * (0.5 + 0.5) + 2.0 * result["cf"]["raw"]
    assert result["raw"]["total"] == pytest.approx(expected)


def test_pair_ranking_summary_counts_passes():
    rows = [
        {"pair_ranking_pass": True, "strict_margin_pass": True, "mean_margin": 1.5},
        {"pair_ranking_pass": True, "strict_margin_pass": False, "mean_margin": 0.2},
        {"pair_ranking_pass": False, "strict_margin_pass": False, "mean_margin": -0.4},
    ]
    summary = pair_ranking_summary(rows)
    assert summary["pair_count"] == 3
    assert summary["pair_ranking_pass"] == 2
    assert summary["strict_margin_pass"] == 1
    assert summary["mean_own_minus_cross_margin"] == pytest.approx((1.5 + 0.2 - 0.4) / 3)


# ------------------------------------------------------------------ recorded pair step


def test_token_setup_records_architecture_identity_and_pair_semantics():
    setup = json.loads((EVAL / "task6h_token_setup.json").read_text(encoding="utf-8"))
    assert setup["passed"] is True
    architecture = setup["architecture"]
    assert architecture["unchanged_from_task6g"] is True
    assert architecture["grid"] == 256
    assert architecture["selected_level_shape"] == [1, 32, 256, 256]
    assert architecture["in_channels"] == 32
    assert architecture["head"]["parameters"] == 270593
    assert setup["causal_placement"]["box_hidden_bit_identical"] is True
    smoke = setup["one_pair_step"]
    assert smoke["sam2_bit_identical"] is True
    assert smoke["shared_feature_bit_identical"] is True
    assert smoke["box_row_delta"] > 0.0 and smoke["seg_row_delta"] > 0.0
    assert (smoke["box_row_grad_norm"] or 0.0) > 0.0
    assert all(value > 0.0 for value in smoke["dense_head_param_max_delta"].values())
    assert smoke["ordinary_rows_unchanged_exactly"] is True
    assert smoke["base_embedding_tensor_bit_identical"] is True
    assert smoke["visual_tower_lora_modules"] == []
    # the pair losses of BOTH samples are recorded in one step
    losses = smoke["losses"]
    assert losses["heatmap_a"] is not None and losses["heatmap_b"] is not None
    assert losses["bce_a"] is not None and losses["bce_b"] is not None
    assert losses["cf_a"] is not None and losses["cf_b"] is not None
    assert setup["region_scores_before"] and setup["region_scores_after"]


def test_pair_step_runs_both_forwards_before_one_backward():
    import inspect

    from buildreasonseg_mvp.runtime import MvpRuntime

    source = inspect.getsource(MvpRuntime.pair_train_step)
    assert source.count("box_query_forward(self.model.qwen") == 2, "both instructions need a forward"
    assert source.count("total.backward()") == 1, "one pair step must use a single backward"
    assert source.index("total.backward()") < source.index("optimizer.step()")
    assert "feature_tensor_for_grid" in source
    # one shared feature tensor, computed once
    assert source.count("spatial_feature = feature_tensor_for_grid") == 1


def test_scheduler_horizon_counts_pair_steps():
    source = (REPO_ROOT / "scripts" / "task6h_h1.py").read_text(encoding="utf-8")
    assert "total_steps = max(1, epochs * pairs_per_epoch)" in source
    assert '"scheduler_horizon": total_steps' in source
    h0 = (REPO_ROOT / "scripts" / "task6h_h0.py").read_text(encoding="utf-8")
    assert "total_steps=max_steps" in h0
    assert '"scheduler_horizon": max_steps' in h0
    config = (REPO_ROOT / "configs" / "mvp" / "task6h_counterfactual_grounding.yaml").read_text(
        encoding="utf-8"
    )
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


# ------------------------------------------------------------------ static hygiene

FORBIDDEN = ("[REF]", "SpatialRelationEncoder", "spatial_consistency", "Qwen3-VL-4B", "4B-Instruct")

FORBIDDEN_6H_SOURCES = (
    "buildreasonseg_mvp/counterfactual.py",
    "buildreasonseg_mvp/pair_eval.py",
    "scripts/task6h_common.py",
    "scripts/task6h_pairs.py",
    "scripts/task6h_token_setup.py",
    "scripts/task6h_h0.py",
    "scripts/task6h_h0_audit.py",
    "scripts/task6h_h1.py",
    "scripts/task6h_h2.py",
    "scripts/task6h_error_analysis.py",
    "scripts/task6h_paired_probe.py",
    "scripts/task6h_verdict.py",
)


def _strip_docstrings(text: str) -> str:
    import re

    stripped = re.sub(r'""".*?"""', '""" """', text, flags=re.DOTALL)
    return re.sub(r"'''.*?'''", "''' '''", stripped, flags=re.DOTALL)


def _code(relative: str) -> str:
    text = _strip_docstrings((REPO_ROOT / relative).read_text(encoding="utf-8"))
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def test_task6h_sources_contain_no_forbidden_components():
    for relative in FORBIDDEN_6H_SOURCES:
        code = _code(relative)
        for marker in FORBIDDEN:
            assert marker not in code, f"{relative} must not use {marker}"
        assert 'read_records("test")' not in code
        assert "read_records('test')" not in code
        assert 'test_split_used": True' not in code


def test_task6e_and_task6f_paths_stay_retired_in_task6h():
    for relative in FORBIDDEN_6H_SOURCES:
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
    # the 6H runtime step must not install or call the 6F box head
    runtime_code = _code("buildreasonseg_mvp/runtime.py")
    assert "install_box_head" in runtime_code  # the 6F method still exists for history
    assert "def pair_train_step" in runtime_code
    pair_step = runtime_code.split("def pair_train_step")[1].split("def build_optimizer")[0]
    assert "box_head" not in pair_step


def test_h2_uses_the_predicted_point_only():
    code = _code("scripts/task6h_h2.py")
    assert "predicted_point" in code or "g2_report" in code
    assert "target_geometry" not in code
    assert "distance_transform_point" not in code


def test_config_keeps_the_task6h_architecture_and_fixed_weights():
    import yaml

    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "mvp" / "task6h_counterfactual_grounding.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["task"] == "6H"
    assert "spatial_tokens" not in cfg
    assert cfg["box_query"]["enabled"] is True
    assert cfg["dense_grounding"]["enabled"] is True
    assert cfg["dense_grounding"]["grid"] == 256
    assert cfg["dense_grounding"]["key_dim"] == 128
    assert cfg["counterfactual"]["pairs_per_epoch"] == 240
    assert float(cfg["counterfactual"]["margin"]) == 1.0
    assert float(cfg["counterfactual"]["pair_reasoning_weight"]) == 0.5
    assert float(cfg["counterfactual"]["pair_heatmap_weight"]) == 1.0
    assert float(cfg["counterfactual"]["pair_cf_weight"]) == 2.0
    assert cfg["training"]["deterministic"] is True
    assert cfg["training"]["deterministic_strict"] is True
    assert cfg["inference"]["do_sample"] is False


def test_no_extra_query_tokens_or_modules_were_added():
    cfg_source = (
        REPO_ROOT / "configs" / "mvp" / "task6h_counterfactual_grounding.yaml"
    ).read_text(encoding="utf-8")
    assert "box_token: \"[BOX]\"" in cfg_source
    assert "<loc_" not in cfg_source
    # the dense head is imported from Task 6G, not reimplemented
    import inspect

    from buildreasonseg_mvp import dense_grounding

    assert "class DenseSpatialGroundingHead" in inspect.getsource(dense_grounding)
    counterfactual_source = inspect.getsource(__import__(
        "buildreasonseg_mvp.counterfactual", fromlist=["x"]
    ))
    assert "class DenseSpatialGroundingHead" not in counterfactual_source
    assert "cross_attention" not in counterfactual_source.lower()


def test_project_state_watt_wording_is_neutral_and_history_is_intact():
    state = (REPO_ROOT / "handoff" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    assert "Watt Toolkit stopped during training" not in state
    assert "Watt is transport-only when needed" in state
    # historical per-task Watt records are preserved elsewhere in the handoff set
    from_dsh = (REPO_ROOT / "handoff" / "FROM_DSH.md").read_text(encoding="utf-8")
    assert "Watt" in from_dsh


def test_json_artifacts_are_written_deterministically():
    from buildreasonseg_mvp.checkpointing import write_json

    path = EVAL / "task6h_verdict_probe.json"
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
    print(f"\n{len(tests) - failures}/{len(tests)} task6h counterfactual checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
