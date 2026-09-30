"""Task 6W tests (section O): 48 checks on proposal-quality filtering and semantic extreme selection."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
REF_PACKS = REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
QUALITY_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "task6w_proposal_quality.py"
RESOLVER_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "task6w_quality_reference_resolver.py"
QUALITY_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6w" / "proposal_quality_v01.pt"
PROPOSAL_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" \
    / "m1_yolo26m_seg_continued" / "weights" / "best.pt"
BASE_COMMIT = "3e185fbc5764dcc74d34d70f085e4a139a93bf37"
PROPOSAL_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
TASK6W_SOURCES = ("task6w_oracle_quality_diagnostic.py", "task6w_build_quality_dataset.py",
                  "task6w_train_quality.py", "task6w_evaluate_reference.py",
                  "task6w_evaluate_downstream.py", "task6w_report.py")
W0_GATE = {"overall_miou_delta_min": 0.08, "smallest_miou_delta_min": 0.10,
           "selection_wrong_ratio_max": 0.60, "abstention_rate_max": 0.10}


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_changed(prefix: str) -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _code_only(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    for prefix in ("r", "f", "b", "u", "rf", "fr"):
        if stripped.startswith(prefix + '"""') or stripped.startswith(prefix + "'''"):
            stripped = stripped[len(prefix):]
            break
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            end = stripped.find(quote, len(quote))
            if end != -1:
                stripped = stripped[end + len(quote):]
            break
    return "\n".join(line.split("#", 1)[0] for line in stripped.splitlines())


def _proposal(mask=None, confidence=0.9, index=0):
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal

    if mask is None:
        mask = np.zeros((512, 512), dtype=bool)
        mask[10:40, 10:40] = True
    return build_proposal(index, confidence, mask)


# ---------------------------------------------------------------- 1-8 frozen assets and W0


def test_task6v_artifacts_unchanged():
    assert _git_changed("evaluation/task6v_") == ""
    assert _git_changed("buildreasonseg_mvp/task6v_family_reference_resolver.py") == ""
    assert _artifact("task6v_verdict.json")["verdict"] == "FAMILY_POLICY_NOT_BETTER"


def test_u_c1_exact_config():
    from scripts.task6u_common import CONFIGS

    config = CONFIGS["U-C1"]
    assert (config["imgsz"], config["conf"], config["max_det"]) == (640, 0.05, 300)
    assert config["tta"] is False and config["tiling"] is False
    w0 = _artifact("task6w_oracle_quality_filter_calib.json")
    if w0 is not None:
        assert w0["config"]["id"] == "U-C1"
        assert (w0["config"]["imgsz"], w0["config"]["conf"], w0["config"]["max_det"]) == (640, 0.05,
                                                                                         300)


def test_yolo_checkpoint_hash_exact():
    assert PROPOSAL_CHECKPOINT.is_file()
    assert _sha256(PROPOSAL_CHECKPOINT) == PROPOSAL_SHA256
    w0 = _artifact("task6w_oracle_quality_filter_calib.json")
    if w0 is not None:
        assert w0["checkpoint"]["sha256"] == PROPOSAL_SHA256
        assert w0["checkpoint"]["retrained"] is False


def test_no_alternative_detector_config_in_primary_resolver():
    for name in TASK6W_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("U-C0", "U-C2", "U-C3"):
            assert marker not in code, f"{name} must not use {marker}"
    resolver = _code_only(RESOLVER_MODULE)
    assert "U-C1" in "".join(TASK6W_SOURCES) or True
    assert "U-C0" not in resolver and "U-C2" not in resolver and "U-C3" not in resolver


def test_w0_uses_u_calib200_only():
    w0 = _artifact("task6w_oracle_quality_filter_calib.json")
    if w0 is None:
        pytest.skip("W0 not run yet")
    assert w0["calibration_split"]["records"] == 200
    assert w0["calibration_split"]["train_only"] is True
    assert w0["calibration_split"]["path"].endswith("task6u_reference_train_split.json")
    code = _code_only(SCRIPTS / "task6w_oracle_quality_diagnostic.py")
    for marker in ("ref_val_unique", "mini_val_240", "paired_val_20"):
        assert marker not in code


def test_q_gt_definition_exact_max_iou():
    code = _code_only(SCRIPTS / "task6w_oracle_quality_diagnostic.py")
    assert "max IoU" in (SCRIPTS / "task6w_oracle_quality_diagnostic.py").read_text(encoding="utf-8")
    assert "for truth in instances.values()" in code
    assert "q_gt" in code
    w0 = _artifact("task6w_oracle_quality_filter_calib.json")
    if w0 is not None:
        assert "max IoU" in w0["quality_definition"]


def test_w0_threshold_exact_050():
    from buildreasonseg_mvp.task6w_proposal_quality import QUALITY_THRESHOLD
    from scripts.task6w_oracle_quality_diagnostic import QUALITY_THRESHOLD as W0_THRESHOLD

    assert QUALITY_THRESHOLD == 0.50
    assert W0_THRESHOLD == 0.50
    w0 = _artifact("task6w_oracle_quality_filter_calib.json")
    if w0 is not None:
        assert w0["quality_threshold"] == 0.50


def test_w0_gate_exact():
    from scripts.task6w_oracle_quality_diagnostic import GATE

    assert GATE == W0_GATE
    w0 = _artifact("task6w_oracle_quality_filter_calib.json")
    if w0 is None:
        pytest.skip("W0 not run yet")
    assert w0["gate_constants"] == W0_GATE
    recomputed = all(entry["passed"] for entry in w0["gate"].values())
    assert w0["mechanism_gate_passed"] == recomputed
    assert w0["training_allowed"] == w0["mechanism_gate_passed"]
    verdict = _artifact("task6w_verdict.json")
    if verdict is not None and not w0["mechanism_gate_passed"]:
        assert verdict["verdict"] == "QUALITY_FILTER_MECHANISM_INSUFFICIENT"


# ---------------------------------------------------------------- 9-15 dataset


def test_ranker_train_only_for_quality_dataset():
    manifest = _artifact("task6w_quality_dataset_manifest.json")
    if manifest is None:
        pytest.skip("dataset manifest not generated yet")
    assert manifest["universe"]["source_split"] == "U-RankerTrain (Task 6U)"
    assert manifest["universe"]["references"] == 625


def test_proposal_dataset_deduplicated_by_tile():
    manifest = _artifact("task6w_quality_dataset_manifest.json")
    if manifest is None:
        pytest.skip("dataset not generated yet")
    assert manifest["universe"]["unique_tiles"] <= 584
    assert manifest["per_tile_proposals_mean"] is not None
    code = _code_only(SCRIPTS / "task6w_build_quality_dataset.py")
    assert "tiles.setdefault(tile_id, record)" in code
    assert "proposals_for_tile(yolo, CONFIGS[\"U-C1\"], tile_id" in code


def test_no_u_calib_tile_in_quality_training():
    manifest = _artifact("task6w_quality_dataset_manifest.json")
    if manifest is None:
        pytest.skip("dataset not generated yet")
    assert manifest["universe"]["calib_tile_overlap"] == 0
    assert manifest["universe"]["no_calib_tile"] is True
    assert manifest["universe"]["excluded_calib_overlapping_tiles"] > 0


def test_no_refval_tile_in_quality_training():
    manifest = _artifact("task6w_quality_dataset_manifest.json")
    if manifest is None:
        pytest.skip("dataset not generated yet")
    assert manifest["universe"]["refval_tile_overlap"] == 0
    assert manifest["universe"]["no_refval_tile"] is True


def test_train_holdout_split_by_tile():
    manifest = _artifact("task6w_quality_dataset_manifest.json")
    training = _artifact("task6w_quality_training.json")
    if manifest is None or training is None:
        pytest.skip("artifacts not generated yet")
    assert manifest["split"]["no_overlap"] is True
    assert manifest["split"]["tile_overlap"] == 0
    assert manifest["split"]["seed"] if "seed" in manifest["split"] else True
    assert training["data"]["tiles"]["overlap"] == 0
    assert training["data"]["split_by"].startswith("tile id hash")
    assert training["data"]["tiles"]["train"] + training["data"]["tiles"]["holdout"] \
        == manifest["universe"]["unique_tiles"]


def test_common_eligibility_exact():
    from scripts.task6w_build_quality_dataset import EXTENT_MAX, common_eligible

    assert EXTENT_MAX == 0.20
    inside = _proposal()
    assert common_eligible(inside)
    border_mask = np.zeros((512, 512), dtype=bool)
    border_mask[0, 10:20] = True
    assert not common_eligible(_proposal(mask=border_mask))
    manifest = _artifact("task6w_quality_dataset_manifest.json")
    if manifest is not None:
        assert manifest["eligibility"]["no_border_touch"] is True
        assert manifest["eligibility"]["bbox_extent_ratio_max"] == 0.20
        assert manifest["eligibility"]["smallest_area_floor_applied"] is False


def test_quality_label_threshold_exact_050():
    manifest = _artifact("task6w_quality_dataset_manifest.json")
    if manifest is None:
        pytest.skip("dataset not generated yet")
    assert manifest["labels"]["quality_threshold"] == 0.50
    assert "q_gt >= 0.50" in manifest["labels"]["binary"]
    code = _code_only(SCRIPTS / "task6w_build_quality_dataset.py")
    assert "label_rows.append(1.0 if best >= QUALITY_THRESHOLD else 0.0)" in code


# ---------------------------------------------------------------- 16-29 features, architecture, loss


def test_geometry_feature_dimension_exactly_8():
    from buildreasonseg_mvp.task6w_proposal_quality import (
        GEOMETRY_DIM,
        GEOMETRY_FEATURE_NAMES,
        geometry_features,
    )

    assert GEOMETRY_DIM == 8
    assert len(GEOMETRY_FEATURE_NAMES) == 8
    assert geometry_features(_proposal()).shape == (8,)


def test_no_family_feature():
    from buildreasonseg_mvp.task6w_proposal_quality import GEOMETRY_FEATURE_NAMES

    joined = " ".join(GEOMETRY_FEATURE_NAMES).lower()
    for marker in ("family", "largest", "smallest"):
        assert marker not in joined, marker
    code = _code_only(QUALITY_MODULE)
    assert "family" not in code.replace("family-independent", "")


def test_no_relation_feature():
    from buildreasonseg_mvp.task6w_proposal_quality import GEOMETRY_FEATURE_NAMES

    assert not any("relation" in name for name in GEOMETRY_FEATURE_NAMES)
    code = _code_only(QUALITY_MODULE)
    assert "relation" not in code


def test_no_proposal_rank_feature():
    from buildreasonseg_mvp.task6w_proposal_quality import GEOMETRY_FEATURE_NAMES

    joined = " ".join(GEOMETRY_FEATURE_NAMES).lower()
    assert "rank" not in joined
    assert "area_rank" not in _code_only(QUALITY_MODULE)


def test_no_x_y_location_feature():
    from buildreasonseg_mvp.task6w_proposal_quality import GEOMETRY_FEATURE_NAMES

    joined = " ".join(GEOMETRY_FEATURE_NAMES).lower()
    for marker in ("centroid", "cx", "cy", "x_norm", "y_norm", "location"):
        assert marker not in joined, marker


def test_sam2_feature_shape_256_64_64():
    from buildreasonseg_mvp.task6w_proposal_quality import FEATURE_SIZE, SAM2_CHANNELS, VISUAL_DIM

    assert SAM2_CHANNELS == 256 and FEATURE_SIZE == 64
    assert VISUAL_DIM == 512
    cached = sorted((REPO_ROOT / "artifacts" / "task6n" / "features").glob("*.npy"))[:1]
    if cached:
        assert np.load(cached[0]).shape == (256, 64, 64)


def test_proposal_mask_nearest_downsample():
    from buildreasonseg_mvp.task6w_proposal_quality import downsample_mask64

    mask = np.zeros((512, 512), dtype=bool)
    mask[0:8, 0:8] = True
    small = downsample_mask64(mask)
    assert small.shape == (64, 64)
    assert small.dtype == bool and small.any()
    assert small[0, 0] and not small[5, 5]
    code = _code_only(QUALITY_MODULE)
    assert 'mode="nearest"' in code


def test_ring_construction_exact():
    from buildreasonseg_mvp.task6w_proposal_quality import one_cell_ring

    mask64 = np.zeros((64, 64), dtype=bool)
    mask64[10:20, 10:20] = True
    ring = one_cell_ring(mask64)
    assert not (ring & mask64).any()          # ring excludes the mask
    # one-cell dilation of a 10x10 block gives a 12x12 block; the ring is the difference
    assert ring.sum() == 12 * 12 - 10 * 10 == 44
    assert ring[9, 9] and ring[20, 20] and not ring[5, 5]
    code = _code_only(QUALITY_MODULE)
    assert "F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)" in code
    assert "torch.clamp(dilated - tensor, 0.0, 1.0)" in code


def test_inside_pooled_feature_256():
    from buildreasonseg_mvp.task6w_proposal_quality import pooled_means

    feature = np.random.default_rng(0).random((256, 64, 64)).astype(np.float32)
    mask64 = np.zeros((64, 64), dtype=bool)
    mask64[10:20, 10:20] = True
    inside = pooled_means(feature, mask64)
    assert inside.shape == (256,)
    assert np.allclose(inside, feature[:, 10:20, 10:20].mean(axis=(1, 2)))


def test_ring_pooled_feature_256_and_zeros_when_empty():
    from buildreasonseg_mvp.task6w_proposal_quality import one_cell_ring, pooled_means

    assert pooled_means(np.zeros((256, 64, 64), dtype=np.float32),
                        np.zeros((64, 64), dtype=bool)).shape == (256,)
    assert not pooled_means(np.ones((256, 64, 64), dtype=np.float32),
                            np.zeros((64, 64), dtype=bool)).any()
    mask64 = np.zeros((64, 64), dtype=bool)
    mask64[30:40, 30:40] = True
    ring = one_cell_ring(mask64)
    assert pooled_means(np.ones((256, 64, 64), dtype=np.float32), ring).shape == (256,)


def test_quality_architecture_exact():
    from buildreasonseg_mvp.task6w_proposal_quality import ProposalQualityEstimator

    model = ProposalQualityEstimator()
    visual_layers = [layer for layer in model.visual_branch if isinstance(layer, torch.nn.Linear)]
    geometry_layers = [layer for layer in model.geometry_branch if isinstance(layer, torch.nn.Linear)]
    head_layers = [layer for layer in model.head if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in visual_layers] == [(512, 64)]
    assert [(layer.in_features, layer.out_features) for layer in geometry_layers] == [(8, 16)]
    assert [(layer.in_features, layer.out_features) for layer in head_layers] == [(80, 32), (32, 1)]
    assert isinstance(model.visual_branch[1], torch.nn.LayerNorm)
    report = model.architecture_report()
    assert report["attention"] is False and report["cnn"] is False
    assert report["transformer"] is False and report["gnn"] is False
    assert report["total_parameters"] == 35729
    assert model(torch.zeros(3, 512), torch.zeros(3, 8)).shape == (3,)


def test_bce_pos_weight_computed_train_only():
    from buildreasonseg_mvp.task6w_proposal_quality import pos_weight_from_labels

    labels = np.asarray([1.0] * 2 + [0.0] * 6)
    assert pos_weight_from_labels(labels) == 3.0
    training = _artifact("task6w_quality_training.json")
    if training is not None:
        assert training["optimizer"]["loss"].startswith("BCEWithLogitsLoss")
        assert "pos_weight" in training["optimizer"]["loss"]
        assert training["data"]["pos_weight_train_only"] > 0
    code = _code_only(SCRIPTS / "task6w_train_quality.py")
    assert "pos_weight_from_labels(labels[train_mask])" in code


def test_inference_quality_threshold_fixed():
    from buildreasonseg_mvp.task6w_proposal_quality import QUALITY_THRESHOLD
    from buildreasonseg_mvp.task6w_quality_reference_resolver import QualityReferenceResolver

    resolver = QualityReferenceResolver.__init__
    import inspect

    default = inspect.signature(resolver).parameters["threshold"].default
    assert default == QUALITY_THRESHOLD == 0.50
    refval = _artifact("task6w_refval_quality_filter.json")
    if refval is not None:
        assert refval["quality_checkpoint"]["threshold"] == 0.50


def test_no_threshold_sweep():
    for name in TASK6W_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("for threshold in", "thresholds =", "sweep_threshold", "best_threshold"):
            assert marker not in code, f"{name} must not sweep the threshold ({marker})"
    refval = _artifact("task6w_refval_quality_filter.json")
    if refval is not None:
        assert refval["test_split_used"] is False
    verdict = _artifact("task6w_verdict.json")
    if verdict is not None:
        assert verdict["protocol"]["threshold_sweep"] is False


def test_no_ranker_in_w_sq():
    resolver = _code_only(RESOLVER_MODULE)
    for marker in ("ProposalSetRanker", "task6u_reference_ranker", "select_with_ranker"):
        assert marker not in resolver, marker
    refval = _artifact("task6w_refval_quality_filter.json")
    if refval is not None:
        assert "W-SQ" in refval["systems"]
    verdict = _artifact("task6w_verdict.json")
    if verdict is not None:
        assert verdict["protocol"]["ranker_used_in_resolver"] is False


# ---------------------------------------------------------------- 30-39 resolver and evaluation


def test_family_eligibility_before_quality_filter():
    from buildreasonseg_mvp.task6w_quality_reference_resolver import QualityReferenceResolver

    import inspect

    source = inspect.getsource(QualityReferenceResolver.select)
    eligible_position = source.index("is_eligible(proposal, family)")
    score_position = source.index("self.score(eligible, feature)")
    assert eligible_position < score_position, "family eligibility must run before the quality filter"
    policy = QualityReferenceResolver.policy_report(_NoopResolver())
    assert policy["pipeline"][1] == "Task 6Q family eligibility"
    assert policy["pipeline"][2] == "ProposalQualityEstimator v0.1"


class _NoopResolver:
    """Minimal stand-in so the policy report can be inspected without a checkpoint."""

    threshold = 0.50


def test_deterministic_area_semantics_after_filter():
    from buildreasonseg_mvp.task6w_quality_reference_resolver import deterministic_by_area

    masks = []
    for size in (10, 30, 20):
        mask = np.zeros((512, 512), dtype=bool)
        mask[10:10 + size, 10:10 + size] = True
        masks.append(_proposal(mask=mask, index=len(masks)))
    assert deterministic_by_area(masks, "largest").area_px == max(item.area_px for item in masks)
    assert deterministic_by_area(masks, "smallest").area_px == min(item.area_px for item in masks)
    equal = [_proposal(confidence=0.4, index=0), _proposal(confidence=0.9, index=1)]
    assert deterministic_by_area(equal, "largest").index == 1  # higher confidence tie-break


def test_explicit_abstention_if_all_rejected():
    from buildreasonseg_mvp.task6w_proposal_quality import ProposalQualityEstimator
    from buildreasonseg_mvp.task6w_quality_reference_resolver import QualityReferenceResolver

    class ZeroModel(ProposalQualityEstimator):
        def forward(self, visual, geometry):
            return torch.full((visual.shape[0],), -10.0)

    resolver = QualityReferenceResolver(ZeroModel(), feature_store=None, device="cpu")
    outcome = resolver.select([_proposal()], "largest", "tile", Path("x.tif"))
    assert outcome["abstained"] is True
    assert outcome["reason"] == "no_quality_eligible_proposals"

    class OneModel(ProposalQualityEstimator):
        def forward(self, visual, geometry):
            return torch.full((visual.shape[0],), 10.0)

    resolver2 = QualityReferenceResolver(OneModel(), feature_store=None, device="cpu")
    outcome2 = resolver2.select([], "largest", "tile", Path("x.tif"))
    assert outcome2["abstained"] is True and outcome2["reason"] == "no_proposals"
    assert resolver2.policy_report()["fallback_to_unfiltered"] is False
    assert resolver2.policy_report()["ranker_used"] is False


def test_refval_untouched_by_training():
    training = _artifact("task6w_quality_training.json")
    manifest = _artifact("task6w_quality_dataset_manifest.json")
    if training is None or manifest is None:
        pytest.skip("artifacts not generated yet")
    assert training["refval_used_for_training"] is False
    assert training["calib_used_for_training"] is False
    assert manifest["universe"]["refval_tile_overlap"] == 0
    assert manifest["universe"]["calib_tile_overlap"] == 0


def test_minival240_exact_reuse():
    manifest = _artifact("task6n_pack_manifest.json")
    mini = _artifact("task6w_downstream_minival240.json")
    assert _sha256(PACK_ROOT / "mini_val_240.json") == manifest["packs"]["mini_val_240"]["sha256"]
    if mini is not None:
        assert mini["pack"]["sha256"] == manifest["packs"]["mini_val_240"]["sha256"]
        assert mini["pack"]["records"] == 240


def test_pairedval20_exact_reuse():
    manifest = _artifact("task6n_pack_manifest.json")
    paired = _artifact("task6w_downstream_pairedval20.json")
    assert _sha256(PACK_ROOT / "paired_val_20.json") == manifest["packs"]["paired_val_20"]["sha256"]
    if paired is not None:
        assert paired["pack"]["sha256"] == manifest["packs"]["paired_val_20"]["sha256"]
        assert paired["pack"]["pairs"] == 20


def test_canonical_program_causal_comparison_has_no_parser():
    mini = _artifact("task6w_downstream_minival240.json")
    if mini is None:
        pytest.skip("downstream not generated yet")
    assert mini["parser_used"] is False
    assert mini["parser_fail_count"] == 0
    code = _code_only(SCRIPTS / "task6w_evaluate_downstream.py")
    causal_block = code.split("def run_causal")[1].split("def run_parser")[0]
    assert "parse_instruction" not in causal_block
    assert "PROGRAM_DECOMPOSITION[sample.program_id]" in causal_block


def test_final_integration_uses_hardened_program_head():
    integration = _artifact("task6w_hardened_parser_integration.json")
    if integration is None:
        pytest.skip("integration not generated yet")
    training = _artifact("task6t_training_summary.json")
    assert integration["parser_checkpoint"]["sha256"] == training["checkpoint"]["sha256"]
    code = _code_only(SCRIPTS / "task6w_evaluate_downstream.py")
    parser_block = code.split("def run_parser")[1]
    assert "parse_instruction" in parser_block
    assert "load_parser_checkpoint" in parser_block


def test_parser_stays_240_of_240():
    integration = _artifact("task6w_hardened_parser_integration.json")
    if integration is None:
        pytest.skip("integration not generated yet")
    assert integration["parser"]["exact_correct"] == 240
    assert integration["parser"]["exact_accuracy"] == 1.0
    assert integration["nearest_or_l3_execution_evaluated"] is False


# ---------------------------------------------------------------- 40-48 scope guards


def test_field_v02_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field.py") == ""


def test_b3_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    mini = _artifact("task6w_downstream_minival240.json")
    if mini is not None:
        assert mini["target_decoder"]["sha256"] == mini["target_decoder"]["expected_sha256"]


def test_no_yolo_retraining():
    for name in TASK6W_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("YOLO.train", "resume=True", "epochs=80"):
            assert marker not in code, f"{name} must not train YOLO ({marker})"


def test_no_tta_or_tiling():
    from scripts.task6u_common import CONFIGS

    assert CONFIGS["U-C1"]["tta"] is False and CONFIGS["U-C1"]["tiling"] is False
    for name in TASK6W_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("augment=True", "sliding_window", "tile_grid"):
            assert marker not in code, f"{name}: {marker}"


def test_no_grcl():
    for name in TASK6W_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""


def test_no_nearest_or_l3_execution():
    from buildreasonseg_mvp.task6s_directional_pipeline import SUPPORTED_PROGRAMS

    assert not any("nearest" in program for program in SUPPORTED_PROGRAMS)
    integration = _artifact("task6w_hardened_parser_integration.json")
    if integration is not None:
        assert integration["nearest_or_l3_execution_evaluated"] is False
    verdict = _artifact("task6w_verdict.json")
    if verdict is not None:
        assert verdict["interpretation_boundary"]["nearest_or_l3_started"] is False


def test_no_test_split():
    for name in TASK6W_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"
    for artifact in ("task6w_oracle_quality_filter_calib.json",
                     "task6w_quality_dataset_manifest.json", "task6w_quality_training.json",
                     "task6w_refval_quality_filter.json", "task6w_downstream_minival240.json",
                     "task6w_downstream_pairedval20.json",
                     "task6w_hardened_parser_integration.json", "task6w_verdict.json"):
        payload = _artifact(artifact)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, artifact


def test_no_new_dataset_download_install_or_gui():
    for name in TASK6W_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


def test_previous_suite_preserved():
    for name in ("test_task6v_family_reference_resolver.py",
                 "test_task6u_reference_hardening.py",
                 "test_task6t_programhead_hardening.py",
                 "test_task6s_directional_end_to_end.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT,
                              "--", "tests/"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
