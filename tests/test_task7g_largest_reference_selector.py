"""Task 7G tests (Part P): 49 checks on the largest-reference set-context selector intervention."""

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
MODULE = REPO_ROOT / "buildreasonseg_mvp" / "task7g_largest_reference_selector.py"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7g" / "largest_set_context_selector_v1.pt"
ROWS_PATH = REPO_ROOT / "artifacts" / "task7g" / "selector_dataset" / "selector_rows.jsonl"
BASE_COMMIT = "c59c6d310849c99b3d8f6b1b2e0116c74dc8e932"
D_B1_SHA256 = "6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0"
YOLO_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
TASK7G_SOURCES = ("task7g_build_selector_dataset.py", "task7g_train_selector.py",
                  "task7g_evaluate_selector.py", "task7g_evaluate_downstream.py",
                  "task7g_report.py")
REQUIRED_ARTIFACTS = ("task7g_training_dataset_manifest.json", "task7g_training.json",
                      "task7g_internal_holdout.json", "task7g_external_reference.json",
                      "task7g_external_downstream.json", "task7g_verdict.json")
INTERNAL_GATE = {"mean_gain": 0.08, "mean_selected_iou": 0.62, "oracle_top1": 0.55,
                 "mean_gap_max": 0.14}
EXTERNAL_GATE = {"1_g_s1_reference_miou": 0.55, "2_reference_gain": 0.08, "3_g_s1_pr_at_0_5": 0.65,
                 "4_mean_gap_max": 0.15, "5_g_s1_strict_miou": 0.30, "6_strict_gain": 0.05,
                 "7_g_s1_answered_miou": 0.30, "8_paired": 12, "9_margin": 0.20,
                 "10_abstention_rate": 0.01}


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _git_changed(prefix: str) -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _proposal(mask: np.ndarray, confidence: float, index: int):
    from buildreasonseg_mvp.task6q_reference_resolver import Proposal

    mask = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(mask)
    bbox = (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1) \
        if rows.size else (0, 0, 0, 0)
    return Proposal(index=index, confidence=confidence, mask=mask, area_px=int(mask.sum()),
                    bbox_xyxy=bbox, bbox_area=int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                    bbox_extent_ratio=0.1, touches_border=False)


# ---------------------------------------------------------------- 1-5 frozen assets


def test_task7f_artifacts_unchanged():
    assert _git_changed("evaluation/task7f_") == ""
    assert _git_changed("buildreasonseg_mvp/task7f_reference_ceiling.py") == ""
    assert _artifact("task7f_verdict.json")["verdict"] == "REFERENCE_SELECTION_DOMINANT"


def test_task7f_erratum_documented_not_mutated():
    doc = (REPO_ROOT / "docs" / "task7g_largest_reference_set_context_selector.md").read_text(
        encoding="utf-8")
    assert "reporting erratum" in doc.lower()
    assert "oracle_selected_confidence" in doc
    assert "does not affect" in doc
    modes = _artifact("task7f_reference_modes.json")
    assert "selected_confidence_mean" in modes["reference"]["F-R1"]
    helper = (REPO_ROOT / "buildreasonseg_mvp" / "task7f_reference_ceiling.py").read_text(
        encoding="utf-8")
    assert "oracle_selected_confidence" in helper and "selected_confidence" in helper


def test_u_c1_exact_config():
    from scripts.task6u_common import CONFIGS

    manifest = _artifact("task7g_training_dataset_manifest.json")
    assert manifest["proposal_generation"]["config"] == "U-C1"
    assert manifest["proposal_generation"]["imgsz"] == 640
    assert manifest["proposal_generation"]["conf"] == 0.05
    assert manifest["proposal_generation"]["max_det"] == 300
    assert manifest["proposal_generation"]["nms"] == "default"
    assert manifest["proposal_generation"]["tta"] is False
    assert manifest["proposal_generation"]["tiling"] is False
    assert CONFIGS["U-C1"] == {key: value for key, value in manifest["proposal_generation"].items()
                               if key in CONFIGS["U-C1"]}


def test_yolo_checkpoint_hash_exact():
    from scripts.task6u_common import PROPOSAL_CHECKPOINT

    manifest = _artifact("task7g_training_dataset_manifest.json")
    assert _sha256(PROPOSAL_CHECKPOINT) == YOLO_SHA256
    assert manifest["proposal_generation"]["checkpoint_sha256"] == YOLO_SHA256
    assert _git_changed("scripts/task6u_common.py") == ""


def test_largest_eligibility_exact():
    from buildreasonseg_mvp.task6q_reference_resolver import is_eligible

    manifest = _artifact("task7g_training_dataset_manifest.json")
    assert manifest["proposal_generation"]["eligibility"] == {
        "non_empty": True, "not_border_touching": True, "bbox_extent_ratio_max": 0.20,
        "area_min_rule": None}
    code = _code_only(SCRIPTS / "task7g_build_selector_dataset.py")
    assert 'is_eligible(proposal, "largest")' in code
    assert "np.asarray(proposal.mask, dtype=bool).any()" in code
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal

    border = np.zeros((512, 512), dtype=bool)
    border[0, :6] = True
    good = np.zeros((512, 512), dtype=bool)
    good[6:8, 6:8] = True
    assert build_proposal(1, 0.9, border).touches_border is True
    assert is_eligible(build_proposal(1, 0.9, border), "largest") is False
    assert is_eligible(build_proposal(2, 0.9, good), "largest") is True


# ---------------------------------------------------------------- 6-12 dataset construction


def test_train_source_is_v02_train_only():
    manifest = _artifact("task7g_training_dataset_manifest.json")
    assert manifest["source"]["split"] == "train"
    assert manifest["source"]["dataset"] == "BuildSpatialReason v0.2"
    code = _code_only(SCRIPTS / "task7g_build_selector_dataset.py")
    assert 'DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"' in code
    assert 'train.jsonl' in code
    for marker in ("val.jsonl", "test.jsonl"):
        assert marker not in code, marker


def test_only_explicit_largest_reference_records_used():
    manifest = _artifact("task7g_training_dataset_manifest.json")
    assert manifest["source"]["largest_reference_rows"] == 4119
    assert manifest["source"]["rows_with_reference_id"] == 3002
    assert manifest["source"]["rows_without_reference_id"] == 1117
    assert manifest["source"]["excluded_smallest_reference_rows"] == 1251
    programs = manifest["source"]["per_program"]
    assert all("largest" in program for program in programs)
    assert not any("smallest" in program for program in programs)
    code = _code_only(SCRIPTS / "task7g_build_selector_dataset.py")
    assert 'if "largest" in str(record["query_type"])' in code
    assert "reference_id_of(record) is not None" in code


def test_dedup_key_exact():
    manifest = _artifact("task7g_training_dataset_manifest.json")
    assert manifest["source"]["unique_references"] == 1063
    assert manifest["source"]["unique_tiles"] == 1063
    assert manifest["source"]["duplicate_rows_removed"] == (
        manifest["source"]["rows_with_reference_id"]
        - manifest["source"]["unique_references"]) == 1939
    code = _code_only(SCRIPTS / "task7g_build_selector_dataset.py")
    assert 'key = (str(record.get("split", "train")), tile_of(record), reference_id_of(record))' in code
    rows = [json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()] if ROWS_PATH.is_file() else []
    if rows:
        keys = {(row["split"], row["tile_id"], row["reference_source_feature_id"]) for row in rows}
        assert len(keys) == len(rows)


def test_no_val_or_test_selector_training():
    manifest = _artifact("task7g_training_dataset_manifest.json")
    assert manifest["test_split_used"] is False
    training = _artifact("task7g_training.json")
    assert training["test_split_used"] is False
    assert training["dataset"]["train_tiles"] + training["dataset"]["holdout_tiles"] <= 1063
    assert training["dataset"]["trainable_train"] == 805
    assert training["dataset"]["trainable_holdout"] == 223
    assert training["dataset"]["tile_assignment"].startswith("SHA256")
    for name in TASK7G_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"val.jsonl"', '"test.jsonl"', 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"


def test_uncovered_and_no_eligible_excluded_and_counted():
    manifest = _artifact("task7g_training_dataset_manifest.json")
    counts = manifest["labels"]["counts"]
    assert counts == {"no_eligible": 2, "untrainable_not_covered": 33, "trainable": 1028}
    assert sum(counts.values()) == 1063
    assert manifest["labels"]["coverage_min_iou"] == 0.50
    code = _code_only(SCRIPTS / "task7g_build_selector_dataset.py")
    assert 'counts["no_eligible"] += 1' in code
    assert 'counts["untrainable_not_covered"] += 1' in code
    internal = _artifact("task7g_internal_holdout.json")
    assert internal["holdout"]["excluded_from_loss"] == {"no_eligible": 2,
                                                         "untrainable_not_covered": 33}
    assert internal["g_i1_learned_selector"]["records"] == 223


def test_label_is_best_eligible_gt_iou_proposal():
    from buildreasonseg_mvp.task7g_largest_reference_selector import oracle_best

    gt = np.zeros((8, 8), dtype=bool)
    gt[:4, :4] = True
    exact = gt.copy()
    partial = np.zeros((8, 8), dtype=bool)
    partial[:2, :2] = True
    proposals = [_proposal(partial, 0.9, 0), _proposal(exact, 0.1, 1)]
    assert oracle_best(proposals, gt) == 1
    code = _code_only(SCRIPTS / "task7g_build_selector_dataset.py")
    assert "target = oracle_best(eligible, gt_reference)" in code
    rows = [json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()] if ROWS_PATH.is_file() else []
    trainable = [row for row in rows if row["state"] == "trainable"]
    if trainable:
        row = trainable[0]
        best = max(row["candidate_ious"])
        assert row["candidate_ious"][row["target_index"]] == pytest.approx(best, abs=1e-12)
        assert row["best_eligible_iou"] == pytest.approx(best, abs=1e-12)


def test_label_tie_break_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import oracle_best

    gt = np.zeros((8, 8), dtype=bool)
    gt[:4, :4] = True
    duplicate = gt.copy()
    # identical IoU -> higher confidence wins
    assert oracle_best([_proposal(duplicate, 0.2, 0), _proposal(gt.copy(), 0.9, 5)], gt) == 1
    # identical IoU and confidence -> lower original index wins
    assert oracle_best([_proposal(duplicate, 0.5, 4), _proposal(gt.copy(), 0.5, 9)], gt) == 0
    code = _code_only(MODULE)
    assert "max(scored)[3]" in code


# ---------------------------------------------------------------- 13-21 features


def test_feature_dim_exactly_18():
    from buildreasonseg_mvp.task7g_largest_reference_selector import (FEATURE_DIM, FEATURE_NAMES,
                                                                     proposal_feature_matrix)

    assert FEATURE_DIM == 18
    assert len(FEATURE_NAMES) == 18
    masks = []
    for position in range(3):
        mask = np.zeros((32, 32), dtype=bool)
        mask[4 * position:4 * position + 6, 2:10] = True
        masks.append(_proposal(mask, 0.5 + 0.1 * position, position))
    features = proposal_feature_matrix(masks)
    assert features.shape == (3, 18)
    rows = [json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()] if ROWS_PATH.is_file() else []
    trainable = [row for row in rows if row["state"] == "trainable"]
    if trainable:
        assert len(trainable[0]["features"][0]) == 18
        assert len(trainable[0]["features"]) == trainable[0]["eligible_count"]


def test_no_gt_derived_input_feature():
    code = _code_only(MODULE)
    block = code.split("def proposal_feature_matrix")[1].split("class SetContextLargestSelector")[0]
    for marker in ("gt_reference", "gt_", "candidate_ious", "target_index", "best_eligible_iou"):
        assert marker not in block, marker
    builder = _code_only(SCRIPTS / "task7g_build_selector_dataset.py")
    assert "features = proposal_feature_matrix(eligible)" in builder
    assert "proposal_feature_matrix(eligible," not in builder


def test_no_centroid_xy():
    code = _code_only(MODULE)
    block = code.split("def proposal_feature_matrix")[1].split("class SetContextLargestSelector")[0]
    for marker in ("centroid", "mean(axis", "argwhere", "np.nonzero"):
        assert marker not in block, marker
    # only width/height/extent statistics enter the vector
    assert "bbox_xyxy" in block


def test_no_relation_direction_program_feature():
    code = _code_only(MODULE)
    block = code.split("def proposal_feature_matrix")[1].split("class SetContextLargestSelector")[0]
    for marker in ("direction", "relation", "program", "left_of", "above"):
        assert marker not in block, marker
    for name in TASK7G_SOURCES:
        source = _code_only(SCRIPTS / name)
        selector_block = source.split("selector_report()")[0]
        if "proposal_feature_matrix" in source:
            assert "record[\"direction\"]" not in source.split("def ")[0]


def test_no_target_feature():
    code = _code_only(MODULE)
    for marker in ("target_source_feature_id", "target_mask", "target_component"):
        assert marker not in code, marker
    builder = _code_only(SCRIPTS / "task7g_build_selector_dataset.py")
    assert "target_source_feature_id" not in builder


def test_no_rgb_or_sam2_feature():
    block = _code_only(MODULE).split("def selector_report")[0]
    for marker in ("sam2", "SAM2", "FrozenFeatureStore", "image_path", "pixel", "rgb", "RGB",
                   "load_frozen_sam2_encoder"):
        assert marker not in block, marker
    from buildreasonseg_mvp.task7g_largest_reference_selector import selector_report

    assert selector_report()["uses_sam2_or_rgb"] is False
    # the trainer never loads visual features either
    trainer = _code_only(SCRIPTS / "task7g_train_selector.py")
    for marker in ("sam2", "FrozenFeatureStore", "load_frozen_sam2_encoder", "image_path"):
        assert marker not in trainer, marker


def test_boundary_feature_formula_exact():
    from scipy import ndimage

    from buildreasonseg_mvp.task7g_largest_reference_selector import proposal_feature_matrix

    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    features = proposal_feature_matrix([_proposal(mask, 0.5, 0)])
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=1,
                                   border_value=0)
    boundary_px = float((mask & ~eroded).sum())
    expected = boundary_px / np.sqrt(max(1.0, float(mask.sum())))
    assert features[0][10] == pytest.approx(expected, abs=1e-6)
    code = _code_only(MODULE)
    assert "ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool)" in code
    assert "iterations=1, border_value=0" in code


def test_overlap_feature_formulas_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import proposal_feature_matrix

    left = np.zeros((16, 16), dtype=bool)
    left[4:12, 4:12] = True          # 64 px
    right = np.zeros((16, 16), dtype=bool)
    right[4:12, 8:14] = True         # 48 px, intersection 32
    features = proposal_feature_matrix([_proposal(left, 0.9, 0), _proposal(right, 0.5, 1)])
    intersection = float((left & right).sum())
    union = float((left | right).sum())
    assert features[0][11] == pytest.approx(intersection / union, abs=1e-6)
    assert features[0][12] == pytest.approx(intersection / union, abs=1e-6)
    assert features[0][13] == pytest.approx(intersection / float(left.sum()), abs=1e-6)
    assert features[0][14] == pytest.approx(intersection / float(right.sum()), abs=1e-6)
    assert float(features[0][15]) == pytest.approx(
        float(intersection / float(left.sum()) >= 0.50), abs=1e-6)
    assert float(features[0][16]) == pytest.approx(
        float(intersection / float(right.sum()) >= 0.50), abs=1e-6)
    single = proposal_feature_matrix([_proposal(left, 0.9, 0)])
    assert single[0][11] == 0.0 and single[0][12] == 0.0
    assert single[0][13] == 0.0 and single[0][14] == 0.0
    assert single[0][15] == 0.0 and single[0][16] == 0.0


def test_proposal_count_feature_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import proposal_feature_matrix

    proposals = []
    for position in range(4):
        mask = np.zeros((32, 32), dtype=bool)
        mask[position:position + 3, position:position + 3] = True
        proposals.append(_proposal(mask, 0.5, position))
    features = proposal_feature_matrix(proposals)
    assert all(row[17] == pytest.approx(4 / 300.0) for row in features)
    code = _code_only(MODULE)
    assert "min(count, 300) / 300.0" in code


def test_geometry_and_rank_features_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import (SOURCE_AREA,
                                                                     proposal_feature_matrix)

    small = np.zeros((16, 16), dtype=bool)
    small[0:2, 0:2] = True           # 4 px
    large = np.zeros((16, 16), dtype=bool)
    large[0:4, 0:4] = True           # 16 px
    features = proposal_feature_matrix([_proposal(small, 0.9, 0), _proposal(large, 0.5, 1)])
    assert features[1][0] == pytest.approx(np.log1p(16) / np.log1p(SOURCE_AREA), abs=1e-6)
    assert features[1][1] == pytest.approx(16 / SOURCE_AREA, abs=1e-9)
    assert features[1][2] == pytest.approx(1.0, abs=1e-6)
    assert features[0][2] == pytest.approx(4 / 16, abs=1e-6)
    assert features[1][3] == pytest.approx(0.0, abs=1e-6)
    assert features[0][3] == pytest.approx(1.0, abs=1e-6)
    assert features[0][4] == pytest.approx(0.9, abs=1e-6)
    assert features[1][6] == pytest.approx(4 / 512.0, abs=1e-6)
    assert features[1][8] == pytest.approx(1.0, abs=1e-6)
    assert features[1][9] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- 22-27 architecture and loss


def test_shared_proposal_encoder_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import SetContextLargestSelector

    model = SetContextLargestSelector()
    layers = list(model.proposal_encoder)
    assert isinstance(layers[0], torch.nn.Linear) and layers[0].in_features == 18
    assert layers[0].out_features == 64
    assert isinstance(layers[1], torch.nn.LayerNorm) and layers[1].normalized_shape == (64,)
    assert isinstance(layers[2], torch.nn.GELU)
    assert isinstance(layers[3], torch.nn.Linear) and layers[3].in_features == 64
    assert layers[3].out_features == 32
    assert isinstance(layers[4], torch.nn.GELU)
    features = torch.randn(5, 18)
    hidden = model.encode(features)
    assert hidden.shape == (5, 32)
    assert torch.allclose(model.encode(features)[0], model.encode(features[:1])[0], atol=1e-6)


def test_set_mean_pool_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import SetContextLargestSelector

    model = SetContextLargestSelector()
    features = torch.randn(6, 18)
    hidden = model.encode(features)
    expected = hidden.mean(dim=0)
    captured = {}
    original = model.score_head[0]

    class Spy(torch.nn.Module):
        def forward(self, value):
            captured["z"] = value
            return original(value)

    model.score_head[0] = Spy()
    model(features)
    z = captured["z"]
    assert torch.allclose(z[:, 32:64], expected.unsqueeze(0).expand(z.shape[0], -1), atol=1e-6)


def test_set_max_pool_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import SetContextLargestSelector

    model = SetContextLargestSelector()
    features = torch.randn(4, 18)
    hidden = model.encode(features)
    expected = hidden.max(dim=0).values
    captured = {}
    original = model.score_head[0]

    class Spy(torch.nn.Module):
        def forward(self, value):
            captured["z"] = value
            return original(value)

    model.score_head[0] = Spy()
    model(features)
    z = captured["z"]
    assert torch.allclose(z[:, 64:96], expected.unsqueeze(0).expand(z.shape[0], -1), atol=1e-6)


def test_candidate_head_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import SetContextLargestSelector

    model = SetContextLargestSelector()
    layers = list(model.score_head)
    assert isinstance(layers[0], torch.nn.Linear) and layers[0].in_features == 96
    assert layers[0].out_features == 32
    assert isinstance(layers[1], torch.nn.GELU)
    assert isinstance(layers[2], torch.nn.Linear) and layers[2].in_features == 32
    assert layers[2].out_features == 1
    assert model.parameter_count() == 6561
    assert model(torch.randn(7, 18)).shape == (7,)
    # permutation invariance of the set context: shuffling rows shuffles scores identically
    features = torch.randn(5, 18)
    order = torch.as_tensor([2, 0, 4, 1, 3])
    assert torch.allclose(model(features)[order], model(features[order]), atol=1e-5)


def test_no_attention_transformer_gnn():
    code = _code_only(MODULE)
    for marker in ("MultiheadAttention", "TransformerEncoder", "nn.Transformer", "MessagePassing",
                   "scaled_dot_product_attention", "nn.MultiheadAttention"):
        assert marker not in code, marker
    from buildreasonseg_mvp.task7g_largest_reference_selector import selector_report

    report = selector_report()
    assert report["transformer"] is False and report["attention"] is False and report["gnn"] is False
    assert report["positional_encoding"] is False and report["proposal_order_feature"] is False
    assert _artifact("task7g_verdict.json")["protocol"]["attention_or_transformer_or_gnn"] is False


def test_listwise_ce_only():
    code = _code_only(SCRIPTS / "task7g_train_selector.py")
    assert "torch.nn.functional.cross_entropy(scores.unsqueeze(0)" in code
    for marker in ("mse_loss", "nll_loss", "margin", "focal", "bce", "ranking_loss",
                   "pairwise", "quality_loss"):
        assert marker not in code, marker
    from buildreasonseg_mvp.task7g_largest_reference_selector import selector_report

    report = selector_report()
    assert report["loss"] == "CrossEntropyLoss (listwise, single loss)"
    assert report["extra_losses"] == []


# ---------------------------------------------------------------- 28-31 split, selection, gate


def test_tile_level_80_20_internal_split():
    training = _artifact("task7g_training.json")
    internal = _artifact("task7g_internal_holdout.json")
    assert training["dataset"]["train_tiles"] == 805
    assert training["dataset"]["holdout_tiles"] == 223
    assert internal["holdout"]["tiles"] == 223
    assert training["split"]["tile_assignment"] if "split" in training else True
    assert "SHA256(seed:tile_id) < 0.80 -> train" in training["dataset"]["tile_assignment"]
    code = _code_only(SCRIPTS / "task7g_train_selector.py")
    assert 'hashlib.sha256(f"{SEED}:{tile_id}".encode("utf-8"))' in code
    assert "TRAIN_TILE_FRACTION = 0.80" in code
    assert "int(digest[:8], 16) / 0xFFFFFFFF < TRAIN_TILE_FRACTION" in code


def test_train_holdout_tile_overlap_zero():
    internal = _artifact("task7g_internal_holdout.json")
    assert internal["holdout"]["tile_overlap"] == 0
    assert internal["holdout"]["reference_overlap"] == 0
    assert internal["holdout"]["tiles"] >= 50
    rows = [json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()] if ROWS_PATH.is_file() else []
    if rows:
        from scripts.task7g_train_selector import tile_assignment

        partitions = {"train": set(), "holdout": set()}
        for row in rows:
            if row["state"] == "trainable":
                partitions[tile_assignment(row["tile_id"])].add(row["tile_id"])
        assert not (partitions["train"] & partitions["holdout"])
        assert len(partitions["holdout"]) == 223


def test_internal_checkpoint_selection_uses_no_external_holdout():
    internal = _artifact("task7g_internal_holdout.json")
    assert internal["external_holdout_used"] is False
    training = _artifact("task7g_training.json")
    assert training["protocol"]["sweep"] is False
    code = _code_only(SCRIPTS / "task7g_train_selector.py")
    for marker in ("task7e_holdout", "holdout_l3_rows", "task7g_external"):
        assert marker not in code, marker
    assert "evaluate_partition(model, holdout_rows)" in code
    assert "patience" in code.lower()
    verdict = _artifact("task7g_verdict.json")
    assert verdict["protocol"]["external_holdout_used_for_selection"] is False
    assert verdict["protocol"]["selector_training_scope"] == "train split only"


def test_internal_gate_exact():
    internal = _artifact("task7g_internal_holdout.json")
    assert internal["gate_constants"] == INTERNAL_GATE
    gate = internal["gate"]
    assert gate["1_mean_gain"]["required"] == INTERNAL_GATE["mean_gain"]
    assert gate["2_mean_selected_iou"]["required"] == INTERNAL_GATE["mean_selected_iou"]
    assert gate["3_oracle_top1"]["required"] == INTERNAL_GATE["oracle_top1"]
    assert gate["4_mean_gap"]["required"] == f"<= {INTERNAL_GATE['mean_gap_max']}"
    measured_gain = (internal["g_i1_learned_selector"]["mean_selected_iou"]
                     - internal["g_i0_deterministic_max_area"]["mean_selected_iou"])
    assert gate["1_mean_gain"]["measured"] == pytest.approx(measured_gain, abs=1e-12)
    assert gate["1_mean_gain"]["passed"] == (measured_gain >= INTERNAL_GATE["mean_gain"])
    assert gate["2_mean_selected_iou"]["passed"] == (
        internal["g_i1_learned_selector"]["mean_selected_iou"] >= INTERNAL_GATE["mean_selected_iou"])
    assert gate["3_oracle_top1"]["passed"] == (
        internal["g_i1_learned_selector"]["oracle_best_top1"] >= INTERNAL_GATE["oracle_top1"])
    assert gate["4_mean_gap"]["passed"] == (
        internal["g_i1_learned_selector"]["mean_best_minus_selected"]
        <= INTERNAL_GATE["mean_gap_max"])
    assert internal["gate_passed"] == all(entry["passed"] for entry in gate.values()) is False
    assert internal["gate_passed"] is False


def test_external_stages_guarded_by_the_stop_gate():
    internal = _artifact("task7g_internal_holdout.json")
    assert internal["gate_passed"] is False
    reference = _artifact("task7g_external_reference.json")
    downstream = _artifact("task7g_external_downstream.json")
    assert reference["executed"] is False
    assert downstream["executed"] is False
    assert "LARGEST_SELECTOR_NOT_LEARNABLE" in reference["reason"]
    assert "LARGEST_SELECTOR_NOT_LEARNABLE" in downstream["reason"]
    assert not (EVAL / "task7g_external_paired.json").exists()
    for name in ("task7g_evaluate_selector.py", "task7g_evaluate_downstream.py"):
        code = _code_only(SCRIPTS / name)
        assert 'if not passed:' in code or "if not internal or not internal.get(\"gate_passed\")" in code
        assert "return 4" in code


# ---------------------------------------------------------------- 32-40 assets and reproductions


def test_e_holdout_hashes_exact():
    manifest = _artifact("task7e_holdout_manifest.json")
    modes = _artifact("task7f_reference_modes.json")
    assert modes["checks"]["record_id_hash"] == manifest["holdout"]["record_id_hash"]
    assert modes["checks"]["pair_id_hash"] == manifest["paired"]["pair_id_hash"]
    assert modes["checks"]["record_id_hash_ok"] is True
    assert modes["checks"]["pair_id_hash_ok"] is True
    assert modes["records"] == 669
    assert manifest["paired"]["pairs"] == 20


def test_g_s0_deterministic_baseline_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import deterministic_max_area

    small = np.zeros((8, 8), dtype=bool)
    small[0, 0] = True
    large = np.zeros((8, 8), dtype=bool)
    large[:2, :2] = True
    assert deterministic_max_area([_proposal(small, 0.9, 0), _proposal(large, 0.1, 1)]) == 1
    other = np.zeros((8, 8), dtype=bool)
    other[4:6, 4:6] = True
    assert deterministic_max_area([_proposal(large, 0.1, 0), _proposal(other, 0.9, 1)]) == 1
    same = np.zeros((8, 8), dtype=bool)
    same[6:8, 6:8] = True
    assert deterministic_max_area([_proposal(same, 0.5, 3), _proposal(other, 0.5, 7)]) == 0
    code = _code_only(MODULE)
    assert "def deterministic_max_area" in code
    external = _artifact("task7g_external_reference.json")
    assert external["executed"] is False


def test_g_s1_uses_no_gt_at_inference():
    from buildreasonseg_mvp.task7g_largest_reference_selector import (select_with_scores,
                                                                     selector_report)

    masks = []
    for position in range(3):
        mask = np.zeros((16, 16), dtype=bool)
        mask[2 * position:2 * position + 4, 2:8] = True
        masks.append(_proposal(mask, 0.4 + 0.1 * position, position))
    scores = np.asarray([0.1, 0.9, 0.2])
    assert select_with_scores(scores, masks) == 1
    # a genuine float tie falls back to confidence, then to the original index
    tied = np.asarray([0.5, 0.5, 0.5])
    assert select_with_scores(tied, masks) == 2
    equal = np.asarray([0.5, 0.5])
    duplicate_mask = np.asarray(masks[0].mask, dtype=bool).copy()
    pair = [_proposal(duplicate_mask, 0.5, 7), _proposal(duplicate_mask.copy(), 0.5, 2)]
    assert select_with_scores(equal, pair) == 1
    report = selector_report()
    assert report["uses_gt_at_inference"] is False
    code = _code_only(MODULE)
    block = code.split("def select_with_scores")[1].split("def deterministic_max_area")[0]
    for marker in ("gt", "iou", "target"):
        assert marker not in block, marker


def test_g_s1_same_proposal_set_as_g_s0():
    code = _code_only(SCRIPTS / "task7g_evaluate_selector.py")
    assert code.count("eligible_of(record[\"tile_id\"]") == 1
    block = code.split("for record in records:")[1].split("paired_rows = []")[0]
    assert block.count("proposal_feature_matrix(eligible)") == 1
    assert "selected = {\"G-S0\": deterministic_max_area(eligible)," in block
    assert "\"G-S1\": select_with_scores(scores, eligible)}" in block
    downstream = _code_only(SCRIPTS / "task7g_evaluate_downstream.py")
    assert downstream.count("eligible_of(record[\"tile_id\"]") == 1
    assert "proposal_feature_matrix(eligible)" in downstream


def test_d_b1_checkpoint_unchanged():
    from buildreasonseg_mvp.task7e_l3_decoder_adapter import D_B1_CHECKPOINT

    assert _sha256(D_B1_CHECKPOINT) == D_B1_SHA256
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""
    verdict = _artifact("task7g_verdict.json")
    assert verdict["protocol"]["d_b1_trained"] is False
    assert verdict["protocol"]["d_b1_checked"] is True


def test_fields_unchanged():
    for path in ("buildreasonseg_mvp/geometric_relation_field_v02.py",
                 "buildreasonseg_mvp/nearest_boundary_field.py",
                 "buildreasonseg_mvp/task6z_field_composition.py"):
        assert _git_changed(path) == ""
    from buildreasonseg_mvp.geometric_relation_field_v02 import ALPHA, S_AXIS, S_MARGIN, TAU
    from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG

    assert (ALPHA, TAU, S_AXIS, S_MARGIN, SIGMA_DIAG) == (1.2, 0.04, 0.02, 0.02, 0.05)
    assert _artifact("task7g_verdict.json")["protocol"]["fields_changed"] is False


def test_sam2_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CONFIG_NAME

    assert SAM2_CONFIG_NAME.endswith("sam2.1_hiera_b+.yaml")
    assert _artifact("task7g_verdict.json")["protocol"]["sam2_changed"] is False


def test_g_s0_downstream_reproduction_guarded():
    downstream = _artifact("task7g_external_downstream.json")
    assert downstream["executed"] is False
    assert downstream.get("reproduction_passed") is None
    task7f = _artifact("task7f_downstream_reference_modes.json")
    assert task7f["results"]["F-R0"]["strict"]["miou"] == 0.24540501038500215
    code = _code_only(SCRIPTS / "task7g_evaluate_downstream.py")
    assert "EXPECTED = {\"strict_miou\": 0.24540501038500215" in code
    assert "0.24614085749260334" in code


def test_paired_reference_reused_within_pair():
    code = _code_only(SCRIPTS / "task7g_evaluate_selector.py")
    pair_block = code.split("for pair in pairs:")[1].split("paired_summary = {}")[0]
    assert pair_block.count("eligible_of(pair[\"tile_id\"]") == 1
    assert pair_block.count("proposal_feature_matrix(eligible)") == 1
    assert "select_with_scores(scores, eligible)" in pair_block


# ---------------------------------------------------------------- 41-49 guards and continuation


def test_external_gate_exact():
    verdict = _artifact("task7g_verdict.json")
    assert verdict["external"]["gate_constants"] == EXTERNAL_GATE
    assert set(verdict["external"]["gate_constants"]) == set(EXTERNAL_GATE)
    assert verdict["external"]["executed"] is False
    assert verdict["external"]["gate_passed"] is False
    assert verdict["external"]["not_executed_reason"].startswith("section 23 internal gate failed")
    assert verdict["ceiling_capture"]["selection_gain_ceiling"] == pytest.approx(
        0.36490938928549665 - 0.24540501038500215, abs=1e-12)
    assert verdict["ceiling_capture"]["learned_selection_gain"] is None


def test_no_proposal_set_ranker_or_quality_estimator():
    for name in TASK7G_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("proposal_set_ranker", "ProposalSetRanker", "ProposalQualityEstimator",
                       "task6w_proposal_quality", "task6u_reference_ranker",
                       "QualityReferenceResolver", "predict_reference"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task7g_verdict.json")
    assert verdict["protocol"]["ranker_or_quality_estimator_used"] is False
    module = _code_only(MODULE)
    for marker in ("ProposalSetRanker", "ProposalQualityEstimator"):
        assert marker not in module, marker


def test_no_mask_refinement():
    for name in TASK7G_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("sam2_refiner", "refine_mask", "task6x_sam2_reference_refiner",
                       "MaskRefiner", "dilate(", "erode("):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task7g_verdict.json")
    assert verdict["protocol"]["mask_refinement_used"] is False
    assert _git_changed("buildreasonseg_mvp/task6x_sam2_reference_refiner.py") == ""


def test_no_yolo_retraining():
    for name in TASK7G_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("YOLO.train", "yolo.train", "weights/best.pt", "epochs=100",
                       "model.train(", "detector.train("):
            if marker == "model.train(" and name == "task7g_train_selector.py":
                continue  # the selector's own nn.Module train mode
            assert marker not in code, f"{name}: {marker}"
    builder = _code_only(SCRIPTS / "task7g_build_selector_dataset.py")
    assert "proposals_for_tile(model, config" in builder
    assert "from ultralytics import YOLO" in builder  # inference-only construction of the frozen detector
    assert _git_changed("scripts/task6u_common.py") == ""
    verdict = _artifact("task7g_verdict.json")
    assert verdict["protocol"]["yolo_retrained"] is False
    assert verdict["decision"]["yolo_retraining_authorized"] is False


def test_no_d_b1_training():
    code = _code_only(SCRIPTS / "task7g_train_selector.py")
    assert "GlobalCompetitionDecoder" not in code
    for name in TASK7G_SOURCES:
        source = _code_only(SCRIPTS / name)
        if name != "task7g_train_selector.py":
            assert "L3TargetDecoder(" not in source
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""
    verdict = _artifact("task7g_verdict.json")
    assert verdict["protocol"]["d_b1_trained"] is False


def test_no_test_split():
    verdict = _artifact("task7g_verdict.json")
    assert verdict["protocol"]["test_split_used"] is False
    assert verdict["test_split_used"] is False
    for name in REQUIRED_ARTIFACTS:
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, name
    for name in TASK7G_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7G_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task7g_verdict.json")
    assert verdict["interpretation_boundary"]["formal_full_data_training_started"] is False


def test_verdict_priority_and_decision():
    verdict = _artifact("task7g_verdict.json")
    manifest = _artifact("task7g_training_dataset_manifest.json")
    internal = _artifact("task7g_internal_holdout.json")
    assert verdict["verdict"] in verdict["allowed_verdicts"]
    if not verdict["protocol"]["clean"]:
        assert verdict["verdict"] == "INVALID_EXPERIMENT"
    elif manifest["verdict"] == "SELECTOR_DEPENDENCY_UNAVAILABLE":
        assert verdict["verdict"] == "SELECTOR_DEPENDENCY_UNAVAILABLE"
    elif manifest["verdict"] != "SELECTOR_TRAIN_DATA_READY":
        assert verdict["verdict"] == "SELECTOR_TRAIN_DATA_INSUFFICIENT"
    elif not internal["gate_passed"]:
        assert verdict["verdict"] == "LARGEST_SELECTOR_NOT_LEARNABLE"
    else:
        assert verdict["verdict"] in ("TASK7F_BASELINE_REPRODUCTION_FAIL",
                                      "LARGEST_SELECTOR_SCENE_DISJOINT_FAIL",
                                      "LARGEST_SELECTOR_DEVELOPMENT_READY")
    assert verdict["verdict"] == "LARGEST_SELECTOR_NOT_LEARNABLE"
    decision = verdict["decision"]
    assert decision["TASK7G_SELECTOR_ADOPTED"] is False
    assert decision["reference_selection_status"] == "unresolved limitation for this project version"
    assert decision["another_selector_authorized"] is False
    assert decision["reference_intervention_stopped"] is True
    assert decision["z_b3_role"] == "frozen decoder baseline/ablation"
    assert decision["deterministic_selector_role"] == "frozen reference baseline/ablation"
    assert verdict["recommendation"].startswith("等待 ChatGPT 根据 Task 7G")
    assert verdict["dataset"]["source"]["split"] == "train"
    assert verdict["internal_holdout"]["gate_passed"] is False


def test_required_artifacts_and_modules_exist():
    assert (REPO_ROOT / "docs" / "task7g_largest_reference_set_context_selector.md").is_file()
    assert MODULE.is_file()
    for name in TASK7G_SOURCES:
        assert (SCRIPTS / name).is_file(), name
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name
    assert CHECKPOINT.is_file()
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["architecture"] == "SetContextLargestSelector"
    assert payload["version"] == "v1"


def test_previous_suite_preserved():
    for name in ("test_task7f_reference_ceiling.py", "test_task7e_holdout_audit.py",
                 "test_task7d_global_competition.py", "test_task7c_rehearsal_hardening.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
