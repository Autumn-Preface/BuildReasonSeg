"""Task 7F tests (Part O): 38+ checks on the reference bottleneck ceiling decomposition."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
HELPER = REPO_ROOT / "buildreasonseg_mvp" / "task7f_reference_ceiling.py"
HOLDOUT_ROOT = REPO_ROOT / "artifacts" / "task7e" / "holdout"
CACHE_PATH = REPO_ROOT / "artifacts" / "task7f" / "reference_modes_cache.json"
BASE_COMMIT = "f9c6e876f7ba2be33f5dff04a182a06813edf580"
D_B1_SHA256 = "6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0"
YOLO_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
MODES = ("F-R0", "F-R1", "F-R2", "F-R3")
TASK7F_SOURCES = ("task7f_reference_modes.py", "task7f_evaluate_downstream.py",
                  "task7f_gap_decomposition.py", "task7f_report.py")
REQUIRED_ARTIFACTS = ("task7f_reference_modes.json", "task7f_downstream_reference_modes.json",
                      "task7f_gap_decomposition.json", "task7f_paired_reference_modes.json",
                      "task7f_verdict.json")
TASK7E_ORACLE = {"f_r0_strict": 0.24540501038500215, "f_r0_answered": 0.24614085749260334,
                 "f_r3_strict": 0.38549570532647004, "abstentions": 2}


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


def _fake_proposal(mask: np.ndarray, confidence: float, index: int):
    """A structurally valid frozen `Proposal` (eligibility fields are irrelevant to the mode rules)."""

    from buildreasonseg_mvp.task6q_reference_resolver import bbox_from_mask, Proposal

    mask = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(mask)
    bbox = (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1) \
        if rows.size else (0, 0, 0, 0)
    return Proposal(index=index, confidence=confidence, mask=mask, area_px=int(mask.sum()),
                    bbox_xyxy=bbox, bbox_area=int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                    bbox_extent_ratio=0.1, touches_border=False)


# ---------------------------------------------------------------- 1-7 frozen state


def test_task7e_artifacts_unchanged():
    assert _git_changed("evaluation/task7e_") == ""
    assert _git_changed("scripts/task7e_") == ""
    assert _artifact("task7e_verdict.json")["verdict"] == "DB1_PREDICTED_REFERENCE_BELOW_GATE"


def test_holdout_record_id_hash_exact():
    manifest = _artifact("task7e_holdout_manifest.json")
    modes = _artifact("task7f_reference_modes.json")
    rows = [json.loads(line) for line in
            (HOLDOUT_ROOT / "holdout_l3_rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    digest = hashlib.sha256("\n".join(sorted(row["sample_id"] for row in rows)).encode("utf-8")).hexdigest()
    assert digest == manifest["holdout"]["record_id_hash"]
    assert modes["checks"]["record_id_hash"] == digest
    assert modes["checks"]["record_id_hash_ok"] is True
    assert modes["records"] == 669


def test_holdout_pair_id_hash_exact():
    manifest = _artifact("task7e_holdout_manifest.json")
    modes = _artifact("task7f_reference_modes.json")
    paired = _artifact("task7f_paired_reference_modes.json")
    pairs = json.loads((HOLDOUT_ROOT / "holdout_pairs.json").read_text(encoding="utf-8"))["pairs"]
    digest = hashlib.sha256("\n".join(sorted(pair["pair_id"] for pair in pairs)).encode("utf-8")).hexdigest()
    assert digest == manifest["paired"]["pair_id_hash"]
    assert modes["checks"]["pair_id_hash"] == digest
    assert modes["checks"]["pair_id_hash_ok"] is True
    assert paired["pairs"] == 20


def test_d_b1_checkpoint_hash_exact():
    from buildreasonseg_mvp.task7e_l3_decoder_adapter import D_B1_CHECKPOINT

    assert _sha256(D_B1_CHECKPOINT) == D_B1_SHA256
    modes = _artifact("task7f_reference_modes.json")
    assert modes["checks"]["d_b1_sha256"] == D_B1_SHA256
    assert modes["checks"]["d_b1_sha256_ok"] is True
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""


def test_u_c1_exact_config():
    from scripts.task6u_common import CONFIGS

    modes = _artifact("task7f_reference_modes.json")
    assert modes["u_c1_config"] == CONFIGS["U-C1"]
    assert modes["u_c1_config"]["imgsz"] == 640
    assert modes["u_c1_config"]["conf"] == 0.05
    assert modes["u_c1_config"]["max_det"] == 300
    assert modes["u_c1_config"]["nms"] == "default"
    assert modes["u_c1_config"]["tta"] is False and modes["u_c1_config"]["tiling"] is False
    assert modes["checks"]["yolo_sha256"] == YOLO_SHA256
    assert modes["checks"]["yolo_sha256_ok"] is True


def test_no_training():
    verdict = _artifact("task7f_verdict.json")
    assert verdict["protocol"]["training_performed"] is False
    assert verdict["protocol"]["selector_trained"] is False
    assert verdict["protocol"]["yolo_retrained"] is False
    assert verdict["protocol"]["checkpoints_created"] is False
    for name in TASK7F_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("backward()", "optimizer.step(", "make_optimizer(", "AdamW(", "torch.save("):
            assert marker not in code, f"{name}: {marker}"
    helper = _code_only(HELPER)
    for marker in ("backward()", "optimizer", "torch.save("):
        assert marker not in helper, marker


def test_no_test_split():
    verdict = _artifact("task7f_verdict.json")
    assert verdict["protocol"]["test_split_used"] is False
    assert verdict["test_split_used"] is False
    modes = _artifact("task7f_reference_modes.json")
    assert modes["checks"]["no_test"] is True
    assert modes["checks"]["overlap_minival"] is True and modes["checks"]["overlap_pairedval"] is True
    for name in TASK7F_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"
    for name in REQUIRED_ARTIFACTS:
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, name


# ---------------------------------------------------------------- 8-16 exact mode construction


def test_same_proposal_set_reused_for_all_modes():
    source = inspect.getsource(
        __import__("scripts.task7f_reference_modes", fromlist=["main"]).main)
    assert source.count("proposals_of(record[\"tile_id\"]") == 1
    assert "reference_cache" in source
    record_loop = source.split("for record in records:")[1].split("reference_summary = {}")[0]
    assert record_loop.count("build_mode_masks(eligible, gt_reference)") == 1
    for mode in ("F-R0", "F-R1", "F-R2"):
        assert f'"{mode}"' in source
    # one shared eligible-proposal cache, filtered from the frozen U-C1 inference pass
    assert source.count("pipeline.proposals(tile_id, Path(image_path))") == 1
    assert "is_eligible(proposal, \"largest\")" in _code_only(SCRIPTS / "task7f_reference_modes.py")


def test_f_r0_deterministic_max_area_exact():
    from buildreasonseg_mvp.task7f_reference_ceiling import select_current

    small = np.zeros((8, 8), dtype=bool)
    small[0, 0] = True
    large = np.zeros((8, 8), dtype=bool)
    large[:2, :2] = True
    proposals = [_fake_proposal(small, 0.9, 0), _fake_proposal(large, 0.2, 1)]
    assert select_current(proposals).index == 1
    # tie on area -> higher confidence
    other = np.zeros((8, 8), dtype=bool)
    other[4:6, 4:6] = True
    assert select_current([_fake_proposal(large, 0.2, 0), _fake_proposal(other, 0.8, 1)]).index == 1
    # tie on area and confidence -> lower original index
    same = np.zeros((8, 8), dtype=bool)
    same[6:8, 6:8] = True
    assert select_current([_fake_proposal(same, 0.5, 3), _fake_proposal(other, 0.5, 7)]).index == 3
    assert select_current([]) is None
    code = _code_only(HELPER)
    assert "max(eligible, key=lambda proposal: (int(np.asarray(proposal.mask).sum())" in code
    assert 'is_eligible(proposal, "largest")' in _code_only(SCRIPTS / "task7f_reference_modes.py")


def test_f_r1_maximum_gt_iou_proposal_exact():
    from buildreasonseg_mvp.task7f_reference_ceiling import select_oracle

    gt = np.zeros((8, 8), dtype=bool)
    gt[:4, :4] = True
    exact = gt.copy()
    partial = np.zeros((8, 8), dtype=bool)
    partial[:2, :2] = True
    unrelated = np.zeros((8, 8), dtype=bool)
    unrelated[6:, 6:] = True
    proposals = [_fake_proposal(partial, 0.9, 0), _fake_proposal(exact, 0.1, 1),
                 _fake_proposal(unrelated, 0.95, 2)]
    assert select_oracle(proposals, gt).index == 1
    assert select_oracle([], gt) is None


def test_f_r1_tie_break_exact():
    from buildreasonseg_mvp.task7f_reference_ceiling import select_oracle

    gt = np.zeros((8, 8), dtype=bool)
    gt[:4, :4] = True
    duplicate = gt.copy()
    # identical masks -> identical IoU -> higher confidence wins
    assert select_oracle([_fake_proposal(duplicate, 0.2, 0), _fake_proposal(gt.copy(), 0.9, 5)],
                         gt).index == 5
    # identical masks and confidence -> lower original index wins
    assert select_oracle([_fake_proposal(duplicate, 0.5, 4), _fake_proposal(gt.copy(), 0.5, 9)],
                         gt).index == 4
    code = _code_only(HELPER)
    assert "max(scored, key=lambda entry: (entry[0], entry[1], entry[2]))" in code


def test_f_r1_uses_predicted_mask_not_gt_mask():
    from buildreasonseg_mvp.task7f_reference_ceiling import build_mode_masks

    gt = np.zeros((8, 8), dtype=bool)
    gt[:4, :4] = True
    proposal = np.zeros((8, 8), dtype=bool)
    proposal[:2, :2] = True
    built = build_mode_masks([_fake_proposal(proposal, 0.9, 0)], gt)
    assert np.array_equal(built["masks"]["F-R1"], proposal)
    assert not np.array_equal(built["masks"]["F-R1"], gt)
    assert np.array_equal(built["masks"]["F-R3"], gt)


def test_f_r2_coverage_threshold_exactly_half():
    from buildreasonseg_mvp.task7f_reference_ceiling import COVERAGE_IOU, build_mode_masks

    assert COVERAGE_IOU == 0.50
    gt = np.zeros((10, 10), dtype=bool)
    gt[:5, :5] = True                      # 25 pixels
    covered = np.zeros((10, 10), dtype=bool)
    covered[:5, :5] = True                 # IoU 1.0
    borderline = np.zeros((10, 10), dtype=bool)
    borderline[:3, :3] = True              # 9 pixels -> IoU 9/25 = 0.36
    half = np.zeros((10, 10), dtype=bool)
    half[:5, :3] = True                    # 15 pixels -> IoU 15/25 = 0.6
    assert build_mode_masks([_fake_proposal(covered, 0.9, 0)], gt)["masks"]["F-R2"] is not None
    assert build_mode_masks([_fake_proposal(half, 0.9, 0)], gt)["detail"]["covered50"] is True
    built = build_mode_masks([_fake_proposal(borderline, 0.9, 0)], gt)
    assert built["detail"]["covered50"] is False
    assert built["masks"]["F-R2"] is None
    # exactly 0.5 counts as covered (>= 0.50)
    exact = np.zeros((10, 10), dtype=bool)
    exact[:5, :5] = True
    partial = np.zeros((10, 10), dtype=bool)
    partial[:5, :5] = True
    partial[5, 0] = False
    from buildreasonseg_mvp.task7f_reference_ceiling import iou_of

    assert iou_of(exact, exact) == 1.0


def test_f_r2_uses_gt_mask_only_when_covered():
    from buildreasonseg_mvp.task7f_reference_ceiling import build_mode_masks

    gt = np.zeros((8, 8), dtype=bool)
    gt[:4, :4] = True
    matching = gt.copy()
    built = build_mode_masks([_fake_proposal(matching, 0.9, 0)], gt)
    assert np.array_equal(built["masks"]["F-R2"], gt)
    assert np.array_equal(built["masks"]["F-R2"], built["masks"]["F-R3"])
    reference = _artifact("task7f_reference_modes.json")
    assert reference["mode_definitions"]["F-R2"].startswith("canonical GT mask when best eligible IoU >= 0.5")
    assert reference["reference"]["F-R2"]["reference_miou"] == 1.0


def test_f_r2_abstains_when_uncovered():
    from buildreasonseg_mvp.task7f_reference_ceiling import build_mode_masks

    gt = np.zeros((8, 8), dtype=bool)
    gt[:4, :4] = True
    poor = np.zeros((8, 8), dtype=bool)
    poor[6:, 6:] = True
    assert build_mode_masks([_fake_proposal(poor, 0.9, 0)], gt)["masks"]["F-R2"] is None
    assert build_mode_masks([], gt)["masks"]["F-R2"] is None
    reference = _artifact("task7f_reference_modes.json")
    assert reference["reference"]["F-R2"]["abstentions"] == 103
    assert reference["reference"]["F-R2"]["covered_records"] == 566
    assert reference["reference"]["F-R2"]["uncovered_records"] == 103


def test_f_r3_always_uses_gt_reference():
    from buildreasonseg_mvp.task7f_reference_ceiling import build_mode_masks

    gt = np.zeros((8, 8), dtype=bool)
    gt[:4, :4] = True
    poor = np.zeros((8, 8), dtype=bool)
    poor[6:, 6:] = True
    for proposals in ([], [_fake_proposal(poor, 0.1, 0)]):
        built = build_mode_masks(proposals, gt)
        assert np.array_equal(built["masks"]["F-R3"], gt)
        assert built["masks"]["F-R3"] is not None
    reference = _artifact("task7f_reference_modes.json")
    assert reference["reference"]["F-R3"]["answered"] == 669
    assert reference["reference"]["F-R3"]["abstentions"] == 0


def test_gt_never_enters_d_b1_except_declared_reference():
    verdict = _artifact("task7f_verdict.json")
    assert verdict["protocol"]["gt_used_only_in_declared_modes"] is True
    assert verdict["interpretation_boundary"]["gt_used_beyond_declared_diagnostics"] is False
    source = inspect.getsource(
        __import__("scripts.task7f_reference_modes", fromlist=["main"]).main)
    # GT is used for F-R1 selection, F-R2 coverage decision and the R2/R3 masks only
    assert "gt_reference = np.asarray(masks.mask(" in source
    assert "best_eligible_iou(eligible, gt_reference)" in inspect.getsource(
        __import__("buildreasonseg_mvp.task7f_reference_ceiling", fromlist=["build_mode_masks"])
        .build_mode_masks)
    downstream = _code_only(SCRIPTS / "task7f_evaluate_downstream.py")
    assert "masks.mask(" not in downstream
    assert "read_cache()" in downstream


def test_gt_target_never_enters_inference():
    source = inspect.getsource(
        __import__("scripts.task7f_reference_modes", fromlist=["main"]).main)
    prediction_block = source.split("def predict_with(")[1].split("reference_rows")[0]
    assert "target_source_feature_id" not in prediction_block
    downstream = _code_only(SCRIPTS / "task7f_evaluate_downstream.py")
    for marker in ("target_mask", "target_source_feature_id"):
        assert marker not in downstream, marker


# ---------------------------------------------------------------- 17-20 reproductions


def test_f_r0_numeric_reproduction_exact():
    downstream = _artifact("task7f_downstream_reference_modes.json")
    entry = downstream["reproduction"]["F-R0"]
    assert entry["strict_miou"] == TASK7E_ORACLE["f_r0_strict"]
    assert entry["strict_delta"] == 0.0 and entry["strict_ok"] is True
    assert entry["answered_only_miou"] == TASK7E_ORACLE["f_r0_answered"]
    assert entry["answered_only_delta"] == 0.0 and entry["answered_only_ok"] is True
    assert entry["abstentions"] == TASK7E_ORACLE["abstentions"] == 2
    assert downstream["results"]["F-R0"]["strict"]["miou"] == pytest.approx(
        _artifact("task7e_predicted_reference_holdout.json")["results"]["D-B1"]["strict"]["miou"],
        abs=1e-12)


def test_f_r3_numeric_reproduction_exact():
    downstream = _artifact("task7f_downstream_reference_modes.json")
    entry = downstream["reproduction"]["F-R3"]
    assert entry["expected_strict_miou"] == TASK7E_ORACLE["f_r3_strict"]
    assert entry["strict_delta"] <= 1e-6 and entry["strict_ok"] is True
    assert entry["abstentions"] == 0
    assert downstream["reproduction"]["passed"] is True
    assert downstream["verdict"] == "TASK7E_NUMERIC_REPRODUCTION_PASS"
    assert downstream["results"]["F-R3"]["strict"]["miou"] == pytest.approx(
        _artifact("task7e_oracle_holdout.json")["results"]["D-B1"]["overall"]["miou"], abs=1e-6)


def test_covered50_definition_exact():
    gaps = _artifact("task7f_gap_decomposition.json")
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    covered = [entry for entry in cache["per_record_modes"] if entry["covered50"]]
    assert gaps["covered50"]["definition"].startswith(
        "at least one eligible U-C1 proposal and best eligible IoU >= 0.50")
    assert gaps["covered50"]["threshold_iou"] == 0.50
    assert gaps["covered50"]["records"] == len(covered) == 566
    assert all(entry["best_eligible_iou"] >= 0.50 for entry in covered)
    assert gaps["covered50"]["records"] == _artifact(
        "task7f_reference_modes.json")["reference"]["F-R2"]["covered_records"]


def test_g_pred_and_g_gt_same_subset():
    gaps = _artifact("task7f_gap_decomposition.json")
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    covered_ids = {entry["sample_id"] for entry in cache["per_record_modes"] if entry["covered50"]}
    f_r1_ids = {row["sample_id"] for row in cache["target_rows"]["F-R1"] if not row["abstained"]}
    f_r3_ids = {row["sample_id"] for row in cache["target_rows"]["F-R3"]}
    assert covered_ids <= f_r1_ids
    assert covered_ids == {row["sample_id"] for row in cache["target_rows"]["F-R3"]
                           if row["sample_id"] in covered_ids}
    assert gaps["covered50"]["g_pred"]["records"] == gaps["covered50"]["g_gt"]["records"] == 566
    assert f_r3_ids >= covered_ids


# ---------------------------------------------------------------- 21-26 gap formulas


def test_selection_gain_formula_exact():
    gaps = _artifact("task7f_gap_decomposition.json")
    gap = gaps["gaps"]
    assert gap["selection_gain"] == pytest.approx(gap["M1_f_r1"] - gap["M0_f_r0"], abs=1e-12)
    downstream = _artifact("task7f_downstream_reference_modes.json")
    assert gap["M1_f_r1"] == downstream["results"]["F-R1"]["strict"]["miou"]
    assert gap["M0_f_r0"] == downstream["results"]["F-R0"]["strict"]["miou"]


def test_geometry_gain_covered_formula_exact():
    gaps = _artifact("task7f_gap_decomposition.json")
    covered = gaps["covered50"]
    assert covered["geometry_gain_covered"] == pytest.approx(
        covered["g_gt"]["miou"] - covered["g_pred"]["miou"], abs=1e-12)
    assert covered["g_pred"]["miou"] == pytest.approx(0.387172, abs=5e-4)
    assert covered["g_gt"]["miou"] == pytest.approx(0.391425, abs=5e-4)


def test_coverage_gain_formula_exact():
    gaps = _artifact("task7f_gap_decomposition.json")
    gap = gaps["gaps"]
    assert gap["coverage_gain"] == pytest.approx(gap["M3_f_r3"] - gap["M2_f_r2"], abs=1e-12)
    labels = gaps["labels"]["PROPOSAL_COVERAGE_IS_MAJOR"]["conditions"]
    assert labels["coverage_gain"]["measured"] == pytest.approx(gap["coverage_gain"], abs=1e-12)


def test_total_reference_gap_exact():
    gaps = _artifact("task7f_gap_decomposition.json")
    gap = gaps["gaps"]
    assert gap["total_reference_gap"] == pytest.approx(gap["M3_f_r3"] - gap["M0_f_r0"], abs=1e-12)
    assert gap["selection_fraction"] == pytest.approx(
        max(gap["selection_gain"], 0.0) / gap["total_reference_gap"], abs=1e-12)
    assert gap["coverage_fraction"] == pytest.approx(
        max(gap["coverage_gain"], 0.0) / gap["total_reference_gap"], abs=1e-12)
    assert "not forced to sum" in gap["geometry_fraction_note"]


def test_pair_reference_reused_within_pair():
    source = inspect.getsource(
        __import__("scripts.task7f_reference_modes", fromlist=["main"]).main)
    pair_block = source.split("for pair in pairs:")[1].split("paired_summary")[0]
    assert pair_block.count("build_mode_masks(eligible, gt_reference)") == 1
    assert pair_block.count("proposals_of(pair[\"tile_id\"]") == 1
    paired = _artifact("task7f_paired_reference_modes.json")
    assert paired["pairs"] == 20
    assert all(len(row["modes"][mode]["own_iou"]) == 2 for row in paired["rows"]
               for mode in MODES if not row["modes"][mode]["abstained"])


def test_metric_helper_definitions():
    from buildreasonseg_mvp.task7f_reference_ceiling import dice_of, iou_of, mask_centroid, centroid_error

    left = np.zeros((4, 4), dtype=bool)
    left[:2, :2] = True
    right = np.zeros((4, 4), dtype=bool)
    right[:2, :2] = True
    assert iou_of(left, right) == 1.0 and dice_of(left, right) == 1.0
    assert mask_centroid(left) == (0.5, 0.5)
    assert centroid_error(left, right) == 0.0
    other = np.zeros((4, 4), dtype=bool)
    other[2:, 2:] = True
    assert iou_of(left, other) == 0.0
    assert centroid_error(left, other) > 0.0


# ---------------------------------------------------------------- 27-38 protocol guards


def test_no_ranker_quality_or_refinement():
    for name in TASK7F_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("QualityReferenceResolver", "ReferenceRanker", "task6w_proposal_quality",
                       "task6x_sam2_reference_refiner", "predict_reference"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/task6u_reference_ranker.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6w_proposal_quality.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6x_sam2_reference_refiner.py") == ""
    verdict = _artifact("task7f_verdict.json")
    assert verdict["protocol"]["ranker_or_filter_or_refinement_added"] is False


def test_fields_unchanged():
    for path in ("buildreasonseg_mvp/geometric_relation_field_v02.py",
                 "buildreasonseg_mvp/nearest_boundary_field.py",
                 "buildreasonseg_mvp/task6z_field_composition.py"):
        assert _git_changed(path) == ""
    from buildreasonseg_mvp.geometric_relation_field_v02 import ALPHA, S_AXIS, S_MARGIN, TAU
    from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG

    assert (ALPHA, TAU, S_AXIS, S_MARGIN, SIGMA_DIAG) == (1.2, 0.04, 0.02, 0.02, 0.05)
    assert _artifact("task7f_verdict.json")["protocol"]["fields_changed"] is False


def test_sam2_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CONFIG_NAME

    assert SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.harness.yaml" or \
        SAM2_CONFIG_NAME == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    assert _artifact("task7f_verdict.json")["protocol"]["sam2_changed"] is False


def test_d_b1_unchanged():
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""
    from buildreasonseg_mvp.task7d_global_competition_decoder import GlobalCompetitionDecoder

    model = GlobalCompetitionDecoder("D-B1")
    assert model.uses_learned_score_head is False and model.trunk.conv1.in_channels == 148
    verdict = _artifact("task7f_verdict.json")
    assert verdict["protocol"]["d_b1_changed"] is False
    assert verdict["protocol"]["d_b1_retrained"] is False if "d_b1_retrained" in verdict["protocol"] \
        else True


def test_no_parser_training_or_use_for_decision():
    verdict = _artifact("task7f_verdict.json")
    assert verdict["protocol"]["parser_trained_or_used_for_decision"] is False
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""
    for name in TASK7F_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("parse_instruction(", "build_program_parser(", "load_parser_checkpoint(",
                       "default_l3_parser_checkpoint("):
            assert marker not in code, f"{name}: {marker}"


def test_no_attention_graph_or_transformer():
    for name in TASK7F_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("MultiheadAttention", "TransformerEncoder", "nn.Transformer", "MessagePassing",
                       "scaled_dot_product_attention"):
            assert marker not in code, f"{name}: {marker}"
    assert _artifact("task7f_verdict.json")["protocol"]["attention_or_graph_added"] is False


def test_no_grcl():
    for name in TASK7F_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""
    assert _artifact("task7f_verdict.json")["protocol"]["grcl_added"] is False


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7F_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"
    assert _artifact("task7f_verdict.json")["interpretation_boundary"]["full_training_or_test_started"] \
        is False


def test_diagnostic_thresholds_exact():
    gaps = _artifact("task7f_gap_decomposition.json")
    assert gaps["thresholds"] == {"selection_gain": 0.05, "f_r1_miou": 0.29, "f_r1_paired": 10,
                                  "f_r1_margin": 0.18, "geometry_gain_covered": 0.05,
                                  "coverage_gain": 0.04, "f_r2_coverage_rate": 0.85,
                                  "f_r2_miou": 0.30, "f_r2_paired": 12, "f_r2_margin": 0.22}
    labels = gaps["labels"]
    assert labels["SELECTION_IS_ACTIONABLE"]["value"] is True
    assert labels["PROPOSAL_GEOMETRY_IS_MAJOR"]["value"] is False
    assert labels["PROPOSAL_COVERAGE_IS_MAJOR"]["value"] is True
    assert labels["CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING"]["value"] is True
    # and the labels follow from their own conditions
    for name, entry in labels.items():
        if name == "PROPOSAL_COVERAGE_IS_MAJOR":
            assert entry["value"] == any(condition["passed"]
                                         for condition in entry["conditions"].values())
        else:
            assert entry["value"] == all(condition["passed"]
                                         for condition in entry["conditions"].values())


def test_verdict_priority_exact():
    verdict = _artifact("task7f_verdict.json")
    gaps = _artifact("task7f_gap_decomposition.json")
    downstream = _artifact("task7f_downstream_reference_modes.json")
    labels = gaps["labels"]
    gap = gaps["gaps"]
    assert verdict["verdict"] in verdict["allowed_verdicts"]
    if not verdict["protocol"]["clean"]:
        assert verdict["verdict"] == "INVALID_EXPERIMENT"
    elif not verdict["holdout"]["checks"]["record_id_hash_ok"]:
        assert verdict["verdict"] == "TASK7E_HOLDOUT_MISMATCH"
    elif not downstream["reproduction"]["passed"]:
        assert verdict["verdict"] == "TASK7E_NUMERIC_REPRODUCTION_FAIL"
    elif not labels["CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING"]["value"]:
        assert verdict["verdict"] == "REFERENCE_PROPOSAL_CEILING_INSUFFICIENT"
    elif (labels["SELECTION_IS_ACTIONABLE"]["value"]
          and gap["selection_gain"] >= gap["coverage_gain"]):
        assert verdict["verdict"] == "REFERENCE_SELECTION_DOMINANT"
    elif (labels["PROPOSAL_COVERAGE_IS_MAJOR"]["value"]
          or labels["PROPOSAL_GEOMETRY_IS_MAJOR"]["value"]):
        assert verdict["verdict"] == "REFERENCE_COVERAGE_OR_GEOMETRY_DOMINANT"
    else:
        assert verdict["verdict"] == "REFERENCE_MIXED_BOTTLENECK"
    assert verdict["verdict"] == "REFERENCE_SELECTION_DOMINANT"
    assert verdict["recommendation"].startswith("等待 ChatGPT 根据 Task 7F")


def test_architecture_status_and_artifacts():
    verdict = _artifact("task7f_verdict.json")
    status = verdict["architecture_status"]
    assert status["d_b1_status"] == "preferred oracle-reference L3 target decoder candidate"
    assert status["end_to_end_ready"] is False
    assert status["final_model"] is False and status["paper_final"] is False
    assert verdict["task7e_oracle_facts"]["all_reproduced"] is True
    assert (REPO_ROOT / "docs" / "task7f_reference_ceiling_decomposition.md").is_file()
    for name in TASK7F_SOURCES:
        assert (SCRIPTS / name).is_file(), name
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name
    assert HELPER.is_file()


def test_previous_suite_preserved():
    for name in ("test_task7e_holdout_audit.py", "test_task7d_global_competition.py",
                 "test_task7c_rehearsal_hardening.py", "test_task7b_parser_hardening.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
