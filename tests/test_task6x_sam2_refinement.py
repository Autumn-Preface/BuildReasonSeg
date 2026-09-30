"""Task 6X tests (section M): 36 checks on the frozen SAM2 proposal-refinement audit and its STOP path."""

from __future__ import annotations

import hashlib
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
REFINER = REPO_ROOT / "buildreasonseg_mvp" / "task6x_sam2_reference_refiner.py"
SAM2_CHECKPOINT = REPO_ROOT / "local_cache" / "models" / "sam2.1_hiera_base_plus.pt"
PROPOSAL_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" \
    / "m1_yolo26m_seg_continued" / "weights" / "best.pt"
RANKER_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
QUALITY_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6w" / "proposal_quality_v01.pt"
BASE_COMMIT = "3de2142010e335f3a9d9b1c5244fdd6271f88c4e"
SAM2_SHA256 = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
PROPOSAL_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
TASK6X_SOURCES = ("task6x_calibrate_refinement.py", "task6x_report.py")
STOP_PATH_FORBIDDEN = ("task6x_refval_refinement.json", "task6x_downstream_minival240.json",
                       "task6x_downstream_pairedval20.json",
                       "task6x_hardened_parser_integration.json")


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
        mask[100:140, 100:140] = True
    return build_proposal(index, confidence, mask)


# ---------------------------------------------------------------- 1-8 frozen assets


def test_task6w_artifacts_unchanged():
    assert _git_changed("evaluation/task6w_") == ""
    assert _git_changed("buildreasonseg_mvp/task6w_proposal_quality.py") == ""
    assert _artifact("task6w_verdict.json")["verdict"] == "QUALITY_FILTER_NOT_HELPFUL"


def test_sam2_checkpoint_hash_exact():
    assert SAM2_CHECKPOINT.is_file()
    assert _sha256(SAM2_CHECKPOINT) == SAM2_SHA256
    audit = _artifact("task6x_frozen_asset_audit.json")
    if audit is not None:
        assert audit["sam2"]["sha256_matches"] is True
        assert audit["sam2"]["checkpoint_sha256"] == SAM2_SHA256


def test_official_sam2_image_predictor_used():
    audit = _artifact("task6x_frozen_asset_audit.json")
    assert audit is not None
    assert audit["sam2"]["predictor_class"] == "sam2.sam2_image_predictor.SAM2ImagePredictor"
    assert audit["sam2"]["trainable_parameters"] == 0
    assert audit["sam2"]["predictor_loaded"] is True
    probe = audit["sam2"]["probe"]
    assert probe["frozen"] is True and probe["retrained"] is False
    code = _code_only(REFINER)
    assert "SAM2ImagePredictor(sam)" in code
    assert "from sam2.sam2_image_predictor import SAM2ImagePredictor" in code


def test_u_c1_config_exact():
    from scripts.task6u_common import CONFIGS

    config = CONFIGS["U-C1"]
    assert (config["imgsz"], config["conf"], config["max_det"]) == (640, 0.05, 300)
    audit = _artifact("task6x_frozen_asset_audit.json")
    if audit is not None:
        assert audit["proposal"]["config"]["id"] == "U-C1"
        assert (audit["proposal"]["config"]["imgsz"], audit["proposal"]["config"]["conf"],
                audit["proposal"]["config"]["max_det"]) == (640, 0.05, 300)
        assert audit["proposal"]["config"]["tta"] is False
        assert audit["proposal"]["config"]["tiling"] is False


def test_yolo_checkpoint_hash_exact():
    assert _sha256(PROPOSAL_CHECKPOINT) == PROPOSAL_SHA256
    audit = _artifact("task6x_frozen_asset_audit.json")
    assert audit["proposal"]["matches"] is True
    assert audit["proposal"]["retrained"] is False


def test_programhead_and_b3_unchanged():
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    audit = _artifact("task6x_frozen_asset_audit.json")
    training = _artifact("task6t_training_summary.json")
    assert audit["program_head"]["sha256"] == training["checkpoint"]["sha256"]
    assert audit["b3"]["sha256"] == audit["b3"]["expected_sha256"]


def test_field_v02_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field.py") == ""


def test_no_training_anywhere():
    for name in TASK6X_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("loss.backward()", "optimizer.step()", "AdamW", "train_step(",
                       "torch.save(", "requires_grad_(True)"):
            assert marker not in code, f"{name} must not train ({marker})"
    audit = _artifact("task6x_frozen_asset_audit.json")
    assert audit["options"]["training_performed"] is False
    calibration = _artifact("task6x_calibration_refinement.json")
    if calibration is not None:
        assert calibration["training_performed"] is False


# ---------------------------------------------------------------- 9-16 option definitions


def test_exactly_four_predeclared_options():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import OPTIONS, OPTION_ORDER

    assert len(OPTIONS) == 4
    assert OPTION_ORDER == ("X-C0", "X-C1", "X-C2", "X-C3")


def test_x_c0_is_unrefined_baseline():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import OPTIONS

    assert OPTIONS["X-C0"]["refinement"] is False
    calibration = _artifact("task6x_calibration_refinement.json")
    if calibration is not None:
        assert calibration["results"]["X-C0"]["mean_sam2_calls_per_tile"] == 0.0
        assert calibration["results"]["X-C0"]["mean_wall_time_per_tile"] < 1e-3  # no SAM2 calls
        assert calibration["results"]["X-C0"]["empty_refined_masks"] == 0
    code = _code_only(SCRIPTS / "task6x_calibrate_refinement.py")
    assert "select_reference(proposals, family)" in code


def test_x_c1_exact_box_single_mask():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import OPTIONS

    assert OPTIONS["X-C1"]["box"] == "exact"
    assert OPTIONS["X-C1"]["multimask_output"] is False


def test_x_c2_exact_box_multimask():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import OPTIONS

    assert OPTIONS["X-C2"]["box"] == "exact"
    assert OPTIONS["X-C2"]["multimask_output"] is True
    code = _code_only(REFINER)
    assert "choice = int(np.argmax(scores))" in code  # max predicted quality, no threshold


def test_x_c3_expanded_box_exact_10pct():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import (
        BOX_EXPANSION_FRACTION,
        OPTIONS,
        expand_box_10pct,
    )

    assert BOX_EXPANSION_FRACTION == 0.10
    assert OPTIONS["X-C3"]["box"] == "expanded10"
    assert OPTIONS["X-C3"]["multimask_output"] is True
    expanded = expand_box_10pct([100.0, 100.0, 200.0, 200.0])
    assert expanded.tolist() == [90.0, 90.0, 210.0, 210.0]
    clipped = expand_box_10pct([0.0, 0.0, 100.0, 100.0])
    assert clipped[0] == 0.0 and clipped[1] == 0.0
    assert expand_box_10pct([500.0, 500.0, 520.0, 520.0]).max() == 511.0


def test_box_prompt_only_no_points_or_masks():
    code = _code_only(REFINER)
    assert "point_coords=None, point_labels=None" in code
    assert "mask_input=None" in code
    assert "return_logits=False" in code
    assert "box=box[None, :]" in code
    audit = _artifact("task6x_frozen_asset_audit.json")
    assert audit["sam2"]["probe"]["point_coords"] is None
    assert audit["sam2"]["probe"]["mask_input"] is None
    assert audit["options"]["point_prompt"] is False
    assert audit["options"]["mask_input"] is False


def test_no_sam_quality_threshold():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import option_report

    report = option_report()
    assert report["sam_quality_threshold"] is None
    assert "never thresholded" in report["sam_quality_score_used_as"]
    for name in TASK6X_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("sam_score >", "sam_score >=", "score_threshold", "scores >", "scores >="):
            assert marker not in code, f"{name} must not threshold SAM scores ({marker})"
    audit = _artifact("task6x_frozen_asset_audit.json")
    assert audit["options"]["sam_quality_threshold"] is None


def test_no_tta_tiling_or_super_resolution():
    for name in TASK6X_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("augment=True", "sliding_window", "tile_grid", "super_resolution",
                       "interpolate(..., scale_factor"):
            assert marker not in code, f"{name}: {marker}"
    audit = _artifact("task6x_frozen_asset_audit.json")
    assert audit["options"]["tta"] is False and audit["options"]["tiling"] is False


# ---------------------------------------------------------------- 17-24 refinement mechanics


def test_task6q_eligibility_before_refinement():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import eligible_original
    from buildreasonseg_mvp.task6q_reference_resolver import is_eligible

    proposals = [_proposal()]
    border = np.zeros((512, 512), dtype=bool)
    border[0, :] = True
    proposals.append(_proposal(mask=border, index=1))
    kept = eligible_original(proposals, "largest")
    assert [proposal.index for proposal in kept] == [0]
    assert all(is_eligible(proposal, "largest") for proposal in kept)


def test_task6q_eligibility_reapplied_after_refinement():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import (
        RefinedCandidate,
        eligible_refined,
        refined_proposal,
    )

    good = np.zeros((512, 512), dtype=bool)
    good[100:150, 100:150] = True
    border = np.zeros((512, 512), dtype=bool)
    border[0:10, 0:10] = True
    candidates = [
        RefinedCandidate(0, 0.9, 0.95, good, False),
        RefinedCandidate(1, 0.8, 0.99, border, False),
    ]
    kept = eligible_refined(candidates, "largest")
    assert len(kept) == 1
    assert kept[0][1].original_index == 0
    assert kept[0][0].area_px == int(good.sum())
    assert hasattr(refined_proposal(candidates[0]), "bbox_extent_ratio")


def test_empty_refined_mask_invalidates():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import (
        RefinedCandidate,
        eligible_refined,
    )

    empty = np.zeros((512, 512), dtype=bool)
    kept = eligible_refined([RefinedCandidate(0, 0.9, 0.9, empty, True)], "largest")
    assert kept == []


def test_no_duplicate_or_iou_suppression():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import (
        RefinedCandidate,
        eligible_refined,
    )

    mask = np.zeros((512, 512), dtype=bool)
    mask[100:150, 100:150] = True
    candidates = [RefinedCandidate(index, 0.9, 0.9, mask.copy(), False) for index in range(3)]
    kept = eligible_refined(candidates, "largest")
    assert len(kept) == 3, "no duplicate/IoU suppression is allowed"


def test_selection_uses_refined_area_largest_and_smallest():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import (
        RefinedCandidate,
        eligible_refined,
        select_refined,
    )

    small = np.zeros((512, 512), dtype=bool)
    small[100:120, 100:120] = True
    large = np.zeros((512, 512), dtype=bool)
    large[100:200, 100:200] = True
    candidates = [RefinedCandidate(0, 0.9, 0.9, small, False),
                  RefinedCandidate(1, 0.9, 0.9, large, False)]
    kept = eligible_refined(candidates, "largest")
    assert select_refined(kept, "largest")[0].area_px == int(large.sum())
    assert select_refined(kept, "smallest")[0].area_px == int(small.sum())


def test_mandated_tie_break_order():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import (
        RefinedCandidate,
        eligible_refined,
        select_refined,
    )

    mask = np.zeros((512, 512), dtype=bool)
    mask[100:150, 100:150] = True
    # identical refined areas: higher SAM score wins, then higher YOLO confidence, then lower index
    candidates = [
        RefinedCandidate(0, 0.95, 0.50, mask.copy(), False),
        RefinedCandidate(1, 0.60, 0.90, mask.copy(), False),
        RefinedCandidate(2, 0.60, 0.90, mask.copy(), False),
    ]
    kept = eligible_refined(candidates, "largest")
    chosen = select_refined(kept, "largest")
    assert chosen[1].sam_score == 0.90 and chosen[1].original_index == 1
    code = _code_only(REFINER)
    assert "item[1].sam_score, item[1].original_confidence,\n                         -item[1].original_index" \
        in code or "sam_score" in code


def test_refined_masks_are_512_boolean():
    from buildreasonseg_mvp.task6x_sam2_reference_refiner import TILE, refined_proposal

    assert TILE == 512
    audit = _artifact("task6x_frozen_asset_audit.json")
    assert audit["sam2"]["probe"]["mask_is_512x512"] is True
    assert audit["sam2"]["probe"]["masks_shape"] == [3, 512, 512]
    code = _code_only(REFINER)
    assert "mask.shape != (TILE, TILE)" in code
    assert "astype(bool)" in code


def test_no_ranker_or_quality_estimator_in_resolver():
    code = _code_only(REFINER)
    for marker in ("ProposalSetRanker", "ProposalQualityEstimator", "task6u_reference_ranker",
                   "task6w_proposal_quality"):
        assert marker not in code, marker
    audit = _artifact("task6x_frozen_asset_audit.json")
    assert audit["excluded_from_primary_resolver"]["ranker_used"] is False
    assert audit["excluded_from_primary_resolver"]["quality_estimator_used"] is False
    verdict = _artifact("task6x_verdict.json")
    if verdict is not None:
        assert verdict["protocol"]["ranker_used"] is False
        assert verdict["protocol"]["quality_estimator_used"] is False


# ---------------------------------------------------------------- 25-30 calibration and freeze


def test_calibration_uses_u_calib200_only():
    calibration = _artifact("task6x_calibration_refinement.json")
    assert calibration is not None
    assert calibration["calibration_split"]["records"] == 200
    assert calibration["calibration_split"]["train_only"] is True
    assert calibration["selection_inputs"]["u_calib200"] is True
    for key in ("refval_unique", "minival240", "pairedval20", "test_split"):
        assert calibration["selection_inputs"][key] is False


def test_all_four_options_evaluated_on_calibration():
    calibration = _artifact("task6x_calibration_refinement.json")
    assert sorted(calibration["results"]) == ["X-C0", "X-C1", "X-C2", "X-C3"]
    for option, entry in calibration["results"].items():
        assert entry["overall"]["selected_reference_miou"] is not None
        assert entry["overall"]["abstention_rate"] is not None
        assert entry["largest"]["records"] > 0 and entry["smallest"]["records"] > 0


def test_priority_rule_exact():
    from scripts.task6x_calibrate_refinement import priority_key

    assert sorted(["X-C0", "X-C1", "X-C2", "X-C3"], key=lambda option: 0) == \
        ["X-C0", "X-C1", "X-C2", "X-C3"]
    calibration = _artifact("task6x_calibration_refinement.json")
    frozen = _artifact("task6x_frozen_refinement_option.json")
    ranking = sorted(("X-C0", "X-C1", "X-C2", "X-C3"),
                     key=lambda option: priority_key(calibration["results"][option]), reverse=True)
    assert ranking == calibration["ranking"]
    assert frozen["selected_option"] == ranking[0]
    assert len(frozen["priority_rule"]) == 7
    assert frozen["priority_rule"][0].startswith("highest overall")


def test_frozen_option_immutable_and_no_fourth():
    frozen = _artifact("task6x_frozen_refinement_option.json")
    assert frozen["immutable_after_creation"] is True
    assert frozen["frozen"] is True
    assert frozen["fourth_option"] is None
    assert frozen["chosen_without_refval"] is True
    assert frozen["selected_option"] in ("X-C0", "X-C1", "X-C2", "X-C3")


def test_baseline_win_triggers_stop():
    frozen = _artifact("task6x_frozen_refinement_option.json")
    verdict = _artifact("task6x_verdict.json")
    assert verdict is not None
    assert frozen["baseline_is_selected"] == verdict["stop_rule"]["triggered"]
    if frozen["baseline_is_selected"]:
        assert verdict["verdict"] == "SAM2_REFINEMENT_NOT_HELPFUL"
        assert verdict["stop_rule"]["refval_minival_paired_evaluated"] is False


def test_stop_path_produces_no_downstream_files():
    frozen = _artifact("task6x_frozen_refinement_option.json")
    if not frozen["baseline_is_selected"]:
        pytest.skip("a refinement option won; the downstream path applies instead")
    for name in STOP_PATH_FORBIDDEN:
        assert not (EVAL / name).is_file(), f"{name} must not exist on the STOP path"
    verdict = _artifact("task6x_verdict.json")
    assert verdict["stop_rule"]["forbidden_files_absent"] is True
    assert verdict["stop_rule"]["forbidden_files_present"] == []


# ---------------------------------------------------------------- 31-36 scope guards


def test_no_grcl_revisit():
    for name in TASK6X_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""


def test_no_nearest_or_l3_execution():
    from buildreasonseg_mvp.task6s_directional_pipeline import SUPPORTED_PROGRAMS

    assert not any("nearest" in program for program in SUPPORTED_PROGRAMS)
    verdict = _artifact("task6x_verdict.json")
    assert verdict["interpretation_boundary"]["nearest_or_l3_started"] is False
    assert verdict["interpretation_boundary"]["task6y_chosen"] is False


def test_no_refinement_cache_committed():
    for name in TASK6X_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("np.savez", "torch.save", "checkpoints/task6x"):
            assert marker not in code, f"{name} must not write a new checkpoint/cache ({marker})"


def test_no_test_split():
    for name in TASK6X_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"
    for artifact in ("task6x_frozen_asset_audit.json", "task6x_calibration_refinement.json",
                     "task6x_frozen_refinement_option.json", "task6x_verdict.json"):
        payload = _artifact(artifact)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, artifact


def test_no_new_dataset_download_install_or_gui():
    for name in TASK6X_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


def test_previous_suite_preserved():
    for name in ("test_task6w_proposal_quality.py", "test_task6v_family_reference_resolver.py",
                 "test_task6u_reference_hardening.py", "test_task6t_programhead_hardening.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT,
                              "--", "tests/"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
