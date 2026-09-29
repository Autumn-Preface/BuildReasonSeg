"""Task 6Q tests (section 15): 38 checks on the frozen-proposal reference resolver and its propagation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
BASE_COMMIT = "b80f3cc04d237a12dbeef537280a371d217d52e3"
TASK6Q_SOURCES = ("task6q_reference_audit.py", "task6q_target_propagation.py", "task6q_report.py")
TASK6Q_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "task6q_reference_resolver.py"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" / "m1_yolo26m_seg_continued" / "weights" / "best.pt"


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_only(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            end = stripped.find(quote, len(quote))
            if end != -1:
                stripped = stripped[end + len(quote):]
            break
    return "\n".join(line.split("#", 1)[0] for line in stripped.splitlines())


def _git_changed(prefix: str) -> str:
    return subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _mask_with_bbox(y0, y1, x0, x1) -> np.ndarray:
    mask = np.zeros((512, 512), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


# ---------------------------------------------------------------- 1-7 frozen configuration


def test_task6p_artifacts_unchanged():
    changed = _git_changed("evaluation/task6p_")
    assert changed == "", f"Task 6P artifacts changed: {changed}"
    assert _artifact("task6p_verdict.json")["verdict"] == "REFERENCE_HEAD_INSUFFICIENT"


def test_task6m1_inference_config_unchanged():
    changed = _git_changed("evaluation/task6m1_inference_config_frozen.json")
    assert changed == ""
    frozen = _artifact("task6m1_inference_config_frozen.json")
    from buildreasonseg_mvp.task6q_reference_resolver import (
        PROPOSAL_CONF, PROPOSAL_IMGSZ, PROPOSAL_MAX_DET,
    )

    assert frozen["conf"] == PROPOSAL_CONF
    assert frozen["max_det"] == PROPOSAL_MAX_DET
    assert frozen.get("imgsz", PROPOSAL_IMGSZ) == PROPOSAL_IMGSZ


def test_proposal_checkpoint_sha_exact():
    from buildreasonseg_mvp.task6q_reference_resolver import (
        PROPOSAL_CHECKPOINT, PROPOSAL_CHECKPOINT_SHA256,
    )

    assert str(PROPOSAL_CHECKPOINT).replace("\\", "/").endswith(
        "artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt"
    )
    assert PROPOSAL_CHECKPOINT_SHA256 == (
        "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
    )
    if CHECKPOINT.is_file():
        assert _sha256(CHECKPOINT) == PROPOSAL_CHECKPOINT_SHA256
    audit = _artifact("task6q_reference_resolver_val.json")
    if audit is not None:
        assert audit["checkpoint"]["matches"] is True
        assert audit["checkpoint"]["actual_sha256"] == PROPOSAL_CHECKPOINT_SHA256


def test_conf_exactly_0_10():
    from buildreasonseg_mvp.task6q_reference_resolver import PROPOSAL_CONF, config_report

    assert PROPOSAL_CONF == 0.10
    assert config_report()["conf"] == 0.10


def test_imgsz_exactly_640():
    from buildreasonseg_mvp.task6q_reference_resolver import PROPOSAL_IMGSZ, config_report

    assert PROPOSAL_IMGSZ == 640
    assert config_report()["imgsz"] == 640


def test_max_det_exactly_100():
    from buildreasonseg_mvp.task6q_reference_resolver import PROPOSAL_MAX_DET, config_report

    assert PROPOSAL_MAX_DET == 100
    assert config_report()["max_det"] == 100


def _function_source(path: Path, name: str) -> str:
    """Exact source text of one function (AST-based, so comments/docstrings are not lost)."""

    import ast

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{name} not found in {path.name}")


def test_no_threshold_sweep():
    from buildreasonseg_mvp.task6q_reference_resolver import config_report

    config = config_report()
    assert config["threshold_sweep"] is False
    assert config["tta"] is False and config["tiling"] is False
    # semantic: no code iterates over alternative conf / max_det values, and the run functions pass
    # the frozen constants directly
    module = _code_only(TASK6Q_MODULE)
    assert "for conf in" not in module and "for max_det in" not in module
    for name in TASK6Q_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("for conf in", "conf_grid", "max_det_grid", "CONF_GRID", "MAX_DET_GRID",
                       "conf=" + "float(", "PROPOSAL_CONF *"):
            assert marker not in code, f"{name} must not sweep thresholds ({marker})"
    run = _function_source(TASK6Q_MODULE, "run_frozen_proposals")
    assert "conf=PROPOSAL_CONF" in run and "max_det=PROPOSAL_MAX_DET" in run
    # the audit/propagation call sites also pass the frozen constants directly
    for name in TASK6Q_SOURCES:
        code = _code_only(SCRIPTS / name)
        if "model.predict(" in code:
            assert "conf=PROPOSAL_CONF" in code and "max_det=PROPOSAL_MAX_DET" in code, name


def test_no_gt_in_reference_selection():
    code = _code_only(TASK6Q_MODULE)
    for marker in ("canonical_instances", "gt_reference_mask", "ground_truth", "read_tile_cache",
                   "source_feature_id"):
        assert marker not in code, f"the resolver must not consult GT ({marker})"
    # every select_reference call site passes only proposals + family
    import re

    for name in TASK6Q_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for match in re.finditer(r"select_reference\(([^)]*)\)", text):
            arguments = [part.strip() for part in match.group(1).split(",")]
            assert arguments == ["proposals", "family"] or arguments == ["proposals", "family "], \
                f"{name}: select_reference must receive only proposals + family, got {arguments}"


def test_no_target_id_in_reference_selection():
    code = _code_only(TASK6Q_MODULE)
    for marker in ("target_source_feature_id", "target_instance_id", "target_mask"):
        assert marker not in code, f"the resolver must not use {marker}"
    selection = _function_source(TASK6Q_MODULE, "select_reference")
    assert "target" not in selection
    eligibility = _function_source(TASK6Q_MODULE, "is_eligible")
    assert "target" not in eligibility


# ---------------------------------------------------------------- 8-16 resolver semantics


def test_masks_restored_to_512():
    from buildreasonseg_mvp.task6q_reference_resolver import TILE_SIZE, build_proposal

    assert TILE_SIZE == 512
    mask = _mask_with_bbox(10, 40, 20, 60)
    proposal = build_proposal(0, 0.9, mask)
    assert proposal.mask.shape == (512, 512)
    assert proposal.area_px == 30 * 40
    with pytest.raises(ValueError):
        build_proposal(0, 0.9, np.zeros((64, 64), dtype=bool))
    from buildreasonseg_mvp.task6m_eval import normalize_mask

    assert normalize_mask(np.ones((640, 640), dtype=np.uint8)).shape == (512, 512)


def test_border_predicate_exact():
    from buildreasonseg_mvp.task6q_reference_resolver import touches_border_exact

    assert touches_border_exact(_mask_with_bbox(0, 5, 100, 120)) is True
    assert touches_border_exact(_mask_with_bbox(507, 512, 100, 120)) is True
    assert touches_border_exact(_mask_with_bbox(100, 120, 0, 5)) is True
    assert touches_border_exact(_mask_with_bbox(100, 120, 507, 512)) is True
    assert touches_border_exact(_mask_with_bbox(100, 120, 100, 120)) is False
    assert touches_border_exact(np.zeros((512, 512), dtype=bool)) is False


def test_merge_threshold_exact_0_20():
    from buildreasonseg_mvp.task6q_reference_resolver import (
        MERGE_BBOX_EXTENT_RATIO_MAX, build_proposal, is_eligible,
    )

    assert MERGE_BBOX_EXTENT_RATIO_MAX == 0.20
    # 512*512*0.20 = 52428.8 px of bbox area; a 229x229 bbox is 52441 -> just over
    over = build_proposal(0, 0.9, _mask_with_bbox(100, 329, 100, 329))
    assert over.bbox_extent_ratio > 0.20
    assert is_eligible(over, "largest") is False
    under = build_proposal(1, 0.9, _mask_with_bbox(100, 320, 100, 320))
    assert under.bbox_extent_ratio <= 0.20
    assert is_eligible(under, "largest") is True


def test_tiny_threshold_exact_150():
    from buildreasonseg_mvp.task6q_reference_resolver import (
        TINY_COMPONENT_AREA_PX, build_proposal, is_eligible,
    )

    assert TINY_COMPONENT_AREA_PX == 150
    tiny = build_proposal(0, 0.9, _mask_with_bbox(10, 22, 10, 22))  # 144 px
    assert tiny.area_px == 144
    assert is_eligible(tiny, "smallest") is False
    assert is_eligible(tiny, "largest") is True  # tiny not rejected for largest
    okay = build_proposal(1, 0.9, _mask_with_bbox(10, 23, 10, 23))  # 169 px
    assert okay.area_px == 169
    assert is_eligible(okay, "smallest") is True


def test_largest_eligibility_exact():
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal, is_eligible

    normal = build_proposal(0, 0.9, _mask_with_bbox(100, 200, 100, 200))
    border = build_proposal(1, 0.9, _mask_with_bbox(0, 100, 100, 200))
    merged = build_proposal(2, 0.9, _mask_with_bbox(0, 400, 0, 400))
    assert is_eligible(normal, "largest") is True
    assert is_eligible(border, "largest") is False
    assert is_eligible(merged, "largest") is False


def test_smallest_eligibility_exact():
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal, is_eligible

    normal = build_proposal(0, 0.9, _mask_with_bbox(100, 200, 100, 200))
    border = build_proposal(1, 0.9, _mask_with_bbox(0, 100, 100, 200))
    merged = build_proposal(2, 0.9, _mask_with_bbox(0, 400, 0, 400))
    tiny = build_proposal(3, 0.9, _mask_with_bbox(10, 20, 10, 20))
    assert is_eligible(normal, "smallest") is True
    assert is_eligible(border, "smallest") is False
    assert is_eligible(merged, "smallest") is False
    assert is_eligible(tiny, "smallest") is False


def test_deterministic_largest_selection():
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal, select_reference

    proposals = [
        build_proposal(0, 0.9, _mask_with_bbox(10, 30, 10, 30)),
        build_proposal(1, 0.8, _mask_with_bbox(100, 200, 100, 200)),
        build_proposal(2, 0.7, _mask_with_bbox(300, 360, 300, 360)),
    ]
    selection = select_reference(proposals, "largest")
    assert selection.abstained is False
    assert selection.proposal.index == 1
    assert selection.proposal.area_px == 100 * 100
    # repeated calls are identical (deterministic)
    assert select_reference(proposals, "largest").proposal.index == 1


def test_deterministic_smallest_selection():
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal, select_reference

    proposals = [
        build_proposal(0, 0.9, _mask_with_bbox(10, 30, 10, 30)),   # 400 px
        build_proposal(1, 0.8, _mask_with_bbox(100, 200, 100, 200)),
        build_proposal(2, 0.7, _mask_with_bbox(300, 316, 300, 316)),  # 256 px
    ]
    selection = select_reference(proposals, "smallest")
    assert selection.abstained is False
    assert selection.proposal.index == 2
    assert selection.proposal.area_px == 256


def test_tie_break_exact():
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal, select_reference

    # identical areas -> higher confidence wins
    proposals = [
        build_proposal(0, 0.50, _mask_with_bbox(10, 30, 10, 30)),
        build_proposal(1, 0.90, _mask_with_bbox(100, 120, 100, 120)),
        build_proposal(2, 0.70, _mask_with_bbox(200, 220, 200, 220)),
    ]
    assert select_reference(proposals, "largest").proposal.index == 1
    # identical areas and confidences -> lower original index wins
    tied = [
        build_proposal(0, 0.90, _mask_with_bbox(10, 30, 10, 30)),
        build_proposal(1, 0.90, _mask_with_bbox(100, 120, 100, 120)),
    ]
    assert select_reference(tied, "largest").proposal.index == 0
    # abstention cases are explicit
    assert select_reference([], "largest").reason == "no_proposals"
    assert select_reference([build_proposal(0, 0.9, _mask_with_bbox(0, 100, 0, 100))],
                            "largest").reason == "no_eligible_proposals"


# ---------------------------------------------------------------- 17-29 chain semantics


def test_refval_unique_reused():
    resolver = _artifact("task6q_reference_resolver_val.json")
    if resolver is None:
        pytest.skip("reference audit not run yet")
    assert resolver["pack"]["name"] == "RefValUnique"
    assert resolver["pack"]["count"] == 219
    pack = REPO_ROOT / "artifacts" / "task6p" / "reference_packs" / "ref_val_unique.json"
    if pack.is_file():
        assert resolver["pack"]["sha256"] == _sha256(pack)
    manifest = _artifact("task6p_reference_pack_manifest.json")
    assert resolver["pack"]["sha256"] == manifest["packs"]["ref_val_unique"]["sha256"]


def test_mini_val_240_reused():
    target = _artifact("task6q_target_val.json")
    if target is None:
        pytest.skip("target propagation not run yet")
    assert target["pack"]["name"] == "MiniVal240"
    assert target["pack"]["count"] == 240
    manifest = _artifact("task6n_pack_manifest.json")
    pack = Path(manifest["packs"]["mini_val_240"]["path"])
    if pack.is_file():
        assert _sha256(pack) == manifest["packs"]["mini_val_240"]["sha256"]


def test_paired_val_20_reused():
    paired = _artifact("task6q_target_paired_val.json")
    if paired is None:
        pytest.skip("paired propagation not run yet")
    assert paired["pairs"] == 20
    for row in paired["rows"]:
        assert "passes" in row and "a" in row and "b" in row


def test_no_test_split():
    for name in TASK6Q_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "split('test')",
                       "task6q_test", "test_fixed120"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"
    for name in ("task6q_reference_resolver_val.json", "task6q_reference_failure_attribution.json",
                 "task6q_target_val.json", "task6q_target_paired_val.json", "task6q_verdict.json",
                 "task6q_proposal_cache_manifest.json"):
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, name


def test_v02_field_unchanged():
    changed = _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py")
    assert changed == "", "Task 6Q must not modify GeometricRelationField v0.2"
    target = _artifact("task6q_target_val.json")
    if target is not None:
        assert target["field"].startswith("GeometricRelationField v0.2")


def test_frozen_b3_checkpoint_unchanged():
    target = _artifact("task6q_target_val.json")
    if target is None:
        pytest.skip("target propagation not run yet")
    assert target["b3"]["matches"] is True
    assert target["b3"]["retrained"] is False
    assert _sha256(Path(target["b3"]["path"])) == target["b3"]["sha256"]
    task6o = _artifact("task6o_mini_val.json")
    assert target["b3"]["sha256"] == \
        task6o["variants"]["B3"]["training"]["checkpoint"]["sha256"]


def test_b3_not_retrained():
    for name in TASK6Q_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("b3.train(", "optimizer = ", "AdamW", "loss.backward()", "optimizer.step()"):
            assert marker not in code, f"{name} must not train anything ({marker})"
    target = _artifact("task6q_target_val.json")
    if target is not None:
        assert target["b3"]["retrained"] is False


def test_no_oracle_reference_in_downstream_chain():
    predict = _function_source(SCRIPTS / "task6q_target_propagation.py", "predict_target")
    for marker in ("reference_source_feature_id", "canonical_instances", "masks.mask"):
        assert marker not in predict, f"the predicted path must not use the oracle reference ({marker})"
    assert "reference_mask" in predict, "the chain must consume the resolved proposal reference"
    target = _artifact("task6q_target_val.json")
    if target is not None:
        assert target["reference_source"] == "frozen_proposal_resolver"


def test_gt_reference_only_evaluation():
    audit = _code_only(SCRIPTS / "task6q_reference_audit.py")
    assert "gt_reference_mask(record)" in audit
    helper = _function_source(SCRIPTS / "task6q_reference_audit.py", "gt_reference_mask")
    assert "canonical_instances" in helper  # explicitly framed as an evaluation helper
    selection = _function_source(TASK6Q_MODULE, "select_reference")
    assert "gt" not in selection
    eligibility = _function_source(TASK6Q_MODULE, "is_eligible")
    assert "gt" not in eligibility


def test_gt_target_only_evaluation():
    code = _code_only(SCRIPTS / "task6q_target_propagation.py")
    for marker in ("loss(", "backward()", "optimizer"):
        assert marker not in code, f"{marker} must not appear in a scoring-only chain"
    assert "target_source_feature_id" in code  # used only to fetch the scoring GT


def test_same_reference_reused_in_paired_same_reference_case():
    paired = _artifact("task6q_target_paired_val.json")
    if paired is None:
        pytest.skip("paired propagation not run yet")
    assert paired["same_resolved_reference_reused_for_pairs"] is True
    for row in paired["rows"]:
        if not row["abstained"]:
            assert row["same_resolved_reference_reused"] is True


# ---------------------------------------------------------------- 30-38 no scope creep


def test_no_yolo_training():
    for name in TASK6Q_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("model.train(", "YOLO(...).train", "resume=True", "epochs=", "patience="):
            assert marker not in code, f"{name} must not train YOLO ({marker})"
    module = _code_only(TASK6Q_MODULE)
    assert "model.train(" not in module


def test_no_dense_reference_head_retraining():
    for name in TASK6Q_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("ReferenceMaskHead", "task6p_reference_head", "reference_loss"):
            assert marker not in code, f"{name} must not touch the Task 6P dense head ({marker})"


def test_no_ref_token():
    for path in [SCRIPTS / name for name in TASK6Q_SOURCES] + [TASK6Q_MODULE]:
        assert "[REF]" not in path.read_text(encoding="utf-8"), path.name


def test_no_grcl_or_scl():
    for path in [SCRIPTS / name for name in TASK6Q_SOURCES] + [TASK6Q_MODULE]:
        code = _code_only(path)
        for marker in ("GRCL", "SCL", "counterfactual_loss", "relation_loss", "focal_loss"):
            assert marker not in code, f"{path.name} must not add {marker}"


def test_no_nearest_or_l3():
    for path in [SCRIPTS / name for name in TASK6Q_SOURCES] + [TASK6Q_MODULE]:
        code = _code_only(path)
        assert "to_nearest" not in code, path.name
        assert "to_above_to_" not in code, path.name


def test_no_graph_transformer():
    for path in [SCRIPTS / name for name in TASK6Q_SOURCES] + [TASK6Q_MODULE]:
        code = _code_only(path)
        for marker in ("MultiheadAttention", "TransformerEncoder", "GraphConv", "MessagePassing",
                       "nn.Transformer"):
            assert marker not in code, f"{path.name} must not add {marker}"


def test_no_4b():
    for path in [SCRIPTS / name for name in TASK6Q_SOURCES] + [TASK6Q_MODULE]:
        text = path.read_text(encoding="utf-8")
        assert "Qwen3-VL-4B" not in text, path.name
        assert "4B" not in text.replace("4B)", ""), path.name


def test_no_new_dataset_download_or_gui():
    for path in [SCRIPTS / name for name in TASK6Q_SOURCES] + [TASK6Q_MODULE]:
        code = _code_only(path)
        for marker in ("urlretrieve", "requests.get", "Invoke-WebRequest", "hf_hub_download",
                       "snapshot_download", "pip install", "tkinter", "PyQt", "gradio",
                       "streamlit", "cv2.imshow", "flask", "fastapi"):
            assert marker not in code, f"{path.name} must not use {marker}"


def test_previous_suite_preserved():
    for name in ("test_task6p_differentiable_field_predicted_reference.py",
                 "test_task6o_field_causal_decomposition.py",
                 "test_task6n_oracle_relation_field.py",
                 "test_task6m1_convergence_demo_fix.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--", "tests/"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert removed == "", f"no test file may be removed: {removed}"
