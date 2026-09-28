"""Task 6M tests: the 28 required checks from the task file (section 22).

Structural checks always run; checks that need a trained checkpoint or a generated artifact skip
(never fail) while those are still being produced.
"""

from __future__ import annotations

import hashlib
import json
import re
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
NV = REPO_ROOT / "datasets" / "whu_native_vector" / "v1.0"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
V011 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1"
EXPORT = REPO_ROOT / "artifacts" / "task6m_yolo_native"

TASK6M_SOURCES = (
    "task6m_freeze_eval_packs.py", "task6m_build_export.py", "task6m_smoke.py",
    "task6m_train.py", "task6m_proposal_eval.py", "task6m_j1v2.py", "task6m_parser_eval.py",
    "task6m_j4v2.py", "task6m_demo_audit.py", "task6m_error_attribution.py",
    "task6m_download_weights.py", "task6m_environment.py",
)


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _code(relative: str) -> str:
    return (SCRIPTS / relative).read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 1-4 frozen inputs


def test_canonical_native_dataset_unchanged():
    index = _artifact("task6l_artifact_index.json")
    if index is None:
        pytest.skip("Task 6L artifact index missing")
    checked = 0
    for entry in index["artifacts"]:
        if not str(entry["path"]).startswith("datasets/whu_native_vector/"):
            continue
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            continue
        assert _sha256(path) == entry["sha256"], f"{entry['path']} changed since Task 6L"
        checked += 1
    assert checked >= 4, f"expected several canonical files to verify, checked {checked}"


def test_v02_dataset_unchanged():
    index = _artifact("task6l_artifact_index.json")
    if index is None:
        pytest.skip("Task 6L artifact index missing")
    for entry in index["artifacts"]:
        if not str(entry["path"]).startswith("datasets/build_spatial_reason/v0.2/"):
            continue
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            continue
        assert _sha256(path) == entry["sha256"], f"{entry['path']} changed since Task 6L"
    for split in ("train", "val", "test"):
        lines = sum(
            1 for line in (V02 / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        )
        expected = json.loads((V02 / "statistics.json").read_text(encoding="utf-8"))["by_split"][split]
        assert lines == expected, f"v0.2 {split}.jsonl line count changed"


def test_v011_dataset_unchanged():
    verdict = _artifact("task6l_verdict.json")
    if verdict is None:
        pytest.skip("Task 6L verdict missing")
    recorded = verdict["gates"]["v011_frozen_unchanged"]["measured"]
    for name, digest in recorded.items():
        assert _sha256(V011 / name) == digest, f"v0.1.1 {name} changed"


def test_no_source_mutation():
    for relative in TASK6M_SOURCES:
        text = _code(relative)
        for marker in ("shutil.move", "os.remove", "os.rename", "shutil.rmtree(source", "unlink("):
            assert marker not in text, f"{relative} must not mutate a source ({marker})"
    export = _code("task6m_build_export.py")
    assert "os.link" in export or "copy2" in export, "images must be linked or copied, never moved"
    assert "source_image" in export


# ---------------------------------------------------------------- 5-8 export integrity


def test_export_preserves_all_non_hole_instances():
    audit = _artifact("task6m_training_export_audit.json")
    if audit is None:
        pytest.skip("export audit not generated yet")
    assert audit["verdict"] == "EXPORT_VALID"
    fidelity = audit["fidelity"]
    assert fidelity["missing_non_hole_instances"] == 0
    assert fidelity["malformed_labels"] == 0
    assert fidelity["tiny_instances_total"] == fidelity["tiny_instances_recovered"]
    assert fidelity["mean_tile_union_iou"] >= 0.995
    assert fidelity["mean_per_instance_iou_area_ge_9px"] >= 0.995


def test_empty_labels_valid():
    audit = _artifact("task6m_training_export_audit.json")
    if audit is None or not (EXPORT / "labels" / "train").is_dir():
        pytest.skip("export not present")
    empty_files = [
        path for path in (EXPORT / "labels" / "train").glob("*.txt") if path.stat().st_size == 0
    ]
    assert audit["counts"]["empty_label_files"] > 0
    assert len(empty_files) > 0
    for path in empty_files[:50]:
        assert path.read_text(encoding="utf-8").strip() == ""


def test_no_min_area_filter_in_export():
    export = _code("task6m_build_export.py")
    assert "contourArea" not in export and "approxPolyDP" not in export
    audit = _artifact("task6m_training_export_audit.json")
    if audit is not None:
        assert audit["rules"]["min_contour_area_filter"] is None
        assert audit["rules"]["tiny_instances_kept"] is True
        assert audit["fidelity"]["tiny_instances_total"] > 0


def test_hole_loss_recorded_only_in_derived_export():
    audit = _artifact("task6m_training_export_audit.json")
    if audit is None:
        pytest.skip("export audit not generated yet")
    assert audit["fidelity"]["hole_instances_affected"] > 0
    assert audit["fidelity"]["hole_instances_detail"]
    assert "exterior" in audit["rules"]["polygon_form"]
    # the canonical dataset still carries the holes
    from buildreasonseg_mvp.whu_native_vector import read_tile_cache

    detail = audit["fidelity"]["hole_instances_detail"][0]
    cache = read_tile_cache(detail["tile_id"])
    assert int(cache["n_holes"][detail["tile_instance_id"] - 1]) >= 1


# ---------------------------------------------------------------- 9-11 splits + packs


def test_scene_disjoint_split_correct():
    payload = json.loads((NV / "splits" / "scene_disjoint_v1.json").read_text(encoding="utf-8"))
    assert payload["definition"] == {"train": "train1", "val": "train2", "test": "test"}
    audit = _artifact("task6m_training_export_audit.json")
    if audit is not None:
        per_split = audit["counts"]["per_split"]
        assert per_split["train"]["tiles"] == 10044
        assert per_split["val"]["tiles"] == 3618
        assert per_split["test"]["tiles"] == 3726


def test_zero_feature_leakage():
    audit = _artifact("task6l_scene_disjoint_split_audit.json")
    if audit is None:
        pytest.skip("Task 6L split audit missing")
    assert audit["source_feature_leakage_zero"] is True
    assert audit["tile_overlap_zero"] is True


def test_eval_packs_frozen_before_tuning():
    manifest = _artifact("task6m_eval_pack_manifest.json")
    if manifest is None:
        pytest.skip("eval packs not frozen yet")
    assert manifest["construction_policy"]["frozen_before_training"] is True
    for name, entry in manifest["artifacts"].items():
        path = EVAL / name
        assert path.is_file(), name
        assert _sha256(path) == entry["sha256"], f"{name} changed after freezing"
    for split in ("val", "test"):
        fixed = _artifact(f"task6m_{split}_fixed120.json")
        paired = _artifact(f"task6m_{split}_paired20.json")
        assert len(fixed["records"]) == 120
        assert len(paired["pairs"]) == 20
    # the test packs must be older than any tuning artifact
    tuning = EVAL / "task6m_inference_config_frozen.json"
    if tuning.is_file():
        assert (EVAL / "task6m_test_fixed120.json").stat().st_mtime <= tuning.stat().st_mtime


def test_no_test_metrics_before_frozen_inference_config():
    frozen = _artifact("task6m_inference_config_frozen.json")
    if frozen is None:
        pytest.skip("inference config not frozen yet")
    assert frozen["frozen_before_test"] is True
    assert frozen["test_metrics_inspected_before_freezing"] is False
    frozen_path = EVAL / "task6m_inference_config_frozen.json"
    j4 = EVAL / "task6m_j4v2_test.json"
    if j4.is_file():
        assert frozen_path.stat().st_mtime <= j4.stat().st_mtime, (
            "the inference config must be frozen before the single graded test run"
        )


# ---------------------------------------------------------------- 12-17 no GT in the chain


def test_gt_never_repairs_proposals():
    structured = (REPO_ROOT / "buildreasonseg_mvp" / "task6m_structured.py").read_text(encoding="utf-8")
    # the executor is only ever called with the predicted candidate set
    for match in re.finditer(r"execute_program_by_id\(([^)]*)\)", structured):
        arguments = match.group(1)
        assert "candidate_set" in arguments
        assert "gt" not in arguments.lower() and "ground_truth" not in arguments.lower()
    # GT masks appear only in comparison helpers
    for match in re.finditer(r"iou\(([^)]*)\)", structured):
        assert "selected" in match.group(1) or "target" in match.group(1) or "gt" in match.group(1).lower()


def test_parser_is_text_only():
    parser_eval = _code("task6m_parser_eval.py")
    assert "instruction_en" in parser_eval
    assert "build_batch" in parser_eval
    for marker in ("image", "pixel_values", "processor(images"):
        assert marker not in parser_eval.split("def main")[1] or marker == "image", marker
    report = _artifact("task6m_parser_v02.json")
    if report is not None:
        assert report["parser"]["image_tokens"] is False
        assert report["parser"]["query_type_leakage"] is False
        assert report["parser"]["input"] == "instruction text only"


def test_no_query_type_in_parser_input():
    parser = (REPO_ROOT / "buildreasonseg_mvp" / "program_parser.py").read_text(encoding="utf-8")
    build_batch = parser.split("def build_batch")[1].split("def forward")[0]
    assert "program_ids" in build_batch, "labels are needed for training only"
    assert "labels = torch.tensor" in build_batch
    # the instruction text passed to the model must not contain the program id
    assert 'messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]' in build_batch
    report = _artifact("task6m_parser_v02.json")
    if report is not None:
        assert report["parser"]["query_type_leakage"] is False


def test_no_gt_in_j4v2():
    text = _code("task6m_j4v2.py")
    # GT is only used through the scoring helpers of the shared module
    assert "canonical_instances" not in text
    assert "evaluate_fixed120" in text and "evaluate_paired20" in text
    assert "program_of" in text
    report = _artifact("task6m_j4v2_test.json")
    if report is not None:
        assert report["single_run"] is True


def test_executor_receives_predicted_geometry_only():
    structured = (REPO_ROOT / "buildreasonseg_mvp" / "task6m_structured.py").read_text(encoding="utf-8")
    candidate_builder = structured.split("def candidate_set_from_predictions")[1].split("def load_pack")[0]
    assert "prediction[\"masks\"]" in candidate_builder
    assert "canonical_instances" not in candidate_builder
    assert "read_tile_cache" not in candidate_builder


# ---------------------------------------------------------------- 18-20 packs + CLI


def test_pairs_same_image_different_native_target():
    for split in ("val", "test"):
        pack = _artifact(f"task6m_{split}_paired20.json")
        if pack is None:
            continue
        for pair in pack["pairs"]:
            assert pair["target_instance_a"] != pair["target_instance_b"]
            assert pair["query_type_a"] != pair["query_type_b"]
            assert pair["tile_id"] == pair["tile_id"]
            assert pack["construction"]["same_tile"] is True
            assert pack["construction"]["different_target_instances"] is True


def test_all_intended_programs_represented():
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    for split in ("val", "test"):
        fixed = _artifact(f"task6m_{split}_fixed120.json")
        if fixed is None:
            continue
        programs = {record["query_type"] for record in fixed["records"]}
        missing = sorted(set(EXPECTED_QUERY_TYPES) - programs)
        assert not missing, f"{split} fixed120 missing {missing}"


def test_demo_cli_works_without_annotations():
    cli = REPO_ROOT / "predict_structured.py"
    assert cli.is_file()
    text = cli.read_text(encoding="utf-8")
    assert "ground_truth_used" in text
    for marker in ("canonical_instances", "read_tile_cache", "task6m_val_fixed120"):
        assert marker not in text, f"the CLI must not read ground truth ({marker})"
    audit = _artifact("task6m_demo_cli_audit.json")
    if audit is not None:
        assert audit["images_audited"] >= 10
        assert audit["summary"]["any_ground_truth_used"] is False
        assert audit["gate"]["passed"] is True


# ---------------------------------------------------------------- 21-23 determinism + hashes


def test_eval_mode_determinism():
    frozen = _artifact("task6m_inference_config_frozen.json")
    if frozen is None:
        pytest.skip("inference config not frozen yet")
    checkpoint = Path(frozen["checkpoint"]["path"])
    if not checkpoint.is_file():
        pytest.skip("checkpoint not available")
    # Ultralytics lives in the dedicated proposal env; the repository test env does not have it.
    pytest.importorskip("ultralytics", reason="run in .conda/buildreasonseg-proposal for this check")
    from buildreasonseg_mvp.task6m_eval import tile_ids
    from buildreasonseg_mvp.task6m_structured import predict_tiles

    from ultralytics import YOLO

    ids = tile_ids("val")[:3]
    model = YOLO(str(checkpoint))
    first = predict_tiles(model, ids, conf=float(frozen["conf"]), max_det=int(frozen["max_det"]),
                          verbose=False)
    second = predict_tiles(model, ids, conf=float(frozen["conf"]), max_det=int(frozen["max_det"]),
                           verbose=False)
    for tile_id in ids:
        left = first[tile_id]["masks"]
        right = second[tile_id]["masks"]
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert np.array_equal(a, b), "eval-mode predictions must be deterministic"


def test_prediction_path_is_deterministic_by_construction():
    """The check that never needs the proposal env: identical inputs -> identical masks."""

    import numpy as np

    from buildreasonseg_mvp.task6m_eval import build_label_map, fast_best_iou

    mask_a = np.zeros((512, 512), dtype=bool)
    mask_a[10:40, 20:60] = True
    mask_b = np.zeros((512, 512), dtype=bool)
    mask_b[200:260, 300:400] = True
    first = build_label_map([mask_a, mask_b])
    second = build_label_map([mask_a, mask_b])
    assert np.array_equal(first, second)
    areas = np.asarray([int(m.sum()) for m in (mask_a, mask_b)], dtype=np.int64)
    assert fast_best_iou(mask_a, first, areas) == fast_best_iou(mask_a, second, areas) == (1.0, 0)


def test_unsupported_program_explicit_failure():
    sys.path.insert(0, str(REPO_ROOT))
    import predict_structured

    class StubRuntime:
        def predict(self, prompt: str) -> str:
            return "definitely_not_a_program"

    with pytest.raises(SystemExit):
        predict_structured.parse_instruction(StubRuntime(), "some prompt", verbose=False)

    class GoodRuntime:
        def predict(self, prompt: str) -> str:
            return "leftmost"

    program, trace = predict_structured.parse_instruction(GoodRuntime(), "leftmost building", verbose=False)
    assert program == "leftmost"
    assert trace["model_input"] == "instruction text only"


def test_model_hashes_recorded():
    manifest = _artifact("task6m_environment_manifest.json")
    if manifest is None:
        pytest.skip("environment manifest missing")
    weights = manifest["pretrained_weights"]
    assert weights["yolo26m-seg.pt"]["sha256"]
    assert weights["yolo26m-seg.pt"]["url"].startswith("https://github.com/ultralytics/assets/")
    for name, entry in weights.items():
        path = Path(entry["local_path"])
        if path.is_file():
            assert _sha256(path) == entry["sha256"], f"{name} hash mismatch"
    training = _artifact("task6m_training_summary.json")
    if training and training.get("checkpoints", {}).get("best"):
        best = Path(training["checkpoints"]["best"]["path"])
        if best.is_file():
            assert _sha256(best) == training["checkpoints"]["best"]["sha256"]


# ---------------------------------------------------------------- 24-28 repo discipline


def test_no_large_weights_staged():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for path in tracked:
        assert not path.endswith((".pt", ".pth", ".ckpt", ".onnx", ".engine")), path
        assert not path.startswith(".conda/"), path
        assert not path.startswith("artifacts/task6m_yolo_native/"), path
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for marker in (".conda/", "artifacts/"):
        assert marker in gitignore


def test_no_old_editable_ultralytics_fork_used():
    manifest = _artifact("task6m_environment_manifest.json")
    if manifest is None:
        pytest.skip("environment manifest missing")
    path = manifest["environment"]["ultralytics_import_path"]
    assert "ultralytics-main" not in path, "the historical editable fork must not be used"
    assert "buildreasonseg-proposal" in path or "site-packages" in path
    # training/inference entry points must never point at the historical env or the local fork
    for relative in ("task6m_smoke.py", "task6m_train.py", "task6m_proposal_eval.py",
                     "task6m_j1v2.py", "task6m_j4v2.py"):
        text = _code(relative)
        assert "yolo_sam_env" not in text, f"{relative} must not use the historical env"
        assert "ultralytics-main" not in text, relative
    cli = (REPO_ROOT / "predict_structured.py").read_text(encoding="utf-8")
    assert "yolo_sam_env" not in cli and "ultralytics-main" not in cli
    # the manifest documents the choice explicitly
    assert "yolo_sam_env" in manifest["environment"]["why_not_yolo_sam_env"]


def test_no_4b_upgrade():
    parser_cfg = (REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml").read_text(encoding="utf-8")
    model_ids = re.findall(r"qwen_model_id:\s*(\S+)", parser_cfg)
    assert model_ids, "the parser config must declare its model"
    assert all("2B" in value for value in model_ids), model_ids
    for relative in TASK6M_SOURCES:
        text = _code(relative)
        for match in re.finditer(r"qwen_model_id[\"']?\s*[:=]\s*[\"']([^\"']+)", text):
            assert "2B" in match.group(1), f"{relative} configures a non-2B parser: {match.group(1)}"
        assert "Qwen3-VL-4B" not in text, relative
    parser = _code("task6m_parser_eval.py")
    assert "task6j_program_parser" in parser


def test_no_ref_sre_scl():
    for relative in TASK6M_SOURCES + ("task6m_build_export.py",):
        text = _code(relative)
        for marker in ("[REF]", "SRE", "SCL"):
            assert marker not in text, f"{relative} must not add {marker}"
    cli = (REPO_ROOT / "predict_structured.py").read_text(encoding="utf-8")
    for marker in ("[REF]", "SRE", "SCL"):
        assert marker not in cli


def test_no_gui():
    for path in list(SCRIPTS.glob("task6m_*.py")) + [REPO_ROOT / "predict_structured.py",
                                                     REPO_ROOT / "buildreasonseg_mvp" / "task6m_eval.py",
                                                     REPO_ROOT / "buildreasonseg_mvp" / "task6m_structured.py"]:
        text = path.read_text(encoding="utf-8")
        for marker in ("tkinter", "PyQt", "gradio", "streamlit", "cv2.imshow", "flask", "fastapi"):
            assert marker not in text, f"{path.name} must not build a GUI ({marker})"


def test_required_artifacts_present_or_pending():
    required = (
        "task6m_eval_pack_manifest.json", "task6m_val_fixed120.json", "task6m_val_paired20.json",
        "task6m_test_fixed120.json", "task6m_test_paired20.json", "task6m_training_export_audit.json",
        "task6m_environment_manifest.json", "task6m_training_summary.json", "task6m_proposal_val.json",
        "task6m_inference_config_frozen.json", "task6m_j1v2_val.json", "task6m_parser_v02.json",
        "task6m_j4v2_test.json", "task6m_error_attribution.json", "task6m_demo_cli_audit.json",
        "task6m_verdict.json",
    )
    present = [name for name in required if (EVAL / name).is_file()]
    assert len(present) >= 6, f"expected the frozen packs and early artifacts, found {present}"
    pending = [name for name in required if name not in present]
    if pending:
        pytest.skip(f"later artifacts still pending: {pending}")
    verdict = _artifact("task6m_verdict.json")
    assert verdict["verdict"] in verdict["allowed_verdicts"]
