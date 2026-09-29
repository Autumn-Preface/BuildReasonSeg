"""Task 6M.1 focused tests (section 18): 22 checks on the continuation and the Demo gate fix.

Structural checks always run; checks that need a finished continuation or a generated 6M.1 artifact
skip (never fail) while the training is still in progress.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
NV = REPO_ROOT / "datasets" / "whu_native_vector" / "v1.0"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
SOURCE_RUN = REPO_ROOT / "artifacts" / "checkpoints" / "task6m" / "runs" / "m1_yolo26m_seg"
SNAPSHOT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "source_epoch18_snapshot"
RUN_6M1 = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" / "m1_yolo26m_seg_continued"

EXPECTED_HASHES = {
    "best.pt": "fd407db634a8a7ef83f09f8096686e73407095105f1b45c70c623d18dbf4ea44",
    "last.pt": "ea998bda37dd2dcb2cc7bb5e19f6d15b7a205a137c9cc2866c508a45513f860e",
}
TASK6M_PACKS = (
    "task6m_val_fixed120.json", "task6m_val_paired20.json",
    "task6m_test_fixed120.json", "task6m_test_paired20.json",
)
TASK6M1_SOURCES = (
    "task6m1_source_snapshot.py", "task6m1_resume_train.py", "task6m1_finalize_training.py",
    "task6m1_domain_gate_check.py", "task6m1_gate_coverage.py", "task6m1_demo_audit.py",
    "task6m1_error_attribution.py", "task6m1_verdict.py",
)


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- 1-5 evidence preservation


def test_task6m_tracked_artifacts_unchanged():
    import subprocess

    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", "evaluation/task6m_", "docs/task6m_"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    # 6M.1 adds its own files; no Task 6M artifact may be modified
    modified = [line for line in changed.splitlines() if line.strip().startswith("M")]
    assert not modified, f"Task 6M artifacts must stay frozen, modified: {modified}"
    manifest = _artifact("task6m_eval_pack_manifest.json")
    assert manifest is not None
    for name, entry in manifest["artifacts"].items():
        assert _sha256(EVAL / name) == entry["sha256"], f"{name} changed after Task 6M"


def test_source_checkpoint_hashes_match_task6m():
    audit = _artifact("task6m1_source_checkpoint_audit.json")
    if audit is None:
        pytest.skip("Task 6M.1 source audit not written yet")
    assert audit["verdict"] == "SOURCE_CHECKPOINT_VERIFIED"
    for name, expected in EXPECTED_HASHES.items():
        assert audit["checkpoints"][name]["actual_sha256"] == expected
        path = SOURCE_RUN / "weights" / name
        if path.is_file():
            assert _sha256(path) == expected, f"{name} drifted after the Task 6M.1 audit"


def test_snapshot_hashes_match_originals():
    audit = _artifact("task6m1_source_checkpoint_audit.json")
    if audit is None:
        pytest.skip("Task 6M.1 source audit not written yet")
    assert audit["snapshot_matches_originals"] is True
    for name in EXPECTED_HASHES:
        snapshot = SNAPSHOT / "weights" / name
        original = SOURCE_RUN / "weights" / name
        if snapshot.is_file() and original.is_file():
            assert _sha256(snapshot) == _sha256(original) == EXPECTED_HASHES[name]


def test_source_data_unchanged():
    index = _artifact("task6l_artifact_index.json")
    if index is None:
        pytest.skip("Task 6L artifact index missing")
    checked = 0
    for entry in index["artifacts"]:
        if not str(entry["path"]).startswith("datasets/whu_native_vector/"):
            continue
        path = REPO_ROOT / entry["path"]
        if path.is_file():
            assert _sha256(path) == entry["sha256"], f"{entry['path']} changed"
            checked += 1
    assert checked >= 4


def test_v02_unchanged():
    index = _artifact("task6l_artifact_index.json")
    if index is None:
        pytest.skip("Task 6L artifact index missing")
    for entry in index["artifacts"]:
        if not str(entry["path"]).startswith("datasets/build_spatial_reason/v0.2/"):
            continue
        path = REPO_ROOT / entry["path"]
        if path.is_file():
            assert _sha256(path) == entry["sha256"], f"{entry['path']} changed"
    statistics = json.loads((V02 / "statistics.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        lines = sum(
            1 for line in (V02 / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        )
        assert lines == statistics["by_split"][split]


# ---------------------------------------------------------------- 6-7 safe resume


def test_resume_starts_next_epoch_19():
    preflight = _artifact("task6m1_resume_preflight.json")
    if preflight is None:
        pytest.skip("resume pre-flight not written yet")
    assert preflight["verdict"] in ("SAFE_RESUME_READY", "SAFE_RESUME_COMPLETED")
    assert preflight["resume_checkpoint"]["stored_epoch_0based"] == 17
    assert preflight["resume_checkpoint"]["next_epoch_1based"] == 19
    assert preflight["checks"]["optimizer_state_present"] is True
    assert preflight["checks"]["ema_state_present"] is True
    assert preflight["checks"]["save_dir_is_task6m1_local"] is True
    # the actual run must really have started at epoch 19
    results = RUN_6M1 / "results.csv"
    if results.is_file():
        epochs = [
            int(line.split(",")[0])
            for line in results.read_text(encoding="utf-8").splitlines()[1:]
            if line.strip() and line.split(",")[0].strip().isdigit()
        ]
        assert epochs, "continuation results.csv must have rows"
        assert 19 in epochs, f"continuation must contain epoch 19, found {epochs[:5]}..."


def test_frozen_training_config_unchanged():
    preflight = _artifact("task6m1_resume_preflight.json")
    if preflight is None:
        pytest.skip("resume pre-flight not written yet")
    for key, entry in preflight["frozen_config_checks"].items():
        assert entry["matches"], f"{key} changed: {entry}"
    assert preflight["checks"]["frozen_config_unchanged"] is True
    assert preflight["checks"]["data_is_native_vector_export"] is True
    args_yaml = RUN_6M1 / "args.yaml"
    if args_yaml.is_file():
        text = args_yaml.read_text(encoding="utf-8")
        for key, value in (("imgsz", "640"), ("batch", "16"), ("workers", "4"), ("seed", "20260812")):
            assert re.search(rf"^{key}:\s*{value}\s*$", text, re.MULTILINE), f"{key} not {value}"


# ---------------------------------------------------------------- 8-10 freeze ordering + pack reuse


def test_no_test_read_before_6m1_config_freeze():
    frozen = EVAL / "task6m1_inference_config_frozen.json"
    j4 = EVAL / "task6m1_j4v2_test.json"
    if not frozen.is_file():
        if j4.is_file():
            pytest.fail("a Task 6M.1 test artifact exists without a frozen 6M.1 inference config")
        pytest.skip("6M.1 freeze not created yet")
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    assert payload["frozen_before_test"] is True
    assert payload["test_metrics_inspected_before_freezing"] is False
    if j4.is_file():
        assert frozen.stat().st_mtime <= j4.stat().st_mtime


def test_test_skipped_if_val_gate_fails():
    j1 = _artifact("task6m1_j1v2_val.json")
    j4 = EVAL / "task6m1_j4v2_test.json"
    if j1 is None:
        pytest.skip("validation gate not evaluated yet")
    if (j1.get("gate") or {}).get("passed") is False:
        assert not j4.is_file(), "section 10 forbids a test run when a validation gate fails"


def test_old_fixed_packs_reused_byte_for_byte():
    manifest = _artifact("task6m_eval_pack_manifest.json")
    if manifest is None:
        pytest.skip("Task 6M pack manifest missing")
    for name in TASK6M_PACKS:
        entry = manifest["artifacts"][name]
        assert _sha256(EVAL / name) == entry["sha256"], f"{name} must be reused unchanged"
    # the 6M.1 protocol must have consumed exactly the frozen packs: the record counts and the
    # fixed120/paired20 shapes recorded in the 6M.1 J1 artifact have to match the frozen packs
    j1 = _artifact("task6m1_j1v2_val.json")
    if j1 is None:
        pytest.skip("Task 6M.1 J1-v2 not run yet")
    fixed = json.loads((EVAL / "task6m_val_fixed120.json").read_text(encoding="utf-8"))
    paired = json.loads((EVAL / "task6m_val_paired20.json").read_text(encoding="utf-8"))
    assert j1["fixed120"]["records"] == len(fixed["records"]) == 120
    assert j1["paired20"]["pairs"] == len(paired["pairs"]) == 20
    assert j1["split"] == "val"
    assert fixed["construction"]["seed"] == 20260810
    assert paired["construction"]["seed"] == 20260910
    assert fixed["construction"]["frozen_before_training"] is True


# ---------------------------------------------------------------- 11-16 Demo gate


def test_no_gt_in_inference():
    cli = (REPO_ROOT / "predict_structured.py").read_text(encoding="utf-8")
    for marker in ("canonical_instances", "read_tile_cache", "task6m_val_fixed120",
                   "task6m_test_fixed120", "build_label_map"):
        assert marker not in cli, f"the CLI must not read ground truth ({marker})"
    assert "ground_truth_used" in cli
    demo = _artifact("task6m1_demo_cli_audit.json")
    if demo is not None:
        assert demo["gate"]["no_ground_truth_used"] is True
        assert all(row["ground_truth_used"] is False for row in demo["supported_runs"])


def test_unsupported_prompt_exits_4():
    import predict_structured

    assert predict_structured.UNSUPPORTED_EXIT_CODE == 4
    result = predict_structured.check_domain("Write a poem about the sea.")
    assert result["supported"] is False and result["reason"] == "missing_object_anchor"
    report = _artifact("task6m1_domain_gate_check.json")
    if report is not None:
        assert report["cli_probe"]["exit_code"] == 4
        assert report["cli_probe"]["status"] == "unsupported_instruction"
        assert report["cli_probe"]["abstention_reason"] == "out_of_domain_prompt"
        assert report["gate"]["passed"] is True


def test_unsupported_prompt_does_not_call_parser(monkeypatch, tmp_path):
    import predict_structured

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the parser must not be constructed for an out-of-domain prompt")

    monkeypatch.setattr(predict_structured, "build_program_parser", explode, raising=False)
    image = tmp_path / "tile.tif"
    image.write_bytes(b"not-an-image")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"x")
    exit_code = predict_structured.main([
        "--image", str(image),
        "--prompt", "今天天气怎么样？",
        "--proposal-checkpoint", str(checkpoint),
        "--parser-checkpoint", str(checkpoint),
        "--out-dir", str(tmp_path / "out"),
        "--quiet",
    ])
    assert exit_code == 4
    payload = json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "unsupported_instruction"
    assert payload["parsed_program"] is None


def test_unsupported_prompt_does_not_call_proposal_model(monkeypatch, tmp_path):
    import predict_structured

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"ultralytics", "torch", "transformers"}:
            raise AssertionError(f"the proposal/parser stack must not be imported for OOD prompts ({name})")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    image = tmp_path / "tile.tif"
    image.write_bytes(b"not-an-image")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"x")
    exit_code = predict_structured.main([
        "--image", str(image),
        "--prompt", "segment the airplane",
        "--proposal-checkpoint", str(checkpoint),
        "--parser-checkpoint", str(checkpoint),
        "--out-dir", str(tmp_path / "out2"),
        "--quiet",
    ])
    assert exit_code == 4
    payload = json.loads((tmp_path / "out2" / "result.json").read_text(encoding="utf-8"))
    assert payload["domain_gate"]["checked_before_parser"] is True
    assert payload["domain_gate"]["checked_before_proposal_model"] is True
    assert payload["proposal_count"] is None


def test_all_six_ood_prompts_rejected_and_five_positives_accepted():
    import predict_structured

    ood = (
        "Write a poem about the sea.",
        "今天天气怎么样？",
        "请总结这张图片。",
        "检测道路。",
        "segment the airplane",
        "   ",
    )
    positives = (
        "分割面积最大的建筑物。",
        "找出最左侧的建筑区域。",
        "分割面积最大的建筑物右侧最近的建筑物。",
        "segment the building nearest to the right of the largest building",
        "找出最小建筑物上方的建筑。",
    )
    for prompt in ood:
        assert predict_structured.check_domain(prompt)["supported"] is False, prompt
    for prompt in positives:
        result = predict_structured.check_domain(prompt)
        assert result["supported"] is True, (prompt, result)
    report = _artifact("task6m1_domain_gate_check.json")
    if report is not None:
        assert report["out_of_domain_all_rejected"] is True
        assert report["positive_prompts_all_accepted"] is True


def test_domain_gate_accepts_every_frozen_v02_template():
    """The gate vocabulary must not falsely reject an in-domain template (zh or en)."""

    import predict_structured

    coverage = _artifact("task6m1_domain_gate_coverage.json")
    if coverage is not None:
        assert coverage["gate"]["passed"] is True
        for split, entry in coverage["per_split"].items():
            assert entry["falsely_rejected_templates"] == [], f"{split} has false rejections"
            assert entry["accepted_zh_rate"] == 1.0, split
            assert entry["accepted_en_rate"] == 1.0, split
        assert coverage["falsely_rejected_template_variants"] == 0
    # the two anchor families the v0.2 templates need beyond the base list
    for anchor in ("最高", "最低", "highest", "lowest", "furthest left", "furthest right", "greatest"):
        assert anchor in predict_structured.DOMAIN_RELATION_ANCHORS, anchor
    # and the completion must not have opened the gate to the required out-of-domain prompts
    for prompt in ("Write a poem about the sea.", "今天天气怎么样？", "请总结这张图片。", "检测道路。",
                   "segment the airplane", "   "):
        assert predict_structured.check_domain(prompt)["supported"] is False, prompt


# ---------------------------------------------------------------- 17-18 CLI audit shape


def test_cli_audit_four_per_level_and_eight_programs():
    demo = _artifact("task6m1_demo_cli_audit.json")
    if demo is None:
        pytest.skip("Task 6M.1 CLI audit not run yet")
    summary = demo["supported_summary"]
    assert summary["runs"] == 12
    assert summary["by_level"] == {"1": 4, "2": 4, "3": 4}
    assert summary["distinct_programs"] >= 8
    assert summary["programs_parsed_correctly"] == 12
    assert demo["gate"]["passed"] is True
    assert demo["out_of_domain_summary"]["all_rejected_with_exit_4"] is True


def test_cli_audit_covers_required_program_families():
    demo = _artifact("task6m1_demo_cli_audit.json")
    if demo is None:
        pytest.skip("Task 6M.1 CLI audit not run yet")
    programs = set(demo["supported_summary"]["programs"])
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    assert programs <= set(EXPECTED_QUERY_TYPES), programs - set(EXPECTED_QUERY_TYPES)
    families = demo.get("supported_summary", {})
    assert families.get("runs") == 12
    # the audit must contain at least one level-3 compositional program
    assert any(row["level"] == 3 for row in demo["supported_runs"])


# ---------------------------------------------------------------- 19-22 no scope creep


def test_no_4b_upgrade_in_6m1():
    config = (REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml").read_text(encoding="utf-8")
    for value in re.findall(r"qwen_model_id:\s*(\S+)", config):
        assert "2B" in value, value
    for name in TASK6M1_SOURCES:
        text = _code(name)
        assert "Qwen3-VL-4B" not in text, name
        for match in re.finditer(r"qwen_model_id[\"']?\s*[:=]\s*[\"']([^\"']+)", text):
            assert "2B" in match.group(1), name


def test_no_model_family_imgsz_or_loss_change():
    preflight = _artifact("task6m1_resume_preflight.json")
    if preflight is not None:
        assert preflight["frozen_config_checks"]["imgsz"]["stored"] == 640
        assert preflight["frozen_config_checks"]["batch"]["stored"] == 16
        assert preflight["frozen_config_checks"]["epochs"]["stored"] == 80
        assert preflight["frozen_config_checks"]["patience"]["stored"] == 15
    resume = _code("task6m1_resume_train.py")
    for marker in ("YOLO26l", "yolo26l", "yolo26x", "YOLO26x", "loss_weights", "custom_loss", "imgsz=1280",
                   "imgsz = 1280", "tile_size", "sliding_window"):
        assert marker not in resume, f"the continuation must not change the configuration ({marker})"
    summary = _artifact("task6m1_training_summary.json")
    if summary is not None:
        config = summary["continuation"]["frozen_config"]
        assert config["model"] == "YOLO26m-seg"
        assert config["imgsz"] == 640
        assert config["batch"] == 16
        assert config["max_epoch_horizon"] == 80
        assert config["patience"] == 15


def test_no_gui_in_6m1():
    paths = [SCRIPTS / name for name in TASK6M1_SOURCES] + [REPO_ROOT / "predict_structured.py"]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for marker in ("tkinter", "PyQt", "gradio", "streamlit", "cv2.imshow", "flask", "fastapi"):
            assert marker not in text, f"{path.name} must not build a GUI ({marker})"


def test_no_new_dataset_or_download():
    manifests = json.loads((EVAL / "task6m_environment_manifest.json").read_text(encoding="utf-8"))
    weights = manifests["pretrained_weights"]
    assert weights["yolo26m-seg.pt"]["sha256"].startswith("16b636f04e8fb6a3")
    assert set(weights) == {"yolo26m-seg.pt", "yolo26s-seg.pt"}
    for name in TASK6M1_SOURCES:
        text = _code(name)
        for marker in ("urlretrieve", "requests.get", "Invoke-WebRequest", "wget ", "hf_hub_download",
                       "snapshot_download"):
            assert marker not in text, f"{name} must not download anything ({marker})"
