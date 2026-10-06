"""Dedicated fake-only Task 8C safety gates. Never invoke a real model/candidate."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("task8c_final_demo_runner")
evaluator = load("task8c_final_demo_evaluate")


def test_candidate_order():
    assert [c.relation for c in runner.CANDIDATES] == ["right", "left", "above", "below"]


def test_exact_sample_ids():
    assert [c.sample_id for c in runner.CANDIDATES] == [
        "buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91",
        "buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3",
        "buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314",
        "buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450"]


def test_exact_raster_hashes():
    assert [c.image_sha256 for c in runner.CANDIDATES] == [
        "1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2",
        "eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38",
        "0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd",
        "c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7"]


def test_exact_prompts():
    assert [c.prompt for c in runner.CANDIDATES] == [
        "最大建筑右侧最近的建筑", "最大建筑左侧最近的建筑", "最大建筑上方最近的建筑", "最大建筑下方最近的建筑"]


def test_exact_programs():
    assert [c.expected_program for c in runner.CANDIDATES] == [
        "largest_to_right_of_to_nearest", "largest_to_left_of_to_nearest",
        "largest_to_above_to_nearest", "largest_to_below_to_nearest"]


@pytest.mark.parametrize("candidate", runner.CANDIDATES)
def test_argv_whitelist(candidate):
    argv = runner.formal_argv(candidate)
    assert argv == [sys.executable, "-B", str(runner.EXTERNAL / "predict.py"),
                    "--image", candidate.image, "--prompt", candidate.prompt, "--confirm-command"]
    assert "--reference-id" not in argv
    assert "--inspect-proposals" not in argv
    assert "--model" not in argv and "--device" not in argv and "--alpha" not in argv


def fake_child(tmp_path, kind, observed, *, fragmented=False):
    marker = {"direct": runner.DIRECT_PROMPT, "suggestion": runner.SUGGESTION_PROMPT,
              "fallback": runner.FALLBACK_PROMPT}[kind]
    text = "[语言] Qwen / ProgramHead  置信度 0.876\n"
    text += f"[解析] unsupported   ({observed if kind == 'direct' else 'unsupported'})\n"
    if kind == "suggestion":
        human = next((k for k, v in runner.DISPLAY_TO_PROGRAM.items() if v == observed), "unknown")
        text += "建议程序：" + human + "\n"
    text += marker
    code = ("import sys,time\n" + f"payload={text!r}.encode('utf-8')\n" +
            ("for b in payload:\n sys.stdout.buffer.write(bytes([b]));sys.stdout.buffer.flush()\n" if fragmented else
             "sys.stdout.buffer.write(payload);sys.stdout.buffer.flush()\n") +
            "answer=sys.stdin.buffer.readline().decode().strip()\n" +
            "print('ANSWER='+answer,flush=True)\n" +
            "sys.exit(0 if answer=='Y' else 19)\n")
    script = tmp_path / "fake_child.py"
    script.write_text(code, encoding="utf-8")
    return [sys.executable, "-B", str(script)]


@pytest.mark.parametrize("kind,observed,answer,label", [
    ("direct", "largest_to_right_of_to_nearest", "Y", "DIRECT_CORRECT"),
    ("direct", "largest_to_left_of_to_nearest", "N", "LANGUAGE_ERROR_SUPPORTED_WRONG"),
    ("suggestion", "largest_to_right_of_to_nearest", "Y", "SUGGESTION_CORRECT"),
    ("suggestion", "largest_to_below_to_nearest", "N", "SUGGESTION_WRONG"),
    ("fallback", None, "N", "LANGUAGE_RUNTIME_ERROR_OR_FALLBACK_REQUEST")])
def test_actual_pipe_interaction(tmp_path, kind, observed, answer, label):
    result = runner.drive_child(fake_child(tmp_path, kind, observed, fragmented=True), tmp_path,
                               runner.CANDIDATES[0].expected_program, timeout=5)
    assert result["driver_decision"][0]["answer"] == answer
    assert result["language_status"] == label
    assert f"ANSWER={answer}" in result["stdout"]
    assert result["exit_code"] == (0 if answer == "Y" else 19)


@pytest.mark.parametrize("kind", ["direct", "suggestion"])
def test_missing_or_near_match_program_rejected(kind):
    for bad in (None, "largest_to_right_of_to_nearest ", "largest_to_right_of_to_nearest_extra"):
        assert runner.decide(kind, bad, runner.CANDIDATES[0].expected_program)[0] == "N"


def test_timeout_silent_child(tmp_path):
    child = tmp_path / "silent.py"
    child.write_text("import time\ntime.sleep(20)\n", encoding="utf-8")
    start = time.monotonic()
    result = runner.drive_child([sys.executable, "-B", str(child)], tmp_path, "expected", timeout=0.3)
    assert result["timed_out"] and result["exit_code"] != 0
    assert time.monotonic() - start < 5
    assert runner.CASE_TIMEOUT_SECONDS == 900.0


def test_persistent_journal_forbids_second_invocation(tmp_path):
    runner.claim_once(tmp_path, "a"*40)
    with pytest.raises(FileExistsError):
        runner.claim_once(tmp_path, "a"*40)


def test_failed_case_continues_without_retry(tmp_path):
    journal = runner.claim_once(tmp_path, "a"*40)
    attempts = []
    def execute(candidate):
        attempts.append(candidate.relation)
        return {"runtime_status": "FAILED" if candidate.relation == "right" else "SUCCESS"}
    rows = runner.execute_sequence(runner.CANDIDATES, tmp_path, journal, execute)
    assert attempts == ["right", "left", "above", "below"]
    assert len(rows) == 4 and journal["all_child_processes_exited"]
    with pytest.raises(ValueError, match="consumed"):
        runner.execute_sequence(runner.CANDIDATES, tmp_path, journal, execute)
    assert len(attempts) == 4


def test_crash_attempt_remains_consumed(tmp_path):
    journal = runner.claim_once(tmp_path, "a"*40)
    def crash(candidate):
        raise RuntimeError("fake harness crash")
    with pytest.raises(RuntimeError):
        runner.execute_sequence(runner.CANDIDATES, tmp_path, journal, crash)
    saved = json.loads((tmp_path / "invocation.json").read_text())
    assert saved["case_attempt_order"] == ["right"] and not saved["all_child_processes_exited"]
    with pytest.raises(FileExistsError):
        runner.claim_once(tmp_path, "a"*40)


def test_runner_has_no_gt_access():
    source = (ROOT / "scripts/task8c_final_demo_runner.py").read_text(encoding="utf-8")
    for forbidden in ("native_vector", "target_component_id", "reference_component_ids",
                      "datasets/", "read_tile_cache", "GT_IDS"):
        assert forbidden not in source


def test_evaluator_cannot_execute_models_or_predict():
    source = (ROOT / "scripts/task8c_final_demo_evaluate.py").read_text(encoding="utf-8")
    for forbidden in ("subprocess", "predict.py", "torch", "ultralytics", "transformers",
                      "PredictRuntime", "detect_global", "run_core_chain"):
        assert forbidden not in source
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert calls.index("verify_runtime") < calls.index("load_truth")


@pytest.mark.parametrize("filename", ["task8c_final_demo_runner.py", "task8c_final_demo_evaluate.py"])
def test_no_delete_or_clear(filename):
    tree = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in ("unlink", "rmtree", "remove", "removedirs", "rmdir")


def test_preexisting_output_cannot_be_overwritten(tmp_path):
    old = tmp_path / "legacy.png"
    old.write_bytes(b"old")
    before = runner.inventory(tmp_path)
    (tmp_path / "new_run").mkdir()
    (tmp_path / "new_run/result.json").write_bytes(b"new")
    runner.verify_inventory(tmp_path, before, allow_new=True)
    old.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        runner.verify_inventory(tmp_path, before, allow_new=True)


@pytest.mark.parametrize("origin", [[0, 0], [-2, -3], [3, 2]])
def test_exact_context_mapping(origin):
    context_mask = np.arange(36).reshape(6, 6) % 3 == 0
    actual = evaluator.context_to_global(context_mask, {"origin": origin, "size": 6}, (8, 9))
    expected = np.zeros((8, 9), dtype=bool)
    for y in range(6):
        for x in range(6):
            if 0 <= y+origin[1] < 8 and 0 <= x+origin[0] < 9:
                expected[y+origin[1], x+origin[0]] = context_mask[y, x]
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("ref,target,status", [(1, 2, "CHAIN_IDENTITY_MATCH"),
    (2, 2, "REFERENCE_IDENTITY_MISMATCH"), (1, 1, "TARGET_IDENTITY_MISMATCH"),
    (2, 1, "REFERENCE_AND_TARGET_IDENTITY_MISMATCH")])
def test_identity_enums_without_quality_threshold(ref, target, status):
    label = np.array([[1, 1, 0], [0, 2, 2]])
    masks = {i: label == i for i in (1, 2)}
    result = evaluator.identity_audit(masks[ref], masks[target], masks, 1, 2)
    assert result["semantic_chain_status"] == status
    assert json.dumps(result, sort_keys=True) == json.dumps(
        evaluator.identity_audit(masks[ref], masks[target], masks, 1, 2), sort_keys=True)


def test_zero_overlap_has_no_identity_match():
    truth = np.array([[True, False]])
    assert evaluator.best_overlap(~truth, {1: truth}) == (None, 0.0)


def test_gt_embargo_before_four_exits(tmp_path):
    path = tmp_path / "runtime.json"
    runner.write_json(path, {"formal_runner_invocation_count": 1, "case_attempt_order": ["right"],
                             "all_child_processes_exited": False, "output_hashes_frozen": False, "cases": []})
    with pytest.raises(ValueError, match="embargo"):
        evaluator.verify_runtime({"frozen_runtime_identity": evaluator.identity(path)})


def test_frozen_runtime_mutation_rejected(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_bytes(b"original")
    frozen = evaluator.identity(path)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        evaluator.verify_runtime({"frozen_runtime_identity": frozen})


def test_binary_mask_not_repainted_or_resized(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(path)
    assert evaluator.read_binary(path, (1, 2)).tolist() == [[False, True]]
    with pytest.raises(ValueError):
        evaluator.read_binary(path, (2, 2))


@pytest.mark.parametrize("language,runtime,status", [(False, "FAILED", "LANGUAGE_FAILED"),
    (True, "FAILED", "RUNTIME_FAILED"), (True, "SUCCESS", "NOT_EVALUABLE_MISSING_ARTIFACT")])
def test_failed_and_missing_artifact_semantics(language, runtime, status):
    row = {"language_expected_program_executed": language, "runtime_status": runtime,
           "diagnostics_path": None}
    audit, ref = evaluator.audit_case(row, {1: np.ones((2, 2), dtype=bool)},
                                     {"canonical_reference_id": 1, "canonical_target_id": 1})
    assert audit["semantic_chain_status"] == status and ref is None


def test_failure_is_in_aggregate():
    cases = [{"language_expected_program_executed": True, "runtime_status": "SUCCESS",
              "gt_audit": {"semantic_chain_status": "CHAIN_IDENTITY_MATCH", "target_iou": 0.2, "target_dice": 1/3}},
             {"language_expected_program_executed": True, "runtime_status": "FAILED",
              "gt_audit": {"semantic_chain_status": "RUNTIME_FAILED"}}]
    result = evaluator.aggregate(cases)
    assert result["cases_attempted"] == 2 and result["runtime_failed"] == 1
    assert result["mean_target_iou_over_runtime_success"] == 0.2


def test_frozen_output_artifact_mutation_rejected(tmp_path):
    raster = tmp_path / "fake.tif"
    Image.new("RGB", (512, 512)).save(raster)
    artifact = tmp_path / "output.png"
    Image.new("L", (512, 512)).save(artifact)
    transcript = tmp_path / "stdout.txt"
    transcript.write_text("fake", encoding="utf-8")
    locks, cases = [], []
    for relation in evaluator.EXPECTED_ORDER:
        lock = {"relation": relation, "image": str(raster), "image_sha256": evaluator.identity(raster)["sha256"]}
        locks.append(lock)
        cases.append({**lock, "run_root": None, "artifacts": [evaluator.identity(artifact)],
                      "transcript_identity": evaluator.identity(transcript)})
    runtime = {"formal_freeze_commit": "a"*40, "formal_runner_invocation_count": 1,
               "case_attempt_order": evaluator.EXPECTED_ORDER, "all_child_processes_exited": True,
               "output_hashes_frozen": True, "cases": cases}
    path = tmp_path / "runtime.json"
    runner.write_json(path, runtime)
    evidence = {"frozen_runtime_identity": evaluator.identity(path), "candidate_lock": locks,
                "formal_freeze_commit": "a"*40}
    assert evaluator.verify_runtime(evidence) == runtime
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        evaluator.verify_runtime(evidence)


def test_review_is_deterministic_and_shows_failure(tmp_path):
    path = tmp_path / "fake.tif"
    Image.new("RGB", (512, 512), (10, 20, 30)).save(path)
    row = {"image": str(path), "relation": "right", "sample_id": "fake_sample",
           "expected_program": "expected", "initial_program": "wrong", "suggested_program": None,
           "language_status": "DIRECT_CORRECT", "runtime_status": "FAILED", "exit_code": 40,
           "reference_id": None, "error_code": "E404", "error_reason": "fake failure"}
    audit = {"canonical_reference_id": 1, "canonical_target_id": 1, "semantic_chain_status": "RUNTIME_FAILED"}
    mask = np.zeros((512, 512), dtype=bool)
    mask[10:20, 10:20] = True
    first = evaluator.review_image(row, audit, None, {1: mask})
    second = evaluator.review_image(row, audit, None, {1: mask})
    assert first.size == (1536, 760) and first.tobytes() == second.tobytes()
    # Original full-size pixels remain intact; evidence is shown in a separate footer.
    assert first.getpixel((0, 42)) == (10, 20, 30)


def test_success_review_saved_overlay_writable_without_inference(tmp_path, monkeypatch):
    import builtins
    import io
    import subprocess

    image_path = tmp_path / "fake_success.tif"
    overlay_path = tmp_path / "saved_overlay.png"
    Image.new("RGB", (512, 512), (10, 20, 30)).save(image_path)
    overlay = np.full((512, 512, 3), (70, 80, 90), dtype=np.uint8)
    overlay[40:60, 40:60] = (210, 50, 50)
    Image.fromarray(overlay).save(overlay_path)
    overlay_before = overlay_path.read_bytes()
    overlay_identity = evaluator.identity(overlay_path)
    reference = np.zeros((512, 512), dtype=bool)
    reference[10:20, 10:20] = True
    target = np.zeros((512, 512), dtype=bool)
    target[40:60, 40:60] = True
    row = {"image": str(image_path), "overlay_path": str(overlay_path), "relation": "right",
           "sample_id": "fake_success", "expected_program": "expected", "initial_program": "expected",
           "suggested_program": None, "language_status": "DIRECT_CORRECT", "runtime_status": "SUCCESS",
           "exit_code": 0, "reference_id": 7, "error_code": None, "error_reason": None}
    audit = {"canonical_reference_id": 1, "canonical_target_id": 2,
             "selected_reference_iou_with_canonical_gt_reference": 1.0,
             "target_iou": 1.0, "target_dice": 1.0, "semantic_chain_status": "CHAIN_IDENTITY_MATCH"}
    execution_calls = []

    def forbid_execution(*args, **kwargs):
        execution_calls.append((args, kwargs))
        pytest.fail("review rendering attempted process/model execution")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbid_execution)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "ultralytics", "transformers", "predict", "buildreasonseg"):
            forbid_execution(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    original_open = Image.open
    overlay_reads = []

    def tracked_open(path, *args, **kwargs):
        if Path(path) == overlay_path:
            overlay_reads.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", tracked_open)
    first = evaluator.review_image(row, audit, reference, {1: reference, 2: target})
    second = evaluator.review_image(row, audit, reference, {1: reference, 2: target})
    assert overlay_reads == [str(overlay_path), str(overlay_path)]
    assert first.getpixel((1024+100, 42+100)) == (70, 80, 90)
    assert first.getpixel((1024+50, 42+50)) == (210, 50, 50)
    assert first.getpixel((1024+40, 42+40)) == (0, 255, 255)
    assert first.tobytes() == second.tobytes()
    first_png, second_png = io.BytesIO(), io.BytesIO()
    first.save(first_png, format="PNG")
    second.save(second_png, format="PNG")
    assert first_png.getvalue() == second_png.getvalue()
    assert overlay_path.read_bytes() == overlay_before
    assert evaluator.identity(overlay_path) == overlay_identity
    assert execution_calls == []
