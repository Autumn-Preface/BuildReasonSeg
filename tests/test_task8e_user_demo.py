"""Presentation/assembly integration tests. No model checkpoint or inference is used."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
APP_SOURCE = REPO / "demo/user_demo_v1"
CANONICAL = REPO / "delivery_src/BuildReasonSeg_Advisor_RC1"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(APP_SOURCE))
sys.path.insert(0, str(CANONICAL))
from _ui import backend as b, environment as env
from scripts import build_task8e_user_demo as build

spec = importlib.util.spec_from_file_location("task8e_gui", APP_SOURCE / "BuildReasonSeg_Demo.py")
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)
spec = importlib.util.spec_from_file_location("task8e_frozen_cli", CANONICAL / "predict.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
from buildreasonseg.language.frontend import DeterministicFallbackFrontend
from buildreasonseg.language.registry import ParsedProgram
from buildreasonseg.language.validator import validate


@pytest.fixture(autouse=True)
def private_environment(monkeypatch, tmp_path):
    import torch
    from buildreasonseg.runtime import pipeline
    def forbidden(*args, **kwargs):
        pytest.fail("Real model/checkpoint execution is forbidden in Demo fake tests")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(pipeline, "predict_one", forbidden)
    for cls, methods in ((pipeline.DetectorRuntime, ("load", "detect_global")),
                         (pipeline.Sam2Runtime, ("load", "encode")),
                         (pipeline.Db1Runtime, ("load", "forward")),
                         (pipeline.ProgramHeadRuntime, ("load", "parse", "generate_suggestion"))):
        for name in methods:
            monkeypatch.setattr(cls, name, forbidden)
    for key in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "PYTHONDONTWRITEBYTECODE", "PYTHONUTF8", "PYTHONIOENCODING"):
        monkeypatch.setenv(key, "test-private")
    env.configure_environment(tmp_path / "test-cache-owner")


class FakeLanguage:
    def __init__(self, program="largest_to_above_to_nearest", available=True, fails=False, suggestion=None):
        self.program, self.is_available, self.fails = program, available, fails
        self.suggestion = suggestion or {"status": "SUGGESTION", "program": "largest_to_above_to_nearest", "display_command": "最大建筑上方最近的建筑"}
        self.parse_calls, self.suggestion_calls = 0, 0
    def available(self):
        return self.is_available, "fake availability"
    def parse(self, prompt):
        self.parse_calls += 1
        if self.fails:
            raise RuntimeError("synthetic Qwen unavailable")
        return NS(program=self.program, confidence=0.9, score_source="fake-only", to_dict=lambda: {"top5": []})
    def generate_suggestion(self, prompt, program):
        self.suggestion_calls += 1
        return json.dumps(self.suggestion, ensure_ascii=False)


def fake_adapter(language):
    adapter = b.EngineAdapter.__new__(b.EngineAdapter)
    adapter.runtime = NS(language_runtime=lambda: language)
    adapter.Parsed, adapter.validate = ParsedProgram, validate
    adapter.Fallback, adapter.helpers = DeterministicFallbackFrontend, helpers
    return adapter


@pytest.mark.parametrize("program", list(b.DIRECTIONS))
def test_direct_requires_confirmation_and_qwen(program):
    lang = FakeLanguage(program)
    confirmations = []
    parsed, info, suggestion = fake_adapter(lang).parse("最大建筑上方最近的建筑", lambda k, m: confirmations.append((k, m)) or True)
    assert lang.parse_calls == 1 and lang.suggestion_calls == 0
    assert parsed.program == program and info["user_confirmation"] == "Y"
    assert confirmations[0][0] == "direct" and b.interpretation(program) in confirmations[0][1]
    assert suggestion is None


def test_direct_rejection_prevents_inference():
    lang = FakeLanguage()
    with pytest.raises(b.DemoFailure) as exc:
        fake_adapter(lang).parse(b.EXAMPLES[2], lambda *_: False)
    assert exc.value.code == "E901" and lang.parse_calls == 1


@pytest.mark.parametrize("accepted", [True, False])
def test_suggestion_uses_existing_helper_validator_and_confirmation(accepted):
    lang = FakeLanguage("largest")
    calls = []
    confirm = lambda kind, text: calls.append((kind, text)) or accepted
    if accepted:
        parsed, info, display = fake_adapter(lang).parse("fake unsupported", confirm)
        assert parsed.program == "largest_to_above_to_nearest" and info["suggestion_used"] and info["user_confirmation"] == "Y"
        assert display == lang.suggestion["display_command"]
    else:
        with pytest.raises(b.DemoFailure):
            fake_adapter(lang).parse("fake unsupported", confirm)
    assert lang.parse_calls == lang.suggestion_calls == 1
    assert calls[0][0] == "suggestion" and "原始指令未能直接映射" in calls[0][1]


@pytest.mark.parametrize("suggestion", [{"status": "NO_SAFE_SUGGESTION"},
    {"status": "SUGGESTION", "program": "smallest", "display_command": "unsupported"}])
def test_unsafe_suggestion_never_confirmed_or_executed(suggestion):
    calls = []
    with pytest.raises(b.DemoFailure):
        fake_adapter(FakeLanguage("largest", suggestion=suggestion)).parse("text", lambda *x: calls.append(x) or True)
    assert not calls


@pytest.mark.parametrize("available,fails,reason", [(False, False, "qwen_asset_missing"), (True, True, "qwen_runtime_failed")])
def test_fallback_requires_consent_and_confirmed_supported_interpretation(available, fails, reason):
    lang = FakeLanguage(available=available, fails=fails)
    calls = []
    parsed, info, _ = fake_adapter(lang).parse(b.EXAMPLES[2], lambda k, m: calls.append((k, m)) or True)
    assert [x[0] for x in calls] == ["fallback", "direct"]
    assert parsed.source == "deterministic_fallback" and validate(parsed).supported
    assert info["language_mode"] == "fallback" and info["fallback_reason"] == reason
    assert lang.parse_calls == int(available)


@pytest.mark.parametrize("fails,code", [(False, "E901"), (True, "E502")])
def test_fallback_rejection_keeps_frozen_error_code(fails, code):
    with pytest.raises(b.DemoFailure) as exc:
        fake_adapter(FakeLanguage(available=fails, fails=fails)).parse(b.EXAMPLES[0], lambda *_: False)
    assert exc.value.code == code


def test_fallback_second_confirmation_rejection_stops():
    with pytest.raises(b.DemoFailure):
        fake_adapter(FakeLanguage(available=False)).parse(b.EXAMPLES[1], lambda k, m: k == "fallback")


def test_unsupported_fallback_rejected_by_validator():
    calls = []
    with pytest.raises(b.DemoFailure):
        fake_adapter(FakeLanguage(available=False)).parse("不存在的关系", lambda k, m: calls.append(k) or True)
    assert calls == ["fallback"]


def test_engine_request_automatic_defaults_single_predict_call():
    from buildreasonseg.runtime.pipeline import PipelineRequest
    adapter = fake_adapter(FakeLanguage())
    adapter.Request = PipelineRequest
    calls = []
    adapter.predict = lambda runtime, request: calls.append(request) or "fake_result"
    out = adapter.run(Path("unlocked_fake.png"), b.EXAMPLES[2], lambda *_: True, lambda *_: None)
    assert out[0] == "fake_result" and len(calls) == 1
    request = calls[0]
    assert request.reference_id is None and not request.inspect_proposals
    assert request.model == "buildreasonseg_advisor" and request.device == "auto" and request.alpha == 0.45


def test_engine_predict_not_called_without_confirmation():
    adapter = fake_adapter(FakeLanguage())
    adapter.predict = lambda *_: pytest.fail("inference before confirmation")
    with pytest.raises(b.DemoFailure):
        adapter.run(Path("fake.png"), b.EXAMPLES[0], lambda *_: False, lambda *_: None)


def fake_result(root, status="SUCCESS"):
    run = root / "_engine/inference/output/fake_run"
    for directory in ("masks", "overlays", "diagnostics"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    mask, overlay = run / "masks/fake_mask.png", run / "overlays/fake_overlay.png"
    if status == "SUCCESS":
        Image.new("L", (40, 20), 255).save(mask)
        Image.new("RGB", (40, 20), "red").save(overlay)
    payload = {"status": status, "validity_scope": "RUNTIME_STRUCTURAL_ONLY", "semantic_status": "NOT_EVALUATED",
               "raw_proposals": ["private"], "output_paths": {"diagnostics": str(run / "diagnostics")}}
    if status == "SUCCESS":
        payload["output_paths"].update({"mask": str(mask), "overlay": str(overlay)})
    return NS(ok=status == "SUCCESS", result_payload=payload, error_code="E404" if status != "SUCCESS" else None)


def test_results_copy_only_user_artifacts_and_preserve_engine(tmp_path):
    result = fake_result(tmp_path)
    before = build.inventory(tmp_path / "_engine")
    out = b.publish_result(tmp_path, Path("fake.png"), "text", next(iter(b.DIRECTIONS)), {"language_mode": "qwen"}, result=result, elapsed=3.4)
    folder = Path(out["directory"])
    assert {p.name for p in folder.iterdir()} == {"mask.png", "overlay.png", "result_summary.json"}
    assert build.inventory(tmp_path / "_engine") == before
    assert out["summary"]["summary_kind"] == "USER_DEMO_SUMMARY_NOT_ENGINE_RESULT"
    assert out["summary"]["semantic_status"] == "NOT_EVALUATED" and "raw_proposals" not in out["summary"]
    assert out["summary"]["engine_run_root"] == "_engine/inference/output/fake_run"


def test_failed_run_has_no_fake_mask_or_overlay(tmp_path):
    out = b.publish_result(tmp_path, Path("fake.png"), "text", None, {}, result=fake_result(tmp_path, "FAILED"))
    assert {p.name for p in Path(out["directory"]).iterdir()} == {"result_summary.json"}
    assert out["summary"]["mask_file"] is out["summary"]["overlay_file"] is None


def test_language_cancel_saves_failure_only(tmp_path):
    out = b.publish_result(tmp_path, Path("fake.png"), "text", None, {}, error=b.DemoFailure("E901"))
    assert {p.name for p in Path(out["directory"]).iterdir()} == {"result_summary.json"}


def test_result_roots_collision_safe_no_overwrite(tmp_path):
    clock = dt.datetime(2026, 10, 7, 12, 0, 0)
    a = b.reserve_result(tmp_path, Path("fake.png"), clock)
    (a / "existing.txt").write_text("preserve")
    c = b.reserve_result(tmp_path, Path("fake.png"), clock)
    assert a != c and c.name.endswith("_001") and (a / "existing.txt").read_text() == "preserve"


def test_publish_rejects_outputs_outside_own_engine(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    r = fake_result(other)
    with pytest.raises(b.DemoFailure):
        b.publish_result(tmp_path, Path("fake.png"), "text", None, {}, result=r)


def test_worker_serializes_duplicate_run_and_confirmation():
    worker = b.Worker()
    done = threading.Event()
    replies = []
    assert worker.start(lambda: (replies.append(worker.confirm("direct", "question")), done.set()))
    assert not worker.start(lambda: pytest.fail("duplicate run"))
    kind, request = worker.events.get(timeout=5)
    assert kind == "confirmation" and not request.accepted
    request.accepted = True
    request.event.set()
    assert done.wait(5)
    worker.thread.join(5)
    assert replies == [True] and not worker.busy


def test_friendly_errors_do_not_show_engine_trace_or_diagnostic_controls():
    for code in b.ERROR_MESSAGES:
        message = b.safe_detail(code, "C:/private/trace.py\n--reference-id 5\nTraceback")
        assert "Traceback" not in message and "--reference-id" not in message and "C:/" not in message
    assert "人工核验" in b.SUCCESS_TEXT and "识别正确" not in b.SUCCESS_TEXT and "分割正确" not in b.SUCCESS_TEXT


def test_preview_same_image_contract_display_only(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "bind_engine", lambda root: CANONICAL)
    path = tmp_path / "unlocked_fake.png"
    Image.new("RGBA", (800, 200), (30, 50, 70, 100)).save(path)
    before = build.identity(path)
    image = b.preview_image(tmp_path, path, (400, 300))
    assert image.size == (400, 100) and image.mode == "RGB" and build.identity(path) == before


def test_preview_does_not_add_grayscale_support(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "bind_engine", lambda root: CANONICAL)
    path = tmp_path / "gray.png"
    Image.new("L", (20, 20)).save(path)
    with pytest.raises(Exception) as e:
        b.preview_image(tmp_path, path)
    assert e.value.code == "E203"


def test_cache_directories_exist_before_import_and_are_private(tmp_path):
    def importer(name):
        for key in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
            assert Path(__import__('os').environ[key]).is_dir()
            assert Path(__import__('os').environ[key]).is_relative_to(tmp_path)
        ver = {"ultralytics": "8.4.164", "torch": "2.13.0", "transformers": "5.17.0", "numpy": "2.0", "PIL": "12.0", "cv2": "4.11", "scipy": "1.15"}.get(name, "1.0")
        return NS(__version__=ver, Qwen3VLForConditionalGeneration=object())
    report = env.probe_runtime(tmp_path, importer)
    assert report["ok"] and report["model_calls"] == 0
    assert report["cache_environment"]["HF_HUB_OFFLINE"] == report["cache_environment"]["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("name", ["torch", "ultralytics", "transformers", "numpy", "PIL", "cv2", "scipy"])
def test_incompatible_runtime_has_friendly_failure(tmp_path, name):
    def importer(module):
        if module == name:
            raise ImportError("raw private traceback")
        return NS(__version__="8.4.164" if module == "ultralytics" else "99.0", Qwen3VLForConditionalGeneration=object())
    report = env.probe_runtime(tmp_path, importer)
    assert not report["ok"] and "环境检查未通过" in report["message"]
    assert "raw private" not in report["message"]


def test_wrong_ultralytics_is_rejected(tmp_path):
    report = env.probe_runtime(tmp_path, lambda name: NS(__version__="8.3.0" if name == "ultralytics" else "99.0", Qwen3VLForConditionalGeneration=object()))
    assert not report["ok"] and "8.4.164" in report["message"]


def test_python_discovery_order_and_fallback(tmp_path):
    (tmp_path / "_ui").mkdir()
    candidates = [tmp_path / "preferred.exe", tmp_path / "runtime/python.exe", tmp_path / "configured.exe", tmp_path / "path.exe"]
    for p in candidates:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake interpreter file")
    (tmp_path / "_ui/runtime.json").write_text(json.dumps({"validated_python": str(candidates[2])}))
    attempts = []
    result = env.discover_python(tmp_path, lambda p: attempts.append(p) or p == str(candidates[2]),
                                env={"BUILDREASONSEG_PYTHON": str(candidates[0])}, which=lambda _: str(candidates[3]))
    assert attempts == list(map(str, candidates[:3])) and result["source"] == "validated_machine_environment"
    assert env.python_candidates(tmp_path, env={}, which=lambda _: str(candidates[3]))[0][0] == "bundled_future"


def test_python_discovery_failure_is_friendly(tmp_path):
    result = env.discover_python(tmp_path, lambda _: False, env={}, which=lambda _: None)
    assert not result["ok"] and "不会安装依赖" in result["message"]


def test_ast_dependency_closure_relative_imports_and_initializers():
    blobs = {"predict.py": b'from pkg import sub\n', "pkg/__init__.py": b'from . import needed\n',
             "pkg/sub.py": b'from .needed import function\n', "pkg/needed.py": b'function = None\n', "unused.py": b''}
    paths, trace = build.dependency_closure(blobs, ("predict",))
    assert paths == ["pkg/__init__.py", "pkg/needed.py", "pkg/sub.py", "predict.py"]
    assert "unused" not in trace["modules"]


@pytest.fixture
def fake_source(tmp_path):
    source = tmp_path / "Advisor"
    source.mkdir()
    blobs = {"predict.py": b'"""fake frozen source"""\n', "buildreasonseg/__init__.py": b''}
    for name, content in blobs.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    (source / "model").mkdir()
    (source / "model/weights.pt").write_bytes(b'fake-heavy-byte-identity' * 2000)
    (source / "logs").mkdir()
    (source / "logs/historical.txt").write_text("do not copy")
    (source / "tests").mkdir()
    (source / "tests/test_old.py").write_text("do not copy")
    return source, blobs


def assemble_fake(tmp_path, fake_source):
    source, blobs = fake_source
    return build.assemble(source, tmp_path / "staging", tmp_path / "final", blobs, list(blobs),
                          ["model/weights.pt"], build.inventory(source), APP_SOURCE, sys.executable)


def test_assembly_preserves_old_source_assets_and_clean_state(tmp_path, fake_source):
    source, _ = fake_source
    before = build.inventory(source)
    assembled = assemble_fake(tmp_path, fake_source)
    package = Path(assembled["staging"])
    assert build.inventory(source) == before and build.forbidden_audit(package)["pass"]
    assert build.identity(source / "model/weights.pt") == build.identity(package / "_engine/model/weights.pt")
    for name in ("results", "_engine/inference/output", "_engine/logs"):
        assert not list((package / name).iterdir())
    assert not (package / "_engine/tests").exists() and not (package / "_engine/logs/historical.txt").exists()
    assert gui.package_check(package, full=True)["ok"]


def test_assembly_rejects_existing_delivery_without_touching_it(tmp_path, fake_source):
    final = tmp_path / "final"
    final.mkdir()
    (final / "preserve.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        assemble_fake(tmp_path, fake_source)
    assert (final / "preserve.txt").read_text() == "preserve" and not (tmp_path / "staging").exists()


def test_assembly_rejects_source_drift_before_write(tmp_path, fake_source):
    source, blobs = fake_source
    before = build.inventory(source)
    (source / "predict.py").write_text("unknown drift")
    with pytest.raises(ValueError):
        build.assemble(source, tmp_path / "staging", tmp_path / "final", blobs, list(blobs), ["model/weights.pt"], before, APP_SOURCE, sys.executable)
    assert not (tmp_path / "staging").exists()


@pytest.mark.parametrize("name", ["handoff/x.txt", "governance/x.yaml", "evaluation/a.png", "tests/test_x.py", ".git/config",
                                 "train.py", "evaluate.py", "prepare_dataset.py", "task8c_report.md", "task8d_report.md"])
def test_forbidden_package_content_detected(tmp_path, fake_source, name):
    out = assemble_fake(tmp_path, fake_source)
    p = Path(out["staging"]) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("forbidden")
    assert not build.forbidden_audit(Path(out["staging"]))["pass"]


def test_whole_folder_move_keeps_relative_source_model_result_paths(tmp_path, fake_source):
    out = assemble_fake(tmp_path, fake_source)
    staging, final = Path(out["staging"]), Path(out["final"])
    source, _ = fake_source
    moved = build.finalize(source, staging, final, out["manifest"])
    assert moved["pass"] and not moved["models_copied_again"]
    assert gui.package_check(final, full=True)["ok"]
    assert (final / "_engine/model/weights.pt").is_file() and not staging.exists()
    result = b.reserve_result(final, Path("fake.png"))
    assert result.is_relative_to(final / "results")


def test_user_docs_are_non_developer_and_all_sections_present():
    text = (APP_SOURCE / "使用说明.md").read_text(encoding="utf-8")
    for forbidden in ("Task", "Codex", "Supervisor", "handoff", "governance"):
        assert forbidden.lower() not in text.lower()
    assert sum(line.startswith("## ") for line in text.splitlines()) == 14
    assert "流程完成”不等于“语义结果一定正确" in text and "跨电脑免配置" in text


def test_no_demo_root_hardcoding_or_diagnostic_ui_controls():
    for relative in ("BuildReasonSeg_Demo.py", "_ui/backend.py", "_ui/environment.py", "_ui/launch.ps1"):
        text = (APP_SOURCE / relative).read_text(encoding="utf-8-sig")
        assert r"C:\D\DeepSeekHarness" not in text and "BuildReasonSeg_Demo_V1" not in text
        for forbidden in ("--reference-id", "--inspect-proposals", "threshold", "ranking"):
            assert forbidden not in text
    assert (APP_SOURCE / "_ui/launch.ps1").read_bytes().startswith(b'\xef\xbb\xbf')


@pytest.fixture
def app(tmp_path):
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    app = gui.DemoApp(root, demo_root=tmp_path, start_check=False)
    yield app
    if app.worker.thread:
        app.worker.thread.join(5)
    assert not app.worker.busy
    app.close()


def test_tk_quick_fill_buttons_only_populate_text(app):
    for i, button in enumerate(app.example_buttons):
        button.invoke()
        assert app.prompt.get("1.0", "end").strip() == b.EXAMPLES[i]
        assert app.worker.thread is None and app.published is None


def test_tk_busy_disables_duplicate_run_and_close(app, monkeypatch):
    from tkinter import messagebox
    done = threading.Event()
    warnings = []
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **kw: warnings.append(a))
    assert app.worker.start(lambda: done.wait(5))
    app.set_busy(True)
    assert app.run_button.instate(["disabled"])
    assert all(button.instate(["disabled"]) for button in app.example_buttons)
    app.run_button.invoke()
    app.close()
    assert warnings and app.window.winfo_exists()
    done.set()
    app.worker.thread.join(5)


def test_tk_one_run_one_fake_predict_and_result_presentation(app, tmp_path, monkeypatch):
    from tkinter import messagebox
    calls, confirmations = [], []
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **kw: confirmations.append(a) or True)
    result = fake_result(tmp_path)
    class Fake:
        def __init__(self, root):
            pass
        def run(self, image, prompt, confirm, progress):
            assert confirm("direct", "fake-only interpreted task")
            calls.append((image, prompt))
            return result, "largest_to_above_to_nearest", {"language_mode": "qwen"}
    app.adapter_factory = Fake
    app.image_path = tmp_path / "unlocked_fake.png"
    Image.new("RGB", (40, 20)).save(app.image_path)
    app.fill_example(2)
    app.run_button.invoke()
    app.run_button.invoke()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        app.window.update()
        if app.published and not app.worker.busy:
            break
        time.sleep(0.01)
    app.window.update()
    assert len(calls) == len(confirmations) == 1
    assert app.published["summary"]["runtime_status"] == "SUCCESS"
    assert "人工核验" in app.status.get() and "语义正确性：未自动验证" in app.summary.get()
    assert app.mask_button.instate(["!disabled"])
    assert app.photos["overlay"]

