"""UI orchestration around the unchanged RC1 APIs; no scientific implementation."""
from __future__ import annotations

import contextlib
import datetime as dt
import importlib.util
import json
import queue
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .environment import configure_environment
from .trace import TraceSession

EXAMPLES = ("最大建筑左侧最近的建筑", "最大建筑右侧最近的建筑", "最大建筑上方最近的建筑", "最大建筑下方最近的建筑")
DIRECTIONS = {"largest_to_left_of_to_nearest": "左侧", "largest_to_right_of_to_nearest": "右侧",
              "largest_to_above_to_nearest": "上方", "largest_to_below_to_nearest": "下方"}
SUCCESS_TEXT = "推理流程已完成。\n结果已生成，请结合原图人工核验目标是否正确。"
ERROR_MESSAGES = {
    "E101": "请输入完整指令，可使用四个示例。", "E102": "该指令不属于当前支持的四类任务，或您未接受建议。",
    "E201": "找不到图片，请重新选择。", "E202": "图片无法读取，请确认文件没有损坏。",
    "E203": "图片格式不受支持，请使用 RGB 光学遥感 PNG、JPG 或 TIFF 图片。",
    "E301": "模型文件不可用，请检查 Demo 文件夹是否完整。", "E302": "模型或运行环境不完整，请运行环境检查。",
    "E303": "模型文件校验未通过，请恢复完整的 Demo 文件夹。", "E304": "模型文件与当前程序不兼容。",
    "E401": "没有检测到可用建筑，请人工检查原图是否清晰、是否包含建筑。",
    "E402": "没有找到满足条件的参考建筑，当前图片无法继续分析。",
    "E403": "所需方向的建筑不在当前可分析范围内，无法继续。",
    "E404": "未生成满足结构与方向条件的目标 Mask，分析未完成。",
    "E501": "可用显存不足，请关闭占用显存的其他程序。", "E502": "运行遇到错误，未自动重试。",
    "E901": "您已取消，本次没有执行图像推理。"}


def interpretation(program: str) -> str:
    if program not in DIRECTIONS:
        raise ValueError("Unsupported program")
    return "参考对象：最大建筑\n方向关系：" + DIRECTIONS[program] + "\n目标关系：最近建筑"


def safe_detail(code: str, detail: str = "") -> str:
    # Do not send scientific internals, paths or a raw traceback to the default UI.
    if code == "E404" and "empty_target_mask" in detail:
        return "未生成非空的目标 Mask。"
    return ERROR_MESSAGES.get(code, "发生运行错误，请查看环境检查结果。")


class DemoFailure(Exception):
    def __init__(self, code: str, detail: str = ""):
        self.code, self.detail = code, safe_detail(code, detail)
        super().__init__(self.detail)


def bind_engine(root: Path):
    engine = Path(root).resolve() / "_engine"
    if not (engine / "buildreasonseg").is_dir():
        raise DemoFailure("E302")
    if str(engine) not in sys.path:
        sys.path.insert(0, str(engine))
    from buildreasonseg import paths
    if paths.project_root().resolve() != engine:
        raise DemoFailure("E304")
    return engine


def preview_image(root: Path, path: Path, bounds=(530, 360)):
    bind_engine(root)
    from buildreasonseg.runtime.imageio import load_image
    from PIL import Image
    loaded = load_image(path)             # exact accepted RGB/RGBA/uint16 image contract
    image = Image.fromarray(loaded.rgb)   # presentation-only copy, never changes the input
    image.thumbnail(bounds, Image.Resampling.LANCZOS)
    return image


class EngineAdapter:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        configure_environment(self.root)
        self.engine = bind_engine(self.root)
        from buildreasonseg.runtime.pipeline import PredictRuntime, PipelineRequest, predict_one
        from buildreasonseg.language.registry import ParsedProgram
        from buildreasonseg.language.validator import validate
        from buildreasonseg.language.frontend import DeterministicFallbackFrontend
        from buildreasonseg.models.package import resolve_model
        from buildreasonseg.errors import BuildReasonSegError
        spec = importlib.util.spec_from_file_location("demo_frozen_cli_helpers", self.engine / "predict.py")
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        resolution = resolve_model()
        if not resolution.ok:
            raise DemoFailure(resolution.error.code if resolution.error else "E301")
        resolution.package.verify_or_raise()
        self.runtime = PredictRuntime(device="auto", verbose=False)
        self.Request, self.predict = PipelineRequest, predict_one
        self.Parsed, self.validate = ParsedProgram, validate
        self.Fallback, self.Error = DeterministicFallbackFrontend, BuildReasonSegError
        self.helpers = helpers

    def observe_language(self, **metadata):
        observer = getattr(self, "observer", None)
        if observer is not None:
            self.language_trace.update(metadata)
            observer.emit(1, "RUNNING", dict(self.language_trace))

    def parse(self, prompt: str, confirm):
        language = self.runtime.language_runtime()
        available, _ = language.available()
        self.observe_language(original_prompt=prompt, qwen_available=available,
                              confidence_scope="模型评分，不是语义正确率")
        fallback_reason = None
        outcome = None
        if available:
            try:
                outcome = language.parse(prompt)    # always Qwen first; examples do not set a program
            except Exception:
                fallback_reason = "qwen_runtime_failed"
        else:
            fallback_reason = "qwen_asset_missing"
        if fallback_reason:
            self.observe_language(initial_program=None, initial_supported=None, language_mode="fallback",
                                  fallback_reason=fallback_reason)
            if not confirm("fallback", "语言模型当前不可用。\n是否使用有限兼容模式继续？"):
                raise DemoFailure("E502" if fallback_reason == "qwen_runtime_failed" else "E901")
            frontend = self.Fallback(fallback_reason)
            frontend.reason = fallback_reason
            parsed = frontend.parse(prompt)
            if not self.validate(parsed).supported:
                raise DemoFailure("E102")
            if not confirm("direct", "有限兼容模式的理解：\n" + interpretation(parsed.program) + "\n\n是否按此理解继续？"):
                raise DemoFailure("E901")
            return parsed, {"language_mode": "fallback", "fallback_reason": fallback_reason,
                            "original_prompt": prompt, "initial_program": parsed.program,
                            "initial_supported": True, "suggestion_used": False, "user_confirmation": "Y"}, None
        parsed = self.Parsed(program=outcome.program, source="qwen_program_head",
                             confidence=outcome.confidence, raw_text=prompt)
        info = {"language_mode": "qwen", "confidence": outcome.confidence,
                "score_source": outcome.score_source, "top5": outcome.to_dict()["top5"],
                "original_prompt": prompt, "initial_program": parsed.program,
                "initial_supported": self.validate(parsed).supported, "suggestion_used": False}
        self.observe_language(initial_program=parsed.program, initial_supported=self.validate(parsed).supported,
                              initial_has_nearest="_to_nearest" in parsed.program,
                              confidence=parsed.confidence, language_mode="qwen")
        if self.validate(parsed).supported:
            if not confirm("direct", "系统理解：\n" + interpretation(parsed.program) + "\n\n是否按此理解继续？"):
                raise DemoFailure("E901")
            info["user_confirmation"] = "Y"
            return parsed, info, None
        suggestion = self.helpers._suggestion(language, prompt, parsed.program)
        suggested = self.helpers._suggestion_program(suggestion)   # existing structured parser + Validator
        self.observe_language(suggestion=suggestion, suggested_program=None if suggested is None else suggested.program,
                              suggestion_validator_pass=suggested is not None)
        if suggested is None:
            raise DemoFailure("E102")
        display = suggestion.get("display_command") or interpretation(suggested.program)
        question = ("原始指令未能直接映射到当前支持的四类任务。\n\n系统建议理解为：\n“" + display +
                    "”\n\n" + interpretation(suggested.program) + "\n\n是否接受此建议并继续？")
        if not confirm("suggestion", question):
            raise DemoFailure("E102")
        info.update({"suggestion_used": True, "suggested_program": suggested.program,
                     "suggestion_display_command": display, "user_confirmation": "Y"})
        return suggested, info, display

    def run(self, image: Path, prompt: str, confirm, progress):
        parsed, info, suggestion = self.parse(prompt, confirm)
        observer = getattr(self, "observer", None)
        if observer is not None:
            self.language_trace.update(final_program=parsed.program, language_mode=info.get("language_mode"),
                                       user_confirmation=info.get("user_confirmation"))
            observer.emit(1, "COMPLETED", dict(self.language_trace))
        progress("正在检测建筑并进行空间推理，请稍候……", parsed.program)
        request = self.Request(image=Path(image), prompt=prompt, parsed=parsed,
                               parsed_info=info, suggestion=suggestion)
        # No override, inspect, model controls or retries. All defaults are the accepted automatic runtime.
        if observer is None:
            result = self.predict(self.runtime, request)
        else:
            result = self.predict(self.runtime, request, observer=observer)
        return result, parsed.program, info


def reserve_result(root: Path, image: Path, clock=None) -> Path:
    stamp = (clock or dt.datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    stem = re.sub(r"[^\w.-]", "_", Path(image).stem)[:80] or "image"
    base = Path(root).resolve() / "results"
    base.mkdir(parents=True, exist_ok=True)
    for index in range(1000000):
        name = stamp + "_" + stem + ("" if index == 0 else f"_{index:03d}")
        target = base / name
        try:
            target.mkdir()
        except FileExistsError:
            continue
        return target
    raise RuntimeError("Result directory reservation exhausted")


def publish_result(root: Path, image: Path, prompt: str, program: str | None, info: dict,
                   result=None, error=None, elapsed=0.0, clock=None, directory=None, observation_failed=False) -> dict:
    root = Path(root).resolve()
    directory = reserve_result(root, image, clock) if directory is None else Path(directory).resolve()
    if not directory.is_relative_to(root / "results"):
        raise DemoFailure("E502")
    payload = {} if result is None else result.result_payload
    success = bool(result is not None and result.ok and not observation_failed)
    summary = {"summary_kind": "USER_DEMO_SUMMARY_NOT_ENGINE_RESULT", "demo_version": "V2",
               "timestamp": (clock or dt.datetime.now().astimezone()).isoformat(), "input_image": str(Path(image).resolve()),
               "prompt": prompt, "interpreted_program": program, "language_mode": info.get("language_mode"),
               "runtime_status": "SUCCESS" if success else "FAILED", "validity_scope": payload.get("validity_scope"),
               "semantic_status": payload.get("semantic_status", "NOT_EVALUATED"),
               "elapsed_seconds": round(elapsed, 3), "mask_file": None, "overlay_file": None,
               "engine_run_root": None}
    summary["underlying_runtime_status"] = payload.get("status", "NOT_EXECUTED")
    summary["trace_directory"] = "trace"
    summary["observation_status"] = "FAILED" if observation_failed else "OK"
    paths = payload.get("output_paths", {})
    diag = paths.get("diagnostics")
    if diag:
        run_root = Path(diag).resolve().parent
        if not run_root.is_relative_to(root / "_engine/inference/output"):
            raise DemoFailure("E502")
        summary["engine_run_root"] = run_root.relative_to(root).as_posix()
    if success:
        sources = [Path(paths[k]).resolve() for k in ("mask", "overlay")]
        for source in sources:
            if not source.is_file() or not source.is_relative_to(root / "_engine/inference/output"):
                raise DemoFailure("E502")
        for key, source in zip(("mask", "overlay"), sources):
            with source.open("rb") as reader, (directory / (key + ".png")).open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            summary[key + "_file"] = key + ".png"
    else:
        code = "E502" if observation_failed else (getattr(error, "code", None) or getattr(result, "error_code", None) or "E502")
        summary.update({"error_code": code, "error_message": safe_detail(code, getattr(error, "detail", ""))})
    if observation_failed:
        summary.update(error_code="E502", error_message="过程观测或记录未完成，本次展示交付失败；请保留记录。",
                       failure_scope="OBSERVATION_ONLY")
    with (directory / "result_summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {"directory": str(directory), "summary": summary}


@dataclass
class Confirmation:
    kind: str
    message: str
    event: threading.Event = field(default_factory=threading.Event)
    accepted: bool = False


class Worker:
    """One background task at a time. All UI interaction uses the queue, never worker-side Tk calls."""
    def __init__(self, events=None):
        self.events = queue.Queue() if events is None else events
        self.lock = threading.Lock()
        self.busy = False
        self.thread = None
        self.adapter = None

    def start(self, action) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        def run():
            try:
                action()
            except Exception as e:
                self.events.put(("worker_error", getattr(e, "code", "E502")))
            finally:
                with self.lock:
                    self.busy = False
                self.events.put(("idle", None))
        self.thread = threading.Thread(target=run, daemon=False, name="BuildReasonSeg-worker")
        self.thread.start()
        return True

    def confirm(self, kind: str, message: str) -> bool:
        request = Confirmation(kind, message)
        self.events.put(("confirmation", request))
        request.event.wait()
        return request.accepted


def run_request(root: Path, image: Path, prompt: str, worker: Worker, adapter_factory=EngineAdapter):
    started = time.perf_counter()
    program, info, result, failure = None, {}, None, None
    directory = reserve_result(root, image)
    trace = TraceSession(directory, worker.events)
    worker.trace = trace
    trace.emit(1, "RUNNING", {"original_prompt": prompt})
    cancelled = False
    def confirm(kind, message):
        nonlocal cancelled
        accepted = worker.confirm(kind, message)
        cancelled = cancelled or not accepted
        adapter = worker.adapter
        if hasattr(adapter, "observe_language"):
            adapter.observe_language(user_confirmation="Y" if accepted else "N", confirmation_kind=kind)
        return accepted
    logs = Path(root) / "_engine/logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        with (logs / "demo_runtime.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            worker.events.put(("progress", ("正在理解指令……", None)))
            if worker.adapter is None:
                worker.adapter = adapter_factory(root)
            adapter = worker.adapter
            adapter.observer = trace
            adapter.language_trace = {}
            result, program, info = adapter.run(image, prompt, confirm,
                lambda text, parsed: worker.events.put(("progress", (text, parsed))))
    except Exception as e:
        failure = e
        # Raw engineering details belong to the private log, never the main screen.
        import traceback
        with (logs / "demo_runtime.log").open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
    worker.events.put(("progress", ("正在保存本次结果……", program)))
    diag = {} if result is None else result.result_payload.get("output_paths", {})
    engine_root = None
    if diag.get("diagnostics"):
        engine_root = Path(diag["diagnostics"]).resolve().parent.relative_to(Path(root).resolve()).as_posix()
    observation_ok = trace.finish(result=result, error=failure, engine_run_root=engine_root, cancelled=cancelled)
    if worker.adapter is not None:
        worker.adapter.observer = None
    published = publish_result(root, image, prompt, program, info, result, failure,
                               time.perf_counter() - started, directory=directory,
                               observation_failed=not observation_ok)
    worker.events.put(("result", published))
