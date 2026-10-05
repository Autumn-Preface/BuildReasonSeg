"""BuildReasonSeg Advisor RC1 — real prediction entry point (Task 8B).

Normal path (always Qwen-first):

    model package check → RGB image load → Qwen/ProgramHead parse → printed program → hard validator
    → tiled global proposals + merge → global reference (automatic or assisted) → one deterministic 512
    reasoning context → frozen SAM2 → P_dir/P_near → D-B1 → target mask → mapped back to the original image
    → mask + translucent overlay + diagnostics

    python predict.py --image inference/input/test.tif --prompt "找出最大的建筑，在它右边寻找离它最近的建筑"
    python predict.py --image inference/input/test.tif --inspect-proposals
    python predict.py --image inference/input/test.tif --prompt "..." --reference-id 12
    python predict.py --input-dir inference/input --prompt "分割最大建筑物右侧最近的建筑物"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from buildreasonseg import DEFAULT_MODEL
from buildreasonseg.cli.common import base_parser, resolve_config, run_cli
from buildreasonseg.errors import BuildReasonSegError
from buildreasonseg.inference.contract import DEFAULT_ALPHA, IMAGE_SUFFIXES
from buildreasonseg.language.frontend import FALLBACK_BANNER, DeterministicFallbackFrontend
from buildreasonseg.language.registry import ParsedProgram
from buildreasonseg.language.suggestion import EXAMPLES, SUGGESTION_QUESTION
from buildreasonseg.language.validator import validate
from buildreasonseg.models.package import resolve_model
from buildreasonseg.runtime.pipeline import (
    PipelineRequest,
    PredictRuntime,
    inspect_proposals,
    predict_one,
)

DESCRIPTION = ("BuildReasonSeg Advisor RC1 — 依据自然语言指令在 RGB 遥感影像中分割目标建筑"
               "（最大建筑 → 方向 → 最近建筑）。")

EXIT_BATCH_PARTIAL = 2
EXIT_BATCH_ALL_FAILED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(DESCRIPTION, config_default="configs/inference.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="单张影像路径（与 --input-dir 二选一）")
    source.add_argument("--input-dir", type=Path, dest="input_dir",
                        help="批量输入目录（只扫描第一层，不递归）")
    parser.add_argument("--prompt", default=None, help="自然语言指令；正常推理必须提供")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型包名称（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--reference-id", type=int, default=None, dest="reference_id",
                        help="Reference-Assisted Mode：手动指定参考建筑编号（仅单图）")
    parser.add_argument("--inspect-proposals", action="store_true", dest="inspect_proposals",
                        help="输出编号后的候选建筑，用于人工选择参考（仅单图，不需要 --prompt）")
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--save-diagnostics", action="store_true", dest="save_diagnostics",
                            help="保存诊断产物（默认开启）")
    diagnostic.add_argument("--no-save-diagnostics", action="store_false", dest="save_diagnostics",
                            help="不保存诊断产物")
    parser.set_defaults(save_diagnostics=True)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help=f"overlay 透明度，范围 (0, 1]（默认 {DEFAULT_ALPHA}）")
    parser.add_argument("--confirm-command", action="store_true", dest="confirm_command",
                        help="执行前显示解析结果并请求确认")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if not 0.0 < args.alpha <= 1.0:
        raise BuildReasonSegError("E101", detail=f"--alpha 必须在 (0, 1] 区间内，收到 {args.alpha}")
    if args.input_dir is not None and args.reference_id is not None:
        raise BuildReasonSegError("E101", detail="--reference-id 只能与 --image 一起使用。")
    if args.input_dir is not None and args.inspect_proposals:
        raise BuildReasonSegError("E101", detail="--inspect-proposals 只支持单图（--image）。")
    if not args.inspect_proposals and not (args.prompt or "").strip():
        raise BuildReasonSegError("E101", detail="正常推理必须提供 --prompt。")
    if args.image is not None:
        if not args.image.exists():
            raise BuildReasonSegError("E201", detail=f"影像不存在: {args.image}")
        if args.image.suffix.lower() not in IMAGE_SUFFIXES:
            raise BuildReasonSegError("E203",
                                      detail=f"不支持的影像类型 {args.image.suffix}"
                                             f"（支持 {', '.join(IMAGE_SUFFIXES)}）")
    if args.input_dir is not None and not args.input_dir.is_dir():
        raise BuildReasonSegError("E201", detail=f"输入目录不存在: {args.input_dir}")


def _resolve_and_report(args: argparse.Namespace) -> str:
    resolution = resolve_model(args.model, confirm_reader=input)
    if resolution.switched_with_confirmation:
        print(f"[模型] 已按用户确认切换到默认模型 {resolution.package.name}")
    elif not resolution.ok:
        if resolution.error is not None:
            raise resolution.error
        raise BuildReasonSegError("E301", detail="模型不可用。")
    else:
        print(f"[模型] {resolution.package.name}")
    verification = resolution.package.verify()
    if not verification["all_verified"]:
        raise BuildReasonSegError("E303", detail="decoder/detector 权重校验失败。")
    return resolution.package.name


def _program_human(program: str) -> str:
    table = {"largest_to_left_of_to_nearest": "largest -> left_of -> nearest",
             "largest_to_right_of_to_nearest": "largest -> right_of -> nearest",
             "largest_to_above_to_nearest": "largest -> above -> nearest",
             "largest_to_below_to_nearest": "largest -> below -> nearest"}
    return table.get(program, program or "(无法解析)")


def _decompose(program: str) -> tuple[str, str, str]:
    reverse = {"largest_to_left_of_to_nearest": ("largest", "left_of", "nearest"),
               "largest_to_right_of_to_nearest": ("largest", "right_of", "nearest"),
               "largest_to_above_to_nearest": ("largest", "above", "nearest"),
               "largest_to_below_to_nearest": ("largest", "below", "nearest")}
    return reverse.get(program, (program, "-", "-"))


def _confirm(question: str) -> bool:
    answer = (input(f"{question} [Y/N]: ") or "").strip().lower()
    return answer in ("y", "yes", "是")


def _suggestion(language, prompt: str, parsed_program: str) -> dict:
    """Ask the local Qwen for a structured, Validator-gated replacement proposal."""

    from buildreasonseg.runtime.program_head import parse_suggestion_response

    try:
        raw = language.generate_suggestion(prompt, parsed_program)
    except BuildReasonSegError as error:
        print(f"[提示] Qwen 建议生成失败：{error.detail}")
        return {"status": "RUNTIME_ERROR", "raw": "", "detail": error.detail}
    parsed = parse_suggestion_response(raw)
    parsed["invoked"] = True
    return parsed


def _suggestion_program(parsed_suggestion: dict) -> ParsedProgram | None:
    """The execution decision comes from the structured `program` field, re-checked by the Validator."""

    if parsed_suggestion.get("status") != "SUGGESTION":
        return None
    candidate = ParsedProgram(program=parsed_suggestion["program"], source="qwen_program_head",
                              confidence=None, raw_text=parsed_suggestion.get("display_command", ""),
                              diagnostics={"suggestion_reason": parsed_suggestion.get("reason", "")})
    if not validate(candidate).supported:
        print(f"[提示] Qwen 建议的 program（{candidate.program}）未通过 Validator，不执行。")
        return None
    return candidate


def _parse_program(runtime: PredictRuntime, prompt: str, *, confirm: bool
                   ) -> tuple[ParsedProgram, dict, str | None]:
    """Qwen-first parse, validator gate, suggestion flow and optional confirmation."""

    language = runtime.language_runtime()
    available, detail = language.available()
    if not available:
        print("语言模型不可用。")
        if not _confirm("是否进入有限兼容模式？"):
            raise BuildReasonSegError("E901", detail="用户拒绝进入 fallback 模式。")
        fallback = DeterministicFallbackFrontend("qwen_asset_missing")
        fallback.reason = "qwen_asset_missing"
        print(FALLBACK_BANNER)
        return (fallback.parse(prompt),
                {"language_mode": "fallback", "fallback_reason": "qwen_asset_missing"}, None)
    try:
        outcome = language.parse(prompt)
    except Exception as error:
        print("语言模型不可用。")
        if not _confirm("是否进入有限兼容模式？"):
            raise BuildReasonSegError("E502",
                                      detail=f"Qwen runtime 失败且用户未启用 fallback: {error}")
        fallback = DeterministicFallbackFrontend("qwen_runtime_failed")
        fallback.reason = "qwen_runtime_failed"
        print(FALLBACK_BANNER)
        return (fallback.parse(prompt),
                {"language_mode": "fallback", "fallback_reason": "qwen_runtime_failed"}, None)

    parsed = ParsedProgram(program=outcome.program, source="qwen_program_head",
                           confidence=outcome.confidence, raw_text=prompt)
    info = {"language_mode": "qwen", "confidence": outcome.confidence,
            "score_source": outcome.score_source, "top5": outcome.to_dict()["top5"]}
    if outcome.confidence is not None:
        print(f"[语言] Qwen / ProgramHead  置信度 {outcome.confidence:.3f}")
    else:
        print("[语言] Qwen / ProgramHead")
    print(f"[解析] {_program_human(parsed.program)}   ({parsed.program})")

    validation = validate(parsed)
    suggestion_trace = None
    if not validation.supported:
        print("当前解析结果不属于 RC1 已开放的四类空间推理语义。")
        print(f"第一次解析：{_program_human(parsed.program)}   ({parsed.program})")
        suggestion = _suggestion(language, prompt, parsed.program)
        suggestion_trace = {"suggestion_invoked": True, "suggestion_status": suggestion.get("status"),
                            "suggestion_raw_text": suggestion.get("raw", ""),
                            "suggested_program": suggestion.get("program"),
                            "suggestion_reason": suggestion.get("reason", "")}
        suggested = _suggestion_program(suggestion)
        suggestion_trace["suggestion_validator_pass"] = suggested is not None
        if suggested is not None:
            display = suggestion.get("display_command") or _program_human(suggested.program)
            print("Qwen 建议的可支持替代指令：")
            print(f"“{display}”")
            if suggestion.get("reason"):
                print(f"（理由：{suggestion['reason']}）")
            print(f"建议程序：{_program_human(suggested.program)}")
            info = {**info, "original_prompt": prompt, "initial_program": parsed.program,
                    "initial_supported": False, "suggestion_used": True,
                    "suggested_program": suggested.program,
                    "suggestion_display_command": display}
            if _confirm(SUGGESTION_QUESTION):
                info["user_confirmation"] = "Y"
                return suggested, info, display
            info["user_confirmation"] = "N"
        else:
            print("Qwen 未给出可安全替代的建议（NO_SAFE_SUGGESTION 或建议未通过 Validator）。")
        print("当前已支持的空间语义（命令示例）: " + " | ".join(EXAMPLES))
        trace = {"original_prompt": prompt, "initial_program": parsed.program,
                 "initial_supported": False, "suggestion_used": False,
                 "suggested_program": suggestion.get("program"),
                 "user_confirmation": suggestion_trace.get("user_confirmation", "N")}
        error = BuildReasonSegError(
            "E102", detail=f"program {parsed.program!r} 不在 RC1 正式支持的四个 program 之内。")
        error.context.update({"language_trace": trace, "suggestion": suggestion_trace})
        raise error
    info = {**info, "original_prompt": prompt, "initial_program": parsed.program,
            "initial_supported": True, "suggestion_used": False, "user_confirmation": None,
            "suggested_program": None}
    if confirm:
        reference, direction, relation = _decompose(parsed.program)
        print("Qwen 解析结果：")
        print(f"Reference : {reference}")
        print(f"Direction : {direction}")
        print(f"Relation  : {relation}")
        if not _confirm("是否按此理解执行？"):
            raise BuildReasonSegError("E901", detail="用户未确认解析结果。")
        info["user_confirmation"] = "Y"
    return parsed, info, None


def _batch_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _run_batch(runtime: PredictRuntime, args: argparse.Namespace, package_name: str,
               parsed: ParsedProgram, language_info: dict, started: float) -> int:
    files = _batch_files(args.input_dir)
    if not files:
        raise BuildReasonSegError("E201", detail=f"目录中没有受支持的影像: {args.input_dir}")
    successes = failures = 0
    for index, path in enumerate(files, 1):
        request = PipelineRequest(image=path, prompt=args.prompt, model=package_name,
                                  device=args.device, alpha=args.alpha,
                                  save_diagnostics=args.save_diagnostics, parsed=parsed)
        request.parsed_info = language_info
        result = predict_one(runtime, request)
        if result.ok:
            successes += 1
            print(f"[{index}/{len(files)}] {path.name} ... SUCCESS [runtime-only; semantic=NOT_EVALUATED]")
        else:
            failures += 1
            print(f"[{index}/{len(files)}] {path.name} ... {result.error_code} "
                  f"{result.error_reason or ''}".rstrip())
    print("Batch complete")
    print(f"Success: {successes}")
    print(f"Failed : {failures}")
    print(f"Total  : {round(time.time() - started, 2)}s")
    if failures == 0:
        return 0
    return EXIT_BATCH_ALL_FAILED if successes == 0 else EXIT_BATCH_PARTIAL


def _report_single(result, args: argparse.Namespace) -> None:
    payload = result.result_payload
    print(f"Result       : {payload['status']}")
    if payload.get("status") == "SUCCESS":
        print(f"Result       : SUCCESS [runtime-only; semantic=NOT_EVALUATED]")
        print(f"Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.")
    if result.ok:
        print()
        print(f"Mask         : {payload['output_paths']['mask']}")
        print(f"Overlay      : {payload['output_paths']['overlay']}")
        print(f"Diagnostics  : {payload['output_paths']['diagnostics']}")
        print(f"Reference ID : {payload.get('reference_id')}")
        print(f"Mask area    : {payload.get('mask_area')}")
    else:
        error = BuildReasonSegError(result.error_code or "E502", detail=payload.get("detail", ""))
        for line in error.user_lines():
            print(line, file=sys.stderr)


def _exit_code(error_code: str | None) -> int:
    if not error_code:
        return 1
    from buildreasonseg.errors import EXIT_CODES

    return EXIT_CODES.get(error_code[:2], 1)


def handler(args: argparse.Namespace) -> int:
    _validate_arguments(args)
    config = resolve_config(args.config)
    if not config.is_file():
        raise BuildReasonSegError("E101", detail=f"配置文件不存在: {config}")
    package_name = _resolve_and_report(args)

    print("BuildReasonSeg Advisor RC1")
    print(f"Model        : {package_name}")

    runtime = PredictRuntime(device=args.device, verbose=args.verbose)
    print(f"Device       : {runtime.device}")

    if args.inspect_proposals:
        request = PipelineRequest(image=args.image, model=package_name, inspect_proposals=True,
                                  device=args.device, save_diagnostics=args.save_diagnostics)
        print("Language     : (--inspect-proposals 不解析自然语言)")
        print("Mode         : INSPECT PROPOSALS")
        result = inspect_proposals(runtime, request)
        print(f"Result       : {result.result_payload['status']}")
        return 0

    started = time.time()
    parsed, language_info, suggestion = _parse_program(runtime, args.prompt,
                                                       confirm=args.confirm_command)
    print("Language     : Qwen / ProgramHead" if language_info["language_mode"] == "qwen"
          else "Language     : FALLBACK COMMAND PARSER")
    print(f"Parsed       : {_program_human(parsed.program)}")
    print(f"Mode         : {'ASSISTED' if args.reference_id is not None else 'AUTOMATIC'}")

    if args.input_dir is not None:
        return _run_batch(runtime, args, package_name, parsed, language_info, started)

    request = PipelineRequest(image=args.image, prompt=args.prompt, model=package_name,
                              reference_id=args.reference_id, device=args.device, alpha=args.alpha,
                              save_diagnostics=args.save_diagnostics, parsed=parsed,
                              suggestion=suggestion)
    request.parsed_info = language_info
    result = predict_one(runtime, request)
    _report_single(result, args)
    return 0 if result.ok else _exit_code(result.error_code)


def main(argv: list[str] | None = None) -> int:
    return run_cli(build_parser(), handler, argv)


if __name__ == "__main__":
    sys.exit(main())
