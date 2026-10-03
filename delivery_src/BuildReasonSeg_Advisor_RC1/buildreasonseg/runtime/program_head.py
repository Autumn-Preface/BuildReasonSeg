"""Real language runtime: Qwen3-VL-2B-Instruct + Task 7C ProgramHead, plus Qwen suggestion generation.

Task 8B sections 4-6: the normal path is always Qwen-first. The parser stack is the **verbatim ported** research
implementation (`program_parser.ProgramParserRuntime` + `qwen_seg` LoRA/token helpers) driven by the delivery
`model/components/program_head/` assets. Nothing here re-implements or replaces the parser, and the deterministic
fallback parser from Task 8A stays a diagnostics-only path that can never be reached while Qwen is available.

Suggestion generation reuses the same local Qwen weights through a `generate` call and releases the parser-only
temporary objects so only one 2B base is resident on the GPU at a time (section 6 memory requirement).
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from buildreasonseg import component_dir, paths
from buildreasonseg.errors import BuildReasonSegError
from buildreasonseg.runtime import frozen_paths

PROGRAM_HEAD_CHECKPOINT = "program_parser_l3_rehearsal_v1.pt"
QWEN_ASSET_DIRNAME = "Qwen3-VL-2B-Instruct"
EXPECTED_PROGRAM_COUNT = 20

SUGGESTION_SYSTEM_PROMPT = (
    "你是遥感建筑分割系统的指令改写助手。系统当前只支持以下四个空间推理程序（不得发明新的算子）：\n"
    "largest_to_left_of_to_nearest：最大建筑物左侧最近的建筑物\n"
    "largest_to_right_of_to_nearest：最大建筑物右侧最近的建筑物\n"
    "largest_to_above_to_nearest：最大建筑物上方最近的建筑物\n"
    "largest_to_below_to_nearest：最大建筑物下方最近的建筑物\n"
    "用户原始指令可能包含系统尚未支持的语义（例如 second largest、second nearest、颜色属性、"
    "方位、多建筑关系、between 等）。请判断用户的**本意方向**，并从上述四个程序中推荐**一个**最接近的"
    "替代方案。不得声称系统支持未支持的语义。若无法可靠对应到这四个程序之一，必须返回 NO_SAFE_SUGGESTION。\n"
    "只输出一个 JSON 对象，不要输出其它文字、不要使用 markdown 代码块。格式二选一：\n"
    '{"status": "SUGGESTION", "program": "<四个程序之一>", '
    '"display_command": "<给用户看的一句中文替代指令>", "reason": "<一句中文理由>"}\n'
    '{"status": "NO_SAFE_SUGGESTION"}'
)

#: frozen generation configuration (Task 8B.1 section 17)
SUGGESTION_GENERATION = {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}


@dataclass
class ParseOutcome:
    program: str
    confidence: float | None
    score_source: str
    runtime: str = "qwen_program_head"
    probabilities: dict = field(default_factory=dict)
    program_ids: tuple = ()
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"program": self.program, "confidence": self.confidence,
                "score_source": self.score_source, "runtime": self.runtime,
                "top5": sorted(self.probabilities.items(), key=lambda item: -item[1])[:5],
                "timings": dict(self.timings)}


def program_head_checkpoint() -> Path:
    return component_dir("program_head") / PROGRAM_HEAD_CHECKPOINT


def qwen_asset_dir() -> Path:
    return component_dir("program_head") / QWEN_ASSET_DIRNAME


class ProgramHeadRuntime:
    """Delivery-owned loader for the frozen Task 7C parser stack."""

    def __init__(self, device: str = "cuda", *, verbose: bool = False) -> None:
        self.device = device
        self.verbose = verbose
        self._runtime = None
        self.load_seconds: float | None = None
        self.load_report: dict = {}

    # ------------------------------------------------------------------ loading

    def available(self) -> tuple[bool, str]:
        checkpoint = program_head_checkpoint()
        assets = qwen_asset_dir()
        if not checkpoint.is_file():
            return False, f"缺少 ProgramHead checkpoint: {checkpoint}"
        if not assets.is_dir():
            return False, f"缺少 Qwen base assets: {assets}"
        return True, f"{assets.name} + {checkpoint.name}"

    def load(self):
        if self._runtime is not None:
            return self._runtime
        available, detail = self.available()
        if not available:
            raise BuildReasonSegError("E302", detail=detail)
        frozen_paths.ensure()
        from buildreasonseg.runtime._frozen.mvp import program_parser as parser_module
        from buildreasonseg.runtime._frozen.mvp import qwen_seg
        from buildreasonseg.runtime._frozen.mvp.task6s_directional_pipeline import (
            parse_instruction as research_parse_instruction,
        )

        started = time.time()
        processor, qwen = qwen_seg.load_qwen(str(qwen_asset_dir()), cache_dir=str(qwen_asset_dir()),
                                             dtype=torch.bfloat16, device=self.device,
                                             attn_implementation="sdpa")
        tokenizer = processor.tokenizer
        token_setup = qwen_seg.setup_seg_token(qwen, tokenizer)
        qwen, _token_holder, lora_report = qwen_seg.attach_lora(
            qwen, token_setup.seg_token_id, rank=16, alpha=32, dropout=0.05, extra_token_ids=[])
        hidden_size = (int(qwen.config.text_config.hidden_size)
                       if hasattr(qwen.config, "text_config") else int(qwen.config.hidden_size))
        head = parser_module.ProgramHead(hidden_dim=hidden_size,
                                         num_programs=parser_module.NUM_PROGRAMS).to(self.device)
        runtime = parser_module.ProgramParserRuntime(
            cfg={"models": {"qwen_model_id": str(qwen_asset_dir())}},
            processor=processor, tokenizer=tokenizer, qwen=qwen, head=head, device=self.device,
            reports={"lora": lora_report.as_dict() if hasattr(lora_report, "as_dict") else {}})
        checkpoint = program_head_checkpoint()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = runtime.qwen.load_state_dict(payload["lora_and_token_state"], strict=False)
        runtime.head.load_state_dict(payload["program_head"])
        runtime.qwen.eval()
        runtime.head.eval()
        self._runtime = runtime
        self._parse_instruction = research_parse_instruction
        self.load_seconds = round(time.time() - started, 2)
        self.load_report = {"checkpoint": str(checkpoint), "checkpoint_step": payload.get("step"),
                            "missing_keys": len(missing), "unexpected_keys": len(unexpected),
                            "program_ids": list(parser_module.EXPECTED_PROGRAM_IDS),
                            "load_seconds": self.load_seconds}
        if self.verbose:
            print(f"[language] ProgramHead loaded in {self.load_seconds}s "
                  f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
        return self._runtime

    # ------------------------------------------------------------------ parsing

    def parse(self, prompt: str) -> ParseOutcome:
        runtime = self.load()
        started = time.time()
        reported = self._parse_instruction(runtime, prompt,
                                           tuple(runtime.reports.get("program_ids")
                                                 or self.load_report["program_ids"]))
        program = reported["program"] if isinstance(reported, dict) else str(reported)
        probabilities = {}
        if isinstance(reported, dict):
            for key, value in (reported.get("probabilities") or {}).items():
                probabilities[str(key)] = float(value)
        confidence = float(reported.get("confidence", 0.0)) if isinstance(reported, dict) else None
        return ParseOutcome(program=program, confidence=confidence,
                            score_source="top1_softmax_from_program_head_logits",
                            probabilities=probabilities,
                            program_ids=tuple(self.load_report.get("program_ids", ())),
                            timings={"language": round(time.time() - started, 3)})

    # ------------------------------------------------------------------ generation

    def release(self) -> None:
        """Free the parser-only runtime (used before loading the generation path)."""

        if self._runtime is not None:
            self._runtime.qwen.to("cpu")
            self._runtime.head.to("cpu")
            self._runtime = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_suggestion(self, prompt: str, parsed_program: str) -> str:
        """Generate a *structured* replacement proposal with the same local Qwen weights (no second model).

        Returns the raw model text; `parse_suggestion_response` turns it into the frozen contract. The model is
        never asked to widen RC1's support: it may only pick one of the four programs or answer
        `NO_SAFE_SUGGESTION`.
        """

        if self._runtime is None:
            self.load()
        runtime = self._runtime
        user_text = (f"原始用户指令：{prompt}\n"
                     f"系统第一次解析结果（20 类 canonical program）：{parsed_program or '无法确定'}\n"
                     f"系统当前只支持上述四个程序。请按约定 JSON 输出推荐结果。")
        messages = [{"role": "system", "content": [{"type": "text", "text": SUGGESTION_SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "text", "text": user_text}]}]
        self.suggestion_count = getattr(self, "suggestion_count", 0) + 1
        try:
            inputs = runtime.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt")
            inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
            with torch.no_grad():
                generated = runtime.qwen.generate(**inputs, **SUGGESTION_GENERATION)
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            return runtime.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as error:
            raise BuildReasonSegError(
                "E502", detail=f"Qwen suggestion generation 失败: {error}") from error


def parse_suggestion_response(raw_text: str) -> dict:
    """Frozen Task 8B.1 suggestion contract parser (pure JSON contract, no keyword routing).

    Returns one of:

    * ``{"status": "SUGGESTION", "program": <one of the four>, "display_command": str, "reason": str}``
    * ``{"status": "NO_SAFE_SUGGESTION"}``
    * ``{"status": "UNPARSABLE", "raw": str}`` — treated as NO_SAFE_SUGGESTION by the caller

    The parser never maps text to a direction by keywords: an invalid or missing ``program`` field is a contract
    violation and therefore *not* a suggestion.
    """

    from buildreasonseg.language.registry import SUPPORTED_PROGRAMS

    if not raw_text or not raw_text.strip():
        return {"status": "UNPARSABLE", "raw": raw_text or ""}
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        if "NO_SAFE_SUGGESTION" in text:
            return {"status": "NO_SAFE_SUGGESTION"}
        return {"status": "UNPARSABLE", "raw": raw_text}
    candidate = text[start:end + 1]
    try:
        payload = json.loads(candidate)
    except Exception:
        if "NO_SAFE_SUGGESTION" in text:
            return {"status": "NO_SAFE_SUGGESTION"}
        return {"status": "UNPARSABLE", "raw": raw_text}
    if not isinstance(payload, dict):
        return {"status": "UNPARSABLE", "raw": raw_text}
    status = str(payload.get("status", "")).strip().upper()
    if status == "NO_SAFE_SUGGESTION":
        return {"status": "NO_SAFE_SUGGESTION"}
    program = str(payload.get("program", "")).strip()
    if status != "SUGGESTION" or program not in SUPPORTED_PROGRAMS:
        return {"status": "UNPARSABLE", "raw": raw_text,
                "reason": "program 字段缺失或不属于四个正式 program"}
    return {"status": "SUGGESTION", "program": program,
            "display_command": str(payload.get("display_command", "")).strip(),
            "reason": str(payload.get("reason", "")).strip(), "raw": raw_text}


def program_head_manifest_path() -> Path:
    return component_dir("program_head") / "qwen_asset_manifest.json"


def runtime_asset_report() -> dict:
    runtime = ProgramHeadRuntime(device="cpu")
    available, detail = runtime.available()
    return {"available": available, "detail": detail,
            "checkpoint": str(program_head_checkpoint()),
            "qwen_dir": str(qwen_asset_dir()),
            "manifest": str(program_head_manifest_path())}


__all__ = ["EXPECTED_PROGRAM_COUNT", "PROGRAM_HEAD_CHECKPOINT", "QWEN_ASSET_DIRNAME",
           "SUGGESTION_GENERATION", "SUGGESTION_SYSTEM_PROMPT", "ParseOutcome", "ProgramHeadRuntime",
           "parse_suggestion_response", "program_head_checkpoint", "program_head_manifest_path",
           "qwen_asset_dir", "runtime_asset_report"]
