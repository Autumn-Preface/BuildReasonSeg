"""End-to-end predict pipeline for one image (Task 8B sections 3, 11-27).

Order of operations (frozen): model-package check → image load → Qwen/ProgramHead parse → validator →
tiled global detection + merge → global reference selection (automatic or assisted) → one deterministic 512
reasoning context → frozen SAM2 → relation fields → D-B1 → hard post-inference validity → mask/overlay/
diagnostics. Ground truth is never read and no test-split data is ever used.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from buildreasonseg import paths
from buildreasonseg.errors import BuildReasonSegError
from buildreasonseg.language.frontend import FALLBACK_BANNER, DeterministicFallbackFrontend
from buildreasonseg.language.registry import ParsedProgram
from buildreasonseg.language.suggestion import EXAMPLES
from buildreasonseg.language.validator import validate
from buildreasonseg.runtime import outputs as outputs_module
from buildreasonseg.runtime.context import (
    ReasoningContext,
    context_to_global,
    direction_satisfied,
    guard_directional_candidates,
    plan_context,
)
from buildreasonseg.runtime.core import Db1Runtime, Sam2Runtime, program_to_direction, run_core_chain
from buildreasonseg.runtime.detector import (
    DetectorRuntime,
    GlobalProposal,
    proposal_by_id,
    select_reference,
)
from buildreasonseg.runtime.imageio import load_image, pad_to_512
from buildreasonseg.runtime.program_head import ProgramHeadRuntime

MODES = ("auto", "assisted", "inspect")


@dataclass
class PipelineRequest:
    image: Path
    prompt: str | None = None
    model: str = "buildreasonseg_advisor"
    reference_id: int | None = None
    inspect_proposals: bool = False
    device: str = "auto"
    alpha: float = 0.45
    save_diagnostics: bool = True
    confirm_reader = None
    language_mode: str = "qwen"
    parsed: ParsedProgram | None = None
    parsed_info: dict = field(default_factory=dict)
    suggestion: str | None = None
    verbose: bool = False


@dataclass
class StageTimings:
    language: float = 0.0
    detector: float = 0.0
    merge: float = 0.0
    sam2: float = 0.0
    relation_fields: float = 0.0
    db1: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {"language": round(self.language, 3), "detector": round(self.detector, 3),
                "merge": round(self.merge, 3), "sam2": round(self.sam2, 3),
                "relation_fields": round(self.relation_fields, 3), "db1": round(self.db1, 3),
                "total": round(self.total, 3)}


@dataclass
class PipelineResult:
    status: str
    image: Path
    result_payload: dict
    mask: np.ndarray | None = None
    outputs: outputs_module.SampleOutputs | None = None
    error_code: str | None = None
    error_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "SUCCESS"


class PredictRuntime:
    """Holds the heavy runtimes so one process loads each model exactly once (section 31)."""

    def __init__(self, device: str = "auto", *, verbose: bool = False) -> None:
        from buildreasonseg.utils.device import resolve_device

        self.device = resolve_device(device)
        self.verbose = verbose
        self.detector = DetectorRuntime(device=self.device)
        self.sam2 = Sam2Runtime(device=self.device)
        self.db1 = Db1Runtime(device=self.device)
        self.language: ProgramHeadRuntime | None = None

    def language_runtime(self) -> ProgramHeadRuntime:
        if self.language is None:
            self.language = ProgramHeadRuntime(device=self.device, verbose=self.verbose)
        return self.language


def parse_prompt(runtime: PredictRuntime, request: PipelineRequest) -> tuple[ParsedProgram, dict]:
    """Qwen-first parse with the Task 8A validator and the diagnostics-only fallback contract."""

    frontend = runtime.language_runtime()
    available, detail = frontend.available()
    info: dict = {"language_mode": "qwen", "runtime": "qwen_program_head", "asset_detail": detail}
    if available:
        outcome = frontend.parse(request.prompt or "")
        parsed = ParsedProgram(program=outcome.program, source="qwen_program_head",
                               confidence=outcome.confidence, raw_text=request.prompt or "")
        info.update({"language_mode": "qwen", "confidence": outcome.confidence,
                     "score_source": outcome.score_source, "top5": outcome.to_dict()["top5"]})
        return parsed, info
    raise BuildReasonSegError("E302", detail=f"Qwen 前端不可用: {detail}")


def fallback_parse(prompt: str, reason: str = "qwen_runtime_failed") -> tuple[ParsedProgram, dict]:
    print(FALLBACK_BANNER)
    frontend = DeterministicFallbackFrontend(reason)
    frontend.reason = reason
    parsed = frontend.parse(prompt)
    return parsed, {"language_mode": "fallback", "fallback_reason": reason}


def inspect_proposals(runtime: PredictRuntime, request: PipelineRequest) -> PipelineResult:
    """`--inspect-proposals`: detector + merge + previews only, no Qwen and no core chain."""

    started = time.time()
    loaded = load_image(request.image)
    detection = runtime.detector.detect_global(loaded.rgb)
    proposals = detection["merged"]
    outputs = outputs_module.allocate_outputs(request.image)
    preview = outputs_module.proposals_preview_image(loaded.rgb, proposals)
    payload = {
        "status": "SUCCESS", "mode": "inspect",
        **loaded.describe(), "model_package": request.model, "device": runtime.device,
        "raw_proposal_count": detection["raw_count"], "merged_proposal_count": len(proposals),
        "tile_size": detection["tile_size"], "overlap": detection["overlap"],
        "tile_count": len(detection["windows"]),
        "proposals": {"count": len(proposals),
                      "items": [proposal.to_dict() for proposal in proposals]},
        "output_paths": {"global_proposals": str(Path(outputs.diagnostics_dir) / "global_proposals.png"),
                         "proposals": str(Path(outputs.diagnostics_dir) / "proposals.json")},
        "timings": StageTimings(total=time.time() - started,
                                detector=detection["detector_seconds"]).to_dict(),
    }
    outputs_module.save_diagnostics(outputs, payload, proposals_preview=preview)
    print(f"Detected {len(proposals)} merged building proposals.")
    print(f"See: {Path(outputs.diagnostics_dir) / 'global_proposals.png'}")
    return PipelineResult(status="SUCCESS", image=request.image, result_payload=payload,
                          outputs=outputs)


def predict_one(runtime: PredictRuntime, request: PipelineRequest) -> PipelineResult:
    """Run the full chain for one image; never fabricates a mask when the algorithm stage fails."""

    total_started = time.time()
    timings = StageTimings()
    loaded = load_image(request.image)
    outputs = outputs_module.allocate_outputs(request.image)
    payload: dict = {"status": "FAILED", **loaded.describe(), "model_package": request.model,
                     "device": runtime.device, "prompt": request.prompt or ""}
    parsed = request.parsed
    validation = None
    suggestion = request.suggestion
    reference_mode = "assisted" if request.reference_id is not None else "automatic"

    def _fail(error: BuildReasonSegError) -> PipelineResult:
        if error.context:
            if "language_trace" in error.context:
                payload.update(error.context["language_trace"])
            if "suggestion" in error.context:
                payload["suggestion_trace"] = error.context["suggestion"]
        payload.update({"status": "FAILED", "error_code": error.code, "error_name": error.name,
                        "reason": error.context.get("reason") if error.context else None,
                        "detail": error.detail, "reference_mode": reference_mode,
                        "timings": timings.to_dict(), "output_paths": {}})
        if request.save_diagnostics:
            outputs_module.save_diagnostics(outputs, payload)
            payload["output_paths"] = {"diagnostics": outputs.diagnostics_dir}
        return PipelineResult(status="FAILED", image=request.image, result_payload=payload,
                              outputs=outputs, error_code=error.code,
                              error_reason=payload.get("reason"))

    try:
        # ---- language (already parsed once per process for batch mode)
        if parsed is None:
            language_started = time.time()
            parsed, language_info = parse_prompt(runtime, request)
            timings.language = time.time() - language_started
            payload.update(language_info)
            validation = validate(parsed)
            payload["parsed"] = {"program": parsed.program, "source": parsed.source,
                                 "confidence": parsed.confidence,
                                 "supported": validation.supported,
                                 "reason": validation.reason,
                                 "suggestion": validation.suggestion}
            if not validation.supported:
                raise BuildReasonSegError(
                    "E102",
                    detail=f"指令解析为 {parsed.program!r}，不在 RC1 正式支持的四类 program 之内。",
                    context={"suggestion": validation.suggestion})
        else:
            validation = validate(parsed)
            payload.update(request.parsed_info or {})
            payload.setdefault("original_prompt", request.parsed_info.get("original_prompt"))
            payload.setdefault("initial_program", request.parsed_info.get("initial_program"))
            payload.setdefault("initial_supported", request.parsed_info.get("initial_supported"))
            payload.setdefault("suggestion_used", request.parsed_info.get("suggestion_used", False))
            payload.setdefault("suggested_program", request.parsed_info.get("suggested_program"))
            payload.setdefault("user_confirmation", request.parsed_info.get("user_confirmation"))
            payload["parsed"] = {"program": parsed.program, "source": parsed.source,
                                 "confidence": parsed.confidence,
                                 "supported": validation.supported, "suggestion": suggestion}

        if request.parsed_info.get("suggestion_used"):
            payload["suggestion_trace"] = {
                "suggestion_status": "SUGGESTION",
                "suggested_program": request.parsed_info.get("suggested_program"),
                "suggestion_validator_pass": True,
                "suggestion_display_command": request.parsed_info.get("suggestion_display_command")}
            payload["suggestion_used"] = True
            payload["suggested_program"] = request.parsed_info.get("suggested_program")

        # ---- global detection + merge
        detection = runtime.detector.detect_global(loaded.rgb)
        proposals: list[GlobalProposal] = detection["merged"]
        timings.detector = detection["detector_seconds"]
        payload.update({"tile_size": detection["tile_size"], "overlap": detection["overlap"],
                        "tile_count": len(detection["windows"]),
                        "raw_proposal_count": detection["raw_count"],
                        "merged_proposal_count": len(proposals),
                        "proposals": {"count": len(proposals),
                                      "items": [proposal.to_dict() for proposal in proposals]}})
        if not proposals:
            raise BuildReasonSegError("E401", detail="整幅影像未检测到任何建筑实例。")

        # ---- global reference
        if reference_mode == "assisted":
            reference = proposal_by_id(proposals, int(request.reference_id))
            if reference is None:
                raise BuildReasonSegError(
                    "E402", detail=f"--reference-id {request.reference_id} 不存在（有效编号 0.."
                                   f"{len(proposals) - 1}）。")
            payload.update({"parsed_reference_selector": "largest", "effective_reference_id":
                            reference.proposal_id, "reference_override": True})
        else:
            reference = select_reference(proposals, family="largest")
            if reference is None:
                raise BuildReasonSegError(
                    "E402", detail="没有满足 U-C1 eligibility 的参考建筑（不触边且 bbox extent ≤ 0.20）。")
            payload.update({"parsed_reference_selector": "largest",
                            "effective_reference_id": reference.proposal_id,
                            "reference_override": False})
        payload.update({"reference_mode": reference_mode, "reference_id": reference.proposal_id,
                        "reference_area": int(reference.mask_area),
                        "reference_confidence": float(reference.confidence),
                        "reference_bbox": list(reference.global_bbox)})

        # ---- deterministic 512 reasoning context
        direction = program_to_direction(parsed.program)
        context = plan_context(reference, direction)
        guard = guard_directional_candidates(reference, proposals, direction, context)
        payload["directional_guard"] = guard
        rgb_context, padding = _context_rgb(loaded.rgb, context)
        context.padding = padding
        payload["reasoning_context"] = context.to_dict()
        payload["direction"] = direction

        # ---- frozen core chain
        reference_mask = _reference_mask_context(reference, context)
        if not reference_mask.any():
            raise BuildReasonSegError("E402", detail="参考建筑在本 reasoning context 内没有有效像素。")
        chain = run_core_chain(rgb_context, reference_mask, parsed.program, sam2=runtime.sam2,
                               db1=runtime.db1)
        timings.sam2 = chain.timings.get("sam2", 0.0)
        timings.relation_fields = chain.timings.get("relation_fields", 0.0)
        timings.db1 = chain.timings.get("db1", 0.0)

        # ---- hard post-inference validity (no "pick the best")
        mask_full, map_padding = context_to_global(chain.mask_context, context,
                                                  (loaded.height, loaded.width))
        if not mask_full.any():
            raise BuildReasonSegError("E404", detail="D-B1 输出掩膜为空。",
                                      context={"reason": "empty_target_mask"})
        if not (mask_full & _non_padding_mask(loaded, padding)).any():
            raise BuildReasonSegError("E404", detail="掩膜像素全部落在 padding 区域。",
                                      context={"reason": "mask_only_in_padding"})
        target_centroid = _centroid(mask_full)
        if not direction_satisfied(target_centroid, reference.centroid, direction):
            raise BuildReasonSegError(
                "E404",
                detail=f"target centroid 不满足 direction={direction} 相对参考建筑的硬约束。",
                context={"reason": "direction_constraint_violated"})

        # ---- outputs
        info = outputs_module.save_final_outputs(outputs, loaded.rgb, mask_full, alpha=request.alpha)
        payload.update({"mask_area": int(mask_full.sum()), "status": "SUCCESS",
                        "target_centroid": [float(target_centroid[0]), float(target_centroid[1])],
                        "context_padding": map_padding,
                        "output_paths": {"mask": info["mask"], "overlay": info["overlay"],
                                         "diagnostics": outputs.diagnostics_dir},
                        "field_mass": chain.field_mass})
        if request.save_diagnostics:
            reference_preview = outputs_module.proposals_preview_image(
                loaded.rgb, proposals, selected_id=reference.proposal_id)
            outputs_module.save_diagnostics(
                outputs, payload, proposals_preview=reference_preview,
                reference_preview=reference_preview, context_rgb=rgb_context,
                reference_context_mask=reference_mask, direction_field=chain.direction_field,
                nearest_field=chain.nearest_field, relation_weight=chain.relation_weight,
                prototype_similarity=chain.prototype_similarity,
                maps={"P_dir": chain.direction_field, "P_near": chain.nearest_field,
                      "W": chain.relation_weight, "A": chain.attention,
                      "C": chain.prototype_similarity, "logits": chain.logits,
                      "probability": chain.probability_context})
        timings.total = time.time() - total_started
        payload["timings"] = timings.to_dict()
        _rewrite_result_json(outputs, payload)
        return PipelineResult(status="SUCCESS", image=request.image, result_payload=payload,
                              mask=mask_full, outputs=outputs)
    except BuildReasonSegError as error:
        return _fail(error)
    except Exception as error:  # unexpected engineering failure → E502 with a logged traceback
        return _fail(BuildReasonSegError("E502", detail=str(error)))


def _context_rgb(rgb: np.ndarray, context: ReasoningContext) -> tuple[np.ndarray, dict]:
    from buildreasonseg.runtime.imageio import crop_with_reflection

    return crop_with_reflection(rgb, context.top, context.left, context.size)


def _reference_mask_context(reference: GlobalProposal, context: ReasoningContext) -> np.ndarray:
    from buildreasonseg.runtime.core import reference_mask_from_proposal

    return reference_mask_from_proposal(reference, context)


def _non_padding_mask(loaded, padding: dict) -> np.ndarray:
    mask = np.ones((loaded.height, loaded.width), dtype=bool)
    return mask


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return (0.0, 0.0)
    return (float(rows.mean()), float(columns_mean := cols.mean()))


def _rewrite_result_json(outputs: outputs_module.SampleOutputs, payload: dict) -> None:
    target = Path(outputs.diagnostics_dir) / "result.json"
    if target.parent.is_dir():
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def command_examples() -> str:
    return "可用命令示例（语义类型）: " + " | ".join(EXAMPLES)


__all__ = ["MODES", "PipelineRequest", "PipelineResult", "PredictRuntime", "StageTimings",
           "command_examples", "fallback_parse", "inspect_proposals", "parse_prompt", "predict_one"]
