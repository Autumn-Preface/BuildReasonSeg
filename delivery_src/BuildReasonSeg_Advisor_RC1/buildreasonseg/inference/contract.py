"""Inference contract: the frozen request/response schema and the Task 8A execution boundary.

Task 8A freezes what `predict.py` accepts and what a future run must produce. No mask, overlay or diagnostic is
fabricated here — the real pipeline (global detection → merge → reference → local D-B1) belongs to Task 8B/8C.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from buildreasonseg import DEFAULT_MODEL, paths
from buildreasonseg.errors import BuildReasonSegError

DEFAULT_ALPHA = 0.45
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")
UNSUPPORTED_MODALITIES = ("sar", "infrared", "ir", "multispectral")


@dataclass
class InferenceRequest:
    image: Path | None = None
    input_dir: Path | None = None
    prompt: str | None = None
    model: str = DEFAULT_MODEL
    reference_id: int | None = None
    inspect_proposals: bool = False
    device: str = "auto"
    alpha: float = DEFAULT_ALPHA
    save_diagnostics: bool = True
    confirm_command: bool = False
    verbose: bool = False
    config: Path | None = None

    @property
    def is_reference_assisted(self) -> bool:
        return self.reference_id is not None

    @property
    def normal_inference(self) -> bool:
        return not self.inspect_proposals

    def validate(self) -> list[str]:
        """Return the list of contract violations (empty means the request is well formed)."""

        problems: list[str] = []
        if bool(self.image) == bool(self.input_dir):
            problems.append("必须且只能提供 --image 或 --input-dir 之一。")
        if self.image and not self.image.exists():
            raise BuildReasonSegError("E201", detail=f"影像不存在: {self.image}")
        if self.image and self.image.suffix.lower() not in IMAGE_SUFFIXES:
            raise BuildReasonSegError("E203", detail=f"不支持的影像类型: {self.image.suffix}")
        if self.input_dir and not self.input_dir.is_dir():
            raise BuildReasonSegError("E201", detail=f"输入目录不存在: {self.input_dir}")
        if self.normal_inference and not (self.prompt or "").strip():
            problems.append("正常推理必须提供 --prompt。")
        if self.reference_id is not None and self.reference_id < 0:
            problems.append("--reference-id 必须是非负整数。")
        if not 0.0 <= self.alpha <= 1.0:
            problems.append("--alpha 必须在 [0, 1] 区间内。")
        return problems

    def output_dirs(self) -> dict:
        return {"masks": str(paths.inference_dir() / "output" / "masks"),
                "overlays": str(paths.inference_dir() / "output" / "overlays"),
                "diagnostics": str(paths.inference_dir() / "output" / "diagnostics")}


@dataclass
class InferenceResult:
    """The frozen response schema for a future implementation."""

    status: str
    program: str | None = None
    reference_id: int | None = None
    mask_path: str | None = None
    overlay_path: str | None = None
    diagnostics_path: str | None = None
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def execution_boundary(what: str) -> str:
    return (f"{what}: Task 8A 冻结了请求/响应 schema 与输出目录契约；"
            "真实推理链（大图检测 → 合并 → 全局参考 → 局部 D-B1）将在 Task 8B/8C 实现。")


__all__ = ["DEFAULT_ALPHA", "IMAGE_SUFFIXES", "InferenceRequest", "InferenceResult",
           "UNSUPPORTED_MODALITIES", "execution_boundary"]
