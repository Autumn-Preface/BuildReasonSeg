"""Unified error-code registry and the user-facing exception framework (Task 8A section 12).

Users see a short Chinese message plus a repair suggestion — never a full traceback. Extended detail is meant
to be written to `logs/` (the CLI wrapper appends it there), and every error maps to a non-zero exit code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ERROR_REGISTRY: dict[str, dict[str, str]] = {
    "E101": {"name": "INVALID_COMMAND", "message": "指令格式无效，无法解析为受支持的推理命令。",
             "suggestion": "请检查指令是否完整描述“最大建筑的某个方向上的最近建筑”。"},
    "E102": {"name": "UNSUPPORTED_PROGRAM", "message": "该指令对应的空间关系尚未在当前 RC1 中正式支持。",
             "suggestion": "当前仅支持 largest→左侧/右侧/上方/下方→最近建筑；可使用建议指令继续。"},
    "E201": {"name": "IMAGE_NOT_FOUND", "message": "找不到输入影像。",
             "suggestion": "请确认 --image / --input-dir 指向存在的文件或目录。"},
    "E202": {"name": "IMAGE_UNREADABLE", "message": "输入影像无法读取。",
             "suggestion": "请确认影像未损坏，且为常见 RGB 遥感影像格式。"},
    "E203": {"name": "UNSUPPORTED_IMAGE_TYPE", "message": "输入影像类型不在支持范围内。",
             "suggestion": "RC1 仅支持 RGB 光学遥感影像；SAR / 红外 / 未处理多光谱暂不支持。"},
    "E301": {"name": "MODEL_NOT_FOUND", "message": "找不到指定的模型包。",
             "suggestion": "请确认 model/<名称>/ 存在，或使用默认模型 buildreasonseg_advisor。"},
    "E302": {"name": "MODEL_INCOMPLETE", "message": "模型包缺少必需文件。",
             "suggestion": "请重新复制完整模型包（decoder.pt / detector.pt / model.yaml / metadata.json）。"},
    "E303": {"name": "MODEL_HASH_MISMATCH", "message": "模型文件校验失败（SHA256 不一致）。",
             "suggestion": "请重新复制权威权重；不要使用被重新保存或重新导出的文件。"},
    "E304": {"name": "MODEL_INCOMPATIBLE", "message": "模型包与当前 RC1 契约不兼容。",
             "suggestion": "请使用与 buildreasonseg_advisor 同构的模型包。"},
    "E401": {"name": "NO_BUILDING_DETECTED", "message": "影像中未检测到可用建筑实例。",
             "suggestion": "请确认影像包含建筑，或调整检测参数后重试。"},
    "E402": {"name": "NO_ELIGIBLE_REFERENCE", "message": "没有满足条件的参考建筑（不触边且范围比例合格）。",
             "suggestion": "可尝试 --inspect-proposals 查看候选，或使用 --reference-id 手动指定参考。"},
    "E403": {"name": "NO_DIRECTIONAL_TARGET", "message": "该方向上没有满足条件的最近建筑目标。",
             "suggestion": "请确认参考建筑周围存在该方向的建筑，或更换指令方向。"},
    "E404": {"name": "REASONING_FAILED", "message": "空间关系推理未能完成。",
             "suggestion": "请重试，或在 logs/ 中查看详细日志后反馈。"},
    "E501": {"name": "CUDA_OUT_OF_MEMORY", "message": "显存不足。",
             "suggestion": "请使用 --device cpu，或减小处理影像尺寸后重试。"},
    "E502": {"name": "INFERENCE_RUNTIME_ERROR", "message": "推理运行时发生错误。",
             "suggestion": "请查看 logs/ 中的详细日志；如持续失败请反馈该样本。"},
    "E900": {"name": "NOT_IMPLEMENTED_IN_TASK_8A", "message": "该功能属于后续构建阶段，尚未在 Task 8A 中实现。",
             "suggestion": "当前 RC1 仅完成工程基础与契约；完整功能将在后续任务中实现。"},
    "E901": {"name": "USER_ABORTED", "message": "用户中止了当前操作。",
             "suggestion": "未执行任何推理；可修改指令后重试。"},
}

EXIT_CODES = {"E1": 10, "E2": 20, "E3": 30, "E4": 40, "E5": 50, "E9": 90}


@dataclass
class BuildReasonSegError(Exception):
    """A user-facing error carrying a registry code, a short message and a suggestion."""

    code: str
    detail: str = ""
    context: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return ERROR_REGISTRY.get(self.code, {}).get("name", "UNKNOWN_ERROR")

    @property
    def message(self) -> str:
        return ERROR_REGISTRY.get(self.code, {}).get("message", "发生未知错误。")

    @property
    def suggestion(self) -> str:
        return ERROR_REGISTRY.get(self.code, {}).get("suggestion", "请查看 logs/ 中的详细日志。")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code[:2], 1)

    def user_lines(self) -> list[str]:
        lines = [f"[{self.code} {self.name}] {self.message}"]
        if self.detail:
            lines.append(f"  细节: {self.detail}")
        lines.append(f"  建议: {self.suggestion}")
        return lines


class MissingDependencyError(BuildReasonSegError):
    def __init__(self, detail: str = "") -> None:
        super().__init__("E302", detail=detail)


class HashMismatchError(BuildReasonSegError):
    def __init__(self, detail: str = "") -> None:
        super().__init__("E303", detail=detail)


class NotInTask8AError(BuildReasonSegError):
    """Raised by every entry point whose real implementation belongs to Task 8B/8C."""

    def __init__(self, detail: str = "") -> None:
        super().__init__("E900",
                         detail=detail or "该功能将在 Task 8B/8C 中实现；Task 8A 只冻结接口与契约。",
                         context={"stage": "8A", "marker": "NOT_IMPLEMENTED_IN_TASK_8A"})


def lookup(code: str) -> dict:
    return {"code": code, **ERROR_REGISTRY.get(code, {})}


def registry() -> dict:
    return {code: dict(entry) for code, entry in ERROR_REGISTRY.items()}


__all__ = ["BuildReasonSegError", "ERROR_REGISTRY", "EXIT_CODES", "HashMismatchError",
           "MissingDependencyError", "NotInTask8AError", "lookup", "registry"]
