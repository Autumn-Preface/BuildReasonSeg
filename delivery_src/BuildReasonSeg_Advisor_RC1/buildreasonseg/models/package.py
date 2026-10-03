"""Model-package contract: loading, hash verification, completeness and the no-silent-switch fallback.

Task 8A sections 5 and 13:

* `model/<name>/` is a self-contained package (`decoder.pt`, `detector.pt`, `model.yaml`, `metadata.json`,
  `metrics.json`) plus shared `model/components/{sam2,program_head}` assets;
* every checkpoint is verified against the SHA256/bytes recorded in the package metadata;
* a user-specified broken model never switches silently — the user must confirm the default model explicitly;
* when the default package itself is unavailable the run stops with an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from buildreasonseg import DEFAULT_MODEL, component_dir, model_dir, paths
from buildreasonseg.errors import BuildReasonSegError, HashMismatchError
from buildreasonseg.utils.hashing import sha256_file, verify_sha256

REQUIRED_PACKAGE_FILES = ("decoder.pt", "detector.pt", "model.yaml", "metadata.json")
OPTIONAL_PACKAGE_FILES = ("metrics.json",)
COMPONENTS = {"sam2": ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml"),
              "program_head": ("program_parser_l3_rehearsal_v1.pt", "Qwen3-VL-2B-Instruct")}


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as error:  # pragma: no cover - PyYAML ships with the project environment
        raise BuildReasonSegError("E502", detail=f"无法导入 PyYAML: {error}") from error
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class ModelPackage:
    name: str
    root: Path
    config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, name: str = DEFAULT_MODEL) -> "ModelPackage":
        root = model_dir(name)
        if not root.is_dir():
            raise BuildReasonSegError("E301", detail=f"模型包目录不存在: {root}")
        package = cls(name=name, root=root)
        missing = [filename for filename in REQUIRED_PACKAGE_FILES
                   if not (root / filename).is_file()]
        if missing:
            raise BuildReasonSegError("E302", detail=f"模型包缺少文件: {', '.join(missing)}")
        package.config = _load_yaml(root / "model.yaml")
        package.metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        metrics_path = root / "metrics.json"
        if metrics_path.is_file():
            package.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return package

    # ------------------------------------------------------------------ contract

    @property
    def decoder_path(self) -> Path:
        return self.root / "decoder.pt"

    @property
    def detector_path(self) -> Path:
        return self.root / "detector.pt"

    @property
    def supported_programs(self) -> list[str]:
        return list((self.config.get("program_set") or
                     self.metadata.get("supported_program_set") or []))

    @property
    def known_limitations(self) -> list[str]:
        return list(self.metadata.get("known_limitations", []))

    def declared_hashes(self) -> dict:
        decoder = self.config.get("decoder", {})
        detector = self.config.get("detector", {})
        return {"decoder": {"sha256": decoder.get("sha256"), "bytes": decoder.get("bytes")},
                "detector": {"sha256": detector.get("sha256"), "bytes": detector.get("bytes")}}

    def verify(self) -> dict:
        declared = self.declared_hashes()
        decoder = verify_sha256(self.decoder_path, declared["decoder"]["sha256"],
                                declared["decoder"].get("bytes"))
        detector = verify_sha256(self.detector_path, declared["detector"]["sha256"],
                                 declared["detector"].get("bytes"))
        return {"decoder": decoder, "detector": detector,
                "all_verified": bool(decoder["matches"] and detector["matches"])}

    def verify_or_raise(self) -> dict:
        report = self.verify()
        if not report["all_verified"]:
            broken = [name for name, entry in report.items()
                      if isinstance(entry, dict) and entry.get("matches") is False]
            raise HashMismatchError(detail=f"校验失败的权重: {', '.join(broken)}")
        return report

    def component_report(self) -> dict:
        report = {}
        for component, expected in COMPONENTS.items():
            root = component_dir(component)
            entries = {}
            for filename in expected:
                target = root / filename
                entries[filename] = {"exists": target.exists(),
                                     "bytes": target.stat().st_size if target.exists() else None}
            report[component] = {"root": str(root), "entries": entries,
                                 "complete": all(entry["exists"] for entry in entries.values())}
        return report

    def is_complete(self) -> bool:
        return (all((self.root / filename).is_file() for filename in REQUIRED_PACKAGE_FILES)
                and self.verify()["all_verified"]
                and all(entry["complete"] for entry in self.component_report().values()))

    def summary(self) -> dict:
        decoder = self.config.get("decoder", {})
        detector = self.config.get("detector", {})
        return {"name": self.name, "root": str(self.root),
                "decoder": {"architecture": decoder.get("architecture"), "seed": decoder.get("seed"),
                            "sha256": decoder.get("sha256"),
                            "selected_by": decoder.get("selected_by")},
                "detector": {"family": detector.get("family"), "imgsz": detector.get("imgsz"),
                             "conf": detector.get("conf"), "max_det": detector.get("max_det"),
                             "tta": detector.get("tta"), "sha256": detector.get("sha256")},
                "supported_programs": self.supported_programs,
                "known_limitations": self.known_limitations,
                "components": self.component_report()}


@dataclass
class ModelResolution:
    ok: bool
    package: ModelPackage | None
    used_default: bool
    requested: str | None
    error: BuildReasonSegError | None = None
    switched_with_confirmation: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "used_default": self.used_default, "requested": self.requested,
                "switched_with_confirmation": self.switched_with_confirmation,
                "error": None if self.error is None else {"code": self.error.code,
                                                          "name": self.error.name,
                                                          "detail": self.error.detail},
                "notes": list(self.notes)}


def default_package_available() -> tuple[bool, str]:
    try:
        package = ModelPackage.load(DEFAULT_MODEL)
    except BuildReasonSegError as error:
        return False, f"{error.code} {error.name}: {error.detail}"
    report = package.verify()
    if not report["all_verified"]:
        return False, "默认模型包权重校验失败"
    components = package.component_report()
    missing = [name for name, entry in components.items() if not entry["complete"]]
    if missing:
        return False, f"缺少组件资产: {', '.join(missing)}"
    return True, "默认模型包 buildreasonseg_advisor 可用"


def resolve_model(requested: str | None = None, *,
                  confirm_reader=None) -> ModelResolution:
    """Frozen fallback semantics (section 13): never switch silently."""

    if requested in (None, "", DEFAULT_MODEL):
        try:
            package = ModelPackage.load(DEFAULT_MODEL)
        except BuildReasonSegError as error:
            return ModelResolution(ok=False, package=None, used_default=True, requested=requested,
                                   error=error, notes=["默认模型不可用，推理不会继续。"])
        return ModelResolution(ok=True, package=package, used_default=True, requested=requested)

    try:
        package = ModelPackage.load(requested)
        report = package.verify()
        if not report["all_verified"]:
            raise HashMismatchError(detail=f"模型 {requested} 权重校验失败")
    except BuildReasonSegError as error:
        available, reason = default_package_available()
        if not available:
            return ModelResolution(ok=False, package=None, used_default=False, requested=requested,
                                   error=BuildReasonSegError(
                                       "E301",
                                       detail=f"指定模型不可用（{error.detail}），默认模型也不可用（{reason}）。"),
                                   notes=["请恢复默认模型，或训练/装载其他兼容模型；推理不会继续。"])
        if confirm_reader is None:
            return ModelResolution(ok=False, package=None, used_default=False, requested=requested,
                                   error=error,
                                   notes=["指定模型不可用；检测到默认模型 "
                                          f"{DEFAULT_MODEL}，需要用户确认后才可切换。"])
        answer = (confirm_reader(f"指定模型不可用。\n已检测到默认模型 {DEFAULT_MODEL}。\n\n"
                                 "是否使用默认模型继续？ [Y/N]: ") or "").strip().lower()
        if answer in ("y", "yes", "是"):
            fallback = ModelPackage.load(DEFAULT_MODEL)
            return ModelResolution(ok=True, package=fallback, used_default=True,
                                   requested=requested, switched_with_confirmation=True,
                                   notes=["用户确认后切换到默认模型。"])
        return ModelResolution(ok=False, package=None, used_default=False, requested=requested,
                               error=error, notes=["用户拒绝了默认模型切换；推理不会继续。"])
    return ModelResolution(ok=True, package=package, used_default=False, requested=requested)


__all__ = ["COMPONENTS", "ModelPackage", "ModelResolution", "OPTIONAL_PACKAGE_FILES",
           "REQUIRED_PACKAGE_FILES", "default_package_available", "resolve_model"]
