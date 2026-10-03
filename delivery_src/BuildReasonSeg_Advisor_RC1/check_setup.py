"""BuildReasonSeg Advisor RC1 — environment and package self-check.

Fully functional in Task 8A. It reports Python/OS/PyTorch/CUDA/GPU, writable paths, the required directory
structure, the default model package, every frozen asset hash (decoder, detector, SAM2, Qwen/ProgramHead),
the `model.yaml` / `metadata.json` parse and the portability rule (no research-workspace dependency). A missing
or invalid item prints `[MISSING]` / `[INVALID]` / `[HASH MISMATCH]` and the exit code is non-zero unless the
final line is `BuildReasonSeg environment: READY`.

    python check_setup.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from buildreasonseg import paths
from buildreasonseg.models.package import ModelPackage
from buildreasonseg.utils.device import cuda_available, gpu_name, torch_available
from buildreasonseg.utils.hashing import verify_sha256

MIN_PYTHON = (3, 10)
REQUIRED_DIRS = ("buildreasonseg", "configs", "model", "model/buildreasonseg_advisor",
                 "model/components/sam2", "model/components/program_head", "datasets",
                 "inference/input", "inference/output/masks", "inference/output/overlays",
                 "inference/output/diagnostics", "runs/train", "runs/eval", "logs", "docs",
                 "tests")


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []

    def ok(self, label: str, extra: str = "") -> None:
        self.lines.append(f"[OK] {label}" + (f" — {extra}" if extra else ""))

    def warn(self, label: str, extra: str = "") -> None:
        self.lines.append(f"[WARN] {label}" + (f" — {extra}" if extra else ""))

    def missing(self, label: str, extra: str = "") -> None:
        self.lines.append(f"[MISSING] {label}" + (f" — {extra}" if extra else ""))
        self.failures.append(f"MISSING {label}")

    def invalid(self, label: str, extra: str = "") -> None:
        self.lines.append(f"[INVALID] {label}" + (f" — {extra}" if extra else ""))
        self.failures.append(f"INVALID {label}")

    def hash_mismatch(self, label: str, extra: str = "") -> None:
        self.lines.append(f"[HASH MISMATCH] {label}" + (f" — {extra}" if extra else ""))
        self.failures.append(f"HASH MISMATCH {label}")

    @property
    def ready(self) -> bool:
        return not self.failures


def check_python(report: Report) -> None:
    version = sys.version_info
    if (version.major, version.minor) >= MIN_PYTHON:
        report.ok("Python", f"{platform.python_version()} ({sys.executable})")
    else:
        report.invalid("Python", f"{platform.python_version()} < {MIN_PYTHON}")


def check_os(report: Report) -> None:
    report.ok("OS", f"{platform.system()} {platform.release()} ({platform.machine()})")


def check_torch(report: Report, *, require_cuda: bool = False) -> None:
    if not torch_available():
        report.missing("PyTorch", "无法 import torch")
        return
    import torch

    report.ok("PyTorch", f"{torch.__version__}")
    if cuda_available():
        report.ok("CUDA", f"可用 (torch {torch.version.cuda})")
        report.ok("GPU", gpu_name() or "unknown")
    else:
        (report.invalid if require_cuda else report.warn)("CUDA", "不可用（将使用 CPU）")


def check_paths(report: Report) -> None:
    root = paths.project_root()
    if paths.is_writable(root):
        report.ok("delivery root 可写", str(root))
    else:
        report.invalid("delivery root 不可写", str(root))
    missing = [name for name in REQUIRED_DIRS if not (root / name).is_dir()]
    if missing:
        report.missing("required directories", ", ".join(missing))
    else:
        report.ok("required directories", f"{len(REQUIRED_DIRS)} 项齐备")
    writable = []
    for directory in ("inference/output/masks", "inference/output/overlays",
                      "inference/output/diagnostics", "runs/train", "runs/eval", "logs"):
        target = root / directory
        target.mkdir(parents=True, exist_ok=True)
        if paths.is_writable(target):
            writable.append(directory)
    if len(writable) == 6:
        report.ok("writable paths", "输出 / 运行 / 日志目录均可写")
    else:
        report.invalid("writable paths", f"不可写: {set(['inference/output/masks', 'runs/train', 'logs']) - set(writable)}")


def check_model_package(report: Report) -> ModelPackage | None:
    try:
        package = ModelPackage.load("buildreasonseg_advisor")
    except Exception as error:
        report.missing("model package", str(error))
        return None
    report.ok("model package", str(package.root))
    try:
        config = package.config
        if config.get("decoder", {}).get("architecture") != "D-B1":
            report.invalid("model.yaml decoder.architecture", str(config.get("decoder")))
        else:
            report.ok("model.yaml", f"decoder={config['decoder']['architecture']} "
                                    f"seed={config['decoder'].get('seed')}")
        if not package.metadata.get("assets"):
            report.invalid("metadata.json", "缺少 assets 记录")
        else:
            report.ok("metadata.json", f"{len(package.metadata['assets'])} 个资产记录")
    except Exception as error:  # pragma: no cover - defensive
        report.invalid("model package parse", str(error))

    verification = package.verify()
    for name, entry in (("decoder", verification["decoder"]), ("detector", verification["detector"])):
        label = f"{'D-B1 decoder' if name == 'decoder' else 'U-C1 detector'}"
        if not entry["exists"]:
            report.missing(label, entry["path"])
        elif entry["matches"]:
            report.ok(label, f"sha256={entry['sha256'][:16]}… bytes={entry['bytes']}")
        elif entry.get("sha256") and entry.get("expected_sha256") \
                and entry["sha256"] != entry["expected_sha256"]:
            report.hash_mismatch(label, f"{entry['sha256'][:16]}… != {entry['expected_sha256'][:16]}…")
        elif entry.get("expected_bytes") and entry.get("bytes") != entry["expected_bytes"]:
            report.hash_mismatch(label, f"bytes {entry['bytes']} != {entry['expected_bytes']}")
        else:
            report.invalid(label, json.dumps(entry, ensure_ascii=False)[:160])
    return package


def check_components(report: Report, package: ModelPackage | None) -> None:
    from buildreasonseg.models.package import COMPONENTS
    from buildreasonseg.runtime import manifest as manifest_module

    # SAM2
    sam2_root = paths.component_dir("sam2")
    sam2_asset = sam2_root / "sam2.1_hiera_base_plus.pt"
    sam2_config = sam2_root / "sam2.1_hiera_b+.yaml"
    expected_sam2 = None
    if package is not None:
        for asset in package.metadata.get("assets", []):
            if asset.get("role") == "sam2_checkpoint":
                expected_sam2 = asset.get("sha256")
    if not sam2_asset.is_file() or not sam2_config.is_file():
        report.missing("SAM2 assets", f"{sam2_root}")
    else:
        record = verify_sha256(sam2_asset, expected_sam2)
        if record["matches"]:
            report.ok("SAM2 assets", f"{sam2_asset.name} sha256={record['sha256'][:16]}…")
        else:
            report.hash_mismatch("SAM2 assets", f"{record.get('sha256', '')[:16]}… != "
                                                f"{(expected_sam2 or '')[:16]}…")

    # Qwen / ProgramHead
    program_head_root = paths.component_dir("program_head")
    checkpoint = program_head_root / "program_parser_l3_rehearsal_v1.pt"
    qwen_root = program_head_root / "Qwen3-VL-2B-Instruct"
    qwen_files = ("config.json", "tokenizer.json", "model.safetensors")
    expected_checkpoint = None
    if package is not None:
        for asset in package.metadata.get("assets", []):
            if asset.get("role") == "program_head":
                expected_checkpoint = asset.get("sha256")
    if not checkpoint.is_file() or not qwen_root.is_dir():
        report.missing("Qwen / ProgramHead assets", f"{program_head_root}")
        return
    missing_qwen = [name for name in qwen_files if not (qwen_root / name).is_file()]
    if missing_qwen:
        report.missing("Qwen base files", ", ".join(missing_qwen))
        return
    record = verify_sha256(checkpoint, expected_checkpoint)
    if record["matches"]:
        report.ok("Qwen / ProgramHead assets",
                  f"ProgramHead sha256={record['sha256'][:16]}… + {len(qwen_files)} base files")
    else:
        report.hash_mismatch("ProgramHead checkpoint",
                            f"{record.get('sha256', '')[:16]}… != {(expected_checkpoint or '')[:16]}…")

    # Task 8B section 30: full Qwen asset manifest (every file, existence + SHA256)
    manifest = manifest_module.load_manifest()
    if manifest is None:
        report.invalid("Qwen asset manifest", f"缺少 {manifest_module.manifest_path()}")
        return
    verification = manifest_module.verify(full=True)
    if verification["ok"]:
        report.ok("Qwen asset manifest",
                  f"{verification['file_count']} 个文件全部校验通过 "
                  f"({manifest['total_bytes'] / 1e9:.2f} GB)")
    else:
        if verification["missing"]:
            report.missing("Qwen asset manifest files", ", ".join(verification["missing"][:4]))
        if verification["mismatched"]:
            report.hash_mismatch("Qwen asset manifest", json.dumps(verification["mismatched"][:3],
                                                                  ensure_ascii=False))


def check_portability(report: Report) -> None:
    root = paths.project_root()
    offenders = []
    for path in list(root.rglob("*.py")) + list(root.rglob("*.yaml")) + list(root.rglob("*.json")) \
            + list(root.rglob("*.bat")) + list(root.rglob("*.md")):
        if any(part in ("model", "logs", "runs", "datasets", "inference") for part in
               path.relative_to(root).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for marker in paths.workspace_references(text):
            offenders.append(f"{path.relative_to(root)} -> {marker}")
    # provenance documentation and reports intentionally mention the research source paths
    excluded = {"RC1_BUILD_REPORT.md", "RC1_TASK8B_REPORT.md",
                "buildreasonseg/runtime/_frozen/PORT_PROVENANCE.md"}
    offenders = [entry for entry in offenders
                 if entry.split(" -> ")[0].replace("\\", "/") not in excluded]
    if offenders:
        report.invalid("portability (no workspace dependency)", "; ".join(offenders[:5]))
    else:
        report.ok("portability", "无 workspace 依赖；项目可整体移动")


def run_checks(*, require_cuda: bool = False) -> Report:
    report = Report()
    check_python(report)
    check_os(report)
    check_torch(report, require_cuda=require_cuda)
    check_paths(report)
    package = check_model_package(report)
    check_components(report, package)
    check_portability(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BuildReasonSeg Advisor RC1 环境与模型包自检",
                                     allow_abbrev=False)
    parser.add_argument("--require-cuda", action="store_true",
                        help="要求 CUDA 可用；否则 CPU 视为可接受")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出检查结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_checks(require_cuda=args.require_cuda)
    if args.json:
        print(json.dumps({"lines": report.lines, "failures": report.failures,
                          "ready": report.ready}, ensure_ascii=False, indent=1))
    else:
        for line in report.lines:
            print(line)
        print()
        print("BuildReasonSeg environment: READY" if report.ready
              else "BuildReasonSeg environment: NOT READY")
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
