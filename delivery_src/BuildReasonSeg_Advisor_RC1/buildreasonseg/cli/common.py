"""Shared CLI helpers: one stable argparse style, short user-facing errors, logged tracebacks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from buildreasonseg import paths
from buildreasonseg.errors import BuildReasonSegError, NotInTask8AError
from buildreasonseg.utils.logging import log_exception

DEVICE_CHOICES = ("auto", "cpu", "cuda")
REFERENCE_MODE_CHOICES = ("oracle", "predicted", "both")
SPLIT_CHOICES = ("train", "val", "test")
DATASET_FORMAT_CHOICES = ("coco", "instance-mask", "vector")
TRAIN_STAGE_CHOICES = ("all", "detector", "decoder")
TRAIN_INIT_CHOICES = ("default", "fresh")

NOT_IMPLEMENTED_STAGE = ("Task 8A foundation build: interface and contract frozen, "
                         "execution pending Task 8B/8C")


def base_parser(description: str, *, config_default: str) -> argparse.ArgumentParser:
    """A parser with the shared `--config` / `--device` / `--verbose` options and `-h` support."""

    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--config", default=config_default,
                        help=f"配置文件路径（默认 {config_default}，相对交付根目录解析）")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                        help="计算设备：auto / cpu / cuda（默认 auto）")
    parser.add_argument("--verbose", action="store_true", help="输出更详细的运行日志")
    return parser


def resolve_config(value: str) -> Path:
    """Resolve a config argument against the project root, never the current working directory."""

    candidate = Path(value)
    return candidate if candidate.is_absolute() else paths.resolve(value)


def require_existing(path: Path, *, code: str, what: str) -> None:
    if not path.exists():
        raise BuildReasonSegError(code, detail=f"{what}不存在: {path}")


def not_implemented(what: str) -> int:
    """Raise the standard Task 8A placeholder error (non-zero exit, no fabricated result)."""

    raise NotInTask8AError(detail=f"{what}：{NOT_IMPLEMENTED_STAGE}")


def print_boundary(what: str) -> None:
    print(f"[Task 8A] {what}")
    print(f"[Task 8A] {NOT_IMPLEMENTED_STAGE}")


def run_cli(parser: argparse.ArgumentParser, handler: Callable[[argparse.Namespace], int],
            argv: list[str] | None = None) -> int:
    """Parse then execute, converting user-facing errors into short messages and exit codes.

    argparse itself keeps the standard behaviour: `-h/--help` prints help and exits 0, a usage error exits 2.
    """

    namespace = parser.parse_args(argv)
    try:
        return int(handler(namespace))
    except BuildReasonSegError as error:
        for line in error.user_lines():
            print(line, file=sys.stderr)
        log_path = log_exception(error, context=error.name)
        print(f"  详细日志: {log_path}", file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        print("[E901 USER_ABORTED] 用户中止了当前操作。", file=sys.stderr)
        return 90
    except Exception as error:  # pragma: no cover - unexpected failures still stay user-friendly
        from buildreasonseg.errors import ERROR_REGISTRY

        print(f"[E502 INFERENCE_RUNTIME_ERROR] {ERROR_REGISTRY['E502']['message']}", file=sys.stderr)
        log_path = log_exception(error, context="unhandled")
        print(f"  详细日志: {log_path}", file=sys.stderr)
        return 50


__all__ = ["DATASET_FORMAT_CHOICES", "DEVICE_CHOICES", "NOT_IMPLEMENTED_STAGE",
           "REFERENCE_MODE_CHOICES", "SPLIT_CHOICES", "TRAIN_INIT_CHOICES",
           "TRAIN_STAGE_CHOICES", "base_parser", "not_implemented", "print_boundary",
           "require_existing", "resolve_config", "run_cli"]
