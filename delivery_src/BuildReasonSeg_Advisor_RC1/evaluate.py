"""BuildReasonSeg Advisor RC1 — evaluation entry point.

Task 8A freezes the CLI contract: `--split` defaults to `val` and `test` can only be reached by an explicit
`--split test`; `--reference-mode` distinguishes the practical predicted-reference chain from the
oracle-reference diagnostic. No evaluation or test run is performed in Task 8A.

    python evaluate.py --help
    python evaluate.py --model buildreasonseg_advisor --dataset datasets/MyDataset --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buildreasonseg import DEFAULT_MODEL
from buildreasonseg.cli.common import (  # noqa: E402
    REFERENCE_MODE_CHOICES,
    SPLIT_CHOICES,
    base_parser,
    not_implemented,
    print_boundary,
    resolve_config,
    run_cli,
)

DESCRIPTION = ("BuildReasonSeg Advisor RC1 — 在数据集上评估模型（默认使用 val）。"
               "Task 8A 仅冻结 CLI 契约，评估实现与任何 test 运行均不在本任务范围内。")


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(DESCRIPTION, config_default="configs/train.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型包名称（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--dataset", type=Path, required=True, help="数据集目录")
    parser.add_argument("--split", choices=SPLIT_CHOICES, default="val",
                        help="评估划分：train / val / test（默认 val；test 必须显式指定）")
    parser.add_argument("--reference-mode", choices=REFERENCE_MODE_CHOICES, default="both",
                        dest="reference_mode",
                        help="参考来源：oracle / predicted / both（默认 both，且必须分开报告）")
    return parser


def handler(args: argparse.Namespace) -> int:
    from buildreasonseg.errors import BuildReasonSegError

    config = resolve_config(args.config)
    if not config.is_file():
        raise BuildReasonSegError("E101", detail=f"配置文件不存在: {config}")
    if not args.dataset.exists():
        raise BuildReasonSegError("E101", detail=f"数据集路径不存在: {args.dataset}")

    print(f"[模型] {args.model}")
    print(f"[数据集] {args.dataset}")
    print(f"[划分] {args.split}（默认 val；test 仅能由用户显式指定）")
    print(f"[参考模式] {args.reference_mode}"
          + ("（oracle 与 predicted 必须分开报告，不得合并为一个数字）"
             if args.reference_mode == "both" else ""))
    if args.split == "test":
        print("[提示] 这是用户显式请求的 test 评估；RC1 不对 test 做任何自动运行或调参。")
    print_boundary("evaluate")
    return not_implemented(f"evaluate（split={args.split}, reference-mode={args.reference_mode}）实现")


def main(argv: list[str] | None = None) -> int:
    return run_cli(build_parser(), handler, argv)


if __name__ == "__main__":
    sys.exit(main())
