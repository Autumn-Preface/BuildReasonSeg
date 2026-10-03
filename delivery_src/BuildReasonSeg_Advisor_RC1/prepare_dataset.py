"""BuildReasonSeg Advisor RC1 — dataset preparation entry point.

Task 8A freezes the CLI contract, the directory layout (`datasets/<name>/raw/` → `datasets/<name>/prepared/`),
the "never modify raw" rule, the instance-annotation requirement and the split-before-tiling rule. The adapters
themselves belong to Task 8C.

    python prepare_dataset.py --help
    python prepare_dataset.py --dataset datasets/MyDataset --format coco
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buildreasonseg.cli.common import (  # noqa: E402
    DATASET_FORMAT_CHOICES,
    base_parser,
    not_implemented,
    print_boundary,
    resolve_config,
    run_cli,
)
from buildreasonseg.data.contract import (  # noqa: E402
    INSTANCE_REQUIRED,
    SPLIT_BEFORE_TILING,
    PrepareRequest,
)

DESCRIPTION = ("BuildReasonSeg Advisor RC1 — 数据集准备（raw → prepared）。"
               "Task 8A 仅冻结 CLI / schema / validation 契约，adapter 完整实现在 Task 8C。")


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(DESCRIPTION, config_default="configs/train.yaml")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="数据集目录（需包含 raw/ 子目录）")
    parser.add_argument("--format", choices=DATASET_FORMAT_CHOICES, required=True,
                        help="原始标注格式：coco / instance-mask / vector")
    parser.add_argument("--overwrite", action="store_true",
                        help="重建 prepared/（raw/ 永不被修改）")
    return parser


def handler(args: argparse.Namespace) -> int:
    from buildreasonseg.errors import BuildReasonSegError

    config = resolve_config(args.config)
    request = PrepareRequest(dataset=args.dataset, format=args.format, config=config,
                             overwrite=args.overwrite, verbose=args.verbose)
    problems = request.validate()
    if problems:
        raise BuildReasonSegError("E101", detail="；".join(problems))

    print(f"[数据集] {request.dataset}")
    print(f"[格式] {request.format}")
    print(f"[阶段划分] 必须在 tiling 之前完成 scene split：{SPLIT_BEFORE_TILING}")
    print(f"[标注要求] 完整 BuildReasonSeg 训练需要建筑实例级标注：{INSTANCE_REQUIRED}")
    print("[规则] raw/ 永不被修改；prepared/ 为生成目录")
    print_boundary("prepare_dataset")
    return not_implemented(f"prepare_dataset（{request.format} adapter）实现")


def main(argv: list[str] | None = None) -> int:
    return run_cli(build_parser(), handler, argv)


if __name__ == "__main__":
    sys.exit(main())
