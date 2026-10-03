"""BuildReasonSeg Advisor RC1 — training entry point.

Task 8A freezes the CLI contract and the training principles (transfer learning by default, ProgramHead never
trained on a new visual dataset, SAM2 frozen, only train+val consumed, artifacts in `runs/train/<name>/`,
deployable package in `model/<name>/`). The actual training loop belongs to a later task.

    python train.py --help
    python train.py --dataset datasets/MyDataset --name my_city_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buildreasonseg.cli.common import (  # noqa: E402
    TRAIN_INIT_CHOICES,
    TRAIN_STAGE_CHOICES,
    base_parser,
    not_implemented,
    print_boundary,
    resolve_config,
    run_cli,
)
from buildreasonseg.training.contract import (  # noqa: E402
    PROGRAM_HEAD_TRAINED_WITH_NEW_DATASET,
    SAM2_FROZEN,
    TEST_CONSUMED_AUTOMATICALLY,
    TrainingRequest,
)

DESCRIPTION = ("BuildReasonSeg Advisor RC1 — 在用户自有建筑数据集上微调检测器与 D-B1 解码器。"
               "Task 8A 仅冻结 CLI 契约与训练原则。")


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(DESCRIPTION, config_default="configs/train.yaml")
    parser.add_argument("--dataset", type=Path, required=True, help="数据集目录（prepared/ 已生成）")
    parser.add_argument("--name", required=True, help="本次训练名称（决定 runs/train/<name>/ 与 model/<name>/）")
    parser.add_argument("--stage", choices=TRAIN_STAGE_CHOICES, default="all",
                        help="训练阶段：all / detector / decoder（默认 all）")
    parser.add_argument("--init", choices=TRAIN_INIT_CHOICES, default="default",
                        help="初始化方式：default=迁移学习 / fresh=从零训练（默认 default）")
    return parser


def handler(args: argparse.Namespace) -> int:
    from buildreasonseg.errors import BuildReasonSegError

    config = resolve_config(args.config)
    request = TrainingRequest(dataset=args.dataset, name=args.name, stage=args.stage,
                              init=args.init, device=args.device, config=config,
                              verbose=args.verbose)
    problems = request.validate()
    if problems:
        raise BuildReasonSegError("E101", detail="；".join(problems))

    print(f"[数据集] {request.dataset}")
    print(f"[名称] {request.name}")
    print(f"[阶段] stage={request.stage} init={request.init} device={args.device}")
    print(f"[训练产物] {request.run_dir()}")
    print(f"[交付模型包] {request.package_dir()}（需与 buildreasonseg_advisor 同构）")
    print(f"[原则] ProgramHead 不随新视觉数据集训练={not PROGRAM_HEAD_TRAINED_WITH_NEW_DATASET}；"
          f"SAM2 frozen={SAM2_FROZEN}；不自动消费 test={not TEST_CONSUMED_AUTOMATICALLY}")
    print_boundary("train")
    return not_implemented(f"train（stage={request.stage}）实现")


def main(argv: list[str] | None = None) -> int:
    return run_cli(build_parser(), handler, argv)


if __name__ == "__main__":
    sys.exit(main())
