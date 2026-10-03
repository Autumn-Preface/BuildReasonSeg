"""Training contract (Task 8A section 9): defaults, stages and the frozen principles.

Frozen principles: transfer learning by default, ProgramHead never trained on a new visual dataset, SAM2 frozen,
only train+val consumed, test never automatic, artifacts in `runs/train/<name>/`, deployable package in
`model/<name>/` and structurally identical to `buildreasonseg_advisor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from buildreasonseg import paths

STAGES = ("all", "detector", "decoder")
INITS = ("default", "fresh")
DEFAULTS = {"stage": "all", "init": "default", "device": "auto", "config": "configs/train.yaml"}
PROGRAM_HEAD_TRAINED_WITH_NEW_DATASET = False
SAM2_FROZEN = True
TEST_CONSUMED_AUTOMATICALLY = False


@dataclass
class TrainingRequest:
    dataset: Path
    name: str
    stage: str = DEFAULTS["stage"]
    init: str = DEFAULTS["init"]
    device: str = DEFAULTS["device"]
    config: Path | None = None
    verbose: bool = False
    extra: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.stage not in STAGES:
            problems.append(f"--stage 必须是 {STAGES} 之一。")
        if self.init not in INITS:
            problems.append(f"--init 必须是 {INITS} 之一。")
        if not self.name or not self.name.strip():
            problems.append("--name 不能为空。")
        if not self.dataset.exists():
            problems.append(f"数据集路径不存在: {self.dataset}")
        return problems

    def run_dir(self) -> Path:
        return paths.runs_dir("train", self.name)

    def package_dir(self) -> Path:
        return paths.model_dir(self.name)


__all__ = ["DEFAULTS", "INITS", "PROGRAM_HEAD_TRAINED_WITH_NEW_DATASET", "SAM2_FROZEN", "STAGES",
           "TEST_CONSUMED_AUTOMATICALLY", "TrainingRequest"]
