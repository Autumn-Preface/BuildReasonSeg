"""Dataset contract (Task 8A section 8): layout, formats and the instance-annotation requirement.

Frozen rules: raw data lives in `datasets/<name>/raw/` and is never modified; generated data goes to
`datasets/<name>/prepared/`; a full BuildReasonSeg training requires **building-instance-level annotations**;
a plain binary semantic mask that cannot reliably recover instances is refused; the scene split must happen
before tiling; an existing user split is respected, otherwise a deterministic source-scene split is planned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from buildreasonseg import paths

FORMATS = ("coco", "instance-mask", "vector")
SPLIT_ORDER = ("train", "val", "test")
INSTANCE_REQUIRED = True
SPLIT_BEFORE_TILING = True


@dataclass
class DatasetLayout:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def prepared(self) -> Path:
        return self.root / "prepared"

    def is_ready(self) -> bool:
        return self.raw.is_dir()


@dataclass
class PrepareRequest:
    dataset: Path
    format: str
    config: Path | None = None
    overwrite: bool = False
    verbose: bool = False

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.format not in FORMATS:
            problems.append(f"--format 必须是 {FORMATS} 之一。")
        if not self.dataset.exists():
            problems.append(f"数据集路径不存在: {self.dataset}")
        elif not (self.dataset / "raw").is_dir():
            problems.append(f"缺少 raw/ 目录: {self.dataset / 'raw'}")
        prepared = self.dataset / "prepared"
        if prepared.exists() and not self.overwrite:
            problems.append(f"prepared/ 已存在；如需重建请显式使用 --overwrite（不会修改 raw/）。")
        return problems


def dataset_root(name: str) -> Path:
    return paths.datasets_dir() / name


def raw_is_never_modified() -> bool:
    return True


__all__ = ["FORMATS", "INSTANCE_REQUIRED", "SPLIT_BEFORE_TILING", "SPLIT_ORDER", "DatasetLayout",
           "PrepareRequest", "dataset_root", "raw_is_never_modified"]
