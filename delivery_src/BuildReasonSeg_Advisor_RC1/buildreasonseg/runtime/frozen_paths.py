"""Redirect the frozen research path constants to the delivery model package.

The ported research modules (`buildreasonseg/runtime/_frozen/mvp/`) contain module-level constants that point at
the research workspace (SAM2 checkpoint, feature caches, source-data roots). `ensure()` rewrites every constant
that the *predict* runtime touches so that no research path is ever opened, then imports the frozen modules.
Import this and call `ensure()` before importing anything from `_frozen.mvp`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from buildreasonseg import component_dir, paths

#: research constants that the predict runtime must never use, mapped to delivery locations
SAM2_CHECKPOINT_TARGET = component_dir("sam2") / "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_b+.yaml"

_probe = ("set to True by ensure(); every patched research constant is listed in `patched`")
PATCHED: dict[str, str] = {}


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def frozen_mvp_import_path() -> str:
    return "buildreasonseg.runtime._frozen.mvp"


def ensure(*, feature_cache_root: Path | None = None) -> dict:
    """Patch the frozen research path constants and return a report of what was redirected."""

    import buildreasonseg.runtime._frozen.mvp.task6n_relation_decoder as relation_decoder
    import buildreasonseg.runtime._frozen.mvp.task6u_common as task6u

    PATCHED.clear()
    PATCHED["task6n_relation_decoder.SAM2_CHECKPOINT"] = str(SAM2_CHECKPOINT_TARGET)
    relation_decoder.SAM2_CHECKPOINT = SAM2_CHECKPOINT_TARGET
    PATCHED["task6n_relation_decoder.SAM2_CONFIG_NAME"] = SAM2_CONFIG_NAME
    relation_decoder.SAM2_CONFIG_NAME = SAM2_CONFIG_NAME
    # the proposal checkpoint lives in the delivery model package
    proposal = paths.default_model_dir() / "detector.pt"
    PATCHED["task6u_common.PROPOSAL_CHECKPOINT"] = str(proposal)
    task6u.PROPOSAL_CHECKPOINT = proposal
    if feature_cache_root is not None:
        target = Path(feature_cache_root)
        target.mkdir(parents=True, exist_ok=True)
        if hasattr(relation_decoder, "FEATURE_ROOT"):
            relation_decoder.FEATURE_ROOT = target
            PATCHED["task6n_relation_decoder.FEATURE_ROOT"] = str(target)
    os.environ.setdefault("SAM2_BUILD_CUDA", "0")
    return {"patched": dict(PATCHED), "sam2_checkpoint": str(SAM2_CHECKPOINT_TARGET),
            "sam2_config": SAM2_CONFIG_NAME, "proposal_checkpoint": str(proposal),
            "frozen_import_root": frozen_mvp_import_path()}


def sam2_assets_ok() -> tuple[bool, str]:
    if not SAM2_CHECKPOINT_TARGET.is_file():
        return False, f"缺少 SAM2 权重: {SAM2_CHECKPOINT_TARGET}"
    config = component_dir("sam2") / "sam2.1_hiera_b+.yaml"
    if not config.is_file():
        return False, f"缺少 SAM2 配置: {config}"
    return True, str(SAM2_CHECKPOINT_TARGET)


__all__ = ["PATCHED", "SAM2_CHECKPOINT_TARGET", "SAM2_CONFIG_NAME", "ensure",
           "frozen_mvp_import_path", "sam2_assets_ok"]
