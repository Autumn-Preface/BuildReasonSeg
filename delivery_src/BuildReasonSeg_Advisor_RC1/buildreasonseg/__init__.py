"""BuildReasonSeg Advisor RC1 — local delivery package.

The package is self-contained: every path is resolved from the project root (the directory that contains this
package), never from the research repository and never from an absolute machine path.

Language chain contract (frozen in Task 8A):

    user prompt -> Qwen / ProgramHead -> parsed program -> hard validator -> execution or suggestion flow

The deterministic parser is **fallback only** (Qwen missing / load failure / runtime failure) and can never
become the normal path.
"""

from __future__ import annotations

from .paths import (  # noqa: F401
    component_dir,
    configs_dir,
    datasets_dir,
    default_model_dir,
    inference_dir,
    logs_dir,
    model_dir,
    project_root,
    resolve,
    runs_dir,
)

__version__ = "RC1"
__product__ = "BuildReasonSeg Advisor"

SUPPORTED_PROGRAMS = (
    "largest_to_left_of_to_nearest",
    "largest_to_right_of_to_nearest",
    "largest_to_above_to_nearest",
    "largest_to_below_to_nearest",
)

DEFAULT_MODEL = "buildreasonseg_advisor"

__all__ = ["DEFAULT_MODEL", "SUPPORTED_PROGRAMS", "__product__", "__version__", "component_dir",
           "configs_dir", "datasets_dir", "default_model_dir", "inference_dir", "logs_dir",
           "model_dir", "project_root", "resolve", "runs_dir"]
