"""Project-root path resolution and the portability contract.

Every path used by the delivery project is derived from the location of this package, so the whole
`BuildReasonSeg_Advisor_RC1` folder can be moved anywhere and still work. No path may point into the research
workspace, and no junction/symlink is required (or allowed) to make the project resolve.
"""

from __future__ import annotations

import os
from pathlib import Path

_RESEARCH_ROOT_NAME = "DeepSeekHarness"
_WORKSPACE_NAME = "workspace"

#: markers of the research repository, assembled from parts so this module never contains the literal marker
RESEARCH_MARKERS = (f"{_RESEARCH_ROOT_NAME}\\{_WORKSPACE_NAME}", f"{_RESEARCH_ROOT_NAME}/{_WORKSPACE_NAME}")


def project_root() -> Path:
    """The delivery project root: the parent directory of the `buildreasonseg` package."""

    return Path(__file__).resolve().parents[1]


def resolve(*parts: str) -> Path:
    """Resolve a project-relative path against the project root."""

    return project_root().joinpath(*parts)


def configs_dir() -> Path:
    return resolve("configs")


def model_dir(name: str | None = None) -> Path:
    base = resolve("model")
    return base if name is None else base / name


def component_dir(name: str) -> Path:
    return resolve("model", "components", name)


def datasets_dir() -> Path:
    return resolve("datasets")


def inference_dir() -> Path:
    return resolve("inference")


def runs_dir(*parts: str) -> Path:
    return resolve("runs", *parts)


def logs_dir() -> Path:
    return resolve("logs")


def docs_dir() -> Path:
    return resolve("docs")


def tests_dir() -> Path:
    return resolve("tests")


def default_model_dir() -> Path:
    return model_dir("buildreasonseg_advisor")


def inference_config() -> Path:
    return configs_dir() / "inference.yaml"


def train_config() -> Path:
    return configs_dir() / "train.yaml"


def ensure_runtime_dirs() -> list[Path]:
    """Create (if needed) every directory the runtime writes to and return them."""

    created = []
    for path in (inference_dir() / "output" / "masks", inference_dir() / "output" / "overlays",
                 inference_dir() / "output" / "diagnostics", runs_dir("train"), runs_dir("eval"),
                 logs_dir()):
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def workspace_references(text: str) -> list[str]:
    """Return the research-workspace markers found in `text` (empty means portable)."""

    lowered = text.replace("/", "\\").lower()
    return [marker for marker in RESEARCH_MARKERS if marker.replace("/", "\\").lower() in lowered]


def absolute_paths_in(text: str) -> list[str]:
    """Return Windows drive-letter paths found in `text` (portability audit helper)."""

    found = []
    for token in text.replace('"', " ").replace("'", " ").replace(",", " ").split():
        cleaned = token.strip("()[]{}")
        if len(cleaned) > 3 and cleaned[1] == ":" and cleaned[2] in "\\/":
            found.append(cleaned)
    return found


def describe() -> dict:
    root = project_root()
    return {"project_root": str(root), "python_executable": os.sys.executable,
            "model_dir": str(default_model_dir()), "configs_dir": str(configs_dir()),
            "datasets_dir": str(datasets_dir()), "inference_dir": str(inference_dir()),
            "workspace_dependency": bool(workspace_references(str(root)))}


__all__ = ["absolute_paths_in", "component_dir", "configs_dir", "datasets_dir",
           "default_model_dir", "describe", "docs_dir", "ensure_runtime_dirs",
           "inference_config", "inference_dir", "is_writable", "logs_dir", "model_dir",
           "project_root", "resolve", "runs_dir", "tests_dir", "train_config",
           "workspace_references"]
