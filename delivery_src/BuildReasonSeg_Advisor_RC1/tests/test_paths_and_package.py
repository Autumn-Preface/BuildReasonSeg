"""Portability and package-structure tests (Task 8A section 19.1)."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from buildreasonseg import paths

REQUIRED_DIRS = ("buildreasonseg", "configs", "model", "model/buildreasonseg_advisor",
                 "model/components/sam2", "model/components/program_head", "datasets",
                 "inference/input", "inference/output/masks", "inference/output/overlays",
                 "inference/output/diagnostics", "runs/train", "runs/eval", "logs", "docs", "tests")
REQUIRED_FILES = ("predict.py", "train.py", "prepare_dataset.py", "evaluate.py", "check_setup.py",
                  "README.md", "VERSION", "environment.yml", "requirements.txt", "setup_env.bat",
                  "configs/inference.yaml", "configs/train.yaml",
                  "model/buildreasonseg_advisor/model.yaml", "model/buildreasonseg_advisor/metadata.json",
                  "model/buildreasonseg_advisor/metrics.json", "docs/command_grammar.md",
                  "docs/dataset_format.md", "docs/model_card.md")


def test_delivery_root_resolution(root: Path) -> None:
    assert paths.project_root() == root
    assert (root / "buildreasonseg" / "paths.py").is_file()


def test_required_structure(root: Path) -> None:
    missing_dirs = [name for name in REQUIRED_DIRS if not (root / name).is_dir()]
    missing_files = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    assert missing_dirs == []
    assert missing_files == []


def test_runtime_paths_are_project_relative(root: Path) -> None:
    for path in (paths.default_model_dir(), paths.configs_dir(), paths.inference_dir(),
                 paths.datasets_dir(), paths.logs_dir(), paths.runs_dir("train")):
        assert str(path).startswith(str(root)), path


def test_no_workspace_reference_in_code(root: Path) -> None:
    offenders = []
    for path in list(root.rglob("*.py")) + list(root.rglob("*.yaml")) + list(root.rglob("*.json")) \
            + list(root.rglob("*.bat")):
        if any(part in ("model", "logs", "runs", "datasets", "inference") for part in
               path.relative_to(root).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if paths.workspace_references(text):
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_no_symlink_dependency(root: Path) -> None:
    links = [str(path.relative_to(root)) for path in root.rglob("*")
             if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())]
    assert links == []


def test_moving_root_keeps_config_and_model_paths(tmp_path: Path, root: Path) -> None:
    """Copy the project to another location (no assets) and confirm paths still resolve."""

    target = tmp_path / "BuildReasonSeg_Advisor_RC1"
    target.mkdir()
    for name in ("buildreasonseg", "configs", "docs", "tests"):
        shutil.copytree(root / name, target / name)
    for name in ("predict.py", "check_setup.py", "VERSION", "README.md"):
        shutil.copy2(root / name, target / name)
    (target / "model" / "buildreasonseg_advisor").mkdir(parents=True)
    shutil.copy2(root / "model" / "buildreasonseg_advisor" / "model.yaml",
                 target / "model" / "buildreasonseg_advisor" / "model.yaml")

    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from buildreasonseg import paths;"
        "print(paths.project_root());"
        "print(paths.inference_config().is_file());"
        "print(paths.default_model_dir().is_dir());" % target
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert lines[0] == str(target)
    assert lines[1] == "True"
    assert lines[2] == "True"


def test_package_imports() -> None:
    for name in ("buildreasonseg", "buildreasonseg.paths", "buildreasonseg.errors",
                 "buildreasonseg.cli", "buildreasonseg.models", "buildreasonseg.language",
                 "buildreasonseg.inference", "buildreasonseg.training", "buildreasonseg.data",
                 "buildreasonseg.diagnostics", "buildreasonseg.utils"):
        assert importlib.import_module(name) is not None


def test_supported_program_constant(root: Path) -> None:
    from buildreasonseg import SUPPORTED_PROGRAMS

    assert len(SUPPORTED_PROGRAMS) == 4
    assert all(program.startswith("largest_to_") and program.endswith("_to_nearest")
               for program in SUPPORTED_PROGRAMS)
