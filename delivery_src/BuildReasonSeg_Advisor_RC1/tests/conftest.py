"""Shared pytest fixtures for the RC1 delivery tests.

The tests are written to run from any location: the delivery root is discovered from the package location, so
the whole project can be moved (or copied to a temporary path by the portability smoke test) without changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RC1_ROOT = Path(__file__).resolve().parents[1]
if str(RC1_ROOT) not in sys.path:
    sys.path.insert(0, str(RC1_ROOT))

from buildreasonseg import paths  # noqa: E402

REQUIRED_DIRS = ("buildreasonseg", "configs", "model", "model/buildreasonseg_advisor",
                 "model/components/sam2", "model/components/program_head", "datasets",
                 "inference/input", "inference/output/masks", "inference/output/overlays",
                 "inference/output/diagnostics", "runs/train", "runs/eval", "logs", "docs", "tests")


@pytest.fixture(scope="session")
def root() -> Path:
    return RC1_ROOT


@pytest.fixture(scope="session")
def project_paths():
    return paths
