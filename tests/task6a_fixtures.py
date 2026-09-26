"""Shared fixtures for the Task 6A tests.

The model-level tests need the downloaded assets. They skip cleanly when the
assets are absent, so `pytest tests/` never forces a download:

* `Qwen3-VL-2B-Instruct` snapshot present under `local_cache/huggingface/hub`;
* `sam2.1_hiera_base_plus.pt` present under `local_cache/models`.

Set `TASK6A_SKIP_MODEL_TESTS=1` to skip them even when the assets exist.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QWEN_SNAPSHOT_GLOB = "models--Qwen--Qwen3-VL-2B-Instruct"
SAM2_CHECKPOINT = REPO_ROOT / "local_cache" / "models" / "sam2.1_hiera_base_plus.pt"
CONFIG_PATH = REPO_ROOT / "configs" / "mvp" / "task6a_2b_seg.yaml"
SUBSET_IDS = REPO_ROOT / "evaluation" / "task6a_subset_ids.json"

_RUNTIME = None


def qwen_assets_present() -> bool:
    hub = REPO_ROOT / "local_cache" / "huggingface" / "hub"
    return any((hub / name).is_dir() for name in (QWEN_SNAPSHOT_GLOB,)) if hub.is_dir() else False


def sam2_assets_present() -> bool:
    return SAM2_CHECKPOINT.is_file()


def model_tests_disabled() -> bool:
    return os.environ.get("TASK6A_SKIP_MODEL_TESTS") == "1"


def require_model_assets() -> None:
    """Raise a pytest skip when the assets or the opt-out say so."""

    import pytest

    if model_tests_disabled():
        pytest.skip("TASK6A_SKIP_MODEL_TESTS=1")
    if not qwen_assets_present():
        pytest.skip("Qwen3-VL-2B-Instruct snapshot not present under local_cache/")
    if not sam2_assets_present():
        pytest.skip("sam2.1_hiera_base_plus.pt not present under local_cache/models")


@lru_cache(maxsize=1)
def runtime():
    """One lazily-built runtime shared by every model-level test.

    No TLS mutation happens here: Task 6B removed the automatic certifi merge, so
    building the runtime only reads the standard trust store.
    """

    from buildreasonseg_mvp.runtime import build_runtime, load_config

    return build_runtime(load_config(CONFIG_PATH), device="cuda", verbose=False)


def smoke_samples():
    import json

    from buildreasonseg_mvp import data as data_mod

    payload = json.loads(SUBSET_IDS.read_text(encoding="utf-8"))
    records = {r["sample_id"]: r for r in data_mod.read_records("train")}
    return [data_mod.to_sample(records[sid]) for sid in payload["smoke_pair"]["sample_ids"]]


def overfit_samples():
    import json

    from buildreasonseg_mvp import data as data_mod

    payload = json.loads(SUBSET_IDS.read_text(encoding="utf-8"))
    records = {r["sample_id"]: r for r in data_mod.read_records("train")}
    return [data_mod.to_sample(records[sid]) for sid in payload["overfit_set"]["sample_ids"]]
