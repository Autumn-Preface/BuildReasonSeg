"""Task 8B.2-R1 tests for the RC1 canonical source sync helper.

Temporary directories only: no model weights, no final test data, no network, no external RC1 delivery access.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sync_advisor_rc1_delivery as sync  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def canonical(tmp_path: Path, monkeypatch) -> Path:
    """A miniature canonical tree + manifest in a temporary directory."""

    root = tmp_path / "delivery_src" / "BuildReasonSeg_Advisor_RC1"
    (root / "buildreasonseg").mkdir(parents=True)
    (root / "check_setup.py").write_bytes(b"print('setup')\n")
    (root / "buildreasonseg" / "runtime.py").write_bytes(b"VALUE = 1\n")
    manifest = {
        "schema": "BuildReasonSeg.AdvisorRC1.SourceManifest.v1",
        "task": "unit-test",
        "files": [{"path": "buildreasonseg/runtime.py", "bytes": 10,
                   "sha256": sha256(root / "buildreasonseg" / "runtime.py")},
                  {"path": "check_setup.py", "bytes": 15,
                   "sha256": sha256(root / "check_setup.py")}],
    }
    (root / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(sync, "CANONICAL_ROOT", root)
    return root


@pytest.fixture()
def destination(tmp_path: Path) -> Path:
    target = tmp_path / "external_delivery"
    target.mkdir()
    return target


def test_manifest_paths_relative_and_normalized(canonical: Path) -> None:
    entries = sync.load_manifest(canonical)
    for entry in entries:
        path = Path(entry["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert entry["path"] == path.as_posix()
        assert entry["bytes"] == (canonical / entry["path"]).stat().st_size
        assert entry["sha256"] == sha256(canonical / entry["path"])


def test_duplicate_manifest_path_rejected(canonical: Path) -> None:
    manifest = json.loads((canonical / "source_manifest.json").read_text(encoding="utf-8"))
    manifest["files"].append(dict(manifest["files"][0]))
    (canonical / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit):
        sync.load_manifest(canonical)


def test_dotdot_path_rejected(canonical: Path) -> None:
    manifest = json.loads((canonical / "source_manifest.json").read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../outside.py"
    (canonical / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit):
        sync.load_manifest(canonical)


def test_absolute_path_rejected(canonical: Path) -> None:
    manifest = json.loads((canonical / "source_manifest.json").read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "C:/absolute/path.py"
    (canonical / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit):
        sync.load_manifest(canonical)


def test_normal_sync_copies_listed_files(canonical: Path, destination: Path) -> None:
    assert sync.main(["--destination", str(destination)]) == 0
    for entry in sync.load_manifest(canonical):
        target = destination / entry["path"]
        assert target.is_file()
        assert sha256(target) == entry["sha256"]


def test_normal_sync_overwrites_only_listed_counterpart(canonical: Path, destination: Path) -> None:
    target = destination / "check_setup.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"stale content")
    assert sync.main(["--destination", str(destination)]) == 0
    assert target.read_bytes() == (canonical / "check_setup.py").read_bytes()


def test_unlisted_destination_file_preserved(canonical: Path, destination: Path) -> None:
    keep = destination / "model" / "buildreasonseg_advisor" / "decoder.pt"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_bytes(b"weights-must-survive")
    extra = destination / "logs" / "run.txt"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"log-must-survive")
    assert sync.main(["--destination", str(destination)]) == 0
    assert keep.read_bytes() == b"weights-must-survive"
    assert extra.read_bytes() == b"log-must-survive"


def test_check_passes_on_exact_match(canonical: Path, destination: Path) -> None:
    assert sync.main(["--destination", str(destination)]) == 0
    assert sync.main(["--destination", str(destination), "--check"]) == 0


def test_check_detects_missing_destination_file(canonical: Path, destination: Path) -> None:
    assert sync.main(["--destination", str(destination)]) == 0
    (destination / "check_setup.py").unlink()
    assert sync.main(["--destination", str(destination), "--check"]) != 0


def test_check_detects_mismatched_destination_file(canonical: Path, destination: Path) -> None:
    assert sync.main(["--destination", str(destination)]) == 0
    (destination / "check_setup.py").write_bytes(b"different content\n")
    assert sync.main(["--destination", str(destination), "--check"]) != 0


def test_source_equals_destination_rejected(canonical: Path) -> None:
    with pytest.raises(SystemExit):
        sync.guard_destination(canonical)


def test_destination_inside_source_rejected(canonical: Path) -> None:
    nested = canonical / "buildreasonseg"
    nested.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit):
        sync.guard_destination(nested)


def test_destination_must_exist(canonical: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        sync.guard_destination(tmp_path / "does_not_exist")


def test_check_does_not_modify_destination(canonical: Path, destination: Path) -> None:
    target = destination / "check_setup.py"
    target.write_bytes(b"pre-existing content")
    before = sha256(target)
    assert sync.main(["--destination", str(destination), "--check"]) != 0
    assert sha256(target) == before


# ---------------------------------------------------------------- canonical policy (Task 8B.2-R1.1)

REAL_MANIFEST = REPO_ROOT / "delivery_src" / "BuildReasonSeg_Advisor_RC1" / "source_manifest.json"
REAL_CANONICAL = REPO_ROOT / "delivery_src" / "BuildReasonSeg_Advisor_RC1"
QWEN_SUBTREE = "model/components/program_head/Qwen3-VL-2B-Instruct/"
INTEGRITY_CACHE = "model/components/program_head/qwen_integrity_cache.json"
SAM2_CONFIG = "model/components/sam2/sam2.1_hiera_b+.yaml"
ASSET_MANIFEST = "model/components/program_head/qwen_asset_manifest.json"


@pytest.fixture(scope="module")
def real_manifest() -> dict:
    return json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))


def test_canonical_manifest_excludes_qwen_downloaded_subtree(real_manifest: dict) -> None:
    assert not [entry["path"] for entry in real_manifest["files"]
                if entry["path"].startswith(QWEN_SUBTREE)]


def test_canonical_manifest_excludes_generated_integrity_cache(real_manifest: dict) -> None:
    paths = {entry["path"] for entry in real_manifest["files"]}
    assert INTEGRITY_CACHE not in paths


def test_canonical_manifest_excludes_downloaded_sam2_config(real_manifest: dict) -> None:
    paths = {entry["path"] for entry in real_manifest["files"]}
    assert SAM2_CONFIG not in paths


def test_canonical_tree_has_no_qwen_subtree_directory() -> None:
    assert not (REAL_CANONICAL / QWEN_SUBTREE).exists()


def test_canonical_tree_has_no_integrity_cache_file() -> None:
    assert not (REAL_CANONICAL / INTEGRITY_CACHE).exists()


def test_canonical_tree_has_no_sam2_config_file() -> None:
    assert not (REAL_CANONICAL / SAM2_CONFIG).exists()


def test_canonical_tree_keeps_project_generated_asset_manifest() -> None:
    assert (REAL_CANONICAL / ASSET_MANIFEST).is_file()
    assert ASSET_MANIFEST in {entry["path"] for entry in
                               json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))["files"]}


def test_canonical_manifest_has_exactly_135_entries(real_manifest: dict) -> None:
    assert len(real_manifest["files"]) == 135


def test_every_real_manifest_entry_matches_canonical_file(real_manifest: dict) -> None:
    for entry in real_manifest["files"]:
        target = REAL_CANONICAL / entry["path"]
        assert target.is_file(), entry["path"]
        assert target.stat().st_size == entry["bytes"], entry["path"]
        assert sha256(target) == entry["sha256"], entry["path"]
