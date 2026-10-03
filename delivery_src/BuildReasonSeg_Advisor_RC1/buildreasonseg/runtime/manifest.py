"""Qwen / ProgramHead asset manifest and runtime integrity checks (Task 8B section 30).

The manifest records every file of the delivery Qwen base plus the ProgramHead checkpoint with
`relative path / bytes / SHA256`. `check_setup.py` verifies all hashes; the predict startup path verifies
existence for all entries and hashes the small files immediately, using a cached integrity state for the large
`safetensors` weight (recorded by the full check) so startup stays fast without ever accepting a missing file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from buildreasonseg import component_dir
from buildreasonseg.utils.hashing import sha256_file

MANIFEST_NAME = "qwen_asset_manifest.json"
QWEN_DIRNAME = "Qwen3-VL-2B-Instruct"
PROGRAM_HEAD_CHECKPOINT = "program_parser_l3_rehearsal_v1.pt"
LARGE_FILE_BYTES = 64 * 1024 * 1024
CACHE_NAME = "qwen_integrity_cache.json"


def manifest_path() -> Path:
    return component_dir("program_head") / MANIFEST_NAME


def cache_path() -> Path:
    return component_dir("program_head") / CACHE_NAME


def build_manifest(*, verify: bool = True) -> dict:
    root = component_dir("program_head")
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in (MANIFEST_NAME, CACHE_NAME):
            continue
        relative = path.relative_to(root)
        entry = {"path": str(relative).replace("\\", "/"), "bytes": path.stat().st_size}
        if verify:
            entry["sha256"] = sha256_file(path)
        entries.append(entry)
    payload = {"_doc": "Task 8B section 30 Qwen / ProgramHead runtime asset manifest.",
               "root": "model/components/program_head",
               "file_count": len(entries),
               "total_bytes": sum(entry["bytes"] for entry in entries),
               "files": entries,
               "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return payload


def write_manifest(payload: dict | None = None) -> dict:
    payload = payload or build_manifest()
    target = manifest_path()
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def load_manifest() -> dict | None:
    target = manifest_path()
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def _large_entry(manifest: dict) -> dict | None:
    for entry in manifest.get("files", []):
        if entry["bytes"] >= LARGE_FILE_BYTES:
            return entry
    return None


def verify(*, full: bool = True) -> dict:
    """Verify existence always; verify every hash when `full`, otherwise the small files + cache."""

    manifest = load_manifest()
    if manifest is None:
        return {"ok": False, "reason": "manifest_missing", "manifest": str(manifest_path())}
    root = component_dir("program_head")
    missing = []
    mismatched = []
    checked = 0
    large = _large_entry(manifest)
    cache = {}
    if cache_path().is_file():
        cache = json.loads(cache_path().read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        target = root / entry["path"]
        if not target.is_file():
            missing.append(entry["path"])
            continue
        if target.stat().st_size != entry["bytes"]:
            mismatched.append({"path": entry["path"], "reason": "size"})
            continue
        if not full and large is not None and entry["path"] == large["path"]:
            cached = cache.get(entry["path"])
            if cached == entry["sha256"]:
                checked += 1
                continue
        digest = sha256_file(target)
        if entry.get("sha256") and digest != entry["sha256"]:
            mismatched.append({"path": entry["path"], "reason": "sha256"})
        else:
            checked += 1
    if full and large is not None and not mismatched and not missing:
        cache[large["path"]] = large["sha256"]
        cache_path().write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": not missing and not mismatched, "missing": missing, "mismatched": mismatched,
            "checked": checked, "file_count": manifest["file_count"], "full_hash": full,
            "manifest": str(manifest_path()), "large_file": None if large is None else large["path"]}


__all__ = ["CACHE_NAME", "LARGE_FILE_BYTES", "MANIFEST_NAME", "PROGRAM_HEAD_CHECKPOINT",
           "QWEN_DIRNAME", "build_manifest", "cache_path", "load_manifest", "manifest_path",
           "verify", "write_manifest"]
