"""Reproducible, no-inference Demo assembly. Accepted RC1 is always read-only.

Heavy files are copied into a new sibling staging directory, verified, then moved
as one folder to the previously absent destination. Nothing is deleted or merged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ACCEPTED_HEAD = "b569dcefa26af9788a9ec75eb52cbb8317305bea"
CANONICAL = "delivery_src/BuildReasonSeg_Advisor_RC1/"
SOURCE = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
FINAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V1")
STAGING = FINAL.with_name("BuildReasonSeg_Demo_V1_task8e_staging_v1")
SEEDS = ("predict", "buildreasonseg.runtime.pipeline", "buildreasonseg.runtime.program_head",
         "buildreasonseg.runtime.imageio", "buildreasonseg.models.package", "buildreasonseg.language.validator")
FORBIDDEN_PARTS = {"handoff", "governance", "evaluation", "tests", ".git", "__pycache__", ".pytest_cache"}
FORBIDDEN_FILES = {"train.py", "evaluate.py", "prepare_dataset.py"}
APP_FILES = ("BuildReasonSeg_Demo.py", "启动BuildReasonSeg Demo.bat", "使用说明.md", "版本说明.txt",
             "_ui/__init__.py", "_ui/environment.py", "_ui/backend.py", "_ui/launch.ps1")
ASSETS = ("model/buildreasonseg_advisor/decoder.pt", "model/buildreasonseg_advisor/detector.pt",
          "model/components/sam2/sam2.1_hiera_base_plus.pt", "model/components/sam2/sam2.1_hiera_b+.yaml",
          "model/components/program_head/program_parser_l3_rehearsal_v1.pt",
          "model/components/program_head/qwen_asset_manifest.json")
QWEN = "model/components/program_head/Qwen3-VL-2B-Instruct/"
QWEN_FILES = ("chat_template.json", "config.json", "generation_config.json", "merges.txt", "model.safetensors",
              "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json")


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def inventory(root: Path) -> list[dict]:
    return [{"path": p.relative_to(root).as_posix(), **identity(p), "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(root.rglob("*")) if p.is_file()]


def verify_inventory(root: Path, expected: list[dict]) -> None:
    if inventory(root) != expected:
        raise ValueError("Source RC1 inventory changed; STOP without writing it")


def module_paths(blobs: dict[str, bytes]) -> dict[str, str]:
    result = {}
    for path in blobs:
        if not path.endswith(".py"):
            continue
        parts = Path(path).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        result[".".join(parts)] = path
    return result


def dependency_closure(blobs: dict[str, bytes], seeds=SEEDS) -> tuple[list[str], dict]:
    """Conservative AST closure includes imports inside functions and required package initializers."""
    modules = module_paths(blobs)
    pending, seen, edges = list(seeds), set(), {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        if name not in modules:
            raise ValueError("Required source module missing: " + name)
        seen.add(name)
        path = modules[name]
        package = name if path.endswith("/__init__.py") else name.rpartition(".")[0]
        dependencies = set()
        for i in range(1, len(name.split("."))):
            parent = ".".join(name.split(".")[:i])
            if parent in modules:
                dependencies.add(parent)
        tree = ast.parse(blobs[path].decode("utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package.split(".") if package else []
                if node.level:
                    prefix = prefix[:len(prefix) - (node.level - 1)]
                    base = ".".join(prefix + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                candidates = [base] + [base + "." + alias.name for alias in node.names if alias.name != "*"]
            else:
                continue
            dependencies.update(candidate for candidate in candidates if candidate in modules)
        edges[name] = sorted(dependencies)
        pending.extend(dependencies - seen)
    paths = sorted(modules[name] for name in seen)
    return paths, {"seeds": list(seeds), "modules": sorted(seen), "import_edges": edges,
                   "policy": "AST imports at all depths plus package initializers; no runtime or model execution"}


def reject_link(path: Path) -> None:
    if path.is_symlink() or (getattr(path.stat(), "st_file_attributes", 0) & 0x400):
        raise ValueError("Links/reparse points are not allowed: " + str(path))


def forbidden_audit(root: Path) -> dict:
    offenders = []
    for p in root.rglob("*"):
        relative = p.relative_to(root)
        parts = {part.lower() for part in relative.parts}
        if parts & FORBIDDEN_PARTS or p.name.lower() in FORBIDDEN_FILES or p.name.lower().startswith(".git"):
            offenders.append(relative.as_posix())
        if p.is_file() and (p.name.lower().startswith("test_") or "task8c" in p.name.lower() or "task8d" in p.name.lower()):
            offenders.append(relative.as_posix())
        reject_link(p)
    for path in (root / "使用说明.md", root / "版本说明.txt"):
        text = path.read_text(encoding="utf-8")
        if any(word.lower() in text.lower() for word in ("Task8", "Codex", "Supervisor", "handoff", "governance")):
            offenders.append(path.name + ":developer_language")
    return {"pass": not offenders, "offenders": sorted(set(offenders)), "links": 0}


def safe_destination(source: Path, staging: Path, final: Path) -> None:
    source, staging, final = source.resolve(), staging.resolve(), final.resolve()
    if source == staging or source == final or source in staging.parents or source in final.parents:
        raise ValueError("Destination must be outside the accepted RC1")
    if staging in source.parents or final in source.parents:
        raise ValueError("Destination must not contain source RC1")
    if staging.parent != final.parent:
        raise ValueError("Use a same-volume sibling staging folder")
    if staging.exists() or final.exists():
        raise FileExistsError("Preserve pre-existing delivery/staging; no overwrite or cleanup")
    if not source.is_dir() or not staging.parent.is_dir():
        raise ValueError("Source and delivery parent must exist")


def assemble(source: Path, staging: Path, final: Path, blobs: dict, source_paths: list[str],
             asset_paths: list[str], expected_inventory: list[dict], app_source: Path, python: str) -> dict:
    safe_destination(source, staging, final)
    verify_inventory(source, expected_inventory)
    expected = {r["path"]: r for r in expected_inventory}
    selected = source_paths + asset_paths
    for relative in selected:
        p = source / relative
        reject_link(p)
        if relative not in expected or identity(p) != {k: expected[relative][k] for k in ("bytes", "sha256")}:
            raise ValueError("Unexplained source drift: " + relative)
        if relative in blobs and p.read_bytes() != blobs[relative]:
            raise ValueError("Accepted Git-canonical source/config differs: " + relative)
    if shutil.disk_usage(staging.parent).free < sum(expected[p]["bytes"] for p in selected) + (32 << 20):
        raise OSError("Insufficient space; source RC1 untouched")
    staging.mkdir()
    records = []
    def copy_file(origin: Path, relative: str, role: str):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with origin.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1 << 20)
        original_identity, copied = identity(origin), identity(target)
        if original_identity != copied:
            raise ValueError("Copied identity mismatch: " + relative)
        records.append({"path": relative, **copied, "role": role})
    for relative in source_paths:
        copy_file(source / relative, "_engine/" + relative, "accepted_engine_source")
    print(f"Verified {len(source_paths)} accepted source/config files", flush=True)
    for relative in asset_paths:
        copy_file(source / relative, "_engine/" + relative, "accepted_engine_version" if relative == "VERSION" else "accepted_model_asset")
        print("Verified asset " + relative, flush=True)
    for relative in APP_FILES:
        copy_file(app_source / relative, relative, "user_demo_source")
    config = staging / "_ui/runtime.json"
    config.write_text(json.dumps({"validated_python": python, "runtime_environment_bundled": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records.append({"path": "_ui/runtime.json", **identity(config), "role": "machine_python_locator_only"})
    for name in ("results", "_engine/inference/input", "_engine/inference/output", "_engine/logs", "_runtime_cache/yolo", "_runtime_cache/matplotlib"):
        (staging / name).mkdir(parents=True, exist_ok=True)
    manifest = {"demo_version": "V1", "files": sorted(records, key=lambda r: r["path"]),
                "total_bytes": sum(r["bytes"] for r in records), "file_count": len(records),
                "runtime_environment_bundled": False, "source_model_portability": "PATH_PORTABLE_ON_CONFIGURED_MACHINE"}
    (staging / "_engine/package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = forbidden_audit(staging)
    if not audit["pass"]:
        raise ValueError("Forbidden-content audit failed: " + str(audit))
    for directory in ("results", "_engine/inference/output", "_engine/logs"):
        if any((staging / directory).iterdir()):
            raise ValueError("New package runtime state must be empty")
    verify_inventory(source, expected_inventory)
    return {"manifest": manifest, "staging": str(staging), "final": str(final), "forbidden_content_audit": audit,
            "source_unchanged": True, "clean_results_output_logs": True, "move_pending": True}


def accepted_blobs() -> dict:
    def git_read(relative):
        return subprocess.run(["git", "-C", str(REPO), "show", ACCEPTED_HEAD + ":" + CANONICAL + relative],
                              capture_output=True, check=True).stdout
    manifest = json.loads(git_read("source_manifest.json"))
    result = {}
    for row in manifest["files"]:
        b = git_read(row["path"])
        if len(b) != row["bytes"] or hashlib.sha256(b).hexdigest() != row["sha256"]:
            raise ValueError("Git-canonical manifest anomaly: " + row["path"])
        result[row["path"]] = b
    return result


def finalize(source: Path, staging: Path, final: Path, manifest: dict) -> dict:
    source, staging, final = source.resolve(), staging.resolve(), final.resolve()
    # Verify the absolute move endpoints stay in the explicitly named sibling delivery scope.
    if staging.parent != final.parent or source in staging.parents or source in final.parents:
        raise ValueError("Unsafe move endpoints")
    if staging == source or final == source or staging in source.parents or final in source.parents:
        raise ValueError("Never move or replace accepted RC1")
    if final.exists() or not staging.is_dir():
        raise FileExistsError("Preserve existing final delivery or missing staging")
    reject_link(staging)
    for record in manifest["files"]:
        path = (staging / record["path"]).resolve()
        if not path.is_relative_to(staging) or identity(path) != {k: record[k] for k in ("bytes", "sha256")}:
            raise ValueError("Staging package identity changed: " + record["path"])
    if not forbidden_audit(staging)["pass"]:
        raise ValueError("Staging forbidden content")
    staging.rename(final)  # one folder move, not another model copy; never delete an existing delivery
    return {"pass": True, "source": str(staging), "destination": str(final),
            "models_copied_again": False, "source_assets_resolve_from_final_root": True}


def main():
    parser = argparse.ArgumentParser(description="Build authorized user Demo without scientific inference")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--assemble", action="store_true")
    operation.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    blobs = accepted_blobs()
    closure, analysis = dependency_closure(blobs)
    source_paths = sorted(set(closure) | {"configs/inference.yaml", "model/buildreasonseg_advisor/model.yaml", "model/buildreasonseg_advisor/metadata.json"})
    assets = list(ASSETS) + [QWEN + name for name in QWEN_FILES] + ["VERSION"]
    evidence_path = REPO / "evaluation/task8e_user_demo_delivery_v1.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    plan = {"accepted_head": ACCEPTED_HEAD, "engine_sources": source_paths, "assets": assets,
            "static_dependency_analysis": analysis, "omitted_optional_model_files": ["metrics.json", "qwen_integrity_cache.json"],
            "no_runtime_execution": True}
    evidence["assembly_plan"] = plan
    if args.assemble:
        result = assemble(SOURCE, STAGING, FINAL, blobs, source_paths, assets,
                          evidence["preflight"]["advisor_rc1_inventory"], REPO / "demo/user_demo_v1", sys.executable)
        evidence["assembly"] = result
        evidence["package_manifest"] = result["manifest"]
        evidence["source_asset_identity"] = [r for r in result["manifest"]["files"] if r["role"] in ("accepted_engine_source", "accepted_engine_version")]
        evidence["model_asset_identity"] = [r for r in result["manifest"]["files"] if r["role"] == "accepted_model_asset"]
    if args.finalize:
        evidence["folder_move"] = finalize(SOURCE, STAGING, FINAL, evidence["package_manifest"])
        evidence["assembly"]["move_pending"] = False
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"engine_source_files": len(source_paths), "assets": len(assets), "assembled": args.assemble}, ensure_ascii=False))


if __name__ == "__main__":
    main()
