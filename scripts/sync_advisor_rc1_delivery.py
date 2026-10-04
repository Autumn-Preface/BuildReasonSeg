"""Deterministic one-way source/config sync for the BuildReasonSeg Advisor RC1 delivery.

Task 8B.2-R1. This helper copies only the manifest-listed lightweight source/config files from the Git-tracked
canonical tree to an external RC1 delivery directory. It never touches model weights, downloaded assets, images,
logs or generated outputs, and it never deletes destination content.

    python scripts/sync_advisor_rc1_delivery.py --destination <path>
    python scripts/sync_advisor_rc1_delivery.py --destination <path> --check

Canonical source is fixed at `delivery_src/BuildReasonSeg_Advisor_RC1` relative to the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / "delivery_src" / "BuildReasonSeg_Advisor_RC1"
MANIFEST_NAME = "source_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


GIT_CANONICAL_BASIS = "GIT_CANONICAL_BLOB_BYTES"
CANONICAL_PREFIX = "delivery_src/BuildReasonSeg_Advisor_RC1/"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def manifest_identity_basis(root: Path | None = None) -> str | None:
    """Return the manifest identity basis, or None for legacy manifests without one."""

    root = CANONICAL_ROOT if root is None else root
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(f"[ERROR] canonical manifest missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    basis = payload.get("identity_basis")
    if basis in (None, ""):
        return None
    if basis != GIT_CANONICAL_BASIS:
        raise SystemExit(f"[ERROR] unsupported identity_basis: {basis!r}")
    return basis


def git_canonical_bytes(relative: str) -> bytes:
    """Exact binary Git canonical bytes for a manifest-listed canonical-relative path."""

    spec = f"HEAD:{CANONICAL_PREFIX}{relative}"
    result = subprocess.run(["git", "-C", str(REPO_ROOT), "show", spec], capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] git canonical read failed for {relative}: "
                         f"{result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout


def entry_source_bytes(entry: dict, basis: str | None, root: Path) -> bytes | None:
    """Source bytes for one entry: Git canonical blob bytes, or working-tree bytes in legacy mode."""

    if basis == GIT_CANONICAL_BASIS:
        payload = git_canonical_bytes(entry["path"])
        if len(payload) != entry.get("bytes") or sha256_bytes(payload) != entry.get("sha256"):
            raise SystemExit(f"[ERROR] manifest identity mismatch: {entry['path']}")
        return payload
    source = root / entry["path"]
    if not source.is_file():
        return None
    return source.read_bytes()


def load_manifest(root: Path | None = None) -> list[dict]:
    """Load and validate the manifest; every listed path must be a safe, unique relative path."""

    root = CANONICAL_ROOT if root is None else root
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(f"[ERROR] canonical manifest missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("[ERROR] manifest has no files")
    seen: set[str] = set()
    for entry in entries:
        relative = str(entry.get("path", ""))
        if not relative:
            raise SystemExit("[ERROR] manifest entry without path")
        if Path(relative).is_absolute() or (len(relative) > 1 and relative[1] == ":"):
            raise SystemExit(f"[ERROR] absolute manifest path: {relative}")
        if ".." in Path(relative).parts:
            raise SystemExit(f"[ERROR] manifest path contains '..': {relative}")
        if relative.startswith("/") or relative.startswith("\\"):
            raise SystemExit(f"[ERROR] non-relative manifest path: {relative}")
        if relative in seen:
            raise SystemExit(f"[ERROR] duplicate manifest path: {relative}")
        seen.add(relative)
    return entries


def guard_destination(destination: Path, canonical_root: Path | None = None) -> Path:
    destination = destination.resolve()
    root = (CANONICAL_ROOT if canonical_root is None else canonical_root).resolve()
    if not destination.exists():
        raise SystemExit(f"[ERROR] destination does not exist: {destination}")
    if destination == root:
        raise SystemExit("[ERROR] destination must not be the canonical source root")
    if root in destination.parents:
        raise SystemExit("[ERROR] destination must not be inside the canonical source root")
    return destination


def run_check(destination: Path, entries: list[dict], canonical_root: Path | None = None) -> int:
    root = CANONICAL_ROOT if canonical_root is None else canonical_root
    matches = missing = mismatched = 0
    basis = manifest_identity_basis(root)
    for entry in entries:
        relative = entry["path"]
        target = destination / relative
        payload = entry_source_bytes(entry, basis, root)
        if payload is None:
            print(f"MISSING  (source) {relative}")
            missing += 1
            continue
        if not target.is_file():
            print(f"MISSING  (destination) {relative}")
            missing += 1
            continue
        destination_bytes = target.read_bytes()
        if len(payload) != len(destination_bytes) or sha256_bytes(payload) != sha256_bytes(destination_bytes):
            print(f"MISMATCH {relative}")
            mismatched += 1
            continue
        matches += 1
    print(f"\nchecked={len(entries)} match={matches} missing={missing} mismatch={mismatched}")
    return 0 if (missing == 0 and mismatched == 0) else 1


def run_sync(destination: Path, entries: list[dict], canonical_root: Path | None = None) -> int:
    root = CANONICAL_ROOT if canonical_root is None else canonical_root
    copied = verified = 0
    failures: list[str] = []
    basis = manifest_identity_basis(root)
    for entry in entries:
        relative = entry["path"]
        target = destination / relative
        payload = entry_source_bytes(entry, basis, root)
        if payload is None:
            failures.append(f"missing canonical source: {relative}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as writer:
            writer.write(payload)
        copied += 1
        check_bytes = target.read_bytes()
        if len(check_bytes) == len(payload) and sha256_bytes(check_bytes) == sha256_bytes(payload):
            verified += 1
        else:
            failures.append(f"verification failed: {relative}")
    print(f"\ncopied={copied} verified={verified} failures={len(failures)}")
    for failure in failures:
        print("  ", failure)
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize the canonical RC1 source/config snapshot to an external RC1 delivery.")
    parser.add_argument("--destination", required=True,
                        help="external RC1 delivery directory (weights/assets/logs are never touched)")
    parser.add_argument("--check", action="store_true",
                        help="read-only comparison of manifest-listed files; non-zero exit on any difference")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canonical_root = CANONICAL_ROOT
    destination = guard_destination(Path(args.destination), canonical_root)
    entries = load_manifest(canonical_root)
    print(f"canonical source : {canonical_root}")
    print(f"destination      : {destination}")
    print(f"manifest files   : {len(entries)}")
    if args.check:
        return run_check(destination, entries, canonical_root)
    return run_sync(destination, entries, canonical_root)


if __name__ == "__main__":
    sys.exit(main())
