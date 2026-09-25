#!/usr/bin/env python
"""CLI: split manifest provenance into generation-time vs current-source.

    python scripts/split_manifest_provenance.py [--check] [--quiet]
    python scripts/split_manifest_provenance.py --current-commit <sha>

Task 5.5 section 3.1. Task 5C edited `scripts/build_spatial_reason.py` *after*
the v0.1.1 JSONL had already been generated, then refreshed that file's digest in
`manifest.json`. That mixed two different facts into one map:

* the source that actually produced the frozen 25,229 records;
* the source currently present in the repository.

This tool separates them:

```
generation_source:  commit + file_sha256   <- Git objects at the generation commit
current_source:     commit + file_sha256   <- Git objects at the snapshot commit
```

Both blocks are read from the **Git object store** and hashed after
line-ending normalisation (CRLF -> LF), so they are comparable, reproducible and
independent of `core.autocrlf`. A raw on-disk hash is *not* used, because on
Windows git may materialise CRLF in the working tree: `spatial_reasoning/relations.py`
is CRLF on disk while its committed blob is LF. That difference is line-ending
noise, not a code difference, and the legacy `generator_file_sha256` map was
polluted by exactly that (`ec0fd613…` = CRLF form instead of the blob's
`d45688a6…`).

The generation-time bytes are reconstructed from Git history, so they describe
code that provably existed at that point and cannot be silently refreshed.

Safety: refuses to run if the frozen JSONL hashes have moved, because then the
"generation commit" premise would no longer describe the accepted dataset.

`--check` verifies the stored manifest without writing. It is idempotent across
later commits because the current-source commit is taken from the stored manifest
unless overridden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))

import dataset_validator as V  # noqa: E402

MANIFEST = _REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1" / "manifest.json"

#: The commit whose working tree produced the frozen v0.1.1 JSONL (Task 5B).
#: This is an immutable historical fact and must never be changed to "make the
#: hashes match".
GENERATION_COMMIT = "a9e5bd69cbc331f79763dd401d5c5f68d2b5f780"

#: Generation-relevant files: anything here can change what a generated sample
#: says or which component it selects.
SOURCE_FILES = (
    "spatial_reasoning/annotator.py",
    "spatial_reasoning/semantic_policy.py",
    "spatial_reasoning/templates.py",
    "spatial_reasoning/relations.py",
    "spatial_reasoning/geometry.py",
    "spatial_reasoning/thresholds.py",
    "spatial_reasoning/component_quality.py",
    "scripts/build_spatial_reason.py",
)

HASH_DEFINITION = (
    "SHA256 of the file content after line-ending normalisation (CRLF -> LF), read from the "
    "Git object store at the commit named above. Normalisation makes the hash independent of "
    "core.autocrlf and comparable with a working-tree file; it is a no-op for blobs, which are "
    "already LF."
)


def sha256_normalised(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def git(repo_root: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, timeout=120,
        text=not binary,
    )
    if result.returncode != 0:
        stderr = result.stderr if binary else (result.stderr or "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return result.stdout


def blob_hash(repo_root: Path, commit: str, path: str) -> str | None:
    """Normalised SHA256 of a blob, or None when absent at that commit."""

    try:
        data = git(repo_root, "show", f"{commit}:{path}", binary=True)
    except RuntimeError:
        return None
    if not isinstance(data, bytes):
        data = data.encode("utf-8")
    return sha256_normalised(data)


def resolve_commit(repo_root: Path, revision: str) -> str:
    return git(repo_root, "rev-parse", revision).strip()


def commit_subject(repo_root: Path, commit: str) -> str | None:
    try:
        return git(repo_root, "log", "-1", "--format=%s", commit).strip() or None
    except RuntimeError:
        return None


def source_block(repo_root: Path, commit: str, definition: str) -> dict:
    files: dict[str, str] = {}
    missing: list[str] = []
    for name in SOURCE_FILES:
        digest = blob_hash(repo_root, commit, name)
        if digest is None:
            missing.append(name)
            continue
        files[name] = digest
    return {
        "commit": commit,
        "commit_subject": commit_subject(repo_root, commit),
        "file_sha256": files,
        "hash_definition": HASH_DEFINITION,
        "missing_at_commit": missing,
        "definition": definition,
    }


def working_tree_divergence(repo_root: Path, current: dict) -> dict:
    """Informational: which files differ on disk from the snapshot block.

    Compared after normalisation, so git's CRLF materialisation is not reported
    as a difference.
    """

    divergent: list[str] = []
    absent: list[str] = []
    for name, recorded in sorted(current["file_sha256"].items()):
        path = repo_root / name
        if not path.is_file():
            absent.append(name)
            continue
        if sha256_normalised(path.read_bytes()) != recorded:
            divergent.append(name)
    return {"divergent_on_disk": divergent, "absent_on_disk": absent}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--generation-commit", default=GENERATION_COMMIT)
    parser.add_argument(
        "--current-commit",
        default=None,
        help="snapshot commit for current_source; defaults to the stored value, else HEAD",
    )
    args = parser.parse_args(argv)

    # ---- safety: the frozen JSONL must not have moved --------------------
    integrity = V.verify_v011_unchanged(_REPO_ROOT / "datasets")
    if integrity.get("unchanged") is not True:
        print(
            "error: v0.1.1 JSONL hashes changed; refusing to rewrite provenance. "
            f"mismatches={integrity.get('mismatches')}",
            file=sys.stderr,
        )
        return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    try:
        generation_commit = resolve_commit(_REPO_ROOT, args.generation_commit)
        stored_current = (manifest.get("current_source") or {}).get("commit")
        current_commit = resolve_commit(
            _REPO_ROOT,
            args.current_commit
            or (stored_current if stored_current and stored_current != "UNCOMMITTED" else "HEAD"),
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    generation = source_block(
        _REPO_ROOT,
        generation_commit,
        "Byte content of the generation-relevant source files AS COMMITTED at the commit "
        "whose working tree produced the frozen v0.1.1 JSONL. Reconstructed from the Git "
        "object store. This block is historical and must never be refreshed.",
    )
    current = source_block(
        _REPO_ROOT,
        current_commit,
        "Byte content of the same files AS COMMITTED at the snapshot commit named in "
        "'current_source.commit' (the repository state when this research task began). It "
        "may move as maintenance edits land; it describes the code a future regeneration "
        "would use, NOT the code that produced the frozen JSONL.",
    )
    divergence = working_tree_divergence(_REPO_ROOT, current)

    if generation["missing_at_commit"]:
        print(
            f"error: files absent at generation commit {generation_commit}: "
            f"{generation['missing_at_commit']}",
            file=sys.stderr,
        )
        return 2

    if args.check:
        ok = True
        if manifest.get("generation_source") != generation:
            ok = False
            print("MISMATCH: generation_source differs from Git history", file=sys.stderr)
        if manifest.get("current_source") != current:
            ok = False
            print("MISMATCH: current_source differs from the recorded commit", file=sys.stderr)
        if "generator_file_sha256" in manifest:
            ok = False
            print("MISMATCH: legacy generator_file_sha256 still present", file=sys.stderr)
        if not args.quiet:
            print(f"provenance check  : {'ok' if ok else 'FAILED'}")
            print(f"generation commit : {generation['commit']} ({generation['commit_subject']})")
            print(f"current commit    : {current['commit']} ({current['commit_subject']})")
            print(f"divergent on disk : {divergence['divergent_on_disk'] or 'none'}")
        return 0 if ok else 1

    legacy = manifest.pop("generator_file_sha256", None)
    manifest["generation_source"] = generation
    manifest["current_source"] = current
    manifest["provenance_note"] = (
        "generation_source describes the code that produced the frozen JSONL and must never "
        "be refreshed; current_source tracks the repository snapshot named in "
        "current_source.commit and may move as maintenance edits land. Both blocks are "
        "derived from Git objects and enforced by scripts/check_artifact_consistency.py. "
        "The single-map field generator_file_sha256 was split in Task 5.5 section 3.1."
    )
    if legacy is not None:
        manifest["provenance_split_note"] = {
            "superseded_field": "generator_file_sha256",
            "superseded_value": legacy,
            "reason": (
                "It mixed two different facts. (1) scripts/build_spatial_reason.py had been "
                "refreshed to its post-generation hash, so the map described no single real "
                "code state. (2) spatial_reasoning/relations.py was hashed in its CRLF "
                "working-tree form (ec0fd6139cf312bcb77bd8dded89c44da4fc6a84c331f0a81a49a058f036c7dc) "
                "rather than its committed LF blob "
                "(d45688a696e8ea102ccd763a14dd20a466a80cab2ed617bb2fcf0d52767ec727); the two "
                "files are semantically identical, differing only in line endings."
            ),
        }

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        diverged = [
            name for name in SOURCE_FILES
            if generation["file_sha256"][name] != current["file_sha256"][name]
        ]
        print(f"generation commit : {generation['commit']} ({generation['commit_subject']})")
        print(f"current  commit   : {current['commit']} ({current['commit_subject']})")
        print(f"files recorded    : {len(generation['file_sha256'])}")
        print(f"diverged generation -> current: {diverged or 'none'}")
        print(f"divergent on disk : {divergence['divergent_on_disk'] or 'none'}")
        print(f"wrote             : {MANIFEST.relative_to(_REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
