#!/usr/bin/env python
"""CLI: repository artifact consistency gate for BuildSpatialReason.

    python scripts/check_artifact_consistency.py [--quiet]

Task 5C section 7. The Task 5B commit shipped a valid-looking dataset whose
*repository documents* still carried stale numbers, and two competing quality
JSONs claimed to be authoritative. This gate makes that class of drift
impossible: every machine-readable artifact, every canonical path and every
Markdown file that opts in with an ``ARTIFACT-FACTS`` block must agree with
``evaluation/build_spatial_reason_artifact_index.json``.

Checks
------
1.  version identity              dataset version, semantic policy version,
                                  generator version, relation config version
2.  counts                        total / split / level / Level-2 A+B /
                                  Level-3 trivial+nontrivial, across the index,
                                  manifest.json, statistics.json and the
                                  evaluation quality JSON
3.  canonical paths               the quality JSON, the sample pack and the
                                  quality doc exist at their canonical names
4.  deprecated paths              the superseded ``v011`` artifacts are gone
5.  provenance                   ``manifest.generation_source`` and
                                  ``manifest.current_source`` both exist, cover
                                  every generation-critical file, and match Git
                                  history / the recorded snapshot commit; the
                                  legacy combined field is gone
6.  frozen data                  v0.1 and v0.1.1 JSONL records hash to their
                                  recorded values
7.  documents                    every ``ARTIFACT-FACTS`` block declares the
                                  authoritative values and canonical paths

Writes ``evaluation/build_spatial_reason_v0.1.1_consistency.json``.
Exit code 0 when consistent, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import dataset_validator as V  # noqa: E402

INDEX_PATH = _REPO_ROOT / "evaluation" / "build_spatial_reason_artifact_index.json"

FACTS_BEGIN = "<!-- ARTIFACT-FACTS:BEGIN -->"
FACTS_END = "<!-- ARTIFACT-FACTS:END -->"

#: Fact keys every ARTIFACT-FACTS block must declare.
REQUIRED_FACT_KEYS = (
    "dataset_version",
    "total_samples",
    "split_train",
    "split_val",
    "split_test",
    "level_1",
    "level_2",
    "level_3",
    "level2_type_a",
    "level2_type_b",
    "level3_trivial",
    "level3_nontrivial",
    "semantic_policy_version",
    "generator_version",
    "quality_json_path",
    "sample_pack_path",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Fact blocks
# --------------------------------------------------------------------------


def expected_facts(index: dict) -> dict:
    """The authoritative fact values declared by the index."""

    dataset = index["dataset"]
    counts = index["counts"]
    paths = index["canonical_paths"]
    return {
        "dataset_version": dataset["version"],
        "total_samples": counts["total_samples"],
        "split_train": counts["by_split"]["train"],
        "split_val": counts["by_split"]["val"],
        "split_test": counts["by_split"]["test"],
        "level_1": counts["by_level"]["1"],
        "level_2": counts["by_level"]["2"],
        "level_3": counts["by_level"]["3"],
        "level2_type_a": counts["level2"]["reference_to_nearest"],
        "level2_type_b": counts["level2"]["reference_to_direction"],
        "level3_trivial": counts["level3"]["trivial"],
        "level3_nontrivial": counts["level3"]["nontrivial"],
        "semantic_policy_version": str(dataset["semantic_policy_version"]),
        "generator_version": dataset["generator_version"],
        "quality_json_path": paths["quality_json"],
        "sample_pack_path": paths["sample_pack"],
    }


def parse_facts(text: str) -> tuple[dict | None, list[str]]:
    """Parse one ARTIFACT-FACTS block. Returns (facts, problems)."""

    if FACTS_BEGIN not in text:
        return None, []
    problems: list[str] = []
    body = text.split(FACTS_BEGIN, 1)[1]
    if FACTS_END not in body:
        return None, [f"missing {FACTS_END}"]
    body = body.split(FACTS_END, 1)[0]

    facts: dict = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            problems.append(f"unparsable fact line: {line!r}")
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value.lstrip("-").isdigit():
            facts[key] = int(value)
        else:
            facts[key] = value
    return facts, problems


def find_fact_documents(repo_root: Path) -> list[Path]:
    """Every Markdown file that carries an ARTIFACT-FACTS block."""

    skip = {".git", "node_modules", "__pycache__", "datasets"}
    found: list[Path] = []
    for path in sorted(repo_root.rglob("*.md")):
        if any(part in skip for part in path.relative_to(repo_root).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if FACTS_BEGIN in text:
            found.append(path)
    return found


# --------------------------------------------------------------------------
# Consistency checks
# --------------------------------------------------------------------------


def run_consistency_check(
    repo_root: Path | None = None,
    index_path: Path | None = None,
    check_frozen_data: bool = True,
    quality_report: dict | None = None,
) -> dict:
    """Run every gate and return a machine-readable report.

    ``quality_report`` lets the caller pass the just-produced quality report in
    memory instead of reading it from disk. The acceptance validator uses this so
    the semantic verdict and the repository-wide agreement check are computed in
    one pass, with no half-written artifact on disk.
    """

    repo_root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    index_path = Path(index_path) if index_path is not None else repo_root / "evaluation" / "build_spatial_reason_artifact_index.json"

    violations: list[dict] = []
    checks: dict[str, str] = {}

    def fail(check: str, detail) -> None:
        violations.append({"check": check, "detail": detail})

    if not index_path.is_file():
        return {
            "status": "inconsistent",
            "checks": {"artifact_index": "missing"},
            "violations": [{"check": "artifact_index_missing", "detail": str(index_path)}],
            "documents_checked": [],
        }

    index = load_json(index_path)
    facts = expected_facts(index)
    paths = index["canonical_paths"]

    def repo_path(relative: str) -> Path:
        return repo_root / relative

    # ---- 1. version identity ------------------------------------------
    manifest_path = repo_path(paths["manifest"])
    statistics_path = repo_path(paths["statistics"])
    quality_path = repo_path(paths["quality_json"])

    missing = [
        name
        for name, path in (
            ("manifest", manifest_path),
            ("statistics", statistics_path),
            ("quality_json", quality_path),
            ("quality_doc", repo_path(paths["quality_doc"])),
            ("dataset_doc", repo_path(paths["dataset_doc"])),
        )
        if not path.is_file()
    ]
    if missing:
        fail("canonical_paths", f"missing canonical artifacts: {missing}")
        checks["canonical_paths"] = "fail"
        return {
            "status": "inconsistent",
            "checks": checks,
            "violations": violations,
            "documents_checked": [],
        }
    checks["canonical_paths"] = "ok"

    manifest = load_json(manifest_path)
    statistics = load_json(statistics_path)
    quality = quality_report if quality_report is not None else load_json(quality_path)

    version_fields = {
        "index.dataset.version": facts["dataset_version"],
        "manifest.dataset_version": manifest.get("dataset_version"),
        "quality.audited_version": quality.get("audited_version"),
    }
    if len(set(version_fields.values())) != 1:
        fail("dataset_version", version_fields)
    checks["dataset_version"] = "ok"

    if str(manifest.get("semantic_visibility_policy_version")) != facts["semantic_policy_version"]:
        fail(
            "semantic_policy_version",
            {
                "index": facts["semantic_policy_version"],
                "manifest": manifest.get("semantic_visibility_policy_version"),
            },
        )
    if manifest.get("generator_version") != facts["generator_version"]:
        fail(
            "generator_version",
            {"index": facts["generator_version"], "manifest": manifest.get("generator_version")},
        )
    if manifest.get("relation_config_version") != index["dataset"]["relation_config_version"]:
        fail(
            "relation_config_version",
            {
                "index": index["dataset"]["relation_config_version"],
                "manifest": manifest.get("relation_config_version"),
            },
        )

    # ---- 2. counts -----------------------------------------------------
    index_counts = index["counts"]
    m_counts = manifest.get("sample_counts", {})
    s_counts = statistics
    q_counts = quality.get("independent_counts", {})

    comparisons = {
        "total_samples": (
            index_counts["total_samples"],
            m_counts.get("total"),
            s_counts.get("total_samples"),
            q_counts.get("total_records"),
        ),
        "split_train": (
            index_counts["by_split"]["train"],
            (m_counts.get("by_split") or {}).get("train"),
            (s_counts.get("by_split") or {}).get("train"),
            (q_counts.get("by_split") or {}).get("train"),
        ),
        "split_val": (
            index_counts["by_split"]["val"],
            (m_counts.get("by_split") or {}).get("val"),
            (s_counts.get("by_split") or {}).get("val"),
            (q_counts.get("by_split") or {}).get("val"),
        ),
        "split_test": (
            index_counts["by_split"]["test"],
            (m_counts.get("by_split") or {}).get("test"),
            (s_counts.get("by_split") or {}).get("test"),
            (q_counts.get("by_split") or {}).get("test"),
        ),
        "level_1": (
            index_counts["by_level"]["1"],
            (m_counts.get("by_level") or {}).get("1"),
            (s_counts.get("by_level") or {}).get("1"),
            (q_counts.get("by_level") or {}).get("1"),
        ),
        "level_2": (
            index_counts["by_level"]["2"],
            (m_counts.get("by_level") or {}).get("2"),
            (s_counts.get("by_level") or {}).get("2"),
            (q_counts.get("by_level") or {}).get("2"),
        ),
        "level_3": (
            index_counts["by_level"]["3"],
            (m_counts.get("by_level") or {}).get("3"),
            (s_counts.get("by_level") or {}).get("3"),
            (q_counts.get("by_level") or {}).get("3"),
        ),
        "level2_type_a": (
            index_counts["level2"]["reference_to_nearest"],
            (m_counts.get("level2") or {}).get("reference_to_nearest"),
            (s_counts.get("level2") or {}).get("reference_to_nearest"),
            (q_counts.get("level2") or {}).get("reference_to_nearest"),
        ),
        "level2_type_b": (
            index_counts["level2"]["reference_to_direction"],
            (m_counts.get("level2") or {}).get("reference_to_direction"),
            (s_counts.get("level2") or {}).get("reference_to_direction"),
            (q_counts.get("level2") or {}).get("reference_to_direction"),
        ),
        "level3_trivial": (
            index_counts["level3"]["trivial"],
            (m_counts.get("level3") or {}).get("trivial"),
            (s_counts.get("level3") or {}).get("trivial"),
            (q_counts.get("level3") or {}).get("trivial"),
        ),
        "level3_nontrivial": (
            index_counts["level3"]["nontrivial"],
            (m_counts.get("level3") or {}).get("nontrivial"),
            (s_counts.get("level3") or {}).get("nontrivial"),
            (q_counts.get("level3") or {}).get("nontrivial"),
        ),
    }
    inconsistent_counts: dict[str, list] = {}
    for name, values in comparisons.items():
        if len(set(values)) != 1:
            inconsistent_counts[name] = list(values)
    if inconsistent_counts:
        fail("counts", inconsistent_counts)
    checks["counts"] = "ok" if not inconsistent_counts else "fail"

    # Level-2 partition invariant, recomputed rather than trusted.
    for label, counts in (
        ("manifest", m_counts),
        ("statistics", s_counts),
    ):
        level2 = counts.get("level2") or {}
        a = level2.get("reference_to_nearest")
        b = level2.get("reference_to_direction")
        total = level2.get("total_level2")
        if None not in (a, b, total) and a + b != total:
            fail("level2_partition", {"artifact": label, "a": a, "b": b, "total": total})
    q_level2 = q_counts.get("level2") or {}
    if None not in (
        q_level2.get("reference_to_nearest"),
        q_level2.get("reference_to_direction"),
        q_level2.get("total_level2"),
    ) and (
        q_level2["reference_to_nearest"] + q_level2["reference_to_direction"]
        != q_level2["total_level2"]
    ):
        fail("level2_partition", {"artifact": "quality_json", **q_level2})

    # ---- 3. deprecated paths -------------------------------------------
    surviving = [p for p in index.get("deprecated_paths", []) if repo_path(p).exists()]
    if surviving:
        fail("deprecated_paths_present", surviving)
    checks["deprecated_paths"] = "ok" if not surviving else "fail"

    # ---- 4. provenance (generation-time vs current-source) --------------
    #
    # Task 5.5 section 3.1. A single `generator_file_sha256` map mixed the code
    # that produced the frozen JSONL with later maintenance edits. The two are
    # now separate blocks and BOTH are verified against Git history, so the
    # generation-time record cannot be silently refreshed.
    required = index.get("provenance_required_files", [])
    generation = manifest.get("generation_source") or {}
    current_source = manifest.get("current_source") or {}
    provenance_report: dict = {
        "legacy_field_present": "generator_file_sha256" in manifest,
        "generation_commit": generation.get("commit"),
        "current_commit": current_source.get("commit"),
        "verified_against_git": None,
        "generation_files": len(generation.get("file_sha256") or {}),
        "current_files": len(current_source.get("file_sha256") or {}),
    }

    if provenance_report["legacy_field_present"]:
        fail(
            "provenance_legacy_field",
            "manifest still carries generator_file_sha256; run "
            "scripts/split_manifest_provenance.py",
        )

    for label, block in (("generation_source", generation), ("current_source", current_source)):
        if not block.get("commit") or not block.get("file_sha256"):
            fail("provenance_missing_block", f"{label} is absent or empty")

    absent = [name for name in required if name not in (generation.get("file_sha256") or {})]
    absent += [
        name for name in required if name not in (current_source.get("file_sha256") or {})
    ]
    if absent:
        fail("provenance_missing_files", sorted(set(absent)))

    # Git-derived verification. Any of the three failure modes below is a hard
    # failure; missing Git metadata is reported as unverified, not as a pass.
    git_problems: list[dict] = []
    git_available = True
    try:
        import split_manifest_provenance as S
    except Exception as exc:  # noqa: BLE001
        git_available = False
        provenance_report["git_error"] = f"import failed: {type(exc).__name__}: {exc}"

    if git_available:
        for label, block in (("generation_source", generation), ("current_source", current_source)):
            commit = block.get("commit")
            if not commit or commit == "UNCOMMITTED":
                git_problems.append({"block": label, "error": "no usable commit recorded"})
                continue
            try:
                resolved = S.resolve_commit(repo_root, commit)
            except Exception as exc:  # noqa: BLE001
                git_problems.append({"block": label, "commit": commit, "error": str(exc)})
                continue
            if resolved != commit:
                git_problems.append(
                    {"block": label, "error": "commit does not resolve to itself", "resolved": resolved}
                )
            for name, recorded in sorted(block["file_sha256"].items()):
                actual = S.blob_hash(repo_root, commit, name)
                if actual is None:
                    git_problems.append({"block": label, "file": name, "error": "absent at commit"})
                elif actual != recorded:
                    git_problems.append(
                        {
                            "block": label,
                            "file": name,
                            "recorded_sha256": recorded,
                            "git_sha256": actual,
                        }
                    )
        provenance_report["verified_against_git"] = not git_problems
        if git_problems:
            fail("provenance_git_mismatch", git_problems)

        # Informational: does the working tree still match the snapshot block?
        try:
            provenance_report["working_tree"] = S.working_tree_divergence(repo_root, current_source)
        except Exception:  # noqa: BLE001
            pass
    else:
        provenance_report["verified_against_git"] = None
        checks["provenance"] = "unverified"
        fail("provenance_git_unavailable", provenance_report.get("git_error"))

    provenance_report["generation_vs_current_diverged"] = sorted(
        name
        for name in set(generation.get("file_sha256") or {}) | set(current_source.get("file_sha256") or {})
        if (generation.get("file_sha256") or {}).get(name)
        != (current_source.get("file_sha256") or {}).get(name)
    )

    if "provenance" not in checks:
        checks["provenance"] = "ok" if not (absent or git_problems or provenance_report["legacy_field_present"]) else "fail"

    # ---- 5. frozen data -------------------------------------------------
    frozen: dict = {}
    if check_frozen_data:
        v01 = V.verify_v01_unchanged(repo_root / "datasets")
        v011 = V.verify_v011_unchanged(repo_root / "datasets")
        frozen = {"v0.1": v01, "v0.1.1": v011}
        if v01.get("unchanged") is not True:
            fail("v01_frozen_modified", v01.get("mismatches"))
        if v011.get("unchanged") is not True:
            fail("v011_frozen_modified", v011.get("mismatches"))
        recorded_quality_integrity = quality.get("v011_frozen_integrity")
        if isinstance(recorded_quality_integrity, dict):
            if recorded_quality_integrity.get("unchanged") is not True:
                fail("quality_json_reports_v011_modified", recorded_quality_integrity.get("mismatches"))
        else:
            fail("quality_json_missing_v011_integrity", "field v011_frozen_integrity absent")
        checks["frozen_data"] = "ok" if not any(
            v.get("unchanged") is not True for v in frozen.values()
        ) else "fail"

    # ---- 6. documents ---------------------------------------------------
    documents_checked: list[str] = []
    document_problems: list[dict] = []
    discovered = find_fact_documents(repo_root)
    discovered_relatives = {p.relative_to(repo_root).as_posix() for p in discovered}

    for required_doc in index.get("facts_documents", []):
        if required_doc not in discovered_relatives:
            document_problems.append(
                {"document": required_doc, "problems": ["required ARTIFACT-FACTS block absent"]}
            )

    for path in discovered:
        relative = path.relative_to(repo_root).as_posix()
        documents_checked.append(relative)
        block, problems = parse_facts(path.read_text(encoding="utf-8"))
        if block is None:
            document_problems.append({"document": relative, "problems": problems or ["no block"]})
            continue
        for problem in problems:
            document_problems.append({"document": relative, "problems": [problem]})
        for key in REQUIRED_FACT_KEYS:
            if key not in block:
                document_problems.append({"document": relative, "problems": [f"missing key {key}"]})
                continue
            if block[key] != facts[key]:
                document_problems.append(
                    {
                        "document": relative,
                        "problems": [f"{key}: declared {block[key]!r} != authoritative {facts[key]!r}"],
                    }
                )
    if document_problems:
        fail("document_facts", document_problems)
    checks["documents"] = "ok" if not document_problems else "fail"
    checks["documents_checked"] = len(documents_checked)

    return {
        "audited_version": facts["dataset_version"],
        "index_path": index_path.relative_to(repo_root).as_posix()
        if index_path.is_relative_to(repo_root)
        else str(index_path),
        "status": "consistent" if not violations else "inconsistent",
        "checks": checks,
        "violations": violations,
        "violation_count": len(violations),
        "documents_checked": documents_checked,
        "authoritative_facts": facts,
        "quality_verdict": quality.get("verdict"),
        "frozen_data": frozen,
        "provenance": provenance_report,
        "provenance_files_verified": provenance_report["generation_files"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-frozen-check", action="store_true",
                        help="skip JSONL re-hashing (faster; not for acceptance)")
    args = parser.parse_args(argv)

    report = run_consistency_check(check_frozen_data=not args.no_frozen_check)

    index = load_json(INDEX_PATH)
    out_path = _REPO_ROOT / index["canonical_paths"]["consistency_json"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"status            : {report['status']}")
        print(f"violations        : {report['violation_count']}")
        for violation in report["violations"][:10]:
            print(f"  - {violation['check']}: {violation['detail']}")
        print(f"documents checked : {len(report['documents_checked'])}")
        print(f"wrote             : {out_path.relative_to(_REPO_ROOT).as_posix()}")
    return 0 if report["status"] == "consistent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
