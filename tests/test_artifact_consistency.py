"""Artifact consistency gate tests (Task 5C sections 7, 8, 9, 10.7-10.9).

Covers:

 7. the artifact consistency checker accepts the real repository and rejects
    injected drift
 8. canonical evaluation artifact path naming
 9. manifest provenance includes ``semantic_policy.py`` and every recorded
    digest matches the file on disk
 10. required Markdown files carry a matching ARTIFACT-FACTS block

Run with pytest, or directly::

    python tests/test_artifact_consistency.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import check_artifact_consistency as C  # noqa: E402

INDEX_PATH = _REPO_ROOT / "evaluation" / "build_spatial_reason_artifact_index.json"


def _load(relative: str) -> dict:
    return json.loads((_REPO_ROOT / relative).read_text(encoding="utf-8"))


def _index() -> dict:
    return _load("evaluation/build_spatial_reason_artifact_index.json")


# --------------------------------------------------------------------------
# 7. The checker itself
# --------------------------------------------------------------------------


def test_consistency_checker_accepts_repository():
    """The real repository must be internally consistent."""

    report = C.run_consistency_check()
    assert report["status"] == "consistent", (
        f"consistency violations: {json.dumps(report['violations'], indent=2)[:4000]}"
    )
    assert report["violation_count"] == 0
    assert report["checks"].get("provenance") == "ok"
    assert report["checks"].get("counts") == "ok"
    assert report["checks"].get("documents") == "ok"
    assert report["checks"].get("frozen_data") == "ok"
    assert report["provenance_files_verified"] >= 8
    print(
        f"  [7a] repository consistent OK "
        f"({report['provenance_files_verified']} provenance files, "
        f"{len(report['documents_checked'])} fact documents)"
    )


def test_consistency_checker_detects_injected_count_drift():
    """A single wrong count in the quality JSON must fail the gate."""

    report_in_memory = _load("evaluation/build_spatial_reason_v0.1.1_quality.json")
    report_in_memory["independent_counts"]["total_records"] += 1

    result = C.run_consistency_check(quality_report=report_in_memory)
    assert result["status"] == "inconsistent", "injected count drift was not detected"
    codes = {violation["check"] for violation in result["violations"]}
    assert "counts" in codes, codes
    print("  [7b] injected count drift detected OK")


def test_consistency_checker_detects_deprecated_artifact():
    """A resurrected v011 quality JSON must fail the gate."""

    stale = _REPO_ROOT / "evaluation" / "build_spatial_reason_v011_quality.json"
    assert not stale.exists(), "deprecated artifact should have been removed"
    try:
        stale.write_text("{}\n", encoding="utf-8")
        result = C.run_consistency_check(check_frozen_data=False)
        codes = {violation["check"] for violation in result["violations"]}
        assert "deprecated_paths_present" in codes, codes
        assert result["status"] == "inconsistent"
    finally:
        stale.unlink(missing_ok=True)
    print("  [7c] deprecated artifact detected OK")


def test_facts_parser_rejects_declared_mismatch():
    """The ARTIFACT-FACTS parser must read numbers as numbers and paths as paths."""

    facts = C.expected_facts(_index())
    block, problems = C.parse_facts(
        f"{C.FACTS_BEGIN}\ntotal_samples: 99999\nquality_json_path: "
        f"{facts['quality_json_path']}\n{C.FACTS_END}\n"
    )
    assert not problems, problems
    assert block["total_samples"] == 99999
    assert block["total_samples"] != facts["total_samples"]
    assert block["quality_json_path"] == facts["quality_json_path"]
    print("  [7d] fact-block parser OK")


# --------------------------------------------------------------------------
# 8. Canonical paths
# --------------------------------------------------------------------------


def test_canonical_evaluation_paths():
    """Evaluation artifacts must use the real dataset version in their names."""

    index = _index()
    paths = index["canonical_paths"]

    assert paths["quality_json"] == "evaluation/build_spatial_reason_v0.1.1_quality.json"
    assert paths["sample_pack"] == "evaluation/build_spatial_reason_v0.1.1_samples"
    assert paths["quality_doc"] == "docs/build_spatial_reason_v0.1.1_quality_audit.md"

    for key in ("quality_json", "quality_doc", "dataset_doc"):
        assert (_REPO_ROOT / paths[key]).is_file(), f"{key} missing: {paths[key]}"
    assert (_REPO_ROOT / paths["sample_pack"]).is_dir(), paths["sample_pack"]

    # The historical "v011" abbreviation must not survive anywhere on disk.
    assert not (_REPO_ROOT / "evaluation" / "build_spatial_reason_v011_quality.json").exists()
    assert not (_REPO_ROOT / "evaluation" / "build_spatial_reason_v011_samples").exists()

    # The validator must derive exactly these names from the version string.
    for version, expected in (
        ("v0.1.1", "build_spatial_reason_v0.1.1_quality.json"),
        ("v0.1", "build_spatial_reason_v0.1_quality.json"),
    ):
        assert f"build_spatial_reason_{version}_quality.json" == expected

    print("  [8] canonical evaluation paths OK")


# --------------------------------------------------------------------------
# 9. Provenance
# --------------------------------------------------------------------------


def test_manifest_provenance_includes_semantic_policy():
    """Provenance must cover semantic_policy.py and match the files on disk."""

    manifest = _load("datasets/build_spatial_reason/v0.1.1/manifest.json")
    provenance = manifest["generator_file_sha256"]

    assert "spatial_reasoning/semantic_policy.py" in provenance, sorted(provenance)

    required = _index()["provenance_required_files"]
    missing = [name for name in required if name not in provenance]
    assert not missing, f"provenance missing generation-critical files: {missing}"

    for name, recorded in sorted(provenance.items()):
        path = _REPO_ROOT / name
        assert path.is_file(), f"provenance lists a missing file: {name}"
        assert C.sha256_file(path) == recorded, f"provenance digest mismatch for {name}"

    # The generator must build the same list at generation time.
    source = (_REPO_ROOT / "scripts" / "build_spatial_reason.py").read_text(encoding="utf-8")
    assert '"semantic_policy.py"' in source
    print(f"  [9] manifest provenance covers {len(provenance)} files and matches disk OK")


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


def test_required_documents_declare_authoritative_facts():
    """Every required Markdown document must carry a matching facts block."""

    facts = C.expected_facts(_index())
    required_docs = _index()["facts_documents"]
    discovered = {
        path.relative_to(_REPO_ROOT).as_posix(): path for path in C.find_fact_documents(_REPO_ROOT)
    }

    for relative in required_docs:
        assert relative in discovered, f"missing ARTIFACT-FACTS block in {relative}"
        block, problems = C.parse_facts(discovered[relative].read_text(encoding="utf-8"))
        assert not problems, f"{relative}: {problems}"
        for key in C.REQUIRED_FACT_KEYS:
            assert key in block, f"{relative}: missing {key}"
            assert block[key] == facts[key], (
                f"{relative}: {key} declared {block[key]!r} != {facts[key]!r}"
            )

    print(f"  [10] {len(required_docs)} documents declare authoritative facts OK")


def test_consistency_report_is_written_and_matches_live_check():
    """The published consistency JSON must agree with a fresh check."""

    report_path = _REPO_ROOT / "evaluation" / "build_spatial_reason_v0.1.1_consistency.json"
    if not report_path.is_file():
        print("  [extra] consistency JSON absent; run scripts/check_artifact_consistency.py")
        return
    published = json.loads(report_path.read_text(encoding="utf-8"))
    live = C.run_consistency_check()

    assert published["status"] == live["status"] == "consistent"
    assert published["violation_count"] == live["violation_count"] == 0
    assert published["authoritative_facts"] == live["authoritative_facts"]
    assert published["documents_checked"] == live["documents_checked"]
    print("  [extra] published consistency JSON matches a live check OK")


def main() -> int:
    tests = [
        ("7a checker accepts repo", test_consistency_checker_accepts_repository),
        ("7b detects count drift", test_consistency_checker_detects_injected_count_drift),
        ("7c detects deprecated artifact", test_consistency_checker_detects_deprecated_artifact),
        ("7d facts parser", test_facts_parser_rejects_declared_mismatch),
        ("8 canonical paths", test_canonical_evaluation_paths),
        ("9 provenance", test_manifest_provenance_includes_semantic_policy),
        ("documents", test_required_documents_declare_authoritative_facts),
        ("extra published report", test_consistency_report_is_written_and_matches_live_check),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failures}/{len(tests)} artifact-consistency checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
