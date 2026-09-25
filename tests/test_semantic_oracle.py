"""Independent-oracle tests (Task 5C section 10).

Covers:

 1. the oracle never imports or calls ``semantic_policy`` (static + runtime)
 2. ``largest`` matches direct area argmax semantics
 3. ``smallest`` matches direct area argmin semantics
 4. ``nearest`` uses all visible candidates BEFORE eligibility filtering
 5. the Level-3 chain uses the full direction-valid candidate set
 6. full-dataset oracle target match (live subset + recorded full-dataset run)
 10. v0.1 and v0.1.1 JSONL hashes remain unchanged

Run with pytest, or directly::

    python tests/test_semantic_oracle.py

The live oracle sweep is bounded by default so the suite stays usable (up to
4,000 records per split). Set ``SPATIAL_ORACLE_FULL=1`` to sweep all 25,229 records
(the acceptance audit in ``scripts/validate_build_spatial_reason.py`` always sweeps
all of them, and its result is asserted here from the quality JSON).
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import dataset_validator as V  # noqa: E402
import geometry as G  # noqa: E402
import semantic_oracle as SO  # noqa: E402
import thresholds as T  # noqa: E402

DATASET_ROOT = _REPO_ROOT / "datasets" / "whu"
GEN_DIR = _REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1"
V01_DIR = _REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1"
QUALITY_JSON = _REPO_ROOT / "evaluation" / "build_spatial_reason_v0.1.1_quality.json"
SPLITS = ("train", "val", "test")
CONFIG = T.load_config()

#: Live sweep size PER SPLIT (4,000 x 3 splits = 11,884 records, spanning every
#: level and query type) so the suite stays quick. The recorded acceptance audit
#: always sweeps all 25,229.
LIVE_LIMIT = None if os.environ.get("SPATIAL_ORACLE_FULL") == "1" else 4000


def _have_v011() -> bool:
    return (GEN_DIR / "train.jsonl").is_file()


def _make_oracle(boxes: dict[int, tuple[int, int, int, int]], width=500, height=500):
    """Build an OracleImage directly from axis-aligned boxes."""

    component_map = np.zeros((height, width), dtype=np.uint8)
    for component_id, (x0, y0, x1, y1) in boxes.items():
        component_map[y0:y1, x0:x1] = component_id

    components = []
    for component_id in sorted(boxes):
        mask = component_map == component_id
        assert mask.any(), f"component {component_id} is empty"
        ys, xs = np.nonzero(mask)
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        components.append(
            G.Component(
                component_id=component_id,
                source_polygon_index=component_id - 1,
                area_px=int(mask.sum()),
                area_ratio=float(mask.sum()) / (width * height),
                centroid_px=(float(xs.mean()), float(ys.mean())),
                bbox_xyxy_px=(bx0, by0, bx1, by1),
                width_px=bx1 - bx0 + 1,
                height_px=by1 - by0 + 1,
                touches_image_border=bool(
                    bx0 == 0 or by0 == 0 or bx1 == width - 1 or by1 == height - 1
                ),
            )
        )
    image = G.ImageGeometry(
        image_id="synthetic",
        split="train",
        width=width,
        height=height,
        component_map_path="unused",
        components=components,
    )
    return SO.OracleImage(image, CONFIG, component_map)


# --------------------------------------------------------------------------
# 1. Independence
# --------------------------------------------------------------------------


def _oracle_code_identifiers(source: str) -> set[str]:
    """Every identifier the module's *code* uses, ignoring comments/strings."""

    tree = ast.parse(source)
    # Drop docstrings so the explanatory prose in this module cannot be mistaken
    # for a reference to the forbidden module.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            for alias in node.names:
                names.add(alias.name)
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_oracle_static_independence():
    """The oracle source must not reference semantic_policy in executable code."""

    source = (_REPO_ROOT / "spatial_reasoning" / "semantic_oracle.py").read_text(encoding="utf-8")
    identifiers = _oracle_code_identifiers(source)

    forbidden = {"semantic_policy", "SP"}
    overlap = identifiers & forbidden
    assert not overlap, f"oracle code references forbidden module/alias: {sorted(overlap)}"

    for banned in (
        "resolve_size_extreme",
        "resolve_nearest",
        "direction_candidates_over_visible",
        "admissible_nearest_ids",
    ):
        # These answers exist in the oracle too (it re-implements the semantics),
        # but under its OWN names and never borrowed from semantic_policy. The
        # identifier scan above proves semantic_policy is unreachable; this
        # asserts the oracle carries its own definitions rather than delegating.
        if banned == "direction_candidates_over_visible":
            assert "def direction_candidates" in source, "oracle must define its own direction set"
            assert f"def {banned}" not in source, "oracle must not shadow the policy name"
            continue
        assert f"def {banned}" in source, f"oracle should define its own {banned}"

    print("  [1a] oracle source has no semantic_policy reference OK")


def test_oracle_runtime_independence():
    """Exercise every oracle entry point with semantic_policy imports poisoned."""

    saved = sys.modules.pop("semantic_policy", None)

    class _Poison:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "semantic_policy" or fullname.endswith(".semantic_policy"):
                raise ImportError("semantic_policy is forbidden on the oracle audit path")
            return None

    poison = _Poison()
    sys.meta_path.insert(0, poison)
    try:
        # The poison must actually work, otherwise the test proves nothing.
        blocked = False
        try:
            importlib.import_module("semantic_policy")
        except ImportError:
            blocked = True
        assert blocked, "poison hook failed to block semantic_policy"

        oracle_module = importlib.reload(SO)
        oracle = _make_oracle(
            {
                1: (20, 20, 120, 100),
                2: (250, 250, 320, 320),
                3: (200, 60, 260, 120),
            }
        )
        oracle_module.resolve_size_extreme(oracle, "largest")
        oracle_module.resolve_size_extreme(oracle, "smallest")
        oracle_module.resolve_extreme(oracle, "leftmost")
        oracle_module.direction_candidates(oracle, 1, "right_of")
        oracle_module.resolve_nearest(oracle, 1, [2, 3], 2, "largest_to_nearest")
        oracle_module.admissible_nearest_ids(oracle, 1, [2, 3])
        oracle_module.resolve_query(oracle, 3, "largest_to_right_of_to_nearest")
        oracle_module.compare_record(
            oracle,
            {
                "sample_id": "synthetic",
                "level": 1,
                "query_type": "largest",
                "target_component_id": 1,
                "reference_component_ids": [],
                "reasoning_steps": [{"step": 1, "operation": "argmax_area"}],
            },
        )
    finally:
        sys.meta_path.remove(poison)
        if saved is not None:
            sys.modules["semantic_policy"] = saved
        importlib.reload(SO)

    print("  [1b] oracle runs end-to-end with semantic_policy poisoned OK")


# --------------------------------------------------------------------------
# 2 / 3. Size extremes
# --------------------------------------------------------------------------


def test_oracle_largest_is_global_argmax():
    """The semantic largest is the global area argmax, even when ineligible."""

    # Component 1 is the global argmax and touches the border -> the frozen
    # 'largest' rule rejects it. The oracle must still report id 1 and mark the
    # query inadmissible; it must never return component 2 as a substitute.
    oracle = _make_oracle(
        {
            1: (0, 0, 80, 60),      # 4800 px, touches border
            2: (200, 200, 260, 260),  # 3600 px, eligible
            3: (300, 300, 330, 330),  # 900 px
        }
    )
    resolution = SO.resolve_size_extreme(oracle, "largest")
    assert resolution.target_id == 1, f"expected global argmax 1, got {resolution.target_id}"
    assert resolution.admissible is False, "border-truncated largest must not be admissible"
    assert resolution.reason == SO.ORACLE_TARGET_INELIGIBLE, resolution.reason

    # Control: move component 1 off the border so the same argmax is eligible.
    oracle = _make_oracle(
        {
            1: (20, 20, 100, 80),     # 4800 px, no border
            2: (200, 200, 260, 260),  # 3600 px
            3: (300, 300, 330, 330),  # 900 px
        }
    )
    resolution = SO.resolve_size_extreme(oracle, "largest")
    assert resolution.target_id == 1
    assert resolution.admissible is True, resolution.reason

    print("  [2] oracle largest == global area argmax (no substitution) OK")


def test_oracle_smallest_is_global_argmin():
    """The semantic smallest is the global area argmin, even when ineligible."""

    # Component 1 is the global argmin but is 'tiny' (< 150 px), which the frozen
    # 'smallest' rule rejects. The oracle must report 1 and not substitute 3.
    oracle = _make_oracle(
        {
            1: (100, 100, 110, 110),  # 100 px, tiny
            2: (250, 250, 310, 310),  # 3600 px
            3: (350, 350, 390, 390),  # 1600 px
        }
    )
    resolution = SO.resolve_size_extreme(oracle, "smallest")
    assert resolution.target_id == 1, f"expected global argmin 1, got {resolution.target_id}"
    assert resolution.admissible is False
    assert resolution.reason == SO.ORACLE_TARGET_INELIGIBLE, resolution.reason

    # Control: 12x13 = 156 px clears the tiny threshold.
    oracle = _make_oracle(
        {
            1: (100, 100, 112, 113),
            2: (250, 250, 310, 310),
            3: (350, 350, 390, 390),
        }
    )
    resolution = SO.resolve_size_extreme(oracle, "smallest")
    assert resolution.target_id == 1
    assert resolution.admissible is True, resolution.reason

    print("  [3] oracle smallest == global area argmin (no substitution) OK")


# --------------------------------------------------------------------------
# 4. Nearest over all visible components
# --------------------------------------------------------------------------


def test_oracle_nearest_ignores_eligibility_for_ranking():
    """The nearest is the closest VISIBLE component, ineligible or not."""

    # Component 2 is tiny (5x5 = 25 px) and much closer to the anchor than the
    # eligible component 3. The oracle must report 2 as the semantic nearest and
    # refuse the query rather than answering 3.
    oracle = _make_oracle(
        {
            1: (100, 100, 140, 140),  # anchor, 1600 px
            2: (145, 100, 150, 105),  # 25 px, gap ~5 px  (ineligible: tiny)
            3: (200, 100, 240, 140),  # 1600 px, gap ~60 px
        }
    )
    resolution = SO.resolve_nearest(oracle, 1, [2, 3], 2, "largest_to_nearest")
    assert resolution.target_id == 2, f"expected semantic nearest 2, got {resolution.target_id}"
    assert resolution.admissible is False
    assert resolution.reason == SO.ORACLE_TARGET_INELIGIBLE, resolution.reason

    # Control: enlarge component 2 so the true nearest is also eligible.
    oracle = _make_oracle(
        {
            1: (100, 100, 140, 140),
            2: (145, 100, 185, 140),  # 1600 px, still the closest
            3: (200, 100, 240, 140),
        }
    )
    resolution = SO.resolve_nearest(oracle, 1, [2, 3], 2, "largest_to_nearest")
    assert resolution.target_id == 2
    assert resolution.admissible is True, resolution.reason
    assert resolution.ranked[0][1] == 2 and resolution.ranked[1][1] == 3

    print("  [4] oracle nearest ranks all visible components before eligibility OK")


# --------------------------------------------------------------------------
# 5. Level-3 chain uses the full direction-valid set
# --------------------------------------------------------------------------


def test_oracle_level3_uses_full_direction_set():
    """The chain's nearest step ranges over the FULL direction-valid set."""

    # Reference 1 (the global largest). Right of it: component 2 (tiny, closest)
    # and component 3 (eligible, farther). Component 4 is to the left and 5 is
    # near-diagonal, so neither belongs to right_of(., 1).
    oracle = _make_oracle(
        {
            1: (100, 100, 180, 180),  # 6400 px, reference
            2: (200, 120, 205, 125),  # 25 px, closest on the right (ineligible)
            3: (300, 140, 340, 180),  # 1600 px, farther on the right
            4: (10, 300, 60, 350),    # 2500 px, left of the reference
            5: (400, 400, 430, 430),  # 900 px, near-diagonal
        }
    )
    resolution = SO.resolve_query(oracle, 3, "largest_to_right_of_to_nearest")
    assert resolution.candidate_ids == (2, 3), (
        f"direction set must be the full visible set, got {resolution.candidate_ids}"
    )
    assert resolution.target_id == 2, (
        f"nearest within the direction set must be 2, got {resolution.target_id}"
    )
    assert resolution.admissible is False
    assert resolution.reason == SO.ORACLE_TARGET_INELIGIBLE, resolution.reason

    # Control: make the closest right-side component eligible.
    oracle = _make_oracle(
        {
            1: (100, 100, 180, 180),
            2: (200, 120, 230, 150),  # 900 px, closest on the right
            3: (300, 140, 340, 180),
            4: (10, 300, 60, 350),
            5: (400, 400, 430, 430),
        }
    )
    resolution = SO.resolve_query(oracle, 3, "largest_to_right_of_to_nearest")
    assert resolution.candidate_ids == (2, 3)
    assert resolution.target_id == 2
    assert resolution.admissible is True, resolution.reason

    print("  [5] oracle Level-3 nearest uses the full direction-valid set OK")


# --------------------------------------------------------------------------
# 6. Full-dataset oracle agreement
# --------------------------------------------------------------------------


def test_oracle_full_dataset_target_match():
    """Every record must match the independent oracle exactly.

    Two independent sources are asserted:
      * the live sweep below (bounded unless SPATIAL_ORACLE_FULL=1);
      * the recorded full-dataset sweep in the acceptance quality JSON.
    """

    if not _have_v011():
        print("  [6] SKIPPED (v0.1.1 not present)")
        return

    checked = matched = 0
    mismatches: list[str] = []
    candidates_checked = 0
    references_checked = 0
    trivial_checked = 0

    for split in SPLITS:
        metadata = {r["image_id"]: r for r in V.iteration_metadata(DATASET_ROOT, split)}
        cache: dict[str, SO.OracleImage] = {}
        with (GEN_DIR / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if LIVE_LIMIT is not None and index >= LIVE_LIMIT:
                    break
                record = json.loads(line)
                image_id = record["image_id"]
                if image_id not in cache:
                    image = G.image_geometry_from_record(metadata[image_id])
                    cache[image_id] = SO.OracleImage(image, CONFIG, image.load_map(DATASET_ROOT))
                comparison = SO.compare_record(cache[image_id], record)
                checked += 1
                matched += int(comparison.target_match)
                candidates_checked += int(comparison.candidate_set_match)
                references_checked += int(comparison.reference_match)
                if comparison.trivial_match is not None:
                    trivial_checked += int(comparison.trivial_match)
                if comparison.mismatch_kinds and len(mismatches) < 10:
                    mismatches.append(f"{comparison.sample_id}: {comparison.mismatch_kinds}")
                hidden = SO.oracle_hidden_violations(cache[image_id], record)
                if hidden and len(mismatches) < 10:
                    mismatches.append(f"{comparison.sample_id}: hidden {hidden}")

    assert checked > 0, "no records were swept"
    assert not mismatches, f"oracle mismatches: {mismatches}"
    assert matched == checked, f"{matched}/{checked} target matches"
    assert candidates_checked == checked, "candidate-set agreement incomplete"
    assert references_checked == checked, "reference agreement incomplete"

    # Recorded full-dataset sweep (always all 25,229).
    if QUALITY_JSON.is_file():
        report = json.loads(QUALITY_JSON.read_text(encoding="utf-8"))
        oracle = report.get("independent_oracle") or {}
        if oracle:
            assert oracle["checked"] == report["independent_counts"]["total_records"], (
                "recorded oracle sweep did not cover every record"
            )
            assert oracle["target_mismatch"] == 0, oracle
            assert oracle["target_match"] == oracle["checked"], oracle
            assert oracle["candidate_set_mismatch"] == 0, oracle
            assert oracle["reference_mismatch"] == 0, oracle
            assert oracle["ambiguity_policy_mismatch"] == 0, oracle
            assert oracle["semantic_violation_unique_records"] == 0, oracle
            assert set(oracle["hidden_semantic_counts"].values()) == {0}, oracle
            print(
                f"  [6] oracle full-dataset sweep OK "
                f"(recorded {oracle['target_match']}/{oracle['checked']} exact matches)"
            )
        else:
            print(f"  [6] live sweep OK ({matched}/{checked}); quality JSON has no oracle block yet")
    else:
        print(f"  [6] live sweep OK ({matched}/{checked}); quality JSON absent")

    print(f"      live sweep: {checked} records, {trivial_checked} trivial-flag agreements")


# --------------------------------------------------------------------------
# 10. Frozen data
# --------------------------------------------------------------------------


def test_jsonl_hashes_unchanged():
    """Both dataset versions must still hash to their recorded values."""

    v01 = V.verify_v01_unchanged(_REPO_ROOT / "datasets")
    assert v01["unchanged"] is True, f"v0.1 modified: {v01['mismatches']}"

    if _have_v011():
        v011 = V.verify_v011_unchanged(_REPO_ROOT / "datasets")
        assert v011["unchanged"] is True, f"v0.1.1 modified: {v011['mismatches']}"
        for name, entry in v011["files"].items():
            assert entry["sha256"] == entry["recorded_sha256"], name
        print("  [10] v0.1 and v0.1.1 JSONL hashes unchanged OK")
    else:
        print("  [10] v0.1 unchanged OK; v0.1.1 absent")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("1a static independence", test_oracle_static_independence),
        ("1b runtime independence", test_oracle_runtime_independence),
        ("2 largest argmax", test_oracle_largest_is_global_argmax),
        ("3 smallest argmin", test_oracle_smallest_is_global_argmin),
        ("4 nearest over visible", test_oracle_nearest_ignores_eligibility_for_ranking),
        ("5 L3 full direction set", test_oracle_level3_uses_full_direction_set),
        ("6 full-dataset match", test_oracle_full_dataset_target_match),
        ("10 JSONL hashes", test_jsonl_hashes_unchanged),
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
    print(f"{len(tests) - failures}/{len(tests)} semantic-oracle checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
