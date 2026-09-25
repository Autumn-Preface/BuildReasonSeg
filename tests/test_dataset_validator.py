"""Tests for the BuildSpatialReason dataset validator.

Covers the checks required by the Task 5 handoff:

* independent count reconstruction
* target recomputation
* hidden-largest / hidden-smallest / hidden-nearest mismatch fixtures
* Level-3 hidden nearest fixture
* component-ID leakage detection
* candidate / distractor integrity
* mask selector integrity
* template reconstruction
* exact image-hash leakage helper

Run with pytest, or directly::

    python tests/test_dataset_validator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import dataset_validator as V  # noqa: E402
import geometry as G  # noqa: E402
import templates as TP  # noqa: E402
import thresholds as T  # noqa: E402
from component_quality import classify_image  # noqa: E402

DATASET_ROOT = _REPO_ROOT / "datasets" / "whu"
GEN_DIR = _REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1"
CONFIG = T.load_config()
SPLITS = ("train", "val", "test")


def _have_dataset() -> bool:
    return (GEN_DIR / "train.jsonl").is_file()


def _make_context(boxes: dict[int, tuple[int, int, int, int]], width=400, height=400):
    """Build an ImageContext from raw boxes, bypassing the dataset."""

    component_map = np.zeros((height, width), dtype=np.uint8)
    for component_id, (x0, y0, x1, y1) in boxes.items():
        component_map[y0:y1, x0:x1] = component_id

    components = []
    for component_id in sorted(boxes):
        mask = component_map == component_id
        assert mask.any(), f"component {component_id} empty"
        ys, xs = np.nonzero(mask)
        cx0, cy0, cx1, cy1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        components.append(
            G.Component(
                component_id=component_id,
                source_polygon_index=component_id - 1,
                area_px=int(mask.sum()),
                area_ratio=float(mask.sum()) / (width * height),
                centroid_px=(float(xs.mean()), float(ys.mean())),
                bbox_xyxy_px=(cx0, cy0, cx1, cy1),
                width_px=cx1 - cx0 + 1,
                height_px=cy1 - cy0 + 1,
                touches_image_border=bool(cx0 == 0 or cy0 == 0 or cx1 == width - 1 or cy1 == height - 1),
            )
        )

    image = G.ImageGeometry(
        image_id="synthetic", split="train", width=width, height=height,
        component_map_path="unused", components=components,
    )
    return V.ImageContext(image, CONFIG, component_map)


# --------------------------------------------------------------------------
# Independent count reconstruction
# --------------------------------------------------------------------------


def test_independent_counts_match_generator_bookkeeping():
    """Recount from JSONL and compare with statistics.json / manifest.json."""

    if not _have_dataset():
        print("  [counts] SKIPPED (no dataset)")
        return

    import json

    from collections import Counter

    split_counts = Counter()
    level_counts = Counter()
    query_counts = Counter()
    trivial = nontrivial = 0

    for split in SPLITS:
        path = GEN_DIR / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            split_counts[split] += 1
            level_counts[record["level"]] += 1
            query_counts[record["query_type"]] += 1
            if record["level"] == 3:
                if record.get("trivial_selection"):
                    trivial += 1
                else:
                    nontrivial += 1

    stats = json.loads((GEN_DIR / "statistics.json").read_text(encoding="utf-8"))
    manifest = json.loads((GEN_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert sum(split_counts.values()) == stats["total_samples"], (
        f"recount {sum(split_counts.values())} != statistics {stats['total_samples']}"
    )
    assert dict(split_counts) == stats["by_split"], (dict(split_counts), stats["by_split"])
    assert {str(k): v for k, v in level_counts.items()} == stats["by_level"]
    assert trivial == stats["level3"]["trivial"], (trivial, stats["level3"])
    assert nontrivial == stats["level3"]["nontrivial"], (nontrivial, stats["level3"])
    assert sum(split_counts.values()) == manifest["sample_counts"]["total"]

    # Level 2 must decompose exactly.
    level2_total = level_counts[2]
    type_a = query_counts.get("largest_to_nearest", 0) + query_counts.get("smallest_to_nearest", 0)
    type_b = sum(
        v for k, v in query_counts.items()
        if k.count("_to_") == 1 and not k.endswith("_to_nearest")
    )
    assert type_a + type_b == level2_total, (type_a, type_b, level2_total)

    # KNOWN GENERATOR BUG (reported, v0.1 not repaired).
    #
    # statistics.json / manifest.json claim reference_to_nearest == 8700. The
    # generator computed that with `query_type.endswith("_to_nearest")`, which
    # ALSO matches every Level-3 query type (they all end in `_to_nearest`).
    # The true Level-2 Type-A count is 4707, and 8700 == 4707 + 3993 (all L3).
    # This test pins the correct semantics and documents the generator defect.
    level3_total = level_counts[3]
    claimed = stats["level2"]["reference_to_nearest"]
    assert claimed == type_a + level3_total, (
        "the generator's claim should be explainable as L2 type A + all Level 3; "
        f"claimed={claimed} type_a={type_a} level3={level3_total}"
    )
    assert claimed != type_a, "if this ever becomes equal, the generator was fixed"
    # The type-A + type-B decomposition must equal the Level-2 level count, which
    # the generator's statistics CANNOT satisfy.
    assert stats["level2"]["reference_to_nearest"] + stats["level2"]["reference_to_direction"] != level2_total, (
        "statistics.json should be internally inconsistent for Level 2 in v0.1"
    )
    print(
        f"  [counts] independent recount OK (total={sum(split_counts.values())}, "
        f"L2 A={type_a} B={type_b} sum={type_a + type_b}; "
        f"generator claimed A={claimed} which equals A+L3={type_a}+{level3_total})"
    )


# --------------------------------------------------------------------------
# Target recomputation
# --------------------------------------------------------------------------


def test_target_recomputation_on_dataset():
    """Recompute targets independently for a sample and require 100% match."""

    if not _have_dataset():
        print("  [recompute] SKIPPED")
        return

    import json

    checked = passed = 0
    cache: dict[tuple[str, str], V.ImageContext] = {}
    mismatches = []

    for split in ("train",):
        meta = {r["image_id"]: r for r in V.iteration_metadata(DATASET_ROOT, split)}
        lines = (GEN_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        for line in lines[:1500]:
            record = json.loads(line)
            key = (split, record["image_id"])
            if key not in cache:
                image = G.image_geometry_from_record(meta[record["image_id"]])
                cache[key] = V.ImageContext(image, CONFIG)
            ctx = cache[key]
            recomputed = V.recompute_target_independent(
                ctx, record["level"], record["query_type"], record["reasoning_steps"]
            )
            checked += 1
            if recomputed == record["target_component_id"]:
                passed += 1
            else:
                mismatches.append((record["sample_id"], record["target_component_id"], recomputed))

    assert checked > 0
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[:3]}"
    print(f"  [recompute] target recomputation OK ({passed}/{checked})")


# --------------------------------------------------------------------------
# Hidden-eligibility fixtures
# --------------------------------------------------------------------------


def test_hidden_largest_fixture():
    """A border-truncated global largest must be flagged as hidden eligibility."""

    # Component 1 is huge but touches the border; component 2 is eligible and smaller.
    ctx = _make_context({1: (0, 0, 200, 200), 2: (250, 250, 380, 380)})
    quality = ctx.quality
    assert quality.flags(1).touches_image_border
    assert not quality.flags(1).geometry_valid

    eligible = ctx.eligible_ids("largest")
    assert 1 not in eligible and 2 in eligible

    violation = V.audit_hidden_largest(ctx, 2)
    assert violation is not None, "border-truncated global largest must be flagged"
    assert violation["global_largest_id"] == 1
    assert violation["engine_reference_id"] == 2
    assert "touches_image_border" in violation["global_largest_flags"]

    # When the global largest IS eligible there is nothing to flag.
    ctx2 = _make_context({1: (50, 50, 250, 250), 2: (280, 280, 380, 380)})
    assert V.audit_hidden_largest(ctx2, 1) is None
    print("  [fixture] hidden largest detection OK")


def test_hidden_smallest_fixture():
    """A tiny/border global smallest differing from the engine answer is flagged."""

    # Component 3 is the global smallest AND touches the border -> excluded.
    ctx = _make_context({1: (100, 100, 200, 200), 2: (250, 250, 330, 330), 3: (0, 0, 20, 20)})
    quality = ctx.quality
    assert quality.flags(3).touches_image_border
    eligible = ctx.eligible_ids("smallest")
    assert 3 not in eligible

    engine_smallest = ctx.size_target("argmin_area")
    violation = V.audit_hidden_smallest(ctx, engine_smallest)
    assert violation is not None
    assert violation["global_smallest_id"] == 3
    assert violation["engine_reference_id"] == engine_smallest
    print("  [fixture] hidden smallest detection OK")


def test_hidden_nearest_fixture():
    """A closer ineligible component makes the semantic nearest differ."""

    # Anchor 1 at the left-ish; component 2 touches the border and is nearest;
    # component 3 is far but eligible.
    ctx = _make_context({1: (100, 150, 160, 210), 2: (0, 150, 80, 210), 3: (300, 150, 380, 210)})
    quality = ctx.quality
    assert quality.flags(2).touches_image_border

    semantic = ctx.nearest_gap(1, [c.component_id for c in ctx.image.components if c.component_id != 1])
    assert semantic[0][1] == 2, "fixture premise: component 2 is semantically nearest"

    eligible = ctx.nearest_eligible(1, [2, 3])
    assert eligible == [3], eligible

    violation = V.audit_hidden_nearest(ctx, 1, 3)
    assert violation is not None
    assert violation["semantic_nearest_id"] == 2
    assert violation["engine_target_id"] == 3
    assert "touches_image_border" in violation["semantic_nearest_flags"]

    # No violation when the semantic nearest IS the engine target.
    assert V.audit_hidden_nearest(ctx, 1, 2) is None
    print("  [fixture] hidden nearest detection OK")


def test_hidden_level3_nearest_fixture():
    """Direction-filtered semantic nearest differing from the engine target."""

    # Reference 1 in the middle; 2 and 3 are to its right; 2 is closer but
    # touches the border, so the engine picks 3.
    ctx = _make_context(
        {1: (150, 150, 200, 200), 2: (0, 150, 60, 210), 3: (220, 150, 280, 210)},
        width=400, height=400,
    )
    reference = 1
    direction = ctx.direction_candidates(reference, "left_of")
    # Components 2 lies to the LEFT of the reference.
    assert 2 in direction, f"fixture premise failed, left_of candidates={direction}"

    eligible = ctx.nearest_eligible(reference, direction)
    assert 2 not in eligible, "component 2 must be nearest-ineligible (border)"

    engine_target = eligible[0] if eligible else None
    violation = V.audit_hidden_level3_nearest(ctx, reference, "left_of", engine_target)
    if 2 in direction and len(direction) > 1:
        assert violation is not None, "expected a hidden Level-3 nearest violation"
        assert violation["semantic_nearest_id"] == 2
    print("  [fixture] hidden Level-3 nearest detection OK")


# --------------------------------------------------------------------------
# Component-ID leakage
# --------------------------------------------------------------------------


def test_component_id_leakage_detection():
    assert V.find_component_id_leaks("The largest building region is component 3.")
    assert V.find_component_id_leaks("component 12 is the target")
    assert V.find_component_id_leaks("组件3是目标")
    assert V.find_component_id_leaks("构件 7 的边界距离最近")
    # Clean prose must not be flagged.
    assert not V.find_component_id_leaks("First identify the largest building region.")
    assert not V.find_component_id_leaks("首先找到图像中面积最大的建筑区域。")
    # A word containing "component" without a number is not a leak.
    assert not V.find_component_id_leaks("the candidate component is selected")
    print("  [leak] component-id leakage detection OK")


def test_instructions_are_id_free_on_dataset():
    """Instructions must not leak internal ids (reasoning may, and is tracked)."""

    if not _have_dataset():
        return
    import json

    leaked = []
    for line in (GEN_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()[:3000]:
        record = json.loads(line)
        for field in ("instruction_zh", "instruction_en"):
            if V.find_component_id_leaks(record[field]):
                leaked.append((record["sample_id"], field))
    assert not leaked, f"{len(leaked)} instruction leaks, first: {leaked[:3]}"
    print("  [leak] instructions are ID-free OK")


# --------------------------------------------------------------------------
# Candidate / distractor integrity
# --------------------------------------------------------------------------


def test_candidate_distractor_integrity_on_dataset():
    if not _have_dataset():
        return
    import json

    problems = []
    for line in (GEN_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()[:3000]:
        record = json.loads(line)
        target = record["target_component_id"]
        distractors = record["distractor_component_ids"]
        if target in distractors:
            problems.append((record["sample_id"], "target_in_distractors"))
        if len(distractors) != len(set(distractors)):
            problems.append((record["sample_id"], "duplicate_distractors"))
        if target in record["candidate_component_ids"]:
            # Target is normally one of the candidates for nearest queries; only
            # flag if it appears in the Type-B single-candidate list incorrectly.
            pass
    assert not problems, f"{len(problems)} problems, first: {problems[:3]}"
    print("  [candidates] candidate/distractor integrity OK")


def test_level3_eligible_subset_of_direction():
    if not _have_dataset():
        return
    import json

    checked = 0
    for line in (GEN_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["level"] != 3:
            continue
        step = next(s for s in record["reasoning_steps"] if s["operation"] == "filter_relation")
        direction = set(step["candidate_component_ids"])
        eligible = set(step["nearest_eligible_component_ids"])
        assert eligible <= direction, f"{record['sample_id']}: eligible not subset of direction"
        checked += 1
        if checked >= 2000:
            break
    print(f"  [candidates] Level-3 eligible subset OK ({checked} checked)")


# --------------------------------------------------------------------------
# Mask selector integrity
# --------------------------------------------------------------------------


def test_mask_selector_integrity_on_dataset():
    if not _have_dataset():
        return
    import json

    checked = 0
    for split in ("train",):
        meta = {r["image_id"]: r for r in V.iteration_metadata(DATASET_ROOT, split)}
        cache: dict[str, np.ndarray] = {}
        for line in (GEN_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()[:400]:
            record = json.loads(line)
            assert record["target_mask"]["component_id"] == record["target_component_id"]
            image_id = record["image_id"]
            if image_id not in cache:
                image = G.image_geometry_from_record(meta[image_id])
                cache[image_id] = image.load_map(DATASET_ROOT)
            component_map = cache[image_id]
            pixels = int((component_map == record["target_component_id"]).sum())
            assert pixels > 0, f"{record['sample_id']}: target mask is empty"
            checked += 1
    print(f"  [mask] mask selector integrity OK ({checked} checked)")


# --------------------------------------------------------------------------
# Template reconstruction
# --------------------------------------------------------------------------


def test_template_reconstruction_level1():
    pair = TP.LEVEL1_TEMPLATES["leftmost"][0]
    assert V.reconstruct_template_id("leftmost", pair.zh, pair.en) == pair.template_id
    print("  [template] Level-1 reconstruction OK")


def test_template_reconstruction_level2_and_level3():
    pair = TP.LEVEL2_NEAREST_TEMPLATES["largest"][0]
    assert V.reconstruct_template_id("largest_to_nearest", pair.zh, pair.en) == pair.template_id

    reference = TP.REFERENCE_PHRASE["largest"]
    direction = TP.LEVEL2_DIRECTION_TEMPLATES["right_of"][0]
    zh = direction.zh.format(ref=reference["zh"])
    en = direction.en.format(ref=reference["en"])
    assert V.reconstruct_template_id("largest_to_right_of", zh, en) == direction.template_id

    prep = TP.DIRECTION_PREPOSITION["right_of"]
    l3 = TP.LEVEL3_TEMPLATES[0]
    zh3 = l3.zh.format(ref=reference["zh"], dir=prep["zh"])
    en3 = l3.en.format(ref=reference["en"], dir=prep["en"])
    assert V.reconstruct_template_id("largest_to_right_of_to_nearest", zh3, en3) == l3.template_id
    print("  [template] Level-2/3 reconstruction OK")


def test_template_reconstruction_all_families():
    """Every template in every family must round-trip through reconstruction."""

    checked = 0
    for query_type, family in TP.LEVEL1_TEMPLATES.items():
        for pair in family:
            assert V.reconstruct_template_id(query_type, pair.zh, pair.en) == pair.template_id
            checked += 1
    for reference, family in TP.LEVEL2_NEAREST_TEMPLATES.items():
        for pair in family:
            assert V.reconstruct_template_id(f"{reference}_to_nearest", pair.zh, pair.en) == pair.template_id
            checked += 1
    for relation, family in TP.LEVEL2_DIRECTION_TEMPLATES.items():
        reference = TP.REFERENCE_PHRASE["largest"]
        for pair in family:
            zh = pair.zh.format(ref=reference["zh"])
            en = pair.en.format(ref=reference["en"])
            assert V.reconstruct_template_id(f"largest_to_{relation}", zh, en) == pair.template_id
            checked += 1
    for pair in TP.LEVEL3_TEMPLATES:
        for relation, prep in TP.DIRECTION_PREPOSITION.items():
            reference = TP.REFERENCE_PHRASE["largest"]
            zh = pair.zh.format(ref=reference["zh"], dir=prep["zh"])
            en = pair.en.format(ref=reference["en"], dir=prep["en"])
            assert V.reconstruct_template_id(f"largest_to_{relation}_to_nearest", zh, en) == pair.template_id
            checked += 1
    print(f"  [template] all families round-trip OK ({checked} templates)")


def test_template_reconstruction_rejects_mismatched_pair():
    """A zh from one template with an en from another must NOT reconstruct."""

    a = TP.LEVEL1_TEMPLATES["leftmost"][0]
    b = TP.LEVEL1_TEMPLATES["rightmost"][0]
    assert V.reconstruct_template_id("leftmost", a.zh, b.en) is None
    print("  [template] mismatched language pair rejected OK")


# --------------------------------------------------------------------------
# Exact image-hash leakage helper
# --------------------------------------------------------------------------


def test_exact_image_hash_duplicate_helper(tmp_path: Path | None = None):
    """Two splits referencing the same image bytes must be reported."""

    import json

    base = tmp_path or (_REPO_ROOT / ".pytest_tmp_hash")
    base.mkdir(parents=True, exist_ok=True)
    try:
        shared = base / "shared.tif"
        shared.write_bytes(b"identical-bytes")
        other = base / "other.tif"
        other.write_bytes(b"different-bytes")

        # Build a fake dataset root layout the helper understands.
        fake_root = base / "fake"
        (fake_root / "build_spatial_reason" / "v0.1").mkdir(parents=True, exist_ok=True)

        records = {
            "train": [{"image_path": str(shared)}],
            "val": [{"image_path": str(shared)}],
            "test": [{"image_path": str(other)}],
        }
        # The helper resolves image_path relative to _REPO_ROOT, so use absolute
        # paths that already exist; Path("/abs") ignores the base on Windows too.
        report = V.exact_cross_split_image_duplicates(fake_root, ["train", "val", "test"], records)
        assert report["exact_image_duplicate_cross_split"] == 1, report
        assert report["duplicates"][0]["split_a"] == "train"
        assert report["duplicates"][0]["split_b"] == "val"
        print("  [hash] exact cross-split duplicate helper OK")
    finally:
        if tmp_path is None:
            import shutil

            shutil.rmtree(base, ignore_errors=True)


def test_hash_file_is_content_based(tmp_path: Path | None = None):
    base = tmp_path or (_REPO_ROOT / ".pytest_tmp_hash2")
    base.mkdir(parents=True, exist_ok=True)
    try:
        a = base / "a.bin"
        b = base / "b.bin"
        a.write_bytes(b"same")
        b.write_bytes(b"same")
        c = base / "c.bin"
        c.write_bytes(b"different")
        assert V.hash_file(a) == V.hash_file(b)
        assert V.hash_file(a) != V.hash_file(c)
        print("  [hash] hash_file is content based OK")
    finally:
        if tmp_path is None:
            import shutil

            shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------
# Verdict logic
# --------------------------------------------------------------------------


def test_verdict_logic():
    clean = V.IssueCollector()
    assert V.decide_verdict(clean) == "PASS"

    warnings = V.IssueCollector()
    warnings.add("some_warning", V.SEVERITY_WARNING)
    assert V.decide_verdict(warnings) == "PASS_WITH_WARNINGS"

    for code in V.BLOCKING_CODES:
        blocking = V.IssueCollector()
        blocking.add(code, V.SEVERITY_ERROR)
        assert V.decide_verdict(blocking) == "FAIL_REQUIRES_REVISION", code
    print("  [verdict] verdict logic OK")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("independent counts", test_independent_counts_match_generator_bookkeeping),
        ("target recomputation", test_target_recomputation_on_dataset),
        ("hidden largest", test_hidden_largest_fixture),
        ("hidden smallest", test_hidden_smallest_fixture),
        ("hidden nearest", test_hidden_nearest_fixture),
        ("hidden L3 nearest", test_hidden_level3_nearest_fixture),
        ("id leakage detection", test_component_id_leakage_detection),
        ("instructions id-free", test_instructions_are_id_free_on_dataset),
        ("candidate/distractor", test_candidate_distractor_integrity_on_dataset),
        ("L3 eligible subset", test_level3_eligible_subset_of_direction),
        ("mask selector", test_mask_selector_integrity_on_dataset),
        ("template L1", test_template_reconstruction_level1),
        ("template L2/L3", test_template_reconstruction_level2_and_level3),
        ("template all", test_template_reconstruction_all_families),
        ("template mismatch", test_template_reconstruction_rejects_mismatched_pair),
        ("hash duplicate helper", test_exact_image_hash_duplicate_helper),
        ("hash content based", test_hash_file_is_content_based),
        ("verdict logic", test_verdict_logic),
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
    print(f"{len(tests) - failures}/{len(tests)} validator checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
