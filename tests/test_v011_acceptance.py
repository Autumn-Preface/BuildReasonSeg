"""Acceptance tests for BuildSpatialReason-v0.1.1.

Covers the Task 5B section 19 requirements:

 1. global largest is never silently replaced
 2. global smallest is never silently replaced
 3. nearest visible target is never replaced by a farther eligible target
 4. Level-3 nearest uses the full direction-valid set
 5. semantic-invalid samples are rejected before acceptance
 6. natural-language reasoning has no internal IDs
 7. structured reasoning_steps may retain IDs
 8. template_id stored and matches zh/en
 9. reference excluded from distractors
10. target excluded from distractors
11. Level-2 Type-A + Type-B invariant
12. validator version parameterization
13. exact image-hash duplicate helper
14. deterministic regeneration on a representative subset
15. v0.1 remains unmodified

Run with pytest, or directly::

    python tests/test_v011_acceptance.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import annotator as A  # noqa: E402
import dataset_validator as V  # noqa: E402
import geometry as G  # noqa: E402
import semantic_policy as SP  # noqa: E402
import templates as TP  # noqa: E402
import thresholds as T  # noqa: E402
from component_quality import classify_image  # noqa: E402

DATASET_ROOT = _REPO_ROOT / "datasets" / "whu"
GEN_DIR = _REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1"
V01_DIR = _REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1"
SPLITS = ("train", "val", "test")
CONFIG = T.load_config()


def _have_v011() -> bool:
    return (GEN_DIR / "train.jsonl").is_file()


def _gen_config() -> dict:
    import yaml

    path = _REPO_ROOT / "configs" / "build_spatial_reason_v0.1.1.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _make_context(boxes: dict[int, tuple[int, int, int, int]], width=400, height=400):
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
                component_id=component_id, source_polygon_index=component_id - 1,
                area_px=int(mask.sum()), area_ratio=float(mask.sum()) / (width * height),
                centroid_px=(float(xs.mean()), float(ys.mean())),
                bbox_xyxy_px=(cx0, cy0, cx1, cy1),
                width_px=cx1 - cx0 + 1, height_px=cy1 - cy0 + 1,
                touches_image_border=bool(cx0 == 0 or cy0 == 0 or cx1 == width - 1 or cy1 == height - 1),
            )
        )
    image = G.ImageGeometry(
        image_id="synthetic", split="train", width=width, height=height,
        component_map_path="unused", components=components,
    )
    return V.ImageContext(image, CONFIG, component_map)


def _v011_records(limit_per_split: int | None = None) -> list[dict]:
    records: list[dict] = []
    for split in SPLITS:
        path = GEN_DIR / f"{split}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if limit_per_split is not None and index >= limit_per_split:
                    break
                if line.strip():
                    records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------
# 1 / 2. Global extremes are never silently replaced
# --------------------------------------------------------------------------


def test_global_largest_never_replaced():
    """The semantic largest must be used, or the query discarded — never swapped."""

    # Canvas is 500x500 so that component 1 clears the suspected_large_merge
    # heuristic (bbox extent <= 0.20) and the only reason it can be rejected is
    # border truncation.
    ctx = _make_context(
        {1: (0, 0, 200, 200), 2: (300, 300, 450, 450)}, width=500, height=500
    )
    flags = ctx.quality.flags(1).as_dict()
    assert flags["touches_image_border"] and not flags["suspected_large_merge"], flags

    outcome = SP.resolve_size_extreme(ctx.image, "largest", CONFIG, ctx.quality)
    assert not outcome.admissible, "a border-truncated global largest must not be admissible"
    assert outcome.semantic_target == 1, "the semantic answer is still component 1"
    assert outcome.reason == SP.REASON_SEMANTIC_TARGET_INELIGIBLE

    # The runner-up must never become the answer.
    assert outcome.semantic_target != 2

    # When the global largest is eligible it is accepted directly.
    ctx2 = _make_context(
        {1: (60, 60, 200, 200), 2: (320, 320, 450, 450)}, width=500, height=500
    )
    assert ctx2.quality.flags(1).as_dict()["geometry_valid"], ctx2.quality.flags(1).as_dict()
    ok = SP.resolve_size_extreme(ctx2.image, "largest", CONFIG, ctx2.quality)
    assert ok.admissible and ok.semantic_target == 1, ok
    print("  [1] global largest never silently replaced OK")


def test_global_smallest_never_replaced():
    ctx = _make_context({1: (100, 100, 200, 200), 2: (250, 250, 330, 330), 3: (0, 0, 20, 20)})
    outcome = SP.resolve_size_extreme(ctx.image, "smallest", CONFIG, ctx.quality)
    assert not outcome.admissible
    assert outcome.semantic_target == 3, "global smallest is component 3"
    assert outcome.semantic_target != 1
    print("  [2] global smallest never silently replaced OK")


# --------------------------------------------------------------------------
# 3. Nearest visible target is never replaced
# --------------------------------------------------------------------------


def test_nearest_visible_never_replaced_by_farther_eligible():
    # Anchor 1; component 2 is nearer but touches the border; 3 is far and fine.
    ctx = _make_context({1: (100, 150, 160, 210), 2: (0, 150, 80, 210), 3: (300, 150, 380, 210)})
    candidates = [2, 3]
    ranked = SP.nearest_over_visible(ctx.image, 1, candidates, ctx.component_map, CONFIG)
    assert ranked[0][1] == 2, "component 2 is the semantic nearest"

    outcome = SP.resolve_nearest(ctx.image, 1, candidates, ctx.component_map, CONFIG, ctx.quality)
    assert not outcome.admissible, "closer-but-ineligible nearest must discard the query"
    assert outcome.semantic_target == 2
    assert outcome.semantic_target != 3, "must NOT fall back to the farther eligible component"
    assert outcome.reason == SP.REASON_SEMANTIC_TARGET_INELIGIBLE
    print("  [3] nearest never replaced by farther eligible OK")


# --------------------------------------------------------------------------
# 4. Level-3 nearest uses the full direction-valid set
# --------------------------------------------------------------------------


def test_level3_nearest_uses_full_direction_set():
    """The Level-3 nearest must consider ineligible direction candidates too."""

    # Reference 1; components 2 and 3 lie left of it; 2 is closer but border.
    ctx = _make_context(
        {1: (150, 150, 200, 200), 2: (0, 150, 60, 210), 3: (220, 150, 280, 210)},
        width=400, height=400,
    )
    direction = SP.direction_candidates_over_visible(ctx.image, 1, "left_of", CONFIG, ctx.quality)
    assert 2 in direction, f"premise: 2 must satisfy left_of, got {direction}"

    full = SP.nearest_over_visible(ctx.image, 1, direction, ctx.component_map, CONFIG)
    assert full[0][1] == 2, "semantic nearest within the FULL direction set is 2"

    outcome = SP.resolve_nearest(ctx.image, 1, direction, ctx.component_map, CONFIG, ctx.quality)
    assert outcome.semantic_target == 2
    if not outcome.admissible:
        assert outcome.reason == SP.REASON_SEMANTIC_TARGET_INELIGIBLE
    print("  [4] Level-3 nearest uses the full direction set OK")


def test_level3_nearest_step_records_full_set():
    """On-disk Level-3 samples must record the full direction set at step 3."""

    if not _have_v011():
        print("  [4b] SKIPPED (no v0.1.1 dataset)")
        return
    checked = 0
    for record in _v011_records(limit_per_split=400):
        if record["level"] != 3:
            continue
        filter_step = next(s for s in record["reasoning_steps"] if s["operation"] == "filter_relation")
        nearest_step = next(s for s in record["reasoning_steps"] if s["operation"] == "argmin_boundary_distance")
        assert sorted(nearest_step["candidate_component_ids"]) == sorted(
            filter_step["candidate_component_ids"]
        ), f"{record['sample_id']}: nearest step must cover the full direction set"
        checked += 1
    assert checked > 0
    print(f"  [4b] Level-3 nearest step records the full set OK ({checked} checked)")


# --------------------------------------------------------------------------
# 5. Semantic-invalid samples rejected before acceptance
# --------------------------------------------------------------------------


def test_no_semantic_violations_on_disk():
    """A stored v0.1.1 sample must never be semantically false."""

    if not _have_v011():
        print("  [5] SKIPPED (no v0.1.1 dataset)")
        return
    violations = 0
    cache: dict[tuple[str, str], V.ImageContext] = {}
    for split in SPLITS:
        meta = {r["image_id"]: r for r in V.iteration_metadata(DATASET_ROOT, split)}
        path = GEN_DIR / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= 1200:
                    break
                record = json.loads(line)
                key = (split, record["image_id"])
                if key not in cache:
                    image = G.image_geometry_from_record(meta[record["image_id"]])
                    cache[key] = V.ImageContext(image, CONFIG)
                ctx = cache[key]
                quality = ctx.quality
                target = record["target_component_id"]
                query_type = record["query_type"]

                if "largest" in query_type:
                    ref = target if record["level"] == 1 else record["reference_component_ids"][0]
                    if V.audit_hidden_largest(ctx, ref):
                        violations += 1
                if "smallest" in query_type:
                    ref = target if record["level"] == 1 else record["reference_component_ids"][0]
                    if V.audit_hidden_smallest(ctx, ref):
                        violations += 1
                if record["level"] == 2 and query_type.endswith("_to_nearest"):
                    if V.audit_hidden_nearest(ctx, record["reference_component_ids"][0], target):
                        violations += 1
                if record["level"] == 3:
                    fs = next(s for s in record["reasoning_steps"] if s["operation"] == "filter_relation")
                    if V.audit_hidden_level3_nearest(
                        ctx, fs["reference_component_id"], fs["relation"], target
                    ):
                        violations += 1
    assert violations == 0, f"{violations} semantic violations survived into v0.1.1"
    print("  [5] no semantic violations on disk OK")


# --------------------------------------------------------------------------
# 6 / 7. ID-free natural language while steps keep IDs
# --------------------------------------------------------------------------


def test_reasoning_is_id_free_on_disk():
    if not _have_v011():
        print("  [6] SKIPPED")
        return
    offenders = []
    total = 0
    for split in SPLITS:
        with (GEN_DIR / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                total += 1
                for field in ("reasoning_zh", "reasoning_en", "instruction_zh", "instruction_en"):
                    if V.find_component_id_leaks(record[field]):
                        offenders.append((record["sample_id"], field))
    assert not offenders, f"{len(offenders)} language fields leak ids, first: {offenders[:3]}"
    print(f"  [6] reasoning/instruction ID-free OK ({total} records)")


def test_structured_steps_retain_ids():
    """The machine layer must still carry numeric ids for verification."""

    if not _have_v011():
        print("  [7] SKIPPED")
        return
    checked = 0
    for record in _v011_records(limit_per_split=200):
        assert isinstance(record["reasoning_steps"], list) and record["reasoning_steps"]
        for step in record["reasoning_steps"]:
            assert step["step"] >= 1
            if step["operation"] in A.LEVEL1_OPERATIONS.values():
                assert isinstance(step["output_component_id"], int), step
        checked += 1
    assert checked > 0
    print(f"  [7] structured steps retain numeric ids OK ({checked} records)")


def test_renderer_output_is_id_free_for_all_operations():
    """Directly exercise the renderer, independent of on-disk data."""

    steps = [
        {"step": 1, "operation": "argmax_area", "output_component_id": 7},
        {
            "step": 2, "operation": "filter_relation", "relation": "right_of",
            "reference_component_id": 7, "candidate_component_ids": [2, 5, 9],
            "nearest_eligible_component_ids": [5, 9],
        },
        {
            "step": 3, "operation": "argmin_boundary_distance",
            "reference_component_id": 7, "candidate_component_ids": [2, 5, 9],
            "output_component_id": 5,
        },
    ]
    zh, en = A.render_reasoning(steps)
    assert not V.find_component_id_leaks(zh), zh
    assert not V.find_component_id_leaks(en), en
    assert "7" not in en and "5" not in en and "9" not in en, en
    print("  [6b] renderer output is ID-free OK")


# --------------------------------------------------------------------------
# 8. template_id
# --------------------------------------------------------------------------


def test_template_id_stored_and_matches():
    if not _have_v011():
        print("  [8] SKIPPED")
        return
    checked = 0
    for record in _v011_records(limit_per_split=600):
        stored = record.get("template_id")
        assert stored, f"{record['sample_id']}: template_id missing"
        reconstructed = V.reconstruct_template_id(
            record["query_type"], record["instruction_zh"], record["instruction_en"]
        )
        assert reconstructed == stored, (
            f"{record['sample_id']}: stored {stored} != reconstructed {reconstructed}"
        )
        checked += 1
    print(f"  [8] template_id stored and matches zh/en OK ({checked} records)")


def test_zh_en_share_template_id():
    """zh and en must come from the same template pair."""

    a = TP.LEVEL1_TEMPLATES["leftmost"][0]
    b = TP.LEVEL1_TEMPLATES["rightmost"][0]
    assert V.reconstruct_template_id("leftmost", a.zh, b.en) is None
    print("  [8b] zh/en must share one template id OK")


# --------------------------------------------------------------------------
# 9 / 10. Distractor policy
# --------------------------------------------------------------------------


def test_reference_and_target_excluded_from_distractors():
    if not _have_v011():
        print("  [9] SKIPPED")
        return
    checked = 0
    for record in _v011_records(limit_per_split=800):
        distractors = set(record["distractor_component_ids"])
        assert record["target_component_id"] not in distractors, record["sample_id"]
        for reference_id in record["reference_component_ids"]:
            assert reference_id not in distractors, (
                f"{record['sample_id']}: reference {reference_id} in distractors"
            )
        checked += 1
    print(f"  [9/10] target and reference excluded from distractors OK ({checked} records)")


# --------------------------------------------------------------------------
# 11. Level-2 invariant
# --------------------------------------------------------------------------


def test_level2_invariant_on_disk_and_in_stats():
    if not _have_v011():
        print("  [11] SKIPPED")
        return
    from collections import Counter

    by_level = Counter()
    type_a = type_b = 0
    for record in _v011_records():
        by_level[record["level"]] += 1
        if record["level"] == 2:
            if record["query_type"].endswith("_to_nearest"):
                type_a += 1
            else:
                type_b += 1
    assert type_a + type_b == by_level[2], (type_a, type_b, by_level[2])

    stats = json.loads((GEN_DIR / "statistics.json").read_text(encoding="utf-8"))
    assert stats["level2"]["reference_to_nearest"] == type_a, (
        f"statistics says {stats['level2']['reference_to_nearest']}, recount {type_a}"
    )
    assert stats["level2"]["reference_to_direction"] == type_b
    assert stats["level2"].get("invariant_holds") is True
    print(f"  [11] Level-2 invariant OK (A={type_a} + B={type_b} = {by_level[2]})")


# --------------------------------------------------------------------------
# 12. Validator version parameterization
# --------------------------------------------------------------------------


def test_validator_version_parameterization():
    assert V.read_dataset_records(_REPO_ROOT / "datasets", "train", "v0.1") if _have_v011() else True
    if not _have_v011():
        print("  [12] SKIPPED")
        return
    records = V.read_dataset_records(_REPO_ROOT / "datasets", "train", "v0.1.1")
    assert records, "v0.1.1 train records must load"
    assert records[0]["dataset_version"] == "v0.1.1"
    # v0.1 must still be separately readable.
    v01 = V.read_dataset_records(_REPO_ROOT / "datasets", "train", "v0.1")
    assert v01 and v01[0]["dataset_version"] == "v0.1"
    print("  [12] validator version parameterization OK")


# --------------------------------------------------------------------------
# 13. Exact image-hash duplicate helper
# --------------------------------------------------------------------------


def test_exact_image_hash_duplicate_helper(tmp_path: Path | None = None):
    base = tmp_path or (_REPO_ROOT / ".pytest_tmp_v011_hash")
    base.mkdir(parents=True, exist_ok=True)
    try:
        shared = base / "shared.tif"
        shared.write_bytes(b"identical")
        other = base / "other.tif"
        other.write_bytes(b"different")
        fake_root = base / "fake"
        (fake_root / "build_spatial_reason" / "v0.1.1").mkdir(parents=True, exist_ok=True)
        records = {
            "train": [{"image_path": str(shared)}],
            "val": [{"image_path": str(shared)}],
            "test": [{"image_path": str(other)}],
        }
        report = V.exact_cross_split_image_duplicates(fake_root, ["train", "val", "test"], records)
        assert report["exact_image_duplicate_cross_split"] == 1, report
        print("  [13] exact image-hash duplicate helper OK")
    finally:
        if tmp_path is None:
            import shutil

            shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------
# 14. Deterministic regeneration on a representative subset
# --------------------------------------------------------------------------


def test_deterministic_regeneration_subset():
    """Regenerating the same subset twice must be byte-identical."""

    if not _have_v011():
        print("  [14] SKIPPED")
        return
    from build_spatial_reason import generate_split

    gen_config = _gen_config()
    relation_config = T.load_config(gen_config["relations"]["config"])
    meta = A.dataset_meta_from_config(gen_config)

    def run() -> str:
        counter = A.DiscardCounter()
        seen: set[str] = set()
        samples = generate_split(
            "train", DATASET_ROOT, relation_config, gen_config, meta, 12, counter, seen, quiet=True
        )
        return "\n".join(json.dumps(s, ensure_ascii=False, sort_keys=True) for s in samples)

    assert run() == run(), "v0.1.1 regeneration is not byte-identical"
    print("  [14] deterministic regeneration on a subset OK")


# --------------------------------------------------------------------------
# 15. v0.1 remains unmodified
# --------------------------------------------------------------------------


def test_v01_remains_unmodified():
    if not V01_DIR.is_dir():
        print("  [15] SKIPPED (no v0.1 directory)")
        return
    report = V.verify_v01_unchanged(_REPO_ROOT / "datasets")
    assert report["unchanged"] is True, report["mismatches"]
    print("  [15] v0.1 remains byte-identical OK")


# --------------------------------------------------------------------------
# Extra: the v0.1.1 manifest records the required provenance
# --------------------------------------------------------------------------


def test_manifest_provenance_fields():
    if not _have_v011():
        print("  [extra] SKIPPED")
        return
    manifest = json.loads((GEN_DIR / "manifest.json").read_text(encoding="utf-8"))
    for key in (
        "dataset_name", "dataset_version", "project_name", "source_dataset",
        "source_subset", "source_component_representation_version",
        "semantic_visibility_policy_version", "relation_config",
        "relation_config_version", "relation_config_sha256",
        "generator_version", "generation_seed", "template_version",
    ):
        assert key in manifest, f"manifest missing {key}"
    assert manifest["dataset_version"] == "v0.1.1"
    assert manifest["semantic_visibility_policy_version"] == "1.0"
    assert manifest["dataset_name"] == "BuildSpatialReason"
    print("  [extra] manifest provenance fields OK")


def test_trivial_flag_matches_admissible_count():
    """trivial_selection must equal 'exactly one admissible candidate'."""

    if not _have_v011():
        print("  [extra2] SKIPPED")
        return
    checked = trivial = 0
    for split in ("train",):
        meta = {r["image_id"]: r for r in V.iteration_metadata(DATASET_ROOT, split)}
        cache: dict[str, V.ImageContext] = {}
        with (GEN_DIR / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= 1500:
                    break
                record = json.loads(line)
                if record["level"] != 3:
                    continue
                image_id = record["image_id"]
                if image_id not in cache:
                    cache[image_id] = V.ImageContext(G.image_geometry_from_record(meta[image_id]), CONFIG)
                ctx = cache[image_id]
                fs = next(s for s in record["reasoning_steps"] if s["operation"] == "filter_relation")
                admissible = SP.admissible_nearest_ids(
                    ctx.image, fs["reference_component_id"],
                    fs["candidate_component_ids"], ctx.component_map, CONFIG, ctx.quality,
                )
                expected = len(admissible) == 1
                assert bool(record["trivial_selection"]) == expected, (
                    f"{record['sample_id']}: trivial={record['trivial_selection']} "
                    f"but admissible={len(admissible)}"
                )
                checked += 1
                trivial += int(expected)
    assert checked > 0
    print(f"  [extra2] trivial flag matches admissible count OK ({checked} checked, {trivial} trivial)")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("1 global largest", test_global_largest_never_replaced),
        ("2 global smallest", test_global_smallest_never_replaced),
        ("3 nearest visible", test_nearest_visible_never_replaced_by_farther_eligible),
        ("4 L3 full direction set", test_level3_nearest_uses_full_direction_set),
        ("4b L3 step records set", test_level3_nearest_step_records_full_set),
        ("5 no semantic violations", test_no_semantic_violations_on_disk),
        ("6 reasoning ID-free", test_reasoning_is_id_free_on_disk),
        ("6b renderer ID-free", test_renderer_output_is_id_free_for_all_operations),
        ("7 steps retain IDs", test_structured_steps_retain_ids),
        ("8 template_id", test_template_id_stored_and_matches),
        ("8b zh/en share id", test_zh_en_share_template_id),
        ("9/10 distractors", test_reference_and_target_excluded_from_distractors),
        ("11 L2 invariant", test_level2_invariant_on_disk_and_in_stats),
        ("12 version param", test_validator_version_parameterization),
        ("13 hash helper", test_exact_image_hash_duplicate_helper),
        ("14 determinism", test_deterministic_regeneration_subset),
        ("15 v0.1 unmodified", test_v01_remains_unmodified),
        ("extra manifest", test_manifest_provenance_fields),
        ("extra2 trivial flag", test_trivial_flag_matches_admissible_count),
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
    print(f"{len(tests) - failures}/{len(tests)} v0.1.1 acceptance checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
