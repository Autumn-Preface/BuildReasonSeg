"""Tests for the BuildSpatialReason-v0.1 annotator.

Covers the nine required checks:

    A. Determinism            same input twice -> byte-identical JSONL
    B. Target recomputation   re-executing reasoning_steps yields the same target
    C. No hidden tie-break    >1 direction candidate -> no Type-B Level-2 sample
    D. Conditional nearest    nearest_within only sees the filtered candidate set
    E. Predicate semantics    "A is left of B" <-> left_of(A, B)
    F. Split preservation     no image leakage across splits
    G. Duplicate prevention   canonical semantic key is unique
    H. Language parity        zh and en come from the same template semantic id
    I. Mask selector          target_component_id == target_mask.component_id

Run with pytest, or directly::

    python tests/test_annotator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import annotator as A  # noqa: E402
import geometry as G  # noqa: E402
import relations as R  # noqa: E402
import templates as TP  # noqa: E402
import thresholds as T  # noqa: E402
from component_quality import classify_image  # noqa: E402

DATASET_ROOT = _REPO_ROOT / "datasets" / "whu"
GENERATED_DIR = _REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1"
SPLITS = ("train", "val", "test")

CONFIG = T.load_config()
GEN_CONFIG = json.loads("{}")  # replaced lazily; kept for clarity


def _gen_config() -> dict:
    import yaml

    with (_REPO_ROOT / "configs" / "build_spatial_reason_v0.1.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dataset_meta() -> dict:
    return A.dataset_meta_from_config(_gen_config())


def _have_dataset() -> bool:
    return (DATASET_ROOT / "metadata" / "train.jsonl").is_file()


def _generate(limit: int = 12, split: str = "train", gen_config: dict | None = None) -> list[dict]:
    """Generate samples for a small deterministic subset."""

    from build_spatial_reason import generate_split

    gen_config = gen_config or _gen_config()
    relation_config = T.load_config(gen_config["relations"]["config"])
    counter = A.DiscardCounter()
    seen: set[str] = set()
    return generate_split(
        split,
        DATASET_ROOT,
        relation_config,
        gen_config,
        A.dataset_meta_from_config(gen_config),
        limit,
        counter,
        seen,
        quiet=True,
    )


def _serialise(samples: list[dict]) -> str:
    return "\n".join(json.dumps(s, ensure_ascii=False, sort_keys=True) for s in samples) + "\n"


# --------------------------------------------------------------------------
# A. Determinism
# --------------------------------------------------------------------------


def test_determinism_byte_identical():
    """Same input twice must produce byte-identical output."""

    if not _have_dataset():
        print("  [A] determinism SKIPPED (no component dataset)")
        return
    first = _serialise(_generate(limit=15))
    second = _serialise(_generate(limit=15))
    assert first == second, "generation is not byte-identical across runs"

    # The on-disk v0.1 artefact was produced with the SUPERSEDED v0.1 semantic
    # policy, so a fresh run no longer reproduces it. The equivalent check for
    # the current policy lives in tests/test_v011_acceptance.py, which compares a
    # fresh v0.1.1 subset run against the on-disk v0.1.1 artefact.
    print("  [A] determinism byte-identical OK")


def test_sample_ids_deterministic():
    if not _have_dataset():
        return
    a = [s["sample_id"] for s in _generate(limit=10)]
    b = [s["sample_id"] for s in _generate(limit=10)]
    assert a == b
    assert len(set(a)) == len(a), "duplicate sample ids"
    for sid in a:
        assert sid.startswith("buildsr_train_")
    print("  [A2] sample ids deterministic and unique OK")


# --------------------------------------------------------------------------
# B. Target recomputation from reasoning_steps
# --------------------------------------------------------------------------


def test_target_recomputable_from_steps():
    """Re-executing reasoning_steps must reproduce target_component_id."""

    if not _have_dataset():
        print("  [B] target recomputation SKIPPED")
        return

    checked = 0
    for record in G.iter_metadata(DATASET_ROOT, "train"):
        image = G.image_geometry_from_record(record)
        samples = A.generate_for_image(
            image, record, CONFIG, _gen_config(), _dataset_meta(), A.DiscardCounter(), set()
        )
        for sample in samples:
            recomputed = A.recompute_target_from_steps(image, sample["reasoning_steps"], CONFIG)
            assert recomputed == sample["target_component_id"], (
                f"{sample['sample_id']}: steps recompute to {recomputed}, "
                f"recorded {sample['target_component_id']}"
            )
            # Independently re-derive the relation filter set and compare with the
            # stored list.
            for step in sample["reasoning_steps"]:
                if step["operation"] != "filter_relation":
                    continue
                derived = A.filter_relation_candidates(image, step, CONFIG)
                assert derived == sorted(step["candidate_component_ids"]), (
                    f"{sample['sample_id']}: stored relation candidates "
                    f"{sorted(step['candidate_component_ids'])} != derived {derived}"
                )
                # The nearest step must reason over the FULL direction-valid set
                # (v0.1.1 semantic policy), so a closer but ineligible candidate
                # is never skipped.
                nearest_eligible = step.get("nearest_eligible_component_ids")
                if nearest_eligible is not None:
                    assert set(nearest_eligible) <= set(derived), (
                        f"{sample['sample_id']}: nearest-eligible {nearest_eligible} is not a "
                        f"subset of the relation candidates {derived}"
                    )
                    nearest_step = next(
                        s for s in sample["reasoning_steps"]
                        if s["operation"] == "argmin_boundary_distance"
                    )
                    assert sorted(nearest_step["candidate_component_ids"]) == sorted(derived), (
                        f"{sample['sample_id']}: nearest step must cover the full direction set "
                        f"{derived}, got {sorted(nearest_step['candidate_component_ids'])}"
                    )
            checked += 1
        if checked >= 120:
            break
    assert checked > 0, "no samples checked"
    print(f"  [B] target recomputation OK on {checked} samples")


# --------------------------------------------------------------------------
# C. No hidden tie-break
# --------------------------------------------------------------------------


def test_no_hidden_tiebreak_for_multi_candidate_direction():
    """A Level-2 direction query with >1 candidate must be discarded, not narrowed.

    Only Level-2 Type-B queries are in scope here: a 2-step program whose second
    step is a single ``filter_relation``. Level-3 chains legitimately carry
    multiple candidates, because they then apply ``argmin_boundary_distance``,
    which is a criterion stated in the instruction.
    """

    if not _have_dataset():
        print("  [C] no-tiebreak SKIPPED")
        return

    from collections import defaultdict

    multi_seen = 0
    checked_images = 0
    for record in G.iter_metadata(DATASET_ROOT, "train"):
        image = G.image_geometry_from_record(record)
        quality = classify_image(image, CONFIG)
        relations = R.evaluate_image(image, CONFIG, compute_nearest=False)

        grouped = defaultdict(list)
        for result in relations.directional:
            # relation(subject, object): the candidate is the SUBJECT.
            if result.valid:
                grouped[(result.object, result.relation)].append(result.subject)
        multi = {k: v for k, v in grouped.items() if len(v) > 1}
        checked_images += 1
        if multi:
            multi_seen += 1

        samples = A.generate_for_image(
            image, record, CONFIG, _gen_config(), _dataset_meta(), A.DiscardCounter(), set()
        )
        for sample in samples:
            if sample["level"] != 2:
                continue
            steps = sample["reasoning_steps"]
            # Level-2 Type B shape: exactly two steps, the second a filter.
            if len(steps) != 2 or steps[1]["operation"] != "filter_relation":
                continue
            step = steps[1]
            key = (step["reference_component_id"], step["relation"])
            if key in multi:
                raise AssertionError(
                    f"{sample['sample_id']}: emitted a Level-2 Type-B query for {key} which "
                    f"has {len(multi[key])} candidates; a hidden tie-break was applied"
                )
        if multi_seen >= 25:
            break
    assert multi_seen > 0, "no multi-candidate direction case was encountered"
    print(
        f"  [C] no hidden tie-break OK ({multi_seen} multi-candidate cases inspected "
        f"across {checked_images} images)"
    )


def test_level3_may_keep_multiple_candidates():
    """Level-3 chains must NOT be discarded just because the filter kept several.

    Discarding them would be over-strict: the instruction states the
    nearest-among-filtered criterion, so the answer is determined.
    """

    if not _have_dataset():
        return
    multi_candidate_chains = 0
    for record in G.iter_metadata(DATASET_ROOT, "train"):
        image = G.image_geometry_from_record(record)
        samples = A.generate_for_image(
            image, record, CONFIG, _gen_config(), _dataset_meta(), A.DiscardCounter(), set()
        )
        for sample in samples:
            if sample["level"] != 3:
                continue
            step = next(s for s in sample["reasoning_steps"] if s["operation"] == "filter_relation")
            if len(step["candidate_component_ids"]) > 1:
                # The instruction must state the nearest criterion for this to be
                # a determined question. Wording varies by template, so accept any
                # of the phrasings the template families actually use.
                zh_ok = any(word in sample["instruction_zh"] for word in ("最近", "最小"))
                en_ok = any(
                    word in sample["instruction_en"].lower()
                    for word in ("nearest", "closest", "smallest boundary")
                )
                assert zh_ok, sample["sample_id"]
                assert en_ok, f"{sample['sample_id']}: {sample['instruction_en']}"
                multi_candidate_chains += 1
        if multi_candidate_chains >= 15:
            break
    print(f"  [C3] Level-3 keeps multi-candidate chains with a stated criterion OK ({multi_candidate_chains})")


def test_level2_direction_samples_have_single_candidate():
    """Every emitted Type-B sample must have exactly one filtered candidate."""

    if not _have_dataset():
        return
    checked = 0
    for record in G.iter_metadata(DATASET_ROOT, "train"):
        image = G.image_geometry_from_record(record)
        samples = A.generate_for_image(
            image, record, CONFIG, _gen_config(), _dataset_meta(), A.DiscardCounter(), set()
        )
        for sample in samples:
            steps = sample["reasoning_steps"]
            filters = [s for s in steps if s["operation"] == "filter_relation"]
            if not filters or len(steps) != 2:
                continue
            # Level-2 Type B: single filter step, single candidate.
            assert len(filters[0]["candidate_component_ids"]) == 1, sample["sample_id"]
            assert sample["target_component_id"] == filters[0]["candidate_component_ids"][0]
            checked += 1
        if checked >= 60:
            break
    print(f"  [C2] Type-B samples have exactly one candidate OK ({checked} checked)")


# --------------------------------------------------------------------------
# D. Conditional nearest
# --------------------------------------------------------------------------


def test_nearest_within_only_sees_filtered_set():
    """nearest_within must ignore candidates outside the supplied set."""

    # Component 2 is globally nearest to 1; component 3 is farther.
    from test_relations import make_image  # noqa: E402

    cm, image = make_image(
        {1: (40, 40, 100, 100), 2: (110, 40, 140, 100), 3: (250, 40, 300, 100)},
        width=400,
        height=400,
    )
    quality = classify_image(image, CONFIG)

    global_nearest = R.evaluate_nearest(image, 1, CONFIG, quality, cm)
    assert global_nearest.valid and global_nearest.object == 2

    # Restricted to {3}: nearest_within must return 3, NOT the global nearest 2.
    within = R.nearest_within(image, 1, [3], CONFIG, quality, cm)
    assert within.valid
    assert within.target == 3, f"nearest_within leaked outside the filter: {within.target}"
    assert within.trivial_selection, "a single candidate must be flagged trivial"

    # Restricted to {2, 3}: the true nearest inside the set is 2.
    within2 = R.nearest_within(image, 1, [2, 3], CONFIG, quality, cm)
    assert within2.valid and within2.target == 2
    assert not within2.trivial_selection

    # Empty set -> invalid, never a silent answer.
    empty = R.nearest_within(image, 1, [], CONFIG, quality, cm)
    assert not empty.valid
    assert empty.reason == R.REASON_TOO_FEW_CANDIDATES
    print("  [D] nearest_within respects the filtered set OK")


def test_nearest_within_margin_policy():
    """The frozen nearest margin policy applies inside the filtered set too."""

    from test_relations import make_image  # noqa: E402

    # Two candidates at almost identical distance from the anchor.
    cm, image = make_image(
        {1: (120, 120, 150, 150), 2: (154, 120, 184, 150), 3: (120, 154, 150, 184)},
        width=400,
        height=400,
    )
    quality = classify_image(image, CONFIG)
    within = R.nearest_within(image, 1, [2, 3], CONFIG, quality, cm)
    assert not within.valid
    assert within.ambiguous, "near-equal distances must be ambiguous, not tie-broken"
    print("  [D2] nearest_within margin policy OK")


def test_nearest_within_requires_non_border_anchor():
    from test_relations import make_image  # noqa: E402

    cm, image = make_image(
        {1: (0, 200, 40, 240), 2: (100, 200, 160, 260), 3: (250, 200, 300, 250)},
        width=400,
        height=400,
    )
    quality = classify_image(image, CONFIG)
    assert quality.flags(1).touches_image_border
    within = R.nearest_within(image, 1, [2, 3], CONFIG, quality, cm)
    assert not within.valid, "border anchor must be rejected by nearest_within too"
    assert within.reason == R.REASON_ANCHOR_NOT_ELIGIBLE
    print("  [D3] nearest_within rejects border anchor OK")


# --------------------------------------------------------------------------
# E. Predicate semantics
# --------------------------------------------------------------------------


def test_predicate_semantics_in_filter_step():
    """"A is left of B" must correspond to left_of(A, B)."""

    from test_relations import make_image  # noqa: E402

    cm, image = make_image(
        {1: (40, 180, 90, 230), 2: (200, 180, 260, 230)},
        width=400,
        height=400,
    )
    quality = classify_image(image, CONFIG)
    a, b = image.get(1), image.get(2)   # a is visually LEFT of b

    # The predicate holds as left_of(subject=a, object=b).
    assert R.evaluate_direction("left_of", image, a, b, CONFIG, quality).valid
    # And it does not hold in reverse.
    assert not R.evaluate_direction("left_of", image, b, a, CONFIG, quality).valid

    # The rendered sentence names subject then object.
    assert R.describe_relation("left_of", "A", "B") == "A is left of B"

    # A filter step with subject=a, reference=b must keep a.
    steps = [
        {"step": 1, "operation": "argmin_centroid_x", "output_component_id": 1},
        {
            "step": 2,
            "operation": "filter_relation",
            "relation": "left_of",
            "reference_component_id": 2,
            "input_component_ids": [1, 2],
            "candidate_component_ids": [1],
        },
    ]
    kept = A.recompute_target_from_steps(image, steps, CONFIG, quality, cm)
    assert kept == 1, f"left_of(a, b) must keep a=1, got {kept}"
    print("  [E] predicate semantics in filter step OK")


# --------------------------------------------------------------------------
# F. Split preservation
# --------------------------------------------------------------------------


def test_split_preservation_no_leakage():
    """No query may cross a source-image split boundary.

    The authoritative split membership comes from the Task 2 metadata
    (`datasets/whu/metadata/<split>.jsonl`), one record per source image.
    """

    if not _have_dataset():
        print("  [F] split preservation SKIPPED")
        return

    split_ids = {
        split: {record["image_id"] for record in G.iter_metadata(DATASET_ROOT, split)}
        for split in SPLITS
    }
    for split in SPLITS:
        assert split_ids[split], f"{split} has no images"

    # Hard disjointness check.
    assert not (split_ids["train"] & split_ids["val"]), list(split_ids["train"] & split_ids["val"])[:5]
    assert not (split_ids["train"] & split_ids["test"]), list(split_ids["train"] & split_ids["test"])[:5]
    assert not (split_ids["val"] & split_ids["test"]), list(split_ids["val"] & split_ids["test"])[:5]

    # Every generated record must claim its own split and belong to that split.
    for split in SPLITS:
        path = GENERATED_DIR / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert record["split"] == split, record["sample_id"]
            assert record["image_id"] in split_ids[split], (
                f"{record['sample_id']}: image {record['image_id']} not in {split}"
            )
    print(
        "  [F] split preservation / no leakage OK "
        f"(train={len(split_ids['train'])}, val={len(split_ids['val'])}, test={len(split_ids['test'])})"
    )


# --------------------------------------------------------------------------
# G. Duplicate prevention
# --------------------------------------------------------------------------


def test_duplicate_prevention():
    """Image + program + target must appear at most once."""

    if not _have_dataset():
        print("  [G] duplicate prevention SKIPPED")
        return
    keys = []
    for record in G.iter_metadata(DATASET_ROOT, "train"):
        image = G.image_geometry_from_record(record)
        samples = A.generate_for_image(
            image, record, CONFIG, _gen_config(), _dataset_meta(), A.DiscardCounter(), set()
        )
        for sample in samples:
            keys.append(A.semantic_key(sample["image_id"], sample["reasoning_steps"], sample["target_component_id"]))
    assert len(keys) == len(set(keys)), "duplicate canonical semantic keys"

    # The on-disk artefact must also be unique per split.
    path = GENERATED_DIR / "train.jsonl"
    if path.is_file():
        ids = [json.loads(l)["sample_id"] for l in path.read_text(encoding="utf-8").splitlines()]
        assert len(ids) == len(set(ids)), "duplicate sample_id on disk"
    print(f"  [G] duplicate prevention OK ({len(keys)} unique keys)")


def test_template_variation_does_not_create_duplicates():
    """Different wording must NOT duplicate a sample: one canonical record only."""

    if not _have_dataset():
        return
    samples = _generate(limit=8)
    keys = [(s["image_id"], json.dumps(A._canonical_steps(s["reasoning_steps"]), sort_keys=True), s["target_component_id"])
            for s in samples]
    assert len(keys) == len(set(keys)), "same program duplicated under different wording"
    print("  [G2] template variation does not duplicate OK")


# --------------------------------------------------------------------------
# H. Language parity
# --------------------------------------------------------------------------


def test_language_parity_from_same_template_id():
    """zh and en must come from the same template pair."""

    families = [
        list(TP.LEVEL1_TEMPLATES.values()),
        list(TP.LEVEL2_NEAREST_TEMPLATES.values()),
        list(TP.LEVEL2_DIRECTION_TEMPLATES.values()),
        [TP.LEVEL3_TEMPLATES],
    ]
    checked = 0
    for group in families:
        for family in group:
            for pair in family:
                assert pair.zh.strip(), pair.template_id
                assert pair.en.strip(), pair.template_id
                checked += 1

    # Every template id must be unique, so a zh string maps to exactly one en.
    all_ids = [
        pair.template_id
        for group in families
        for family in group
        for pair in family
    ]
    assert len(all_ids) == len(set(all_ids)), "duplicate template ids break parity"
    print(f"  [H] language parity: {checked} template pairs, unique ids OK")


def test_bilingual_instructions_emitted_together():
    if not _have_dataset():
        return
    for sample in _generate(limit=8):
        assert sample["instruction_zh"] and sample["instruction_en"]
        assert sample["reasoning_zh"] and sample["reasoning_en"]
        # No architecture-specific token may appear in any language field.
        for field_name in ("instruction_zh", "instruction_en", "reasoning_zh", "reasoning_en"):
            text = sample[field_name]
            for token in ("[SEG]", "[REF]"):
                assert token not in text, f"{token} leaked into {field_name}"
    print("  [H2] bilingual instructions present, no architecture tokens OK")


def test_family_sizes_meet_minimum():
    """Every query type must offer at least 3 templates per language."""

    for query_type, family in TP.LEVEL1_TEMPLATES.items():
        assert len(family) >= 3, query_type
    for reference, family in TP.LEVEL2_NEAREST_TEMPLATES.items():
        assert len(family) >= 3, reference
    for relation, family in TP.LEVEL2_DIRECTION_TEMPLATES.items():
        assert len(family) >= 3, relation
    assert len(TP.LEVEL3_TEMPLATES) >= 3
    print("  [H3] template families all >= 3 OK")


# --------------------------------------------------------------------------
# I. Mask selector
# --------------------------------------------------------------------------


def test_mask_selector_matches_target():
    if not _have_dataset():
        print("  [I] mask selector SKIPPED")
        return
    for sample in _generate(limit=10):
        assert sample["target_mask"]["representation"] == "component_map_selector"
        assert sample["target_mask"]["component_id"] == sample["target_component_id"], sample["sample_id"]
    for split in SPLITS:
        path = GENERATED_DIR / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines()[:200]:
            record = json.loads(line)
            assert record["target_mask"]["component_id"] == record["target_component_id"]
    print("  [I] mask selector matches target OK")


def test_schema_fields_present():
    """Required schema keys must exist and geometry must NOT be duplicated."""

    required = {
        "sample_id", "project", "dataset_name", "dataset_version", "source_dataset",
        "source_subset", "image_id", "split", "image_path", "component_map_path",
        "image_metadata_ref", "level", "query_type", "instruction_zh", "instruction_en",
        "reference_component_ids", "target_component_id", "candidate_component_ids",
        "distractor_component_ids", "reasoning_steps", "reasoning_zh", "reasoning_en",
        "target_mask", "relation_config_version", "generator_version",
    }
    if not _have_dataset():
        return
    for sample in _generate(limit=6):
        missing = required - set(sample)
        assert not missing, f"missing keys: {missing}"
        # Geometry must not be duplicated into the sample.
        for banned in ("polygon_normalized", "bbox_xyxy_px", "centroid_px", "area_px"):
            assert banned not in sample, f"geometry field {banned} duplicated into sample"
        assert sample["source_dataset"] == "WHU Building Dataset"
        assert sample["dataset_name"] == "BuildSpatialReason"
    print("  [I2] schema complete and geometry not duplicated OK")


# --------------------------------------------------------------------------
# Level-3 trivial / nontrivial separation
# --------------------------------------------------------------------------


def test_level3_trivial_flag_matches_candidate_count():
    """trivial_selection == "exactly one ADMISSIBLE candidate after filtering".

    v0.1.1 changed the criterion from the raw direction-set size to the
    admissible-candidate count, so a chain whose direction set has several
    members but only one admissible candidate is correctly marked trivial.
    """

    if not _have_dataset():
        return
    import semantic_policy as SP

    checked = trivial = 0
    for split in ("train",):
        for image_record in list(G.iter_metadata(DATASET_ROOT, split))[:25]:
            image = G.image_geometry_from_record(image_record)
            samples = A.generate_for_image(
                image, image_record, CONFIG, _gen_config(), _dataset_meta(),
                A.DiscardCounter(), set()
            )
            for sample in samples:
                if sample["level"] != 3:
                    continue
                steps = sample["reasoning_steps"]
                nearest_step = next(s for s in steps if s["operation"] == "argmin_boundary_distance")
                admissible = SP.admissible_nearest_ids(
                    image,
                    nearest_step["reference_component_id"],
                    nearest_step["candidate_component_ids"],
                    image.load_map(DATASET_ROOT),
                    CONFIG,
                )
                if len(admissible) == 1:
                    assert sample["trivial_selection"], sample["sample_id"]
                    trivial += 1
                else:
                    assert not sample["trivial_selection"], (
                        f"{sample['sample_id']}: admissible={len(admissible)} but flagged trivial"
                    )
                checked += 1
    assert checked > 0, "no Level-3 samples generated"
    print(f"  [extra] Level-3 trivial flag consistent OK ({checked} samples, {trivial} trivial)")


def test_level3_nontrivial_actually_needs_distance():
    """A nontrivial Level-3 sample must have >=2 candidates to compare."""

    if not _have_dataset():
        return
    found = 0
    for sample in _generate(limit=40):
        if sample["level"] != 3 or sample["trivial_selection"]:
            continue
        nearest_step = next(s for s in sample["reasoning_steps"] if s["operation"] == "argmin_boundary_distance")
        assert len(nearest_step["candidate_component_ids"]) >= 2
        found += 1
    print(f"  [extra] nontrivial Level-3 requires >=2 candidates OK ({found} found)")


# --------------------------------------------------------------------------
# Thresholds must not be overridden
# --------------------------------------------------------------------------


def test_generator_uses_frozen_thresholds():
    gen_config = _gen_config()
    frozen = gen_config["relations"]["frozen"]
    assert abs(CONFIG.size_rank.ratio_margin - frozen["ratio_margin"]) < 1e-9
    assert CONFIG.direction.active_candidate == frozen["direction_preset"]
    assert CONFIG.eligibility_for("nearest").reject_touches_image_border_anchor is True
    assert CONFIG.eligibility_for("nearest").reject_touches_image_border is True
    print("  [extra] generator honours frozen thresholds OK")


def test_no_architecture_tokens_anywhere_in_artifact():
    """No [SEG] / [REF] may appear anywhere in the generated dataset."""

    for split in SPLITS:
        path = GENERATED_DIR / f"{split}.jsonl"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in ("[SEG]", "[REF]"):
            assert token not in text, f"{token} found in {path.name}"
    print("  [extra] no architecture tokens in artifact OK")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("A determinism", test_determinism_byte_identical),
        ("A2 sample ids", test_sample_ids_deterministic),
        ("B target recompute", test_target_recomputable_from_steps),
        ("C no tiebreak", test_no_hidden_tiebreak_for_multi_candidate_direction),
        ("C2 type-B single cand", test_level2_direction_samples_have_single_candidate),
        ("C3 level3 multi cand", test_level3_may_keep_multiple_candidates),
        ("D nearest_within", test_nearest_within_only_sees_filtered_set),
        ("D2 nearest margin", test_nearest_within_margin_policy),
        ("D3 nearest border", test_nearest_within_requires_non_border_anchor),
        ("E predicate", test_predicate_semantics_in_filter_step),
        ("F split", test_split_preservation_no_leakage),
        ("G duplicates", test_duplicate_prevention),
        ("G2 template dup", test_template_variation_does_not_create_duplicates),
        ("H parity", test_language_parity_from_same_template_id),
        ("H2 bilingual", test_bilingual_instructions_emitted_together),
        ("H3 family size", test_family_sizes_meet_minimum),
        ("I mask selector", test_mask_selector_matches_target),
        ("I2 schema", test_schema_fields_present),
        ("L3 trivial flag", test_level3_trivial_flag_matches_candidate_count),
        ("L3 nontrivial", test_level3_nontrivial_actually_needs_distance),
        ("frozen thresholds", test_generator_uses_frozen_thresholds),
        ("no arch tokens", test_no_architecture_tokens_anywhere_in_artifact),
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
    print(f"{len(tests) - failures}/{len(tests)} annotator checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
