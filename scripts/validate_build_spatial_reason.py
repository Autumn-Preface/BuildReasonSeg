#!/usr/bin/env python
"""CLI: validate the BuildSpatialReason-v0.1 dataset.

    python scripts/validate_build_spatial_reason.py [--splits train val test]
                                                    [--limit N] [--no-samples] [--quiet]

Writes:
    evaluation/build_spatial_reason_v0.1_quality.json
    docs/build_spatial_reason_v0.1_quality_audit.md
    evaluation/build_spatial_reason_v0.1_samples/*.png  (+ contact_sheet.png)

This script never modifies the dataset it audits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import dataset_validator as V  # noqa: E402
import geometry as G  # noqa: E402
import relations as R  # noqa: E402
import semantic_oracle as SO  # noqa: E402
import thresholds as T  # noqa: E402

DATASET_ROOT = _REPO_ROOT / "datasets" / "whu"
GEN_DIR = _REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1"
EVAL_DIR = _REPO_ROOT / "evaluation"
DOCS_DIR = _REPO_ROOT / "docs"
SAMPLES_DIR = EVAL_DIR / "build_spatial_reason_v0.1_samples"
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
# Per-record audits
# --------------------------------------------------------------------------


def audit_record(
    record: dict,
    ctx: V.ImageContext,
    issues: V.IssueCollector,
    counters: dict,
    enforce_v011: bool = True,
) -> None:
    """Run every per-sample integrity and semantic check."""

    sample_id = record["sample_id"]
    image_id = record["image_id"]
    level = record["level"]
    query_type = record["query_type"]
    steps = record["reasoning_steps"]
    target = record["target_component_id"]
    references = record["reference_component_ids"]

    # ---- mask selector -------------------------------------------------
    mask = record.get("target_mask") or {}
    if mask.get("component_id") != target:
        issues.add("target_mask_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                   mask_component_id=mask.get("component_id"), target=target)
    if mask.get("representation") != "component_map_selector":
        issues.add("mask_representation_unexpected", V.SEVERITY_ERROR, sample_id, image_id,
                   representation=mask.get("representation"))

    # ---- target exists and yields a non-empty mask ---------------------
    mask_pixels = int((ctx.component_map == target).sum())
    if ctx.component(target) is None:
        issues.add("target_missing_component", V.SEVERITY_ERROR, sample_id, image_id, target=target)
    elif mask_pixels == 0:
        issues.add("target_mask_empty", V.SEVERITY_ERROR, sample_id, image_id, target=target)
    counters["mask_checked"] += 1

    # ---- target recomputation ------------------------------------------
    recomputed = V.recompute_target_independent(ctx, level, query_type, steps)
    if recomputed != target:
        issues.add("target_recompute_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                   recorded=target, recomputed=recomputed, query_type=query_type, level=level)
    counters["recompute_checked"] += 1
    counters["recompute_pass"] += int(recomputed == target)

    # ---- structured program integrity ----------------------------------
    for index, step in enumerate(steps, start=1):
        if step.get("step") != index:
            issues.add("step_numbering", V.SEVERITY_ERROR, sample_id, image_id,
                       expected=index, found=step.get("step"))
        for key in ("reference_component_id", "output_component_id"):
            value = step.get(key)
            if value is not None and ctx.component(int(value)) is None:
                issues.add("step_references_missing_component", V.SEVERITY_ERROR, sample_id, image_id,
                           step=index, key=key, value=value)
        for key in ("input_component_ids", "candidate_component_ids", "nearest_eligible_component_ids"):
            values = step.get(key)
            if values is None:
                continue
            if len(values) != len(set(values)):
                issues.add("duplicate_ids_in_step", V.SEVERITY_ERROR, sample_id, image_id,
                           step=index, key=key)
            for value in values:
                if ctx.component(int(value)) is None:
                    issues.add("step_references_missing_component", V.SEVERITY_ERROR, sample_id, image_id,
                               step=index, key=key, value=value)

    expected_ops = {1: 1, 2: 2, 3: 3}[level]
    if len(steps) != expected_ops:
        issues.add("program_length_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                   level=level, expected=expected_ops, found=len(steps))

    # ---- reference integrity -------------------------------------------
    if level == 1:
        if references:
            issues.add("level1_has_reference", V.SEVERITY_ERROR, sample_id, image_id, references=references)
    else:
        if not references:
            issues.add("missing_reference", V.SEVERITY_ERROR, sample_id, image_id)
        else:
            reference_id = references[0]
            if ctx.component(reference_id) is None:
                issues.add("reference_missing", V.SEVERITY_ERROR, sample_id, image_id, reference=reference_id)
            if reference_id == target:
                issues.add("reference_equals_target", V.SEVERITY_ERROR, sample_id, image_id,
                           reference=reference_id)
            step1_output = steps[0].get("output_component_id") if steps else None
            if step1_output != reference_id:
                issues.add("reference_disagrees_with_step1", V.SEVERITY_ERROR, sample_id, image_id,
                           reference=reference_id, step1_output=step1_output)

    # ---- candidate / distractor integrity -------------------------------
    distractors = record["distractor_component_ids"]
    if target in distractors:
        issues.add("target_in_distractors", V.SEVERITY_ERROR, sample_id, image_id, target=target)
    if len(distractors) != len(set(distractors)):
        issues.add("duplicate_distractors", V.SEVERITY_ERROR, sample_id, image_id)
    for value in distractors:
        if ctx.component(value) is None:
            issues.add("distractor_missing_component", V.SEVERITY_ERROR, sample_id, image_id, value=value)
    # Task 5B section 9: distractors must exclude every explicit reference too.
    for reference_id in references:
        if reference_id in distractors:
            issues.add("reference_in_distractors", V.SEVERITY_ERROR, sample_id, image_id,
                       reference=reference_id)

    # ---- Level-2 Type B: uniqueness ------------------------------------
    if level == 2 and query_type.count("_to_") == 1 and not query_type.endswith("_to_nearest"):
        relation = query_type.rsplit("_to_", 1)[1]
        reference_id = references[0] if references else None
        if reference_id is not None:
            derived = ctx.direction_candidates(reference_id, relation)
            stored = record["candidate_component_ids"]
            if len(derived) != 1:
                issues.add("typeB_not_unique", V.SEVERITY_ERROR, sample_id, image_id,
                           relation=relation, derived=derived)
            if derived != sorted(stored):
                issues.add("typeB_candidate_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                           derived=derived, stored=sorted(stored))

    # ---- Level-2A candidate recomputation -------------------------------
    if level == 2 and query_type.endswith("_to_nearest"):
        stored = sorted(record["candidate_component_ids"])
        all_others = sorted(c.component_id for c in ctx.image.components if c.component_id != references[0])
        if stored != all_others:
            issues.add("typeA_candidate_mismatch", V.SEVERITY_WARNING, sample_id, image_id,
                       derived_count=len(all_others), stored_count=len(stored))

    # ---- Level-3 candidate sets ----------------------------------------
    if level == 3:
        filter_step = next((s for s in steps if s["operation"] == "filter_relation"), None)
        nearest_step = next((s for s in steps if s["operation"] == "argmin_boundary_distance"), None)
        if filter_step is None or nearest_step is None:
            issues.add("level3_missing_operation", V.SEVERITY_ERROR, sample_id, image_id)
        else:
            relation = filter_step["relation"]
            reference_id = filter_step["reference_component_id"]
            derived_direction = V.SP.direction_candidates_over_visible(
                ctx.image, reference_id, relation, ctx.config, ctx.quality
            )
            stored_direction = sorted(filter_step.get("candidate_component_ids", []))
            if derived_direction != stored_direction:
                issues.add("level3_direction_candidates_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                           derived=derived_direction, stored=stored_direction)

            stored_eligible = sorted(filter_step.get("nearest_eligible_component_ids", []))
            derived_eligible = ctx.nearest_eligible(reference_id, derived_direction)
            if stored_eligible != derived_eligible:
                issues.add("level3_nearest_eligible_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                           derived=derived_eligible, stored=stored_eligible)
            if not set(stored_eligible) <= set(stored_direction):
                issues.add("level3_eligible_not_subset", V.SEVERITY_ERROR, sample_id, image_id,
                           eligible=stored_eligible, direction=stored_direction)
            # The nearest step must reason over the FULL direction-valid set (so a
            # closer but ineligible component is never skipped), not over the
            # eligible subset.
            if sorted(nearest_step.get("candidate_component_ids", [])) != stored_direction:
                issues.add("level3_nearest_step_candidates_mismatch", V.SEVERITY_ERROR,
                           sample_id, image_id,
                           expected=stored_direction,
                           found=sorted(nearest_step.get("candidate_component_ids", [])))

            # trivial_selection must equal "exactly one admissible candidate
            # after the direction filter and the frozen margin rule".
            admissible = V.SP.admissible_nearest_ids(
                ctx.image, reference_id, derived_direction, ctx.component_map,
                ctx.config, ctx.quality,
            )
            expected_trivial = len(admissible) == 1
            if bool(record.get("trivial_selection")) != expected_trivial:
                issues.add("trivial_flag_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                           recorded=record.get("trivial_selection"), expected=expected_trivial,
                           n_admissible=len(admissible), n_direction=len(derived_direction))

    # ---- component id leakage in natural language -----------------------
    # Task 5B section 11: three distinct counters, not one ambiguous field.
    #
    #   leak_reasoning_records -> records with >=1 leak in any reasoning field
    #   leak_reasoning_fields  -> (record, field) pairs that leak
    #   leak_reasoning_mentions-> total id mentions
    record_leaks = 0
    for field_name, kind in (
        ("instruction_zh", "instruction"),
        ("instruction_en", "instruction"),
        ("reasoning_zh", "reasoning"),
        ("reasoning_en", "reasoning"),
    ):
        leaks = V.find_component_id_leaks(record.get(field_name, ""))
        if not leaks:
            continue
        counters[f"leak_{kind}_fields"] += 1
        counters[f"leak_{kind}_mentions"] += len(leaks)
        if kind == "reasoning":
            record_leaks += 1
        issues.add(f"component_id_leak_{kind}", V.SEVERITY_WARNING, sample_id, image_id,
                   field=field_name, leaks=leaks[:3])
    if record_leaks:
        counters["leak_reasoning_records"] += 1
    if any(
        V.find_component_id_leaks(record.get(f, ""))
        for f in ("instruction_zh", "instruction_en")
    ):
        counters["leak_instruction_records"] += 1

    # ---- template_id integrity (v0.1.1 profile) --------------------------
    stored_template_id = record.get("template_id")
    if not stored_template_id and not enforce_v011:
        # Historical v0.1 had no template_id field; skip rather than flag it.
        counters["template_checked"] += 1
        counters["template_id_absent_legacy"] += 1
        return
    if not stored_template_id:
        issues.add("template_id_missing", V.SEVERITY_ERROR, sample_id, image_id)
    else:
        reconstructed = V.reconstruct_template_id(
            query_type, record["instruction_zh"], record["instruction_en"]
        )
        if reconstructed is None:
            issues.add("template_not_reconstructed", V.SEVERITY_ERROR, sample_id, image_id,
                       query_type=query_type, stored_template_id=stored_template_id)
        elif reconstructed != stored_template_id:
            issues.add("template_id_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                       stored=stored_template_id, reconstructed=reconstructed)
        else:
            counters["template_verified"] += 1
            counters.setdefault("template_usage", Counter())[stored_template_id] += 1
    counters["template_checked"] += 1


def audit_semantics(
    record: dict,
    ctx: V.ImageContext,
    issues: V.IssueCollector,
    counters: dict,
) -> bool:
    """Hidden-eligibility semantic audit (stricter than engine eligibility).

    Returns True when this record carries at least one semantic violation, so the
    caller can maintain the unique-record union as a machine-readable counter
    rather than deriving it only in prose.
    """

    sample_id = record["sample_id"]
    image_id = record["image_id"]
    level = record["level"]
    query_type = record["query_type"]
    target = record["target_component_id"]
    references = record["reference_component_ids"]
    violated = False

    # ---- largest / smallest ---------------------------------------------
    if "largest" in query_type:
        reference_id = references[0] if references else (target if level == 1 else None)
        if level == 1:
            reference_id = target
        violation = V.audit_hidden_largest(ctx, reference_id)
        if violation:
            counters["hidden_largest"] += 1
            violated = True
            issues.add("hidden_eligibility_largest", V.SEVERITY_ERROR, sample_id, image_id, **violation)
        else:
            counters["hidden_largest_ok"] += 1

    if "smallest" in query_type:
        reference_id = target if level == 1 else (references[0] if references else None)
        violation = V.audit_hidden_smallest(ctx, reference_id)
        if violation:
            counters["hidden_smallest"] += 1
            violated = True
            issues.add("hidden_eligibility_smallest", V.SEVERITY_ERROR, sample_id, image_id, **violation)
        else:
            counters["hidden_smallest_ok"] += 1

    # ---- nearest ---------------------------------------------------------
    if level == 2 and query_type.endswith("_to_nearest"):
        anchor = references[0]
        violation = V.audit_hidden_nearest(ctx, anchor, target)
        if violation:
            counters["hidden_nearest"] += 1
            violated = True
            issues.add("hidden_eligibility_nearest", V.SEVERITY_ERROR, sample_id, image_id, **violation)
        else:
            counters["hidden_nearest_ok"] += 1

    # ---- Level-3 nearest-within -----------------------------------------
    if level == 3:
        filter_step = next((s for s in record["reasoning_steps"] if s["operation"] == "filter_relation"), None)
        if filter_step is not None:
            violation = V.audit_hidden_level3_nearest(
                ctx, filter_step["reference_component_id"], filter_step["relation"], target
            )
            if violation:
                counters["hidden_level3_nearest"] += 1
                violated = True
                issues.add("hidden_eligibility_level3_nearest", V.SEVERITY_ERROR,
                           sample_id, image_id, **violation)
            else:
                counters["hidden_level3_nearest_ok"] += 1

    return violated


def audit_with_oracle(
    record: dict,
    so_ctx: SO.OracleImage,
    issues: V.IssueCollector,
    counters: dict,
) -> bool:
    """Independent-oracle acceptance audit (Task 5C section 5).

    Recomputes the whole semantic program — reference, direction candidate set,
    semantic nearest, final target, ambiguity/admissibility — with an
    implementation that never imports ``semantic_policy``, and compares every
    field with the stored record.

    Returns True when at least one oracle mismatch was found.
    """

    sample_id = record["sample_id"]
    image_id = record["image_id"]
    comparison = SO.compare_record(so_ctx, record)

    counters["oracle_checked"] += 1
    counters["oracle_target_match"] += int(comparison.target_match)
    counters["oracle_reference_match"] += int(comparison.reference_match)
    counters["oracle_candidate_set_match"] += int(comparison.candidate_set_match)
    if comparison.trivial_match is not None:
        counters["oracle_trivial_checked"] += 1
        counters["oracle_trivial_match"] += int(comparison.trivial_match)

    mismatched = False
    detail = {
        "level": comparison.level,
        "query_type": comparison.query_type,
        "oracle_target": comparison.oracle_target,
        "stored_target": comparison.stored_target,
        "oracle_reference": comparison.oracle_reference,
        "stored_references": list(comparison.stored_references),
        "oracle_candidates": list(comparison.oracle_candidates),
        "stored_candidates": list(comparison.stored_candidates),
    }

    if not comparison.admissible:
        mismatched = True
        counters["oracle_ambiguity_mismatch"] += 1
        issues.add("oracle_ambiguity_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                   reason=comparison.oracle_reason, **detail)
    if not comparison.target_match:
        mismatched = True
        counters["oracle_target_mismatch"] += 1
        issues.add("oracle_target_mismatch", V.SEVERITY_ERROR, sample_id, image_id, **detail)
    if not comparison.reference_match:
        mismatched = True
        counters["oracle_reference_mismatch"] += 1
        issues.add("oracle_reference_mismatch", V.SEVERITY_ERROR, sample_id, image_id, **detail)
    if not comparison.candidate_set_match:
        mismatched = True
        counters["oracle_candidate_set_mismatch"] += 1
        issues.add("oracle_candidate_set_mismatch", V.SEVERITY_ERROR, sample_id, image_id, **detail)
    if comparison.trivial_match is False:
        mismatched = True
        counters["oracle_trivial_flag_mismatch"] += 1
        issues.add("oracle_trivial_flag_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                   stored_trivial=comparison.stored_trivial,
                   oracle_admissible_nearest=len(comparison.oracle_admissible_nearest_ids),
                   **detail)

    # ---- independent hidden-eligibility audit (section 6) ---------------
    hidden = SO.oracle_hidden_violations(so_ctx, record)
    if hidden:
        mismatched = True
        counters["oracle_semantic_violation_unique_records"] += 1
        for code in hidden:
            counters[f"oracle_{code}"] += 1
            issues.add(f"oracle_{code}", V.SEVERITY_ERROR, sample_id, image_id, **detail)
    else:
        counters["oracle_semantic_clean_records"] += 1

    return mismatched


# --------------------------------------------------------------------------
# Main audit
# --------------------------------------------------------------------------


def run_audit(splits: Sequence[str], limit: int | None, quiet: bool,
              version: str = "v0.1", enforce_v011: bool | None = None) -> dict:
    """Audit one dataset version.

    ``enforce_v011`` selects the rule set:
      * ``True``  -> v0.1.1 acceptance profile (template_id required, semantic
        candidate derivation, distractor policy strict).
      * ``False`` -> historical v0.1 profile (no template_id field existed).
      * ``None``  -> inferred: enabled for versions other than ``v0.1``.
    """

    if enforce_v011 is None:
        enforce_v011 = version != "v0.1"

    config = T.load_config()
    issues = V.IssueCollector()
    counters: dict = defaultdict(int)
    counters["template_usage"] = Counter()
    counters["audited_version"] = version

    level_counts = Counter()
    query_type_counts = Counter()
    split_counts = Counter()
    split_level = defaultdict(Counter)
    split_query_type = defaultdict(Counter)
    per_image = Counter()
    trivial = nontrivial = 0
    target_areas = []
    target_centroids = []
    sample_ids = Counter()
    semantic_keys = Counter()

    record_by_split: dict[str, list[dict]] = {}
    started = time.time()

    for split in splits:
        records = V.read_dataset_records(_REPO_ROOT / "datasets", split, version)
        if limit is not None:
            records = records[:limit]
        record_by_split[split] = records
        if not quiet:
            print(f"  [{split}] {len(records)} records", flush=True)

        metadata_by_image = {
            rec["image_id"]: rec for rec in V.iteration_metadata(DATASET_ROOT, split)
        }
        cache: dict[str, V.ImageContext] = {}
        oracle_cache: dict[str, SO.OracleImage] = {}
        counters["component_maps_loaded"] = counters.get("component_maps_loaded", 0)

        for index, record in enumerate(records, start=1):
            image_id = record["image_id"]
            if image_id not in cache:
                meta = metadata_by_image.get(image_id)
                if meta is None:
                    issues.add("image_not_in_metadata", V.SEVERITY_ERROR, record["sample_id"], image_id)
                    continue
                image = G.image_geometry_from_record(meta)
                cache[image_id] = V.ImageContext(image, config)
                counters["component_maps_loaded"] += 1

            ctx = cache[image_id]

            # Independent-oracle context: shares the already-loaded component
            # map (read-only) but performs its own semantic reasoning.
            so_ctx = oracle_cache.get(image_id)
            if so_ctx is None:
                so_ctx = SO.OracleImage(ctx.image, config, ctx.component_map)
                oracle_cache[image_id] = so_ctx

            split_counts[split] += 1
            level_counts[record["level"]] += 1
            query_type_counts[record["query_type"]] += 1
            split_level[split][record["level"]] += 1
            split_query_type[split][record["query_type"]] += 1
            per_image[(split, image_id)] += 1
            sample_ids[record["sample_id"]] += 1
            semantic_keys[(image_id, json.dumps(record["reasoning_steps"], sort_keys=True),
                           record["target_component_id"])] += 1

            if record["level"] == 3:
                if record.get("trivial_selection"):
                    trivial += 1
                else:
                    nontrivial += 1

            component = ctx.component(record["target_component_id"])
            if component is not None:
                target_areas.append(component.area_px)
                target_centroids.append(
                    (component.centroid_x / ctx.image.width, component.centroid_y / ctx.image.height)
                )

            audit_record(record, ctx, issues, counters, enforce_v011)
            if audit_semantics(record, ctx, issues, counters):
                counters["semantic_violation_unique_records"] += 1

            # ---- independent oracle (Task 5C) --------------------------
            audit_with_oracle(record, so_ctx, issues, counters)

            if not quiet and index % 4000 == 0:
                print(f"    {split} {index}/{len(records)} ({time.time() - started:.0f}s)", flush=True)

    # ---- duplicate detection -------------------------------------------
    for sid, count in sample_ids.items():
        if count > 1:
            issues.add("duplicate_sample_id", V.SEVERITY_ERROR, sid, detail={"count": count})
    for key, count in semantic_keys.items():
        if count > 1:
            issues.add("duplicate_semantic_key", V.SEVERITY_ERROR, detail={"key": list(key[:1]), "count": count})

    # ---- language parity (zh/en from the same template) ------------------
    if counters.get("template_checked"):
        parity_ok = counters.get("template_verified", 0)
        if enforce_v011 and parity_ok != counters["template_checked"]:
            issues.add("language_parity_incomplete", V.SEVERITY_ERROR,
                       detail={"verified": parity_ok, "checked": counters["template_checked"]})

    # ---- image hash leakage --------------------------------------------
    duplicate_report = V.exact_cross_split_image_duplicates(
        DATASET_ROOT, [s for s in splits], record_by_split
    )
    if duplicate_report["exact_image_duplicate_cross_split"] > 0:
        issues.add("exact_image_duplicate_cross_split", V.SEVERITY_ERROR,
                   detail=duplicate_report["duplicates"][:3])

    # ---- v0.1 must remain byte-identical --------------------------------
    frozen_report = V.verify_v01_unchanged(_REPO_ROOT / "datasets")

    if enforce_v011 and frozen_report.get("unchanged") is False:
        issues.add("v01_modified", V.SEVERITY_ERROR, detail=frozen_report.get("mismatches"))

    # ---- v0.1.1 JSONL must remain byte-identical (Task 5C section 11) ----
    frozen_v011 = V.verify_v011_unchanged(_REPO_ROOT / "datasets")
    if enforce_v011 and frozen_v011.get("unchanged") is False:
        issues.add("v011_modified", V.SEVERITY_ERROR, detail=frozen_v011.get("mismatches"))

    # ---- independent counts --------------------------------------------
    total = sum(split_counts.values())
    independent_counts = {
        "total_records": total,
        "by_split": {k: int(v) for k, v in sorted(split_counts.items())},
        "by_level": {str(k): int(v) for k, v in sorted(level_counts.items())},
        "by_query_type": {k: int(v) for k, v in sorted(query_type_counts.items())},
        "by_split_level": {s: {str(k): int(v) for k, v in sorted(c.items())}
                           for s, c in sorted(split_level.items())},
        "by_split_query_type": {s: {k: int(v) for k, v in sorted(c.items())}
                                for s, c in sorted(split_query_type.items())},
        "level2": {
            "reference_to_nearest": int(sum(v for k, v in query_type_counts.items() if k.endswith("_to_nearest") and k.count("_to_") == 1)),
            "reference_to_direction": int(sum(v for k, v in query_type_counts.items()
                                              if k.count("_to_") == 1 and not k.endswith("_to_nearest"))),
            "total_level2": int(level_counts.get(2, 0)),
        },
        "level3": {"trivial": trivial, "nontrivial": nontrivial, "total": trivial + nontrivial},
        "images_with_samples": len(per_image),
        "samples_per_image": V.quantiles(list(per_image.values())),
    }

    runtime = time.time() - started

    # ---- required machine-readable audit fields (Task 5B sections 11, 20) ----
    leak_records = int(counters.get("leak_reasoning_records", 0))
    leak_fields = int(counters.get("leak_reasoning_fields", 0))
    leak_mentions = int(counters.get("leak_reasoning_mentions", 0))

    semantic_flag_codes = (
        "hidden_eligibility_largest",
        "hidden_eligibility_smallest",
        "hidden_eligibility_nearest",
        "hidden_eligibility_level3_nearest",
    )
    semantic_violations = sum(int(issues.counts.get(code, 0)) for code in semantic_flag_codes)
    semantic_unique = int(counters.get("semantic_violation_unique_records", 0))

    total_records = int(independent_counts["total_records"])
    counters["template_checked"] = int(counters.get("template_checked", 0))
    counters["template_verified"] = int(counters.get("template_verified", 0))

    report = {
        "audit_target": f"BuildSpatialReason-{version}",
        "audited_version": version,
        "auditor": "dataset_validator.py",
        "runtime_seconds": round(runtime, 2),
        "splits_audited": list(splits),
        "limit_applied": limit,
        "verdict": "PENDING_CONSISTENCY_GATE",
        "independent_counts": independent_counts,
        "target_recomputation": {
            "checked": int(counters.get("recompute_checked", 0)),
            "pass": int(counters.get("recompute_pass", 0)),
            "fail": int(counters.get("recompute_checked", 0)) - int(counters.get("recompute_pass", 0)),
        },
        "hidden_semantic_counts": {
            "hidden_eligibility_largest": int(issues.counts.get("hidden_eligibility_largest", 0)),
            "hidden_eligibility_smallest": int(issues.counts.get("hidden_eligibility_smallest", 0)),
            "hidden_eligibility_nearest": int(issues.counts.get("hidden_eligibility_nearest", 0)),
            "hidden_eligibility_level3_nearest": int(
                issues.counts.get("hidden_eligibility_level3_nearest", 0)
            ),
            "semantic_violation_flag_instances": semantic_violations,
        },
        "semantic_violation_unique_records": semantic_unique,
        "semantic_clean_records": total_records - semantic_unique,
        "reasoning_leakage": {
            "leak_reasoning_records": leak_records,
            "leak_reasoning_fields": leak_fields,
            "leak_reasoning_mentions": leak_mentions,
            "leak_instruction_records": int(counters.get("leak_instruction_records", 0)),
        },
        "distractor_integrity": {
            "target_in_distractors": int(issues.counts.get("target_in_distractors", 0)),
            "reference_in_distractors": int(issues.counts.get("reference_in_distractors", 0)),
            "duplicate_distractors": int(issues.counts.get("duplicate_distractors", 0)),
            "distractor_missing_component": int(issues.counts.get("distractor_missing_component", 0)),
        },
        "template_id_integrity": {
            "checked": int(counters.get("template_checked", 0)),
            "verified": int(counters.get("template_verified", 0)),
            "mismatch": int(issues.counts.get("template_id_mismatch", 0)),
            "missing": int(issues.counts.get("template_id_missing", 0)),
            "not_reconstructed": int(issues.counts.get("template_not_reconstructed", 0)),
        },
        "mask_selector_integrity": {
            "checked": int(counters.get("mask_checked", 0)),
            "mismatch": int(issues.counts.get("target_mask_mismatch", 0)),
            "empty": int(issues.counts.get("target_mask_empty", 0)),
        },
        "exact_cross_split_duplicates": int(
            duplicate_report["exact_image_duplicate_cross_split"]
        ),
        "counters": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in counters.items()},
        "issues": issues.as_dict(),
        "duplicate_image_audit": duplicate_report,
        "v01_frozen_integrity": frozen_report,
        "v011_frozen_integrity": frozen_v011,
        "independent_oracle": {
            "oracle_version": SO.ORACLE_VERSION,
            "oracle_module": "spatial_reasoning/semantic_oracle.py",
            "imports_semantic_policy": False,
            "checked": int(counters.get("oracle_checked", 0)),
            "target_match": int(counters.get("oracle_target_match", 0)),
            "target_mismatch": int(issues.counts.get("oracle_target_mismatch", 0)),
            "candidate_set_mismatch": int(issues.counts.get("oracle_candidate_set_mismatch", 0)),
            "reference_mismatch": int(issues.counts.get("oracle_reference_mismatch", 0)),
            "ambiguity_policy_mismatch": int(issues.counts.get("oracle_ambiguity_mismatch", 0)),
            "trivial_flag_checked": int(counters.get("oracle_trivial_checked", 0)),
            "trivial_flag_match": int(counters.get("oracle_trivial_match", 0)),
            "trivial_flag_mismatch": int(issues.counts.get("oracle_trivial_flag_mismatch", 0)),
            "semantic_violation_unique_records": int(
                counters.get("oracle_semantic_violation_unique_records", 0)
            ),
            "semantic_clean_records": int(counters.get("oracle_semantic_clean_records", 0)),
            "hidden_semantic_counts": {
                "hidden_eligibility_largest": int(
                    issues.counts.get("oracle_hidden_eligibility_largest", 0)
                ),
                "hidden_eligibility_smallest": int(
                    issues.counts.get("oracle_hidden_eligibility_smallest", 0)
                ),
                "hidden_eligibility_nearest": int(
                    issues.counts.get("oracle_hidden_eligibility_nearest", 0)
                ),
                "hidden_eligibility_level3_nearest": int(
                    issues.counts.get("oracle_hidden_eligibility_level3_nearest", 0)
                ),
                "semantic_violation_flag_instances": sum(
                    int(issues.counts.get(code, 0))
                    for code in V.ORACLE_HIDDEN_CODES
                ),
            },
        },
        "artifact_consistency": {"status": "not_run", "violations": []},
        "scene_level_split_leakage": "unverified",
        "scene_level_split_leakage_reason": (
            "The source WHU tiles carry no scene/geographic grouping metadata in the derived "
            "YOLO dataset, and the val split was a random 20% subset of the original train pool. "
            "Scene-level disjointness therefore cannot be established from available metadata."
        ),
        "target_distribution": {
            "area_px": V.quantiles(target_areas),
            "centroid_x_norm": V.quantiles([c[0] for c in target_centroids]),
            "centroid_y_norm": V.quantiles([c[1] for c in target_centroids]),
        },
    }

    # ---- artifact consistency gate (Task 5C section 7) -------------------
    #
    # Run BEFORE the verdict is fixed, and fed this in-memory report, so a
    # repository-level inconsistency (stale counts in docs, a competing quality
    # JSON, a missing/mismatched provenance hash) is a blocking acceptance
    # failure rather than a footnote. The gate's own frozen-data check re-reads
    # the JSONL from disk independently of the checks above.
    report["artifact_consistency"] = run_consistency_gate(version, report)
    if report["artifact_consistency"].get("status") != "consistent":
        issues.add(
            "artifact_inconsistent",
            V.SEVERITY_ERROR,
            detail={
                "status": report["artifact_consistency"].get("status"),
                "violations": report["artifact_consistency"].get("violations", [])[:5],
            },
        )

    report["verdict"] = V.decide_verdict(issues)
    report["issues"] = issues.as_dict()
    report["counters"] = {
        k: (dict(v) if isinstance(v, Counter) else v) for k, v in counters.items()
    }
    return report


def run_consistency_gate(version: str, report: dict) -> dict:
    """Run the repository consistency gate for the version just audited."""

    try:
        import check_artifact_consistency as C
    except Exception as exc:  # noqa: BLE001
        return {"status": "not_run", "error": f"{type(exc).__name__}: {exc}", "violations": []}

    try:
        result = C.run_consistency_check(quality_report=report)
    except Exception as exc:  # noqa: BLE001
        return {"status": "not_run", "error": f"{type(exc).__name__}: {exc}", "violations": []}

    # The gate audits the repository state for the indexed dataset version; if
    # that is not the version just audited, say so instead of claiming agreement.
    if result.get("audited_version") not in (None, version):
        result["status"] = "version_mismatch"
    return result


def decide_verdict(issues: V.IssueCollector) -> str:
    """Deprecated shim: the authoritative implementation lives in the validator."""

    return V.decide_verdict(issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--version", default="v0.1.1",
                        help="dataset version under datasets/build_spatial_reason/")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-samples", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    version = args.version
    version_dir = _REPO_ROOT / "datasets" / "build_spatial_reason" / version
    if not (version_dir / "train.jsonl").is_file():
        print(f"error: dataset not found at {version_dir}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"auditing {version_dir}")

    report = run_audit(args.splits, args.limit, args.quiet, version)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    # Canonical artifact naming (Task 5C section 8): the file name carries the
    # real dataset version, so v0.1.1 writes build_spatial_reason_v0.1.1_*.json
    # rather than the historical "v011" abbreviation that docs never used.
    json_path = EVAL_DIR / f"build_spatial_reason_{version}_quality.json"
    samples_dir = EVAL_DIR / f"build_spatial_reason_{version}_samples"

    samples: list[str] = []
    if not args.no_samples:
        try:
            from build_spatial_reason_samples import build_sample_pack

            samples = build_sample_pack(samples_dir, report, args.limit, version)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: sample pack generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.quiet:
        oracle = report["independent_oracle"]
        print(f"verdict : {report['verdict']}")
        print(f"issues  : {report['issues']['counts']}")
        print(f"counts  : {report['independent_counts']['by_level']}")
        print(f"oracle  : {oracle['target_match']}/{oracle['checked']} exact target matches, "
              f"{oracle['semantic_violation_unique_records']} semantic violations")
        print(f"consist.: {report['artifact_consistency'].get('status')}")
        print(f"wrote   : {json_path}")
        print(f"samples : {len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
