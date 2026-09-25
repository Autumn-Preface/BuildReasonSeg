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

import dataset_validator as V  # noqa: E402
import geometry as G  # noqa: E402
import relations as R  # noqa: E402
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
) -> None:
    """Run every per-sample integrity and semantic check."""

    sample_id = record["sample_id"]
    image_id = record["image_id"]
    level = record["level"]
    query_type = record["query_type"]
    steps = record["reasoning_steps"]
    target = record["target_component_id"]

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
    references = record["reference_component_ids"]
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
    if references and references[0] in distractors:
        issues.add("reference_in_distractors", V.SEVERITY_WARNING, sample_id, image_id,
                   reference=references[0])

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
            derived_direction = ctx.direction_candidates(reference_id, relation)
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
            if sorted(nearest_step.get("candidate_component_ids", [])) != stored_eligible:
                issues.add("level3_nearest_step_candidates_mismatch", V.SEVERITY_ERROR, sample_id, image_id)

            # trivial flag must follow the eligible candidate count
            expected_trivial = len(derived_eligible) == 1
            if bool(record.get("trivial_selection")) != expected_trivial:
                issues.add("trivial_flag_mismatch", V.SEVERITY_ERROR, sample_id, image_id,
                           recorded=record.get("trivial_selection"), expected=expected_trivial,
                           n_eligible=len(derived_eligible))

    # ---- component id leakage in natural language -----------------------
    for field_name, kind in (
        ("instruction_zh", "instruction"),
        ("instruction_en", "instruction"),
        ("reasoning_zh", "reasoning"),
        ("reasoning_en", "reasoning"),
    ):
        leaks = V.find_component_id_leaks(record.get(field_name, ""))
        if leaks:
            counters[f"leak_{kind}_samples"] += 1
            counters[f"leak_{kind}_mentions"] += len(leaks)
            issues.add(f"component_id_leak_{kind}", V.SEVERITY_WARNING, sample_id, image_id,
                       field=field_name, leaks=leaks[:3])

    # ---- language hygiene -----------------------------------------------
    for field_name in ("instruction_zh", "instruction_en", "reasoning_zh", "reasoning_en"):
        text = record.get(field_name, "")
        if not text or not text.strip():
            issues.add("empty_language_field", V.SEVERITY_ERROR, sample_id, image_id, field=field_name)
        if "{" in text or "}" in text:
            issues.add("raw_placeholder", V.SEVERITY_ERROR, sample_id, image_id, field=field_name)
        if "None" in text or "null" in text:
            issues.add("literal_none", V.SEVERITY_ERROR, sample_id, image_id, field=field_name)
        if "。。" in text or ".." in text or "，，" in text:
            issues.add("doubled_punctuation", V.SEVERITY_WARNING, sample_id, image_id, field=field_name)

    # ---- template reconstruction ----------------------------------------
    template_id = V.reconstruct_template_id(query_type, record["instruction_zh"], record["instruction_en"])
    if template_id is None:
        issues.add("template_not_reconstructed", V.SEVERITY_ERROR, sample_id, image_id,
                   query_type=query_type)
    else:
        counters["template_matched"] += 1
        counters.setdefault("template_usage", Counter())[template_id] += 1
    counters["template_checked"] += 1


def audit_semantics(
    record: dict,
    ctx: V.ImageContext,
    issues: V.IssueCollector,
    counters: dict,
) -> None:
    """Hidden-eligibility semantic audit (stricter than engine eligibility)."""

    sample_id = record["sample_id"]
    image_id = record["image_id"]
    level = record["level"]
    query_type = record["query_type"]
    target = record["target_component_id"]
    references = record["reference_component_ids"]

    # ---- largest / smallest ---------------------------------------------
    if "largest" in query_type:
        reference_id = references[0] if references else (target if level == 1 else None)
        if level == 1:
            reference_id = target
        violation = V.audit_hidden_largest(ctx, reference_id)
        if violation:
            counters["hidden_largest"] += 1
            issues.add("hidden_eligibility_largest", V.SEVERITY_ERROR, sample_id, image_id, **violation)
        else:
            counters["hidden_largest_ok"] += 1

    if "smallest" in query_type:
        reference_id = target if level == 1 else (references[0] if references else None)
        violation = V.audit_hidden_smallest(ctx, reference_id)
        if violation:
            counters["hidden_smallest"] += 1
            issues.add("hidden_eligibility_smallest", V.SEVERITY_ERROR, sample_id, image_id, **violation)
        else:
            counters["hidden_smallest_ok"] += 1

    # ---- nearest ---------------------------------------------------------
    if level == 2 and query_type.endswith("_to_nearest"):
        anchor = references[0]
        violation = V.audit_hidden_nearest(ctx, anchor, target)
        if violation:
            counters["hidden_nearest"] += 1
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
                issues.add("hidden_eligibility_level3_nearest", V.SEVERITY_ERROR,
                           sample_id, image_id, **violation)
            else:
                counters["hidden_level3_nearest_ok"] += 1


# --------------------------------------------------------------------------
# Main audit
# --------------------------------------------------------------------------


def run_audit(splits: Sequence[str], limit: int | None, quiet: bool) -> dict:
    config = T.load_config()
    issues = V.IssueCollector()
    counters: dict = defaultdict(int)
    counters["template_usage"] = Counter()

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
        records = V.read_dataset_records(_REPO_ROOT / "datasets", split)
        if limit is not None:
            records = records[:limit]
        record_by_split[split] = records
        if not quiet:
            print(f"  [{split}] {len(records)} records", flush=True)

        metadata_by_image = {
            rec["image_id"]: rec for rec in V.iteration_metadata(DATASET_ROOT, split)
        }
        cache: dict[str, V.ImageContext] = {}
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

            audit_record(record, ctx, issues, counters)
            audit_semantics(record, ctx, issues, counters)

            if not quiet and index % 4000 == 0:
                print(f"    {split} {index}/{len(records)} ({time.time() - started:.0f}s)", flush=True)

    # ---- duplicate detection -------------------------------------------
    for sid, count in sample_ids.items():
        if count > 1:
            issues.add("duplicate_sample_id", V.SEVERITY_ERROR, sid, detail={"count": count})
    for key, count in semantic_keys.items():
        if count > 1:
            issues.add("duplicate_semantic_key", V.SEVERITY_ERROR, detail={"key": list(key[:1]), "count": count})

    # ---- image hash leakage --------------------------------------------
    duplicate_report = V.exact_cross_split_image_duplicates(
        DATASET_ROOT, [s for s in splits], record_by_split
    )
    if duplicate_report["exact_image_duplicate_cross_split"] > 0:
        issues.add("exact_image_duplicate_cross_split", V.SEVERITY_ERROR,
                   detail=duplicate_report["duplicates"][:3])

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
    verdict = V.decide_verdict(issues)

    return {
        "audit_target": "BuildSpatialReason-v0.1",
        "auditor": "dataset_validator.py",
        "runtime_seconds": round(runtime, 2),
        "splits_audited": list(splits),
        "limit_applied": limit,
        "verdict": verdict,
        "independent_counts": independent_counts,
        "counters": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in counters.items()},
        "issues": issues.as_dict(),
        "duplicate_image_audit": duplicate_report,
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


def decide_verdict(issues: V.IssueCollector) -> str:
    """Deprecated shim: the authoritative implementation lives in the validator."""

    return V.decide_verdict(issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-samples", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not (GEN_DIR / "train.jsonl").is_file():
        print(f"error: dataset not found at {GEN_DIR}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"auditing {GEN_DIR}")

    report = run_audit(args.splits, args.limit, args.quiet)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DIR / "build_spatial_reason_v0.1_quality.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    samples: list[str] = []
    if not args.no_samples:
        try:
            from build_spatial_reason_samples import build_sample_pack

            samples = build_sample_pack(SAMPLES_DIR, report, args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: sample pack generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not args.quiet:
        print(f"verdict : {report['verdict']}")
        print(f"issues  : {report['issues']['counts']}")
        print(f"counts  : {report['independent_counts']['by_level']}")
        print(f"wrote   : {json_path}")
        print(f"samples : {len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
