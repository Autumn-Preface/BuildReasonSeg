#!/usr/bin/env python
"""Dataset-wide spatial relation diagnostics.

    python scripts/analyze_spatial_relations.py [--limit N] [--no-samples] [--quiet]

Scans the whole WHU component dataset and writes:

    evaluation/relation_statistics_v1.json     machine-readable statistics
    docs/relation_statistics_v1.md             readable report
    evaluation/relation_samples/*.png          a small manual-review sample

This script does NOT generate instructions and does NOT use a model. It only
measures how much reliable spatial relation ground truth the dataset can
actually support.
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

import geometry as G  # noqa: E402
import relations as R  # noqa: E402
import thresholds as T  # noqa: E402
from component_quality import ALL_FLAGS, classify_image  # noqa: E402

DATASET_ROOT = _REPO_ROOT / "datasets" / "whu"
EVAL_DIR = _REPO_ROOT / "evaluation"
DOCS_DIR = _REPO_ROOT / "docs"
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------


def _repo_relative(path_like) -> str:
    """Return a repository-relative POSIX path, never an absolute one.

    Reports must survive a repository rename, so no absolute path is recorded.
    """

    resolved = Path(path_like).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        # Outside the repository: record the basename rather than an absolute path.
        return resolved.name


def quantiles(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p1": float(np.percentile(array, 1)),
        "p5": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


# --------------------------------------------------------------------------
# Level-2 chain feasibility (statistics only -- no instruction is generated)
# --------------------------------------------------------------------------


def _pair_holds(image_relations: R.ImageRelations, relation: str, subject: int, target: int) -> bool:
    """True when ``relation(subject, target)`` is a valid evaluated relation."""

    for result in image_relations.directional:
        if result.valid and result.relation == relation and result.subject == subject and result.object == target:
            return True
    return False


def chain_feasibility(image_relations: R.ImageRelations) -> dict[str, bool]:
    """Which Level-2 relation chains are uniquely determined for this image."""

    out: dict[str, bool] = {}

    largest = image_relations.size_rank.get("largest")
    smallest = image_relations.size_rank.get("smallest")
    leftmost = image_relations.extremes.get("leftmost")
    rightmost = image_relations.extremes.get("rightmost")
    topmost = image_relations.extremes.get("topmost")
    bottommost = image_relations.extremes.get("bottommost")

    # A: a named extreme anchor -> valid directional relation to its nearest neighbour
    for name, extreme in (
        ("leftmost", leftmost),
        ("rightmost", rightmost),
        ("topmost", topmost),
        ("bottommost", bottommost),
    ):
        if extreme is None or not extreme.valid:
            out[f"{name}->nearest"] = False
            out[f"{name}->any_direction"] = False
            continue
        anchor = extreme.subject
        nearest = image_relations.nearest.get(anchor)
        out[f"{name}->nearest"] = bool(nearest is not None and nearest.valid)
        out[f"{name}->any_direction"] = any(
            r.valid and r.subject == anchor for r in image_relations.directional
        )

    # B: largest/smallest -> directional -> nearest
    for name, size in (("largest", largest), ("smallest", smallest)):
        if size is None or not size.valid:
            out[f"{name}->direction->nearest"] = False
            continue
        anchor = size.subject
        found = False
        for result in image_relations.directional:
            if result.valid and result.subject == anchor:
                inner = image_relations.nearest.get(result.object)
                if inner is not None and inner.valid:
                    found = True
                    break
        out[f"{name}->direction->nearest"] = found

    # C: largest -> right_of specifically (the canonical example)
    if largest is not None and largest.valid:
        anchor = largest.subject
        out["largest->right_of"] = _pair_holds(image_relations, "right_of", anchor, next(
            (r.object for r in image_relations.directional
             if r.valid and r.relation == "right_of" and r.subject == anchor), -1))
        out["largest->right_of->nearest"] = any(
            r.valid and r.relation == "right_of" and r.subject == anchor
            and image_relations.nearest.get(r.object) is not None
            and image_relations.nearest[r.object].valid
            for r in image_relations.directional
        )
    else:
        out["largest->right_of"] = False
        out["largest->right_of->nearest"] = False

    # D: topmost -> below
    if topmost is not None and topmost.valid:
        anchor = topmost.subject
        out["topmost->below"] = any(
            r.valid and r.relation == "below" and r.subject == anchor
            for r in image_relations.directional
        )
    else:
        out["topmost->below"] = False

    return out


def level1_counts(image_relations: R.ImageRelations) -> dict[str, int]:
    """How many reliable Level-1 style queries this image can support.

    Counts, per category, the number of valid unique-answer extreme and
    size-rank relations available.
    """

    counts = {
        "extreme": sum(1 for r in image_relations.extremes.values() if r.valid),
        "size_rank": sum(1 for r in image_relations.size_rank.values() if r.valid),
        "nearest": sum(1 for r in image_relations.nearest.values() if r.valid),
    }
    counts["total"] = counts["extreme"] + counts["size_rank"] + counts["nearest"]
    return counts


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------


def analyze(limit: int | None, quiet: bool) -> dict:
    config = T.load_config()

    component_stats = defaultdict(list)
    flag_counts = Counter()
    border_by_relation = defaultdict(Counter)
    pair_stats = defaultdict(list)

    relation_counts = defaultdict(lambda: {"valid": 0, "ambiguous": 0, "discarded": 0})
    relation_counts_by_split = defaultdict(lambda: defaultdict(lambda: {"valid": 0, "ambiguous": 0, "discarded": 0}))
    discard_reasons = Counter()
    ambiguous_by_relation = Counter()

    per_image_components = []
    level1_hist = Counter()
    chain_counter = Counter()
    chain_by_split = defaultdict(Counter)
    images_processed = 0
    images_by_split = Counter()
    scope_counts = Counter()

    started = time.time()

    for split in SPLITS:
        for record in G.iter_metadata(DATASET_ROOT, split):
            if limit is not None and images_by_split[split] >= limit:
                continue
            image = G.image_geometry_from_record(record)
            images_by_split[split] += 1
            images_processed += 1
            per_image_components.append(len(image.components))

            quality = classify_image(image, config)
            for component in image.components:
                flags = quality.flags(component.component_id)
                component_stats["area_px"].append(component.area_px)
                component_stats["area_ratio"].append(component.area_ratio)
                component_stats["width_px"].append(component.width_px)
                component_stats["height_px"].append(component.height_px)
                component_stats["aspect_ratio"].append(
                    component.aspect_ratio if np.isfinite(component.aspect_ratio) else 1e9
                )
                component_stats["bbox_extent_ratio"].append(flags.bbox_extent_ratio)
                component_stats["fill_ratio"].append(component.fill_ratio)
                component_stats["centroid_x_norm"].append(component.centroid_x / image.width)
                component_stats["centroid_y_norm"].append(component.centroid_y / image.height)
                component_stats["equivalent_scale_px"].append(component.equivalent_scale_px)
                for flag in ALL_FLAGS:
                    if flags.as_dict()[flag]:
                        flag_counts[flag] += 1
                flag_counts["total_components"] += 1
                if flags.touches_image_border:
                    flag_counts["border_components"] += 1

            # pairwise geometry
            components = image.components
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    a, b = components[i], components[j]
                    dx, dy = G.subject_to_object_delta(a, b)
                    pair_stats["abs_dx_norm"].append(abs(dx) / image.width)
                    pair_stats["abs_dy_norm"].append(abs(dy) / image.height)
                    pair_stats["centroid_distance_px"].append(G.centroid_distance(a, b))
                    pair_stats["bbox_gap_px"].append(G.bbox_gap(a, b))
                    pair_stats["area_ratio"].append(
                        max(a.area_px, b.area_px) / max(min(a.area_px, b.area_px), 1)
                    )
                    pair_stats["centroid_separation_norm"].append(
                        G.centroid_distance(a, b) / image.diagonal
                    )

            # relations
            image_relations = R.evaluate_image(image, config)

            for result in image_relations.all_results():
                bucket = relation_counts[result.relation]
                split_bucket = relation_counts_by_split[split][result.relation]
                if result.valid:
                    bucket["valid"] += 1
                    split_bucket["valid"] += 1
                elif result.ambiguous:
                    bucket["ambiguous"] += 1
                    split_bucket["ambiguous"] += 1
                    ambiguous_by_relation[result.relation] += 1
                else:
                    bucket["discarded"] += 1
                    split_bucket["discarded"] += 1
                    discard_reasons[result.reason or "unknown"] += 1

            # border component eligibility, per relation
            for relation in T.CORE_RELATIONS:
                eligible = set(quality.eligible_ids(relation, config))
                for component in image.components:
                    flags = quality.flags(component.component_id)
                    if component.component_id in eligible:
                        border_by_relation[relation]["eligible"] += 1
                        if flags.touches_image_border:
                            border_by_relation[relation]["eligible_border"] += 1
                    else:
                        border_by_relation[relation]["rejected"] += 1
                        if flags.touches_image_border:
                            border_by_relation[relation]["rejected_border"] += 1

            counts = level1_counts(image_relations)
            bucket = counts["total"]
            if bucket >= 4:
                level1_hist["4+"] += 1
            else:
                level1_hist[str(bucket)] += 1

            chains = chain_feasibility(image_relations)
            for name, ok in chains.items():
                if ok:
                    chain_counter[name] += 1
                    chain_by_split[split][name] += 1

            scope_counts[f"images_with_1_component"] += 1 if len(image.components) == 1 else 0

            if not quiet and images_processed % 500 == 0:
                print(f"  ...{images_processed} images ({time.time() - started:.0f}s)", flush=True)

    elapsed = time.time() - started

    report = {
        "version": config.version,
        "scope": {"kind": config.scope_kind, "note": config.scope_note},
        # Repository-relative so a repository rename cannot invalidate the report.
        "config_source": _repo_relative(config.source_path),
        "images_processed": images_processed,
        "images_by_split": dict(images_by_split),
        "elapsed_seconds": round(elapsed, 2),
        "active_thresholds": {
            "direction": {
                "candidate": config.direction.active_candidate,
                "alpha": config.direction.alpha,
                "tau": config.direction.tau,
                "candidates": config.direction.candidates,
            },
            "extreme_margin_px": config.extreme_margin_px,
            "nearest": {
                "distance_metric": config.nearest.distance_metric,
                "margin_px_floor": config.nearest.margin_px_floor,
                "margin_diag_fraction": config.nearest.margin_diag_fraction,
                "margin_mode": config.nearest.margin_mode,
            },
            "size_rank_ratio_margin": config.size_rank.ratio_margin,
            "quality": {
                "merge_bbox_extent_ratio": config.quality.merge_bbox_extent_ratio,
                "tiny_area_px": config.quality.tiny_area_px,
            },
        },
        "component_statistics": {k: quantiles(v) for k, v in component_stats.items()},
        "pairwise_statistics": {k: quantiles(v) for k, v in pair_stats.items()},
        "components_per_image": quantiles(per_image_components),
        "component_quality": dict(flag_counts),
        "relation_counts": {rel: dict(v) for rel, v in sorted(relation_counts.items())},
        "relation_counts_by_split": {
            split: {rel: dict(v) for rel, v in sorted(relation_counts_by_split[split].items())}
            for split in SPLITS
        },
        "ambiguity_by_relation": dict(sorted(ambiguous_by_relation.items())),
        "discard_reasons": dict(sorted(discard_reasons.items())),
        "border_eligibility_by_relation": {
            rel: dict(counter) for rel, counter in sorted(border_by_relation.items())
        },
        "level1_feasibility": dict(sorted(level1_hist.items())),
        "level2_feasibility": dict(sorted(chain_counter.items())),
        "level2_feasibility_by_split": {
            split: dict(sorted(chain_by_split[split].items())) for split in SPLITS
        },
    }
    return report


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    thresholds = report["active_thresholds"]
    out: list[str] = []
    out.append("# Spatial Relation Statistics (v1)")
    out.append("")
    out.append(
        "Generated by `scripts/analyze_spatial_relations.py`. Measured over the full "
        "WHU component dataset; no instructions were generated and no model was used."
    )
    out.append("")
    split_desc = ", ".join(f"{k}={v}" for k, v in report["images_by_split"].items())
    out.append(f"- images processed: **{report['images_processed']}** ({split_desc})")
    out.append(f"- runtime: {report['elapsed_seconds']} s")
    out.append(f"- config: `{Path(report['config_source']).name}`")
    out.append(f"- scope: **{report['scope']['kind']}**")
    out.append("")
    out.append("## Active thresholds")
    out.append("")
    out.append(
        _table(
            ["setting", "value"],
            [
                ["direction candidate", f"`{thresholds['direction']['candidate']}`"],
                ["direction alpha", str(thresholds["direction"]["alpha"])],
                ["direction tau", str(thresholds["direction"]["tau"])],
                ["extreme margin (px)", str(thresholds["extreme_margin_px"])],
                ["nearest distance metric", f"`{thresholds['nearest']['distance_metric']}`"],
                ["nearest margin floor (px)", str(thresholds["nearest"]["margin_px_floor"])],
                ["nearest margin (x diagonal)", str(thresholds["nearest"]["margin_diag_fraction"])],
                ["size rank ratio margin", str(thresholds["size_rank_ratio_margin"])],
                ["merge bbox extent ratio", str(thresholds["quality"]["merge_bbox_extent_ratio"])],
                ["tiny area (px^2)", str(thresholds["quality"]["tiny_area_px"])],
            ],
        )
    )
    out.append("")

    out.append("## Component statistics")
    out.append("")
    rows = []
    for key, stats in report["component_statistics"].items():
        if not stats.get("n"):
            continue
        rows.append([
            f"`{key}`",
            str(stats["n"]),
            f"{stats['min']:.4g}",
            f"{stats['p5']:.4g}",
            f"{stats['median']:.4g}",
            f"{stats['p90']:.4g}",
            f"{stats['p95']:.4g}",
            f"{stats['p99']:.4g}",
            f"{stats['max']:.4g}",
        ])
    out.append(_table(["metric", "n", "min", "p5", "median", "p90", "p95", "p99", "max"], rows))
    out.append("")

    out.append("## Pairwise statistics")
    out.append("")
    rows = []
    for key, stats in report["pairwise_statistics"].items():
        if not stats.get("n"):
            continue
        rows.append([
            f"`{key}`",
            str(stats["n"]),
            f"{stats['min']:.4g}",
            f"{stats['p5']:.4g}",
            f"{stats['median']:.4g}",
            f"{stats['p90']:.4g}",
            f"{stats['p95']:.4g}",
            f"{stats['p99']:.4g}",
            f"{stats['max']:.4g}",
        ])
    out.append(_table(["metric", "n", "min", "p5", "median", "p90", "p95", "p99", "max"], rows))
    out.append("")

    out.append("## Component quality")
    out.append("")
    quality = report["component_quality"]
    total = quality.get("total_components", 0) or 1
    rows = []
    for flag in list(ALL_FLAGS) + ["border_components"]:
        value = quality.get(flag, 0)
        rows.append([f"`{flag}`", str(value), f"{100 * value / total:.3f}%"])
    rows.append(["`total_components`", str(total), "100%"])
    out.append(_table(["flag", "count", "share"], rows))
    out.append("")
    out.append(
        "> `suspected_large_merge` is a **conservative heuristic, not merge ground truth**."
    )
    out.append("")

    out.append("## Relation counts")
    out.append("")
    rows = []
    for relation, counts in report["relation_counts"].items():
        total_rel = sum(counts.values()) or 1
        rows.append([
            f"`{relation}`",
            str(counts.get("valid", 0)),
            str(counts.get("ambiguous", 0)),
            str(counts.get("discarded", 0)),
            f"{100 * counts.get('valid', 0) / total_rel:.2f}%",
        ])
    out.append(_table(["relation", "valid", "ambiguous", "discarded", "valid share"], rows))
    out.append("")

    out.append("## Relation counts per split")
    out.append("")
    rows = []
    for relation in report["relation_counts"]:
        row = [f"`{relation}`"]
        for split in SPLITS:
            counts = report["relation_counts_by_split"].get(split, {}).get(relation, {})
            row.append(str(counts.get("valid", 0)))
        rows.append(row)
    out.append(_table(["relation"] + [f"{s} valid" for s in SPLITS], rows))
    out.append("")

    out.append("## Ambiguity")
    out.append("")
    rows = [[f"`{rel}`", str(count)] for rel, count in report["ambiguity_by_relation"].items()]
    out.append(_table(["relation", "ambiguous"], rows) if rows else "_none_")
    out.append("")
    out.append("Discard reasons (non-ambiguous rejections):")
    out.append("")
    rows = [[f"`{reason}`", str(count)] for reason, count in report["discard_reasons"].items()]
    out.append(_table(["reason", "count"], rows) if rows else "_none_")
    out.append("")

    out.append("## Border component analysis")
    out.append("")
    out.append(
        "Tile-relative relations must tolerate border truncation; magnitude relations must not."
    )
    out.append("")
    rows = []
    for relation, counter in report["border_eligibility_by_relation"].items():
        eligible = counter.get("eligible", 0)
        eb = counter.get("eligible_border", 0)
        rejected = counter.get("rejected", 0)
        rb = counter.get("rejected_border", 0)
        rows.append([
            f"`{relation}`",
            str(eligible),
            str(eb),
            f"{100 * eb / eligible:.2f}%" if eligible else "-",
            str(rejected),
            str(rb),
        ])
    out.append(_table(["relation", "eligible", "eligible&border", "share", "rejected", "rejected&border"], rows))
    out.append("")

    out.append("## Level-1 feasibility")
    out.append("")
    out.append("Number of images by how many reliable unique-answer Level-1 relations they support.")
    out.append("")
    rows = [[f"`{k}`", str(v)] for k, v in report["level1_feasibility"].items()]
    out.append(_table(["reliable relations", "images"], rows))
    out.append("")

    out.append("## Level-2 feasibility preview")
    out.append("")
    out.append(
        "Counts of images where the whole chain is uniquely determined. "
        "**No instruction text is generated here**; this is a feasibility measurement only."
    )
    out.append("")
    rows = [[f"`{name}`", str(count)] for name, count in report["level2_feasibility"].items()]
    out.append(_table(["chain", "images"], rows))
    out.append("")

    viz = report.get("visualization")
    if viz:
        out.append("## Visualization audit")
        out.append("")
        out.append(
            _table(
                ["item", "count"],
                [
                    ["visualizations_generated", str(viz.get("visualizations_generated", 0))],
                    ["contact_sheet_tiles", str(viz.get("contact_sheet_tiles", 0))],
                    ["ground_truth_panel_tiles", str(viz.get("ground_truth_panel_tiles", 0))],
                ],
            )
        )
        out.append("")
        out.append(
            f"- `evaluation/relation_samples/contact_sheet.png` — every sample in one image.\n"
            f"- `evaluation/relation_samples/ground_truth_panel.png` — raw component maps with ids.\n"
            f"- Generator: `{viz.get('sample_dir')}`.\n"
        )
        out.append(
            "> `visualizations_generated` counts images produced. It is NOT a manual review count.\n"
            "> The number of samples actually inspected by a human is recorded separately in\n"
            "> this document's maintainer note, not inferred from the generation count.\n"
        )
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Sample visualisation
# --------------------------------------------------------------------------


def render_samples(config: T.RelationConfig, out_dir: Path, n_train: int, n_val: int, n_test: int) -> list[str]:
    """Render a small manual-review sample. Not a full-dataset visualisation."""

    import random

    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    palette = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 190), (0, 128, 128), (170, 110, 40),
    ]

    rng = random.Random(20260804)
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        records = list(G.iter_metadata(DATASET_ROOT, split))
        rng.shuffle(records)
        taken = 0
        for record in records:
            if taken >= n:
                break
            image = G.image_geometry_from_record(record)
            if len(image.components) < 2:
                continue
            cmap = image.load_map(DATASET_ROOT)
            image_relations = R.evaluate_image(image, config)

            canvas = np.zeros((image.height, image.width, 3), np.uint8)
            for component in image.components:
                colour = palette[component.component_id % len(palette)]
                canvas[cmap == component.component_id] = colour

            for component in image.components:
                cx, cy = int(round(component.centroid_x)), int(round(component.centroid_y))
                cv2.circle(canvas, (cx, cy), 2, (255, 255, 255), -1)
                x0, y0, x1, y1 = component.bbox_xyxy_px
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)
                cv2.putText(canvas, str(component.component_id), (x0, max(10, y0 - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # draw a handful of decided relations as arrows
            drawn = 0
            for result in image_relations.directional:
                if not result.valid or drawn >= 6:
                    continue
                a = image.get(result.subject)
                b = image.get(result.object)
                cv2.arrowedLine(canvas, (int(a.centroid_x), int(a.centroid_y)),
                                (int(b.centroid_x), int(b.centroid_y)), (0, 255, 0), 1,
                                cv2.LINE_AA, tipLength=0.15)
                drawn += 1

            # annotate extreme / nearest decisions
            lines = []
            for name in ("leftmost", "rightmost", "topmost", "bottommost"):
                res = image_relations.extremes.get(name)
                if res and res.valid:
                    lines.append(f"{name}={res.subject}")
            for name in ("largest", "smallest"):
                res = image_relations.size_rank.get(name)
                if res and res.valid:
                    lines.append(f"{name}={res.subject}")
            nearest_valid = sum(1 for r in image_relations.nearest.values() if r.valid)
            lines.append(f"nearest_valid={nearest_valid}")

            for i, text in enumerate(lines[:10]):
                cv2.putText(canvas, text, (4, 12 + i * 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (255, 255, 0), 1, cv2.LINE_AA)

            path = out_dir / f"{split}_{image.image_id}.png"
            cv2.imwrite(str(path), canvas)
            written.append(path.name)
            taken += 1
    return written


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_contact_sheet(sample_dir: Path, out_path: Path, columns: int = 8) -> int:
    """Tile the sample visualisations into one contact sheet for quick review.

    Keeps the individual PNGs too; this is just a single image a human can open
    to review every sample at once.
    """

    import cv2

    paths = sorted(sample_dir.glob("*.png"))
    if not paths:
        return 0

    thumbs = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        thumb = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, path.stem, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    if not thumbs:
        return 0

    rows = (len(thumbs) + columns - 1) // columns
    sheet = np.zeros((rows * 192, columns * 192, 3), np.uint8)
    for index, thumb in enumerate(thumbs):
        r, c = divmod(index, columns)
        sheet[r * 192:(r + 1) * 192, c * 192:(c + 1) * 192] = thumb

    cv2.imwrite(str(out_path), sheet)
    return len(thumbs)


def render_ground_truth_panel(config: T.RelationConfig, out_path: Path, n: int = 6) -> int:
    """Render a small panel showing the raw component map with ids for review."""

    import cv2

    palette = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    ]
    tiles = []
    for record in G.iter_metadata(DATASET_ROOT, "train"):
        image = G.image_geometry_from_record(record)
        if len(image.components) < 3:
            continue
        cmap = image.load_map(DATASET_ROOT)
        canvas = np.zeros((image.height, image.width, 3), np.uint8)
        for component in image.components:
            canvas[cmap == component.component_id] = palette[component.component_id % len(palette)]
            x0, y0, _, _ = component.bbox_xyxy_px
            cv2.putText(canvas, str(component.component_id), (x0, max(10, y0 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_NEAREST))
        if len(tiles) >= n:
            break

    if not tiles:
        return 0
    sheet = np.zeros((256, 256 * len(tiles), 3), np.uint8)
    for index, tile in enumerate(tiles):
        sheet[:, index * 256:(index + 1) * 256] = tile
    cv2.imwrite(str(out_path), sheet)
    return len(tiles)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="max images per split (debug)")
    parser.add_argument("--no-samples", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = T.load_config()
    if not args.quiet:
        print(f"config   : {config.source_path}")
        print(f"dataset  : {DATASET_ROOT}")

    report = analyze(args.limit, args.quiet)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DIR / "relation_statistics_v1.json"

    samples: list[str] = []
    contact_count = 0
    gt_count = 0
    if not args.no_samples:
        sample_dir = EVAL_DIR / "relation_samples"
        for stale in sample_dir.glob("*.png"):
            stale.unlink()
        samples = render_samples(config, sample_dir, 20, 10, 10)
        contact_count = build_contact_sheet(sample_dir, sample_dir / "contact_sheet.png")
        gt_count = render_ground_truth_panel(config, sample_dir / "ground_truth_panel.png")

    report["visualization"] = {
        "visualizations_generated": len(samples),
        "contact_sheet_tiles": contact_count,
        "ground_truth_panel_tiles": gt_count,
        "manual_visual_inspection": "see report",
        "sample_dir": "evaluation/relation_samples",
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    markdown_path = DOCS_DIR / "relation_statistics_v1.md"
    markdown_path.write_text(render_markdown(report) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"wrote    : {json_path}")
        print(f"wrote    : {markdown_path}")
        print(f"samples  : {len(samples)} images + contact_sheet.png ({contact_count} tiles)")
        print()
        print("relation valid counts:")
        for relation, counts in report["relation_counts"].items():
            print(f"  {relation:12s} valid={counts['valid']:7d} ambiguous={counts['ambiguous']:6d} discarded={counts['discarded']:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
