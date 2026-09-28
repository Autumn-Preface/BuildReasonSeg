"""Task 6K.1 Part E (section 9): native vector vs pseudo-instance / semantic-component matching.

Read-only. On every corpus tile the native vector instances are compared with

* the raw 8-connected components of the cropped raster semantic label, and
* the historical pseudo-instance view (the current component map = rasterized YOLO polygons),

and the following are measured:

* counts (distinct native features, clipped instances, instances per tile, pseudo instances/tile);
* **many-vector -> one semantic component** (the PRIMARY touching-building merge measure): per
  component the number of overlapping native features, with 1 / 2 / 3+ rates and the maximum;
* **one-vector -> many components** splits, attributed to rasterization/disconnection, tile-boundary
  clipping or other causes;
* pseudo <-> vector matching at IoU 0.25 / 0.50 / 0.75 (one-to-one rate, unmatched vector,
  unmatched pseudo, merged pseudo, split pseudo) with breakdowns by small area, border truncation,
  dense tile and split.

Writes `evaluation/task6k1_vector_vs_pseudo_matching.json`.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_source_audit import binarize, raw_components, read_semantic_label  # noqa: E402
from buildreasonseg_mvp.whu_vector_audit import (  # noqa: E402
    count_features_per_component,
    count_targets_per_feature,
)

from task6k1_common import (  # noqa: E402
    CROPPED_ROOT,
    EVAL,
    iou_masks,
    load_instance_view,
    masks_from_label_map,
    match_instances,
    percentile_summary,
    write_json,
)

OUT = EVAL / "task6k1_vector_vs_pseudo_matching.json"
THRESHOLDS = (0.25, 0.50, 0.75)
CONTAINMENT = 0.80          # a native building "sits inside" a component/pseudo instance
OVERLAP_SHARE = 0.50        # a native building "belongs to" a component for the primary merge count
SMALL_AREA_PX = 50          # the historical `<50` filter boundary
DENSE_TILE_INSTANCES = 10


def raster_label_path(stem: str) -> Path | None:
    for split in ("train", "train_no", "test", "test_no"):
        candidate = CROPPED_ROOT / split / "label" / f"{stem}.tif"
        if candidate.is_file():
            return candidate
    return None


def component_count(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(count) - 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    from task6k_common import aligned_tiles, converted_candidate_set

    tiles = aligned_tiles()
    if args.limit:
        tiles = tiles[: int(args.limit)]
    print(f"[task6k1.match] {len(tiles)} tiles", flush=True)

    totals = Counter()
    features_per_component = []
    features_contained_per_pseudo = []
    components_contained_per_pseudo = []
    pseudos_per_vector = []
    matching = {threshold: Counter() for threshold in THRESHOLDS}
    matched_area_ratios = []
    split_causes = Counter()
    breakdown = {
        "small_area": Counter(), "border_truncated": Counter(),
        "dense_tile": Counter(), "not_dense_tile": Counter(),
    }
    by_split = {}
    rows = []
    distinct_feature_ids: set[int] = set()

    for position, tile in enumerate(tiles, start=1):
        view = load_instance_view(tile.stem)
        pseudo_set = converted_candidate_set(tile)
        if view is None or pseudo_set is None:
            continue
        label_map = np.asarray(view["label_map"], dtype=np.int32)
        vector_masks = masks_from_label_map(label_map)
        vector_global_ids = [int(value) for value in view["ids"].tolist()]
        for global_id in vector_global_ids:
            distinct_feature_ids.add(global_id)
        pseudo_masks = [candidate.mask for candidate in pseudo_set.candidates]

        component_masks = []
        label_path = raster_label_path(tile.stem)
        if label_path is not None:
            binary = binarize(read_semantic_label(label_path))
            labels, records = raw_components(binary, connectivity=8)
            component_masks = [labels == record.component_id for record in records]

        # ---- matching at the three IoU thresholds
        per_threshold = {
            threshold: match_instances(vector_masks, pseudo_masks, threshold) for threshold in THRESHOLDS
        }
        for threshold, result in per_threshold.items():
            matching[threshold]["pairs"] += len(result["pairs"])
            matching[threshold]["unmatched_vector"] += len(result["unmatched_left"])
            matching[threshold]["unmatched_pseudo"] += len(result["unmatched_right"])
            matching[threshold]["vectors"] += len(vector_masks)
            matching[threshold]["pseudos"] += len(pseudo_masks)

        # ---- PRIMARY merge measure: native features per SEMANTIC COMPONENT
        component_counts = count_features_per_component(component_masks, vector_masks, OVERLAP_SHARE)
        features_per_component.extend(component_counts)

        # ---- native features inside one PSEUDO instance (containment)
        contained_counts = count_features_per_component(pseudo_masks, vector_masks, CONTAINMENT)
        features_contained_per_pseudo.extend(contained_counts)

        for pseudo_mask in pseudo_masks:
            contained = 0
            for component_mask in component_masks:
                area = int(component_mask.sum())
                if area == 0:
                    continue
                if int(np.logical_and(component_mask, pseudo_mask).sum()) / area >= CONTAINMENT:
                    contained += 1
            components_contained_per_pseudo.append(contained)

        # ---- splits: one native feature over several pseudo instances (and its cause)
        pseudo_multi_vector = sum(1 for count in contained_counts if count >= 2)
        span_counts = count_targets_per_feature(vector_masks, pseudo_masks, 0.25)
        for local_index, spanned in enumerate(span_counts):
            pseudos_per_vector.append(spanned)
            if spanned >= 2:
                vector_mask = vector_masks[local_index]
                disconnected = component_count(vector_mask) > 1
                border = bool(view["touches_border"][local_index])
                clipped = bool(view["clipped"][local_index])
                if disconnected:
                    cause = "rasterization_or_disconnection"
                elif border or clipped:
                    cause = "tile_boundary"
                else:
                    cause = "other"
                split_causes[cause] += 1

        # ---- breakdowns on the 0.50 matching
        dense = len(vector_masks) >= DENSE_TILE_INSTANCES
        bucket_key = "dense_tile" if dense else "not_dense_tile"
        for pair in per_threshold[0.50]["pairs"]:
            vector_area = int(vector_masks[pair["left"]].sum())
            pseudo_area = int(pseudo_masks[pair["right"]].sum())
            if vector_area and pseudo_area:
                matched_area_ratios.append(pseudo_area / vector_area)
            breakdown[bucket_key]["matched"] += 1
            if vector_area < SMALL_AREA_PX:
                breakdown["small_area"]["matched"] += 1
            if bool(view["touches_border"][pair["left"]]):
                breakdown["border_truncated"]["matched"] += 1
        for left in per_threshold[0.50]["unmatched_left"]:
            vector_area = int(vector_masks[left].sum())
            breakdown[bucket_key]["unmatched_vector"] += 1
            if vector_area < SMALL_AREA_PX:
                breakdown["small_area"]["unmatched_vector"] += 1
            if bool(view["touches_border"][left]):
                breakdown["border_truncated"]["unmatched_vector"] += 1
        breakdown[bucket_key]["vectors"] += len(vector_masks)
        breakdown["small_area"]["vectors"] += sum(1 for mask in vector_masks if int(mask.sum()) < SMALL_AREA_PX)
        breakdown["border_truncated"]["vectors"] += int(sum(1 for flag in view["touches_border"] if flag))

        split_bucket = by_split.setdefault(tile.split, Counter())
        split_bucket["tiles"] += 1
        split_bucket["vectors"] += len(vector_masks)
        split_bucket["pseudos"] += len(pseudo_masks)
        split_bucket["components"] += len(component_masks)
        split_bucket["matched_0_50"] += len(per_threshold[0.50]["pairs"])
        split_bucket["unmatched_vector_0_50"] += len(per_threshold[0.50]["unmatched_left"])

        totals["tiles"] += 1
        totals["vectors"] += len(vector_masks)
        totals["pseudos"] += len(pseudo_masks)
        totals["components"] += len(component_masks)
        totals["pseudo_multi_vector"] += pseudo_multi_vector
        rows.append({
            "stem": tile.stem,
            "split": tile.split,
            "vector_instances": len(vector_masks),
            "pseudo_instances": len(pseudo_masks),
            "raster_components": len(component_masks),
            "matched_0_50": len(per_threshold[0.50]["pairs"]),
            "unmatched_vector_0_50": len(per_threshold[0.50]["unmatched_left"]),
            "unmatched_pseudo_0_50": len(per_threshold[0.50]["unmatched_right"]),
            "pseudo_multi_vector": pseudo_multi_vector,
            "features_per_component_max": max((count for count in features_per_component[-len(component_masks):]), default=0)
            if component_masks else 0,
        })
        if position % 250 == 0:
            print(f"[task6k1.match] {position}/{len(tiles)} ({time.time() - started:.0f}s)", flush=True)

    vector_total = max(totals["vectors"], 1)
    pseudo_total = max(totals["pseudos"], 1)
    component_total = max(totals["components"], 1)
    histogram = {
        "1": sum(1 for count in features_per_component if count == 1),
        "2": sum(1 for count in features_per_component if count == 2),
        "3+": sum(1 for count in features_per_component if count >= 3),
        "0": sum(1 for count in features_per_component if count == 0),
    }

    report = {
        "_doc": (
            "Task 6K.1 section 9. Native vector instances vs the raw semantic components and the "
            "historical pseudo-instance view. The PRIMARY touching-building merge measure is the "
            "number of native features per SEMANTIC COMPONENT; containment-based counts are also "
            "reported per pseudo-instance, and splits are attributed to rasterization/disconnection, "
            "tile-boundary clipping or other causes."
        ),
        "task": "6K.1",
        "definitions": {
            "feature_belongs_to_component": (
                f">= {OVERLAP_SHARE:.2f} of the native feature's pixels inside the component"
            ),
            "feature_inside_pseudo": f">= {CONTAINMENT:.2f} of the native feature's pixels inside the pseudo-instance",
            "small_area_px": SMALL_AREA_PX,
            "dense_tile_instances": DENSE_TILE_INSTANCES,
        },
        "counts": {
            "tiles": totals["tiles"],
            "distinct_native_features_represented": len(distinct_feature_ids),
            "native_features_total": 34085,
            "clipped_native_instances": totals["vectors"],
            "native_instances_per_tile": totals["vectors"] / max(totals["tiles"], 1),
            "pseudo_instances": totals["pseudos"],
            "pseudo_instances_per_tile": totals["pseudos"] / max(totals["tiles"], 1),
            "raster_semantic_components": totals["components"],
            "raster_components_per_tile": totals["components"] / max(totals["tiles"], 1),
        },
        "many_vector_to_one_semantic_component": {
            "components_evaluated": len(features_per_component),
            "features_per_component_histogram": histogram,
            "rate_with_1": histogram["1"] / component_total,
            "rate_with_2": histogram["2"] / component_total,
            "rate_with_3_plus": histogram["3+"] / component_total,
            "rate_with_0": histogram["0"] / component_total,
            "max_features_in_one_component": max(features_per_component) if features_per_component else 0,
            "features_per_component_summary": percentile_summary(features_per_component),
            "components_containing_multiple_features_rate": (
                (histogram["2"] + histogram["3+"]) / component_total
            ),
            "interpretation": (
                "This is the primary measure of touching-building merge in the semantic raster: a "
                "component with >= 2 native features contains that many distinct manually delineated "
                "buildings."
            ),
        },
        "many_vector_to_one_pseudo_instance": {
            "pseudo_instances_evaluated": len(features_contained_per_pseudo),
            "histogram": {
                str(count): sum(1 for value in features_contained_per_pseudo if value == count)
                for count in range(0, 6)
            },
            "pseudo_instances_containing_multiple_features": totals["pseudo_multi_vector"],
            "rate": totals["pseudo_multi_vector"] / pseudo_total,
            "features_inside_multi_feature_pseudo_rate": (
                sum(count for count in features_contained_per_pseudo if count >= 2) / vector_total
            ),
            "components_contained_per_pseudo_summary": percentile_summary(components_contained_per_pseudo),
        },
        "one_vector_to_many_components": {
            "native_features_spanning_multiple_pseudo": sum(1 for count in pseudos_per_vector if count >= 2),
            "rate": sum(1 for count in pseudos_per_vector if count >= 2) / vector_total,
            "pseudos_per_vector_summary": percentile_summary(pseudos_per_vector),
            "split_causes": dict(split_causes),
        },
        "matching": {
            str(threshold): {
                "pairs": matching[threshold]["pairs"],
                "one_to_one_rate_over_vector": matching[threshold]["pairs"] / vector_total,
                "one_to_one_rate_over_pseudo": matching[threshold]["pairs"] / pseudo_total,
                "unmatched_vector_instances": matching[threshold]["unmatched_vector"],
                "unmatched_vector_rate": matching[threshold]["unmatched_vector"] / vector_total,
                "unmatched_pseudo_instances": matching[threshold]["unmatched_pseudo"],
                "unmatched_pseudo_rate": matching[threshold]["unmatched_pseudo"] / pseudo_total,
                "merged_pseudo_instances": totals["pseudo_multi_vector"],
                "split_pseudo_instances": sum(1 for count in pseudos_per_vector if count >= 2),
            }
            for threshold in THRESHOLDS
        },
        "matched_pseudo_over_vector_area_ratio_0_50": percentile_summary(matched_area_ratios),
        "breakdown": {
            name: {
                **dict(counter),
                "unmatched_vector_rate": (
                    counter["unmatched_vector"] / counter["vectors"] if counter["vectors"] else None
                ),
            }
            for name, counter in breakdown.items()
        },
        "by_split": {name: dict(counter) for name, counter in sorted(by_split.items())},
        "per_tile": rows,
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6k1.match] tiles {totals['tiles']}; features {totals['vectors']} vs pseudo "
        f"{totals['pseudos']} vs components {totals['components']}; "
        f"PRIMARY components with >=2 features "
        f"{report['many_vector_to_one_semantic_component']['components_containing_multiple_features_rate']:.4f} "
        f"(max {report['many_vector_to_one_semantic_component']['max_features_in_one_component']}); "
        f"pseudo with >=2 features {report['many_vector_to_one_pseudo_instance']['rate']:.4f}; "
        f"match@0.5 {report['matching']['0.5']['one_to_one_rate_over_vector']:.4f}",
        flush=True,
    )
    print(f"[task6k1.match] wrote {OUT.name} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
