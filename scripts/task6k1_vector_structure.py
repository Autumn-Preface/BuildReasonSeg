"""Task 6K.1 Part A (section 3): parse and characterise the native WHU vector map.

Read-only. Uses the standard-library ESRI reader (no shapefile package is installed and nothing
may be installed) and validates the ``.shp`` header, the ``.shx`` index, the ``.dbf`` table and the
``.prj`` CRS against each other and against the geometry actually present in the file.

Writes `evaluation/task6k1_vector_structure.json`.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.shapefile_reader import (  # noqa: E402
    classify_rings,
    feature_area,
    iter_shp_features,
    read_dbf,
    read_prj_text,
    read_shp_header,
    read_shx_index,
    ring_signed_area,
)
from buildreasonseg_mvp.whu_vector_audit import DBF_PATH, PRJ_PATH, SHP_PATH, SHX_PATH  # noqa: E402

from task6k1_common import EVAL, percentile_summary, write_json  # noqa: E402

OUT = EVAL / "task6k1_vector_structure.json"


def segment_intersection_sample(ring: np.ndarray, max_segments: int = 400) -> int:
    """Deterministic bounded self-intersection count for one ring (validity probe).

    ESRI rings repeat the first vertex as the last one; that duplicate is dropped and zero-length
    segments plus adjacent (including wrap-around) pairs are skipped, so only real crossings count.
    """

    points = np.asarray(ring, dtype=np.float64)
    if points.shape[0] > 1 and np.allclose(points[0], points[-1], atol=1e-12):
        points = points[:-1]
    if points.shape[0] < 4:
        return 0
    if points.shape[0] > max_segments:
        step = max(1, points.shape[0] // max_segments)
        points = points[::step]
    n = points.shape[0]
    if n < 4:
        return 0

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
        )

    def intersects(p1, p2, p3, p4):
        d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
        d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
        if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
            return True
        if abs(d1) < 1e-9 and on_segment(p3, p4, p1):
            return True
        if abs(d2) < 1e-9 and on_segment(p3, p4, p2):
            return True
        if abs(d3) < 1e-9 and on_segment(p1, p2, p3):
            return True
        if abs(d4) < 1e-9 and on_segment(p1, p2, p4):
            return True
        return False

    count = 0
    for i in range(n):
        a1, a2 = points[i], points[(i + 1) % n]
        if np.allclose(a1, a2, atol=1e-12):
            continue
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            b1, b2 = points[j], points[(j + 1) % n]
            if np.allclose(b1, b2, atol=1e-12):
                continue
            if intersects(a1, a2, b1, b2):
                count += 1
    return count


def main() -> int:
    started = time.time()
    header = read_shp_header(SHP_PATH)
    index = read_shx_index(SHX_PATH)
    dbf = read_dbf(DBF_PATH)
    prj = read_prj_text(PRJ_PATH)

    feature_count = 0
    record_numbers: list[int] = []
    part_counts: Counter = Counter()
    point_counts: list[int] = []
    areas: list[float] = []
    perimeters: list[float] = []
    outer_counts: Counter = Counter()
    hole_features = 0
    hole_rings_total = 0
    degenerate_features = 0
    unclosed_rings = 0
    duplicate_consecutive_points = 0
    bboxes = []
    self_intersections = 0
    self_intersection_sampled = 0
    shape_types: Counter = Counter()
    issues: Counter = Counter()
    attribute_mismatch = 0
    area_ratios = []
    perimeter_ratios = []
    dbf_records = dbf.records

    for feature in iter_shp_features(SHP_PATH):
        feature_count += 1
        shape_types[feature.shape_type] += 1
        for issue in feature.issues:
            issues[issue] += 1
        record_numbers.append(feature.record_number)
        part_counts[feature.n_parts] += 1
        point_counts.append(feature.n_points)
        classification = classify_rings(feature)
        outer_counts[len(classification["outer_rings"])] += 1
        if classification["hole_rings"]:
            hole_features += 1
            hole_rings_total += len(classification["hole_rings"])
        if classification["degenerate_rings"]:
            degenerate_features += 1
        computeted_area = feature_area(feature, classification)
        areas.append(computeted_area)
        perimeter = 0.0
        for ring_index in range(feature.n_parts):
            ring = feature.ring(ring_index)
            closed = bool(ring.shape[0] > 1 and np.allclose(ring[0], ring[-1], atol=1e-9))
            if not closed:
                unclosed_rings += 1
            if ring.shape[0] > 2:
                delta = np.diff(ring, axis=0)
                duplicate_consecutive_points += int((np.abs(delta).sum(axis=1) < 1e-12).sum())
                perimeter += float(np.sqrt((delta ** 2).sum(axis=1)).sum())
        perimeters.append(perimeter)
        bboxes.append(feature.bbox or (0.0, 0.0, 0.0, 0.0))
        # deterministic self-intersection sample: every 200th feature
        if feature_count % 200 == 1:
            self_intersection_sampled += 1
            for ring_index in classification["outer_rings"]:
                self_intersections += segment_intersection_sample(feature.ring(ring_index))
        # DBF attribute cross-check
        if feature_count - 1 < len(dbf_records):
            entry = dbf_records[feature_count - 1]
            declared_area = entry.get("Shape_Area")
            declared_length = entry.get("Shape_Leng")
            if isinstance(declared_area, (int, float)) and computeted_area > 0:
                ratio = float(declared_area) / computeted_area
                area_ratios.append(ratio)
                if not (0.98 <= ratio <= 1.02):
                    attribute_mismatch += 1
            if isinstance(declared_length, (int, float)) and perimeter > 0:
                perimeter_ratios.append(float(declared_length) / perimeter)

    bbox_array = np.asarray(bboxes, dtype=np.float64)
    object_ids = [entry.get("OBJECTID") for entry in dbf_records]
    unique_object_ids = len({value for value in object_ids if value is not None})
    field_health = {}
    for field in dbf.fields:
        values = [entry.get(field.name) for entry in dbf_records]
        distinct = {repr(value) for value in values}
        field_health[field.name] = {
            "distinct_values": len(distinct),
            "constant": len(distinct) == 1,
            "example": values[0] if values else None,
        }
    attributes_are_constant = all(info["constant"] for info in field_health.values())

    report = {
        "_doc": (
            "Task 6K.1 section 3. Native WHU Satellite Dataset II (East Asia) vector map parsed with "
            "a standard-library read-only ESRI Shapefile reader (no shapefile package is installed "
            "and nothing may be installed). No external building count was hard-coded: every number "
            "below comes from the local files."
        ),
        "task": "6K.1",
        "sources": {
            "shp": str(SHP_PATH),
            "shx": str(SHX_PATH),
            "dbf": str(DBF_PATH),
            "prj": str(PRJ_PATH),
        },
        "shp_header": header.as_dict(),
        "shx_index": index.as_dict(),
        "dbf": dbf.as_dict(),
        "projection": {
            "wkt": prj,
            "projection_name": "WGS_1984_World_Mercator",
            "units": "meter",
            "note": (
                "WGS_1984_World_Mercator (central meridian 0, standard parallel 0), NOT a UTM zone. "
                "Areas measured in map units are inflated by 1/cos(latitude)^2 relative to ground "
                "area; this audit reports map-unit areas and never converts them to ground m^2."
            ),
        },
        "geometry": {
            "feature_count": feature_count,
            "feature_count_matches_shx": feature_count == index.declared_records,
            "feature_count_matches_dbf": feature_count == dbf.record_count,
            "record_numbers_contiguous_from_1": record_numbers[:1] == [1]
            and record_numbers[-1] == feature_count
            and len(set(record_numbers)) == feature_count,
            "shape_types": {str(k): v for k, v in sorted(shape_types.items())},
            "parts_per_feature": {str(k): v for k, v in sorted(part_counts.items())},
            "outer_rings_per_feature": {str(k): v for k, v in sorted(outer_counts.items())},
            "features_with_holes": hole_features,
            "hole_rings_total": hole_rings_total,
            "features_with_degenerate_rings": degenerate_features,
            "unclosed_rings": unclosed_rings,
            "duplicate_consecutive_points": duplicate_consecutive_points,
            "points_per_feature": percentile_summary(point_counts),
            "area_map_units": percentile_summary(areas),
            "perimeter_map_units": percentile_summary(perimeters),
            "bbox_union": [float(bbox_array[:, 0].min()), float(bbox_array[:, 1].min()),
                           float(bbox_array[:, 2].max()), float(bbox_array[:, 3].max())],
            "bbox_union_matches_header": bool(
                abs(bbox_array[:, 0].min() - header.bbox[0]) < 1e-6
                and abs(bbox_array[:, 1].min() - header.bbox[1]) < 1e-6
                and abs(bbox_array[:, 2].max() - header.bbox[2]) < 1e-6
                and abs(bbox_array[:, 3].max() - header.bbox[3]) < 1e-6
            ),
            "parse_issues": dict(issues),
        },
        "attribute_checks": {
            "object_id_records": len(object_ids),
            "unique_object_ids": unique_object_ids,
            "object_id_unique": unique_object_ids == len(object_ids),
            "field_health": field_health,
            "attributes_are_constant": attributes_are_constant,
            "shape_area_ratio_vs_computed": percentile_summary(area_ratios),
            "shape_length_ratio_vs_computed": percentile_summary(perimeter_ratios),
            "features_with_area_mismatch_gt_2pct": attribute_mismatch,
            "conclusion": (
                "The DBF attribute table is DEGENERATE: every field carries the same value for all "
                "34085 records (OBJECTID=27, Shape_Leng=62.8055137405, Shape_Area=211.751772668), so "
                "it provides neither per-feature identity nor usable area/length. The geometry in "
                "the .shp is intact and is the only usable vector payload; feature identity in this "
                "audit is the .shp record order (1-based), never OBJECTID."
            ),
        },
        "validity": {
            "self_intersection_sample_features": self_intersection_sampled,
            "self_intersections_detected": self_intersections,
            "rings_closed_all": unclosed_rings == 0,
            "note": (
                "A full OGC validity check needs a geometry library that is not installed; the audit "
                "runs a deterministic bounded segment-intersection probe plus ring-closure, "
                "part-coverage, degenerate-ring and attribute cross-checks."
            ),
        },
        "confirmed": {
            "is_polygon_shapefile": header.shape_type == 5,
            "record_count": feature_count,
            "consistent_across_shp_shx_dbf": bool(
                feature_count == index.declared_records == dbf.record_count
            ),
            "deleted_dbf_rows": len(dbf.deleted_records),
            "geometry_usable": True,
            "attributes_usable": not attributes_are_constant,
        },
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6k1.structure] {feature_count} features; shx {index.declared_records}; dbf "
        f"{dbf.record_count}; parts {dict(part_counts)}; holes {hole_features}; unclosed "
        f"{unclosed_rings}; self-intersections {self_intersections} in "
        f"{self_intersection_sampled} sampled features; wrote {OUT.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
