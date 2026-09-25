"""Tests for the polygon -> building component conversion.

Covers the seven required checks:

    A. Count preservation   n_components == number of non-empty polygon lines
    B. ID validity          map ids == {0, 1, ..., n_components}, no gaps
    C. Source mapping       component_id == source_polygon_index + 1
    D. Bounds               centroids / bboxes / pixel vertices in image range
    E. Area                 every component has area_px > 0
    F. Split preservation   the original train/val/test assignment is unchanged
    G. Determinism          converting the same label twice is byte-identical

Run either way:

    pytest tests/test_component_conversion.py -v
    python tests/test_component_conversion.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "datasets" / "transforms"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from polygon_to_component_map import (  # noqa: E402
    BACKGROUND_ID,
    build_component_map,
    check_id_contiguity,
    iter_split,
    parse_yolo_polygon_label,
    rasterize_polygon,
    read_image_size,
    read_indexed_png,
    unique_component_ids,
    write_indexed_png,
)

DATASET_ROOT = _REPO_ROOT.parent / "WHU_Building_Segment" / "dataset" / "WHU_YOLO_dataset"
OUT_DIR = _REPO_ROOT / "datasets" / "whu"
SPLITS = ("train", "val", "test")

#: Number of images sampled per split. Conversion of all 4038 images is already
#: covered by the CLI run; these tests sample to stay fast.
SAMPLE_PER_SPLIT = 40


def _sample(split: str, limit: int = SAMPLE_PER_SPLIT):
    items = []
    for i, triple in enumerate(iter_split(DATASET_ROOT, split)):
        if i >= limit:
            break
        items.append(triple)
    return items


def _convert(image_id: str, image_path: Path, label_path: Path, split: str):
    size = read_image_size(image_path)
    return build_component_map(label_path, size, image_id, split)


# ---------------------------------------------------------------------------
# A. Count preservation
# ---------------------------------------------------------------------------


def test_count_preservation():
    """n_components must equal the number of non-empty polygon lines."""

    checked = 0
    for split in SPLITS:
        for image_id, image_path, label_path in _sample(split):
            result = _convert(image_id, image_path, label_path, split)

            polygons, _, _ = parse_yolo_polygon_label(label_path)
            n_lines = len(polygons)

            assert len(result.components) == n_lines, (
                f"{image_id}: {len(result.components)} components vs {n_lines} polygon lines"
            )
            assert result.diagnostics.n_polygons_read == n_lines
            checked += 1
    assert checked > 0
    print(f"  [A] count preservation OK on {checked} images")


# ---------------------------------------------------------------------------
# B. ID validity
# ---------------------------------------------------------------------------


def test_id_validity():
    """Map ids must be exactly {0} U {1..n_components}, with no gaps."""

    for split in SPLITS:
        for image_id, image_path, label_path in _sample(split):
            result = _convert(image_id, image_path, label_path, split)
            cmap = result.component_map

            n = len(result.components)
            ids = unique_component_ids(cmap)

            assert BACKGROUND_ID not in ids
            if n > 0:
                assert ids.max() <= 255, f"{image_id}: max id {ids.max()} exceeds uint8"
                assert ids.min() >= 1

            if not result.diagnostics.empty_mask_polygons:
                assert list(ids) == list(range(1, n + 1)), (
                    f"{image_id}: ids {list(ids)[:10]}... != 1..{n}"
                )
                assert check_id_contiguity(cmap, n) == []
    print("  [B] ID validity OK")


# ---------------------------------------------------------------------------
# C. Source mapping
# ---------------------------------------------------------------------------


def test_source_mapping():
    """component_id == source_polygon_index + 1, and the map agrees."""

    for split in SPLITS:
        for image_id, image_path, label_path in _sample(split):
            result = _convert(image_id, image_path, label_path, split)

            seen = set()
            for geom in result.components:
                assert geom.component_id == geom.source_polygon_index + 1, (
                    f"{image_id}: component_id {geom.component_id} != "
                    f"source_polygon_index {geom.source_polygon_index} + 1"
                )
                assert geom.source_polygon_index not in seen, f"{image_id}: duplicate source index"
                seen.add(geom.source_polygon_index)

                if geom.area_px > 0:
                    # The id must actually be present in the map.
                    assert (result.component_map == geom.component_id).any(), (
                        f"{image_id}: id {geom.component_id} recorded but absent from map"
                    )
    print("  [C] source mapping OK")


# ---------------------------------------------------------------------------
# D. Bounds
# ---------------------------------------------------------------------------


def test_bounds():
    """Centroids, bboxes and pixel vertices must lie inside the image."""

    for split in SPLITS:
        for image_id, image_path, label_path in _sample(split):
            result = _convert(image_id, image_path, label_path, split)
            width, height = result.width, result.height

            for geom in result.components:
                if geom.area_px == 0:
                    continue
                cx, cy = geom.centroid_px
                assert 0.0 <= cx <= width - 1, f"{image_id}: centroid x {cx} out of range"
                assert 0.0 <= cy <= height - 1, f"{image_id}: centroid y {cy} out of range"

                x0, y0, x1, y1 = geom.bbox_xyxy_px
                assert 0 <= x0 <= x1 <= width - 1, f"{image_id}: bbox x {x0}..{x1}"
                assert 0 <= y0 <= y1 <= height - 1, f"{image_id}: bbox y {y0}..{y1}"

                assert geom.width_px == x1 - x0 + 1
                assert geom.height_px == y1 - y0 + 1

                ncx, ncy = geom.centroid_normalized
                assert 0.0 <= ncx <= 1.0 and 0.0 <= ncy <= 1.0

            for polygon in result.polygons:
                v = polygon.vertices_normalized
                assert np.all(np.isfinite(v)), f"{image_id}: non-finite vertex"
    print("  [D] bounds OK")


# ---------------------------------------------------------------------------
# E. Area
# ---------------------------------------------------------------------------


def test_area_positive():
    """Every rasterized component must have area_px > 0."""

    total = 0
    for split in SPLITS:
        for image_id, image_path, label_path in _sample(split):
            result = _convert(image_id, image_path, label_path, split)

            assert not result.diagnostics.empty_mask_polygons, (
                f"{image_id}: empty masks for polygons "
                f"{result.diagnostics.empty_mask_polygons}"
            )

            for geom in result.components:
                assert geom.area_px > 0, f"{image_id}: component {geom.component_id} area=0"
                assert geom.area_ratio > 0.0
                assert geom.area_ratio <= 1.0
                total += 1

            # area_px must agree with the number of pixels carrying that id.
            for geom in result.components:
                pixels = int((result.component_map == geom.component_id).sum())
                assert pixels == geom.area_px, (
                    f"{image_id}: id {geom.component_id} map has {pixels} px, "
                    f"metadata says {geom.area_px}"
                )
    print(f"  [E] area OK on {total} components")


# ---------------------------------------------------------------------------
# F. Split preservation
# ---------------------------------------------------------------------------


def test_split_preservation():
    """Every image must stay in its original split."""

    for split in SPLITS:
        images_dir = DATASET_ROOT / "images" / split
        labels_dir = DATASET_ROOT / "labels" / split

        for image_id, image_path, label_path in _sample(split):
            assert image_path.parent.name == split, (
                f"{image_id}: image came from split {image_path.parent.name}, expected {split}"
            )
            assert label_path.parent.name == split, (
                f"{image_id}: label came from split {label_path.parent.name}, expected {split}"
            )
            assert image_path.is_file() and label_path.is_file()

        # No image id may appear in two splits.
        stems = {p.stem for p in labels_dir.glob("*.txt")}
        for other in SPLITS:
            if other == split:
                continue
            other_stems = {p.stem for p in (DATASET_ROOT / "labels" / other).glob("*.txt")}
            overlap = stems & other_stems
            assert not overlap, f"split leak between {split} and {other}: {list(overlap)[:5]}"
    print("  [F] split preservation OK")


# ---------------------------------------------------------------------------
# G. Determinism
# ---------------------------------------------------------------------------


def test_determinism(tmp_dir: Path | None = None):
    """Converting the same label twice must give an identical map and PNG."""

    import shutil

    tmp = tmp_dir or (_REPO_ROOT / ".pytest_tmp_determinism")
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        for split in SPLITS:
            for image_id, image_path, label_path in _sample(split, limit=6):
                a = _convert(image_id, image_path, label_path, split)
                b = _convert(image_id, image_path, label_path, split)

                assert np.array_equal(a.component_map, b.component_map), (
                    f"{image_id}: component map differs between runs"
                )

                p1 = tmp / f"{image_id}_a.png"
                p2 = tmp / f"{image_id}_b.png"
                write_indexed_png(a.component_map, p1)
                write_indexed_png(b.component_map, p2)

                assert p1.read_bytes() == p2.read_bytes(), (
                    f"{image_id}: PNG encoding is not byte-identical"
                )
                assert np.array_equal(read_indexed_png(p1), a.component_map)
                assert read_indexed_png(p1).dtype == np.uint8
    finally:
        # Leave no scratch behind.
        if tmp_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    print("  [G] determinism OK")


# ---------------------------------------------------------------------------
# Extra: rasterization backend parity and dtype safety
# ---------------------------------------------------------------------------


def test_rasterizer_matches_opencv():
    """The rasterizer must agree with cv2.fillPoly on real polygons.

    This pins the rasterization backend: the legacy labels were produced with
    cv2, so rasterizing them must reproduce the same coverage.
    """

    import cv2

    checked = 0
    for split in ("train",):
        for image_id, image_path, label_path in _sample(split, limit=10):
            size = read_image_size(image_path)
            width, height = size
            result = _convert(image_id, image_path, label_path, split)

            for geom in result.components:
                if geom.area_px == 0:
                    continue
                from polygon_to_component_map import (
                    normalized_to_pixel,
                    to_raster_vertices,
                )

                polygon = result.polygons[geom.source_polygon_index]
                vertices_px = normalized_to_pixel(polygon.vertices_normalized, width, height)
                vertices_int, _ = to_raster_vertices(vertices_px, width, height)
                mine = rasterize_polygon(vertices_int, width, height)

                ref = np.zeros((height, width), np.uint8)
                cv2.fillPoly(ref, [vertices_int.astype(np.int32).reshape(-1, 1, 2)], 1)

                assert np.array_equal(mine, ref.astype(bool)), (
                    f"{image_id} polygon {geom.source_polygon_index}: rasterizer "
                    f"disagrees with cv2.fillPoly"
                )
                checked += 1
    assert checked > 0
    print(f"  [extra] rasterizer parity with cv2.fillPoly OK on {checked} polygons")


def test_component_id_overflow_is_loud():
    """Exceeding uint8 capacity must raise, never overflow silently."""

    from polygon_to_component_map import ComponentIdOverflowError

    # Build a synthetic label with 256 polygons, each a tiny triangle.
    lines = []
    for i in range(256):
        x = (i % 16) * 0.05
        y = (i // 16) * 0.05
        lines.append(
            f"0 {x:.4f} {y:.4f} {x + 0.01:.4f} {y:.4f} {x:.4f} {y + 0.01:.4f}"
        )
    tmp = _REPO_ROOT / ".pytest_tmp_overflow"
    tmp.mkdir(parents=True, exist_ok=True)
    label = tmp / "synthetic.txt"
    label.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        build_component_map(label, (64, 64), "synthetic", "train")
    except ComponentIdOverflowError:
        print("  [extra] uint8 overflow correctly raised")
        return
    finally:
        label.unlink(missing_ok=True)
        try:
            tmp.rmdir()
        except OSError:
            pass
    raise AssertionError("256 components did not raise ComponentIdOverflowError")


def test_metadata_schema_when_present():
    """If a conversion has been run, its JSONL must satisfy the schema."""

    jsonl = OUT_DIR / "metadata" / "train.jsonl"
    if not jsonl.is_file():
        print("  [extra] metadata JSONL not present yet (run the CLI first)")
        return

    required_component_keys = {
        "component_id",
        "source_polygon_index",
        "polygon_vertices",
        "polygon_vertices_ref",
        "area_px",
        "centroid_px",
        "bbox_xyxy_px",
        "entity_semantics",
    }
    required_record_keys = {
        "image_id",
        "split",
        "image_path",
        "label_path",
        "component_map",
        "width",
        "height",
        "n_components",
        "components",
        "representation_version",
    }

    with jsonl.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= 25:
                break
            record = json.loads(line)
            missing = required_record_keys - set(record)
            assert not missing, f"record {i} missing keys {missing}"
            assert record["n_components"] == len(record["components"])
            assert not os.path.isabs(record["image_path"]), "absolute path recorded"
            assert not os.path.isabs(record["component_map"]), "absolute path recorded"
            assert ":" not in record["image_path"][:3], "drive-letter path recorded"
            for component in record["components"]:
                cmissing = required_component_keys - set(component)
                assert not cmissing, f"component missing keys {cmissing}"
                assert component["entity_semantics"] == "connected_component"
    print("  [extra] metadata schema OK")


def test_polygon_archive_roundtrip():
    """The polygon archive must reproduce the legacy polygon vertices exactly.

    ``polygon_normalized`` lives in the compressed archive rather than inline in
    the JSONL. This test proves the provenance is still complete: every stored
    polygon must match the vertex count recorded inline and re-rasterize to the
    exact component map written to disk.
    """

    for split in SPLITS:
        jsonl = OUT_DIR / "metadata" / f"{split}.jsonl"
        archive_path = OUT_DIR / "polygons" / f"{split}.npz"
        if not jsonl.is_file() or not archive_path.is_file():
            print(f"  [extra] {split}: archive not present yet (run the CLI first)")
            continue

        data = np.load(archive_path)
        vertices_all = data["vertices"]
        offsets = data["offsets"]
        lengths = data["lengths"]

        records = [json.loads(line) for line in jsonl.open(encoding="utf-8")]
        n_components = sum(len(r["components"]) for r in records)
        assert offsets.shape[0] == n_components, (
            f"{split}: archive has {offsets.shape[0]} entries for {n_components} components"
        )

        cursor = 0
        checked = 0
        for record in records:
            for component in record["components"]:
                length = lengths[cursor]
                verts = vertices_all[offsets[cursor]:offsets[cursor] + length]
                assert verts.shape[0] == component["polygon_vertices"], (
                    f"{split} {record['image_id']} id {component['component_id']}: "
                    f"archive has {verts.shape[0]} vertices, "
                    f"metadata says {component['polygon_vertices']}"
                )
                assert verts.ndim == 2 and verts.shape[1] == 2
                assert np.all(np.isfinite(verts))

                # Re-rasterize from the archived polygon and compare with the map.
                if checked < 30 and component["area_px"] > 0:
                    from polygon_to_component_map import (
                        normalized_to_pixel,
                        to_raster_vertices,
                    )

                    vpx = normalized_to_pixel(verts, record["width"], record["height"])
                    vint, _ = to_raster_vertices(vpx, record["width"], record["height"])
                    mask = rasterize_polygon(vint, record["width"], record["height"])
                    assert int(mask.sum()) == component["area_px"], (
                        f"{split} {record['image_id']} id {component['component_id']}: "
                        f"re-rasterized area {int(mask.sum())} != recorded {component['area_px']}"
                    )
                    # And it must equal this component's pixels in the stored map.
                    stored = read_indexed_png(
                        OUT_DIR / "components" / split / f"{record['image_id']}.png"
                    )
                    assert np.array_equal(mask, stored == component["component_id"]), (
                        f"{split} {record['image_id']} id {component['component_id']}: "
                        f"re-rasterized mask differs from stored component map"
                    )
                cursor += 1
                checked += 1
        assert cursor == n_components
        print(f"  [extra] {split}: polygon archive roundtrip OK ({n_components} components, "
              f"{int(vertices_all.shape[0])} vertices)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("A count preservation", test_count_preservation),
        ("B id validity", test_id_validity),
        ("C source mapping", test_source_mapping),
        ("D bounds", test_bounds),
        ("E area", test_area_positive),
        ("F split preservation", test_split_preservation),
        ("G determinism", test_determinism),
        ("E2 rasterizer parity", test_rasterizer_matches_opencv),
        ("E3 uint8 overflow", test_component_id_overflow_is_loud),
        ("E4 metadata schema", test_metadata_schema_when_present),
        ("E5 polygon archive", test_polygon_archive_roundtrip),
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
    print(f"{len(tests) - failures}/{len(tests)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
