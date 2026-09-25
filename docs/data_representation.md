# WHU Building Component Representation (v1.0)

How the legacy YOLO polygon labels are turned into a stable component-index
representation, and what that representation does and does not mean.

---

## 1. Terminology: component, not instance

A polygon in the legacy labels is a **building connected component**. It is
**not** a verified physical building instance.

The chain that produced it:

```
original raster label (binary semantic mask, values {0, 255})
    -> connected component
    -> RETR_EXTERNAL contour
    -> polygon                <- the legacy YOLO label
    -> deterministic rasterization
    -> component-index map     <- this representation
```

The source raster carried no instance identity, only one foreground class. So
the word "instance" is deliberately **not** used here. Where a legacy term must
be referenced it is written `pseudo_instance` / `connected_component`.

Throughout this repository:

| Term | Meaning |
|---|---|
| `component_id` | Integer id of a connected component within one image, `>= 1` |
| `component_map` | Indexed image where `0` = background and `k` = `component_id k` |
| `building_component` | One rasterized polygon |
| `pseudo_instance` | Legacy synonym, used only when quoting legacy code |

---

## 2. Storage layout

```
datasets/whu/
├── components/
│   ├── train/<image_id>.png     indexed PNG, uint8
│   ├── val/<image_id>.png
│   └── test/<image_id>.png
├── polygons/
│   ├── train.npz                full-precision polygon provenance
│   ├── val.npz
│   └── test.npz
├── metadata/
│   ├── train.jsonl              one JSON line per source image
│   ├── val.jsonl
│   └── test.jsonl
└── component_manifest.json      dataset-level summary
```

**One image produces one component-index map.** There is deliberately no
one-PNG-per-component layout: 36,926 separate binary masks would be wasteful
and would lose the intra-image component relationship that spatial reasoning
needs.

### Where the polygon vertices live

The polygon vertices are exact provenance (float64, unrounded, as written in the
legacy labels). They are stored in the compressed `polygons/<split>.npz`
archives rather than inline in the JSONL, because embedding 36,926
high-vertex-count polygons as decimal text inflated the metadata to ~82 MB for
no benefit.

There are **2,329,082 polygon vertices in total**. The archive stores one flat
`vertices` array (Nx2 float64) plus `offsets` and `lengths` giving each
component's slice, in the same order as the JSONL. Each component records
`polygon_vertices` (count) and `polygon_vertices_ref` (archive path), so
provenance stays complete and reachable. A test re-rasterizes polygons straight
from the archive and asserts they reproduce the stored component map exactly.

### Component id assignment

```
component_id = source_polygon_index + 1
```

`source_polygon_index` is the 0-based line number of the polygon in the label
file. This mapping is **stable and required**: `tests/test_component_conversion.py`
asserts it. Ids are never renumbered, so an id gap always corresponds to a
recorded defect rather than silent re-indexing.

### dtype

`uint8`, so ids `0..255`. The measured maximum is **52 components per image**,
far inside that range. The converter checks this explicitly and raises
`ComponentIdOverflowError` if a future dataset exceeds 255, rather than
overflowing silently. A test verifies the raise with a synthetic 256-polygon
label.

---

## 3. Geometry comes from the rasterized map

All geometry is derived from the **rasterized component pixel set**:

```
component_map
    -> pixel set per component
        -> area_px
        -> centroid_px
        -> bbox_xyxy_px
        -> width_px / height_px
```

It is **not** derived from polygon vertex means, bbox centres, the original
raster labels, or any model prediction. The reason is consistency: mask
supervision and geometry supervision must describe the same object. If area
came from the polygon and the mask from rasterization, they could disagree.

`centroid_px` is the mean of the component's pixel coordinates, not the mean of
its vertices. Vertex means are biased wherever `approxPolyDP` left vertices
unevenly distributed along the boundary.

The polygon vertices are still stored in the metadata, as **provenance only**,
under `polygon_normalized` and marked `geometry_source: rasterized_component_map`.

Each component also carries `continuous_polygon_area_px` and
`rasterization_area_error_ratio` (shoelace area vs rasterized area). These are
**diagnostics**, not ground truth. There is intentionally **no** global
assertion that the two agree within some percentage: for small components,
pixel discretization and boundary rounding make such a rule unstable. The
distribution is reported instead.

---

## 4. Rasterization

Backend: **`cv2.fillPoly`**.

### Provenance of the two OpenCV steps (do not conflate them)

| Step | Routine | Where | What it produced |
|---|---|---|---|
| Legacy label generation | `cv2.findContours` (RETR_EXTERNAL) + `cv2.approxPolyDP` | `WHU_Building_Segment/scripts/mask_to_yolo.py` | the polygon labels, from the binary raster |
| **This representation** | `cv2.fillPoly` | `datasets/transforms/polygon_to_component_map.py` | the component-index maps, from the existing polygons |

**These are different routines doing opposite jobs.** The legacy pipeline never
called `fillPoly`; it went mask -> contour -> polygon. This project goes
polygon -> filled region, and `fillPoly` is simply the rasterization backend of
the current canonical representation.

### What this does and does not guarantee

- It guarantees that rasterizing a stored polygon reproduces the coverage of
  that polygon **under OpenCV's closed-region fill convention**, i.e. it is
  self-consistent and deterministic.
- It does **not** mean the component map equals the original raster ground
  truth. The original raster is gone from the pipeline at the `findContours`
  step: interior holes were dropped by `RETR_EXTERNAL`, sub-50 px contours were
  filtered out, and `approxPolyDP` simplified the boundary. See section 7.

The polygon -> mask direction is therefore a best available reconstruction of
the polygon, **not** a recovery of the source mask.

A hand-written scanline rasterizer was implemented first and validated. It
matched `cv2.fillPoly` exactly on all axis-aligned shapes, but **over-filled
non-convex polygons by up to 46%**, because a min/max span fill implicitly
takes the convex hull (one measured case: 16,764 px rasterized vs 9,154 px
true). Reproducing OpenCV's exact closed-region boundary rule for non-convex
input would mean reimplementing its span-fill conventions with no benefit, so
the backend delegates to OpenCV and a test pins the parity.

Determinism: `cv2.fillPoly` on integer vertices is a pure integer scanline fill
with no threading or sampled state. A test converts the same label twice and
asserts the component maps **and the encoded PNG bytes** are identical.

Coordinates: `x_px = x_normalized * width`, matching the denominator convention
the legacy conversion used. Vertices are rounded with `np.rint` and are **not
clipped**; rasterization clamps to the viewport itself, and out-of-range
vertices are counted and reported rather than silently moved.

**No external raster label is read.** Only polygon vertices already present in
the workspace are used.

---

## 5. Conflict policy

Before painting a component, the converter tests:

```
new_component_mask AND existing_component_map != 0
```

If any pixel overlaps an existing component, the converter:

1. records the conflict (`source_polygon_index`, conflicting index, overlap px);
2. marks the image `rasterization.is_valid = false`;
3. **does not paint over the existing id** — the earlier component wins;
4. surfaces the conflict in the JSONL and in `component_manifest.json`.

It never resolves a conflict silently.

**Measured result: 0 conflicts across all 4,038 images.** This is expected: the
legacy polygons came from separate external contours of a binary mask and are
mutually disjoint by construction.

---

## 6. Small-polygon policy

The converter does **not** re-apply the historical `contourArea < 50` filter.
The legacy labels were already filtered when they were produced; filtering again
would change the legacy geometry ground truth.

Every non-empty line in an existing label file becomes exactly one component.
If a polygon rasterizes to zero pixels, that is recorded under
`rasterization.empty_mask_polygons` and the image is marked invalid — the
component is not dropped and ids are not renumbered.

**Measured result: 0 empty masks across all 36,926 components.**

---

## 7. Hole semantics (explicit limitation)

The legacy polygons came from `RETR_EXTERNAL` contours, therefore:

- **internal holes are already lost.** Courtyards, light wells and enclosed
  gaps were filled before this representation existed;
- a rasterized component is a **filled exterior polygon**;
- it is **not** a pixel-exact recovery of the original building footprint;
- downstream relations that use area (`largest`, `smallest`, relative size) are
  computed **on this representation** and inherit its bias.

> **This representation must never be described as equal to the original raster
> ground truth.**

Measured upstream: 74 of 300 sampled tiles had fewer polygons than connected
components, consistent with hole filling plus the `area < 50` filter. Area is
therefore slightly overestimated for buildings with interior voids.

---

## 8. Metadata schema

One JSON line per source image in `metadata/<split>.jsonl`:

```jsonc
{
  "image_id": "1_0",
  "split": "train",
  "image_path": "../WHU_Building_Segment/dataset/WHU_YOLO_dataset/images/train/1_0.tif",
  "label_path": "../WHU_Building_Segment/dataset/WHU_YOLO_dataset/labels/train/1_0.txt",
  "component_map": "datasets/whu/components/train/1_0.png",
  "width": 512,
  "height": 512,
  "n_components": 3,
  "components": [
    {
      "component_id": 1,
      "source_polygon_index": 0,
      "entity_semantics": "connected_component",
      "source_polygon_class_id": 0,
      "polygon_vertices": 110,
      "polygon_vertices_ref": "datasets/whu/polygons/train.npz",
      "geometry_source": "rasterized_component_map",
      "area_px": 2134,
      "area_ratio": 0.00814,
      "centroid_px": [474.94, 229.82],
      "centroid_normalized": [0.9276, 0.4489],
      "bbox_xyxy_px": [444, 203, 506, 256],
      "bbox_xyxy_normalized": [0.8672, 0.3965, 0.9883, 0.5],
      "width_px": 63,
      "height_px": 54,
      "touches_image_border": false,
      "continuous_polygon_area_px": 2046.0,
      "rasterization_area_error_ratio": 0.0430
    }
  ],
  "rasterization": {
    "is_valid": true,
    "empty_mask_polygons": [],
    "conflict_pairs": [],
    "malformed_lines": [],
    "non_building_class_lines": [],
    "out_of_bound_vertices": 0
  },
  "representation_version": "v1.0"
}
```

All paths are **relative**; no absolute path is ever written.

---

## 9. Regenerating

```bash
python scripts/build_whu_components.py --split all
```

Useful flags: `--limit N` (debug subset), `--dataset-root`, `--out-dir`, `--quiet`.

The legacy dataset is only ever read. New artifacts are written solely under
this repository.
