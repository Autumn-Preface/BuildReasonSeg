# WHU Data Reference

> **No WHU images or labels are copied into this repository.**
> The legacy dataset is used **in place**, read-only, via relative paths.

---

## Naming and layout of future datasets

This directory is **source-specific**: it holds raw/derived data for the *first*
source dataset (WHU), which is why it is still called `whu/`.

| Path | Contents | Status |
|---|---|---|
| `datasets/whu/` | source-specific derived data for the WHU source dataset | **exists** (Task 2 output) |
| `datasets/build_spatial_reason/` | the planned **BuildSpatialReason** reasoning annotation dataset | **not generated yet** |

The reasoning dataset is named **BuildSpatialReason** (`build_spatial_reason`).
It is deliberately *not* named after the source dataset, because the project is
not tied to WHU: further public building / remote-sensing datasets are expected
to be added later, and each will get its own source-specific directory alongside
`whu/`. See `docs/architecture_decisions.md` → ADR-009.

---

## 1. Locations

Resolved from this directory (`BuildReasonSeg/datasets/whu/`):
| What | Relative path | Access |
|---|---|---|
| YOLO-format dataset (images + polygon labels) | `../../../WHU_Building_Segment/dataset/WHU_YOLO_dataset` | **read-only source** |
| Legacy conversion script | `../../../WHU_Building_Segment/scripts/mask_to_yolo.py` | **read-only reference** |
| Legacy dataset config | `../../../WHU_Building_Segment/building_dataset.yaml` | read-only (copy archived in `baseline/`) |
| Original WHU raster dataset | *not reachable from this workspace* | read-only, external |

The original raster dataset lives outside this workspace. Its absolute location is recorded
in `baseline/yolo_whu/manifest.json` under a key marked `_historical_absolute` **for
provenance only**. No program in this repository may read that field.

**No absolute paths are permitted in any config or source file in this repository.**

---

## 2. Dataset shapes

| Split | Images | Labels | Image bytes | Label bytes |
|---|---|---|---|---|
| train | 2508 | 2508 | 1,950,200,830 | 30,695,290 |
| val | 627 | 627 | 487,831,670 | 8,710,898 |
| test | 903 | 903 | 714,535,770 | 11,951,639 |
| **total** | **4038** | **4038** | — | — |

- Images: `tif`, RGB, 512x512, `uint8`.
- Image/label pairing was verified complete in all three splits
  (0 images without labels, 0 labels without images).

### Split provenance

- `val` is a **random 20% subset of the original WHU train pool**
  (`VALIDATION_RATIO = 0.2`, `RANDOM_SEED = 42`) — see `mask_to_yolo.py` lines 13-14, 145-147.
- `test` is the original WHU test set.
- Consequently `train` + `val` together reconstruct the original WHU train set.

> This split is acceptable for baseline instance segmentation, but it is a random split of a
> geographically contiguous source. Scene-level leakage between train and val cannot be ruled
> out from the available metadata. Reasoning-query evaluation will need a stronger split
> discipline than this.

---

## 3. Label format (geometry GT source)

YOLO polygon instance labels. One instance per line:

```
<class_id> x1 y1 x2 y2 ... xN yN
```

- `class_id` is always `0` (`building`).
- Coordinates are **normalized to [0, 1]** — multiply by 512 to obtain pixels.
- Vertex counts vary; they are not fixed-length.

Verified label statistics (measured over all 4038 label files):

| Statistic | Value |
|---|---|
| Total instances | 36,926 |
| Instances per image | min 1, median 7, mean 9.14, p90 19, max 52 |
| Vertices per polygon | min 3, median 54, mean 63.1, max 266 |
| Instance area (px^2) | min 50, p10 310, median 1176, p90 3072, max 131,054 |
| Area as fraction of image | median 0.449%, p90 1.17%, max 49.99% |
| Malformed lines | 0 |

### How these labels were produced

`mask_to_yolo.py` reads a **binary** raster label (pixel values `{0, 255}` only), thresholds
it, finds contours with `cv2.findContours(..., RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`,
drops contours with area < 50 px, simplifies with `approxPolyDP(eps = 0.001 * arcLength)`,
and writes normalized vertices.

**Consequences that constrain all downstream reasoning annotation:**

1. **An instance is a connected component.** Because the source raster is binary (a semantic
   mask, not an instance map), any mutually **touching buildings were merged into one
   instance** and cannot be separated again from these labels. Measured: 2.601% of sampled
   instance pairs have a contour distance <= 1.5 px.
2. **Instances are mutually disjoint by construction.** Because only external contours were
   used, no two polygons overlap. Measured over 25,454 sampled instance pairs:
   **0 overlapping pairs, 0 pairs with IoU > 0.05, 0 containment pairs.**
   Therefore `overlap`, `inside`, and `contain` relations are structurally empty on this data
   and must not be used.
3. **Holes were filled.** Buildings with interior courtyards yield a solid outer polygon, so
   the recorded area is slightly **larger** than the true building footprint. 74 of 300
   sampled tiles had fewer polygons than connected components, all explained by the
   `area < 50` filter and hole filling (never more polygons than components).

These are properties of the data, not bugs to be fixed. See
`docs/architecture_decisions.md` → ADR-003.

---

## 4. Usage rules

- The legacy dataset is **read-only**. Never modify, reorganize, or delete it.
- The legacy polygon labels are the **geometry GT source** for annotation generation.
- **Newly generated annotations** (`BuildSpatialReason` instructions, relations, target
  instance ids, derived masks) are written **only** under this repository.
- Any derived artifact must record which `image_id` and which `annotation_version` it came
  from, so that regenerating it is possible and auditable.
- Geographic/scene grouping metadata for a stronger split is **unconfirmed** — it is not
  present in the labels and has not been verified in the source dataset.
