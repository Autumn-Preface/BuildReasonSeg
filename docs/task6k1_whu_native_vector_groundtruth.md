# Task 6K.1 — Recover and Validate the Native WHU East-Asia Vector Ground Truth

> Task: `handoff/TO_DSH.md` · Verdict: **`MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`**
> Evidence: `evaluation/task6k1_*.json`, `evaluation/task6k1_samples/` · ADR-024 ·
> Tests: `tests/test_task6k1_whu_vector_audit.py`

Task 6K discovered `2. The shape file of the whole images\EA.shp` but did not parse it. This task
does: it parses the local shapefile with a **standard-library** reader (no pyshp/fiona/geopandas/GDAL/
rasterio/tifffile is installed and nothing may be installed), recovers the tile → whole-image
mapping, validates the vector against the raster labels, and re-answers the Task 6K and Task 6J
questions against **true instance ground truth**.

Everything is read-only: no file under the WHU archive, the converted dataset or the legacy project
was modified (`evaluation/task6k1_read_only_proof.json`), no model was trained, no data downloaded,
no package installed.

## 1. The native vector is a real building footprint map (§3)

| Property | Value |
|---|---|
| File | `EA.shp` 4,999,904 B · `.shx` 272,780 B · `.dbf` 1,636,210 B · `.prj` 354 B |
| Header | ESRI shapefile, version 1000, shape type **5 = Polygon**, declared size == actual size |
| Records | **34,085** — identical in `.shp`, `.shx` index and `.dbf` table; 0 deleted rows |
| Geometry | 34,080 single-part + **5 two-part** features; **5 features with interior holes**; 0 unclosed rings; 0 duplicate consecutive points |
| Vertices | min 4, median **5**, mean 5.7, max 60 per feature (simple, manually drawn quadrilaterals) |
| Areas (map units) | min 0.004, median 157.5, mean 226.8, p95 573, max 18,976 |
| Bounding box | matches the `.shp` header exactly: X 13,942,403.8–13,998,804.1 · Y 4,781,504.2–4,790,946.9 |
| CRS | `PROJCS["WGS_1984_World_Mercator", …]` — **World Mercator**, not UTM (areas are inflated by 1/cos²(lat); only map units are reported) |
| Validity | bounded segment-intersection probe on 171 sampled features: **0 self-intersections** |

**Two findings that change how the file must be used:**

1. **The DBF attribute table is degenerate.** All three fields are *constant* over all 34,085
   records (`OBJECTID = 27`, `Shape_Leng = 62.8055137405`, `Shape_Area = 211.751772668`). It
   therefore provides **no per-feature identity and no usable area/length**; the ArcGIS-computed
   values cannot even be compared with the geometry. Identity in this audit is the **`.shp` record
   order (1-based)**.
2. **The published counts are irrelevant here.** The local archive authoritatively contains
   **34,085** polygons; the 29,085/34,085 figures quoted externally are context only.

## 2. Whole-image georeferencing (§4)

| Raster | Size (px) | Type | Tiles | Compression |
|---|---|---|---|---|
| `image/train1.tif` | 95,521 × 27,801 (2.66 Gpx) | RGB uint8 | 128×128, 162,846 tiles | none (BigTIFF) |
| `image/train2.tif` | 34,772 × 27,802 | RGB uint8 | 128×128 | none (BigTIFF) |
| `image/test.tif` | 35,765 × 27,802 | RGB uint8 | 128×128 | none (BigTIFF) |
| `label/train1.tif` | 95,522 × 27,801 | 1-bit | 128×128 | LZW |
| `label/train2.tif` | 34,772 × 27,802 | LZW | | |
| `label/test.tif` | 35,765 × 27,802 | LZW | | |

The rasters are **BigTIFF** (magic 43) and were read with a minimal standard-library IFD parser that
extracts metadata only; pixel access uses small PIL windowed crops. **No multi-gigapixel raster is
ever fully decoded** (`full_decode_performed: false`). The label-side `.tfw` world files give a
pixel size of **0.33956958 m** and the origins:

```
train1: X0 = 13,942,416.196  Y0 = 4,790,946.487
test:   X0 = 13,974,852.277  Y0 = 4,790,946.621
train2: X0 = 13,986,996.983  Y0 = 4,790,946.621
```

These three strips tile the complete vector extent with ~1 m of overlap between neighbours, and the
label raster for `train1` is **one pixel wider** than its image raster (95,522 vs 95,521) — recorded,
and harmless for the 512-tile grid.

## 3. Tile mapping recovered and VALIDATED (§5)

Only **full** 512×512 windows were cropped, so the grid is `floor(width/512) × floor(height/512)` and
the tile index is `row * columns + col`:

| Region | Grid | Capacity | Cropped tiles in the region | Difference |
|---|---|---|---|---|
| `1_*` → train1 | 186 × 54 | 10,044 | 10,044 | **0** |
| `2_*` → train2 | 67 × 54 | 3,618 | 3,618 | **0** |
| bare → test | 69 × 54 | 3,726 | 3,726 | **0** |

Validation was performed rather than assumed:

* **RGB identity**: 40 sampled tiles per raster (120 total) are **pixel-identical**
  (`max_abs_diff = 0`) to their whole-image window;
* **label identity**: 120 sampled tiles per raster — every cropped label equals the whole-label
  window exactly (or both are empty);
* **residual shift**: a ±2 px offset search over 240 tiles finds the **best offset (0,0) in 240/240**,
  mean gain ≈ 0 → no systematic misalignment;
* **vector agreement inside the same windows**: mean IoU **0.975–0.985** per raster.

Mapping confidence: **validated**.

## 4. Vector ↔ raster alignment (§6-7)

Every one of the 4,038 positive tiles plus a 300-tile empty-tile sample was evaluated:

| Metric | Value |
|---|---|
| mean IoU (native vector vs raster label) | **0.9505** |
| tiles with IoU ≥ 0.90 | **4,036 / 4,038** |
| tiles with IoU < 0.50 | **0** |
| empty-sample agreement | 277/300 both empty |
| best offset = (0,0) | 240/240 sampled tiles |
| exact local feature count | **34,085** (`.shp` == `.shx` == `.dbf`) |

The ~5 % IoU gap is boundary discretisation between polygon rasterisation and the original label
raster (plus a few native features with no raster counterpart). **The native vector is confirmed as
the manually delineated building footprint map of this archive.**

## 5. Native vector vs semantic components vs pseudo-instances (§8-9)

| | Value |
|---|---|
| native features (corpus) | 38,824 clipped instances over 4,038 tiles · **32,590 distinct** features touched |
| instances per tile | mean 9.62 · border-truncated 34.1 % · multipart 11 |
| pseudo-instances | 36,926 |
| raw 8-connected semantic components | 38,309 |
| vector ↔ pseudo match at IoU 0.50 | **94.8 %** of native instances |
| vector instances without a pseudo match | **5.24 %** |

**The merge question, measured properly.** A merged pseudo-instance is *much larger* than either
building, so IoU under-detects merges; containment (`≥ 80 %` of a native building's pixels inside one
pseudo-instance) is the primary metric. The **primary** touching-building measure the task asks for is
the number of native features per **semantic component**:

| Native features per semantic component | Count | Rate |
|---|---|---|
| 1 | 37,414 | 97.66 % |
| **2** | 471 | **1.23 %** |
| **3+** | 33 | **0.086 %** |
| 0 (no native building inside the component) | 391 | 1.02 % |
| max in one component | **4** | — |

| Merge / split metric | Value |
|---|---|
| semantic components containing ≥ 2 native buildings | **1.32 %** |
| pseudo-instances containing ≥ 2 native buildings | **1.34 %** (496 instances) |
| native buildings sitting inside such a merged pseudo-instance | **2.65 %** |
| native buildings spanning ≥ 2 pseudo-instances (split) | **0.13 %** (51 instances: 46 rasterisation/disconnection, 5 tile boundary) |

**Matching at IoU 0.25 / 0.50 / 0.75:**

| Threshold | Pairs | One-to-one over vector / pseudo | Unmatched vector | Unmatched pseudo |
|---|---|---|---|---|
| 0.25 | 36,835 | 94.88 % / 99.75 % | 1,989 (5.12 %) | 91 (0.25 %) |
| **0.50** | **36,791** | **94.76 % / 99.63 %** | **2,033 (5.24 %)** | 135 (0.37 %) |
| 0.75 | 36,395 | 93.74 % / 98.56 % | 2,429 (6.26 %) | 531 (1.44 %) |

Breakdowns (at IoU 0.50): border-truncated instances have a 11.8 % unmatched rate vs 5.2 % overall;
dense tiles 5.5 % vs 4.7 % in sparse tiles; **native buildings smaller than 50 px are unmatched
100 % of the time (1,039 / 1,039)** — the historical `<50 contourArea` filter is the single
identifiable cause; and the unmatched rate is flat across splits (train 5.4 %, val 5.1 %, test 5.0 %),
so this is a conversion property, not a split artefact.

**The hypothesis that touching buildings merge into one pseudo-instance is therefore quantitatively
small (1.3 %)**, and the dominant deviation is the opposite direction: **5.24 % of native buildings
are absent from the pseudo view** (the historical `<50 contourArea` filter, hole filling and polygon
simplification — exactly Task 6K's measured conversion losses).

## 6. VECTOR vs PSEUDO relation drift (§10)

All 20 canonical programs (frozen Task 3B semantics) were executed on both candidate sets over the
4,038-tile corpus. Targets are compared through mask-overlap matching; ids are never inference inputs.

| Metric | Value |
|---|---|
| tiles with any relation-answer change | **1,596 / 4,038 (39.5 %)** |
| **weighted current-query target-change rate** | **6.96 %** (Task 6K's RAW-vs-CONVERTED figure was 4.33 %) |
| L1 / L2 / L3 target-change rate | **7.87 %** / 3.38 % / 5.03 % |
| per-program extremes | `leftmost` 9.4 %, `bottommost` 9.2 %, `topmost` 9.0 %, `rightmost` 8.8 %; `smallest_to_left_of` 0.4 % |
| native buildings without a pseudo counterpart | 2,033 (5.24 %) |

Attribution of the changed answers (`eligibility_ranking_consequences` 1,110, `below_50_removal` 782,
`border_clipping` 373, `multiple_vector_merged_into_one_pseudo` 75, `geometry_approximation` 27,
`one_vector_to_multiple_pseudo` 18, `vector_target_missing_in_pseudo` 17): the two largest classes are
the ranking/nearest consequences of the different candidate sets and the **`<50` removal** of small
native buildings — i.e. the conversion, not the merging of touching buildings, drives the drift.

**This 6.96 % — not Task 6K's 4.33 % — is the number that should drive the annotation decision**, and
it is above the declared 5 % materiality threshold.

## 7. Task 6J re-attributed against native truth (§11)

| Category (65 J1 failures) | Count | Share |
|---|---|---|
| proposal-model error against a **clean single native building** | **51** | **78.5 %** |
| representation mismatch (single native building, no proposal ≥ 0.5) | 13 | 20.0 % |
| pseudo-label definition error (target merges ≥ 2 native buildings) | **0** | **0 %** |
| ambiguous / inseparable | 1 | 1.5 % |

YOLO proposal recall on the failures that **do** have a clean vector target: **79.7 %**. No J1
failure is explained by a merged pseudo-instance target, so Task 6J/6K's proposal-model conclusion is
**confirmed on native ground truth** — the failure is the proposal chain, not the instance definition.

## 8. Split and geographic structure (§12)

With the mapping recovered, the split question becomes geometric:

| Metric | Value |
|---|---|
| val tiles directly adjacent to effective training tiles | **627 / 627 = 100 %** |
| val nearest effective-training tile (tile-grid units) | **1 for every val tile** (histogram {1: 627}) |
| mean training neighbours per val tile (of 8) | **6.91**; fully surrounded 40.7 % |
| val tiles sharing a whole-image raster with train | **100 %** (train1 333, train2 294) |
| test tiles adjacent to train/val | **0 %** (separate raster, no shared region) |
| approximate extent | train+val ≈ 56.2 km × 9.4 km · test ≈ 12.0 km × 9.4 km · lon 125.25–125.75, lat 39.61–39.67 |

The comparison uses `train_effective` = the source train pool **minus the val tiles** (the raw pool
contains the val tiles by construction, so comparing against it would trivially report distance 0).

**What the split supports:** random-tile generalisation within the same two source scenes (train vs
val) and one spatially disjoint test scene. **What it cannot support:** any claim of geographic
generalisation across cities or regions, or of unseen-scene robustness beyond the single held-out
test raster. Task 6K's region-level statement is now a per-tile adjacency fact.

## 9. Verdict (§13)

**`MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`**, with `keep_whu_imagery: true` and
`keep_historical_pseudo_baseline: true`.

Gates (declared before the verdict, all measured values in `task6k1_dataset_decision.json`):

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| vector confirmed as building footprints | mean IoU ≥ 0.90, ≥ 90 % tiles ≥ 0.75 | 0.9505, 99.95 % | PASS |
| tile mapping validated | pixel identity + exact grid capacity | all three rasters | PASS |
| true instance identity available | distinct valid polygons | 34,085 | PASS |
| pseudo-instances materially merge true buildings | > 0.05 | **0.0134** | FAIL (not the mechanism) |
| relation answers materially change | > 0.05 | **0.0696** | **PASS** |

Per the task's own criteria, `KEEP_CURRENT_PSEUDO_INSTANCES` would require that the shapefile is not
a footprint map, cannot be aligned, or gives **no meaningful instance advantage**; none holds. The
native vector (a) is the manual footprint map, (b) aligns at IoU 0.95 with zero residual shift,
(c) carries true instance identity where the semantic raster cannot, (d) removes every measured
conversion loss (no `<50` deletion, no hole filling, no polygon simplification), and (e) changes
6.96 % of current relation answers. **It is a label-source upgrade that needs no change to the
imagery.**

**What this does *not* authorise:** this audit task changes nothing by itself. The frozen artifacts
(36,926-component mapping, Task 3B relations, J1/J4 results) stay valid and reproducible; building a
vector-derived canonical instance dataset and BuildSpatialReason-v0.2 is the *next* task.

## 10. Reusable dataset-audit schema (§14)

`evaluation/dataset_audit_schema_whu_native_vector.json` fills schema v1 for the native vector so it
can be compared field-by-field with Task 6K's pseudo-instance instance: true instance identity
**true** (vs false), identity key `.shp` record order, 34,085 features, border-truncation 34.1 %,
merge/split structure, constructibility per level, geographic metadata (approximate lon/lat,
non-geographic split), storage, license `UNKNOWN`, source resolution `UNKNOWN`.

## 11. Reproduction

| Script | Artifact |
|---|---|
| `scripts/task6k1_vector_structure.py` | `task6k1_vector_structure.json` |
| `scripts/task6k1_georef_mapping.py` | `task6k1_whole_image_georef.json`, `task6k1_tile_mapping.json` |
| `scripts/task6k1_alignment_validation.py` | `task6k1_vector_raster_alignment.json` |
| `scripts/task6k1_vector_instances.py` | `task6k1_vector_instance_stats.json` + gitignored per-tile cache |
| `scripts/task6k1_vector_vs_pseudo_matching.py` | `task6k1_vector_vs_pseudo_matching.json` |
| `scripts/task6k1_relation_drift.py` | `task6k1_vector_vs_pseudo_relation_drift.json` |
| `scripts/task6k1_cross_6j.py` | `task6k1_task6j_vector_cross_analysis.json` |
| `scripts/task6k1_split_geographic_audit.py` | `task6k1_split_geographic_audit.json` |
| `scripts/task6k1_samples.py` | `evaluation/task6k1_samples/` (small panels) |
| `scripts/task6k1_verdict.py`, `scripts/task6k1_schema.py`, `scripts/task6k1_read_only_proof.py` | decision, schema instance, read-only proof |
| `scripts/task6k1_slim_artifacts.py` | final stage: moves per-tile rows into the gitignored cache so the committed JSON stays small |

Pipeline order: measurement scripts → `task6k1_verdict.py` → `task6k1_schema.py` →
`task6k1_read_only_proof.py` → `task6k1_slim_artifacts.py` (last). Nothing is lost: the per-tile rows
land in `artifacts/task6k1/*_per_tile.json` (gitignored) and are regenerated by re-running the
measurement scripts.

Library code: `buildreasonseg_mvp/shapefile_reader.py` (stdlib ESRI shapefile/DBF reader) and
`buildreasonseg_mvp/whu_vector_audit.py` (BigTIFF metadata, world files, tile grids, feature
rasterisation, merge/split metrics). Everything is deterministic and CPU-only; the whole fan-out is
read-only.
