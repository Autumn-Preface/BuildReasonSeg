# TO_DSH — Task 6K.1: Recover and Validate Native WHU East-Asia Vector Ground Truth

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Goal: determine whether the original WHU Satellite Dataset II (East Asia) archive already contains a native manually delineated building vector map that can replace our semantic→connected-component pseudo-instances **without changing the imagery dataset**.
>
> This task exists because Task 6K discovered:
>
> `C:\D\resources\Satellite dataset Ⅱ (East Asia)\2. The shape file of the whole images\EA.shp`
>
> with the associated `.shx/.dbf/.prj/...`, but Task 6K did not inspect or align those vector records.
>
> Do not accept Task 6K's `KEEP_WHU_AS_PRIMARY_FOR_NOW` as a final project-level dataset decision until this vector source is resolved.

## 0. User-facing language

All DSH narrative/UI output must be **Chinese**.

Code, paths and metric keys may remain English.

## 1. Read-only sources

### Original WHU archive — READ ONLY

```text
C:\D\resources\Satellite dataset Ⅱ (East Asia)
```

Important subtrees:

```text
1. The cropped image data and raster labels
2. The shape file of the whole images
3. The whole images
```

Vector files discovered by Task 6K:

```text
C:\D\resources\Satellite dataset Ⅱ (East Asia)\2. The shape file of the whole images\EA.shp
C:\D\resources\Satellite dataset Ⅱ (East Asia)\2. The shape file of the whole images\EA.shx
C:\D\resources\Satellite dataset Ⅱ (East Asia)\2. The shape file of the whole images\EA.dbf
C:\D\resources\Satellite dataset Ⅱ (East Asia)\2. The shape file of the whole images\EA.prj
```

Task 6K recorded:
- `EA.shp`: 4,999,904 bytes
- `EA.shx`: 272,780 bytes
- `EA.dbf`: 1,636,210 bytes
- `EA.prj`: 354 bytes

For a standard ESRI `.shx`, `(272780 - 100) / 8 = 34085` index records. Treat this only as a **strong hypothesis** until the local shapefile is parsed and validated.

### Historical converted YOLO dataset — READ ONLY

```text
C:\D\resources\WHU_YOLO_dataset
```

### Canonical repository

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
```

### Legacy project — READ ONLY

```text
C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment
```

No source mutation, no downloads, no retraining.

## 2. External facts already checked by ChatGPT

These are context, not substitutes for local validation:

1. The official WHU Building Dataset paper states the released dataset contains both raster labels and vector maps and can support building instance segmentation.
2. Published descriptions of Satellite Dataset II (East Asia) state that its vector building map was manually delineated in ArcGIS.
3. Published counts are inconsistent across versions/papers (e.g. 29,085 vs 34,085), so **the local archive is authoritative for this project**.
4. Task 6K's local `.shx` byte size is consistent with 34,085 standard shapefile index records.

Do not hard-code any external building count. Parse the local files.

# PART A — Native vector structure

## 3. Parse `EA.shp/.shx/.dbf/.prj`

First check existing packages:
- `shapefile` / pyshp;
- `fiona`;
- `geopandas`;
- GDAL/OGR CLI;
- `rasterio`;
- `tifffile`.

Do not install anything.

If no shapefile reader is available, implement the minimal read-only ESRI Shapefile parsing needed for polygon records using Python standard library. The `.shx` and `.dbf` headers can also be parsed directly.

Report:
- `.shp` file header validity;
- shape type;
- record count from `.shx`;
- record count from `.dbf`;
- deleted DBF rows if any;
- DBF fields;
- CRS/WKT from `.prj`;
- geometry-type distribution;
- polygon vs multipart polygon counts;
- null/empty/invalid records;
- ring counts;
- vertices per feature;
- bounding box of all features;
- feature area distribution in coordinate units;
- stable vector instance id = shapefile record index or another explicit stable id if present.

Do not call records “buildings” yet until spatial validation against labels/images succeeds.

# PART B — Whole-image georeferencing and tile reconstruction

## 4. Inspect whole-image files safely

Task 6K found very large TIFFs such as:

```text
train1.tif
train2.tif
test.tif
```

and `.tfw` files.

Do NOT load whole images into RAM.

Use, in preference order:
- rasterio metadata/window reads if already installed;
- tifffile metadata/memmap/windowed access if already installed;
- another existing read-only TIFF tool;
- only if necessary, a minimal metadata parser.

Do not disable safety and then decode the full multi-gigapixel images.

Report for each whole image:
- width/height;
- band count/dtype;
- affine/world-file transform;
- CRS if available;
- geographic/projected bounds;
- source file size.

## 5. Recover exact cropped-tile → whole-image mapping

This is critical.

Do not assume row-major naming without validation.

For every cropped source group:
- determine which whole image it belongs to;
- determine tile row/column/global pixel extent;
- determine geospatial extent from the `.tfw`/affine transform.

Possible evidence:
- filename/prefix/index convention;
- whole-image dimensions;
- 512×512 seamless crop ordering;
- pixel-content verification on deterministic sample windows.

Validate the recovered mapping on at least 100 deterministic tiles per source image by comparing the cropped RGB tile with the corresponding whole-image window:
- exact byte/pixel agreement if possible;
- otherwise quantify differences and explain why.

If exact mapping cannot be established, STOP before claiming vector-derived instance labels.

# PART C — Validate that EA.shp is the East-Asia building footprint map

## 6. Raster/vector spatial validation

Overlay/clip vector features against the whole-image raster-label coordinate system.

Use vector geometry + georeferencing to rasterize deterministic windows/tiles.

Compare vector-union rasterization with the original binary semantic labels.

Report:
- tile-level IoU/Dice/precision/recall;
- global foreground agreement using streaming/windowed aggregation;
- vector features with no raster-label overlap;
- raster foreground with no vector overlap;
- systematic offset if present;
- best simple alignment correction ONLY as a diagnostic (do not mutate source geometry);
- train1/train2/test separately.

The spatial agreement must clearly establish whether the shapefile represents the same building annotation source as the raster labels.

Use representative visual overlays under:
`evaluation/task6k1_samples/`.

## 7. Resolve local feature count

Report the exact local count.

If it is 34,085, record that this matches the `.shx` structural expectation.

If it is another number, report the actual number and do not force agreement with literature.

# PART D — Build a temporary canonical true-instance view

## 8. Clip vector features to 512×512 tiles

Without modifying the original data, create a temporary/audit-only canonical instance view for the 4,038 positive tiles currently used by BuildReasonSeg.

For each tile:
- preserve stable source vector feature id;
- clip polygon to tile boundary;
- preserve whether the feature is truncated by tile border;
- support multipart geometry correctly;
- compute raster mask/bbox/centroid/area;
- distinguish:
  - one physical/vector instance appearing in multiple tiles;
  - multiple vector instances touching in raster space.

Write audit metadata under gitignored artifacts, with only summary JSON committed.

Do NOT create a full new dataset yet.

# PART E — True-vector vs semantic connected-component comparison

## 9. Quantify the exact pseudo-instance error

This replaces Task 6K's heuristic merge-risk analysis with vector-grounded evidence.

For the 4,038 positive tiles compare:
- vector instances;
- raw 8-connected semantic components;
- current 36,926 pseudo-instances.

Measure:

### Counts
- total unique source vector features represented;
- total clipped vector instances across tiles;
- vector instances/tile;
- pseudo-instances/tile.

### Many-vector → one-semantic-component merges
For each semantic connected component:
- number of overlapping vector feature ids;
- count/rate with 1;
- count/rate with 2;
- 3+;
- max.

This is the primary measure of touching-building merge.

### One-vector → multiple semantic/pseudo components
Measure splits caused by:
- rasterization/disconnection;
- tile boundaries;
- other causes.

### Pseudo ↔ vector matching
At IoU thresholds 0.25 / 0.50 / 0.75:
- one-to-one match rate;
- unmatched vector instances;
- unmatched pseudo-instances;
- merged pseudo-instances;
- split pseudo-instances.

Break down by:
- small area;
- border truncated;
- dense tile;
- train/val/test.

# PART F — Re-evaluate spatial reasoning semantics using native vector instances

## 10. VECTOR vs current PSEUDO relation drift

This is more important than Task 6K's RAW-semantic vs CONVERTED comparison.

Use:
- VECTOR candidate set = clipped native vector building instances;
- PSEUDO candidate set = current BuildReasonSeg pseudo-instances.

Use the same frozen Task 3B relation semantics where meaningful.

Run all 20 canonical programs.

Report:
- constructible queries in VECTOR;
- constructible queries in PSEUDO;
- comparable;
- target unchanged;
- target changed;
- became invalid;
- became newly valid;
- by L1/L2/L3;
- by query type.

When matching target identity across representations, use stable source-vector ids and geometric overlap; do not use annotation ids as inference inputs.

Attribute differences:
- multiple true vector instances merged into one pseudo-instance;
- true vector instance split;
- `<50` removal;
- border clipping;
- geometry approximation;
- eligibility/ranking consequences.

This result, not Task 6K's 4.33% RAW-vs-CONVERTED number, should drive the final annotation decision if native vector GT is confirmed.

# PART G — Revisit Task 6J against vector truth

## 11. J1 proposal diagnosis

For the fixed Task 6J 120 records / 20 pairs:
- map current pseudo target and YOLO proposals to native vector instances;
- identify cases where current pseudo target itself merges/splits true vector instances;
- measure YOLO proposal recall against vector-instance targets where a clean correspondence exists;
- classify failures:
  - proposal-model error against clean vector target;
  - pseudo-label definition error;
  - representation mismatch;
  - ambiguous/inseparable.

Do not retrain YOLO in this task.

# PART H — Split / geographic structure

## 12. Use recovered tile coordinates

If tile→whole-image mapping is successfully recovered:
- quantify train/val spatial adjacency;
- nearest train-tile distance for each val tile in tile-grid units;
- same whole-image source rate;
- identify whether val is spatially interleaved with train;
- quantify test separation by source image/region.

State precisely what the split can and cannot support as a paper claim.

# PART I — Decision

## 13. Final verdict: exactly one

### `MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`
Use if:
- `EA.shp` is confirmed as the native manually delineated building footprint map;
- tile mapping is reliable;
- vector instances can be aligned with the current imagery/raster labels;
- true instance IDs/polygons materially improve the annotation basis.

This means:
- KEEP WHU imagery;
- STOP using semantic connected components as the primary instance truth;
- next task builds a vector-derived canonical instance dataset and BuildSpatialReason-v0.2.

### `KEEP_CURRENT_PSEUDO_INSTANCES`
Use only if:
- shapefile is not building footprints, or
- it cannot be reliably aligned, or
- it provides no meaningful instance advantage.

### `REPLACE_WHU_WITH_NEW_DATASET`
Use only if:
- native vector GT is unusable AND the semantic/pseudo path is structurally inadequate for final goals.

### `INSUFFICIENT_EVIDENCE`
Use if local files cannot be parsed/aligned.

Also report:
- `keep_whu_imagery: true/false`
- `keep_historical_pseudo_baseline: true/false`

# PART J — Required artifacts

Create:

```text
evaluation/task6k1_vector_structure.json
evaluation/task6k1_whole_image_georef.json
evaluation/task6k1_tile_mapping.json
evaluation/task6k1_vector_raster_alignment.json
evaluation/task6k1_vector_instance_stats.json
evaluation/task6k1_vector_vs_pseudo_matching.json
evaluation/task6k1_vector_vs_pseudo_relation_drift.json
evaluation/task6k1_task6j_vector_cross_analysis.json
evaluation/task6k1_split_geographic_audit.json
evaluation/task6k1_dataset_decision.json
docs/task6k1_whu_native_vector_groundtruth.md
```

Small representative overlays:
`evaluation/task6k1_samples/`

Large per-tile caches must be gitignored.

# PART K — Tests

## 14. Required checks

At minimum:
1. no writes under original WHU root;
2. no writes under `WHU_YOLO_dataset`;
3. no writes under legacy project;
4. no package installation;
5. shapefile record count independently agrees between `.shx` and `.dbf`;
6. geometry parsing is deterministic;
7. CRS parsed from local `.prj`;
8. whole TIFFs never fully decoded into RAM;
9. tile mapping validated by image-window evidence;
10. vector/raster alignment measured, not assumed;
11. stable source vector ids preserved;
12. tile clipping preserves border-truncation metadata;
13. many-vector→one-component merge metric correct on synthetic test;
14. one-vector→many-component split metric correct;
15. no vector id/GT target leakage into inference execution;
16. relation drift uses frozen Task 3B semantics;
17. Task 6J artifacts are read-only/frozen;
18. no model training;
19. no dataset regeneration;
20. final verdict is one of the four allowed values.

Run:
`python -m pytest tests/ -q`

# PART L — Runtime / dependencies

No model training.
No external download.
No new Conda env.
No package install.

Use existing installed readers when available.

If alignment becomes difficult:
- keep DeepSeek V4.1 Flash;
- it is acceptable to increase reasoning effort from High → Max;
- do not switch to V4 Pro merely because the task is important.

# PART M — Git / Watt

Do not stage source shapefiles, whole TIFFs, cropped imagery, generated full vector tiles, weights, or caches.

Commit scripts/tests/small JSON/docs/handoff and a small number of overlays only.

Recommended commit:
`audit: validate WHU native vector ground truth`

Use established Watt ownership rules only if needed for final push.

# PART N — Handoff

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

The handoff must explicitly state:
1. exact local vector record count;
2. whether the shapefile is confirmed building footprints;
3. vector↔raster alignment quality;
4. tile mapping confidence;
5. exact true-vector merge/split rates against pseudo-instances;
6. VECTOR↔PSEUDO relation drift;
7. revised interpretation of Task 6K;
8. revised Task 6J attribution;
9. split/geographic result;
10. final four-way verdict;
11. tests;
12. commit/push.

# 15. STOP

After Task 6K.1 STOP.

Do not automatically regenerate BuildSpatialReason, convert all vector labels into production form, retrain YOLO, train a new proposal backbone, switch Qwen to 4B, download SpaceNet/WHU-Mix, add `[REF]`/SRE/SCL, run formal training, or build GUI.

Wait for ChatGPT review.
