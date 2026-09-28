# TO_DSH — Task 6L: WHU Native-Vector Canonical Dataset + BuildSpatialReason v0.2

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Predecessor verdict: Task 6K.1 → `MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`
>
> Goal: migrate the PRIMARY annotation source from semantic connected-component pseudo-instances to the validated native WHU East-Asia vector polygons, while preserving all historical artifacts as frozen baselines. Build a production-ready canonical vector-instance representation and a new reasoning dataset `BuildSpatialReason v0.2`.
>
> This is a **data migration / regeneration task only**. Do not train a proposal model, MLLM, SAM2, or final model.

## 0. User-facing language

All DSH narrative/UI output must be Chinese.

Code, file names, metric keys and canonical tokens may remain English.

## 1. Frozen evidence from Task 6K.1

Treat the following as already measured and immutable unless a direct implementation bug is found:

- `EA.shp`: 34,085 polygon records (`.shp == .shx == .dbf`);
- native vector ↔ raster mean IoU: 0.9505;
- tile mapping validated by pixel/window identity;
- 38,824 clipped native instances on the historical 4,038 positive tiles;
- 32,590 distinct vector features represented there;
- current pseudo-instances: 36,926;
- unmatched native instances vs pseudo at IoU 0.50: 5.24%;
- semantic-component merge rate: 1.32%;
- pseudo merge rate: 1.34%;
- vector→pseudo split rate: 0.13%;
- VECTOR↔PSEUDO weighted relation target-change rate: 6.96%;
- Task 6J failure remains proposal-dominated: 51/65 = 78.5% on clean single native buildings;
- current random val split is spatially interleaved with train; test is a separate raster.

Do not rerun Task 6K.1 except for small consistency assertions.

## 2. Read-only external sources

Original WHU archive — READ ONLY:

`C:\D\resources\Satellite dataset Ⅱ (East Asia)`

Historical YOLO dataset — READ ONLY:

`C:\D\resources\WHU_YOLO_dataset`

Legacy project — READ ONLY:

`C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment`

Canonical repo:

`C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`

No source mutation. No external download. No package installation.

## 3. Historical artifacts are frozen

Keep unchanged:
- `datasets/whu/`
- `datasets/build_spatial_reason/v0.1.1/`
- all Task 1–6K.1 evaluation artifacts;
- old YOLO baseline/provenance;
- current `BuildSpatialReason v0.1.1` artifact-fact block.

Do NOT silently rewrite old manifests to pretend they came from vector GT.

`v0.1.1` remains the pseudo-instance historical baseline.

# PART A — Canonical native-vector dataset

## 4. Dataset identity

Create a new canonical dataset namespace:

`datasets/whu_native_vector/`

Suggested version:

`v1.0`

Canonical name:

`WHU-EA-NativeVector`

Primary annotation truth:
- native `EA.shp` polygon geometry;
- `source_feature_id` = 1-based `.shp` record order within this exact archive;
- define a canonical reproducible UID as `(EA.shp SHA256, source_feature_id)` and record the shapefile hash in the manifest, because record order alone is only stable for this exact file/version;
- DBF fields are NOT identity because Task 6K.1 proved them degenerate.

A tile-clipped annotation identity must explicitly include:
- `source_feature_id`
- `tile_id`
- optional deterministic `tile_instance_id`

Never replace the stable source id with only a per-tile renumbering.

## 5. Cover the COMPLETE cropped archive, not only the historical 4,038 positive tiles

Task 6K found:
- train: 3,135
- train_no: 10,527
- test: 903
- test_no: 2,823
- total: 17,388 tiles

Task 6K.1 proved the whole-image grid capacity equals the cropped-tile count.

Therefore the canonical vector dataset must index **all 17,388 tiles**, including:
- truly empty tiles;
- `_no` tiles with small/partial building labels;
- positive historical tiles.

Do not discard `_no` tiles merely because the old semantic workflow did.

For every tile record include at least:
- canonical tile id;
- source whole raster: `train1 | train2 | test`;
- grid row/column;
- logical source image path relative to the external WHU root;
- logical source raster-label path;
- image size;
- native vector instances clipped to tile;
- instance count;
- empty/non-empty flag;
- old folder category (`train`, `train_no`, `test`, `test_no`);
- historical pseudo split if applicable (`train`, `val`, `test`, or null);
- source geospatial bounds if available;
- provenance version.

Committed metadata must NOT hard-code `C:\D\...` absolute paths.

Use a logical source root / CLI `--source-root` for runtime.

## 6. Per-instance schema

For each clipped native instance store:
- `source_feature_id`;
- `tile_instance_id`;
- polygon/multipolygon geometry in tile pixel coordinates;
- bbox xyxy;
- centroid;
- clipped area px;
- full source-feature area in the same projected/image-compatible units where safely computed;
- `touches_tile_border`;
- `visible_fraction` when computable as clipped/full feature area;
- ring/hole metadata;
- multipart flag;
- tiny-area flags (do not remove);
- source provenance.

Do not convert map-coordinate area into “ground m²”. The local `.tfw`/CRS-derived scale must be treated as authoritative for alignment only; do not reuse the public webpage's nominal 2.7 m GSD to convert pixel area unless that discrepancy is separately reconciled.

## 7. Preserve all native instances

Unlike the historical converter:
- NO `<50 contourArea` deletion;
- NO semantic connected-component merging;
- NO loss of holes via `RETR_EXTERNAL`;
- NO `approxPolyDP` simplification unless a separately documented derived export needs it.

The canonical ground truth must preserve native vector geometry as faithfully as tile clipping permits.

# PART B — Split design

## 8. Maintain TWO split views

### A. `legacy_compat_v1`

Purpose: fair comparison with old Task 1–6K.1 experiments.

For the historical 4,038 positive tiles:
- reuse the exact recovered old train/val/test stems;
- do not recreate from seed.

Other tiles may be `unassigned` in this view.

### B. `scene_disjoint_v1`

Purpose: cleaner future model development and paper evaluation.

Use whole-raster provenance:

- train = `train1`
- val = `train2`
- test = `test`

for all mapped cropped tiles.

Before freezing, prove:
- zero tile overlap;
- zero `source_feature_id` overlap between train/val/test, OR explicitly list and exclude rare boundary-crossing features/tiles until identity leakage is zero;
- zero RGB duplicate hash overlap where feasible;
- train/val/test query constructibility is non-zero for every intended program family.

If any program class has insufficient validation/test support, do not silently fall back to random splitting. Report and STOP for review.

## 9. Split interpretation

Record explicitly:
- `legacy_compat_v1` is for historical comparability only;
- `scene_disjoint_v1` is the new primary development/evaluation split;
- it provides scene/raster separation, NOT unseen-city or broad geographic generalization.

Do not claim cross-city generalization.

# PART C — Canonical adapter

## 10. Implement reusable dataset adapter

Create a reusable interface, e.g.:

```python
load_tile(tile_id, split_view=...)
list_instances(tile_id)
get_instance_geometry(tile_id, tile_instance_id)
get_source_feature_id(...)
iter_tiles(split, split_view=...)
```

It must be independent of YOLO and independent of the old pseudo-component format.

Provide an adapter layer that exposes the geometry fields expected by the existing Task 3B relation engine.

Do not rewrite the relation engine unless a genuine incompatibility is found.

# PART D — BuildSpatialReason v0.2

## 11. Dataset version

Create:

`datasets/build_spatial_reason/v0.2/`

Primary source:
`WHU-EA-NativeVector v1.0`

Use native vector instances as candidate truth.

Use `scene_disjoint_v1` as the primary split.

Keep `legacy_compat_v1` support for comparison, but not as the primary v0.2 split.

## 12. Program vocabulary

Keep the current **20 canonical program/query types** unchanged.

Reason:
- isolate annotation truth + split migration;
- retain Task 6J ProgramHead/executor comparability;
- avoid changing language semantics and instance truth in the same task.

No new relation types in Task 6L.

## 13. Relation semantics

Start from the frozen Task 3B semantics:
- left/right/above/below;
- area ranking;
- boundary-distance nearest;
- ambiguity/discard rules.

Candidate geometry now comes from native vector instances.

Do not use pseudo ids.

## 14. Visibility / truncation metadata

Compute/store:
- `touches_tile_border`;
- `visible_fraction` where robust;
- tiny-area flags.

Do NOT invent a complex new visibility threshold merely to improve counts.

Canonical GT must preserve all instances; filtering may happen only at reasoning-query eligibility time.

If eligibility must change because vector geometry exposes better visibility:
- make it explicit as `semantic_visibility_policy_version = 2.0`;
- document the reason;
- provide a compatibility audit against v0.1.1;
- do not tune on test.

## 15. Reasoning records

Each reasoning record must include enough provenance to audit the target:
- `tile_id`;
- source-feature / tile-instance metadata for reference and target;
- query type/program id;
- reasoning steps;
- instruction zh/en;
- target geometry reference;
- split view/version;
- dataset version;
- relation config version;
- visibility policy version.

Source/instance ids may exist in metadata/evaluation, but must not leak into the natural-language model input.

# PART E — v0.2 validation

## 16. Structural checks

Report:
- total tiles indexed;
- total native source features;
- total clipped tile instances;
- empty/non-empty tile counts;
- instances/tile distribution;
- per-split tile and instance counts;
- border-truncated rate;
- visible-fraction distribution;
- tiny-instance distribution;
- holes/multipart preservation.

## 17. Query-generation checks

For `BuildSpatialReason v0.2` report:
- total samples;
- by split;
- by L1/L2/L3;
- by all 20 query types;
- trivial/nontrivial L3;
- paired counterfactual availability;
- samples/image;
- discard reasons;
- ambiguity rate;
- reference/target border/tiny distributions.

Every intended program class must have non-zero val and test support.

## 18. v0.1.1 comparison

On the exact historical 4,038 positive tiles under `legacy_compat_v1` compare:
- common query count;
- target unchanged/changed;
- became valid/invalid;
- weighted answer change;
- by level/query type.

This should broadly reconcile with Task 6K.1's ~6.96% VECTOR↔PSEUDO drift.

Do not require exact equality if an explicit visibility-policy change explains the delta.

## 19. Acceptance gates

Accept only if:
- all 17,388 tiles have deterministic canonical records;
- vector provenance resolves to native `EA.shp`;
- stable source ids survive clipping;
- no cross-split source-feature leakage in `scene_disjoint_v1`;
- every intended program class is represented in val and test;
- generator/validator deterministic;
- no test data influences thresholds;
- random audit samples agree with native vector overlays;
- v0.1.1 comparison reconciles with Task 6K.1;
- artifact consistency checks pass.

Verdict:
- `VECTOR_DATASET_MIGRATION_PASS`
or
- `VECTOR_DATASET_MIGRATION_NEEDS_FIX`

# PART F — Task 6J bridge

## 20. Do NOT train yet

Do not retrain YOLO or a new proposal backbone.

Prepare only the reusable native-vector evaluation interface needed next:
- proposal recall vs native vector masks;
- selected-mask mIoU;
- proposal geometry extraction;
- structured executor candidate API.

Old Task 6J J0/J1/J2/J3 artifacts remain frozen.

Do not run J4 with an untrained/new proposal model.

# PART G — Storage / Git

## 21. Avoid duplicating imagery

Do not copy 17,388 TIFF images into Git.

Canonical metadata should reference logical paths under the external source root.

Large regenerable per-tile geometry caches may be gitignored under:
`artifacts/whu_native_vector/`

Do not stage raw shapefiles or source rasters. The official WHU page exposes the dataset for public/free download, but Task 6L must not assume a formal redistribution license that is not explicitly present in the local archive/official page; record provenance/citation and keep raw data external.

# PART H — Required artifacts

Create at minimum:

```text
datasets/whu_native_vector/v1.0/manifest.json
datasets/whu_native_vector/v1.0/statistics.json
datasets/whu_native_vector/v1.0/splits/legacy_compat_v1.json
datasets/whu_native_vector/v1.0/splits/scene_disjoint_v1.json

datasets/build_spatial_reason/v0.2/manifest.json
datasets/build_spatial_reason/v0.2/statistics.json

evaluation/task6l_vector_dataset_integrity.json
evaluation/task6l_scene_disjoint_split_audit.json
evaluation/task6l_build_spatial_reason_v0.2_quality.json
evaluation/task6l_v01_vs_v02_comparison.json
evaluation/task6l_artifact_index.json
evaluation/task6l_verdict.json

docs/task6l_vector_dataset_migration.md
```

Use JSONL/index files as needed.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`
- architecture decisions if warranted.

# PART I — Tests

## 22. Required tests

At minimum:

1. original WHU source read-only;
2. old YOLO dataset read-only;
3. legacy project read-only;
4. v0.1.1 frozen unchanged;
5. native source feature ids stable;
6. all 17,388 cropped tiles accounted for;
7. tile→whole-raster mapping deterministic;
8. vector clipping deterministic;
9. holes/multipart not silently dropped;
10. no `<50` global deletion in canonical GT;
11. empty tiles supported;
12. `legacy_compat_v1` exact historical stems;
13. `scene_disjoint_v1` train1/train2/test mapping correct;
14. zero tile overlap across scene-disjoint splits;
15. zero source-feature-id split leakage after any explicitly documented exclusions;
16. no test-driven threshold tuning;
17. relation engine consumes vector geometry;
18. all 20 program ids preserved;
19. ids absent from model-input instruction text;
20. zh/en semantic parity;
21. generator deterministic;
22. validator catches invalid target provenance;
23. every program has val/test support;
24. v0.1.1 comparison reconciles with Task 6K.1;
25. no model training;
26. no downloads/installations;
27. no GUI;
28. full artifact consistency.

Run:
`python -m pytest tests/ -q`

# PART J — Model choice / runtime

Recommended:
- DeepSeek V4.1 Flash + High.

If a genuinely difficult cross-file/geospatial bug appears:
- V4.1 Flash + Max.

Do not move to V4 Pro merely because the task is important.

# PART K — Git / Watt

Recommended commit:
`feat: migrate WHU to native vector instances`

Then a separate docs/handoff commit if needed.

Do not commit raw TIFFs, source shapefile, model weights, per-instance raster dumps, huge caches, or `.conda`.

Use established Watt ownership rules only if needed for push.

# PART L — Final handoff

`handoff/FROM_DSH.md` must include:

1. Verdict
2. Canonical Dataset Identity
3. Full 17,388-Tile Coverage
4. Native Instance Statistics
5. Split Design
6. Leakage Audit
7. Adapter/API
8. BuildSpatialReason v0.2
9. v0.2 Distribution
10. v0.1.1 vs v0.2 Comparison
11. Visibility/Truncation Policy
12. Task 6J Bridge Preparedness
13. Reproducibility
14. Tests
15. Git/Watt
16. Recommended Next Step

Final UI in Chinese must report at minimum:
- Task verdict;
- total indexed tiles;
- total clipped native instances / distinct source features;
- scene-disjoint split sizes;
- excluded leakage boundary cases if any;
- v0.2 sample counts by split/level;
- all 20 program support status;
- v0.1.1↔v0.2 target-change rate;
- acceptance checks;
- tests;
- commit/push.

# 23. STOP

After Task 6L STOP.

Do not automatically retrain YOLO, select a new proposal architecture, run Task 6J J4, upgrade Qwen to 4B, add `[REF]`/SRE/SCL, download another dataset, perform formal final-model training, or build GUI.

Wait for ChatGPT review.
