# Task 6L — WHU Native-Vector Canonical Dataset + BuildSpatialReason v0.2

> Task: `handoff/TO_DSH.md` · Verdict: **`VECTOR_DATASET_MIGRATION_PASS`**
> Predecessor: Task 6K.1 → `MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES` · ADR-025
> Tests: `tests/test_task6l_native_vector_dataset.py` (28 checks) ·
> Evidence: `evaluation/task6l_*.json`, `datasets/whu_native_vector/v1.0/`,
> `datasets/build_spatial_reason/v0.2/`

This task migrates the PRIMARY annotation source from semantic connected-component pseudo-instances
to the validated native WHU East-Asia vector polygons, while keeping every historical artifact
(`datasets/whu/`, `datasets/build_spatial_reason/v0.1.1/`, Task 1–6K.1 evaluations) frozen.

No model was trained, no proposal architecture was selected, no Task 6J J4 was run, no dataset was
downloaded and nothing was installed.

## 1. Canonical dataset identity

`datasets/whu_native_vector/v1.0/` — canonical name **`WHU-EA-NativeVector`**.

| Element | Value |
|---|---|
| Annotation truth | native `EA.shp` polygon geometry |
| `source_feature_id` | 1-based `.shp` record order in this exact archive |
| UID | `(EA.shp SHA256, source_feature_id)` — the hash is recorded in the manifest |
| DBF fields | **never identity** (Task 6K.1 proved them degenerate) |
| Clipped identity | `source_feature_id` + `tile_id` + deterministic `tile_instance_id` |
| `tile_id` | the archive's own tile name (`1_0`, `0`, …), unique across the cropped archive |
| `grid_id` | `{raster}:{row}:{col}` — the whole-raster coordinate identity |

Committed metadata stores **no absolute path**: the external root is supplied at runtime
(`--source-root`) and records carry logical paths relative to it (`image_path_root:
"whu_source_root"`).

## 2. Full 17,388-tile coverage

| Category | Tiles | Raster |
|---|---|---|
| train | 3,135 | train1 / train2 |
| train_no | 10,527 | train1 / train2 |
| test | 903 | test |
| test_no | 2,823 | test |
| **total** | **17,388** | |

Every tile has a canonical record (`tiles/index.jsonl`, 17,388 lines), a unique `tile_id` and a
unique `grid_id`; **empty tiles and `_no` tiles are kept** (11,909 empty, 5,479 non-empty). The old
semantic workflow discarded the `_no` tiles; this dataset does not.

## 3. Native instance statistics

| Metric | Value |
|---|---|
| native source features | 34,085 |
| distinct features represented in the corpus | **33,788** |
| total clipped tile instances | **41,186** |
| instances per tile | mean 2.37, median 0, p90 8, max **55** (uint8 label map safe) |
| instance area px | min 1, p5 116, median 1,234, p90 3,118, max 132,014 |
| visible fraction (clipped / full) | p5 0.069, median 1.0, mean 0.819 |
| border-truncated instances | 14,583 (**35.4 %**) |
| tiny instances (< 50 px, **retained**) | 1,235 (3.0 %) |
| instances with holes | 6 (source has 5 features with holes) |
| multipart instances | 0 (the 5 multipart source features have parts that never share a tile) |
| instance overlap / boundary rounding | 370 tiles, 2,966 px total, max 96 px |

**Geometry preservation guarantees (per instance):** no `contourArea` filter, no connected-component
merging, no `RETR_EXTERNAL` hole loss, no `approxPolyDP` simplification. Holes and multipart parts
are carried as explicit `outer`/`hole` rings in tile pixel coordinates; tiny instances are flagged
but kept, so filtering can only happen later, at query-eligibility time.

The per-instance schema is documented in `instances/schema.json`; per-instance scalars for the whole
corpus are in `instances/index.jsonl`; full pixel-space rings are in
`instances/sample_geometry.jsonl` (a deterministic sample) and in the gitignored per-tile cache
`artifacts/whu_native_vector/instances/<tile_id>.npz`, which is regenerated deterministically by
`scripts/task6l_build_dataset.py`.

## 4. Split design (two views)

### `legacy_compat_v1` — historical comparability only

Recovered from the existing converted folders (never regenerated from a seed): train 2,508 · val 627
· test 903 = the 4,038 historical positive tiles; the remaining 13,350 tiles are `unassigned`.

### `scene_disjoint_v1` — the new primary split

| Split | Raster | Tiles | Instances |
|---|---|---|---|
| train | train1 | 10,044 | 15,721 |
| val | train2 | 3,618 | 15,674 |
| test | test | 3,726 | 9,791 |

**Leakage audit (all clean):**

| Check | Result |
|---|---|
| tile overlap train∩val / train∩test / val∩test | **0 / 0 / 0** |
| source-feature overlap after documented exclusions | **0 / 0 / 0** |
| boundary-crossing features needing exclusion | **0** |
| RGB duplicate groups across splits | **0** (1,302 duplicate tiles exist, all within a split and all content-degenerate) |

**Interpretation:** `scene_disjoint_v1` provides **scene/raster separation**, not unseen-city or
broad geographic generalization. No cross-city claim may be made from it.

## 5. Adapter / API

`buildreasonseg_mvp/native_vector_adapter.py` is independent of YOLO and of the old pseudo-component
format:

```python
dataset = NativeVectorDataset()
dataset.load_tile(tile_id, split_view=...)        # TileView metadata
dataset.list_instances(tile_id)                   # per-instance scalars
dataset.get_instance_geometry(tile_id, tile_instance_id)   # pixel-space rings
dataset.get_source_feature_id(tile_id, tile_instance_id)
dataset.iter_tiles(split, split_view=...)
dataset.label_map(tile_id)                        # uint8 instance-index map
```

The relation-engine bridge builds the exact record schema the frozen Task 3B stack already consumes
(`image_record_for_reasoning`, `candidate_set_for_tile`) and a provenance validator
(`validate_reasoning_record`). **The relation engine itself was not modified** — no incompatibility
was found.

## 6. BuildSpatialReason v0.2

`datasets/build_spatial_reason/v0.2/` — generated by `scripts/task6l_build_v0_2.py` from
`configs/build_spatial_reason_v0.2.yaml`.

* primary source: **WHU-EA-NativeVector v1.0** (`source_component_representation_version:
  whu-native-vector-v1.0`);
* primary split: **`scene_disjoint_v1`** (`legacy_compat_v1` retained for comparison);
* program vocabulary: **the same 20 canonical query types** as v0.1.1 (unchanged, by design);
* relation semantics: the **same frozen** `configs/spatial_relations_v1.yaml`
  (sha256 identical to v0.1.1's manifest entry);
* generator code: the **frozen v0.1.1 annotator/templates/relations**, reused unchanged
  (`generator_version: v0.1.1`, `generator_config` v0.2);
* **visibility policy stays `1.0`**: Task 6L section 14 permits a v2.0 policy when vector geometry
  exposes better visibility, but this migration deliberately does not change eligibility, so the
  v0.1.1↔v0.2 delta isolates annotation truth + split. `touches_tile_border`, `visible_fraction` and
  `tiny_area` are stored per instance for a future, separately-audited policy.

Every record carries full provenance: `tile_id`, target/reference `tile_instance_id` +
`source_feature_id`, `target_geometry_ref` (dataset, version, cache path), `target_mask`,
`split_view`, `dataset_version`, `relation_config_version`,
`semantic_visibility_policy_version`, `generator_version`. **Ids never enter the instruction text.**

## 7. v0.2 distribution

| | Value |
|---|---|
| total samples | **28,108** |
| by split | train 12,778 · val 9,111 · test 6,219 |
| by level | L1 19,769 · L2 5,323 · L3 3,016 |
| L2 decomposition | nearest 2,286 · direction 3,037 (invariant holds) |
| L3 | trivial 1,392 · nontrivial 1,624 |
| samples per tile | min 1, median 6, mean 6.17, p90 10, max 12 |
| tiles considered | 17,388 (all canonical tiles) |
| top discard reasons | `single_component_image` 12,830 · `semantic_target_ineligible` 13,602 · `no_direction_candidate` 11,301 · `semantic_ambiguous` 5,460 · `multiple_direction_candidates` 4,024 |
| target border/tiny | 10,902 border · 1,243 tiny (of 28,108) |
| program support | **all 20 programs present in train, val AND test** (smallest val count 78: `smallest_to_below`; smallest test count 34: `smallest_to_below`) |

## 8. v0.1.1 ↔ v0.2 comparison (`legacy_compat_v1`, the same 4,038 tiles)

| Metric | Value |
|---|---|
| v0.1.1 answers / v0.2 answers | 25,229 / 24,370 |
| common `(tile_id, query_type)` groups | **23,210** |
| target unchanged (mask IoU ≥ 0.5) | **21,886** |
| target changed | **1,324** → target-change rate **5.70 %** |
| became invalid in v0.2 / newly valid | 2,019 / 1,160 |
| **weighted current-query answer-change rate** | **5.45 %** |
| by level | L1 7.61 % · L2 0.74 % · L3 1.61 % |
| Task 6K.1 reference (VECTOR↔PSEUDO) | 6.96 % → **reconciles** |

The residual 1.5-point difference from Task 6K.1 is explained by dataset-level effects that the
candidate-set comparison does not have: quota selection, per-query eligibility and the
`single_component_image` discard interacting differently with the new instance structure. The
direction and magnitude agree.

## 9. Acceptance gates (Task 6L section 19)

| Gate | Result |
|---|---|
| all 17,388 tiles have canonical records | PASS |
| vector provenance resolves to native `EA.shp` (SHA256 recomputed) | PASS |
| stable source ids survive clipping (0 instances without id) | PASS |
| no cross-split source-feature leakage in `scene_disjoint_v1` | PASS |
| every intended program represented in val and test | PASS |
| generator/validator deterministic (byte-identical reruns) | PASS |
| no test data influences thresholds (relation config hash identical) | PASS |
| random audit samples agree with native-vector overlays (min IoU **1.0**) | PASS |
| v0.1.1 comparison reconciles with Task 6K.1 | PASS |
| artifact consistency | PASS |
| v0.1.1 frozen unchanged | PASS |

**Verdict: `VECTOR_DATASET_MIGRATION_PASS`.**

## 10. Task 6J bridge (prepared, not trained)

`buildreasonseg_mvp/native_vector_adapter.py::proposal_evaluation_interface` scores any proposal set
against native instance masks and returns per-proposal best-IoU plus native recall@0.5, and
`candidate_set_for_tile` exposes native instances to the frozen structured executor. The old Task 6J
J0/J1/J2/J3 artifacts remain frozen and **J4 was not run** with an untrained or new proposal model.

## 11. Storage / Git

Committed: canonical manifest/statistics/index/splits/schema/sample geometry, the v0.2 dataset, the
evaluation artifacts, scripts, tests and this document. **Not committed:** raw TIFFs, the source
shapefile, per-instance raster dumps, per-tile geometry caches or the reasoning view. Those live
under the gitignored `artifacts/whu_native_vector/` and are regenerated deterministically:

```text
python scripts/task6l_build_dataset.py            # canonical instances (cache)
python scripts/task6l_build_reasoning_view.py     # reasoning view (cache)
python scripts/task6l_build_v0_2.py               # BuildSpatialReason v0.2
python scripts/task6l_compare_v01_v02.py          # section 18 comparison
python scripts/task6l_validate.py                 # gates, verdict, artifact index
```

The WHU archive is publicly distributed by its authors, but this project holds **no explicit
redistribution licence**; raw data therefore stays external and only derived metadata is committed.
Citation: WHU Building Dataset, "Satellite dataset II (East Asia)" subset (Ji, Wei & Lu, 2018).
