<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

# FROM_DSH — Task 6L Report: Native-Vector Canonical Dataset + BuildSpatialReason v0.2

> The ARTIFACT-FACTS block above still describes **v0.1.1**, which is frozen and unchanged; v0.2 is a
> separate dataset version with its own manifest (`datasets/build_spatial_reason/v0.2/manifest.json`).

_This file holds the Task 6L report; the Task 6K.1 report is preserved in git history and in
`docs/task6k1_whu_native_vector_groundtruth.md`._

Full design notes: `docs/task6l_vector_dataset_migration.md`, ADR-025.

## 1. Verdict

**`VECTOR_DATASET_MIGRATION_PASS`** — all 11 acceptance gates passed, 0 failed.

The primary instance truth is now the validated native WHU East-Asia vector map (Task 6K.1:
`MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`). The canonical dataset indexes **all 17,388** cropped tiles
with **41,186** clipped instances from **33,788** distinct `EA.shp` features, preserves native
geometry exactly (no `<50` deletion, no merging, no hole loss, no simplification; tiny instances
flagged but kept), provides two split views (`legacy_compat_v1` for historical comparability and the
new primary `scene_disjoint_v1` with zero leakage), and BuildSpatialReason **v0.2** (28,108 samples,
same 20 programs, same frozen relation config) changes **5.70 %** of targets (**5.45 %** weighted) on
the historical tiles — reconciling with Task 6K.1's 6.96 % VECTOR↔PSEUDO drift. No model training,
no downloads, no installations, no GUI; `datasets/whu/` and v0.1.1 stay frozen.

## 2. Canonical Dataset Identity

`datasets/whu_native_vector/v1.0/` — **`WHU-EA-NativeVector`** v1.0, schema 1.0.

* annotation truth: native `EA.shp` polygon geometry;
* `source_feature_id`: 1-based `.shp` record order in this exact archive;
* UID: **`(EA.shp SHA256, source_feature_id)`** — `shp_sha256 9bbd1e06d6cd…`, 4,999,904 bytes,
  recorded in the manifest;
* DBF fields are **not** identity (Task 6K.1 proved them degenerate);
* clipped identity: `source_feature_id` + `tile_id` + deterministic `tile_instance_id`
  (never a per-tile renumbering alone);
* `tile_id` = the archive's own tile name; `grid_id` = `{raster}:{row}:{col}`;
* committed metadata contains **no absolute path**; the external root is supplied via `--source-root`
  and records carry logical paths (`image_path_root: "whu_source_root"`).

## 3. Full 17,388-Tile Coverage

train 3,135 · train_no 10,527 · test 903 · test_no 2,823 = **17,388** tiles, each with a canonical
record, a unique `tile_id` and a unique `grid_id`: 11,909 empty and 5,479 non-empty. The `_no` tiles
the semantic workflow discarded are **included**, and empty tiles are first-class records.

## 4. Native Instance Statistics

| Metric | Value |
|---|---|
| native source features | 34,085 |
| distinct features represented | **33,788** |
| total clipped instances | **41,186** |
| instances/tile | mean 2.37 · median 0 · p90 8 · max **55** (uint8-safe) |
| instance area px | p5 116 · median 1,234 · p90 3,118 · max 132,014 |
| visible fraction | p5 0.069 · median 1.0 · mean 0.819 |
| border-truncated | 14,583 (**35.4 %**) |
| tiny (< 50 px, retained) | 1,235 (3.0 %) |
| instances with holes | 6 (5 source features carry holes) |
| multipart instances | 0 (the 5 multipart features' parts never share a tile) |
| overlap/rounding delta | 370 tiles, 2,966 px total, max 96 px |

Preservation guarantees are recorded in `statistics.json → preservation_guarantees` and asserted by
tests: no contour-area filter, no component merging, no `RETR_EXTERNAL` hole loss, no polygon
simplification.

## 5. Split Design

* **`legacy_compat_v1`** (comparability only): the recovered historical stems — train 2,508 · val 627
  · test 903 = the 4,038 positive tiles; the other 13,350 tiles are `unassigned`.
* **`scene_disjoint_v1`** (new primary): train = train1 (10,044 tiles, 15,721 instances) · val =
  train2 (3,618, 15,674) · test = test (3,726, 9,791).

`scene_disjoint_v1` gives **scene/raster separation only** — not unseen-city or broad geographic
generalization; no cross-city claim is made anywhere.

## 6. Leakage Audit

| Check | Result |
|---|---|
| tile overlap train∩val / train∩test / val∩test | 0 / 0 / 0 |
| source-feature overlap after exclusions | 0 / 0 / 0 |
| boundary-crossing features excluded | **0** (none needed) |
| RGB duplicate groups across splits | **0** (1,302 duplicate tiles, all intra-split and content-degenerate) |

## 7. Adapter / API

`buildreasonseg_mvp/native_vector_adapter.py`, independent of YOLO and of the pseudo-component
format: `load_tile`, `list_instances`, `get_instance_geometry`, `get_source_feature_id`,
`iter_tiles`, `label_map`, plus the relation-engine bridge (`image_record_for_reasoning`,
`candidate_set_for_tile`) and `validate_reasoning_record`. **`spatial_reasoning/*` was not modified.**

## 8. BuildSpatialReason v0.2

`datasets/build_spatial_reason/v0.2/` from `configs/build_spatial_reason_v0.2.yaml`: source
`WHU-EA-NativeVector v1.0`, primary split `scene_disjoint_v1`, **the same 20 programs**, the **same
frozen** `configs/spatial_relations_v1.yaml` (sha256 identical to v0.1.1's manifest), the frozen
v0.1.1 generator code reused unchanged, and visibility policy deliberately unchanged at **1.0** so the
delta isolates annotation truth + split. Every record carries `tile_id`, target/reference
`tile_instance_id` + `source_feature_id`, `target_geometry_ref`, `target_mask`, `split_view`,
`dataset_version`, `relation_config_version`, `semantic_visibility_policy_version`,
`generator_version`; ids never appear in the instruction text.

## 9. v0.2 Distribution

| | Value |
|---|---|
| total | **28,108** |
| by split | train 12,778 · val 9,111 · test 6,219 |
| by level | L1 19,769 · L2 5,323 · L3 3,016 |
| L2 | nearest 2,286 · direction 3,037 (invariant holds) |
| L3 | trivial 1,392 · nontrivial 1,624 |
| samples/tile | median 6 · mean 6.17 · p90 10 · max 12 |
| discards | `single_component_image` 12,830 · `semantic_target_ineligible` 13,602 · `no_direction_candidate` 11,301 · `semantic_ambiguous` 5,460 · others |

## 10. v0.1.1 vs v0.2 Comparison

On the identical 4,038 historical tiles under `legacy_compat_v1`: 25,229 v0.1.1 answers vs 24,370
v0.2 answers; **23,210** common `(tile_id, query_type)` groups; 21,886 unchanged and **1,324 changed**
(target-change rate **5.70 %**); 2,019 became invalid and 1,160 newly valid; **weighted answer-change
rate 5.45 %** (L1 7.61 %, L2 0.74 %, L3 1.61 %). Task 6K.1's VECTOR↔PSEUDO reference is 6.96 %, so the
result **reconciles**; the residual difference is quota/eligibility interaction, not a different
relation definition.

## 11. Visibility / Truncation Policy

Unchanged: `semantic_visibility_policy_version = "1.0"`. `touches_tile_border`, `visible_fraction`
and `tiny_area` are computed and stored per instance (35.4 % of instances are border-truncated, 3.0 %
are tiny, 10,902 sample targets are border-truncated and 1,243 tiny), but eligibility still follows
the frozen relation rules. A v2.0 policy is deliberately **not** claimed, because changing
eligibility in the same task as the annotation truth would confound the comparison; the metadata is
in place for a separately-audited policy later.

## 12. Task 6J Bridge Preparedness

`proposal_evaluation_interface` scores any proposal set against native instance masks (per-proposal
best IoU, native recall@0.5) and `candidate_set_for_tile` exposes native instances to the frozen
structured executor. Old J0/J1/J2/J3 artifacts remain frozen; **J4 was not run** and no proposal
model was trained, selected or downloaded.

## 13. Reproducibility

```text
python scripts/task6l_build_dataset.py          # canonical dataset + integrity + split audit
python scripts/task6l_build_reasoning_view.py   # reasoning view (gitignored cache)
python scripts/task6l_build_v0_2.py             # BuildSpatialReason v0.2
python scripts/task6l_compare_v01_v02.py        # section 18 comparison
python scripts/task6l_validate.py               # gates, verdict, artifact index
```

Determinism was verified by regenerating a 75-tile subset twice into a scratch directory and
comparing bytes (identical), and by an independent re-clip audit over a deterministic tile sample
(min IoU **1.0**, i.e. the committed label maps are exactly reproducible from `EA.shp`).

## 14. Tests

`python -m pytest tests/ -q` → **518 passed** (489 before Task 6L + 29 new checks). The new file
`tests/test_task6l_native_vector_dataset.py` covers all 28 required items: read-only sources (×3),
holes/multipart preservation, no global area deletion, empty-tile support, exact historical stems,
scene-disjoint mapping, zero tile overlap, zero feature leakage, no test-driven threshold tuning, the
relation engine consuming vector geometry, all 20 program ids, no id leakage into instructions, zh/en
parity, generator determinism, the provenance validator rejecting corrupted targets, val/test program
support, Task 6K.1 reconciliation, no training, no downloads/installations, no GUI, and full artifact
consistency (including per-file SHA256 against `task6l_artifact_index.json`).

## 15. Git / Watt

Commit `feat: migrate WHU to native vector instances` plus a `docs:` handoff commit. Not committed:
raw TIFFs, the source shapefile, per-instance raster dumps, per-tile geometry caches, the reasoning
view, weights or `.conda`. The committed migration is ~92 MB of derived metadata and records
(canonical `tiles/index.jsonl` 11.4 MB + `instances/index.jsonl` 17.4 MB + the v0.2 JSONL 62 MB),
comparable to the 39 MB v0.1.1 baseline it sits beside; the WHU archive itself is publicly
distributed by its authors but this project holds no explicit redistribution licence, so raw data
stays external and only derived metadata is committed (citation recorded in the docs). Watt was not
required for this task; established ownership rules were respected for the push.

## 16. Recommended Next Step

Use `scene_disjoint_v1` + v0.2 as the primary development/evaluation setting and give the Task 6J
bridge its proposal backbone: a training task that (a) trains or selects an instance-segmentation
proposal model whose instances align with native vector geometry, (b) runs J1/J4 against native
instance masks with the prepared interface, and (c) reports deltas against the frozen v0.1.1
baseline while never mixing the two dataset versions in one table. Two follow-ups are recorded but
not started: a separately-audited `semantic_visibility_policy_version = 2.0` study, and recovery of
real instance attributes (the shipped DBF is degenerate). Per the STOP section, nothing was
retrained, selected, downloaded, converted beyond this migration, or GUI-built; waiting for ChatGPT
review.
