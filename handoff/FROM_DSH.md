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

# FROM_DSH — Task 6K.1 Report: Native WHU East-Asia Vector Ground Truth

_This file holds the Task 6K.1 report; the Task 6K report is preserved in git history and in
`docs/task6k_whu_source_pseudoinstance_audit.md`._

Full design notes: `docs/task6k1_whu_native_vector_groundtruth.md`, ADR-024.

## 1. Verdict

**`MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`** — `keep_whu_imagery: true`,
`keep_historical_pseudo_baseline: true`.

The original archive ships a genuine manually delineated building vector map. It parses to **34,085**
polygons (identical count in `.shp`, `.shx` and `.dbf`), the tile→whole-image mapping is recovered
**and validated** (RGB windows pixel-identical, labels exact, grid capacity == cropped tile count for
all three rasters), and the vector reproduces the raster labels at **mean IoU 0.9505** (4,036/4,038
tiles ≥ 0.90). Measured against the pseudo-instance view it changes **6.96 %** of current relation
answers (weighted; L1 7.9 %, L2 3.4 %, L3 5.0 %) — above the declared 5 % materiality bar — and
**5.24 %** of native buildings are entirely absent from the pseudo view. The Task 6K hypothesis that
touching buildings merge into pseudo-instances is real but small: **1.32 %** of semantic components
and **1.34 %** of pseudo-instances contain ≥ 2 native buildings. Read-only throughout: no source,
converted or legacy file was modified (verified empirically), no model trained, nothing installed,
no dataset regenerated.

## 2. Exact local vector record count

**34,085 polygon features**, corroborated three ways — the `.shp` record iteration, the `.shx` index
(34,085 × 8 B + 100 B header = 272,780 B exactly) and the `.dbf` header (34,085 records, 0 deleted
rows) — with 0 unclosed rings, 5 features carrying interior holes, 5 multipart features and a median
of 5 vertices per feature. The published 29,085/34,085 figures were treated as context only.

**Two findings change how the file may be used.** (1) The **DBF attribute table is degenerate**: all
three fields are constant over all 34,085 records (`OBJECTID = 27`,
`Shape_Leng = 62.8055137405`, `Shape_Area = 211.751772668`), so it provides neither identity nor
usable area/length — identity in this audit is the **`.shp` record order**. (2) The CRS is
`WGS_1984_World_Mercator`, so map-unit areas must never be quoted as ground m².

## 3. Is the shapefile confirmed building footprints?

**Yes.** Shape type 5 (Polygon), 0 self-intersections in a bounded segment-intersection probe over
171 sampled features, and — decisively — the clipped vector reproduces the raster semantic labels:
mean IoU **0.9505** over all 4,038 positive tiles, **4,036/4,038 ≥ 0.90**, **0 below 0.50**, with the
best alignment offset **(0,0) in 240/240** sampled tiles (mean gain ≈ 0, i.e. no systematic shift).
Empty-tile behaviour agrees as well (277/300 sampled empty tiles are empty on both sides).

## 4. Vector ↔ raster alignment quality

Excellent but not perfect, in the expected way: the residual ~5 % IoU is boundary discretisation
between polygon rasterisation and the original label raster. No spatial offset, no scale error, no
rotation. Tile-level agreement is uniform across splits.

## 5. Tile mapping confidence

**Validated, not inferred.** The whole-area rasters are **BigTIFF** (magic 43) and were read with a
standard-library IFD parser (metadata only); pixel access used small PIL windowed crops, never a
full decode. Only full 512×512 windows were cropped, so `columns = floor(width/512)`:
train1 186×54 = 10,044 (cropped 10,044, diff **0**), train2 67×54 = 3,618 (diff **0**),
test 69×54 = 3,726 (diff **0**). The `.tfw` world files give a 0.33956958 m pixel and origins that
tile the whole vector extent (train1 → test → train2, ~1 m overlap), and `label/train1.tif` is one
pixel wider than its image raster (95,522 vs 95,521) — recorded, harmless for the tile grid.

## 6. Exact true-vector merge/split rates against pseudo-instances

Corpus: 4,038 tiles, 38,824 clipped native instances (**32,590 distinct features**), 36,926
pseudo-instances, 38,309 raw 8-connected semantic components.

| Metric | Value |
|---|---|
| native instance ↔ pseudo match at IoU 0.50 | 36,791 pairs = **94.76 %** of native instances |
| unmatched native instances (IoU 0.50) | **2,033 = 5.24 %** |
| unmatched pseudo-instances (IoU 0.50) | 135 = 0.37 % |
| **semantic components containing ≥ 2 native buildings (PRIMARY)** | **1.32 %** (471 with 2, 33 with 3+, max 4) |
| pseudo-instances containing ≥ 2 native buildings (containment ≥ 0.80) | **1.34 %** (496) |
| native buildings inside such a merged pseudo-instance | **2.65 %** |
| native buildings spanning ≥ 2 pseudo-instances (split) | **0.13 %** (51: 46 rasterisation/disconnection, 5 tile boundary) |

Breakdowns at IoU 0.50: border-truncated instances 11.8 % unmatched (vs 5.2 % overall), dense tiles
5.5 % (vs 4.7 % sparse), and **native buildings below 50 px are unmatched 100 % of the time
(1,039/1,039)** — the historical `<50 contourArea` filter is the single identifiable cause. The
unmatched rate is flat across splits (train 5.4 %, val 5.1 %, test 5.0 %), so this is a conversion
property, not a split artefact.

## 7. VECTOR ↔ PSEUDO relation drift

All 20 canonical programs under the frozen Task 3B semantics, targets compared by mask overlap (ids
are never inference inputs):

* tiles with any relation-answer change: **1,596/4,038 = 39.5 %**;
* **weighted current-query target-change rate 6.96 %** (Task 6K's RAW-vs-CONVERTED figure was 4.33 %);
* L1 **7.87 %**, L2 3.38 %, L3 5.03 %; per-program extremes `leftmost` 9.4 %, `bottommost` 9.2 %,
  `topmost` 9.0 %, `rightmost` 8.8 %; `smallest_to_left_of` 0.4 %;
* native buildings without a pseudo counterpart: 2,033 (5.24 %);
* attribution of the 2,402 changed program-tile pairs: `eligibility_ranking_consequences` 1,110,
  `below_50_removal` 782, `border_clipping` 373, `multiple_vector_merged_into_one_pseudo` 75,
  `geometry_approximation` 27, `one_vector_to_multiple_pseudo` 18,
  `vector_target_missing_in_pseudo` 17.

The two largest classes are ranking/nearest consequences of the different candidate sets and the
`<50` removal of small native buildings — i.e. **the conversion, not touching-building merging,
drives the drift.**

## 8. Revised interpretation of Task 6K

* Still correct: the semantic-raster → polygon conversion is faithfully reproduced (YOLO re-emulation
  IoU 1.00000), the conversion loss decomposition stands (1,383 contours removed = 0.058 % of
  foreground, hole filling +0.021 %, approximation IoU 0.99955), and the imagery corpus is untouched.
* **Revised:** Task 6K concluded "keep WHU as primary" while treating the semantic raster as the only
  label source. The archive does contain a native vector map with **true instance identity**, so the
  pseudo-instance limitation is only 1.3 % merging; the real, larger deviation is **5.24 % of native
  buildings missing from the pseudo view**. The annotation decision should be driven by the
  **6.96 %** VECTOR-vs-PSEUDO number, not by Task 6K's 4.33 % RAW-vs-CONVERTED number.
* `legacy_baseline_role` stays `keep`: the historical pseudo-instance baseline remains frozen,
  reproducible and comparable; it simply stops being the primary instance truth.

## 9. Revised Task 6J attribution

65/120 J1 failures against native truth:

| Category | Count | Share |
|---|---|---|
| proposal-model error against a clean single native building | **51** | **78.5 %** |
| representation mismatch (single native building, no proposal ≥ 0.50) | 13 | 20.0 % |
| pseudo-label definition error (merged target) | **0** | **0 %** |
| ambiguous / inseparable | 1 | 1.5 % |

YOLO proposal recall on the failures that have a clean vector target: **79.7 %**. **No** J1 failure is
explained by a merged pseudo-instance target, so Task 6J/6K's proposal-model conclusion is confirmed
on native ground truth.

## 10. Split / geographic result

With the mapping recovered the split becomes a per-tile geometric fact: **100 %** of the 627 val tiles
are directly adjacent to effective-training tiles (nearest training tile at distance **1** for all
627; mean 6.91 of 8 neighbours), val spans both train rasters (train1 333, train2 294) and 100 % of
val tiles share a raster with train, while the test scene is a separate raster with **0 %** adjacency.
Extents: train+val ≈ 56.2 km × 9.4 km, test ≈ 12.0 km × 9.4 km. **Supported claim:** random-tile
generalisation within the same two scenes plus one spatially disjoint test scene. **Unsupported:**
any geographic/unseen-city generalisation claim.

## 11. Tests

`python -m pytest tests/ -q` → **489 passed** (468 before Task 6K.1 + the 21 new checks). The new file
`tests/test_task6k1_whu_vector_audit.py` (21 checks) covers all 20 required items: no writes under the
original WHU root / converted dataset / legacy project (plus the empirical
`evaluation/task6k1_read_only_proof.json` check), no package installation, `.shx`/`.dbf` record-count
agreement, deterministic geometry parsing, CRS parsed from the local `.prj`, no full raster decode,
tile mapping validated by image-window evidence, measured (not assumed) vector/raster alignment,
stable source ids, border-truncation metadata, synthetic correct merge and split metrics, no GT
target leakage into inference execution, frozen Task 3B semantics, frozen Task 6J artifacts, no model
training, no dataset regeneration, small tracked overlays, and a final verdict restricted to the four
allowed values.

Metrics are in the artifacts, not the tests: the synthetic merge/split checks assert the *metric
definitions* (`count_features_per_component`, `count_targets_per_feature` in
`buildreasonseg_mvp/whu_vector_audit.py`), and the artifact checks assert the measured values.

## 12. Git / Watt

Task commit `audit: validate WHU native vector ground truth` plus the `docs:` handoff commit. No
source shapefiles, whole TIFFs, cropped imagery, generated per-tile instance caches, weights or
Conda state staged; only scripts, tests, small JSON artifacts, the small overlay panels and docs.
Watt was not required for this task and the established ownership rules were respected for the push.

## 13. Recommended next step (not started)

Build the vector-derived canonical instance dataset and BuildSpatialReason-v0.2 on top of the frozen
imagery: keep the imagery and tiling, replace the instance labels with the native vector polygons
(identity = `.shp` record order), re-derive the relation labels, and re-run the J1/J4 gates with the
measured **6.96 %** answer-change budget as the reference delta. Two open items are recorded but not
started: recovering the true identity attributes (the shipped DBF is degenerate, so a documented
external join or a fresh ArcGIS export would be needed) and confirming the published count/licence
externally. Per the task's STOP section, nothing was regenerated, retrained, downloaded, converted to
production form or GUI-built; waiting for ChatGPT review.
