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

# FROM_DSH — Task 6K Report: WHU Source → Pseudo-Instance Data Audit

_This file holds the Task 6K report. The Task 6J report is preserved in git history and in
`docs/task6j_structured_proposal_grounding.md`; Task 6I in `docs/task6i_visual_query_refinement.md`;
Task 6H.1 in `docs/task6h1_bounded_point_counterfactual.md`._

Full design notes: `docs/task6k_whu_source_pseudoinstance_audit.md`, ADR-023.

## 1. Verdict

**`KEEP_WHU_AS_PRIMARY_FOR_NOW`** — `legacy_baseline_role: keep`.

The conversion is faithfully implemented and numerically light: the current 36,926 components are
exactly the 36,926 surviving polygons of the historical pipeline (all 36,926 objects re-emulate at
union IoU 1.00000, object-count agreement 4,038/4,038 tiles), the `<50` filter deletes 1,383 of
38,309 contours (3.61 % of components, **0.058 %** of foreground area), hole filling adds 0.021 %,
and polygon simplification is effectively lossless (IoU 0.99955, boundary 0.0086 px). The decisive
relation-semantic measurement — RAW (all 8-connected semantic components, no filter) vs CONVERTED
(current representation) under the frozen Task 3B semantics — gives **4.33 %** weighted
current-query target change (below the declared 5 % materiality threshold), concentrated in the
level-1 extremes (5.0 %) and essentially zero for L2/L3 (0.03 %); heuristic high merge risk is
1.40 %. Task 6J's J1 failures are dominated by the **proposal model** (49/65 = 75.4 % on clean
targets; only 2/65 = 3.1 % conversion/data related). WHU therefore remains the primary corpus, with
two recorded structural limitations (no true instance identity anywhere in the chain; train/val is
a random split of the same contiguous regions). Read-only throughout: no source, converted or
legacy file was modified, no dataset regenerated, no model trained, nothing installed.

## 2. Source Inventory

17,388 cropped tiles: train 3,135 · train_no 10,527 · test 903 · test_no 2,823; all labels present,
0 missing on either side, `.tif` only, 512×512 uint8, **binary** (`[0, 255]`, or `[0]` for empty
tiles). Empty masks: train 0, test 0, train_no 9,378, test_no 2,537 (so the `_no` folders are
mostly — not purely — empty). Foreground fraction (train): min 1.0 %, median 3.7 %, mean 6.4 %,
max 57.3 %. Images: RGB uint8 (300-tile deterministic sample), median 789 KB. Whole-area subtree:
3 georeferenced rasters (994 Mpx / 2.66 Gpx / 967 Mpx) with `.tfw` world files, which Pillow
refuses to decode (decompression-bomb limit — recorded, not bypassed). Shapefile subtree: `EA.shp`
+ `.dbf/.prj/.shx/.sbn/.sbx/.shp.xml`.

## 3. Historical Split Recovery

Recovered from the existing converted folders, never regenerated: train 2,508 + val 627 = 3,135 =
the **entire** source train pool; test 903 = the source test pool; zero train∩val, train∩test,
val∩test overlaps; zero missing or extra stems; `int(3135 × 0.2) = 627` matches the observed val
size. **Reproducibility caveat recorded**: the historical `get_image_stems` returned
`list(set(...))`, whose iteration order depends on the per-process hash seed, so `random.seed(42)`
alone does not guarantee the same split across processes — the folders are the authority.
**Geographic finding**: tile-prefix regions (`1_x`/`2_x` vs bare test index) show that **100 % of
val tiles come from regions also present in train** (train/val scene-level correlation is certain
at region granularity), while the test region is disjoint. Per-tile adjacency/overlap cannot be
quantified (no coordinates in the cropped tiles) and none were invented. The split supports
random-tile generalization only; only the test region is unseen.

## 4. Raw Semantic Component Statistics

After threshold 127, **8-connected (primary): 38,309 components** over the 4,038 aligned tiles
(4-connected sensitivity: 38,491). Components/tile mean 9.49, median 7, p90 20, max 53. Area px:
min 1, p5 118, p25 646, median 1,209, p75 1,921, p95 4,513, max 131,549. Border-touching 13,091
(**34.2 %**). Tiles by component count: 1 → 104, 2–4 → 1,091, 5–9 → 1,358, 10–19 → 1,056, ≥20 →
429. Foreground/tile mean 6.1 %, median 3.7 %. **Terminology enforced**: these are semantic
connected components, NOT verified physical-building instances.

## 5. `<50` Filter Loss

External contours before filter 38,309 → removed **1,383 (3.61 %)**, kept 36,926; removed raster
pixels 37,589 = **0.058 % of foreground**; 1,099 tiles affected (27.2 %); removed-per-affected-tile
median 1; the `len(approx) < 3` skip removed **0**; 1,116 raw components have raster area < 50 px
(2.91 %, 0.033 % of foreground). Both raster pixel area and `cv2.contourArea` are reported.

## 6. `RETR_EXTERNAL` Topology Loss

6,757 components contain interior holes; 297 tiles have holes; **13,399 hole pixels are added**
(0.021 % of foreground) by external-contour filling; reconstruction difference
(filled ∧ ¬foreground) equals exactly the hole pixels. Effect: components are filled exterior
polygons, slightly *larger* than the true footprint — area is added, never lost.

## 7. Polygon Approximation Loss

Post-filter raster vs simplified polygons: IoU mean **0.99955** (p1 0.99178), boundary displacement
mean **0.0086 px** (p95 0.26 px), area bias ≈ 0; only 25/4,038 tiles below IoU 0.99. Cumulative
raw-vs-post-filter IoU mean 0.99902. Loss ranking: filter (0.058 % of foreground) ≫ hole filling
(0.021 %) ≫ approximation (≈0).

## 8. Actual YOLO Fidelity

All 4,038 label files re-parsed and rasterized object by object: 36,926 actual polygons, **36,926
matched** to the in-memory emulator at per-object IoU ≥ 0.5, **object-count agreement 4,038/4,038
tiles**, mean union raster IoU **1.00000**, 0 malformed lines, 0 wrong-class lines, 0 degenerate
(0-pixel) polygons. Difference vs the supplied snippet: the snippet omits `if len(approx) < 3:
continue` and the `[0, 1]` clipping that the historical script contains; in this dataset the
`< 3` branch never fires (0 polygons), so both agree here — recorded because a future label dump
built from the snippet alone could differ. Pre-clip coordinates are unrecoverable from the saved
labels; vertices exactly on 0/1 are the recoverable clipping evidence.

## 9. Component Lineage

```
38,309 raw 8-connected semantic components
  → 38,309 external contours (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
  → 1,383 removed by `contourArea < 50`
  → 36,926 simplified polygons == 36,926 actual YOLO label lines
  → 36,926 current components (component_id = source_polygon_index + 1; 0 zero-area)
```

Identities verified: raw == external contours, kept contours == actual polygons, actual polygons ==
current components, 4,038/4,038 tiles equal at the polygon/component stage, 2,939/4,038 tiles equal
at the raw/current stage. **36,926 is the number of surviving polygons, not of raw semantic
components**; the difference is the conversion loss.

## 10. Relation-Semantic Drift

All 20 canonical programs on RAW (38,309 candidates, no filter) vs CONVERTED (36,926; 36,926/36,926
matched) under the frozen relation config: **1,069/4,038 tiles (26.5 %) show any change**;
**weighted current-query target-change rate 4.33 %**; L1 **5.0 %** (1,071/21,430), L2 0.03 %
(4/12,569), L3 0.03 % (2/6,496). Per program: `topmost` 8.2 %, `leftmost` 7.9 %, `bottommost`
7.5 %, `rightmost` 7.4 %, `smallest` 0.16 %, `largest` 0.03 %, all `*_to_nearest` ≤ 0.13 %, all
`*_to_{dir}` 0 %. Attribution: L1 extremes are `small_component_removed` (≈300 tiles each) plus a
few `polygon_geometry_changed`; size ranks from `ranking_changed`; nearest from `nearest_changed`.
Answerability flips both ways (569 became-invalid, 653 became-newly-valid) but **cancel**: corpus
constructibility per program moves only +39…−18 out of ~3,500. Measures conversion sensitivity
only — not physical-building truth.

## 11. Split / Geographic Audit

See §3. Additional recorded facts: no geographic/scene metadata in the cropped tiles; the
whole-area georeferenced rasters and the EA shapefile exist locally but were not joined to tiles
(no coordinates fabricated); `val` is a random 20 % of a contiguous source pool, so scene-level
train/val leakage is certain at region granularity and unquantifiable at tile granularity.

## 12. Merge-Risk Heuristic

**Clearly labelled heuristic — not instance ground truth.** Raw components: high **538 (1.40 %)**,
medium 1,642 (4.29 %), low 36,129 (94.31 %); current components: high 399, medium 1,708, low
34,819. High risk concentrates slightly in dense tiles (1.48 % vs 1.25 % at < 10 components).
A first version classified "1-px erosion yields exactly one piece" as medium — 92.7 % of components,
the base rate — and was revised so the classes are discriminative; the revision is recorded in the
artifact.

## 13. Task 6J Cross-Analysis

Frozen 6J facts (J0 1.000 / paired 20-20; J1 mIoU 0.3712, 35/120 abstentions, paired 5/20; YOLO
target recall@0.5 0.869 and tiny recall 0.391; J2 1.000 fixed-120 and full-val; J3 1.000 / paired
20-20) cross-referenced against source evidence for every J1 failure (65/120):
**49 (75.4 %) proposal-model related on clean targets** (a proposal at IoU ≥ 0.5 exists and no
conversion flag holds); **2 (3.1 %) plausibly conversion/data related** (HIGH merge risk / measured
RAW-vs-CONVERTED target change); **14 (21.5 %) inseparable** with current evidence. Conversion-flag
discriminative power (failures vs successes): `raw_vs_converted_target_changed` 1.5 % vs 0 %,
`target_merge_risk_high` 1.5 % vs 0 %, `target_below_area_50` 0 % vs 0 %, missing raw counterpart
0 % vs 0 %. The medium merge-risk class fires for 63/65 failures **and** 54/55 successes, so it is
recorded but not used as evidence. Conclusion: the binding constraint is the proposal model, not
the source data or the conversion.

## 14. Qwen2B Parser Qualification

Task 6J proves **Qwen3-VL-2B is sufficient for the current closed-template 20-program
classification task** (20 closed program classes, program id 1:1 with `query_type`, finite
templated instruction families with ≥ 3 variants each). It does **not** prove 2B suffices for
arbitrary natural language, for paraphrase/OOD instructions, or that 4B cannot help the final
system. No 2B-vs-4B experiment was run in Task 6K.

## 15. Primary-Dataset Decision

`REPLACE_PRIMARY_DATASET` requires material conversion-induced drift, substantial small-object
deletion, high merge risk, identity limits becoming binding, or an unsuitable split. Measured
against the declared thresholds (drift > 0.05, contour deletion > 0.05, foreground deletion > 0.01,
high merge risk > 0.15, proposal-model dominance): **0.0433 / 0.0361 / 0.00058 / 0.0140** and
75.4 % proposal-model dominance → no replace condition fires, so **`KEEP_WHU_AS_PRIMARY_FOR_NOW`**
with `legacy_baseline_role: keep`. Recorded limitations: no true instance identity in the chain
(instances are connected components) and a random train/val split over the same regions.

## 16. Future Dataset Audit Protocol

`evaluation/dataset_audit_schema_v1.json` (schema v1) with the required fields, the WHU instance
filled from this audit, a blank `candidate_template` for SpaceNet 2 / WHU-Mix Vector or later
candidates, and the decision gates recorded verbatim. License and source resolution remain
`UNKNOWN` — no external dataset fact was fabricated.

## 17. Runtime

Deterministic CPU work only: inventory 25 s · per-tile scan 369 s · aggregates 83 s · YOLO fidelity
+ lineage 199 s · relation drift 152 s. No model training, no downloads, no package installation,
no GPU. All reads are read-only; the only writes are BuildReasonSeg artifacts (large per-tile caches
under gitignored `artifacts/task6k/`). **Empirical read-only proof** (`evaluation/task6k_read_only_proof.json`):
the newest modification time under every read-only root predates the audit — original WHU root
2026-08-03 (`mask_to_yolo.py`), cropped subtree 2018-12-07, converted dataset 2026-08-03
(`labels/val.cache`), legacy project 2026-08-03 — so no source, converted or legacy file was
touched (`modified_after_audit_start: false` for all four roots).

## 18. Tests

`python -m pytest tests/ -q` → **468 passed** (24 new Task 6K checks). The new file
`tests/test_task6k_whu_audit.py` covers the required 20 items: no writes under `C:\D\resources`, no
writes in the legacy project, threshold 127 exactness, `RETR_EXTERNAL` exactness, `<50 contourArea`
exactness (including the 8×8 → 49 px boundary case), `0.001 * arcLength` epsilon exactness,
OpenCV/Pillow fallback binary equivalence, split recovered-not-regenerated, reproducibility caveat
recorded, YOLO parser/rasterizer correctness, three-stage loss decomposition, component terminology
discipline, frozen relation config use, no target-id leakage in the drift experiment, merge-risk
heuristic labelling, Task 6J frozen-artifact cross-analysis, the Qwen qualification, no model
training, no dataset mutation, deterministic JSON, plus the empirical read-only proof and artifact
existence/alignment/verdict checks.

## 19. Git / Watt

Task commit `audit: compare WHU source and pseudo-instance conversion` followed by the `docs:`
handoff commit; no source data, converted dataset, masks/raster dumps, weights, caches or `.conda`
staged. Watt is not required for this task; the established ownership rules were respected for the
final push only.

## 20. Recommended Next Step

Keep the WHU corpus and the frozen relation/program stack unchanged. The measured binding
constraint is the **proposal model**, so the next step remains a proposal-backbone task (an
instance-segmentation model whose instances align with the component semantics, or deterministic
proposal post-processing), evaluated with the same J1/J4 gates. Two dataset-level follow-ups are
recorded but **not** started: (a) quantify geographic train/val correlation by joining the
whole-area rasters + `EA.shp` to the cropped tiles if coordinates can be recovered without
inventing them; (b) run the same audit (schema v1) against a candidate dataset that provides true
instance identity, e.g. SpaceNet 2 or WHU-Mix Vector, before any replacement decision. Per section
17 the task stops here — no dataset download, no WHU deletion, no BuildSpatialReason regeneration,
no YOLO retraining, no new instance backbone, no 4B, no `[REF]`/SRE/SCL, no full training, no GUI —
waiting for ChatGPT review.
