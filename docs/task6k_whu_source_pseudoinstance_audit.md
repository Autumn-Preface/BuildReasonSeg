# Task 6K — WHU Source → Pseudo-Instance Data Audit

> Task: `handoff/TO_DSH.md` · Verdict: **`KEEP_WHU_AS_PRIMARY_FOR_NOW`** (`legacy_baseline_role: keep`)
> Evidence: `evaluation/task6k_*.json`, `evaluation/dataset_audit_schema_v1.json` · ADR-023 ·
> Tests: `tests/test_task6k_whu_audit.py`

This is a **read-only** audit of the chain

```
original WHU Satellite Dataset II (East Asia) binary semantic raster
  → historical semantic→polygon conversion (mask_to_yolo.py)
  → current BuildReasonSeg building-connected-component representation (36,926 components)
```

Nothing was written below `C:\D\resources`, the legacy project or the converted dataset; no model
was trained; no dataset was regenerated; no package was installed.

## 1. Source inventory (section 3)

| Split | Images | Labels | Matched | Empty masks | Label values | Foreground mean |
|---|---|---|---|---|---|---|
| train | 3,135 | 3,135 | 3,135 | 0 | `[0, 255]` | 6.41 % |
| train_no | 10,527 | 10,527 | 10,527 | 9,378 | `[0, 255]` / `[0]` | 0.05 % |
| test | 903 | 903 | 903 | 0 | `[0, 255]` | 5.04 % |
| test_no | 2,823 | 2,823 | 2,823 | 2,537 | `[0, 255]` / `[0]` | 0.05 % |
| **total** | **17,388** | **17,388** | **17,388** | 11,915 | — | — |

* 0 images without labels, 0 labels without images; only `.tif` in the cropped subtrees.
* Images: 512×512, RGB, uint8 (deterministic 300-per-split sample), median 789 KB.
* Labels: 512×512, uint8, **binary** — no unusual value set anywhere.
* Foreground fraction (train): min 1.0 %, p5 1.2 %, median 3.7 %, mean 6.4 %, max 57.3 %.
* The `_no` folders are *mostly* empty tiles but not purely so: 1,149 `train_no` and 286 `test_no`
  tiles do contain buildings.
* Whole-area subtree: 3 georeferenced rasters (`test.tif` 3.0 GB / 994 Mpx, `train1.tif` 8.0 GB /
  2.66 Gpx, `train2.tif` 2.9 GB / 967 Mpx) plus their label rasters (16.7/19.5/17.1 MB) and `.tfw`
  world files (92 bytes each). Pillow refuses to decode them (decompression-bomb limit), which is
  recorded as such; only byte sizes and the reported pixel counts are used.
* Shapefile subtree: `EA.shp` (5.0 MB) + `.dbf/.prj/.shx/.sbn/.sbx/.shp.xml`.

## 2. Historical split recovery (sections 4, 10)

The split is **recovered from the existing converted folders**, never regenerated:

| Check | Result |
|---|---|
| converted train / val / test | 2,508 / 627 / 903 (total 4,038) |
| train + val == source train | **True** (3,135) |
| test == source test | **True** (903) |
| train∩val, train∩test, val∩test | 0 / 0 / 0 |
| missing or extra stems | 0 / 0 |
| `int(len(train_stems) * 0.2) == 627` | True |

**Reproducibility caveat (recorded):** the historical code built the stem list with
`stems = set(); … return list(stems)`, and `list(set(...))` iteration order depends on the
per-process string-hash seed (`PYTHONHASHSEED`). `random.seed(42)` alone therefore does **not**
guarantee the same split across processes; the surviving folders are the authority.

**Geographic audit (section 11):** the numeric tile prefixes are recoverable region labels
(`1_x`/`2_x` = the two source train rasters; bare index = the test raster). All 627 val tiles come
from regions that also appear in train (`val_same_region_mixing_rate = 1.0`), so **train/val
scene-level correlation is certain at region granularity**; the test split is a separate region
and is geographically disjoint. Per-tile adjacency and exact overlap cannot be quantified (the
cropped tiles carry no world files or coordinates) and **no coordinates were invented**. The split
therefore supports *random-tile* generalization, and only the test region provides an unseen
region.

## 3. Raw semantic components (section 5)

8-connected components of the thresholded (`>127`) source label — **primary diagnostic**:
**38,309 components** across 4,038 tiles (4-connectivity sensitivity: 38,491).

| Statistic | Value |
|---|---|
| components / tile | mean 9.49, median 7, p90 20, max 53 |
| area px | min 1, p5 118, p25 646, median 1,209, p75 1,921, p95 4,513, max 131,549 |
| border-touching | 13,091 (34.2 %) |
| tiles by count | 1 comp: 104 · 2–4: 1,091 · 5–9: 1,358 · 10–19: 1,056 · ≥20: 429 |
| foreground / tile | mean 6.1 %, median 3.7 % |

**Terminology (enforced in every artifact):** these are *semantic connected components*, **not**
verified physical-building instances. The source raster is binary, so mutually touching buildings
are already merged here.

## 4. Conversion loss (section 6)

The historical pipeline was reproduced **in memory** for all 4,038 tiles, and the loss of each
stage is reported separately.

**Stage 1 — `cv2.contourArea < 50` filter** (plus the `len(approx) < 3` skip)

| Metric | Value |
|---|---|
| external contours before filter | 38,309 |
| removed | **1,383 (3.61 %)** |
| kept | 36,926 |
| removed raster pixels | 37,589 = **0.058 % of foreground** |
| tiles affected | 1,099 (27.2 %) |
| removed by the `len(approx) < 3` skip | 0 |
| raw components with raster area < 50 px | 1,116 (2.91 %), 0.033 % of foreground |

**Stage 2 — `RETR_EXTERNAL` topology (hole filling)**

6,757 components contain interior holes (297 tiles); 13,399 hole pixels are **added** to the mask
(0.021 % of foreground). This makes a component a filled exterior polygon — slightly *larger* than
the true footprint — and never removes building area.

**Stage 3 — `approxPolyDP(0.001 * arcLength)`**

| Metric | Value |
|---|---|
| post-filter raster vs simplified polygons, IoU | mean **0.99955**, p1 0.99178 |
| boundary displacement | mean **0.0086 px**, p95 0.26 px |
| tiles with IoU < 0.99 | 25 / 4,038 |
| cumulative raw-vs-post-filter IoU | mean 0.99902 |

**Loss ranking:** the only non-trivial stage is the small-object filter (3.6 % of contours,
0.06 % of foreground area); hole filling adds 0.02 %; polygon approximation is effectively
lossless.

## 5. Actual YOLO fidelity and lineage (sections 7-8)

All 4,038 current label files were re-parsed and re-rasterized object by object:

| Metric | Value |
|---|---|
| actual YOLO polygons | 36,926 |
| matched to the emulator at per-object IoU ≥ 0.5 | **36,926 / 36,926** |
| tiles with equal object count | **4,038 / 4,038** |
| mean union raster IoU actual vs emulator | **1.00000** |
| malformed lines / wrong-class lines / degenerate polygons | 0 / 0 / 0 |

**Component lineage — how 36,926 arises:**

```
38,309 raw 8-connected semantic components
   → 38,309 external contours (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
   → 1,383 removed by `contourArea < 50`
   → 36,926 simplified polygons  ==  36,926 YOLO label lines
   → 36,926 current components (component_id = source_polygon_index + 1)
```

2,939 / 4,038 tiles have identical raw and current counts. The current component count is the
number of **surviving polygons**, not of raw semantic components — the difference *is* the
conversion loss.

**Difference vs the supplied snippet (section 7):** the snippet in the task description omits two
details that the historical script contains — `if len(approx) < 3: continue` (lines 117-118) and
the `[0, 1]` coordinate clipping (lines 123-124). In *this* dataset the `len(approx) < 3` branch
never fires (0 polygons), so the supplied snippet and the emulator agree here; the difference is
still classified and recorded because a future label dump produced with the snippet alone could
differ. Pre-clip coordinates are **not** recoverable from the saved labels (already clipped);
vertices exactly on 0/1 are the recoverable clipping evidence.

## 6. Relation-semantic drift — RAW vs CONVERTED (section 9, critical experiment)

All 20 canonical Task 6J programs were executed on both candidate sets with the **frozen Task 3B
relation configuration**:

* **RAW** = every 8-connected semantic component, no `<50` filter (38,309 candidates);
* **CONVERTED** = the current component representation (36,926 candidates; 36,926/36,926 matched
  to raw components at IoU ≥ 0.5).

| Metric | Value |
|---|---|
| tiles with **any** relation-semantic change | 1,069 / 4,038 = **26.5 %** |
| **overall current-query target-change rate** (weighted by the v0.1.1 query mix) | **4.33 %** |
| L1 target-change rate | **5.0 %** (1,071 / 21,430) |
| L2 target-change rate | 0.03 % (4 / 12,569) |
| L3 target-change rate | 0.03 % (2 / 6,496) |
| validity flips | 569 became-invalid, 653 became-newly-valid |

Per program (target-change rate): `leftmost` 7.9 %, `topmost` 8.2 %, `bottommost` 7.5 %,
`rightmost` 7.4 %; `largest` 0.03 %, `smallest` 0.16 %; `*_to_nearest` ≤ 0.13 %; `*_to_{dir}` 0 %.

**Attribution:** the L1 extremes are dominated by `small_component_removed` (≈300 tiles per
program) — a tiny component that sits at the extreme position is deleted by the `<50` filter and
the answer moves to the runner-up. `largest`/`smallest` drift comes from `ranking_changed` (the
deleted speck perturbs the size rank in 7 cases) and the nearest programs from `nearest_changed`.
This measures **conversion sensitivity only**; it does not establish physical-building truth.

**Constructibility (answerability) nuance.** Tile-level answerability flips in both directions:
569 program-tile pairs become invalid and 653 become newly valid (1.5 % of the 80,760 pairs). The
L2-B direction programs (`*_to_{dir}`) flip most often in relative terms (~10 % of their pairs),
because whether a direction filter leaves *exactly one* candidate depends on whether the small
specks survive. Those flips largely **cancel**: the corpus-level constructible-tile count per
program changes by only +39…−18 out of ~3,500 (`constructibility_summary` in the artifact). The
conversion therefore redistributes *which* tiles are answerable far more than it changes *how
many*.

**Interpretation:** the drift is concentrated exactly where the frozen semantics are most fragile
(extreme-relation targets on tiles that contain sub-50 px specks) and is essentially zero for the
compositional L2/L3 programs. The absolute foreground area affected is 0.06 %.

## 7. Merge-risk heuristic (section 12)

**Clearly labelled heuristic — a binary semantic mask cannot reveal physical instance identity.**

| Class | Criterion | Raw semantic components | Current components |
|---|---|---|---|
| high | 1-px erosion splits into ≥ 2 pieces, or ≥ 3 distance-transform peaks | 538 (**1.40 %**) | 399 |
| medium | exactly 2 peaks, or solidity < 0.75, or vanishes under 1-px erosion | 1,642 (4.29 %) | 1,708 |
| low | otherwise | 36,129 (94.31 %) | 34,819 |

High risk is slightly more frequent in dense tiles (1.48 % vs 1.25 % in tiles with < 10
components). Representative high-risk samples are listed in `task6k_merge_risk.json`. A first
version of the heuristic treated "1-px erosion yields exactly one piece" as medium; that fired for
92.7 % of components (the base rate) and was removed so the classes are discriminative — the
revision is recorded in the artifact.

## 8. Task 6J cross-analysis (section 13)

Frozen Task 6J facts: J0 1.000 (120/120, paired 20/20); J1 mIoU 0.3712, 35/120 abstentions, paired
5/20; YOLO target recall@0.5 0.869, tiny-component recall 0.391; J2 1.000 fixed-120 and full-val;
J3 1.000, paired 20/20.

Every J1 failure (65/120) was cross-referenced against source-side evidence:

| Category | Count | Share of failures |
|---|---|---|
| proposal-model related **on clean targets** (a proposal at IoU ≥ 0.5 exists; no conversion flag) | 49 | **75.4 %** |
| plausibly conversion/data related (would-be-deleted < 50 px target, missing raw counterpart, HIGH merge risk, or a measured RAW-vs-CONVERTED target change) | 2 | 3.1 % |
| inseparable with current evidence | 14 | 21.5 % |

Discriminative power of the conversion flags (rate in failures vs successes):
`raw_vs_converted_target_changed` 1.5 % vs 0 %; `target_merge_risk_high` 1.5 % vs 0 %;
`target_below_area_50` 0 % vs 0 %; `target_raw_counterpart_missing` 0 % vs 0 %. The heuristic
*medium* merge-risk class fires for 63/65 failures **and** 54/55 successes, so it is recorded but
deliberately not used as evidence.

**Conclusion:** Task 6J's J1 failure is dominated by the proposal model (YOLO instance geometry vs
the component-calibrated frozen semantics), not by source-data or conversion defects.

## 9. Qwen2B qualification (section 14)

Task 6J proves **Qwen3-VL-2B is sufficient for the current closed-template 20-program
classification task** (20 closed program classes, program id 1:1 with `query_type`, finite
templated instruction families with ≥ 3 template variants each). It does **not** prove that 2B
suffices for arbitrary natural language, for paraphrase/OOD instructions, or that 4B cannot help
the final system. No 2B-vs-4B experiment was run.

## 10. Dataset-role verdict (section 15)

**`KEEP_WHU_AS_PRIMARY_FOR_NOW`**, `legacy_baseline_role: keep`.

Measured gates (thresholds declared in `task6k_dataset_decision.json` before reading the verdict):

| Gate | Threshold | Measured | Material? |
|---|---|---|---|
| conversion-induced relation drift | > 0.05 | **0.0433** | no |
| small-object deletion (contour rate) | > 0.05 | **0.0361** | no |
| small-object deletion (foreground share) | > 0.01 | **0.00058** | no |
| heuristic merge risk (high rate) | > 0.15 | **0.0140** | no |
| Task 6J failure attribution | proposal-model dominance | **0.754 proposal vs 0.031 conversion** | keep |
| alignment completeness | required | complete (exact identities) | — |

WHU therefore remains the primary corpus for now, with two recorded limitations: (a) there is **no
true instance identity** anywhere in the chain — instances are connected components; (b) the
train/val split is a random split of the same contiguous regions, so only random-tile
generalization is supported. WHU stays a historical baseline regardless (the dataset is never
deleted). A replacement becomes justified when a candidate passes the same measurement with real
instance identity and a geographic split.

## 11. Reusable dataset-audit schema (section 16)

`evaluation/dataset_audit_schema_v1.json` defines schema **v1**: annotation type, true stable
instance id availability, image/instance counts, instances per tile, area distribution,
tiny-target rate, border-truncation rate, multi-instance tile rate, candidate density, L1/L2/L3
constructible-query rate, nearest and directional constructibility, relation ambiguity rate,
geographic diversity metadata, local storage footprint, license (`UNKNOWN`) and source resolution
(`UNKNOWN`). The WHU instance is filled from this audit; a `candidate_template` is provided for
SpaceNet 2 / WHU-Mix Vector or any later candidate, with the decision gates recorded verbatim. No
external dataset fact is fabricated.

## 12. Reproduction

| Script | Artifact |
|---|---|
| `scripts/task6k_inventory.py` | `task6k_source_inventory.json`, `task6k_split_audit.json` |
| `scripts/task6k_scan.py` | gitignored per-tile evidence cache |
| `scripts/task6k_aggregate.py` | `task6k_raw_component_stats.json`, `task6k_conversion_loss.json`, `task6k_merge_risk.json` |
| `scripts/task6k_yolo_fidelity.py` | `task6k_actual_yolo_fidelity.json`, `task6k_component_lineage.json` |
| `scripts/task6k_relation_drift.py` | `task6k_relation_semantic_drift.json` |
| `scripts/task6k_cross_6j.py` | `task6k_task6j_cross_analysis.json` |
| `scripts/task6k_schema.py`, `scripts/task6k_decision.py` | `dataset_audit_schema_v1.json`, `task6k_dataset_decision.json` |

Runtime: inventory 25 s · scan 369 s · aggregate 83 s · fidelity 199 s · drift 182 s (all CPU,
read-only). `python -m pytest tests/ -q` → see `handoff/FROM_DSH.md` §18.
