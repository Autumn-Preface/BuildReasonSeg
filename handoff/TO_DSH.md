# TO_DSH — Task 6K: WHU Source → Pseudo-Instance Data Audit

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Goal: audit the original WHU Satellite Dataset II (East Asia) semantic masks against the historical semantic→YOLO pseudo-instance conversion and the current BuildReasonSeg component representation, so we can make an evidence-based **KEEP-AS-BASELINE vs REPLACE-PRIMARY-DATASET** decision.
>
> This is a read-only dataset audit. Do not retrain any model and do not modify any dataset.

## 0. UI language
All DSH narrative/UI output must be Chinese. Code/paths/metric keys may remain English.

## 1. Read-only sources

Original WHU semantic dataset:
`C:\D\resources\Satellite dataset Ⅱ (East Asia)`

Likely historical subtree:
`C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels`

Historical converted YOLO dataset:
`C:\D\resources\WHU_YOLO_dataset`

Canonical repo:
`C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`

Legacy project:
`C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment`

Rules:
- original WHU: READ ONLY;
- converted YOLO: READ ONLY;
- legacy project: READ ONLY;
- do not move/rename/re-encode/regenerate source files;
- write only to BuildReasonSeg gitignored temp/artifact paths, evaluation/docs/tests/scripts/handoff as appropriate.

## 2. Historical conversion semantics

Audit against the exact user-supplied conversion semantics:

```python
mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
if mask is None:
    from PIL import Image
    mask = np.array(Image.open(label_path).convert("L"))

_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

for contour in contours:
    if cv2.contourArea(contour) < 50:
        continue
    epsilon = 0.001 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
```

Important:
- threshold 127;
- `RETR_EXTERNAL`;
- `CHAIN_APPROX_SIMPLE`;
- filter is `cv2.contourArea(contour) < 50`;
- simplification epsilon = `0.001 * arcLength`;
- one class: building;
- Pillow is I/O fallback only.

Do NOT rerun and overwrite the historical split.

Historical split code used:
```python
stems = set(...)
return list(stems)
random.seed(42)
random.shuffle(train_stems_all)
```

Record the reproducibility caveat:
`list(set(...))` order is not cross-process guaranteed, so seed 42 alone does not guarantee the same split. Recover the actual split from the existing converted folders.

# PART A — Source inventory

## 3. Inventory original data
For every discovered original split/subtree report:
- image count;
- label count;
- matched stems;
- missing images/labels;
- extensions;
- image shape distribution;
- label shape distribution;
- image dtype/channels;
- label dtype;
- label unique values;
- empty/non-empty mask counts;
- foreground-pixel fraction distribution.

Measure before thresholding. Record exact resolved paths.

## 4. Recover actual historical split
From `WHU_YOLO_dataset/images/{train,val,test}` and labels:
- exact stem counts;
- train/val/test overlap;
- map current stems to original source split;
- whether current train+val exactly partitions original source train;
- whether current test exactly matches original source test.

Do not regenerate split from seed 42.

# PART B — Raw semantic component audit

## 5. Raw connected-component statistics
After threshold 127:
- primary diagnostic: 8-connectivity;
- secondary sensitivity diagnostic: 4-connectivity.

For 8-connectivity report:
- total components;
- components/tile;
- area px distribution: min, p1, p5, p10, p25, median, p75, p90, p95, p99, max;
- bbox width/height distribution;
- border-touch count/rate;
- foreground area/tile;
- tiles with 1 / 2–4 / 5–9 / 10–19 / >=20 components.

Terminology rule:
These are semantic connected components, NOT verified physical-building instances.

# PART C — Conversion-loss audit

## 6. Reproduce historical contour pipeline in memory
For each raw binary mask reproduce:
- external contours before filter;
- contourArea;
- `<50` removal;
- polygon simplification.

Report:

### `<50` filter
- external contour count before filter;
- count removed;
- percentage removed;
- pixel foreground area removed;
- percentage of total foreground pixels removed;
- tiles affected;
- mean/median removed contours per affected tile.

Report both raster pixel area and `cv2.contourArea`.

### `RETR_EXTERNAL` topology loss
Measure:
- source components containing holes;
- hole pixel count;
- reconstruction difference from external-contour-only rasterization.

Do not overstate physical meaning.

### `approxPolyDP` loss
Rasterize simplified polygons and compare against post-filter external-contour target:
- IoU;
- Dice;
- precision;
- recall;
- area bias;
- boundary displacement if inexpensive.

Separate loss due to:
1. `<50` filtering;
2. `RETR_EXTERNAL`;
3. polygon approximation.

# PART D — Actual current YOLO fidelity

## 7. Compare actual YOLO labels with the historical emulator
For every current stem:
- parse actual YOLO polygons;
- rasterize to source-label resolution;
- compare to emulator.

Report:
- object-count agreement;
- raster IoU/Dice;
- missing/extra objects;
- malformed/degenerate labels;
- coordinate clipping if recoverable.

If current labels differ from the supplied converter, classify why; do not assume the supplied code was the final exact converter.

## 8. Component lineage
Compare:
- raw thresholded semantic connected components;
- historical conversion emulator;
- actual `WHU_YOLO_dataset`;
- current `datasets/whu/` / BuildSpatialReason component metadata.

Explain how the recorded current `36,926` components arise.

# PART E — Relation-semantic drift

## 9. Critical experiment: RAW vs CONVERTED candidate sets

For each aligned image construct:

RAW:
- all 8-connected semantic components after threshold 127;
- no `<50` filter.

CONVERTED:
- candidate set implied by the actual historical conversion/current component representation.

Use the same frozen BuildSpatialReason relation semantics.

Compare target identity for all applicable canonical programs:
- leftmost/rightmost/topmost/bottommost;
- largest/smallest;
- largest/smallest → nearest;
- largest/smallest → left/right/above/below;
- Level-3 reference → direction → nearest.

For each program report:
- valid in both;
- target unchanged;
- target changed;
- became invalid/ambiguous;
- became newly valid;
- change rate.

Attribute changes:
- small component removed;
- ranking changed;
- nearest changed;
- directional candidate-set changed;
- border/eligibility changed;
- polygon geometry changed.

Also report:
- fraction of images with any relation-semantic change;
- overall current-query target-change rate;
- by L1/L2/L3.

This measures conversion sensitivity only. It does not establish physical-building truth.

# PART F — Split/geographic audit

## 10. Historical split reproducibility
Record that `set -> list -> shuffle(seed=42)` is not guaranteed cross-process reproducible without controlled hash/set order.

Current folder split is the historical authority.

## 11. Geographic/spatial correlation
Inspect local naming/layout/metadata.

If source-scene/tile adjacency is recoverable:
- estimate train↔val same-scene/adjacency mixing;
- quantify spatial-correlation risk.

If not recoverable:
- explicitly say geographic leakage cannot be quantified from available metadata;
- do not invent coordinates.

State whether current split supports:
- random tile generalization;
- geographic/cross-city generalization.

# PART G — Merge-risk heuristic

## 12. Touching/merged-building risk
Binary semantic masks cannot reveal physical instance identity.

Do NOT claim true instance recovery.

Provide only a clearly labelled heuristic risk analysis, e.g.:
- thin-neck/bridge split sensitivity under 1–2 px erosion;
- multiple large distance-transform peaks;
- strongly multi-lobed shapes.

Report:
- low/medium/high risk counts;
- representative sample IDs;
- whether risk concentrates in dense-building tiles.

# PART H — Cross-analyze Task 6J

## 13. Freeze Task 6J facts
Use existing artifacts only:
- J0: 120/120, paired 20/20;
- J2: current-template parser 1.000 fixed-120 and full-val;
- J3: 120/120, paired 20/20;
- YOLO target proposal recall@0.5 = 0.869;
- tiny-component recall@0.5 = 0.391;
- J1 mIoU = 0.3712;
- J1 abstentions = 35/120;
- J1 paired mask selection = 5/20.

Cross-reference J1 failing sample IDs against:
- raw source component size;
- distance to `<50` threshold;
- border status;
- merge-risk status;
- proposal count;
- RAW-vs-CONVERTED relation drift.

Estimate what share of J1 failures are:
- plausibly conversion/data related;
- proposal-model related on otherwise clean targets;
- inseparable with current evidence.

Do not force attribution when unsupported.

# PART I — Qwen2B interpretation

## 14. Qualify Task 6J J2 correctly
Current BuildSpatialReason has:
- 20 closed program classes;
- finite templated instruction families (at least 3 template variants per family in the current generator);
- program id is 1:1 with current `query_type`.

Therefore record:

> Task 6J proves Qwen3-VL-2B is sufficient for the current closed-template 20-program classification task.

It does NOT prove:
- 2B is sufficient for arbitrary natural language;
- 2B is sufficient for paraphrase/OOD instructions;
- 4B cannot help the final system.

Do not run a 2B-vs-4B experiment in Task 6K.

# PART J — Dataset-role verdict

## 15. Use exactly one

### `REPLACE_PRIMARY_DATASET`
Use if evidence shows the current semantic→pseudo-instance source is structurally unsuitable as the main corpus for instance-level spatial reasoning, including material conversion-induced relation drift, substantial small-object deletion, high merge risk, lack of real instance identity becoming binding, Task 6J mismatch tied to pseudo-instance semantics, or split structure unsuitable for intended paper claims.

WHU still remains a historical baseline.

### `KEEP_WHU_AS_PRIMARY_FOR_NOW`
Only if conversion loss is small, relation targets are very stable, pseudo-instance identity is sufficiently reliable, and Task 6J failure is mainly a replaceable proposal-model problem.

### `INSUFFICIENT_EVIDENCE`
Only if source/converted data cannot be aligned reliably.

Also report:
`legacy_baseline_role: keep / do_not_keep`

Expected default is `keep`.

# PART K — Reusable future dataset audit protocol

## 16. Create a reusable schema
Create `evaluation/dataset_audit_schema_v1.json` covering:
- annotation type: semantic / raster instance / vector polygon;
- true stable instance ID available;
- image count / instance count;
- instances per tile;
- area distribution;
- tiny-target rate;
- border-truncation rate;
- multi-instance tile rate;
- candidate density;
- L1/L2/L3 constructible-query rate;
- nearest constructibility;
- directional relation constructibility;
- relation ambiguity rate;
- geographic diversity metadata;
- local storage footprint if known;
- license: UNKNOWN unless externally verified;
- source resolution: UNKNOWN unless locally/officially verified.

This schema will later be reused for SpaceNet 2 / WHU-Mix Vector or other candidates.

Do not fabricate external dataset facts.

# PART L — Required artifacts

Create:
```text
evaluation/task6k_source_inventory.json
evaluation/task6k_raw_component_stats.json
evaluation/task6k_conversion_loss.json
evaluation/task6k_actual_yolo_fidelity.json
evaluation/task6k_component_lineage.json
evaluation/task6k_relation_semantic_drift.json
evaluation/task6k_split_audit.json
evaluation/task6k_merge_risk.json
evaluation/task6k_task6j_cross_analysis.json
evaluation/task6k_dataset_decision.json
evaluation/dataset_audit_schema_v1.json
docs/task6k_whu_source_pseudoinstance_audit.md
```

Optional small diagnostics:
`evaluation/task6k_samples/`

No large raster dumps.

# PART M — Tests

At minimum cover:
1. no writes under `C:\D\resources`;
2. no writes under legacy project;
3. threshold 127 exact;
4. `RETR_EXTERNAL` exact;
5. `<50 contourArea` exact;
6. epsilon `0.001*arcLength` exact;
7. OpenCV/Pillow fallback does not change binary semantics;
8. historical split recovered from current folders, not regenerated;
9. split reproducibility caveat recorded;
10. YOLO rasterization parser correct;
11. loss decomposition separates filter/topology/approximation;
12. raw components never called true physical instances;
13. relation drift uses frozen relation config;
14. no target-id leakage;
15. merge-risk labelled heuristic;
16. Task 6J cross-analysis uses frozen artifacts;
17. J2 qualification recorded;
18. no model training;
19. no dataset mutation;
20. deterministic JSON where applicable.

Run:
`python -m pytest tests/ -q`

# PART N — Runtime/dependencies

This is deterministic data/statistics work.

No model training.
No external downloads.
Use existing environments only.

If a core audit cannot run without a missing package, STOP and report it. Do not install packages automatically.

# PART O — Git/Watt

Do not stage:
- original data;
- converted YOLO dataset;
- masks/raster dumps;
- weights/checkpoints;
- caches;
- `.conda`.

Commit only scripts/tests/small JSON/docs/handoff and a few small diagnostic images if useful.

Recommended commit:
`audit: compare WHU source and pseudo-instance conversion`

Use established Watt ownership rules only for final Git push if needed.

# PART P — Handoff

`handoff/FROM_DSH.md` must include:
1. Verdict
2. Source Inventory
3. Historical Split Recovery
4. Raw Semantic Component Statistics
5. `<50` Filter Loss
6. `RETR_EXTERNAL` Topology Loss
7. Polygon Approximation Loss
8. Actual YOLO Fidelity
9. Component Lineage
10. Relation-Semantic Drift
11. Split/Geographic Audit
12. Merge-Risk Heuristic
13. Task 6J Cross-Analysis
14. Qwen2B Parser Qualification
15. Primary-Dataset Decision
16. Future Dataset Audit Protocol
17. Runtime
18. Tests
19. Git/Watt
20. Recommended Next Step

Final DSH UI in Chinese must report:
- dataset-role verdict;
- source image/label counts;
- raw semantic component count;
- current pseudo-instance/component count;
- `<50` removed component count/rate;
- foreground area removed rate;
- current-YOLO raster vs source IoU/recall;
- fraction of images with any relation-semantic drift;
- target-change rate overall and L1/L2/L3;
- merge-risk heuristic rate;
- split reproducibility/geographic finding;
- share of Task 6J failures plausibly data/conversion vs proposal-model;
- whether WHU should remain only historical baseline;
- tests;
- commit/push.

# 17. STOP

After Task 6K STOP.

Do not automatically:
- download a new dataset;
- delete WHU;
- regenerate BuildSpatialReason;
- retrain YOLO;
- train a new instance backbone;
- switch Qwen to 4B;
- add `[REF]`, SRE or SCL;
- run full training;
- build GUI.

Wait for ChatGPT review.
