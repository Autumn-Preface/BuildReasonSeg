# BuildSpatialReason v0.1

**BuildSpatialReason** is the reasoning segmentation dataset of the
**BuildReasonSeg** project. It is programmatically generated, not human-authored.

Current version **v0.1** uses the **WHU Building Dataset**,
subset **`Satellite dataset II (East Asia)`**, as its
**source dataset**. WHU appears only as provenance: it is not part of the dataset
identity, and nothing in the schema, relation representation, or reasoning program
assumes WHU. Further public building / remote-sensing datasets are expected to be
added later.

---

## Where it lives

```
datasets/build_spatial_reason/v0.1/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── manifest.json
└── statistics.json
```

Generation config: `configs/build_spatial_reason_v0.1.yaml`
Generator: `scripts/build_spatial_reason.py`, `spatial_reasoning/annotator.py`,
`spatial_reasoning/templates.py`

```bash
python scripts/build_spatial_reason.py
```

## Terminology

A **"building region"** (中文「建筑区域」) in an instruction refers to a **building
connected component** in the source annotation. It is **not** guaranteed to
correspond one-to-one with a real-world independent physical building.

Reason: the source raster was a binary semantic mask, so mutually touching
buildings were merged into a single component before this dataset existed. See
`docs/data_representation.md`.

---

## Core generation chain

```
source image
  -> building component representation      (datasets/whu, Task 2)
  -> frozen Spatial Relation Engine         (spatial_reasoning, Task 3A/3B)
  -> structured reasoning program           (reasoning_steps -- SOURCE OF TRUTH)
  -> deterministic target
  -> bilingual natural-language instruction
```

Every `target_component_id` can be **deterministically recomputed** from its
`reasoning_steps`. No target is ever guessed by a language model.

### The structured program is the source of truth

`reasoning_steps` is the ground truth of the reasoning. `reasoning_zh` /
`reasoning_en` are **rendered from** the steps. Steps are never inferred from
text. `annotator.recompute_target_from_steps` re-executes the stored program and
must reproduce the target.

Operations:

| operation | meaning |
|---|---|
| `argmax_area` / `argmin_area` | largest / smallest eligible component |
| `argmin_centroid_x` / `argmax_centroid_x` | leftmost / rightmost |
| `argmin_centroid_y` / `argmax_centroid_y` | topmost / bottommost |
| `filter_relation` | keep components satisfying `relation(subject, reference)` |
| `argmin_boundary_distance` | nearest within the supplied candidate set |

---

## Query levels

### Level 1 — direct spatial grounding

One reference-free extreme per query type: `leftmost`, `rightmost`, `topmost`,
`bottommost`, `largest`, `smallest`. Only `valid == true` and
`ambiguous == false` results are emitted; anything ambiguous is discarded.

### Level 2 — reference-based reasoning

**Type A: reference → nearest.** Reference is `largest` or `smallest`. The
reference must itself be eligible as a `nearest` anchor, the nearest relation must
be valid, and reference ≠ target.

**Type B: reference → direction.** e.g. "the building region to the right of the
largest building region". Emitted **only when exactly one candidate satisfies the
stated relation**. If more than one does, the query is **discarded** — a hidden
tie-break such as "the nearest of them" is never applied, because that criterion
would not appear in the instruction (ADR-004).

### Level 3 — multi-hop spatial reasoning

Structure: **reference → direction filter → nearest within the filtered set**.

Example: *"先找到面积最大的建筑区域，在位于它右侧的建筑区域中，分割距离它最近的一个。"*

```
Step 1: argmax_area
Step 2: filter_relation(right_of)
Step 3: argmin_boundary_distance within the filtered set
```

`nearest_within(anchor, candidates)` is a **separate** operation from
`nearest(anchor)` followed by a filter. The two differ: the global nearest
neighbour may lie outside the filtered set. `nearest_within` compares only inside
the restricted set and otherwise follows the frozen `nearest` semantics
(boundary distance, non-border anchor, non-border candidates, margin policy).

#### Trivial vs nontrivial Level-3

| candidate count after filtering | meaning |
|---|---|
| `0` | discarded |
| `1` | valid, `trivial_selection = true` — the filter alone determines the answer; no distance comparison is needed |
| `>= 2` and margin passes | valid, nontrivial — a genuine nearest comparison |

`trivial_selection` is recorded per sample and reported separately in the
statistics. **Future reasoning evaluation should prefer the nontrivial subset**,
because a trivial sample does not test distance reasoning at all.

---

## Frozen relation rules

The dataset uses `configs/spatial_relations_v1.yaml` unchanged. The generator
**never overrides a threshold**, and a startup guard fails the run if the
generator config disagrees with the frozen values.

- `ratio_margin = 1.10`
- direction preset `medium` (`alpha = 1.2`, `tau = 0.04`)
- `nearest`: **anchor non-border AND target non-border**
- ambiguous → **discard**, never tie-break

### Predicate convention (frozen, ADR-006)

`relation(subject, object)` means the **subject** satisfies the relation with
respect to the object:

```
left_of(A, B)  ==  "A is left of B"   ==  cx_A < cx_B
```

Because Level 2/3 ask about *"the region to the right of the reference"*, the
**candidate is the subject** and the **reference is the object**:
`right_of(candidate, reference)`.

---

## Sample schema

One JSON object per line. Geometry is deliberately **not** duplicated: polygons,
bboxes, centroids and areas are referenced through `image_metadata_ref`, which
points at the Task 2 metadata as the single geometry source.

```jsonc
{
  "sample_id": "buildsr_train_1_0_2_largest_to_right_of_e794686d4fdb",
  "project": "BuildReasonSeg",
  "dataset_name": "BuildSpatialReason",
  "dataset_version": "v0.1",
  "source_dataset": "WHU Building Dataset",
  "source_subset": "Satellite dataset II (East Asia)",

  "image_id": "1_0",
  "split": "train",
  "image_path": "../WHU_Building_Segment/dataset/WHU_YOLO_dataset/images/train/1_0.tif",
  "component_map_path": "datasets/whu/components/train/1_0.png",
  "image_metadata_ref": "datasets/whu/metadata/train.jsonl",

  "level": 2,
  "query_type": "largest_to_right_of",

  "instruction_zh": "...",
  "instruction_en": "...",

  "reference_component_ids": [1],
  "target_component_id": 2,
  "candidate_component_ids": [2],
  "distractor_component_ids": [3],

  "reasoning_steps": [ /* structured program */ ],
  "reasoning_zh": "...",
  "reasoning_en": "...",

  "target_mask": {
    "representation": "component_map_selector",
    "component_id": 2
  },

  "trivial_selection": false,
  "relation_config_version": "spatial_relations_v1",
  "generator_version": "v0.1"
}
```

### Model-agnostic by design

No architecture-specific token appears anywhere in the dataset. `[SEG]`, `[REF]`
and similar belong to a future concrete MLLM conversation format and are
**not** dataset semantics. Converting this dataset into a specific MLLM
conversation format is a later, separate step.

---

## Determinism

- Template choice: `SHA256(semantic_key | template_version)` bucketed into the
  family. Python's salted `hash()` is never used.
- `sample_id`: `SHA256` over image id, structured program, target id, generator
  version.
- Any quota downsampling: ascending `SHA256(seed | image_id | level | type | key)`.
- Regenerating with the same seed and config produces **byte-identical** JSONL.

## Duplicate policy

One canonical record per **(image_id, structured reasoning program,
target_component_id)**. Wording variants do not create duplicate training
samples.

## Split policy

Queries inherit their source image's split. Query-level random splitting is never
performed, and the `image_id` sets of train / val / test are disjoint.

---

## What this dataset CAN be used to evaluate

- **Direct spatial grounding** — locate a component by a stated spatial extreme.
- **Reference-based reasoning** — resolve a reference, then a relation to it.
- **Multi-hop geometric reasoning** — chain reference → relation filter → nearest.
- **Target segmentation** — predict the mask of the selected component.

## What this dataset must NOT be claimed to provide

- **Physical-building instance ground truth.** Components are connected
  components, not verified physical buildings.
- **Building functional semantics.** There is no use/type/function annotation.
- **General world reasoning.** All queries are geometric and tile-relative.
- **Façade / indoor structure understanding.** The source imagery is nadir
  remote sensing; there is no façade or interior label.
