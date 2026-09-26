# Architecture Decision Records

Decisions already fixed for this project. Each record states the decision, the reasoning, and
what the decision forbids.

Nothing in this file describes implemented code — only binding design constraints.

---

## ADR-001 — YOLO is baseline only

**Status:** Accepted

**Decision**

`YOLOv8m-seg` is retained strictly as a **baseline** for comparison and sanity checking.
The proposed method does **not** import YOLO code, and YOLO is not a backbone, proposal
generator, or trainable component of the proposed pipeline.

**Reasoning**

- Architecturally mismatched: YOLO is a feed-forward per-instance detector/segmenter, whereas
  reasoning segmentation is conditioned on language and must produce a mask for a
  *linguistically described* target.
- Inserting YOLO mid-pipeline would break the differentiable path from the `[SEG]` embedding
  to the mask (NMS and instance discretization are non-differentiable), so `[SEG]` could not
  learn.
- Empirically, the baseline's instance decomposition is unstable at the pixel level. On one
  test tile the baseline split a single building into five overlapping detections at
  confidences 0.74 / 0.65 / 0.55 / 0.53 / 0.41. Importing that instability into the
  reasoning pipeline would inject noise into `[SEG]` supervision.

**Forbids**

- Any `import` of YOLO/ultralytics code inside `models/`, `spatial_reasoning/`,
  `training/`, or `evaluation/`.
- Treating the baseline checkpoint as an initialization for the proposed model.

**Permits**

- Citing the baseline's validation metrics as a performance floor.
- Using it offline, outside the gradient path, in an explicitly labelled ablation, if ever
  needed. This is not the current plan.

---

## ADR-002 — GT geometry is supervision only

**Status:** Accepted

**Decision**

Geometry derived from the GT polygon labels — `centroid`, `area`, bounding box,
`relation`, `polygon` (and any mask derived from them) — may be used **only** as
supervision and evaluation targets. It must **never** be a model input at inference time.

**Reasoning**

Using GT geometry as an input would make the model depend on information that does not exist
when the system is actually used. Any reported metric would then be unattainable in practice.
This is also the failure mode ADR-005 exists to prevent.

**Forbids**

- Feeding GT boxes, GT centroids, GT areas, GT relation matrices, or GT masks into any
  inference-time forward pass.
- Reporting results from a configuration where such inputs were used, without labelling it
  explicitly as an oracle/upper-bound experiment.

**Permits**

- Using GT geometry to *generate* instruction text, relation labels, and target ids.
- Using GT geometry in the loss (this is supervision, not input).
- A clearly labelled **oracle ablation** whose purpose is to bound achievable performance.
  Such a run must never be presented as the main result.

---

## ADR-003 — Current reliable spatial relation candidates

**Status:** Accepted (scoped to the current WHU annotation topology)

**Decision**

Relation priority for `BuildSpatialReason`:

| Tier | Relations |
|---|---|
| **Core (preferred)** | `left_of`, `right_of`, `above`, `below`, `nearest` |
| **Conditional (threshold / margin required)** | `largest`, `smallest`, `near`, `far` |
| **Not used** | `overlap`, `contain`, `inside`, strict-touching `adjacent_to` |

**Reasoning**

This follows from the measured topology of the current WHU labels, **not** from a general
claim about spatial reasoning:

1. All polygons derive from `RETR_EXTERNAL` contours of a **binary** raster mask, so instances
   are mutually disjoint by construction. Measured across 25,454 sampled instance pairs:
   **0 overlapping pairs, 0 pairs with IoU > 0.05, 0 containment pairs.**
   `overlap` / `contain` / `inside` would therefore be permanently empty or permanently false
   — a field that can never fire is worse than an absent field.
2. Because the source raster is binary, **touching buildings were merged into a single
   instance** and cannot be recovered. Measured: 2.601% of instance pairs are within 1.5 px.
   Strict-touch adjacency therefore has no dependable ground truth.
3. `largest` / `smallest` are geometrically well defined but depend on area, and the recorded
   area is inflated for buildings with interior courtyards (holes were filled). The largest
   measured instance covers 49.99% of its tile and is a strong candidate for a merged
   multi-building blob. These relations require a stability margin before use.
4. `near` / `far` are only meaningful relative to scene scale (median instance area 1176 px^2,
   equivalent scale ~34 px). Absolute pixel thresholds do not transfer across images, so any
   such relation must be defined relative to a per-image scale statistic.

**This is a scoped conclusion.** It is a statement about what the current WHU annotation can
support, and must be re-derived if a different or richer annotation source is introduced.

**Forbids**

- Including `overlap`, `contain`, `inside`, or strict-touching `adjacent_to` in any relation
  schema, loss, or metric for the current dataset.
- Using `largest` / `smallest` / `near` / `far` without an explicit, recorded margin or
  threshold and a uniqueness check.

---

## ADR-004 — Instruction semantics must be complete

**Status:** Accepted

**Decision**

Every criterion that determines the target must appear explicitly **in the instruction text**.
The generator may not use a hidden criterion to force uniqueness.

**Forbidden pattern**

```
instruction:  "the building to the right of the largest building"
generator silently applies:  nearest-on-the-right, to make the answer unique
```

The instruction names a set; the generator quietly reduces that set by an unstated rule. The
instruction as written does **not** determine the recorded target, so the sample is
unanswerable and the supervision is wrong.

**Correct pattern**

If the intended disambiguation is proximity, the instruction must say so:

```
"the building closest to the largest building on its right side"
```

**Reasoning**

If the instruction does not uniquely determine the answer, the model is asked to guess an
unstated rule. Gradient signal then teaches the model to reproduce the generator's arbitrary
tie-break, and reported accuracy measures agreement with a hidden convention rather than
spatial reasoning ability.

**Enforcement requirements**

The annotation generator must implement the following, and the validator must confirm it:

1. Enumerate **all** instances satisfying the criteria stated in the instruction.
2. Require exactly one result. If zero or more than one, **discard the sample** — never add an
   unstated tie-break.
3. Record the decision margin, and discard samples whose margin is below a threshold.
4. Record the thresholds used inside the sample, so train/val/test apply identical rules.

**Forbids**

- Any fallback that selects among multiple valid answers.
- Post-hoc filtering to one answer.
- Ties resolved by instance id, array order, or any criterion absent from the instruction text.

---

## ADR-005 — Future Spatial Relation Encoder must be inference-valid

**Status:** Accepted as a constraint; **final design deferred to Task 8**

**Decision**

The Spatial Relation Encoder must operate only on information available at inference time. It
may not consume GT geometry, in accordance with ADR-002.

The final mechanism is **not yet decided**. The candidate direction to be evaluated in Task 8:

```
reference prediction
  -> predicted geometry / spatial prior
  -> target segmentation
```

That is: the model first produces a prediction for a reference object named in the
instruction, derives geometry from *that prediction*, and then uses the derived geometry to
locate the target. The geometry is therefore model-produced, not GT-derived.

**Reasoning**

The obvious way to build a spatial relation module is to feed it a relation matrix computed
from GT instances. That is invalid at inference time, because the set of GT instances is not
available for an unseen image. Any such design would work in training and fail in deployment.

**Open questions to settle in Task 8**

- Where does the reference object come from if the instruction names a set before the target
  is known (e.g. "the largest building")?
- Should reference selection and target segmentation share one forward pass or two?
- How are candidate/reference instances obtained at inference time without reintroducing a
  detector whose output is non-differentiable (see ADR-001)?

**Forbids (now)**

- Implementing a relation encoder that takes GT geometry as input.
- Claiming a spatial relation module exists before this question is resolved.

---

## ADR-006 — Predicate convention is frozen

**Status:** Accepted (Task 3B)

**Decision**

> **`relation(subject, object)` always means: the SUBJECT satisfies the relation
> with respect to the OBJECT.**

| Call | Unique natural-language reading | Geometric meaning |
|---|---|---|
| `left_of(A, B)` | **A is left of B** | `cx_A < cx_B` |
| `right_of(A, B)` | **A is right of B** | `cx_A > cx_B` |
| `above(A, B)` | **A is above B** | `cy_A < cy_B` |
| `below(A, B)` | **A is below B** | `cy_A > cy_B` |

**Reasoning**

An earlier revision implemented the **inverted** convention: `left_of(X, Y)`
held when Y was to the left of X. That contradicts standard predicate semantics
and made every consumer — instructions, reasoning chains, losses, evaluation —
liable to read the relation backwards. Because the whole project downstream of
this layer is built on these predicates, the convention had to be fixed in the
code before any dataset was generated, not documented around.

`boundary_distance` and relation evaluation were unaffected numerically (the
relation set is symmetric under argument swap, so counts barely move), but the
*meaning* changed completely.

**Forbids**

- Re-interpreting the convention anywhere downstream. The dataset generator,
  reasoning chains, MLLM training/prompting, evaluation, and the Spatial
  Consistency Loss must all use "subject satisfies relation w.r.t. object".
- Introducing new offset helpers with ambiguous names. The ambiguous
  `centroid_delta` is removed from the direction path; use
  `subject_to_object_delta` (object − subject) or
  `subject_minus_object_delta` (subject − object), whose names state the order.

**Enforcement**

- `test_predicate_semantics_left_right_absolute_coordinates` and
  `test_predicate_semantics_above_below_absolute_coordinates` pin absolute
  coordinates (centroids at exactly (100, 200) / (300, 200) and (200, 100) /
  (200, 300)).
- `test_round_trip_semantic_mapping` converts each relation to a sentence via
  `relations.describe_relation` and then checks the geometry predicate the
  sentence denotes.
- `RelationResult.evidence` carries the absolute centroids and the explicit
  `subject_minus_object_dx/dy` plus a `relationship` string, so the sign is
  readable from evidence alone.

---

## ADR-007 — `nearest` excludes border components at BOTH ends

**Status:** Accepted (Task 3B)

**Decision**

For `nearest`, **both the anchor and the target** must have
`touches_image_border == false`.

**Reasoning**

`nearest` is the only relation whose metric is a boundary distance. A component
clipped by the tile edge has an incomplete boundary, so the distance measured
from it, and the ordering of distances around it, are unreliable. Admitting a
border anchor would yield supervision that looks plausible but is not
trustworthy.

This resolves a contradiction in the previous revision, which excluded border
*targets* while permitting border *anchors*.

**Forbids**

- Relaxing the anchor rule to gain more `nearest` samples. If coverage drops,
  the drop is reported, not compensated for.

**Does NOT affect**

`leftmost`, `rightmost`, `topmost`, `bottommost`, and the four directional
predicates still accept border components, because they are tile-relative
questions where a clipped component is still a legitimate answer.

---

## ADR-008 — Threshold presets must be monotonic

**Status:** Accepted (Task 3B)

**Decision**

The `strict -> medium -> loose` preset family relaxes **both** direction
parameters, so that

```
Valid(strict)  ⊆  Valid(medium)  ⊆  Valid(loose)
```

| preset | alpha | tau |
|---|---|---|
| `strict` | 2.0 | 0.06 |
| **`medium` (active)** | **1.2** | **0.04** |
| `loose` | 1.0 | 0.02 |

**Reasoning**

The previous preset set was not monotonic: `strict` had `alpha = 2.0` but
`tau = 0.04`, the same as `medium`, so "strict" was not uniformly stricter. A
preset name that does not correspond to a uniform ordering is misleading in
ablations and in any claim about operating points.

**Enforcement**

`test_preset_monotonicity_on_all_pairs` asserts the subset property on synthetic
pairs spanning clean, diagonal and small-margin geometry, and
`test_preset_monotonicity_over_real_dataset_sample` asserts it on real image
pairs from the dataset.

**Forbids**

- Choosing a preset by sample count alone. `ratio_margin` and the presets are
  chosen for supervision reliability; if a relaxation increases sample count but
  drops below the rasterization / discretization discrepancy, it is rejected. See
  `docs/relation_definitions.md` for the `ratio_margin` decision.

---

## ADR-009 — Project naming

**Status:** Accepted

**Decision**

| Kind | Name |
|---|---|
| Project codename / repository | **BuildReasonSeg** |
| Future reasoning dataset | **BuildSpatialReason** (`build_spatial_reason`) |
| Source dataset directory (first source) | `datasets/whu/` |
| Legacy baseline experiment | `WHU_Building_Segment` (unchanged) |
| Legacy baseline record | `baseline/yolo_whu/` (unchanged) |

- **The project is not tied to WHU.** WHU is merely the **first source dataset**.
  Additional public building / remote-sensing datasets are expected to be added
  later, and nothing in the architecture may assume WHU specifically.
- The official Chinese competition title is unchanged. The codename names the
  repository only.
- `SpatialReasoningSeg` is superseded as the repository name and must not be
  used for new files, paths, or documentation.
- `WHU-SpatialReason` / `whu_spatial_reason` are superseded as the **future
  dataset** name; the planned dataset is `BuildSpatialReason` /
  `build_spatial_reason` and lives at `datasets/build_spatial_reason/`.
  It has **not been generated yet**.

**Reasoning**

The previous repository name embedded the research *method* rather than the
*project*, and the previous dataset name embedded the *source dataset* rather
than the *task*. Both would become misleading once a second source dataset is
introduced. Renaming now — before any dataset is generated — avoids a migration
that would otherwise touch every generated artifact.

**Retained names (deliberately NOT renamed)**

These describe real historical artefacts or provenance, so renaming them would
falsify the record:

- `WHU_Building_Segment` — the legacy experiment directory
- `WHU Building Dataset` — the source dataset
- `YOLOv8m-seg-WHU` — the frozen baseline name in `baseline/yolo_whu/manifest.json`
- `baseline/yolo_whu/` — the frozen baseline record path
- `datasets/whu/` — source-specific raw/derived data for the first source dataset

**Forbids**

- Introducing a dependency on the repository name. All program paths resolve
  relative to `__file__` or to the repository root, never from a hard-coded
  directory string, so the repository can be renamed again without breaking code
  or recorded relative paths.

---

## ADR-010 — Semantic visibility policy for natural language

**Status:** Accepted (Task 5B)

**Decision**

1. **The natural-language semantic universe is ALL VISIBLE COMPONENTS.**
   "the largest building region" means the largest component the reader can see
   in the image, not the largest component that happens to survive a hidden
   quality filter.
2. **Eligibility may REJECT a sample; it may never silently CHANGE its answer.**
   If the true semantic answer is excluded by border/tiny/merge quality logic,
   the query is **discarded**. The runner-up is never substituted.
3. **Quality / GT geometry remains annotation and supervision logic, never
   inference input** (restates ADR-002 for the generator).
4. **Reference components are excluded from `distractor_component_ids`.**
   A reference is reasoning context, not a wrong answer.
5. **Natural-language reasoning is ID-free, while structured fields keep IDs.**
   `reasoning_steps` retains numeric component ids for machine verification;
   `reasoning_zh` / `reasoning_en` contain **zero** ids.

**Reasoning**

v0.1 generated all targets with the relation engine, which answers over an
*eligibility-filtered* subset — then rendered those answers in language that
reads as being over *all visible* components. A Task 5 audit measured the
consequence: **7,086 of 32,284 records (21.9%)** carried an instruction whose
plain reading was false. The clearest case selected a component **4.7× farther**
from the reference than the visibly nearest one, while the instruction said
"nearest".

Separately, every one of the 32,284 records leaked internal ids
(`component 2`) into its reasoning text. The input image displays no such ids, so
this trains a model to emit identifiers it cannot ground — an annotation artifact
rather than spatial reasoning.

Because v0.1's generator and its relation engine disagreed about what a question
*means*, v0.1.1 introduces
`spatial_reasoning/semantic_policy.py` as the **single shared implementation**
used by the generator, the target recomputation and the acceptance validator.
One definition, so they cannot drift apart again.

**Scope note.** This decision concerns the *meaning* of an instruction. It does
not alter any relation threshold: `ratio_margin = 1.10`, direction preset
`medium`, and the `nearest` non-border anchor/target rule are unchanged and
remain frozen in `configs/spatial_relations_v1.yaml` (ADR-008).

**Rejected alternative.** Making the eligibility explicit in the instruction
("among building regions that do not touch the image boundary, segment the
largest one") was rejected: tile-edge truncation is an artifact of how the source
tiles were cropped, not a property of buildings or of spatial reasoning. Teaching
a model to condition on it would encode a non-transferable dataset quirk.

**Enforcement**

`tests/test_dataset_validator.py` and `tests/test_annotator.py` assert:
global largest/smallest are never silently replaced; the nearest visible target is
never replaced by a farther eligible one; Level-3 nearest uses the full
direction-valid set; semantic-invalid samples are rejected before acceptance;
reasoning text has no internal ids; `template_id` matches the rendered zh/en
pair; and references never appear in distractors.

---

## ADR-011 — Acceptance semantics must have an independent implementation

**Status:** Accepted (Task 5C)

**Decision**

1. **A shared implementation proves consistency, not truth.** The acceptance
   oracle for BuildSpatialReason is a **second, independent implementation**
   (`spatial_reasoning/semantic_oracle.py`) that must not import, call, or
   delegate to `spatial_reasoning/semantic_policy.py`.
2. **The oracle re-derives the semantics from frozen inputs only**: raw component
   geometry, the frozen relation config, the frozen direction predicate, its own
   recomputation of the component quality flags, and the low-level
   `geometry.component_box_distance` primitive.
3. **Every accepted record must agree with the oracle on all five semantic
   quantities** — semantic reference, direction candidate set, semantic nearest,
   final target and the ambiguity/admissibility decision. A single disagreement
   is blocking (`FAIL_REQUIRES_REVISION`).
4. **Machine-readable artifacts are a single declared truth.** Counts, versions
   and canonical paths are declared once in
   `evaluation/build_spatial_reason_artifact_index.json` and enforced by
   `scripts/check_artifact_consistency.py` against `manifest.json`,
   `statistics.json`, the quality JSON and every Markdown file carrying an
   `ARTIFACT-FACTS` block.
5. **An artifact name must carry the real dataset version.** Evaluation
   artifacts are `build_spatial_reason_<version>_*`; the historical `v011`
   abbreviation is prohibited.

**Reasoning**

v0.1.1 was produced by a *single* shared policy module
(`semantic_policy.py`) that the generator, the recomputation and the validator all
call. That removes drift between them, but it also means the audit is circular: a
wrong shared definition is wrong three times and the audit still reports PASS.
Task 5B's own report illustrated the failure mode in miniature — it published
Level-2 Type A/B as 2,272/2,764 and Level-3 trivial/nontrivial as 1,261/1,657,
while `statistics.json`, `manifest.json` and the quality JSON all said
2,275/2,761 and 1,256/1,662. The data was right; a hand-written artifact was
wrong, and nothing in the repository could tell.

Task 5C therefore adds:
* an independent oracle over all 25,229 records (non-circular acceptance), and
* a consistency gate over every machine-readable and declared artifact
  (self-checking documentation).

**Rejected alternative.** Trusting the shared policy because "the generator and
the validator cannot disagree" was rejected: it converts a testable claim
("the dataset is semantically correct") into an untestable one ("two copies of
the same code agree with each other").

**Enforcement**

`tests/test_semantic_oracle.py` asserts static independence (AST scan of the
oracle source for any reference to `semantic_policy` or an `SP` alias), runtime
independence (every oracle entry point is exercised while a `sys.meta_path`
poison makes importing `semantic_policy` raise), the five semantic identities on
synthetic geometry, full-dataset target agreement, and unchanged JSONL hashes.
`tests/test_artifact_consistency.py` asserts that the gate accepts the real
repository and rejects injected drift, a resurrected `v011` artifact, and a
declared fact that contradicts the authoritative index.

---

## ADR-012 — BuildReasonSeg-MVP model stack

**Status:** Accepted (Task 5.5), with the **`PROVISIONAL`** sub-decisions listed in §Provisional clauses.

**Decision**

1. **Base MLLM: `Qwen/Qwen3-VL-4B-Instruct`** (Apache-2.0, verified from the Hugging Face API licence
   tag and the ModelScope first-party `License` field; BF16; 8.27 GiB of weights).
   **Stages 0–2 run on `Qwen/Qwen3-VL-2B-Instruct`** so the pipeline is proved before the 4B budget is
   spent. Switching is a configuration change: same family, same token ids.
2. **Mask decoder: `facebook/sam2.1-hiera-large`** (Apache-2.0, LICENSE file read first-hand; 1.67 GiB).
   The image encoder and memory module stay **frozen**; only the **mask decoder** trains.
3. **Pathway: LISA-style `[SEG]` for the MVP; `[REF] + [SEG]` from Stage 3.** The `[REF]` **token** is
   acknowledged prior art (SegLLM, PSALM) and is never claimed as a contribution.
4. **Trainable:** LoRA rank 16 on `q,k,v,o,gate,up,down`, the `[SEG]`/`[REF]` embedding rows, the
   projection MLP, and the SAM2 mask decoder. **Frozen:** vision tower, LLM base weights, SAM2 image
   encoder and memory module.
5. **Precision:** BF16 LoRA. 4-bit QLoRA is the OOM fallback only.
6. **Image policy: native 512×512, no upscaling** (256 visual tokens per tile), with the pixel budget set
   **explicitly** because Qwen3-VL's released preprocessor config carries no `min_pixels`/`max_pixels`.
7. **Environment: native Windows 11 in a new dedicated conda environment** (`yolo_sam_env` is never
   modified), PyTorch SDPA rather than FlashAttention. **WSL2 Ubuntu is the documented fallback.**
8. **No external dataset is merged into training.** Warm-up is unnecessary; the licence-clean candidates
   (EarthReason, RefSegRS) are reserved for evaluation.
9. **The Spatial Relation Encoder is the lead contribution**, and the MVP must not block it: the
   projection MLP and the relation encoder remain separate modules, and every inference-time geometric
   quantity is derived from the model's own predicted masks.

**Reasoning**

The MVP must be buildable on a single 16 GB laptop GPU by one person under competition time pressure.
That constraint, plus a strict licence policy, selects the stack almost uniquely:

* Every component on the critical path has a **verified** licence and a **public** reference recipe
  (LISA for the `[SEG]` mechanism; Sa2VA for the `[SEG]`-hidden-state → projection → SAM2 pattern on the
  *same* Qwen3-VL base family).
* The local Blackwell question is already **closed by measurement**: the installed
  `torch 2.13.0+cu132` advertises `sm_120` and initialises the RTX 5080 Laptop GPU, and the total VRAM is
  15.89 GiB. So the usual reason to move to Linux — PyTorch not supporting the new GPU — does not apply.
* FlashAttention, the usual *other* reason to move to Linux, is **optional** here and has no official
  Windows support, no prebuilt wheels and no `sm_120` in its documented GPU list. At ~768-token
  sequences with a 256-token image prefix it offers the least benefit of any regime, so it is not
  allowed to dictate the operating system.
* `Qwen2.5-VL-3B-Instruct` was the most attractive size on paper and is **excluded on licence grounds
  alone**: no reachable first-party source states a licence for its weights.

The novelty audit (Task 5.5 §6) forces the architectural emphasis: `[SEG]` is ubiquitous and `[REF]` is
already published, so the defensible contribution is the **explicit relation pathway over predicted
region geometry**, backed by a dataset whose relations can be re-verified geometrically.

**Provisional clauses (`PROVISIONAL` — do not treat as settled)**

| Sub-decision | Why provisional | What resolves it |
|---|---|---|
| 4-bit QLoRA as the OOM fallback | `bitsandbytes` documents Windows 11 + QLoRA 4-bit, but the combination with `sm_120` is not documented | Stage 0/3 measurement on the target machine |
| WSL2 as the fallback environment | SAM 2's Windows build status rests on non-first-party text; its compiled extension is optional but its necessity here is unmeasured | Stage 0 installation attempt |
| Any future adoption of Sa2VA code | the repository's **code licence** was not verifiable at research time | reading the repository LICENSE from a machine with unimpeded GitHub access |
| The narrow dataset-novelty claim | "no public dataset verified to combine instance-level building masks with multi-hop spatial instructions" is an evidenced absence; the ISPRS Annals XI-2-2026 building-VLM paper was read only at abstract level | reading that paper's full text |

**Enforcement**

`evaluation/task5_5_stack_decision.json` is the machine-readable form of this ADR.
`docs/research/task5_5_mvp_decision.md` §2.2 lists the Task 6 entry criteria, including the licence and
weight-acquisition checks that must pass before any implementation begins. Task 6 must not begin from a
different base model without superseding this ADR.

---

## ADR-013 — Task 6A measured MVP proof stack

**Status:** **Accepted** (Task 6A). Acceptance was conditional on Stage 1 succeeding,
and Stage 1 passed all nine of its checks; the Stage 2 outcome is recorded in
`evaluation/task6a_smoke_report.json` and summarised in §Outcome.

**Decision**

1. **The measured MVP proof stack is Qwen3-VL-2B-Instruct + SAM2.1 Hiera Base+
   with a single `[SEG]` token.** Measured on the target laptop: 2.223 B total
   parameters, 24.0 M trainable (1.08 %), 256 visual tokens per 512×512 tile,
   sequence lengths 295–331, idle combined VRAM 5.10 GiB, Stage 1 peak 7.25 GiB
   allocated / 8.45 GiB reserved.
2. **Qwen3-VL-4B remains UNMEASURED and therefore still `BORDERLINE_LOCAL`.**
   Task 5.5 estimated 11.6–12.6 GiB for 4B BF16 LoRA; Task 6A did not test it, so
   that estimate is unchanged but unconfirmed. Nothing in Task 6A authorises
   treating 4B as proven.
3. **`[SEG]` is selectively trainable, and the tying is preserved.** PEFT 0.21
   `trainable_token_indices` wraps the input embedding **and** the tied `lm_head`, and the two adapters
   share the **same delta parameter object**, so there is exactly **one** trainable vocabulary row used
   on both sides — which is what "tied embeddings" should mean. The full 151,670 × 2048 embedding table
   stays frozen, and after an optimizer step the `[SEG]` row changes while six sampled ordinary
   vocabulary rows move by exactly 0.0. Two measured details are worth keeping: the optimizer covers all
   528 trainable tensors with none missing, and `resize_token_embeddings` **shrinks** Qwen's padded table
   (151,936 rows → 151,670) rather than only appending a row.
4. **The Qwen and SAM2 image branches are separate and measured separately.**
   One 512×512 RGB tile becomes 256 Qwen visual tokens (`image_grid_thw =
   [1,32,32]`, patch 16, merge 2) **and** a 1×3×1024×1024 normalized SAM2 tensor
   producing a 1×256×64×64 image embedding plus high-resolution features of
   1×32×256×256 and 1×64×128×128. The earlier Task 5.5 phrase "512×512
   everywhere" is explicitly retracted.
5. **The SAM2 optional compiled CUDA extension is intentionally disabled on
   native Windows** (`SAM2_BUILD_CUDA=0`, SAM2 revision `2b90b9f5…`), because no
   system CUDA toolkit is installed and the extension only affects mask
   hole/sprinkle post-processing.
6. **Task 6A measurements supersede every overlapping Task 5.5 estimate.** Where
   the two disagree, the measured value wins.

**Reasoning**

Task 5.5 produced a design; Task 6A produced the first numbers measured on the
real machine. Three of them changed the picture:

* **The image-budget estimate was right, but for a reason worth recording.**
  Task 5.5 predicted 256 visual tokens from the config; the processor confirms it,
  and the released preprocessor config carries no top-level
  `min_pixels`/`max_pixels` — the budget lives on `image_processor.size`, which is
  where the explicit 512×512 setting is applied.
* **`[SEG]` trainability had a subtlety.** The naive reading of "tied embeddings"
  suggests the output row must be trained separately. In practice PEFT's token
  adapter covers both sides, and the output-side delta is created lazily on the
  first forward — which is why the trainable parameter count is 2048 before any
  step and 4096 afterwards. Recorded so this is not "rediscovered" later.
* **Instruction dependence is real and measurable.** On the fixed 20-sample set
  (10 images × 2 different targets), every prediction overlaps its own ground
  truth far more than the other target on the same image (typically ≈0.9 vs
  0.000). This is the evidence that the mask pathway consumes the instruction
  rather than an image-only shortcut.

**Consequences**

* Task 6B may start from this stack; it must not silently switch to 4B without
  measuring it first.
* The `[SEG]` token is plumbing, not a contribution (ADR-011 and the Task 5.5
  novelty audit); the reference pathway is still future work.
* Estimates that Task 6A replaced are labelled as superseded rather than deleted,
  so the difference between design and measurement stays visible.

**Outcome**

Stage 0, Stage 1 and Stage 2 all passed. Stage 2 (deterministic 20-sample overfit) finished at 2,000
optimizer steps with **all 20 samples ≥ 0.9366 training mIoU (mean 0.9807)**, **10/10** same-image paired
instruction dependence and **20/20** free-generation `[SEG]` emission. Two corrections made during the
task are recorded rather than hidden: the paired-instruction criterion was initially implemented wrongly
(it reported 0/10 because it compared the two predictions against each other instead of each prediction
against both ground truths), and the primary IoU interpolation was aligned with SAM2's own
`postprocess_masks` (bilinear) after reading the installed source, with the stricter nearest view also
reported. Full numbers, every recipe revision and the runs that did **not** meet the gate are in
`evaluation/task6a_smoke_report.json`, `docs/task6a_mvp_smoke.md` and `handoff/FROM_DSH.md`.

**What Task 6A did not establish.** The assistant reasoning text is not reproduced (assistant token
accuracy ≈ 0), and the model emits `[SEG]` on 0/4 unseen test records. This is an overfit pipeline proof,
not a generalisation or language-quality result.

---

## ADR-014 — Task 6B: the `[SEG]`→prompt interface is the MVP bottleneck, not the mask decoder

**Status:** **Accepted** (Task 6B). This is a measurement ADR: it records what a real 2B mini-train
established, because it changes where the next iteration should spend effort.

**Context.** Task 6A proved the pipeline end to end on a 20-sample overfit (mean training mIoU 0.9807,
10/10 paired instruction dependence). Task 6A could not say whether that was generalisation or
memorisation. Task 6B ran the first real mini-train — 480 train / 120 val / 20 paired val images, drawn
deterministically from BuildSpatialReason v0.1.1, with no test-split contact — on the frozen ADR-013
stack.

**Decision**

1. **The language pathway generalises; the mask pathway does not.** On 120 unseen val records the
   headline recipe reaches valid `[SEG]` emission **120/120**, generated reasoning exact match **1.000**
   and operation-chain accuracy **1.000**, but strict end-to-end mIoU only **0.1102** (Dice 0.1807)
   against 0.98 on the 20-sample overfit. Teacher-forced mIoU is identical to free-generation mIoU to
   1e-16, so the mask error is not a generation problem.
2. **The measured mechanism is a collapsed prompt interface.** Along
   `[SEG] hidden → Projection MLP → 256-d sparse prompt → SAM2 mask decoder`, the cosine similarity of the
   `[SEG]` hidden state for two *different instructions on the same image* is 0.9656 (headline run) /
   0.9952 (adjusted run), versus 0.9975 / 0.9996 for two *different images*; after the projection the
   256-d prompt is a near-constant vector (cosine 0.999952 within an image and 0.999997 across images in
   the headline run), and the two decoded masks have IoU 0.9970. The paired instruction probe therefore
   scores **0/20** with mean own-target IoU 0.13885 versus mean cross-target IoU 0.13873. The mask decoder
   is not the failing component; the prompt it is handed carries no referent identity.
3. **The failure is not a loss-weighting or learning-rate artefact.** The single bounded section 12
   adjustment — Task 6A's mask-dominant weighting (`lm_ce 1.5 / mask_bce 4.0 / mask_dice 3.0`) plus a
   3.3× mask-pathway learning rate — made the result slightly **worse** (strict e2e mIoU 0.11018 →
   0.09498, paired probe still 0/20). The training mask Dice loss stayed in 0.65–1.00 in every epoch of
   both runs, the signature of a near-all-background prediction on small targets.
4. **The gradient budget is dominated by the mask decoder.** At the first Phase-B step the gradient norms
   were `sam_mask_decoder 114.3`, `projection 13.2`, `seg_token 1.57`, `lora 1.04` against a clip norm of
   1.0, so the decoder consumes ≈99 % of the clipped budget while the two places a referent-specific
   prompt could come from receive ≈11 % and <1 %.
5. **Language metrics on this dataset cannot support a reasoning claim.** `reasoning_zh` is
   template-generated: the whole 480-record mini-train contains **21 distinct reasoning strings** (L1 6,
   L2 11, L3 4) and every one of the 20 distinct val strings also occurs in train. Training LM CE
   therefore collapses to ~0 within 40 optimizer steps. `operation_chain_accuracy` is the only language
   metric here that carries information.
6. **Task 6B's verdict is `FAIL_REQUIRES_DEBUG`.** Nine of the ten section 17 minimums are met; paired
   instruction dependence (0/20 against a ≥14/20 bar) fails as badly as it can, and section 17 names
   "pair generalization fails badly" as a debug-required condition.
7. **Next effort belongs at the prompt interface, not at model scale.** The collapse sits at the
   `[SEG]`→prompt interface, which a 4B base MLLM would plausibly reproduce, so a 4B run is not the fix;
   it is only a scale measurement once the interface is fixed.
8. **Phase B is not bit-reproducible, and the +0.10 improvement bar is therefore marginal.**
   `training.deterministic: true` is never consumed — `set_seed` is called but
   `torch.use_deterministic_algorithms` / `torch.backends.cudnn.deterministic` are not — so re-running the
   identical headline recipe reproduced the baseline and Phase A bit-identically and Phase B only
   approximately (epoch-1 mIoU 0.1008 vs 0.0932; best-epoch 0.11018644 vs 0.11017864). The +0.10 bar was
   met by both executions, but the epoch-to-epoch spread is larger than the margin, so the criterion should
   not be read as comfortable. The flag was left alone because enforcing it would change the numerics of an
   already-recorded recipe; it is the first recommended change for Task 6C.

**Consequences**

* The paired instruction-dependence probe stays the primary gate for the MVP. Strict end-to-end mIoU
  alone presented this failure as a modest success (+0.110 absolute over baseline); the paired probe
  exposed it as a total failure of instruction conditioning.
* Any future prompt design must be justified by a measurement at this interface
  (`scripts/task6b_diagnose_prompt.py`), not by the final mIoU.
* The three Task 6A numbers that Task 6B supersedes are: 20/20 emission no longer implies generalisation,
  0.98 mIoU is an overfit ceiling rather than a capability, and "assistant token accuracy ≈ 0" was partly
  a measurement bug (see below) rather than only a frozen-LM artefact.
* Task 6B found and fixed two measurement defects, recorded rather than silently restated: the
  teacher-forced token accuracy indexed `lm_logits` one position early (baseline reasoning-token accuracy
  was 0.4522, reported as 0.0160), and the teacher-forced path hard-coded its text-quality fields to
  zero by construction.

---

## Summary

| ADR | Decision | Primary constraint |
|---|---|---|
| 001 | YOLO is baseline only | Proposed method does not import YOLO code |
| 002 | GT geometry is supervision only | Never a model input at inference time |
| 003 | Relation set limited by data topology | No `overlap` / `contain` / `inside` / strict adjacency |
| 004 | Instructions must be semantically complete | No hidden tie-breaks; ambiguity causes discard |
| 005 | Relation encoder must be inference-valid | Design deferred to Task 8 |
| 006 | Predicate convention frozen | `relation(subject, object)` = subject satisfies it w.r.t. object |
| 007 | `nearest` excludes border at both ends | Boundary metric needs a complete boundary |
| 008 | Presets monotonic | Valid(strict) ⊆ Valid(medium) ⊆ Valid(loose) |
| 009 | Project naming | `BuildReasonSeg`; not tied to WHU; WHU is the first source dataset |
| 010 | Semantic visibility policy | Language is over ALL visible components; eligibility may reject but never silently change an answer |
| 011 | Acceptance needs an independent oracle | The oracle never imports `semantic_policy`; artifacts are checked against one declared truth |
| 012 | **MVP stack frozen** | Qwen3-VL-4B-Instruct + SAM 2.1 hiera-large + BF16 LoRA `[SEG]`, native Windows, no external training data; `[REF]` is prior art, the relation encoder is the contribution |
| 013 | **Task 6A measured proof stack** | 2B + Base+ measured working; 4B still unmeasured; `[SEG]` selectively trainable on both tied rows; separate Qwen/SAM preprocessing; SAM2 CUDA extension disabled |
| 014 | **Task 6B: the `[SEG]`→prompt interface is the MVP bottleneck** | Language generalises (120/120 emission, 1.000 chain accuracy) but the mask does not (0.1102 mIoU, paired probe 0/20); the projection collapses to a near-constant 256-d prompt; loss weighting and mask LR do not fix it, so effort goes to the interface, not to 4B |
