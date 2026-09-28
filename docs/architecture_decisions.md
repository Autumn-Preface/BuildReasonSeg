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

> **Status amendment (Task 6C, section 3): the *causal* diagnosis in this ADR is PROVISIONAL.**
> Every measurement below is preserved unchanged. What is provisional is the attribution — the claim
> that the `[SEG]`→prompt interface is *the* bottleneck. Task 6B had two confounds that a single
> training run could not separate:
>
> 1. **Unique-image training confound.** All 480 training records came from 480 *distinct* images, so
>    the subset never forced two different instructions on the *same* image to select two different
>    targets. A model can minimise the objective without ever learning instruction-conditional
>    selection, and Task 6B's own 0/20 paired result is consistent with that.
> 2. **Fixed positive-centre prompt confound.** The SAM bridge always injected a real point prompt at
>    the tile centre with `label = positive` and then *added* the projected language vector to that
>    point embedding. That is a genuine spatial prior, not a neutral or "content-free" placeholder,
>    and it can dominate the language contribution.
>
> `docs/task6c_prompt_ablation.md` and ADR-015 report the controlled 2×2 that separates them:
> training-sample structure (U = unique images, P = 240 images × 2 counterfactual instructions) ×
> SAM bridge (C = the Task 6B centre-positive prompt, L = language-only sparse token). The Task 6B
> verdict and every number in this ADR remain as measured; only the causal reading is downgraded.

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

## ADR-015 — Task 6C: the instruction-conditioning failure is not sampling and not the centre point

**Status:** **Accepted** (Task 6C) for what the 2×2 experiment measures; nothing here is claimed beyond it.

**Context.** ADR-014's causal reading was provisional because Task 6B could not separate two confounds: a
training subset drawn from 480 *distinct* images (so two instructions never had to pick two targets on one
image), and a bridge that injected a fixed positive point at the tile centre. Task 6C ran the controlled
2×2 — sampling {U = Task 6B's 480 unique images, P = 240 images × 2 counterfactual instructions} × bridge
{C = the Task 6B centre-positive prompt, L = language-only sparse token} — with one recipe, one seed, one
validation set and one metric path.

**Decision**

1. **The experiment is valid and reproducible.** All four arms start from the same initial trainable
   fingerprint (`97daa58a4cff0ae9…`, 528 tensors, 24,010,309 parameters), write to four distinct
   config-driven checkpoint directories, and run in strict deterministic mode
   (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) whose cross-process bit
   reproducibility is demonstrated on a two-sample forward/backward plus optimizer step. `training.
   deterministic` is now consumed; in Task 6B it was a dead flag.
2. **Neither factor fixes instruction conditioning.** All four arms score **0/20** on the paired unseen
   validation probe with mean own-minus-cross margins between −0.0035 and +0.000007, against a required
   > 0.05, and strict end-to-end mIoU between 0.0928 and 0.1087 against a 0.11 bar. Paired counterfactual
   training, the removal of the fixed centre point, and their combination all fail.
3. **Paired training does change the representation, without creating conditioning.** The projected
   prompt's effective rank rises from 1.50 (U arms) to 3.74 (P arms) and its top-1 variance share falls
   from 0.905 to 0.616, while the same-image projected cosine stays above 0.99999 and the
   prediction-to-prediction IoU stays at 0.999. The projection uses more dimensions; two instructions on
   one image still produce the same mask.
4. **The fixed positive centre point is not the dominant cause.** For the C arms the point embedding's norm
   is 11.39 against a projected language norm of 266.8 — a ratio of 23.4× in the language vector's favour —
   and removing the point prompt entirely (the L arms) leaves the paired probe at 0/20 and does not move
   the prompt geometry materially (effective rank 2.43 vs 2.81). Task 6B's bridge was a real spatial prior,
   but it was not what suppressed the language signal.
5. **The collapse is directional and low-rank, not a constant.** Same-image projected cosine > 0.9999 with
   a top-1 variance share of 0.59–0.91 and a non-zero effective rank (1.5–4.1). Wording that Task 6B used
   loosely is fixed here: *directional collapse / near-constant direction*, not "the prompt is constant".
6. **The remaining problem is after the prompt.** Since the sampling scheme and the point prior are both
   excluded, what is left is how the `[SEG]` hidden state is formed and how SAM's decoder turns a prompt
   direction into a region. Prompt normalisation, a multi-token prompt, auxiliary point supervision or
   `[REF]` become justified *now*, and not before.
7. **Verdict: `EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`.** No arm clears the fix gate; `P_C` and `P_L`
   materially improve the prompt representation over `U_C` (effective rank +2.58 and +1.85, top-1 variance
   −0.313 and −0.259).
8. **The Task 6B measurements stand unchanged**, including the 0.11018 strict e2e mIoU, the 0/20 paired
   result and the 0.99995 projected cosine; only Task 6B's *attribution* is superseded, and it is
   superseded in the direction of "later in the pipeline", not "somewhere else entirely".

**Consequences**

* The paired instruction-dependence probe remains the primary gate, and strict end-to-end mIoU remains a
  weak signal: the arms that changed the representation most had *lower* mIoU (`P_L` 0.0928) than the
  baseline (`U_C` 0.1052).
* A 4B scale-up is still not justified by anything measured here. Nothing in Task 6C suggests that more
  parameters would fix a prompt that is not instruction-conditional.
* Task 6C's four correctness fixes (real determinism, config-driven per-arm checkpoints, train-only
  operation lookup, and a 480-image feature cache inside an 8 GiB budget) are prerequisites for any future
  ablation and should be kept.
* Language metrics remain template metrics: `reasoning_zh` has 21 distinct values in the whole training
  mini-set, so reasoning exact match and operation-chain accuracy are format measurements, and the mask
  target selection on paired unseen images stays the reasoning evidence.

---

## ADR-016 — Task 6E: making the box an explicit language target is necessary but not sufficient at 2B/3-epoch scale

**Status:** **Accepted** (Task 6E) for what the E0/E1 experiments measure. No novelty is claimed for coordinate tokens themselves.

**Context.** Task 6D.1 established (ADR-014/015 lineage) that the frozen `[SEG]` hidden state of the Task 6C `P_C` checkpoint contains no *practically decodable* target geometry, while the downstream machinery is sound: given the correct box, SAM2 reaches 0.7506 mIoU / paired 20/20. Task 6E replaced the implicit `[SEG] hidden → geometry` readout with an explicit autoregressive target `reasoning [BOX] <loc_x1> <loc_y1> <loc_x2> <loc_y2> [SEG]` over one shared 256-token location family, with `L = 1.0·L_assistant + 5.0·L_location` and no mask loss.

**Decision**

1. **The quantized-box pathway is viable.** Deterministic enclosing quantization at `B = 256` (the smallest bin count whose oracle keeps paired 20/20 within `oracle − 0.03` mIoU) recovers **0.7343 mIoU / paired 20/20** on the official frozen SAM2 box prompt, versus 0.7506 continuous and 0.10604 for Task 6C `P_C`. Quantization is not the bottleneck (0/120 validation samples have a quantized GT box below 0.5 IoU against the continuous box).
2. **The tokenizer/row/loss implementation is verified, not assumed.** 257 new tokens (each single-id and round-tripping); 258 trainable rows through PEFT `trainable_token_indices` with the base table frozen; a real one-step smoke shows `[SEG]`, `[BOX]` and the four target `<loc_*>` rows moving while 16 ordinary rows and the base embedding tensor are bit-identical, and every new **output** row receiving gradient from an output-head-only loss (so the tokens are emittable). Causal label positions are asserted against `input_ids[i+1]`, and the parser reads token ids, never text.
3. **E0 overfits by construction.** 20 records (10 same-image/different-target pairs) reach **20/20 structural, 20/20 exact four-token sequences and 0.9166 mean box IoU** in 500 steps; per-coordinate token accuracy is 1.0, so the residual is the quantization ceiling, not model error.
4. **E1 fails on the location values, not on the format.** On the 480-record paired `P` subset, 3 epochs × 480 steps: structural validity **0/120** at every epoch. Teacher forcing reaches reasoning 1.000, `[BOX]` 1.000 and EOS 1.000, but location-token accuracy only 0.021 → 0.058 → 0.075 with `[SEG]` accuracy **0.000**; free generation emits exactly one `[BOX]` and then a 51–81-token location run with no `[SEG]`. The training location CE stalls at 5.15 against the 256-way uniform floor `ln 256 = 5.545`, i.e. the objective does not fit even its own training data inside the fixed budget — while the *same* code path, loss and schedule overfit the first 20 of those 480 records (E0).
5. **Verdict `EXPLICIT_SPATIAL_TOKENS_FAILED`; E2 not run** (section 14 gates it on the E1 gate). SAM2 was never trained, no GT geometry entered any prompt, and no escalation (`[REF]`, SRE, SCL, 4B, dataset change, full training, GUI) was attempted.
6. **The dataset is not the constraint.** Error classification finds 120 structural and **0 localization** failures, 0 quantization-limited samples, and `whu_data_quality_dominates: null`.

**Consequences**

* "Make geometry an explicit language target" is **necessary but not sufficient**: it removes the decodability question entirely, and it exposes a vocabulary-learning problem that the hidden-state route never had. The next task should attack that signal/budget question (loss weighting or a `[BOX]`-first curriculum, coarse-to-fine bins, or a location-only logit head over the existing 256 tokens) before adding new modules.
* The 0.7343 mIoU / 20/20 quantized-oracle ceiling is the number to beat, and it is a *box-quality* ceiling: any future geometry improvement can be measured there without running SAM2.
* The `[SEG]` hidden-state route stays closed as diagnosed by Task 6D.1; Task 6E does not reopen it.
* Coordinate tokens are not the paper's novelty claim; geometry-verifiable supervision, reference/relation mechanisms, SRE and SCL remain the intended contribution and are now testable against a verified geometry *interface*.

---

## ADR-017 — Task 6F: the target-aware `[BOX]` query token — pre-reasoning placement, box readout, and the F1 outcome

**Status:** **Accepted** (Task 6F) for what the F0/F1 experiments measure. No novelty is claimed for the query token itself.

**Context.** Task 6E retired the autoregressive coordinate vocabulary (E0 overfits 20 records; E1 fails 0/120 structurally). Task 6F keeps the box an explicit target but forms the representation in a dedicated query token placed **before** reasoning: `image + instruction → [BOX] → reasoning_zh [SEG] EOS`, where `[BOX]` is a fixed learned input token (never predicted) and its hidden state feeds the same simple readout Task 6D used (`LayerNorm → Linear → GELU → Linear → sigmoid → min/max`) to regress `(x1,y1,x2,y2)` under `L = 1.0·L_reasoning + 5.0·L_box` (SmoothL1, no mask loss).

**Decision**

1. **The query hidden is provably conditioned only by its allowed context.** Structurally, `[BOX]` sits at `prompt_length` and is never a label; functionally, replacing the whole reasoning tail with unrelated text leaves the `[BOX]` hidden **bit-identical** on the real model (recorded in `evaluation/task6f_token_setup.json`). This is the direct fix for the confound Task 6D's end-of-reasoning `[SEG]` readout had.
2. **F0 proves the mechanism.** The same 20 records Task 6E E0 used reach **train box IoU 0.9046, geometry paired 10/10 and non-identical same-image boxes (L1 0.2285)** at step 1000, on the inference-form query path, with no GT leakage. The 6-tensor head (1 055 236 parameters) and the `{[SEG], [BOX]}` rows are the only new trainables beyond the text LoRA.
3. **F1 underfits: `TARGET_AWARE_QUERY_FAILED`.** On 480 paired records (8 epochs × 480 steps, one corrected cosine horizon), the box head never fits even its own training set (train SmoothL1 ≈0.003 ≈ 8 % RMS coordinate error; val box IoU **0.025**, center-inside 0.017, geometry paired **0/20** at the selected final epoch). The predictions are canonical and spread (per-coordinate spread 0.05 → 0.25) but wrongly placed on inherently tiny targets (mean GT area 1.2 % of the tile; 93 % of failures carry `tiny_target`). The language side is untouched (exactly-one `[SEG]` 1.000, EOS 1.000), and F2 is not run.
4. **The query is image-dominated, not target-dominated.** Same-image/different-instruction centered cosine **0.626** vs **0.018** across images; hidden-distance vs GT-box-distance correlation **0.476** (vs ≈0 for the legacy `[SEG]`); effective rank 7.07 (legacy 8.33). Moving the query before reasoning removed the template confound but replaced it with image domination — measurable target variation exists, but it is far below the ~5 % placement accuracy WHU-scale targets require.
5. **Verdict:** see `evaluation/task6f_verdict.json` (section 17). Task 6E machinery stays retired either way: no `<loc_*>` vocabulary, no location CE, no location-run parsing, no Task 6E checkpoint initialization.

**Consequences**

* Whatever the F1 gate outcome, the causal placement machinery (query at the generation prefix, box head on its hidden, bit-identity proof) is a reusable, verified interface for any future geometry readout — including geometry-verifiable supervision and relation-aware heads.
* The Task 6D/6D.1 `[SEG]` readout remains retired as diagnosed; the Task 6E coordinate vocabulary remains retired as measured.
* SAM2 stays the frozen oracle/downstream path: 0.7506 (continuous box) and 0.7343 (Task 6E B=256 quantized box) remain the ceilings to compare any predicted-box segmentation against.
* Coordinate tokens, query tokens and box heads are none of them the paper's novelty claim; they are interfaces for the project's intended contribution (relation reasoning with verifiable geometry).

---

## ADR-018 — Task 6G: dense query-to-visual-map grounding does not localize either — the query's instruction signal is the measured bottleneck

**Status:** **Accepted** (Task 6G) for what the grid oracle and the G0 audit measure. No novelty is claimed for dot-product query-to-map fusion (section 19).

**Context.** Task 6F's global box regression failed on 480 records with an image-dominated query (same-image centered cosine 0.626, hidden-vs-box correlation 0.476). Task 6G tested whether retaining the 2-D visual grid fixes the failure: the causally clean pre-reasoning `[BOX]` query is matched against the frozen SAM2 dense feature map (`q·K/√128` heatmap) with `1.0·L_reasoning + 2.0·(BCE + SoftDice)`, and the predicted point is the heatmap argmax cell centre.

**Decision**

1. **The point-oracle pathway is verified at every grid.** Snapping the frozen Task 6D interior point to the nearest cell centre costs essentially nothing at 256×256 (mIoU 0.4883 vs 0.4876 continuous, paired 18/20, mean displacement 0.50 px); 64/128 pass the mIoU bar (0.4950/0.4894) but drop a paired image to 17/20. Selected grid: **256** (the 32-channel SAM2 high-res feature, verified by spatial size).
2. **The dense head is implemented faithfully and audited, not assumed.** One-step smoke: both token rows and all head parameters move with non-zero gradient, 16 ordinary rows and the base embedding stay exactly unchanged, **SAM2 stays bit-identical** through a training step, and the query hidden stays bit-identical under future-text mutation. The G0 audit adds: train/eval query-path equivalence (bit-identical hiddens), healthy logits (std 3.3, no sigmoid saturation), and comparable BCE/Dice gradient norms (0.395 vs 0.456) — the Dice supervision is not vanishing.
3. **G0 fails anyway: the map does not localize 20 memorized records.** Inside-target 0.10, heatmap Dice 0.125, paired point selection 0/10 after 1500 steps; the heatmap is a near-flat field (sigmoid mean 0.013 ≈ target occupancy; peakiness 0.086) whose argmax is image-content noise — errors 44–225 px, and one image's two instructions pin the identical cell.
4. **Verdict: `DENSE_GROUNDING_IMPLEMENTATION_FAILED`** (section 11 — G0 cannot pass after the audit). G1, the representation diagnosis and G2 are not run; no SAM2 training, no `[REF]`/SRE/SCL/4B/dataset change/GUI anywhere in the task.

**Consequences**

* The bottleneck is now measured as a *query-signal* problem, not a *readout-format* problem: four readouts (6D.1 `[SEG]` probes, 6E coordinate tokens, 6F box regression, 6G dense map) all fail the same way — a query representation whose instruction-dependent component (r ≈ 0.48, image-dominated cosines) is too weak to select the target, in whatever output space it is read. Further readout engineering alone is unlikely to help; the next task should strengthen the query's target-specificity itself (multi-token/instruction-aware querying, stronger counterfactual supervision, or a representation objective) or revisit the training budget the specs have capped.
* The grid-snapped point oracle (0.4883/18-20 at 256) is a cheap, frozen ceiling for any future localization readout, and the `evaluation/task6g_g0_audit.json` equivalence/saturation/gradient checks are a reusable implementation-audit template.
* All Task 6E/6F machinery stays retired; the `[BOX]` query placement and causal-placement proof remain the shared, verified interface for the next candidate.

---

## ADR-019 — Task 6H: an own-vs-cross region-ranking loss on unbounded logits is satisfiable without localizing

**Status:** **Accepted** (Task 6H) for what the pair-manifest, H0 and the focused audit measure. No novelty is claimed for same-image counterfactual supervision (section 21).

**Context.** Task 6G's dense head was implementation-clean yet localized nothing (G0 inside 0.10, Dice 0.125). Task 6H kept the architecture frozen and changed only the training semantics: one optimizer step per canonical same-image counterfactual pair (240 pairs over 240 images from the Task 6C `P` subset, identity hash recorded, 0/240 overlapping), adding `L_total = 0.5*(L_reasoning_A+L_reasoning_B) + 1.0*(L_heatmap_A+L_heatmap_B) + 2.0*L_cf` with `L_cf = 0.5*(softplus(1-(s_AA-s_AB)) + softplus(1-(s_BB-s_BA)))` and `s_XY = sum(H_X·M_Y)/sum(M_Y)` on the 256×256 soft target masks.

**Decision**

1. **The pair machinery is correct and verified.** The one-pair step shares one frozen SAM2 feature (bit-identical through the step), runs both query forwards with retained graphs and a single backward; SAM2 stays bit-frozen, the `[BOX]`/`[SEG]` rows and all head parameters move with gradient, the Task 6G head is unchanged (270 593 parameters, level `[1,32,256,256]`), and the scheduler horizon counts pair steps. The pair loss has the correct sign (own-score up lowers it, cross-score up raises it) and its gradient reaches every trainable group (query projection, query norm, visual 1×1, Qwen LoRA, token rows).
2. **H0 fails on the point gates while passing the ranking gates.** After 1500 pair steps: pair ranking **10/10**, strict margin **10/10**, mean own−cross margin **+5.58** — but point-inside-own **3/20**, paired point selection **0/10**, heatmap Dice **0.134**.
3. **The audit explains the failure precisely: the specified margin is scale-degenerate.** Between clean initialization and the H0 checkpoint the mean absolute logit grows 0.148 → **16.32**, the logit margin grows −0.0001 → **+5.58**, but the *probability* margin only reaches **+0.112** and the point-inside rate **0.15**. The pair loss is minimized (1.31 → 0.021) by magnifying the logit field, not by sharpening or relocating the peak; A/B separation explodes by magnitude (projected-query distance 1.49 → 779.5) rather than becoming target-specific. 10/10 failing pairs are classified `pair_ranking_correct_point_outside`.
4. **Verdict `COUNTERFACTUAL_QUERY_SIGNAL_FAILED`; the task stops at H0** (section 12). H1/H2 are not run; no SAM2 training; no `[REF]`/SRE/SCL/4B/dataset change/GUI.

**Consequences**

* **A ranking objective on unbounded logits is not a localization objective.** Any future pair/contrastive term for this project must be defined on a bounded scale (probability/softmax) or must carry an explicit peak-sharpness/coverage term; otherwise the ranking gate passes at chance localization, as it did here. This is the sharpest formulation yet of the project's bottleneck: five readouts (6D.1 `[SEG]`, 6E tokens, 6F box, 6G dense map, 6H ranked dense map) fail while their implementations are verified.
* The canonical pair construction (manifest + hash + zero-overlap audit) and the audit template (before/after geometry, logit scale, probability margin, peakiness, point-inside rate, gradient groups) are reusable for the next candidate.
* The frozen ceilings remain the comparison points: point oracle 0.4876/18-20, box oracle 0.7506/20-20, grid-256 point oracle 0.4883/18-20, Task 6C `P_C` 0.10604/0-20.

---

## ADR-020 — Task 6H.1: the point-aligned bounded objective is healthy; the single `[BOX]` query's spatial signal is the limit

**Status:** **Accepted** (Task 6H.1) for what the objective setup, H0-R and its audit measure. No novelty is claimed for the objective itself.

**Context.** Task 6H's own-vs-cross ranking was defined on unbounded logits and could be minimized by magnifying the field (mean |logit| 0.148 → 16.32) without moving the peak. Task 6H.1 froze the Task 6G/6H architecture and replaced only the objective: a 65,536-class spatial cross-entropy on the deterministic target point-cell plus a **bounded** own-vs-cross target **probability-mass** preference (`L_total = 0.5*(reasoning_A+reasoning_B) + 1.0*(L_point_A+L_point_B) + 1.0*L_cf`), with Task 6G's BCE+Dice and Task 6H's logit ranking detached to zero gradient.

**Decision**

1. **The objective is implemented faithfully and verified.** Target cells follow the frozen interior point + 256-cell convention (indices `y*256+x`); the softmax sums to 1; the four masses are bounded in [0,1] and not area-normalised; the pair preference is shift-invariant; the point-CE and bounded-`L_cf` gradients reach the query projection, the query norm, the visual 1×1 projection, the Qwen LoRA and the token rows; the retired terms carry `requires_grad == False` and zero gradient; SAM2 and the shared frozen feature stay bit-identical; the `[BOX]` query stays causally clean.
2. **The objective works, and it is *not* degenerate.** Over 1500 pair steps on 10 memorized pairs: point CE **11.0965 → 4.7427** (chance = `ln 65536` = 11.090), spatial entropy **11.09 → 6.21** nats, target-cell probability ×2 500, normalized point error **0.4658 → 0.1673**, bounded preference margin **−0.0000 → +0.1385**, own mass 0.184 vs cross mass 0.045. Contrast Task 6H, where the "improvement" was logit magnification with a probability margin of 0.112.
3. **H0-R still fails: `BOUNDED_POINT_OBJECTIVE_FAILED`.** Inside-target **7/20** (gate 18), paired point **2/10** (gate 9), bounded ranking **8/10** (gate 9), final mean |logit| 13.2 (limit 10). The audit shows why: the largest **non-target** cell holds 0.0532 probability against a 0.0515 mean target probability, so top-1 is 25 % and the mode is not reliably the target. The single `[BOX]` query does not carry enough target-specific spatial signal even under a directly aligned, bounded objective. The task stops before H1-R/H2-R (section 13).
4. **Task 6H's verdict and numbers are preserved**; only its causal wording is narrowed: the failure was objective degeneracy, not proof that the query signal is unlearnable. Task 6H.1 supplies the missing bounded test and shows the query *does* produce real, insufficient spatial signal.

**Consequences**

* Six readouts have now failed with verified implementations (6D.1 `[SEG]` probes, 6E coordinate tokens, 6F box regression, 6G dense map, 6H logit-ranked dense map, 6H.1 point-aligned bounded dense map). The two most recent tasks rule out both the readout format (6G) and the objective scale/sign (6H.1) — what remains is the **query representation and the strength of the MLLM's instruction-conditioned spatial signal**.
* Per section 23 the next candidates are therefore representation-level, not another scalar loss: instruction-aware multi-query / iterative query refinement, a stronger MLLM, or architecture-level reference/relation grounding. None may be implemented without review.
* The reusable pieces stay: canonical pair construction + manifest hash (6H), the audit template for objective health (6H/6H.1), the frozen point oracle ceiling (0.4883 mIoU / 18-20 at grid 256) and the `[BOX]`-query causal-placement proof.

---

## ADR-021 — Task 6I: one free cross-attention refinement step improves the query but cannot aim itself

**Status:** **Accepted** (Task 6I) for what the architecture setup, I0 and its audit measure.

**Context.** Task 6H.1 left the query representation as the measured bottleneck. Task 6I is the first architecture change since 6G: the causally clean `[BOX]` query q0 cross-attends the frozen SAM2 64×64 embedding exactly once (embed 256, 4 heads, residual + FFN(256→512→256)), and the refined q1 scores the frozen 256×256/32-ch feature (`dot(q1,K256)/sqrt(256)` + bias). The Task 6H.1 point-cell CE + bounded probability-mass pair objective is frozen byte-for-byte; the old Task 6G head is frozen and unused.

**Decision**

1. **The implementation is clean.** Exactly one cross-attention layer; F64 `[1,256,64,64]` and F256 `[1,32,256,256]` verified by size; q1 feeds the scorer (bit-identical to the combined forward); gradients reach every block component and the LoRA/token rows with zero gradient into SAM2, the old head or the box head; SAM2 and the shared features stay bit-identical; the `[BOX]` query stays causally clean (fp32 probe).
2. **The refinement step works but under-shoots.** I0 over 1500 pair steps: inside-own 7 → **13/20**, paired point 2 → **5/10**, bounded ranking 8 → **10/10**, own mass 0.184 → **0.417** vs cross 0.046 → 0.024, normalized point error 0.167 → **0.084** — versus gates 18 / 9 / 9 / <0.08.
3. **The attention cannot aim itself.** The 1×4096 cross-attention peaks (entropy 8.31 → 1.46 nats) but only ~2.5 % of its mass lands inside the target's 64×64 region, and that share barely moves (0.0079 → 0.0250); q1's same-image L2 distance grows to 1756 while q0's stays at 163, and the logit scale grows again (36). One free look at the whole map gives the query no target hypothesis to test, so it locks onto a salient but task-irrelevant location and the scorer amplifies it.
4. **Verdict `VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`**: I0 fails its gates cleanly after the audit; per the task the run stops before I1/I2 (no 240-pair train, no SAM2 segmentation, no further architecture changes).

**Consequences**

* Seven readouts have now failed with verified implementations. The remaining hypothesis is structural: a single query slot cannot *direct itself* toward the target even with one allowed visual inspection. The next candidates are therefore **multiple learned query slots** (object-query sets, e.g. OMG-Seg/OMG-LLaVA style, with the head picking the most peaked refined heatmap), a **stronger/larger MLLM**, or architecture-level reference/relation grounding. None may be implemented without review.
* Two methodology fixes are adopted for all later inference paths: inference runs the Qwen model in **eval mode** (LoRA dropout p=0.05 is training-only noise, 0.56–1.39 measured same-batch delta in train mode vs 0.0 in eval), and causal bit-identity probes run in **fp32** (bf16 sdpa rounds the shared prefix differently for different sequence lengths: 0.31 delta vs 0.0 in fp32).
* The frozen baselines are unchanged: grid-256 point oracle 0.4883/18-20, box oracle 0.7506/20-20, Task 6C `P_C` 0.10604/0-20.

---

## ADR-022 — Task 6J: program parsing + relation execution are solved; the proposal backbone is the binding limit

**Status:** **Accepted** (Task 6J) for what J0-J3 and the J1 audit measure.

**Context.** After seven failed direct pixel-grounding readouts (6D.1–6I), Task 6J audits the structured route: instruction → canonical relation program → building candidates → explicit geometry execution with the frozen Task 3B relation engine → selected mask. 20 canonical programs derive 1:1 from the frozen v0.1.1 query types (verified against all 25,229 records); a compact executor consumes only program + candidate geometry and never reads target id / GT reasoning / target mask.

**Decision**

1. **The executor semantics are exact (J0).** The independent executor agrees with the frozen generator's own `recompute_target_from_steps` on 120/120 fixed val samples; exact accuracy 1.000, paired 20/20, zero abstentions with oracle candidates.
2. **Instruction → program is solved at 2B (J2).** Text-only Qwen LoRA + `LayerNorm → Linear(2048, 20)` over the last prompt position reaches 1.000 accuracy on the fixed 120, 1.000 macro F1, 20/20 paired and 1.000 on the full 3,884-record val split. Predicted program + oracle candidates (J3) is therefore also exact: 1.000 / 20-20. Program parsing is not the bottleneck.
3. **The frozen YOLO proposal chain is the binding limit (J1).** Recall passes (0.919/0.869/0.594 at IoU 0.25/0.5/0.75; missing 13.1 %; tiny components 0.391) and oracle-program mIoU passes (0.3712 ≥ 0.30), but paired mask selection is **5/20** (gate 12/20): 35/120 executor abstentions plus proposal-geometry-induced wrong selections — the frozen semantics were calibrated on connected components, and YOLO instances (split/merged/border-clipped) change relation outcomes. J4 is therefore correctly not run.
4. **Verdict `PROPOSAL_QUALITY_LIMIT`** (J0/J2/J3 healthy; the proposal chain is the binding failure). Per the interpretation discipline nothing more is claimed than: explicit program parsing + proposal-level geometry execution is a viable functional target-selection architecture conditional on a better proposal backbone.

**Consequences**

* The next step is a **proposal-backbone/dataset task** (instance segmentation aligned with the component semantics, or proposal post-processing), not a parser or executor change. No YOLO retraining, no dataset migration, no `[REF]`/SRE/SCL/4B/GUI.
* The reusable API (`parse_program`, `extract_building_candidates`, `execute_program`, `predict_structured_mask` in `buildreasonseg_mvp/structured_grounding.py`) and the 20-program vocabulary (`evaluation/task6j_program_spec.json`) are the durable artifacts; the frozen YOLO baseline is invoked read-only with hashed provenance (`d9a6a65b…`, yolo_sam_env, ultralytics 8.4.67).

---

## ADR-023 — Task 6K: the WHU semantic→pseudo-instance conversion is faithful and light; keep WHU as primary

**Status:** **Accepted** (Task 6K) for the read-only measurements recorded in `evaluation/task6k_*.json`.

**Context.** After Task 6J attributed the structured-route failure to the proposal chain, the open question was whether the *source and conversion* are structurally unsuitable for instance-level spatial reasoning. Task 6K audited the chain original WHU Satellite Dataset II (East Asia) binary semantic raster → historical `mask_to_yolo.py` polygon conversion → current BuildReasonSeg component representation, entirely read-only (no source/converted/legacy file modified, no model trained, nothing installed).

**Decision**

1. **The conversion is exactly reproduced and numerically light.** All 4,038 tiles were re-emulated in memory and all 36,926 actual YOLO objects match object-by-object (per-object IoU ≥ 0.5 for 36,926/36,926, object-count agreement 4,038/4,038, mean union IoU **1.00000**). The current 36,926 components are exactly the surviving polygons (`component_id = polygon index + 1`) of 38,309 raw 8-connected semantic components.
2. **Loss decomposition.** `<50 contourArea` filter: 1,383 contours (3.61 %) removed = **0.058 %** of foreground area; `RETR_EXTERNAL` hole filling: 13,399 px **added** (0.021 %); `approxPolyDP`: IoU 0.99955, boundary 0.0086 px, 25/4,038 tiles below 0.99. The supplied snippet omits the historical `len(approx) < 3` skip and the `[0, 1]` clipping; the skip never fires in this dataset (0 polygons) and the difference is recorded.
3. **Relation-semantic drift is modest and concentrated.** RAW (38,309 candidates, no filter) vs CONVERTED (36,926) under the frozen Task 3B semantics: **4.33 %** weighted current-query target change (L1 5.0 %, L2 0.03 %, L3 0.03 %); the L1 extremes move because a deleted sub-50 px speck can be the extreme component; corpus constructibility changes ≤ ±0.5 % per program.
4. **Heuristic merge risk is low** (high 1.40 %, medium 4.29 % of raw components) and is explicitly labelled a heuristic, never instance ground truth.
5. **The historical split is recovered, not regenerated** (train 2,508 + val 627 = the whole source train pool; test 903 = the source test pool), with the `set→list→shuffle(42)` reproducibility caveat recorded and region-level geographic correlation shown to be certain for train/val.
6. **Task 6J's failure is not a data defect:** 49/65 (75.4 %) of J1 failures are proposal-model related on clean targets, 2/65 (3.1 %) plausibly conversion/data related, 14/65 inseparable.
7. **Verdict `KEEP_WHU_AS_PRIMARY_FOR_NOW`** (`legacy_baseline_role: keep`) — no replace gate fires against the declared thresholds (drift > 0.05, contour deletion > 0.05, foreground deletion > 0.01, high merge risk > 0.15).

**Consequences**

* WHU stays the primary corpus and the historical baseline, with two recorded structural limitations: **no true instance identity** anywhere in the chain (instances are connected components) and a **random train/val split** over the same contiguous regions (supports random-tile generalization only; only the test region is unseen).
* The next step remains a **proposal-backbone task** (instance segmentation aligned with component semantics, or deterministic proposal post-processing) evaluated with the same J1/J4 gates. Any candidate replacement dataset must first be audited with `evaluation/dataset_audit_schema_v1.json` (schema v1); license and source resolution stay `UNKNOWN` until externally verified and no external facts are fabricated.
* Reusable pieces: `buildreasonseg_mvp/whu_source_audit.py` (read-only historical-pipeline emulator + raw component/merge-risk/heuristic tooling), `scripts/task6k_*.py`, and the gitignored per-tile evidence cache `artifacts/task6k/`.

---

## ADR-024 — Task 6K.1: the archive contains a native manually delineated vector map; migrate the instance source to it

**Status:** **Accepted** (Task 6K.1) for the read-only measurements in `evaluation/task6k1_*.json`.

**Context.** Task 6K found `2. The shape file of the whole images\EA.shp` but never parsed it, and its `KEEP_WHU_AS_PRIMARY_FOR_NOW` verdict rested on the assumption that the semantic raster is the only label source. Task 6K.1 resolves that vector source before any project-level dataset decision.

**Decision**

1. **The shapefile is a genuine building footprint map.** 34,085 polygon features, consistent across `.shp`/`.shx`/`.dbf`, 0 unclosed rings, 0 self-intersections in the sampled validity probe, median 5 vertices per feature. No shapefile library is installed (no pyshp/fiona/geopandas/GDAL/rasterio/tifffile) and nothing may be installed, so the audit implements a standard-library read-only ESRI reader.
2. **Two operational findings:** the DBF attribute table is **degenerate** (every field constant over all 34,085 records → no per-feature identity, no usable area/length; identity is the `.shp` record order), and the CRS is **WGS_1984_World_Mercator**, so map-unit areas must never be presented as ground m².
3. **The tile mapping is validated, not assumed.** The whole-area rasters are BigTIFF; only full 512×512 windows were cropped, so `columns = floor(width/512)`; the derived grid capacity equals the cropped tile count exactly for all three rasters (10,044 / 3,618 / 3,726), the sampled RGB windows are pixel-identical to the cropped tiles, the labels match exactly, and the best offset is (0,0) in 240/240 tiles.
4. **The vector reproduces the raster labels:** mean IoU 0.9505 over all 4,038 positive tiles, 4,036/4,038 ≥ 0.90, none below 0.50.
5. **The original merge hypothesis is quantitatively small.** Measured by containment (IoU under-detects merges because a merged pseudo is much larger than either building): **1.34 %** of pseudo-instances contain ≥ 2 native buildings (2.65 % of native buildings live inside one). The dominant deviation is the opposite: **5.24 %** of native buildings are absent from the pseudo view — the `<50 contourArea` filter, hole filling and polygon simplification measured in Task 6K.
6. **Relation answers change by 6.96 % weighted** (L1 7.9 %, L2 3.4 %, L3 5.0 %; 39.5 % of tiles change at least one program), above the 5 % materiality bar. This, not Task 6K's 4.33 % RAW-vs-CONVERTED figure, is the number that drives the annotation decision.
7. **Task 6J is confirmed on native truth:** 78.5 % of J1 failures are proposal-model errors against *single* native buildings, and **0 %** involve merged pseudo-instance targets.
8. **Verdict `MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`**, `keep_whu_imagery: true`, `keep_historical_pseudo_baseline: true`.

**Consequences**

* WHU imagery stays primary and the historical pseudo-instance baseline stays frozen and reproducible (the 36,926-component mapping, Task 3B relations and J1/J4 results are untouched by this audit).
* The **instance truth source changes** for future work: semantic connected components stop being the primary instance truth. The next task builds a vector-derived canonical instance dataset and BuildSpatialReason-v0.2, and must budget for the measured **6.96 %** relation-answer change.
* The split remains the recorded limitation: 100 % of val tiles are directly adjacent to training tiles (94.7 % fully surrounded) and share their source rasters, while the test scene is spatially disjoint — so only random-tile generalisation within the same scenes is supported, never geographic generalisation.
* Reusable pieces: `buildreasonseg_mvp/shapefile_reader.py` (stdlib ESRI reader), `buildreasonseg_mvp/whu_vector_audit.py` (BigTIFF metadata, world files, tile grids, rasterisation, merge/split metrics), `scripts/task6k1_*.py`, small tracked overlays in `evaluation/task6k1_samples/`, and gitignored per-tile caches.

---

## ADR-025 — Task 6L: canonical native-vector dataset + BuildSpatialReason v0.2; the pseudo-instance baseline becomes historical

**Status:** **Accepted** (Task 6L), verdict `VECTOR_DATASET_MIGRATION_PASS`.

**Context.** Task 6K.1 confirmed the archive ships a manually delineated building vector map and that using it would change ~7 % of relation answers. Task 6L executes that migration as a *data* task: canonical instance dataset, split redesign, and a v0.2 reasoning dataset — with no model training.

**Decision**

1. **Canonical dataset `WHU-EA-NativeVector` v1.0** under `datasets/whu_native_vector/v1.0/` covering **all 17,388** cropped tiles (train 3,135 · train_no 10,527 · test 903 · test_no 2,823), including the `_no` and empty tiles the semantic workflow discarded: **41,186** clipped instances from **33,788** distinct features, per-instance polygons in tile pixel coordinates, `source_feature_id` = `.shp` record order, UID `(EA.shp SHA256, source_feature_id)`.
2. **Nothing is simplified away:** no `<50` deletion, no connected-component merging, no `RETR_EXTERNAL` hole loss, no `approxPolyDP`; holes and multipart parts are explicit rings; tiny instances are flagged (`tiny_area`) but retained, so any filtering must happen at query-eligibility time.
3. **Two split views:** `legacy_compat_v1` (the exact recovered historical 4,038 tiles, for comparison with Tasks 1–6K.1) and `scene_disjoint_v1`, the new primary split: train1 10,044 / train2 3,618 / test 3,726 with **0 tile overlap, 0 source-feature leakage and 0 cross-split RGB duplicates** (no boundary-crossing feature needed excluding). It provides scene separation, **not** cross-city generalization.
4. **Adapter layer, not an engine rewrite:** `native_vector_adapter.py` exposes the geometry fields the frozen Task 3B stack already consumes, so `spatial_reasoning/*` is untouched.
5. **BuildSpatialReason v0.2** uses native instances with the **same 20 programs** and the **same frozen relation config**; the semantic visibility policy stays at `1.0` on purpose so the version delta isolates annotation truth + split. 28,108 samples (train 12,778 / val 9,111 / test 6,219; L1 19,769 / L2 5,323 / L3 3,016) with every program supported in val and test.
6. **The migration is quantified, not assumed:** on the identical historical tiles v0.2 changes 5.70 % of targets (5.45 % weighted; L1 7.6 %, L2 0.7 %, L3 1.6 %), reconciling with Task 6K.1's 6.96 %.

**Consequences**

* `datasets/whu/` and `datasets/build_spatial_reason/v0.1.1/` stay frozen and reproducible; v0.1.1 remains the pseudo-instance historical baseline and is never rewritten to look vector-derived.
* v0.2 is **not answer-compatible** with v0.1.1 (5.45 % weighted change is expected and documented); results from the two versions must never be mixed in one table without stating the version.
* The Task 6J bridge (`proposal_evaluation_interface`, `candidate_set_for_tile`) is prepared; **J4 was not run** and no proposal model was trained or selected.
* Reusable pieces: `buildreasonseg_mvp/whu_native_vector.py` (canonical build/read), `native_vector_adapter.py`, `scripts/task6l_*.py`, `configs/build_spatial_reason_v0.2.yaml`, and the gitignored caches under `artifacts/whu_native_vector/`.

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
| 014 | **Task 6B: the `[SEG]`→prompt interface is the MVP bottleneck** | Language generalises (120/120 emission, 1.000 chain accuracy) but the mask does not (0.1102 mIoU, paired probe 0/20); the projection collapses to a near-constant 256-d prompt; loss weighting and mask LR do not fix it, so effort goes to the interface, not to 4B. **Causal reading PROVISIONAL — see 015** |
| 015 | **Task 6C: neither paired sampling nor the centre point fixes conditioning** | Valid, bit-reproducible 2×2; all four arms 0/20 paired with margins within ±0.0036; paired training raises projected effective rank 1.50→3.74 without producing conditioning; the centre point is 1/23 of the language norm, so removing it changes little — the remaining problem is after the prompt |
| 016 | **Task 6E: explicit box tokens are necessary but not sufficient at 2B / 3 epochs** | Enclosing quantization at `B = 256` keeps the oracle at 0.7343 mIoU / 20/20 and E0 overfits 20 records to 20/20 exact tokens (0.9166 box IoU), but E1 on 480 paired records fails with structural 0/120: `[BOX]`/EOS/reasoning are learned (accuracy 1.0) while the 256-way location values stall at the uniform floor (train CE 5.15 vs `ln 256 = 5.545`) and `[SEG]` is never predicted after the location run — fix the location learning signal/budget before adding `[REF]`, SRE, SCL or scale |
| 017 | **Task 6F: pre-reasoning `[BOX]` query with a direct box readout** | The query hidden is provably conditioned only by image + instruction (bit-identical under future-text mutation); the same 20 records reach F0 box IoU 0.9046 / paired 10/10 on the inference-form query path; F1 (480 paired, ≤8 epochs) answers whether the target-aware representation generalizes — the Task 6E coordinate vocabulary stays retired regardless |
| 018 | **Task 6G: dense query-to-map grounding does not localize either** | The 256×256 grid-snapped point oracle keeps 0.4883 mIoU / 18-20; the dense head is audited clean (SAM2 bit-frozen, train/eval bit-identical, healthy gradients) yet G0 cannot overfit 20 records (inside 0.10, Dice 0.125) — the heatmap is a near-flat image-dominated field, so the measured bottleneck is the query's weak instruction signal, not the readout format |
| 019 | **Task 6H: own-vs-cross ranking on unbounded logits is satisfiable without localizing** | 240 canonical same-image counterfactual pairs, one pair step with one shared frozen feature and a single backward; H0 reaches pair ranking 10/10 and mean margin +5.58 while inside-own is 3/20 — the audit shows the margin grew with a 110× logit-scale increase (probability margin only +0.112, point-inside 0.15), so the specified objective is scale-degenerate and the task stops at H0 |
| 020 | **Task 6H.1: the point-aligned bounded objective is healthy; the query's spatial signal is the limit** | Spatial cross-entropy on the deterministic point-cell plus a bounded probability-mass preference (BCE/Dice and logit ranking detached to zero gradient); H0-R shows point CE 11.10 → 4.74, entropy 11.09 → 6.21, point error 0.466 → 0.167, bounded margin +0.139, yet inside-own is 7/20 and the max non-target probability (0.0532) still rivals the target cell (0.0515) — a clean failure of the single `[BOX]` query representation, so the next candidates are representation-level (multi-query refinement, stronger MLLM, reference/relation grounding) |
| 021 | **Task 6I: one free cross-attention refinement step improves the query but cannot aim itself** | `[BOX]` q0 cross-attends the frozen 64×64 SAM2 embedding exactly once (256/4 heads, residual + FFN); refined q1 scores the 256×256 feature; frozen 6H.1 objective; I0 raises inside-own 7→13/20 and paired point 2→5/10 (gates 18/9) while the 1×4096 attention peaks (entropy 8.31→1.46) with only ~2.5 % mass on the target — a single query slot cannot direct itself at the target even with one allowed look, so the next candidates are multiple learned query slots, a stronger MLLM, or reference/relation grounding |
| 022 | **Task 6J: program parsing + relation execution are solved; the proposal backbone is the binding limit** | 20 canonical programs derived 1:1 from the frozen query types; the independent executor reproduces the frozen generator on 120/120 samples (J0 1.000, paired 20/20); text-only Qwen ProgramHead parses instructions perfectly (J2 1.000 acc / 1.000 macro F1 / 20-20 paired / 1.000 full val; J3 1.000 / 20-20) — but the frozen YOLO proposal chain fails the J1 paired gate (5/20 vs 12/20; recall@0.5 0.869; 35/120 abstentions from proposal-geometry mismatch), so the next step is a proposal-backbone/dataset task, not parser or executor changes |
| 023 | **Task 6K: the WHU semantic→pseudo-instance conversion is faithful and light; keep WHU as primary** | Read-only audit: 36,926/36,926 actual YOLO objects re-emulate exactly (union IoU 1.00000, count agreement 4,038/4,038) from 38,309 raw 8-connected semantic components; `<50` deletion 3.61 % of contours but only 0.058 % of foreground; hole filling +0.021 %; approx IoU 0.99955; RAW-vs-CONVERTED relation target change 4.33 % weighted (L1 5.0 %, L2/L3 0.03 %); merge high risk 1.40 %; 75.4 % of Task 6J J1 failures are proposal-model related → `KEEP_WHU_AS_PRIMARY_FOR_NOW` with no-true-instance-identity and random-train/val-split limitations recorded, and a reusable schema v1 for future candidates |
| 024 | **Task 6K.1: the archive ships a native vector building map with true instance identity → migrate the instance source** | The local `EA.shp` parses to **34,085** polygon features (consistent across `.shp`/`.shx`/`.dbf`, 0 unclosed rings, 0 self-intersections in the sampled validity probe), CRS `WGS_1984_World_Mercator`, DBF attributes **degenerate** (all fields constant → identity is the record order); the tile→whole-image mapping is **validated** (RGB windows pixel-identical, labels exact, grid capacity == cropped tiles for all three rasters, best offset (0,0) in 240/240); the vector reproduces the raster labels at **mean IoU 0.9505** (4,036/4,038 tiles ≥ 0.90); vs the pseudo view: 94.8 % match at IoU 0.5, **merges are small (1.34 % of pseudo-instances contain ≥ 2 buildings)** but **5.24 % of native buildings are absent from the pseudo view**, and VECTOR-vs-PSEUDO relation answers change by **6.96 % weighted** (L1 7.9 %, L2 3.4 %, L3 5.0 %); Task 6J re-attribution: 78.5 % of J1 failures are proposal-model errors on clean single native buildings, **0 %** merged-target cases → verdict **`MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`** with `keep_whu_imagery: true` and `keep_historical_pseudo_baseline: true`; the frozen artifacts stay valid and the next task builds a vector-derived canonical instance dataset + BuildSpatialReason-v0.2 |
| 025 | **Task 6L: the primary instance truth is migrated to the native vector map with a scene-disjoint split → `VECTOR_DATASET_MIGRATION_PASS`** | Canonical `WHU-EA-NativeVector` v1.0 indexes **all 17,388** cropped tiles (including `_no` and empty tiles) with **41,186** clipped instances from **33,788** distinct `EA.shp` features, identity `(EA.shp SHA256, source_feature_id)`, no `<50` deletion / no component merging / no hole loss / no simplification (tiny instances flagged only), full per-instance polygons + `visible_fraction` + border flags in a gitignored regenerable cache; two split views: `legacy_compat_v1` (the exact historical 4,038 tiles, for comparability) and the new primary `scene_disjoint_v1` (**train1 10,044 / train2 3,618 / test 3,726**, 0 tile overlap, 0 source-feature leakage, 0 cross-split RGB duplicates); BuildSpatialReason **v0.2** uses the native instances, the same 20 programs and the same frozen relation config (visibility policy deliberately unchanged at 1.0) and yields **28,108** samples (train 12,778 / val 9,111 / test 6,219) with **every program supported in val and test**; on the historical tiles v0.2 changes **5.70 %** of targets (**5.45 %** weighted), reconciling with Task 6K.1's 6.96 %; all 11 acceptance gates pass (including a byte-identical determinism rerun and an independent overlay audit at min IoU 1.0) |
