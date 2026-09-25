# Task 5.5 — Evaluation Protocol Plan

**Status:** plan only. Nothing here has been executed.
**Research/access date:** 2026-09-26
**Dataset under evaluation:** BuildSpatialReason **v0.1.1** (the only accepted version).

Authoritative counts are declared once in
`evaluation/build_spatial_reason_artifact_index.json` and enforced by
`scripts/check_artifact_consistency.py`. Do not hand-copy them.

---

## 1. Split protocol

| Split | Samples | Images | Use |
|---|---|---|---|
| train | 15,592 | 2,423 | training / LoRA fitting |
| val | 3,884 | 614 | model selection, early stopping, hyper-parameters |
| test | 5,753 | 883 | final reported numbers only |

Rules:

1. **test is touched once.** No hyper-parameter, prompt, threshold or checkpoint
   decision may be made by looking at test.
2. **Metric denominators must be reported together with the subset** — full test
   set, Level-1/2/3 breakdown, and the nontrivial Level-3 subset separately.
3. **Trivial Level-3 must never be folded into the headline multi-hop number.**
   A trivial chain (exactly one admissible candidate after the direction filter,
   1,256 of 2,918 Level-3 samples) is answerable without any distance comparison,
   so including it inflates multi-hop success.
4. **Scene-level split leakage is `UNVERIFIED`** in v0.1.1: the source tiles carry
   no scene/geographic grouping metadata and val was a random 20% subset of the
   original train pool. Any test-set number must carry this caveat, and no claim
   of geographic generalisation may be made from it.

---

## 2. Segmentation metrics

The dataset target is a **single connected component**, so a prediction can be
scored two ways. Both are reported; they answer different questions.

### 2.1 Pixel-level

| Metric | Definition | Notes |
|---|---|---|
| **mIoU** | mean over samples of `|P∩G| / |P∪G|` | primary pixel metric |
| **cIoU** | `Σ|P∩G| / Σ|P∪G|` over the whole split | dominated by large targets; report alongside mIoU, never instead of it |
| **gIoU** | mean over samples of `|P∩G| / |B(P)∪B(G)|` using bounding boxes | more forgiving to small localisation error; useful for adjacency-heavy building scenes |
| **Dice / F1** | `2|P∩G| / (|P|+|G|)` | redundant with IoU but commonly reported; include only if a comparison table needs it |
| **Boundary F1** | boundary-matching F1 at a stated tolerance | include **only if** a boundary argument is actually made in the paper; otherwise it is decoration |

### 2.2 Component-level (this project's distinctive axis)

Because every target is a component of the source component map, snap the
predicted mask to components and score the discrete decision:

| Metric | Definition |
|---|---|
| **Target selection accuracy** | argmax-overlap component of the prediction equals `target_component_id` |
| **Component IoU after snapping** | IoU between the snapped component and the ground-truth component; a clean measure of "right building, right extent" |
| **Adjacency confusion rate** | fraction of errors whose predicted component is the *reference* component, or a component adjacent/nearest to it — the characteristic failure of this task |

`Adjacency confusion rate` matters more than aggregate IoU for this project: it
separates "the model segmented a building badly" from "the model segmented the
wrong building, specifically the reference one".

### 2.3 Reporting discipline

- mIoU and cIoU are **not interchangeable**; never report one and label it the
  other.
- Confidence intervals or per-image bootstrap variance **only if** the paper
  actually needs them; a bare single run must be labelled as a single run.
- No metric from another dataset may be placed in the same column as a
  BuildSpatialReason metric as if directly comparable (different annotation
  density, different image domain, different component definitions).

---

## 3. Reasoning metrics

These are computed from the model's produced reasoning text plus the target it
finally selected, and they are what makes the paper a *reasoning* paper.

| Metric | Definition | Applies to |
|---|---|---|
| **Reference selection accuracy** | the reference the model used equals `reference_component_ids[0]` | L2, L3 |
| **Relation satisfaction rate** | the ground-truth target genuinely satisfies the stated relation **and** the model selected it | all |
| **Target selection accuracy** | see §2.2 | all |
| **Multi-hop success** | full chain correct end-to-end (reference correct AND final target correct) | **nontrivial L3 = 1,662 samples, primary metric** |
| **Multi-hop success (trivial)** | same, on the 1,256 trivial samples | separate column, never merged |
| **Reasoning-order correctness** | the emitted steps follow reference → filter → comparison | L2, L3 (requires structured output) |

The `relation satisfaction rate` is the geometry-verifiable metric that the
dataset was built to make checkable: the relation predicate is frozen
(`configs/spatial_relations_v1.yaml`), so satisfaction can be re-evaluated
geometrically from the predicted mask instead of being judged by a language model.
This is the project's main measurement advantage and should be stated as such.

### 3.1 Language-side checks (cheap, honest)

- **ID leakage rate**: fraction of reasoning strings containing a bare component
  id. The training data has **0**; a model that emits one is imitating an artifact
  it never saw, and that is a finding worth reporting.
- **Template-copy rate**: fraction of reasoning strings that reproduce a training
  template verbatim. The dataset uses 39 templates with explicit `template_id`, so
  this is measurable rather than impressionistic.
- **Instruction-language consistency**: does the model answer in the language it
  was asked in. Cheap to compute, and a real failure mode for bilingual data.

---

## 4. Spatial consistency metrics

These are the metrics the final model is *for*, so they are defined now even
though the MVP will not yet have a Spatial Relation Encoder.

| Metric | Definition | Availability in MVP |
|---|---|---|
| **Direction consistency** | given the reference and the predicted target, the frozen direction predicate holds | computable from the MVP output |
| **Size consistency** | for `largest`/`smallest` queries, the predicted target really is the global extreme over the visible components, or is discarded | computable |
| **Nearest consistency** | the predicted target is the boundary-distance minimum among the admissible set | computable |
| **Reference-mask quality** | IoU of the predicted reference region against the ground-truth reference component — reported only for models that actually predict a reference | needs `[REF]` |

Direction / size / nearest consistency are defined so that they are computable
**without** a Spatial Relation Encoder, which is what lets the MVP produce a
baseline number that the final model must beat. They must not be redefined after
the final model exists.

---

## 5. Evaluation harness requirements

- **Deterministic subsetting.** Any subset evaluation (nontrivial L3, per-level)
  selects by the stored `query_type`, `level` and `trivial_selection` fields —
  never by re-deriving them, so the subsets cannot drift from the audit.
- **Component snapping uses the stored component map** referenced by
  `component_map_path`. No geometry is regenerated.
- **Failure accounting.** Every failed sample is written out with its
  `sample_id`, so qualitative error analysis and the paper's failure table come
  from the same run that produced the metrics.
- **No test-time peeking.** The harness records which split produced each number.

---

## 6. Baselines

| Tier | Baseline | Purpose | Status |
|---|---|---|---|
| B0 | **Frozen YOLOv8m-seg on WHU** (val mask mAP50 = 0.80693 at epoch 100, peak 0.84373 at epoch 44) | performance floor from the pre-existing baseline line; not a reasoning model | available as a frozen record in `baseline/yolo_whu/` |
| B1 | **Non-reasoning referring segmentation**: feed the instruction text to a plain text-conditioned segmenter with no spatial reasoning supervision | isolates how much of the performance comes from spatial reasoning versus from language-conditioned segmentation | to be built in Task 6 |
| B2 | **LISA-style / Sa2VA-style reasoning segmentation** fine-tuned on BuildSpatialReason | the honest "does our method beat the standard approach on our own data" comparison | to be built in Task 6 (stack chosen in `docs/research/task5_5_mvp_decision.md`) |
| B3 | **A published remote-sensing reasoning-segmentation method**, if it is reproducible from public code and weights | external comparison | conditional — only if the license and reproducibility are both clean; otherwise report as related work rather than as a table row |

Baselines B0 and B1 use the same splits and the same evaluation code as the
proposed model. B0's historical numbers were measured on the WHU *instance*
task, not on BuildSpatialReason, so it is a floor and a sanity check, **not** a
like-for-like comparison row; if it appears in a table it must be footnoted as a
different task.

---

## 7. Ablations to schedule (design, not execution)

1. instruction without spatial relations (language only) → isolates reasoning.
2. `[SEG]` only vs `[REF] + [SEG]` → isolates the reference pathway.
3. with vs without the spatial prior in the decoder → isolates the fusion claim.
4. with vs without the Spatial Consistency Loss → the final-model claim.
5. trivial vs nontrivial Level-3 → shows where the gain actually comes from.

Each ablation is a claim in the paper; nothing above should be run before the
corresponding mechanism exists.

---

## 8. What this protocol deliberately does not claim

- It does not claim the metrics are comparable to published results on other
  datasets.
- It does not claim scene-level generalisation (leakage `UNVERIFIED`).
- It does not claim statistical significance from a single run.
- It does not define any metric in a way that requires ground-truth geometry at
  inference time (ADR-002). Every metric above is computable from the model's own
  outputs plus the annotation used only for scoring.
