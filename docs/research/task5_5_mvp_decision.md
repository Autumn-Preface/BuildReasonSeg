# Task 5.5 — MVP Decision

**Research/access date:** 2026-09-26
**Status:** design freeze for **BuildReasonSeg-MVP**. No implementation, no installation, no download and
no training was performed in Task 5.5.

Supporting documents: `task5_5_model_stack.md` · `task5_5_hardware_feasibility.md` ·
`task5_5_novelty_collision.md` · `task5_5_external_datasets.md` · `task5_5_evaluation_plan.md` ·
`task5_5_literature_matrix.md` · `task5_5_sources.md`.
Machine-readable form: `evaluation/task5_5_stack_decision.json`.

---

## 1. The twelve required decisions

### Q1. What exact MLLM/checkpoint should Task 6 start from?

**`Qwen/Qwen3-VL-4B-Instruct`** — Apache-2.0 (verified from both the Hugging Face API licence tag and the
ModelScope first-party `License` field), BF16, 8.27 GiB of weights, text hidden 2560 × 36 layers,
262,144-token context, vocabulary 151,936.

**Stage 0–2 run on `Qwen/Qwen3-VL-2B-Instruct`** (same family, same token ids, 3.96 GiB) so that the
pipeline is proved on the smaller model before the 4B training budget is spent. Switching between them is
a configuration change, not a rewrite.

### Q2. What exact segmentation model/decoder should Task 6 use?

**`facebook/sam2.1-hiera-large`** — Apache-2.0, verified by reading the repository's `LICENSE` file
first-hand; 1.67 GiB on disk (≈0.83 GiB per copy). The **image encoder and memory module stay frozen**;
only the **mask decoder** is trained, matching the published LISA and Sa2VA recipes.

Fallback if VRAM is tight: `facebook/sam2.1-hiera-base-plus` (0.60 GiB, same licence).

SAM 1 (`facebook/segment-anything`, Apache-2.0, LICENSE read) is *not* chosen: `sam-vit-huge` is 4.78 GiB
for no benefit here, and a `segment_anything` install already exists only because of the legacy baseline.

### Q3. `[SEG]` only, `[REF] + [SEG]`, or structured prompts?

**Three tiers, in this order:**

1. **MVP primary: `[SEG]` only.** One new special token appended to the answer; its last-layer hidden
   state drives the mask decoder. This is the smallest thing that can work and it produces the baseline
   that every later claim is measured against.
2. **Stage 3+ of the MVP: add `[REF]`.** BuildSpatialReason already stores `reference_component_ids` for
   all Level-2/3 samples, so a ground-truth reference mask is available **at zero annotation cost** — the
   reference pathway can be supervised directly. Note explicitly that the `[REF]` *token* is prior art
   (SegLLM, PSALM); the contribution is what is done with the reference's **predicted geometry**.
3. **Baseline, not contribution: structured prompts (Think2Seg-RS-style), no training of the LLM.**
   Prompts (points/boxes) drive a frozen SAM2. This is the honest engineering baseline and answers "how
   far does prompting get?" — it is never presented as the method.

### Q4. Which modules are frozen vs trainable?

| Module | State | Reason |
|---|---|---|
| Qwen3-VL vision tower | **frozen** | visual adaptation is not the claim, and training it costs ≈3–4 GiB |
| Qwen3-VL language weights | **frozen** (LoRA only) | 16 GB budget; full fine-tuning is out of scope |
| LoRA adapters on `q,k,v,o,gate,up,down` | **trainable** | ≈31.3 M parameters at rank 16 |
| `[SEG]` (and later `[REF]`) embedding rows | **trainable** | new tokens must learn embeddings |
| Projection MLP (LLM hidden → decoder prompt space) | **trainable** | the bridge; must stay a *separate, replaceable* module |
| SAM2.1 image encoder | **frozen** | published recipes freeze it; large VRAM saving |
| SAM2.1 memory/attention module | **frozen** | same |
| SAM2.1 mask decoder | **trainable** | the standard LISA/Sa2VA choice |
| Spatial Relation Encoder (future) | not built in the MVP | interface reserved — see Q10 |

### Q5. LoRA, QLoRA, or another PEFT method?

**LoRA, BF16, rank 16, alpha 32, dropout 0.05**, applied to `q,k,v,o,gate,up,down`.
Rationale: it avoids every quantisation dependency, and the measured headroom at 4B is ≈3 GiB
(11.6–12.6 GiB of ≈15.89 GiB, ESTIMATE). `bitsandbytes` officially supports Windows 11 with QLoRA 4-bit,
so **4-bit QLoRA rank 16 is the documented OOM-recovery fallback** — but the `sm_120` + bnb + Windows
combination is **UNVERIFIED**, so it must not be the primary plan. Full fine-tuning is not attempted.
Adapters on the vision tower are excluded.

### Q6. First image resolution / max-pixel policy?

**Native 512 × 512, no upscaling, no tiling beyond the source tile.** Qwen3-VL uses `patch_size = 16`
and `merge_size = 2`, so one visual token covers 32 × 32 px ⇒ **256 visual tokens per tile**. The image
budget must be **set explicitly** (`min_pixels = max_pixels = 512 × 512`) because Qwen3-VL's released
`preprocessor_config.json` contains no `min_pixels`/`max_pixels` keys at all, so the Qwen2.5-VL defaults
(`max_pixels = 12,845,056`) must **not** be inherited.

Total sequence for training: ≈256 visual tokens + instruction + reasoning + special tokens, targeted at
**≤768 tokens**. Context length is not a constraint for this dataset.

### Q7. Local OS/environment?

**Native Windows 11**, in a **new dedicated conda environment** (never `yolo_sam_env`, which holds the
frozen YOLO baseline line). Python 3.10/3.11; `torch` already 2.13.0+cu132 with `sm_120` support
(**measured working**); `transformers` ≥ 4.57.0; `peft`, `accelerate`, `safetensors`, `tokenizers`,
`qwen-vl-utils`; `sam2`; **SDPA attention, not FlashAttention** (FlashAttention has no official Windows
support, no prebuilt wheels, and no `sm_120` in its documented GPU list).

**WSL2 Ubuntu is the documented fallback**, triggered only by (a) SAM 2's optional compiled CUDA
extension proving necessary, or (b) a decision to enable FlashAttention. Rationale for not starting
there: every critical component is measured working natively, and the workload (≈768-token sequences,
256-token image prefix) is exactly where FlashAttention's benefit is smallest.

### Q8. Smallest staged experiment proving the pipeline works?

**Stage 1: a 2-sample forward/backward pass** that (i) appends `[SEG]`, (ii) reads its hidden state from
`output_hidden_states`, (iii) projects it, (iv) decodes a mask with frozen SAM2, and (v) produces a
finite, non-zero gradient on the projection MLP and the LoRA adapters. Passing Stage 1 is what proves the
`hidden([SEG]) → projection → mask` path exists at all. Everything before it is environment plumbing.

### Q9. Any external dataset before/after BuildSpatialReason?

**No external dataset before BuildSpatialReason.** Warm-up is unnecessary: the model is already
instruction-tuned, the task is narrow, and the licence status of almost every candidate is unresolved
(22 of 25 surveyed datasets are `UNVERIFIED`). Adding an unresolved-licence corpus to buy an unproven
warm-up gain is a bad trade.

If warm-up is ever wanted, the only licence-clean options verified during this task are
**EarthReason** (HF `earth-insights/EarthReason`, `apache-2.0`, **1.32 GiB**) and, with a caveat,
**LaSeRS** (`earth-insights/LaSeRS`, `apache-2.0`, 11.65 GiB — but its SAMRS-derived masks inherit
academic/non-commercial upstream terms).

**After** the MVP: EarthReason and LaSeRS for **evaluation / robustness only**; published RRSIS-D and
RISBench numbers for a **literature comparison** (never re-measured and presented as our own run);
DRSeg is `cc-by-nc-4.0` and therefore excluded from anything beyond citation. No external dataset is
merged into training.

### Q10. What path stays open for the Spatial Relation Encoder and Spatial Consistency Loss?

An explicit **interface contract**, fixed now so the MVP cannot block the final model:

```
inputs : predicted reference geometry (mask → centroid, bbox, area, border contact)
         predicted candidate/target geometry
         visual feature map from the frozen SAM2 encoder
outputs: a relation-aware conditioning vector and/or refined mask logits
```
Rules that make the contract real:
1. The **projection MLP and the relation encoder stay separate modules.** The MVP has no relation
   encoder, so the pipeline must be written so one can be inserted between the projection and the
   decoder without touching the LLM path.
2. Every inference-time geometric quantity is derived from the **model's own predicted masks**, never
   from the annotation (ADR-002). This is what makes the encoder inference-valid (ADR-005), and it is
   also what distinguishes the approach from SegLLM's GT-free-but-LLM-side geometry injection.
3. The MVP keeps, per sample, the record needed to compute a relation contradiction at loss time:
   predicted reference mask, predicted target mask, and the stated relation. The Spatial Consistency
   Loss is then computable on top of an unchanged MVP output.
4. The dataset side is already prepared: `configs/spatial_relations_v1.yaml` is frozen and the
   validator's independent oracle shows the relation semantics are correct, so relation-consistency
   evaluation can be implemented without new annotation.

### Q11. What literature result most threatens current novelty?

**SegLLM** (arXiv 2410.18923, Oct 2024) — it already introduces `[REF]` **and** `[SEG]` with supervision
on both masks, already re-injects predicted-region geometry (mask-derived bounding-box positional
embedding + masked-object CLIP embedding) into the LLM, and already performs multi-round relational
reasoning over previously segmented regions. Secondary: **PSALM** (ECCV 2024) for the `[REF]` token as a
generic device, and the **ISPRS Annals XI-2-2026** building-VLM paper for the "instruction-tuned VLM for
geospatial buildings" framing.

### Q12. How should the Challenge Cup technical narrative change?

From a *mechanism* claim to a **verification-first** claim:

1. **Foundation:** spatial supervision that can be **proved** correct — 25,229 samples, every stated
   relation re-checked geometrically by an independent oracle, zero semantic violations, enforced by a
   repository-wide artifact gate. Annotation auditability is the differentiator, not annotation volume.
2. **Model:** the contribution is the **explicit relation pathway** — predicted reference geometry →
   Spatial Relation Encoder → semantic–spatial–visual fusion → target mask, with a Spatial Consistency
   Loss. `[REF]`/`[SEG]` are plumbing and are cited as prior art.
3. **Evidence:** relation-level metrics that only a geometry-verifiable dataset permits — relation
   satisfaction rate, direction/size/nearest consistency, and multi-hop success on the **nontrivial**
   Level-3 subset (1,662 samples), against a `[SEG]`-only model, a structured-prompting baseline and a
   referring-segmentation baseline.

See `task5_5_novelty_collision.md` §4 for the retain / narrow / reframe / postpone / drop list.

---

## 2. Task 6 staged plan (design only — do not execute)

Common settings unless a stage says otherwise: batch 1 with gradient accumulation, AdamW on adapters,
BF16, SDPA attention, vision tower frozen, SAM2 frozen except the mask decoder, images at 512 × 512,
gradient checkpointing enabled from Stage 3 onward.

| Stage | What it proves | Runtime scale (ESTIMATE) | Success criteria | Failure criteria | Checkpoint policy | VRAM safety | Logging |
|---|---|---|---|---|---|---|---|
| **0. Environment + model load smoke test** | the stack exists and loads on this machine | 30–60 min | `Qwen3-VL-2B` loads in BF16; tokenizer extended with `[SEG]`; `resize_token_embeddings` OK; SAM2.1 weights load; a text-only forward pass returns a finite logit; **weights are obtainable (HF or ModelScope route confirmed)** | any component fails to load, or weight acquisition is blocked | none | watch `torch.cuda.max_memory_allocated()` after load; must be < 6 GiB for 2B | versions of torch/transformers/bnb/OS/driver; peak VRAM; load time |
| **1. 2-sample forward/backward** | the `[SEG]` → projection → mask path exists and is differentiable | 15–30 min | loss finite; gradients non-zero on the projection MLP, the `[SEG]` embedding row and the LoRA adapters; the decoded mask is non-empty; `hidden([SEG])` shape is `(B, 2560)` | NaN/inf loss, all-zero gradients, empty mask, or an OOM that survives checkpointing | none | must fit for the 2B model with ≥8 GiB free | loss, grad norms per module, token counts, peak VRAM |
| **2. Overfit 20 samples** | the architecture can memorise; the data pipeline and mask supervision are correct | 30 min – 2 h | training mIoU on those 20 samples → > 0.9; the `[SEG]` token is emitted reliably; target-selection accuracy → 20/20 | loss plateaus at a high value; `[SEG]` not emitted; masks collapse to empty or full-image | save `last` after the stage | 2B model; 4B optional | per-step loss, mIoU, emitted-token rate, peak VRAM |
| **3. 200–500 sample mini-train** | the model generalises off the overfit set, and `[REF]` can be added | 2–6 h (ESTIMATE) | held-out subset (from train, **not** val) mIoU clearly above the Stage-2-untrained baseline; **switch to Qwen3-VL-4B here**; add `[REF]` and verify the reference mask is supervised | no improvement over zero-shot; unstable loss; OOM at 4B that survives checkpointing + QLoRA fallback | `best` + `last`, every 200 steps | 4B BF16 (≈11.6–12.6 GiB EST); fall back to 2B or QLoRA | full metric set + periodic qualitative dumps |
| **4. Full train** | the MVP result | target **< 7 days** wall-clock on the 4B model, subject to the measured step time from Stage 3 | improves on Stage 3 and on every baseline; no overfitting to val; clean early stop | val degrades early and irreversibly; run exceeds the budget without a usable checkpoint | `best` + `last` + periodic; **resumable** | monitor continuously; abort-and-resume guidance | loss curves, val metrics per epoch, LR, memory, throughput |
| **5. Validation + demo** | the paper numbers and the demo | 2–4 h | full evaluation per `task5_5_evaluation_plan.md`: mIoU/cIoU/gIoU, target-selection accuracy, relation satisfaction, spatial consistency, nontrivial-L3 multi-hop | test-set numbers below baseline; relation-satisfaction collapse | freeze the checkpoint hash used for every reported number | inference only (≈9–10 GiB EST for 4B) | record the split, checkpoint hash and metric definitions with every number |

### 2.1 Requirements the Task 6 training code must satisfy

- best **and** last checkpoints; periodic saves; **resume** from any checkpoint;
- mixed precision (BF16) with an explicit switch;
- gradient accumulation (effective batch recorded in the log);
- gradient clipping (norm logged, not silently applied);
- **NaN/inf detection** with a clear abort, not a silent skip;
- OOM recovery guidance: gradient checkpointing → smaller image budget → 4-bit QLoRA → smaller model;
- validation each epoch with the metric set above;
- early stopping where appropriate, with the patience recorded;
- every reported number stamped with the split, checkpoint hash and metric definition.

### 2.2 Task 6 entry criteria (all must hold before implementation starts)

1. Licences cleared for every component actually used: Qwen3-VL-4B/2B (apache-2.0 ✔), SAM 2.1
   (Apache-2.0 ✔). Any QLoRA fallback additionally requires resolving `bitsandbytes` + `sm_120` on Windows.
2. **Weight acquisition confirmed** on this machine (verify at Stage 0, because the local accelerator's
   state determines whether the Hugging Face route or the ModelScope route is used).
3. A **new dedicated environment** — `yolo_sam_env` is not touched.
4. The `[SEG]` token id and vocabulary extension approach fixed and recorded before training.
5. Dataset frozen and verified: v0.1.1 JSONL hashes unchanged, quality JSON `PASS`, consistency gate
   `consistent` — all re-checkable by `python scripts/check_artifact_consistency.py`.
6. `datasets/build_spatial_reason/v0.1/` is **not** used for any training decision.

---

## 3. What this decision deliberately does not do

- It does not implement, install, download or train anything.
- It does not claim the MVP will reach any particular accuracy; every number in this document is an
  ESTIMATE or a MEASURED environment fact, never a model result.
- It does not adopt a licence-unverified component on the critical path: `Qwen2.5-VL-3B-Instruct` is
  excluded for that reason alone, and Sa2VA is used as a *recipe*, not as a downloadable dependency.
- It does not merge any external dataset into training.
