# FROM_DSH — Task 5.5 Report: External Research, Model-Stack Verification & MVP Design Freeze

**Date:** 2026-09-26
**Actor:** DSH
**Task:** Task 5.5 (from `handoff/TO_DSH.md`)
**Verdict:** **COMPLETE** — MVP stack frozen (ADR-012), with explicitly listed `PROVISIONAL` sub-items.
**Research confidence:** **high on the chosen stack, medium on the surrounding landscape, low on licences
that require GitHub/Hugging Face to resolve** — see §1.

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

---

## 1. Verdict / Research Confidence

**Task 5.5 is complete.** A primary MVP stack, a fallback stack, a baseline-only stack and a
not-recommended list are all fixed and machine-readable in `evaluation/task5_5_stack_decision.json`,
with the reasoning in `docs/research/`. Nothing was downloaded, installed, executed or trained.

Confidence is deliberately uneven, and the unevenness is recorded rather than smoothed over:

| Area | Confidence | Why |
|---|---|---|
| The chosen stack (Qwen3-VL-4B + SAM 2.1 + BF16 LoRA) | **High** | Every component licence was verified first-hand or from an official platform field, the local GPU support was *measured*, model dimensions came from the official config files, and the `[SEG]`→projection→SAM2 pattern has a public reference implementation on the same base family |
| Local hardware feasibility | **High for Windows/PyTorch, medium for the training budget** | `sm_120` and 15.89 GiB VRAM are measured; all VRAM totals are arithmetic (ESTIMATE) and will be replaced by Stage 0–2 measurements |
| Literature landscape | **Medium–high** | 43 works surveyed from primary sources; repository-level facts (stars, releases, training code) could not be checked through the research tools |
| Licences needing GitHub/Hugging Face | **Low where unresolved** | A local accelerator blackholes those hosts for the research tools; they were reachable through ordinary HTTP clients, which is how the important licences *were* verified — but the residue is genuinely unverified (§16) |
| Dataset novelty claim | **Medium** | The absence is well evidenced across 28 datasets, but it is an *evidenced absence*, and one paper was read only at abstract level |

---

## 2. Preflight Provenance Fix

### 2.1 §3.1 — generation-time vs current-source provenance

`datasets/build_spatial_reason/v0.1.1/manifest.json` no longer carries the single combined
`generator_file_sha256` map. It now has:

```
generation_source:  commit a9e5bd69cbc331f79763dd401d5c5f68d2b5f780  (Task 5B generation commit)
                    file_sha256  <- read from the GIT OBJECT STORE at that commit
current_source:     commit ac7f8b1778a25fdf5cba4f7402e6306fd6784016  (repo state at Task 5.5 start)
                    file_sha256  <- read from the GIT OBJECT STORE at that commit
```

Both blocks are hashed after **CRLF→LF normalisation**, which makes them comparable and independent of
`core.autocrlf`. New tool: `scripts/split_manifest_provenance.py` (re-runnable, `--check` mode,
refuses to run if the frozen JSONL hashes have moved).

**Two real defects were found in the old map, not one.** The task described the post-generation refresh
of `scripts/build_spatial_reason.py`; investigating it also exposed a second, previously unnoticed one:

| File | Old recorded value | Truth | Defect |
|---|---|---|---|
| `scripts/build_spatial_reason.py` | `29c27bb2…` | generation-time `9345feda…` | **silently refreshed** after generation — the reported defect |
| `spatial_reasoning/relations.py` | `ec0fd613…` | committed blob `d45688a6…` | hashed in **CRLF working-tree form**, not the committed LF blob: 900 CRLF pairs vs 0. Semantically identical, but the map described no reproducible code state |

The other six files were correct. The old map, with its superseded values and the reason, is preserved in
the manifest as `provenance_split_note` rather than deleted.

**Verification result:** exactly **one** file differs between `generation_source` and `current_source` —
`scripts/build_spatial_reason.py` — which is precisely the maintenance edit the task described. The
consistency gate now verifies *both* blocks against Git history, so a future silent refresh of the
generation-time record fails the gate.

### 2.2 §3.2 — nested `PENDING_CONSISTENCY_GATE`

Confirmed and fixed. Before the fix, the quality JSON read:

```
top-level verdict        : PASS
embedded quality_verdict : PENDING_CONSISTENCY_GATE
```

Cause: the consistency gate ran before the final verdict existed, so it recorded the pre-verdict
placeholder. Fix: `run_audit()` now computes the per-record verdict, runs the gate with it, recomputes
the verdict if the gate itself fails, and then **writes the final verdict back into the embedded
`artifact_consistency.quality_verdict` field**. A full re-run was required and was performed; the
top-level verdict and the embedded verdict now agree, and the live consistency check is `consistent`.

---

## 3. Sources Reviewed

The consolidated, deduplicated register is `docs/research/task5_5_sources.md`; the raw evidence notes are
in `docs/research/_raw/` (six files, retained as the evidence trail).

**A network reality that matters for reading this package.** The research tools (`web_fetch`,
`web_search`) **cannot** retrieve `github.com`, `raw.githubusercontent.com` or `huggingface.co` on this
machine: the Windows `hosts` file contains a 210-entry blocklist installed by **Steam++ / Watt Toolkit**
(`# Steam++ Start` … `# Steam++ End`) mapping those hosts to `127.0.0.1`. However, a local listener on
`0.0.0.0:443` from the same tool transparently proxies them, so **ordinary HTTP clients succeed** —
verified first-hand by retrieving genuine Hugging Face API JSON, real `LICENSE` file text from
`raw.githubusercontent.com`, and successful `git ls-remote` calls against GitHub.

Consequences, stated plainly:

* The six delegated research agents were restricted to `web_fetch`/`web_search` and therefore recorded
  many hosts as unreachable. Two of them declined to use a shell on principle; that is defensible and is
  recorded as such, not as an error.
* **I re-verified the decision-critical licences and sizes myself through the working HTTP path.** Those
  values are labelled `VERIFIED` / `PLATFORM_FIELD` in the deliverables; everything still unresolved is
  labelled `UNVERIFIED` and is not relied on.
* `hf-mirror.com` resolves but is a third-party mirror and was never used as licence evidence.

---

## 4. Literature Landscape

43 works are compared in `docs/research/task5_5_literature_matrix.md` across architecture, mechanism,
resources, licences, training cost and a nine-mechanism inventory. Highlights:

* **LISA** (CVPR 2024) established the `[SEG]`-token → SAM mask-decoder pattern; **LISA++** exists but has
  **no venue** (author page: technical report); **GSVA** (CVPR 2024) generalises to multiple `[SEG]` plus
  a `[REJ]` token; **PixelLM** (CVPR 2024) uses a codebook with a lightweight decoder; **GLaMM**
  (CVPR 2024) adds a Region Encoder and the GranD corpus; **PSALM** (ECCV 2024) uses a `[REF]` token.
* **Sa2VA** (arXiv 2501.04001, accepted IEEE TPAMI 2026) is the MLLM + SAM-2 family, single-stage
  instruction tuning, SAM-2 decoder **and memory frozen**; **SAMTok** (CVPR 2026) is a *separate*
  mask-tokenizer contribution (2 tokens per mask) and is **not** the tokenizer inside Sa2VA.
* The remote-sensing line is active: **SegEarth-R1** (arXiv 2504.09644) and **SegEarth-R2** (CVPR 2026
  open access), **EarthReason**, **FIRM** (benchmarks: LaSeRS, EarthReason, DRSeg, RRSIS-D, RISBench;
  trains on LaSeRS), **Think2Seg-RS** (arXiv 2512.19302 — official source verified to exist),
  **PixDLM/DRSeg** (CVPR 2026), **TerraScope/TerraLogic**, **GeoSeg**.
* **Corrections to seed assumptions:** the seed's `READ` entry was wrong in **title and arXiv id** (the
  real work is *Reasoning to Attend…*, CVPR 2025, arXiv 2412.17741); "OMGM" does not exist;
  LISA++ has no venue; Sa2VA is not a GLaMM successor; RS-RefSeg is `RSRefSeg`.

**The three gaps that matter for this project** (each an evidenced absence, per the survey):

1. **No work supervises the *relation* geometrically** — nothing stores a spatial-relation triple and
   re-checks it against predicted region geometry. The nearest cases are all something else: a mask-IoU
   GRPO reward (Think2Seg-RS), an attention-space loss from a downsampled GT mask (SegEarth-R2),
   reference-mask supervision (SegLLM), deterministic GT mask codes (FIRM).
2. **No learned relation encoder over the model's own predicted masks/boxes** (`NO_EVIDENCE`).
   SegLLM's mask-encoding loop is the closest, and it re-injects a CLIP masked-crop embedding plus a
   mask-derived bounding-box embedding into the **LLM**, not into a relation module.
3. **`[REF]`-style reference tokens already exist** (SegLLM with supervised reference-mask decoding;
   SAMTok and OMG-LLaVA with mask-in tokens) — yet they are absent from the entire remote-sensing line,
   and no remote-sensing work is building-relation-specific.

---

## 5. Novelty Collision

Full audit: `docs/research/task5_5_novelty_collision.md`.

**The damaging finding: the `[REF]` token is prior art.** **SegLLM** (arXiv 2410.18923, Oct 2024)
generates both `[REF]` and `[SEG]`, decodes **two** masks (`F([REF],[PAD]) → M_ref`,
`F([REF],[SEG]) → M_tgt`), supervises **both**, re-injects predicted-region geometry (mask-derived
bounding-box positional embedding + masked-object CLIP embedding), and performs multi-round relational
reasoning. **PSALM** also uses a `[REF]` token. On top of that, the **ISPRS Annals XI-2-2026** paper
(DOI `10.5194/isprs-annals-XI-2-2026-857-2026`, online 2026-07-03, abstract read first-hand) adapts
PaliGemma 2 into a "unified geospatial building analyzer" from a 16,500-sample instruction-tuning dataset
built out of building polygons — closing the "instruction-tuned VLM for buildings" framing.

| Claim | Verdict | Action |
|---|---|---|
| Geometry-verifiable spatial reasoning dataset | `NO_EVIDENCE` for relation-level verification | **retain, sharpen** |
| `[REF] + [SEG]` | **`DIRECT_OVERLAP`** | **narrow** to the geometric use of the predicted reference |
| Semantic–spatial–visual fusion | `PARTIAL_OVERLAP`, crowded | **reframe** as an ablation |
| Spatial Relation Encoder | **`NO_EVIDENCE`** | **retain as the lead contribution** |
| Spatial Consistency Loss | `NO_EVIDENCE` for a relation-consistency loss | **retain, define precisely** |
| Building/RS specialisation | `PARTIAL_OVERLAP` | **retain as the setting, not a contribution** |
| Tokenised mask representation (FIRM/SAMTok) | relevant, different stage | **postpone** |
| "first"/"novel" for `[SEG]`, `[REF]`, fusion, building instruction-tuning | false or unproven | **drop** |

**Most threatening prior work: SegLLM.** The defence is narrow but real: no surveyed work has a
dedicated relation encoder over predicted geometry, a relation-level consistency loss, or
geometry-verifiable relation supervision and evaluation.

**Narrative change:** move from a *mechanism* claim to a **verification-first** claim — (1) spatial
supervision that can be *proved* correct (25,229 samples, independent oracle, zero violations, artifact
gate); (2) the model contribution is the **explicit relation pathway**; (3) the evidence is
relation-level metrics that only a geometry-verifiable dataset makes possible.

---

## 6. Base MLLM Comparison

All sizes and licences below were verified first-hand (official config files, HF API file listings,
ModelScope first-party API).

| Model | Params | BF16 weights | Weight licence | Verdict |
|---|---|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | ~4.44 B | 8.27 GiB | **apache-2.0** | **PRIMARY** |
| `Qwen/Qwen3-VL-2B-Instruct` | ~2.13 B | 3.96 GiB | **apache-2.0** | **FALLBACK + Stages 0–2** |
| `Qwen/Qwen3-VL-4B-Thinking` | ~4.44 B | 8.27 GiB | apache-2.0 | not for the MVP — forced CoT before the mask token |
| `Qwen/Qwen2.5-VL-7B-Instruct` | ~8.29 B | 15.45 GiB | apache-2.0 | BF16 exceeds VRAM; QLoRA only |
| `Qwen/Qwen2.5-VL-3B-Instruct` | ~3.75 B | 6.99 GiB | **UNVERIFIED** | **EXCLUDED on licence grounds** — HF exposes no licence tag and ModelScope's `License` field is empty |
| `Qwen/Qwen3.5-2B` / `-4B` | 2.28 / 4.66 B | 4.24 / 8.68 GiB | apache-2.0 | watch list (2026 family, no pixel-level recipe yet) |

Verified architecture (config.json): Qwen3-VL-4B text hidden **2560**, **36 layers**, FFN 9728, 32/8
heads, context 262,144, vocab 151,936, vision patch **16** merge 2; documented special-token ids
`vision_start 151652`, `vision_end 151653`, `image 151655`. A `[SEG]` id can therefore be added at
151,936+ with `resize_token_embeddings` and read from `output_hidden_states`.

Notes: Qwen3-VL needs `transformers` ≥ 4.57.0; **FlashAttention is recommended but optional**;
Qwen3-VL's released `preprocessor_config.json` contains **no** `min_pixels`/`max_pixels` keys, so the
image budget must be set explicitly (Qwen2.5-VL's defaults are `min_pixels 3136`,
`max_pixels 12,845,056`).

---

## 7. Segmentation Pathways

| Route | Assessment |
|---|---|
| **A — LISA-style direct `[SEG]`** | **chosen.** Only route where every component is licence-verified and locally feasible, with public reference implementations |
| **B — Sa2VA-style adaptation** | its *recipe* is valuable and free; the *checkpoints* are not (18.84 GiB for 4B, 9.93 GiB for 2B), the code is research-grade, and its training scale is unreachable. Use as a reference, optionally as a weight initialiser for the projection |
| **C — Think2Seg-RS-style decoupled prompting** | **baseline only.** Cheapest, inference-only, excellent as "how far does prompting get?", but it contains no learnable relation representation and cannot support the contribution |
| **D — SegEarth-R1/R2-style learned mask decoder** | related work; not adopted — it is a general geospatial decoder, not a relation-aware one |
| **E — FIRM / SAMTok tokenized masks** | relevant to small/adjacent buildings, but replaces the decoder: postponed to related/future work |

---

## 8. `[REF] + [SEG]` Feasibility

The target architecture is feasible, and the **critical rule is preserved**: ground-truth geometry is
never an inference-time input (ADR-002). Three inference-valid reference sources were assessed; the
recommendation is **source (1), the predicted `[REF]` mask**, because BuildSpatialReason already stores
`reference_component_ids` for every Level-2/3 sample, so **direct reference-mask supervision is available
at zero annotation cost** — a property of this dataset, not of the prior work.

Staging: the **MVP trains `[SEG]` only**; `[REF]` is added at Stage 3. That keeps the moving parts at a
minimum and produces the baseline the final model must beat. The MVP cannot block the Spatial Relation
Encoder provided the interface is fixed now (see §13), because the encoder consumes *predicted* reference
geometry, *predicted* candidate geometry and visual features — all of which the MVP already produces.

---

## 9. Hardware Feasibility

Full analysis: `docs/research/task5_5_hardware_feasibility.md`. **MEASURED on the target machine:**

| Property | Value |
|---|---|
| GPU / capability | RTX 5080 Laptop GPU, **(12, 0)** = `sm_120` |
| Total VRAM | **15.89 GiB** |
| PyTorch / CUDA | **2.13.0+cu132** (CUDA 13.2), `sm_120` present in `get_arch_list()`, `cuda.is_available() == True` |

This closes the Blackwell risk natively on Windows with no installation. Estimated totals (ESTIMATE, not
measurements): Qwen3-VL-2B BF16 LoRA ≈ **6.6–7.1 GiB** (SAFE_LOCAL); Qwen3-VL-4B BF16 LoRA ≈
**11.6–12.6 GiB** (SAFE_LOCAL, ≈3 GiB headroom); Qwen3-VL-4B 4-bit QLoRA ≈ 5.9–6.9 GiB;
Qwen2.5-VL-7B BF16 **does not fit**; Sa2VA-Qwen3-VL-4B **does not fit**.

Crucially, **this project's images are 512×512**, i.e. **256 visual tokens** per tile with Qwen3-VL's
patch-16/merge-2 tokenisation — roughly 50× smaller than Qwen2.5-VL's default pixel budget. Image
resolution is the largest single lever on activation memory, and the dataset's native resolution is
already the economical setting. Storage is a non-issue: the whole recommended stack plus the primary
dataset is well under 30 GB.

---

## 10. Windows vs WSL2/Linux

**Recommendation: native Windows 11, in a new dedicated conda environment** (`yolo_sam_env` is never
modified). **WSL2 Ubuntu is the documented fallback.**

The deciding facts point in opposite directions, and both are recorded:
* **For WSL2:** FlashAttention has **no official Windows support and no prebuilt Windows wheels**
  (`flash-attn` ships sdist only; classifier "Operating System :: Unix"), and its documented GPU list
  stops at Hopper — **no `sm_120`**. SAM 2's own guidance for Windows is reportedly to use WSL (that
  source was not retrievable first-hand).
* **For native Windows:** FlashAttention is **optional** for Qwen3-VL, and SDPA is measured working; at
  ~768-token sequences with a 256-token image prefix it is the least valuable regime for FlashAttention.
  `bitsandbytes` **does** officially support Windows 11 with QLoRA 4-bit. And `sm_120` + PyTorch is
  already measured working natively.

Choosing the environment whose critical path is already measured working, rather than adding a second
filesystem and a GPU-passthrough layer for a marginal attention-kernel gain, is the least disruptive
technically reliable choice.

---

## 11. External Dataset Decisions

Full detail: `docs/research/task5_5_external_datasets.md` (28 datasets).

**The decisive answer: NO.** No public remote-sensing reasoning-segmentation dataset was verified to
combine **instance-level building masks** with **multi-hop spatial instructions**. The evidence is
per-dataset: BRIGHT has ~291k building instance polygons but no language; LaSeRS and DRSeg have instance
masks but unverified building coverage; EarthReason has reasoning but region masks; RISBench and RRSIS-D
have no building class and only explicit referring; DOTA-v2/DIOR/VRSBench/SAMRS have neither.

* **Licence-clean (verified):** EarthReason (`earth-insights/EarthReason`, **apache-2.0**, **1.32 GiB**),
  RefSegRS (`JessicaYuan/RefSegRS`, cc-by-4.0, 2.77 GiB), and LaSeRS (`earth-insights/LaSeRS`,
  apache-2.0, 11.65 GiB — with the caveat that its SAMRS-derived masks inherit academic/non-commercial
  upstream terms). Sizes are verified from the HF API file listings.
* **Rejected / non-commercial:** DRSeg (`cc-by-nc-4.0`), DIOR (`CC BY-NC 4.0`), DOTA-v2 (academic only),
  RISBench and RRSIS-D (own terms unresolved plus upstream non-commercial constraints).
* **22 of 28 datasets remain `UNVERIFIED / DO NOT USE UNTIL RESOLVED`.**
* **Decision: no external dataset is merged into training.** Warm-up is unnecessary; the clean candidates
  are reserved for evaluation and robustness, and published RRSIS-D/RISBench numbers may be used only as
  a literature comparison, never re-presented as our own run.

The ISPRS Annals XI-2-2026 building-VLM paper was the one item that could have overturned this; I read
its abstract and citation metadata first-hand. It is a model/adaptation study whose dataset spans
segmentation, detection, VQA and captioning — **the answer stays NO** and the paper's effect is confined
to the novelty audit (§5).

---

## 12. Evaluation Plan

Full protocol: `docs/research/task5_5_evaluation_plan.md`. Summary:

* **Splits:** train 15,592 / val 3,884 / test 5,753; test touched once; scene-level leakage remains
  `unverified` and must be stated as a caveat.
* **Segmentation:** mIoU (primary), cIoU (reported alongside, never instead), gIoU, Dice; boundary F1
  only if a boundary argument is actually made.
* **Component-level (this project's distinctive axis):** target-selection accuracy,
  component-IoU-after-snapping, and an **adjacency confusion rate** distinguishing "wrong building,
  specifically the reference" from merely poor masks.
* **Reasoning:** reference-selection accuracy, **relation satisfaction rate** (the geometry-verifiable
  metric the dataset exists to enable), target-selection accuracy, **multi-hop success on the
  nontrivial Level-3 subset (1,662 samples) as the primary metric**, trivial Level-3 (1,256) reported
  separately and never merged.
* **Spatial consistency:** direction / size / nearest consistency — defined now so the MVP produces a
  baseline number that the final model must beat, and so they cannot be redefined later.
* **Baselines:** frozen YOLOv8m-seg-WHU (floor; a *different task*, must be footnoted), non-reasoning
  referring segmentation, a `[SEG]`-only model, and a structured-prompting route.
* **Discipline:** no cross-dataset metric comparison as if equivalent; a single run is labelled a single
  run.

---

## 13. Primary MVP Stack

**`Qwen/Qwen3-VL-4B-Instruct` + `facebook/sam2.1-hiera-large` + LISA-style `[SEG]`, BF16 LoRA rank 16.**
(Stages 0–2 use `Qwen/Qwen3-VL-2B-Instruct`.)

* Trainable: LoRA r=16 on `q,k,v,o,gate,up,down` (≈31.3 M params), `[SEG]`/`[REF]` embedding rows,
  projection MLP, SAM2 mask decoder.
* Frozen: Qwen3-VL vision tower, LLM base weights, SAM2 image encoder and memory module.
* Images: native 512×512, no upscaling, pixel budget set **explicitly**, target sequence ≤768 tokens.
* Interface reserved for the final model: projection MLP and relation encoder stay separate modules; all
  inference-time geometry comes from the model's own predicted masks; the MVP records, per sample, the
  predicted reference mask, the predicted target mask and the stated relation, so the Spatial
  Consistency Loss is computable later without changing the MVP output.

---

## 14. Fallback / Baselines

* **Fallback:** `Qwen/Qwen3-VL-2B-Instruct` + `facebook/sam2.1-hiera-base-plus`, same recipe. Triggered by
  OOM at 4B after gradient checkpointing and image-budget reduction, or unacceptable step time. It is a
  configuration change, not a rewrite.
* **Baseline-only:** Think2Seg-RS-style decoupled prompting (frozen SAM2, no LLM training); a
  non-reasoning referring-segmentation model; and the frozen YOLOv8m-seg WHU floor.
* **Not recommended now:** `Qwen2.5-VL-3B` (licence unverifiable), `Qwen2.5-VL-7B` in BF16 (exceeds
  VRAM), Sa2VA-Qwen3-VL-4B as a fine-tuning target (18.84 GiB), `Qwen3-VL-4B-Thinking` for the MVP, and
  Route E tokenized masks.

---

## 15. Task 6 Staged Plan

Design only — nothing executed. Full table with success/failure criteria, checkpoint policy, VRAM safety
and logging in `docs/research/task5_5_mvp_decision.md` §2.

| Stage | Goal | Model | Runtime (ESTIMATE) |
|---|---|---|---|
| 0 | environment + model-load smoke test (incl. **weight-acquisition check**) | 2B | 30–60 min |
| 1 | 2-sample forward/backward proving `hidden([SEG]) → projection → mask` | 2B | 15–30 min |
| 2 | overfit 20 samples to > 0.9 training mIoU | 2B | 30 min – 2 h |
| 3 | 200–500-sample mini-train; **switch to 4B**; add `[REF]` | 4B | 2–6 h |
| 4 | full train, target **< 7 days** | 4B | days |
| 5 | validation + demo per the evaluation plan | 4B | 2–4 h |

The training code must provide: best **and** last checkpoints, periodic saves, resume, mixed precision,
gradient accumulation, gradient clipping (logged), **NaN/inf detection with a clear abort**, documented
OOM-recovery order (checkpointing → smaller image budget → 4-bit QLoRA → smaller model), per-epoch
validation, and early stopping with recorded patience. Every reported number must be stamped with its
split, checkpoint hash and metric definition.

**Entry criteria** are listed in §2.2 of the decision document; they include licence clearance, the
Stage-0 weight-acquisition check, a new dedicated environment, a fixed `[SEG]` token id, and a
re-verification of the frozen dataset.

---

## 16. Remaining Risks

1. **A local accelerator (Steam++ / Watt Toolkit) blackholes `github.com` and `huggingface.co` in the
   hosts file while proxying them via a local listener.** Ordinary HTTP clients currently succeed, which
   is how the licences in this report were verified — but if the accelerator is disabled, Hugging Face
   weight downloads fail and the ModelScope route must be used. **This is a Stage 0 check, and it is not
   merely academic: it decides how Task 6 obtains its weights.**
2. **`bitsandbytes` + `sm_120` + Windows is undocumented** — the QLoRA fallback is not guaranteed.
3. **Every VRAM figure is analytic.** Stage 0–2 exist to replace them with measurements.
4. **Sa2VA repository code licence unresolved**; Sa2VA is used as a *recipe*, never as a dependency.
5. **Residual `UNVERIFIED` licences** across most external datasets — handled by not using them.
6. **The ISPRS Annals building-VLM paper was read at abstract level only.** If its full text released a
   dataset with instance-level building masks plus multi-hop spatial instructions, the dataset-novelty
   claim would need narrowing.
7. **`scene_level_split_leakage = unverified`** in BuildSpatialReason-v0.1.1 — no claim of geographic
   generalisation may be made from test numbers.
8. **The dataset-novelty claim is an evidenced absence**, not a proof. It is phrased as such everywhere.

---

## 17. Files Created / Modified

**Created — deliverables**

| File | Contents |
|---|---|
| `docs/research/task5_5_sources.md` | consolidated deduplicated source register + licence register |
| `docs/research/task5_5_literature_matrix.md` | 43 works, architecture/resources tables, 9-mechanism inventory, gaps |
| `docs/research/task5_5_model_stack.md` | base MLLM comparison, routes A–E, `[REF]+[SEG]` feasibility, decision matrix |
| `docs/research/task5_5_hardware_feasibility.md` | measured environment facts, memory accounting, per-config estimates, OS decision |
| `docs/research/task5_5_novelty_collision.md` | collision matrix, retain/narrow/reframe/postpone/drop, narrative change |
| `docs/research/task5_5_mvp_decision.md` | the 12 required decisions + the staged Task 6 plan |
| `docs/research/task5_5_external_datasets.md` | 28 datasets, the decisive-question answer, licence groups, use plan |
| `docs/research/task5_5_evaluation_plan.md` | paper-ready evaluation protocol |
| `evaluation/task5_5_stack_decision.json` | machine-readable stack decision with evidence labels |
| `docs/research/_raw/*.md` (6 files) | raw evidence notes, retained as the trail |

**Created — tooling**

| File | Purpose |
|---|---|
| `scripts/split_manifest_provenance.py` | reconstructs generation-time provenance from Git objects; `--check` mode |

**Modified**

| File | Change |
|---|---|
| `datasets/build_spatial_reason/v0.1.1/manifest.json` | provenance split into `generation_source` / `current_source`; legacy map preserved under `provenance_split_note` |
| `scripts/build_spatial_reason.py` | emits the two provenance blocks; CRLF-normalised source hashing; `_git_head()` |
| `scripts/check_artifact_consistency.py` | provenance gate now verifies **both** blocks against Git history and rejects the legacy field |
| `scripts/validate_build_spatial_reason.py` | embedded consistency result now quotes the **final** verdict (§3.2 fix) |
| `tests/test_artifact_consistency.py` | provenance test rewritten for the split + Git verification |
| `docs/architecture_decisions.md` | **ADR-012** added, with `PROVISIONAL` sub-decisions and the summary row |
| `README.md` | status now records the MVP stack freeze; new "Frozen MVP stack" and "Contribution framing" sections |
| `handoff/PROJECT_STATE.md` | Task 5.5 state |

**Not modified:** all `*.jsonl` in both dataset versions, `docs/build_spatial_reason_v0.1*`,
`../WHU_Building_Segment/**`, `handoff/TO_DSH.md`.

---

## 18. Git Commit / Push

Commit message: `research: select BuildReasonSeg MVP stack`.
Pre-staging checks: `git status --short` inspected; no JSONL, no weights, no dataset archives, no cloned
repositories, no caches staged; frozen JSONL hashes re-verified; no out-of-repo writes. The commit hash
and push result are in the final DSH chat summary.

---

## 19. Ready for Task 6?

**Yes — the design is frozen and implementable without repeating this survey.** Task 6 can start from
`evaluation/task5_5_stack_decision.json` and `docs/research/task5_5_mvp_decision.md`, which contain the
exact model ids, the frozen/trainable split, the precision, the image policy, the environment, the staged
plan and the entry criteria.

**Four things must be checked at Stage 0 before any implementation effort is spent:**

1. **Weight acquisition** — confirm whether the Hugging Face route works or the ModelScope route is
   needed (risk 1 in §16). This is the only item that could force a change to how Task 6 starts.
2. **Licence clearance of anything newly introduced** — the chosen stack is clean; a QLoRA fallback is
   not, until `bitsandbytes` + `sm_120` on Windows is confirmed.
3. **A new dedicated environment** — `yolo_sam_env` must not be touched.
4. **Dataset re-verification** — `python scripts/check_artifact_consistency.py` must report
   `consistent` and the v0.1.1 JSONL hashes must be unchanged.

**Task 5.5 did not begin Task 6**: no environment installed, no weights or datasets downloaded, no
inference run, no training, no cloud resources.
