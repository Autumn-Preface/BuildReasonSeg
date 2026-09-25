# Task 5.5 — Model Stack Selection

**Research/access date:** 2026-09-26
**Scope:** which base multimodal LLM, which segmentation pathway, and which module split BuildReasonSeg-MVP
(Task 6) should start from. **Design only — nothing here was downloaded, installed, run or trained.**

Evidence labels used throughout: `VERIFIED` (read first-hand from an official source: model config/API,
LICENSE file, publisher page, vendor documentation), `PLATFORM_FIELD` (an official platform field such as
an HF API `license:` tag or a GitHub API `license.spdx_id` — strong, but not the same as reading a
LICENSE file), `UNVERIFIED` (no reachable source; nothing asserted), `ESTIMATE` (the author's arithmetic
or judgement, explicitly not a measurement).

---

## 1. Base MLLM candidates

### 1.1 Verified facts

| Model ID | Params | Weights (BF16) | **Weight licence** | Context | Text hidden / layers / FFN | Vision | Weight format | Verified by |
|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | ~2.13 B | **3.96 GiB** | **apache-2.0** | 262,144 | 2048 / 28 / 6144 | depth 24, h1024, patch 16, merge 2 | safetensors, BF16 | HF API tag + `config.json` + ModelScope `License` |
| `Qwen/Qwen3-VL-4B-Instruct` | ~4.44 B | **8.27 GiB** | **apache-2.0** | 262,144 | **2560 / 36 / 9728** | depth 24, h1024, patch 16, merge 2 | safetensors, BF16 | HF API tag + `config.json` + ModelScope |
| `Qwen/Qwen3-VL-4B-Thinking` | ~4.44 B | 8.27 GiB | **apache-2.0** | 262,144 | same as 4B-Instruct | same | safetensors, BF16 | HF API tag |
| `Qwen/Qwen2.5-VL-3B-Instruct` | ~3.75 B | 6.99 GiB | **UNVERIFIED / DO NOT USE UNTIL RESOLVED** | — | — | — | safetensors, BF16 | **HF API exposes NO licence tag; ModelScope `License` field empty** |
| `Qwen/Qwen2.5-VL-7B-Instruct` | ~8.29 B | **15.45 GiB** | **apache-2.0** | 128,000 | 3584 / 28 / 18944 | depth 32, h1280, patch 14, merge 2 | safetensors, BF16 | HF API tag + `config.json` |
| `Qwen/Qwen3.5-2B` | ~2.28 B | 4.24 GiB | **apache-2.0** | 262,144 → ~1.01 M (YaRN) | — | natively multimodal (`Qwen3_5ForConditionalGeneration`, task `image-text-to-text`) | safetensors, BF16 | ModelScope API |
| `Qwen/Qwen3.5-4B` | ~4.66 B | 8.68 GiB | **apache-2.0** | 262,144 → ~1.01 M | — | natively multimodal | safetensors, BF16 | ModelScope API |

All three Qwen3-VL variants expose the same special-token ids in `config.json`:
`vision_start_token_id = 151652`, `vision_end_token_id = 151653`, `image_token_id = 151655`,
`video_token_id = 151656`, vocabulary 151,936. **A `[SEG]` token can therefore be added as a new id at
151,936+ with `resize_token_embeddings`, and read back from `output_hidden_states`** — this is exactly
what the Sa2VA Qwen3-VL release does, which confirms the mechanism is workable on this architecture.

### 1.2 Compatibility and runtime

| Question | Finding | Status |
|---|---|---|
| Minimum framework version | Qwen3-VL needs `transformers` ≥ 4.57.0; the model card at capture time said 4.57.0 was unreleased and advised building from source | VERIFIED (ModelScope card text) |
| FlashAttention required? | **Optional.** The card *recommends* `attn_implementation="flash_attention_2"`; the default path works without it | VERIFIED |
| BF16 support | Yes — the published tensor type is BF16 | VERIFIED |
| FP16 support | **Not documented** | UNVERIFIED |
| `min_pixels` / `max_pixels` | Qwen2.5-VL ships `min_pixels = 3136`, `max_pixels = 12,845,056` in `preprocessor_config.json`. **Qwen3-VL's released preprocessor config contains neither key** — only `patch_size 16`, `merge_size 2`, `temporal_patch_size 2`. The Qwen3-VL image budget must therefore be set explicitly and verified in Stage 0 | VERIFIED (config files) |
| QLoRA / bitsandbytes | Not documented for the Qwen families; `bitsandbytes` itself officially supports Windows 11 with QLoRA 4-bit | UNVERIFIED for these models / VERIFIED for bnb |
| Hidden state of a custom token readable | Yes — `output_hidden_states` + index selection; confirmed by the Sa2VA Qwen3-VL release | VERIFIED (architecture) / PLATFORM_FIELD (Sa2VA) |
| Windows-native feasibility | PyTorch + `sm_120` **measured working** locally; `decord` is not installable on non-Linux (use `torchvision`) | MEASURED |
| Official Windows/WSL guidance | None found for Qwen3-VL | UNVERIFIED |

### 1.3 Choosing the size

| Candidate | Fit | Reasoning |
|---|---|---|
| **Qwen3-VL-4B-Instruct** | **primary** | 8.27 GiB BF16 LoRA ≈ 11.6–12.6 GiB total (ESTIMATE) — fits 15.89 GiB with ≈3 GiB headroom; Apache-2.0 verified; strongest spatial-perception claims in the Qwen3-VL family (2D grounding + 3D grounding); same token layout as the Sa2VA reference implementation, so the integration is de-risked by an existing public recipe. |
| **Qwen3-VL-2B-Instruct** | **fallback / pipeline prover** | 3.96 GiB ⇒ ≈6.6–7.1 GiB total: fits with ~9 GiB to spare, so it is the right vehicle for Stage 0–2 smoke tests and the immediate fallback if 4B training turns out to be tight in practice. Same architecture and token ids, so code is portable between them. |
| Qwen3-VL-4B-**Thinking** | not for the MVP | Apache-2.0 and the same size, but the shipped chat template forces a long chain-of-thought before any answer, and the segmentation token can only appear *after* it. That adds a large supervised-token budget that is irrelevant to segmentation and complicates mask-token supervision. Useful later as a reasoning-quality ablation. |
| Qwen2.5-VL-7B-Instruct | not for the MVP | Licence is clean (apache-2.0) but 15.45 GiB of BF16 weights **exceed VRAM**; only a 4-bit QLoRA run fits, which then depends on an unverified bnb+sm_120 combination. Also a vision **encoder patch of 14** vs 16, and no public pixel-level recipe at this size. |
| Qwen2.5-VL-3B-Instruct | **excluded** | The licence cannot be established from any reachable first-party source (HF exposes no licence tag; ModelScope's field is empty). Per the evidence policy this is `UNVERIFIED / DO NOT USE UNTIL RESOLVED` and it is removed from consideration regardless of its attractive size. |
| Qwen3.5-2B / Qwen3.5-4B | watch list | Newer (2026) Apache-2.0 compact multimodal family with thinking-by-default and an `enable_thinking=False` switch. Attractive, but no pixel-level integration recipe exists yet, so adopting it would mean building the `[SEG]` pathway from scratch on an unproven base. Revisit after the MVP pipeline works. |

---

## 2. Existing pixel-level integrations

| Option | What exists | Licence | Fit for this project |
|---|---|---|---|
| **Sa2VA-Qwen3-VL-4B** (`ByteDance/Sa2VA-Qwen3-VL-4B`) | Released; 5 shards, **18.84 GiB**; HF API tag `apache-2.0`, not gated. Paper: arXiv 2501.04001, accepted IEEE TPAMI 2026. SAM-2 decoder **and memory module frozen**, single-stage instruction tuning. | `apache-2.0` (PLATFORM_FIELD) | **Too large to fine-tune in BF16 on 16 GB.** Its *value here is as a reference recipe*, not as a training target: it demonstrates precisely the `[SEG]`-token → hidden-state → projection → SAM2 path this project needs. |
| `ByteDance/Sa2VA-Qwen3-VL-2B` | Released; 3 shards, **9.93 GiB**; `apache-2.0` | PLATFORM_FIELD | Borderline fit (≈12–13 GiB ESTIMATE). Viable as a *fallback* integration if building the pathway from scratch fails, but it arrives with a research-grade code base and no documentation of the exact recipe. |
| `ByteDance/Sa2VA-Qwen2_5-VL-3B` (16.00 GiB) / `ByteDance/Sa2VA-InternVL3-2B` (8.06 GiB) | Released; `apache-2.0` tags | PLATFORM_FIELD | The InternVL3-2B variant is small, but InternVL is a third base family and would add a second integration path for no benefit. |
| SAMTok | CVPR 2026, pp. 37852–37863. A SAM2-based residual-VQ **mask tokenizer** producing exactly 2 tokens per mask. **Not** the tokenizer inside Sa2VA — Sa2VA emits one continuous `[SEG]` hidden state. | UNVERIFIED | Interesting for a *tokenized-mask* route (Route E) but it is a separate mechanism; out of scope for the MVP. |

**Important negative finding:** the Sa2VA **repository code licence** could not be verified first-hand
during this task, and its training recipe is documented at a scale (large curated video/instruction
corpus) that a 16 GB laptop cannot reproduce. Its *papers* are clear; its *release* is research-grade.

---

## 3. Segmentation pathways

### Route A — LISA-style direct `[SEG]` (recommended)
```
MLLM hidden([SEG])  ->  projection MLP  ->  mask decoder  ->  mask
```
- **Mechanism:** a single `[SEG]` token is appended to the answer; its last-layer hidden state (2560-d
  for Qwen3-VL-4B) is projected to the decoder's prompt space.
- **Decoder:** SAM 2.1 `hiera-large` (**1.67 GiB on disk**, licence **Apache-2.0 verified from the
  LICENSE file**) — or `hiera-base-plus` (0.60 GiB) if VRAM is tight. SAM 2's image encoder and memory
  module are **frozen** in the published LISA/Sa2VA recipes; only the mask decoder is trained.
- **Original SAM vs SAM2:** SAM 1 ViT-H is 4.78 GiB and architecturally heavier for no gain here; SAM 2.1
  hiera-large is smaller, newer and equally licensed. Both are Apache-2.0 (SAM 1 licence also verified
  first-hand).
- **Train/freeze:** freeze the vision tower, freeze SAM2; train LoRA adapters on the LLM plus the
  projection MLP plus the SAM2 mask decoder.
- **`[REF] + [SEG]` compatibility:** excellent — adding a second token to the same answer stream is a
  one-line vocabulary extension, and the projection/decoder path is reused unchanged.
- **Risk:** the `[SEG]` hidden state must carry enough information. This is the standard, well-trodden
  route with public reference implementations, and it is the only route whose every component is
  licence-verified and locally feasible.

### Route B — Sa2VA-style MLLM + SAM2 adaptation
- **Lower risk than building from scratch?** Only partially. The *architectural* knowledge is valuable
  and free (paper + public release), but the released checkpoints are 18.84 GiB (4B) / 9.93 GiB (2B) and
  are not sized for a 16 GB LoRA fine-tune, the code is research-grade, and the recipe assumes a data
  scale this project does not have.
- **Verdict:** use Sa2VA as the **reference recipe and a possible pretrained initialisation**, not as the
  thing to fine-tune. Concretely: copy its projection design; consider initialising the projection from
  `Sa2VA-Qwen3-VL-2B` later if a cold start fails.

### Route C — Think2Seg-RS-style decoupled prompting (baseline only)
- `MLLM -> structured point/box/geometric prompts -> frozen SAM2`.
- **Think2Seg-RS is real** and its official source was verified first-hand: arXiv **2512.19302**
  (v1 2025-12-22, v2 2026-04-21), positioned in the ISPRS Journal of Photogrammetry and Remote Sensing.
- **Implementation simplicity:** highest of all routes — no new tokens, no projection, no decoder
  training; SAM2 stays frozen and is driven by prompts.
- **16 GB feasibility:** excellent, and it is *inference-only* if used as a baseline.
- **Value as a baseline:** high. It is the strongest "reasonable engineering alternative" and it makes
  the ablation "does explicit relational reasoning beat structured prompting?" measurable.
- **Divergence from the narrative:** total. It contains no learned spatial relation representation, so it
  cannot support the Spatial Relation Encoder or the Spatial Consistency Loss. It is a baseline, never
  the contribution.
- **Novelty collision:** none for this project (it is a different, decoupled mechanism).

### Route D — SegEarth-R1-style learned mask decoder
- Remote-sensing-specific, and the family is active (SegEarth-R1, then **SegEarth-R2** in CVPR 2026
  open access). Relevant prior art, but it is a *general geospatial* mask decoder, not a relation-aware
  one, and reproducing it requires its own training pipeline.
- **Verdict:** read as related work; do not adopt as the MVP pathway. Revisit if the project needs a
  remote-sensing-domain decoder.

### Route E — FIRM / tokenized mask representation
- Representing a mask as discrete tokens (SAMTok: 2 tokens per mask; FIRM: fine-grained intra-token mask
  representation) is genuinely interesting for **small and adjacent remote-sensing objects**, which is
  exactly this project's hardest regime (small buildings, touching components).
- **Engineering complexity:** high — it replaces the mask decoder with a tokenizer/decoder pair and
  changes the supervision target.
- **Verdict:** **not for the MVP**; keep it as a documented future direction and as related work. It
  belongs to the final model or to the discussion, not to the first working pipeline.

---

## 4. `[REF] + [SEG]` feasibility

Target architecture:
```
Image + Instruction -> MLLM -> reasoning text + [REF] + [SEG]
  -> [REF] hidden -> predicted reference mask / representation
  -> inference-available geometry / spatial prior
  -> [SEG] hidden + spatial prior + visual features -> target mask decoder
```
**Critical rule (ADR-002, restated): ground-truth geometry must never become an inference-time input.**

Three inference-valid reference sources:

| | (1) Predicted `[REF]` mask | (2) Predicted point / box / proposal | (3) Learned latent spatial map |
|---|---|---|---|
| **Supervision** | **Directly available and free in BuildSpatialReason**: every Level-2/Level-3 sample stores `reference_component_ids`, so a GT reference mask exists for 7,954 of 25,229 samples without any new annotation | Same supervision, but only at box/point precision | Indirect only; no reference-region supervision exists |
| **Differentiability** | Yes — the same projection + mask head as the target, a second query | Yes — box/point heads are differentiable | Yes |
| **Inference availability** | Yes: the model predicts the reference, then consumes its own prediction | Yes | Yes |
| **Implementation complexity** | Moderate: a second token, a second decoder query, a second loss term | Low | Moderate–high (needs a bespoke geometry encoder and a way to train it) |
| **Failure modes** | Reference error propagates into the spatial prior; a wrong reference yields a confidently wrong target. Mitigated because the reference is itself supervised | Weaker spatial prior; boxes over-cover irregular buildings | Opaque; hard to attribute failure; risks being a second, uninterpretable attention pathway |
| **VRAM impact** | + one SAM2 decode pass per step (small; SAM2.1 hiera-large is 1.67 GiB on disk and frozen) | negligible | moderate |
| **Literature overlap** | **SegLLM already does this** (supervised reference mask + mask-derived bounding-box embedding re-injected into the LLM input) ⇒ DIRECT/PARTIAL overlap on the *token*, PARTIAL on the *mechanism* | common in referring segmentation | NO_EVIDENCE for this exact form |
| **MVP vs final** | **final-model mechanism**, staged in after the single-token pipeline works | could be an MVP stepping stone | final-model research direction |

**Recommendation.**
- **MVP (Task 6): `[SEG]` only.** Get the single-token pipeline working and measured first. This keeps
  the number of moving parts at its minimum and produces the baseline the final model must beat.
- **Stage 3+ of the MVP: add `[REF]`.** The dataset already provides GT reference regions for Level-2/3,
  so this is the single highest-value extension available at zero annotation cost. It is also the point
  at which the project's "reference-guided" narrative becomes true rather than aspirational.
- **Reference-source choice: (1), the predicted `[REF]` mask**, because it is the only option with free
  direct supervision — and that free supervision is a genuine property of BuildSpatialReason, not of the
  prior work.
- **Not now: (3).** A latent spatial map without an explicit reference cannot be supervised, cannot be
  interpreted, and cannot be ablated cleanly.

**Does the MVP block the future Spatial Relation Encoder?** No, provided the interface is fixed now:
the encoder consumes *(predicted reference geometry, predicted target candidate geometry, visual
features)*. So the MVP must (a) expose the predicted reference mask when `[REF]` is added, and (b) keep
the geometry derivation in a separate, replaceable module rather than fused into the projection MLP.
ADR-005's rule — the encoder must be inference-valid — is satisfied because the geometry comes from the
model's own predictions.

---

## 5. Final decision matrix

Rows are the candidates required by the task specification. Ratings are judgements informed by the
verified facts above; `EST` marks the estimate cells.

| Criterion | **Qwen3-VL-2B + SAM2.1** | **Qwen3-VL-4B + SAM2.1** | Sa2VA-Qwen3-VL-2B adapt | Sa2VA-Qwen3-VL-4B adapt | Think2Seg-RS-style decoupled | Qwen2.5-VL-7B + SAM2.1 |
|---|---|---|---|---|---|---|
| 16 GB feasibility (LoRA) | SAFE_LOCAL (≈6.6–7.1 GiB EST) | SAFE_LOCAL (≈11.6–12.6 GiB EST) | BORDERLINE (≈12–13 GiB EST) | **NOT feasible BF16** (18.84 GiB weights) | SAFE_LOCAL (inference-only) | BF16 infeasible; QLoRA SAFE_LOCAL |
| Implementation effort | moderate | moderate | **high** (adopt research code) | high | **lowest** | moderate |
| Windows/WSL complexity | native Windows OK | native Windows OK | native OK | — | native OK | Linux-preferred (QLoRA path) |
| Licence clarity | **clean** (apache-2.0 + Apache-2.0) | **clean** | weights clean, **code UNVERIFIED** | weights clean, code UNVERIFIED | paper verified; artifacts UNVERIFIED | clean (apache-2.0) |
| Code maturity | reference recipes public (LISA) | reference recipes public | research-grade | research-grade | paper-level | reference recipes public |
| Special-token flexibility | high (151,936 vocab, documented ids) | high | high (already does it) | high | none needed | high |
| `[REF] + [SEG]` compatibility | **excellent** | **excellent** | good (already has `[SEG]`) | good | not applicable | excellent |
| Remote-sensing relevance | generic VLM | generic VLM | generic | generic | **RS-specific** | generic |
| Inference speed | fastest | fast | slower | slowest | **fastest** | slow |
| Paper extensibility | good | **best** (headroom for the encoder + loss) | constrained | constrained | weak (no learnable relation module) | moderate |
| Novelty collision risk | none (base model) | none | low | low | **none, but no novelty either** | none |

### 5.1 Outputs

**1. Primary MVP stack — `Qwen3-VL-4B-Instruct` + `SAM 2.1 hiera-large` + LISA-style `[SEG]`, BF16 LoRA r=16.**
- LLM: `Qwen/Qwen3-VL-4B-Instruct` (apache-2.0, verified two ways).
- Decoder: `facebook/sam2.1-hiera-large` (Apache-2.0, LICENSE read), image encoder + memory **frozen**,
  mask decoder trainable.
- Trainable: LoRA rank 16 on `q,k,v,o,gate,up,down` (≈31.3 M adapter parameters), the `[SEG]` embedding
  row, and the projection MLP; vision tower frozen.
- Why: everything on the critical path is licence-clean and locally feasible, and the exact
  `[SEG] → hidden → projection → SAM2` path has a public reference implementation on the *same* base
  family, which removes the largest unknown.

**2. Fallback stack — `Qwen3-VL-2B-Instruct` + `SAM 2.1 hiera-base-plus` + the same recipe.**
Trigger conditions: OOM at 4B after gradient checkpointing and image-budget reduction; or an
unacceptably slow step time. Same token ids and architecture, so switching is a configuration change
plus a re-run, not a re-implementation. This is also the stack for Stage 0–2, deliberately.

**3. Baseline-only stack — Think2Seg-RS-style decoupled prompting (Route C) with a frozen SAM2.**
No training of the LLM; used to answer "how far does structured prompting get?" It is the honest
engineering baseline and requires no new tokens.

**4. Not recommended now**
- `Qwen/Qwen2.5-VL-3B-Instruct` — **licence unverifiable** (`UNVERIFIED / DO NOT USE UNTIL RESOLVED`).
- `Qwen/Qwen2.5-VL-7B-Instruct` in BF16 — exceeds VRAM; only QLoRA fits and that depends on an unverified
  bnb + sm_120 combination.
- Sa2VA-Qwen3-VL-4B as a fine-tuning target — 18.84 GiB weights.
- Route E (FIRM / SAMTok tokenized masks) — wrong stage; high complexity before the pipeline exists.
- `Qwen3-VL-4B-Thinking` for the MVP — the forced reasoning trace complicates mask-token supervision.
- Any external dataset merged into training — see `task5_5_external_datasets.md`; warm-up is not needed
  and most external licences are unresolved.
