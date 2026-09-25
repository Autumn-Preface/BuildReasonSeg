# FROM_DSH — Task 6A Report: Native-Windows Environment Bootstrap + 2B `[SEG]` MVP Smoke/Overfit

**Date:** 2026-09-26
**Actor:** DSH
**Task:** Task 6A (from `handoff/TO_DSH.md`)
**Verdict: `PASS`**

> Reported against the Task 6A success criteria: all 20 overfit samples ≥ 0.90 teacher-forced training
> mIoU, ≥ 9/10 same-image paired instruction dependence, 20/20 free-generation `[SEG]` emission, no
> NaN/Inf, no collapse. Measured: **min 0.9366 / mean 0.9807 mIoU**, **10/10 pairs**, **20/20 emission**.

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

## 1. Verdict

**`PASS`** — the smallest end-to-end BuildReasonSeg segmentation pipeline was built and measured on the
user's actual laptop:

```
image + instruction -> Qwen3-VL-2B-Instruct -> reasoning text + [SEG]
                    -> [SEG] hidden state -> Projection MLP
                    -> one sparse prompt embedding -> vanilla SAM2.1 Hiera Base+ mask decoder -> mask
```

| Gate | Result |
|---|---|
| Environment created, existing environments untouched | ✅ |
| Models loaded on 16 GB VRAM, native Windows | ✅ |
| Stage 1 real 2-sample forward/backward, all 9 checks | ✅ |
| Stage 2 teacher-forced mIoU, all 20 samples | **min 0.9366, mean 0.9807** (≥ 0.90 required) |
| Same-image paired instruction dependence | **10 / 10** (≥ 9 required) |
| Free-generation `[SEG]` emission | **20 / 20** |
| No NaN/Inf, no empty/full collapse | ✅ |
| Tests | **156 / 156 passed**, exit 0 |

Machine-readable results: `evaluation/task6a_smoke_report.json` (sections: `environment`,
`model_revisions`, `stage0`, `trainable_params`, `stage1`, `stage2`, `pairwise_instruction_dependence`,
`free_generation_seg_emission`, `vram`, `timings`, `checkpoint_hashes`), plus
`evaluation/task6a_preflight.json`, `evaluation/task6a_environment.json`,
`evaluation/task6a_subset_ids.json`, `evaluation/task6a_checkpoint_manifest.json`.

---

## 2. Environment Created

One dedicated environment, created with **`conda create --prefix`** as instructed — **not venv**, and not
a clone of anything.

| Item | Value |
|---|---|
| Prefix | `.conda/buildreasonseg-mvp` (project-local) |
| Python | 3.11.16 (conda 25.11.0) |
| PyTorch | `2.13.0+cu132` from `https://download.pytorch.org/whl/cu132` |
| `torch.cuda.is_available()` | True |
| Compute capability | (12, 0) = `sm_120` |
| BF16 matmul, SDPA | verified inside the new environment |
| Disk (`local_cache/`) | 4.81 GB |

`base`, `jupyter` and `yolo_sam_env` were **never modified**. `.conda/`, `local_cache/` and `artifacts/`
were added to `.gitignore` **after** the existing "MUST STAY TRACKED" negation block, because the
unanchored `!*.yaml` / `!*.yml` rules match at any depth and would otherwise have re-included
environment files.

**Local TLS trust configuration (necessary, and disclosed).** The machine runs Steam++ / Watt Toolkit,
which terminates TLS for some hosts and re-signs them with its own root ("SteamTools Certificate",
BeyondDimension). That root is in the Windows certificate store, so `ssl.create_default_context()`
trusts it, but the `certifi` bundle used by `httpx`/`huggingface_hub` does not, which made Hub downloads
fail with `CERTIFICATE_VERIFY_FAILED`. `buildreasonseg_mvp/local_env.py` appends the Windows ROOT/CA
stores to this environment's certifi bundle. This is a **process-local** setting: it touches only the
project's own environment and `local_cache/`, changes no registry key, driver, PATH or system variable,
and backs the original bundle up as `cacert.pem.orig`. Both SteamTools roots were located this way.

---

## 3. Packages / Model Revisions

| Component | Version / revision |
|---|---|
| `transformers` | 5.17.0 |
| `peft` | 0.21.0 |
| `accelerate` / `safetensors` / `huggingface_hub` / `tokenizers` | 1.15.0 / 0.8.0 / 1.33.0 / 0.23.2 |
| `qwen-vl-utils` | 0.0.14 |
| `SAM-2` | 1.0 @ **`2b90b9f5ceec907a1c18123530e92e794ad901a4`** |
| Qwen3-VL-2B-Instruct snapshot | **`89644892e4d85e24eaac8bacfd4f463576704203`**, 4.27 GB on disk |
| sam2.1_hiera_base_plus.pt | 323,606,802 bytes |

Lock file: `environment/task6a_requirements_lock.txt` (SAM-2 pinned to the upstream commit, not to a
local path, so the lock is reproducible elsewhere).

SAM2 was installed with **`SAM2_BUILD_CUDA=0`**: the optional compiled CUDA extension is intentionally
skipped on native Windows (no system CUDA toolkit present) and only affects mask hole/sprinkle
post-processing. The SAM2 checkpoint could not be fetched via `hf_hub_download` because the local
accelerator strips the `X-Repo-Commit` header from the resolve redirect; `scripts/task6a_download.py`
falls back to a plain `urllib` download of the same official URL and records which path was used.

---

## 4. Stage 0 Measurements

**Qwen3-VL-2B-Instruct:** BF16; text hidden 2048; vocab 151,669 → 151,670; **`[SEG]` id 151,669**;
single-token round-trip verified; weight tying present before and after resize; total parameters
**2,227,632,642**, trainable **24,010,309 (1.08 %)**; RAM 0.53 → 0.84 GiB.

**Qwen preprocessing (measured, not assumed):** source (512, 512, 3) → `pixel_values` (1024, 1536),
`image_grid_thw = [[1, 32, 32]]` → **256 visual tokens**. Sequence lengths per level: L1 296, L2 324,
L3 332. The image budget is set explicitly on `image_processor.size` (`min_pixels = max_pixels =
262144`) because the released Qwen3-VL preprocessor config carries no top-level pixel keys.

**SAM2.1 Hiera Base+:** `image_size` 1024, `backbone_stride` 16, prompt embed dim **256**, image
embedding (64, 64), `use_high_res_features_in_sam` True; parameters — total 80,850,178, image encoder
69,106,816 (frozen), prompt encoder 6,220 (frozen), **mask decoder 4,215,109 (trainable)**. Transformed
input (1, 3, 1024, 1024) → image embedding (1, 256, 64, 64) plus high-res features (1, 32, 256, 256) and
(1, 64, 128, 128).

**VRAM at rest (both models loaded):** 4.53 GiB allocated / 5.10 GiB reserved of 15.89 GiB.

The Qwen and SAM2 branches are confirmed to be **separate preprocessing paths** from the same 512×512 RGB
tile.

---

## 5. Qwen Preprocessing

Covered in §4. One correction to Task 5.5 is recorded: the design note "512×512 everywhere" is wrong as
a description of the model inputs — the Qwen branch sees 256 visual tokens and the SAM2 branch sees a
1024×1024 normalised tensor. Both derive from the same source tile.

## 6. SAM2 Preprocessing

The official `SAM2ImagePredictor` transform and encoder are used, so the visual path is byte-for-byte the
official behaviour. `Sam2Encoder.encode` reproduces `set_image` exactly: transform → `forward_image` →
`_prepare_backbone_features` → `no_mem_embed` addition → the documented feature reshaping. The official
prompt encoder is consulted only for (a) the positional encoding of one sparse prompt slot and (b) the
no-mask dense embedding; no point, box, mask, centroid, bbox or component id from the annotation is ever
supplied.

---

## 7. `[SEG]` Token / PEFT Verification

* `[SEG]` is **exactly one token** (id 151,669); encode → `[151669]`, decode round-trips.
* Vocab 151,669 → 151,670; embeddings resized **once**; weight tying still True afterwards.
* **Measured detail:** `resize_token_embeddings` *shrinks* Qwen's padded table from 151,936 rows to
  151,670 (tokenizer length + the new token). Those padding rows were unreachable from the tokenizer, so
  this removes dead weight — but it is a real change and is asserted, not assumed.
* `trainable_token_indices=[151669]` is used (PEFT 0.21 supports it). The base embedding table stays
  **frozen** (`requires_grad == False`, 310,620,160 elements).
* **PEFT preserves the tying for the trainable token:** the input adapter and the output head's adapter
  share the **same** delta parameter object, giving exactly **one** trainable `[SEG]` row used on both
  sides. No separate output-row wrapper is needed.
* The optimizer covers **all 528 trainable tensors with 0 missing** (asserted).
* Regression: after one optimizer step the `[SEG]` row moved (max abs delta 1.0e-4) while ordinary rows
  for ids 0, 1, 100, 1000, 50000 and 151000 moved by exactly **0.0**.

---

## 8. LoRA Scope Verification

* 196 target modules, all language-model projections (`...language_model.layers.N.{q,k,v,o,gate,up,down}_proj`).
* **Zero** trainable parameters anywhere under the visual tower, and **zero** LoRA modules under it —
  raised as an error by `attach_lora` if violated, and asserted by the tests.
* LoRA parameters: 17,432,576.

---

## 9. Stage 1 Forward/Backward

Fixed smoke pair: same image `1_0`, different instructions, different query types, different targets
(`bottommost` → 1, `leftmost` → 2). All nine checks passed:

all losses finite · projected shape == SAM prompt dim (256) · mask logits finite and non-constant ·
non-zero finite gradients on LoRA / `[SEG]` / projection / SAM2 mask decoder · zero trainable parameters
in the Qwen visual tower · frozen SAM2 image encoder has no gradient · `[SEG]` row changed after the
optimizer step · ordinary vocabulary rows unchanged · no OOM.

```
lm_logits (1, 290, 151670)  seg_hidden (1, 2048)  projected (1, 256)
sparse prompt (1, 2, 256)   mask logits (1, 1, 256, 256)
grad norms: lora 11.45 | [SEG] 8.60 | projection 5.61 | SAM2 decoder 5.70 | SAM2 encoder 0.0
losses: total 5.038 = lm_ce 4.110 + mask_bce 0.214 + mask_dice 0.9999
```

**Stage 1 peak VRAM: 7.25 GiB allocated / 8.45 GiB reserved.** Stage 1 wall time 35 s including load.

---

## 10. Stage 2 Overfit

Deterministic subset: **10 images × 2 instructions = 20 records**, all train, decided before any training
and never revised using results. Levels: L1 14, L2 1, nontrivial L3 5. Every image has exactly two
different targets.

Five runs were performed; **all are kept**, including the three that did not meet the gate:

| Run | Supervision | Loss (lm/bce/dice) | LR | Steps | Result |
|---|---|---|---|---|---|
| 1 | 256 logit | 1.0/2.0/0.5 | 1e-4 | 1000 | mean 0.836, min 0.649 — unstable |
| 2 | 256 logit | 1.0/2.0/0.5 | 2e-4 | 1000 | mean 0.706, min 0.304 — worse without a schedule |
| 3 | 256 logit | 0.5/4.0/2.0 | 3-group + cosine | 1000 | mean 0.8995, min 0.696, **pairs 10/10**, emission 0/20 |
| 4 | 256 logit | 1.5/4.0/2.0 | 3-group + cosine | 2000 | mean 0.9237, min 0.8667, **pairs 10/10**, **emission 20/20** |
| **5** | **512 original** | 1.5/4.0/3.0 | 3-group + cosine | 2000 | **min 0.9366, mean 0.9807, pairs 10/10, emission 20/20** |

Final run: **2,000 optimizer steps**, stopped by the step bound (the wall-clock bound was not reached),
no NaN/Inf, no collapse, **Stage 2 peak VRAM 5.99 GiB allocated / 8.76 GiB reserved**.

Three engineering findings drove the revisions:

1. **The model was never taught to stop.** Without EOS appended to the training target, free generation
   ran to the token cap and emitted `[SEG]` 4–13 times. Appending EOS took emission from 0/20 to
   **20/20** with generated sequences of 17–47 tokens and exactly one `[SEG]`.
2. **Separate learning rates + a cosine schedule** removed the late-training oscillation that ruined
   runs 1 and 2 (the projection MLP is randomly initialised and the SAM2 decoder is being repurposed, so
   both need a higher rate than the pretrained LoRA adapters).
3. **Supervision resolution.** SAM2's logits are 256×256, so a 148 px target occupies ~37 px there,
   capping the achievable 512×512 IoU. Task 6A §11.4 permits post-processing the logits to the original
   size instead of resizing the ground truth; run 5 supervises at 512×512.

**Recipe adjustments were made only after the initial results were recorded**, as §12 permits, and every
prior run remains in the record.

---

## 11. Same-Image Pair Test

**10 / 10 pairs** (requirement ≥ 9). Each prediction is compared against **both** ground truths on its
image:

| Image | A own / other | B own / other | Pass |
|---|---|---|---|
| `1_0` | 0.929 / 0.000 | 0.901 / 0.000 | ✅ |
| `1_1` | 0.911 / 0.000 | 0.949 / 0.000 | ✅ |
| `1_10026` | 0.875 / 0.000 | 0.953 / 0.000 | ✅ |
| … 7 more | own ≈ 0.94–0.98 | **other = 0.000** | ✅ |

**A correction recorded rather than hidden.** The first implementation of this criterion compared the two
predictions' own-target IoUs against each other, which can never satisfy a "both directions" requirement
because the two conditions are mutually exclusive — it reported **0/10** for runs 1, 2 and 3. The
corrected implementation (each prediction against both ground truths, exactly as §16.4 states) reports
10/10 with a very wide margin. The wrong 0/10 figure appears in two run logs and is explained in
`docs/task6a_mvp_smoke.md` §5.4.

---

## 12. Free-Generation Test

**20 / 20** emit `[SEG]` exactly once. The path uses **no** ground-truth reasoning, target id, geometry
or reference: generate from image + instruction → locate the generated `[SEG]` → re-forward the complete
generated sequence → read that token's hidden state → projection → SAM2 decoder → mask.

Standalone demo (`scripts/task6a_infer.py`) on the recorded smoke pair from the best checkpoint:

| Sample | query | `[SEG]` count | IoU |
|---|---|---|---|
| `buildsr_train_1_0_1_bottommost_979bd303d0ec` | `bottommost` | 1 | 0.9639 |
| `buildsr_train_1_0_1_leftmost_274a8750ef95` | `leftmost` | 1 | 0.9755 |

**Honest negative result:** the same script on four **unseen test** records emits `[SEG]` **0/4** times.
That is expected for a model overfitted on 20 training records, and it is reported because it is the
correct evidence that Stage 2 is an overfit sanity test and **not** a generalisation result.

---

## 13. VRAM / Runtime

| Point | allocated | reserved |
|---|---|---|
| Both models loaded, idle | 4.53 GiB | 5.10 GiB |
| Stage 1 peak (forward + backward) | **7.25 GiB** | 8.45 GiB |
| Stage 2 peak (training + eval + generation) | **5.99 GiB** | 8.76 GiB |
| Evaluation-only re-run peak | 5.69 GiB | 6.38 GiB |
| Total VRAM | 15.89 GiB | |

Runtimes: environment + downloads ≈ 20 min; Stage 0 ≈ 20 s; Stage 1 ≈ 35 s; Stage 2 run 5 ≈ 80 min for
2,000 steps plus final evaluation.

The Task 5.5 estimate for the 2B stack (6.6–7.1 GiB) is close to the measured Stage 1 peak of 7.25 GiB
and is superseded by it. The 4B estimate remains **unmeasured**.

---

## 14. Checkpoint Manifest

`evaluation/task6a_checkpoint_manifest.json` — paths, sizes, SHA-256 and base model ids/revisions. Only
the reproducible state is saved: LoRA + trainable `[SEG]` row, projection MLP, SAM2 mask decoder,
optimizer state, config, step, metrics and RNG state. **No base Qwen or SAM2 weights are duplicated**, and
no checkpoint bytes are committed (all under `artifacts/`, gitignored).

---

## 15. Tests

```
python -m pytest tests/ -q
```

| Metric | Value |
|---|---|
| collected | 156 |
| passed | **156** |
| failed | 0 |
| skipped | 0 |
| exit code | **0** |
| wall time | 266 s |

Twelve new Task 6A checks (four files) cover the required list; the pre-existing 144 checks still pass.
The model-level Task 6A tests skip cleanly when the assets are absent or `TASK6A_SKIP_MODEL_TESTS=1`, so
`pytest` never forces a download.

Three failures encountered while building the suite were **my own test bugs**, not product bugs, and are
worth recording: the embedding-table assertion ignored Qwen's padding and the shrink on resize; two
static-scan assertions flagged their own explanatory docstrings; and one asserted a separate output-row
delta that PEFT in fact shares with the input adapter.

---

## 16. Git Status / Push

Commit message: `feat: prove 2B SEG MVP pipeline`.
Pre-staging checks: environment directory, HF cache, model weights, SAM2 source and checkpoints all
confirmed gitignored; no `.safetensors`/`.pt`/`.pth`/`.bin` staged; no BuildSpatialReason JSONL changed;
`scripts/check_artifact_consistency.py` re-run before committing. The commit hash and push result are in
the final DSH chat summary.

---

## 17. Known Problems

1. **The assistant reasoning text is not learned.** Assistant token accuracy ≈ 0 because the smoke recipe
   is mask-dominant. The model emits `[SEG]` reliably and the mask is correct, but the reasoning trace
   itself is not usable. **This is the single most important limitation of Task 6A** and it is a
   deliberate consequence of an overfit/segmentation-first recipe, not a claim about the final model.
2. **No generalisation.** 0/4 `[SEG]` emission on unseen test records. Expected from a 20-sample overfit.
3. **Small targets cap out.** Under the strict nearest-interpolation view, sub-300 px targets reach only
   ~0.87 IoU, because SAM2's decoder output is 256×256 and a ~6×6-pixel object loses ~30 % IoU to a
   1-pixel boundary error. The primary metric uses SAM2's own bilinear post-processing; both are reported.
4. **A local accelerator mediates the network.** Weight acquisition could break if Steam++ is disabled;
   the ModelScope route is the documented alternative for Qwen models. Stage 0 confirms acquisition
   rather than assuming it.
5. **`SAM2_BUILD_CUDA=0`** means mask hole/sprinkle post-processing is unavailable. Not exercised in
   Task 6A and documented in `evaluation/task6a_environment.json`.
6. **Qwen3-VL-4B was never loaded**, so ADR-012's 4B estimate remains an estimate.

---

## 18. Recommendation for Task 6B

Task 6B can proceed from this stack. Recommended order of work, in priority order:

1. **Make the language trace real.** Raise the language weight and/or train on more than 20 samples so
   the model actually produces `reasoning_zh`; the current recipe proves the mask pathway only. Without
   this, an MLLM narrative is not supportable.
2. **Scale the data.** Move from the 20-sample overfit to a real mini-train (200–500 samples, then the
   full 15,592) using the same code path; the pipeline, checkpointing, resume, metrics and inference
   script are already in place.
3. **Then add `[REF]`** — the second special token — and only afterwards the Spatial Relation Encoder and
   the Spatial Consistency Loss. The interface is reserved: the projection MLP and any future relation
   module are separate, and all inference-time geometry comes from the model's own predicted masks.
4. **Keep the deterministic-subset habit.** Every subset used for a claim should be recorded before the
   run and never revised using results — the Task 6A subset file is the template.

**Task 6A did not start Task 6B**: no `[REF]`, no Spatial Relation Encoder, no Spatial Consistency Loss,
no full-dataset training, no 4B download.
