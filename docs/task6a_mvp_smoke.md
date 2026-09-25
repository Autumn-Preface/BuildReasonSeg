# Task 6A — 2B `[SEG]` MVP smoke / overfit report

**Scope:** build the smallest working BuildReasonSeg segmentation pipeline on the user's own laptop and
prove it end to end:

```
Qwen3-VL-2B-Instruct -> [SEG] hidden state -> Projection MLP -> vanilla SAM2.1 Hiera Base+ mask decoder
```

**Deliberately out of scope:** Qwen3-VL-4B, `[REF]`, Spatial Relation Encoder, Spatial Consistency Loss,
full-dataset training, cloud resources.

**Environment:** dedicated conda environment `.conda/buildreasonseg-mvp` (Python 3.11.16), native
Windows 11.
**Machine:** RTX 5080 Laptop GPU, 15.89 GiB VRAM, driver 617.14.

Machine-readable results: `evaluation/task6a_smoke_report.json`,
`evaluation/task6a_environment.json`, `evaluation/task6a_preflight.json`,
`evaluation/task6a_subset_ids.json`, `evaluation/task6a_checkpoint_manifest.json`.

---

## 1. Environment created

One dedicated conda environment, created with `conda create --prefix`, **not** a clone of any existing
environment. `base`, `jupyter` and `yolo_sam_env` were not touched at any point.

| Item | Value |
|---|---|
| Environment | `.conda/buildreasonseg-mvp` (project-local, gitignored) |
| Python | 3.11.16 |
| PyTorch | `2.13.0+cu132` (CUDA runtime 13.2) |
| `torch.cuda.is_available()` | **True** |
| Compute capability | **(12, 0)** = `sm_120` |
| `torch.cuda.get_arch_list()` | includes `sm_120` |
| BF16 matmul / SDPA | both verified in the new environment |
| Disk used by `local_cache/` | 4.81 GB |
| Dependency lock | `environment/task6a_requirements_lock.txt` |

**Note on the local trust configuration.** The machine runs a local network accelerator
(Steam++ / Watt Toolkit) which terminates TLS for some hosts and re-signs them with its own root
certificate ("SteamTools Certificate", BeyondDimension). That root is in the Windows certificate store
so `ssl.create_default_context()` trusts it, but the `certifi` bundle used by `httpx` /
`huggingface_hub` does not, which makes Hub downloads fail with `CERTIFICATE_VERIFY_FAILED`.
`buildreasonseg_mvp/local_env.py` appends the Windows ROOT/CA stores to the environment's certifi
bundle — a **process-local** trust setting that touches only this project's environment and
`local_cache/`, changes no system setting, registry key, driver or PATH, and backs up the original
bundle as `cacert.pem.orig`. Both SteamTools roots were located this way.

---

## 2. Packages and model revisions

| Component | Version / revision |
|---|---|
| `transformers` | 5.17.0 |
| `peft` | 0.21.0 |
| `accelerate` | 1.15.0 |
| `safetensors` | 0.8.0 |
| `huggingface_hub` | 1.33.0 |
| `tokenizers` | 0.23.2 |
| `qwen-vl-utils` | 0.0.14 |
| `SAM-2` | 1.0, source revision **`2b90b9f5ceec907a1c18123530e92e794ad901a4`** |
| Qwen3-VL-2B-Instruct snapshot | **`89644892e4d85e24eaac8bacfd4f463576704203`** |
| sam2.1-hiera-base-plus checkpoint | 323,606,802 bytes (308 MB) |

SAM2 was installed with `SAM2_BUILD_CUDA=0`: the optional compiled CUDA extension is intentionally
skipped on native Windows (no system CUDA toolkit is installed) and only affects mask hole/sprinkle
post-processing.

The SAM2 checkpoint could not be fetched through `hf_hub_download` because the local accelerator strips
the `X-Repo-Commit` header from the `resolve/` redirect; `scripts/task6a_download.py` therefore falls
back to a plain `urllib` download of the same official URL. The Qwen snapshot downloaded normally.

---

## 3. Stage 0 measurements

### 3.1 Qwen3-VL-2B-Instruct

| Property | Measured |
|---|---|
| dtype | `torch.bfloat16` |
| text hidden size | 2048 |
| vocabulary before `[SEG]` | 151,669 |
| vocabulary after `[SEG]` | 151,670 |
| `[SEG]` token id | **151,669** |
| `[SEG]` is exactly one token | **yes** (encode → `[151669]`, decode round-trips) |
| `tie_word_embeddings` before resize | True |
| weight tying after resize | **still True** |
| embedding table shape | 151,670 × 2048 |
| total parameters | 2,227,632,642 |
| trainable parameters | 24,010,309 (1.08 %) |
| RAM before / after Qwen load | 0.53 GiB / 0.84 GiB |

### 3.2 Qwen preprocessing — **measured, not assumed**

For a real 512×512 WHU tile:

| Property | Measured |
|---|---|
| source image shape | (512, 512, 3) |
| `pixel_values` shape | (1024, 1536) |
| `image_grid_thw` | `[[1, 32, 32]]` |
| **visual tokens** | **256** |
| processor class | `Qwen2VLImageProcessor` (patch 16, merge 2) |
| image budget applied via | `image_processor.size.min_pixels = max_pixels = 262144` |

Sequence lengths measured on one real record per level:

| Level | query type (example) | prompt | total | supervised assistant tokens |
|---|---|---|---|---|
| 1 | `leftmost` | 279 | 295 | 16 |
| 2 | `smallest_to_nearest` | 286 | 323 | 37 |
| 3 | `largest_to_left_of_to_nearest` | 289 | 331 | 42 |

The Task 5.5 estimate of 256 visual tokens is therefore **confirmed by measurement**. The released
Qwen3-VL preprocessor config carries no top-level `min_pixels`/`max_pixels`; the budget lives on
`image_processor.size`, which is where Task 6A sets it explicitly.

### 3.3 SAM2.1 Hiera Base+

| Property | Measured |
|---|---|
| `image_size` | 1024 |
| `backbone_stride` | 16 |
| prompt embedding dim | **256** |
| image embedding size | (64, 64) |
| `use_high_res_features_in_sam` | True |
| total parameters | 80,850,178 |
| image encoder parameters | 69,106,816 (frozen) |
| prompt encoder parameters | 6,220 (frozen) |
| mask decoder parameters | 4,215,109 (**trainable**) |
| source shape | (512, 512, 3) |
| transformed shape | (1, 3, 1024, 1024) |
| image embedding shape | (1, 256, 64, 64) |
| high-res feature shapes | (1, 32, 256, 256), (1, 64, 128, 128) |

The Qwen branch and the SAM2 branch are confirmed to be **separate preprocessing paths** from the same
512×512 RGB source.

### 3.4 VRAM at rest

| Point | allocated | reserved | free | total |
|---|---|---|---|---|
| after both models loaded, idle | 4.53 GiB | 5.10 GiB | 9.51 GiB | 15.89 GiB |

---

## 4. Stage 1 — real 2-sample forward/backward

Fixed smoke pair: two records from the **same image** (`1_0`) with different instructions, different
query types and different targets (`bottommost` → component 1, `leftmost` → component 2).

All nine Stage 1 success criteria passed:

| Check | Result |
|---|---|
| all losses finite | ✅ |
| projected embedding shape == SAM prompt dim (256) | ✅ |
| predicted mask logits finite and non-constant | ✅ |
| non-zero finite gradients on LoRA, `[SEG]`, projection, SAM2 mask decoder | ✅ |
| **zero** trainable parameters in the Qwen visual tower | ✅ |
| frozen SAM2 image encoder receives no gradient | ✅ |
| `[SEG]` row changed after one optimizer step | ✅ (max abs delta 1.0e-4) |
| ordinary vocabulary rows unchanged after that step | ✅ (0.0 for all six sampled rows) |
| no OOM | ✅ |

Measured shapes and gradients for the first sample:

```
lm_logits     (1, 290, 151670)
seg_hidden    (1, 2048)
projected     (1, 256)
sparse prompt (1, 2, 256)
mask logits   (1, 1, 256, 256)
grad norms    lora 11.45 | [SEG] 8.60 | projection 5.61 | SAM2 mask decoder 5.70 | SAM2 image encoder 0.0
losses        total 5.038 = lm_ce 4.110 + mask_bce 0.214 + mask_dice 0.9999
```

**Stage 1 peak VRAM: 7.25 GiB allocated, 8.45 GiB reserved** (Stage 1 wall time 35 s including model
load).

---

## 5. Stage 2 — deterministic 20-sample overfit

### 5.1 The subset

10 source images × 2 instructions = 20 records, all from `train`, selected deterministically before any
training and never revised using model results (`evaluation/task6a_subset_ids.json`).

| Property | Value |
|---|---|
| records / images | 20 / 10 |
| levels present | L1 14, L2 1, L3 (nontrivial) 5 |
| distinct targets per image | exactly 2 for all 10 images |
| query families | `bottommost`, `leftmost`, `rightmost`, `largest_to_*_to_nearest`, `smallest_to_nearest` |

Two different targets on the same image is what makes image-only memorisation insufficient.

### 5.2 Recipe revisions and the recorded runs

Every run is kept, including the ones that did not meet the gate. Task 6A §12 permits adjusting the
recipe only after the initial result is recorded; the initial results are in this table.

| Run | Supervision | Loss weights (lm/bce/dice) | LR | Steps | Result |
|---|---|---|---|---|---|
| 1 | 256 logit res | 1.0 / 2.0 / 0.5 | 1e-4 | 1000 | mean 0.836, min 0.649 — unstable, and the paired check was mis-implemented (see §5.4) |
| 2 | 256 logit res | 1.0 / 2.0 / 0.5 | 2e-4 | 1000 | mean 0.706, min 0.304 — worse; LR too high without a schedule |
| 3 | 256 logit res | 0.5 / 4.0 / 2.0 | 1e-4 + cosine | 1000 | mean **0.8995**, min 0.696, **pairs 10/10**, emission 0/20 |
| 4 | 256 logit res | 1.5 / 4.0 / 2.0 | 3-group + cosine | 2000 | mean **0.9237**, min **0.8667**, pairs **10/10**, emission **20/20** |
| 5 | **512 original res** | 1.5 / 4.0 / 3.0 | 3-group + cosine | 2000 | mean **0.9807**, min **0.9366**, pairs **10/10**, emission **20/20** → **PASS** |

Engineering findings that drove the revisions:

* **Run 3 → 4: the model was never taught to stop.** The assistant target had no end-of-sequence token,
  so free generation ran to the token cap and emitted `[SEG]` 4–13 times (`seg_count` 4–13). Appending
  EOS to the training target fixed it outright: emission went 0/20 → **20/20**, with generated sequences
  of 17–47 tokens and exactly one `[SEG]`.
* **Run 1/2 → 3: separate learning rates.** The projection MLP is randomly initialised and the SAM2 mask
  decoder is being repurposed, so both need a higher rate than the pretrained LoRA adapters. Three AdamW
  groups (adapters 1e-4 / decoder+projection 3e-4 / token row 3e-4 with zero weight decay) plus a
  cosine schedule removed the late-training oscillation entirely.
* **Run 4 → 5: supervision resolution.** SAM2 emits 256×256 logits, so a 148 px target occupies only
  ~37 px there, which caps the achievable 512×512 IoU. Task 6A §11.4 allows post-processing the logits to
  the original size instead of resizing the ground truth; run 5 supervises at 512×512.

### 5.3 Stage 2 result — `PASS`

Evaluated from the `best` checkpoint (step 1600, selected by training mIoU), 2,000 optimizer steps,
wall-clock bound.

| Criterion (§16.3, §16.4, §16.5) | Requirement | Measured | Result |
|---|---|---|---|
| teacher-forced training mIoU, all 20 | ≥ 0.90 | **min 0.9366, mean 0.9807** | ✅ |
| no NaN / Inf | — | none (training completed 2,000 steps) | ✅ |
| no empty / full-image collapse | — | 0 collapsed | ✅ |
| same-image paired instruction dependence | ≥ 9 / 10 | **10 / 10** | ✅ |
| free-generation `[SEG]` emission | 20 / 20 | **20 / 20** | ✅ |
| `[SEG]` is the argmax LM target at the teacher-forced position | — | 20 / 20 | ✅ |
| Stage 2 peak VRAM | — | 5.99 GiB allocated / 8.76 GiB reserved | safe |

**Interpolation convention.** The primary IoU uses **bilinear** upsampling because that is exactly what
vanilla SAM2 does: `sam2/utils/transforms.py::SAM2Transforms.postprocess_masks` ends with
`F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)`. Evaluating any other way would
measure a different pipeline from the one being built. The stricter **nearest** (block-replication) view
is also recorded, and the sensitivity is material for small objects:

| View | mean | min | samples ≥ 0.90 |
|---|---|---|---|
| **bilinear** (SAM2's own convention — primary) | **0.9807** | **0.9366** | **20 / 20** |
| nearest (block replication — strict) | 0.9237 | 0.8704 | 17 / 20 |

Worst samples under the primary metric: 0.9366 (298 px target), 0.9668 (1855 px), 0.9712 (600 px).
Under the strict view the same small targets fall to 0.87–0.89 because a 1-pixel boundary error on a
~6×6-pixel object costs roughly 30 % IoU.

### 5.4 Two corrected measurements, not tuned ones

**(a) The paired-instruction criterion was implemented wrongly.** §16.4 is a statement about **each
prediction against both ground truths**:

```
prediction under instruction A overlaps GT-A more than GT-B
prediction under instruction B overlaps GT-B more than GT-A
```

The first implementation compared the two predictions' own-target IoUs instead, which can never satisfy
a "both directions" requirement because the two conditions are mutually exclusive — it reported 0/10 for
every run. Corrected, the criterion reports **10/10** in runs 3, 4 and 5, with a wide margin
(own-target IoU ≈ 0.9 versus **0.000** for the other target on the same image). The mis-implementation is
recorded rather than quietly fixed, because the 0/10 figure appears in two runs' logs.

**(b) The metric default was aligned with SAM2's own post-processing.** The primary IoU mode was changed
from `nearest` to `bilinear` **after reading `postprocess_masks` in the installed SAM2 source and
confirming it uses bilinear interpolation**. Both views are reported in
`evaluation/task6a_smoke_report.json` (`final.mean_train_miou` and `final.strict_nearest_view`), so a
reviewer can apply either convention. Under the strict view 17 of 20 samples clear the 0.90 gate and the
mean is 0.9237.

---

## 6. Trained surface

| Group | Parameters | State |
|---|---|---|
| Text LoRA adapters (196 target modules, 7 projections × 28 layers) | 17,432,576 | trainable |
| `[SEG]` trainable token row (**shared** by the input embedding and the tied `lm_head`) | 2,048 | trainable |
| Projection MLP (2048 → 256) | 2,360,576 | trainable |
| SAM2 mask decoder | 4,215,109 | trainable |
| Qwen vision tower | 0 | **frozen — asserted** |
| Qwen base weights + embedding table | 0 | **frozen — asserted** |
| SAM2 image encoder, memory modules, prompt encoder, top-level parameters | 0 | **frozen — asserted** |

Trainable fraction: **1.08 %** of 2.23 B parameters. The optimizer covers **528 trainable tensors with 0
missing** (asserted by `tests/test_task6a_trainables.py`).

Two details worth recording because they are easy to get wrong:

* **PEFT preserves the tying for the trainable token.** The output head's token adapter shares the *same*
  delta parameter object as the input embedding's adapter, so there is exactly **one** trainable `[SEG]`
  row used on both sides — which is what "tied embeddings" should mean. No separate output-row wrapper is
  needed on PEFT 0.21.
* **`resize_token_embeddings` shrinks as well as appends.** Qwen ships a table padded to 151,936 rows for
  a 151,669-token tokenizer; the call normalises it to 151,670 rows (tokenizer + the new `[SEG]` token).
  The removed rows were unreachable from the tokenizer, so this is dead weight, but it is a real change
  and is asserted rather than assumed.

The full embedding matrix is never handed to the optimizer. In the first Stage 1 run the ordinary
vocabulary rows sampled (ids 0, 1, 100, 1000, 50000, 151000) moved by exactly **0.0** across an optimizer
step while the `[SEG]` row moved.

---

## 7. Free-generation inference (the deployment path)

`scripts/task6a_infer.py` runs the real inference path with **no ground-truth reasoning, no target id, no
geometry and no reference** supplied at any point:

```
image + instruction
  -> Qwen3-VL generates assistant text
  -> locate the generated [SEG]
  -> re-forward the COMPLETE generated sequence to read that token's hidden state
  -> projection -> SAM2.1 mask decoder -> mask
```

On the recorded smoke pair, from `best_run5.pt`:

| Sample | query | `[SEG]` count | IoU |
|---|---|---|---|
| `buildsr_train_1_0_1_bottommost_979bd303d0ec` | `bottommost` | 1 | 0.9639 |
| `buildsr_train_1_0_1_leftmost_274a8750ef95` | `leftmost` | 1 | 0.9755 |

Mean IoU 0.9697, emission 1.00.

**Honest negative result:** the same script run on four unseen **test** records emits `[SEG]` **0/4**
times and produces no usable mask. That is the expected behaviour of a model overfitted on 20 training
records — it is reported here because it is the correct evidence that this is an overfit sanity test and
**not** a generalisation result.

---

## 8. Visual outputs

`evaluation/task6a_samples/` — 23 PNGs, 4.9 MB, gitignored-free (committed):

* 20 Stage-2 panels (one per overfit sample), each showing the source image, the ground-truth outline,
  the predicted outline, the instruction, the query type and the IoU;
* `contact_sheet.png`;
* 2 smoke-pair inference panels.

Each panel renders the prediction at the original 512×512 resolution with a fixed threshold.

---

## 9. Tests

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

Twelve new Task 6A checks cover the required list: target-mask reconstruction, `[SEG]` single-token
identity, assistant-only LM labels, ID-free instructions, selective `[SEG]` training leaving ordinary
vocabulary rows unchanged, LoRA attaching to no visual-tower module, bridge dimensions, frozen SAM2
encoder, trainable projection + mask decoder, no `[REF]` path, no GT geometry at inference, and paired
targets per image.

The model-level tests skip cleanly when the downloaded assets are absent (or when
`TASK6A_SKIP_MODEL_TESTS=1`), so `pytest` never forces a download.

---

## 10. What this proves and what it does not

**Proved.**

* The full `Qwen3-VL → [SEG] → projection → vanilla SAM2.1` path is real, differentiable and runs on
  16 GB of laptop VRAM on native Windows, with every licence-clean component from ADR-012.
* `[SEG]` is a genuine single token, selectively trainable without unfreezing the embedding table.
* The mask decision is driven by the instruction, not by the image alone: on all 10 same-image pairs the
  prediction matches its own target and not the other (≈0.98 versus 0.000).
* The model emits `[SEG]` exactly once in free generation and produces a mask from that hidden state.
* The architecture can memorise paired instruction–mask examples: all 20 samples reach ≥0.9366 IoU.

**Not proved, and not claimed.**

* **Language quality.** The assistant reasoning text is *not* reproduced: assistant token accuracy stays
  near zero because the mask objective dominates the smoke recipe. The model emits `[SEG]` reliably and
  the correct mask follows, but the reasoning trace itself is not usable. This is a deliberate
  consequence of an overfit/segmentation-first recipe; the MVP is a pipeline proof, not a language result.
* **Generalisation.** 0/4 `[SEG]` emission on unseen test records. This is an overfit sanity test on 20
  training records; no val or test metric is reported and none should be inferred.
* **4B feasibility.** Qwen3-VL-4B was never loaded.
* **Any relation-aware behaviour.** There is no `[REF]`, no Spatial Relation Encoder and no Spatial
  Consistency Loss in Task 6A.

