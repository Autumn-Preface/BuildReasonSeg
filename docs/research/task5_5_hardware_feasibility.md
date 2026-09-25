# Task 5.5 — Hardware Feasibility

**Research/access date:** 2026-09-26
**Target machine:** Windows 11 64-bit · Intel Core Ultra 9 275HX · 32 GB DDR5-5600 · NVIDIA RTX 5080 **Laptop**, 16 GB VRAM · 1 TB NVMe.

> **Everything in this document is an ESTIMATE unless it is explicitly marked MEASURED or VERIFIED.**
> No model was run, no weights were downloaded, and no training was performed in Task 5.5. The
> arithmetic below is analytic, with every assumption stated in §2. Per the task specification,
> estimates must not be presented as measured results.

---

## 1. MEASURED facts (first-hand, from the actual target machine)

These were read directly from the local environment during Task 5.5 (read-only; nothing installed).

| Property | Value | How obtained |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU | `torch.cuda.get_device_name(0)` |
| Compute capability | **(12, 0)** — i.e. `sm_120`, Blackwell | `torch.cuda.get_device_capability(0)` |
| Total VRAM | **15.89 GiB** | `torch.cuda.get_device_properties(0).total_memory` |
| PyTorch | **2.13.0+cu132**, `cuda.is_available() == True` | `torch.__version__` |
| CUDA runtime in the wheel | **13.2** | `torch.version.cuda` |
| Compiled arch list | `sm_75, sm_80, sm_86, sm_90, sm_100, **sm_120**` | `torch.cuda.get_arch_list()` |
| SDPA available | yes | `torch.nn.functional.scaled_dot_product_attention` |
| Python | 3.10.20 | `sys.version` |
| Already installed | torch, torchvision 0.28.0+cu132, numpy 2.2.6, scipy 1.15.3, opencv 5.0.0, Pillow 12.3.0, PyYAML 6.0.3, **`segment_anything` (SAM 1)** | import probe |
| Missing | `transformers`, `peft`, `bitsandbytes`, `accelerate`, `safetensors`, `tokenizers`, `qwen_vl_utils`, `sam2`, `flash_attn`, `triton`, `xformers`, `deepspeed` | import probe |
| Free disk | 1 TB NVMe, 250–300 GB can be freed (user-stated) | user-stated |

**The single most important measured fact:** the already-installed PyTorch build has `sm_120` in its
arch list and successfully initialises the RTX 5080 Laptop GPU. The Blackwell/sm_120 compatibility
question — historically the main risk for 50-series laptops — is therefore **closed on this machine,
on native Windows, with no installation required.**

Note also that a dedicated environment will be needed for Task 6, because the missing packages above
must not be installed into `yolo_sam_env` (that environment holds the frozen YOLO baseline line).

---

## 2. VERIFIED model dimensions used by the arithmetic

Read first-hand from official model configuration files (`config.json` / model API), not assumed.

| Model | Text hidden | Layers | FFN | Heads / KV heads | Vision | Weight size (BF16) |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | 2048 | 28 | 6144 | 16 / 8, head_dim 128 | depth 24, hidden 1024, patch 16, merge 2 | **3.96 GiB** |
| Qwen3-VL-4B-Instruct | **2560** | **36** | **9728** | 32 / 8, head_dim 128 | depth 24, hidden 1024, patch 16, merge 2 | **8.27 GiB** |
| Qwen3-VL-4B-Thinking | same as 4B-Instruct | | | | | 8.27 GiB |
| Qwen2.5-VL-7B-Instruct | 3584 | 28 | 18944 | 28 / 4, head_dim 128 | depth 32, hidden 1280, patch 14, merge 2 | **15.45 GiB** |
| SAM 2.1 hiera-large | — | — | — | — | image encoder + memory attention | **1.67 GiB on disk** (≈0.83 GiB per copy) |
| SAM 2.1 hiera-base-plus | — | — | — | — | | 0.60 GiB |
| SAM ViT-H (SAM 1) | — | — | — | — | | 4.78 GiB |
| Sa2VA-Qwen3-VL-4B (whole release) | — | — | — | — | | **18.84 GiB** |
| Sa2VA-Qwen3-VL-2B (whole release) | — | — | — | — | | 9.93 GiB |

Context length: Qwen3-VL family 262,144 (config), Qwen2.5-VL-7B 128,000 (config).
`tie_word_embeddings = True` for Qwen3-VL 2B/4B; `False` for Qwen2.5-VL-7B.

### 2.1 This project's images are small — and that dominates the budget

BuildSpatialReason-v0.1.1 tiles are **512 × 512**. With Qwen3-VL's `patch_size = 16` and
`merge_size = 2`, one visual token covers 32 × 32 px, so a native-resolution tile is
**256 visual tokens**. Qwen2.5-VL (patch 14, merge 2 → 28 × 28 px per token) gives **≈324 tokens**.

For comparison, Qwen2.5-VL's released default budget is `max_pixels = 12,845,056` (≈12,800 tokens) —
**~50× larger than this project needs**. Setting the image budget to the native tile is not a
compromise; it is the faithful setting for this dataset, and it removes the largest single source of
activation memory in typical VLM fine-tuning.

Caveat: Qwen3-VL's released `preprocessor_config.json` contains **no** `min_pixels` / `max_pixels`
keys (only `patch_size`, `merge_size`, `temporal_patch_size`). The Qwen3-VL image budget therefore has
to be set explicitly in Task 6 Stage 0 and its behaviour confirmed, rather than assumed from
Qwen2.5-VL's documented defaults. Qwen2.5-VL's own defaults **are** in its config
(`min_pixels = 3136`, `max_pixels = 12,845,056`).

---

## 3. Memory accounting model

Five terms, kept separate as the task requires:

| Term | Formula | Notes |
|---|---|---|
| **Parameter memory** | `params × bytes_per_param` | BF16 = 2 B; NF4 4-bit ≈ 0.55 B effective incl. quantisation metadata; INT8 ≈ 1.05 B |
| **Optimizer memory** | adapters are the only trainable parameters → `adapter_params × 8 B` (AdamW fp32, 2 moments) | full fine-tuning is out of scope on this GPU |
| **Gradient memory** | `adapter_params × 4 B` (fp32 grads) | base gradients are never materialised with a frozen base |
| **Activation memory** | dominated by the per-layer MLP intermediate `batch × seq × ffn × 2 B`, plus block inputs and QKV | SDPA avoids materialising the `seq²` attention matrix; **no** `seq²` term is counted for that reason |
| **CUDA/runtime overhead** | ~0.8–1.0 GiB | context, cuDNN/cuBLAS workspaces, allocator fragmentation |

**Adapter size (computed, not guessed).** With rank 16 on `q,k,v,o,gate,up,down`:
- Qwen3-VL-4B: 54,272 × 16 × 36 layers ≈ **31.3 M** adapter parameters
  → optimizer ≈ 250 MB, gradients ≈ 125 MB, adapter weights ≈ 63 MB ⇒ **≈ 0.44 GiB total**
- Qwen3-VL-2B: 38,912 × 16 × 28 ≈ **17.4 M** → ≈ **0.25 GiB total**

**Activation estimate.** Qwen3-VL-4B, batch 1, `seq = 768` (256 visual tokens + instruction + reasoning
trace + special tokens, padded), BF16, SDPA, no gradient checkpointing:
`768 × 9728 × 2 B = 14.9 MB` per layer for the MLP term alone → 538 MB across 36 layers; adding block
inputs, attention projections and the 24-layer vision tower gives a realistic band of
**1.5–2.5 GiB**. With gradient checkpointing this falls substantially (the MLP term becomes
`seq × hidden` per layer rather than `seq × ffn`), at the cost of a recompute pass.

---

## 4. Configuration estimates

All rows: batch size 1 with gradient accumulation for the effective batch, AdamW, LoRA rank 16,
vision tower frozen, SAM2 frozen, images at 512 × 512 native, `seq ≈ 768`.

| Configuration | Weights | Adapter + optimizer + grads | Activations | SAM2 (frozen) | Runtime | **Total (est.)** | Class |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B, BF16 LoRA | 3.96 | 0.25 | 1.0–1.5 | 0.45 | 0.9 | **≈ 6.6–7.1 GiB** | **SAFE_LOCAL** |
| Qwen3-VL-4B, BF16 LoRA | 8.27 | 0.44 | 1.5–2.5 | 0.45 | 0.9 | **≈ 11.6–12.6 GiB** | **SAFE_LOCAL** (with ~3 GiB headroom) |
| Qwen3-VL-4B, 4-bit QLoRA | 2.6 | 0.44 | 1.5–2.5 | 0.45 | 0.9 | **≈ 5.9–6.9 GiB** | **SAFE_LOCAL** — *only if* bitsandbytes works (see §6) |
| Qwen2.5-VL-7B, BF16 LoRA | 15.45 | — | — | — | — | **> 15.89 GiB ⇒ does not fit** | **NOT_RECOMMENDED_LOCAL** |
| Qwen2.5-VL-7B, 4-bit QLoRA | 4.9 | 0.5 | 2.0–3.0 | 0.45 | 0.9 | **≈ 8.8–9.8 GiB** | **SAFE_LOCAL** — licence is clean, but see §6 |
| Sa2VA-Qwen3-VL-4B, BF16 | 18.84 | — | — | (bundled) | — | **> VRAM ⇒ does not fit** | **NOT_RECOMMENDED_LOCAL** |
| Sa2VA-Qwen3-VL-2B, BF16 | 9.93 | 0.3 | 1.0–1.5 | (bundled) | 0.9 | **≈ 12.1–12.6 GiB** | **BORDERLINE_LOCAL** |

**Inference only (no optimizer, no gradients):** subtract the adapter/optimizer/gradient term and
roughly half the activation term. Qwen3-VL-4B + SAM2 inference lands at **≈ 9–10 GiB** — comfortably
local, which matters because evaluation and demo generation are inference workloads.

### 4.1 Per-effect sensitivity

| Effect | Impact (estimate) | Comment |
|---|---|---|
| **Image resolution** | 512→1024 px multiplies visual tokens by 4 (256 → ~1,024) | the single largest lever; keep native resolution for the MVP |
| **Gradient checkpointing** | saves roughly 0.5–1.5 GiB at ≈20–30 % extra time | the standard OOM-recovery lever, preferred over reducing batch |
| **Batch size** | activations scale linearly; weights do not | batch 1 + accumulation is the right default here |
| **Gradient accumulation** | no VRAM cost | use it to reach an effective batch of 8–16 |
| **Vision tower trainable** | + gradients/optimizer for ~0.4 B params → **+3–4 GiB** | keep it frozen; the task does not require visual adaptation |
| **SAM2 trainable** | SAM2.1 hiera-large decoder/encoder training adds optimizer state and mask-decode activations at high resolution | keep SAM2 frozen for the MVP (also the published LISA/Sa2VA recipes freeze it) |

### 4.2 Classification summary

- **SAFE_LOCAL** — Qwen3-VL-2B (BF16 LoRA), Qwen3-VL-4B (BF16 LoRA), Qwen3-VL-4B (4-bit QLoRA),
  Qwen2.5-VL-7B (4-bit QLoRA)
- **BORDERLINE_LOCAL** — Sa2VA-Qwen3-VL-2B (BF16), Qwen3-VL-4B BF16 LoRA with a larger image budget
- **NOT_RECOMMENDED_LOCAL** — Qwen2.5-VL-7B (BF16), Sa2VA-Qwen3-VL-4B (BF16)

---

## 5. Dataset storage (VERIFIED sizes where marked)

| Item | Size | Status |
|---|---|---|
| BuildSpatialReason-v0.1.1 JSONL | 40.0 MB (train 24.7 + val 6.1 + test 9.1) | VERIFIED (local files) |
| Source component maps + WHU tiles | already on disk; not copied into the repo | VERIFIED |
| EarthReason (HF `earth-insights/EarthReason`) | **1.32 GiB** | VERIFIED (HF API file listing) |
| LaSeRS (HF `earth-insights/LaSeRS`) | **11.65 GiB** | VERIFIED |
| DRSeg (HF `WhynotHug/DRSeg`) | **2.47 GiB** | VERIFIED |
| VRSBench (`xiang709/VRSBench`) | **11.61 GiB** | VERIFIED |
| RefSegRS (`JessicaYuan/RefSegRS`) | **2.77 GiB** | VERIFIED |
| BRIGHT (`xlangai/BRIGHT`) | **0.56 GiB** | VERIFIED |
| Qwen3-VL-4B weights | 8.27 GiB | VERIFIED |
| SAM 2.1 hiera-large | 1.67 GiB | VERIFIED |

Total for the recommended MVP (model + SAM2 + primary dataset + one optional warm-up corpus) is
**well under 30 GB**, comfortably inside the stated < 200 GB preference and the 250–300 GB that can be
freed. Storage is not a constraint for this project.

---

## 6. Environment decision — native Windows 11 vs WSL2 Ubuntu

### 6.1 VERIFIED on reachable first-party sources

| Component | Native Windows 11 | WSL2 Ubuntu | Evidence strength |
|---|---|---|---|
| PyTorch + Blackwell (sm_120) | **MEASURED WORKING** on this machine (torch 2.13.0+cu132, `sm_120` in arch list) | documented supported | MEASURED (Windows), documented (Linux) |
| FlashAttention 2 | **no official Windows support, no prebuilt wheels** (`flash-attn` ships sdist only; classifier "Operating System :: Unix"; its own description says Windows compilation "still requires more testing"); FA2's documented GPU list stops at Hopper, i.e. **no sm_120** | Linux is the only officially supported OS | VERIFIED via `pypi.org` (reachable) |
| bitsandbytes (QLoRA 4-bit / LLM.int8) | **officially supported** — README (0.50.2, 2026-08-27) has a "Windows 11 / Server 2022+" row with QLoRA 4-bit ✅ and `win_amd64` wheels; first Windows wheel 0.43.0 (2024-03-08) | supported | VERIFIED via `pypi.org` |
| DeepSpeed | partly supported; AIO and GDS not supported on Windows | fully supported | VERIFIED (PyPI-published README) |
| SAM 2 CUDA extension | compiles an optional `sam2._C` kernel via nvcc; **optional** — build failure can be ignored, `SAM2_BUILD_CUDA=0` skips it, losing only mask hole/sprinkle post-processing; documented Windows blocker is an MSVC/nvcc version conflict | SAM 2's own guidance for Windows is to use WSL | **UNVERIFIED via primary host** (SAM2 repo not fetchable by the research tool); treat as advisory |
| WSL2 GPU passthrough | — | **current, not deprecated**; Microsoft Learn documents CUDA on WSL2 (page last updated 2025-09-18); documented training caveat is limited pinned memory, and data should live in the WSL filesystem | VERIFIED (`learn.microsoft.com`) |
| Qwen3-VL runtime | needs `transformers` ≥ 4.57.0 (build-from-source at card time); **FlashAttention optional** (recommended only) | same | VERIFIED via ModelScope card text |
| `decord` (video) | not installable on non-Linux; use `torchvision` | installable | VERIFIED |

### 6.2 Judgement (explicitly not a measured result)

The two components that decide this question point in **opposite** directions:

- **FlashAttention has no Windows story at all** — no official support, no wheels, and no `sm_120` in
  its documented GPU list. That argues for WSL2/Linux.
- **FlashAttention is optional for this stack.** Qwen3-VL's own card recommends it but does not require
  it, and SDPA is available and measured working locally. The sequences here are ~768 tokens with a
  256-token image prefix — exactly the regime where FlashAttention's advantage is smallest. So the
  FlashAttention argument is **real but weak for this workload**.
- **bitsandbytes works on Windows**, so the QLoRA fallback that usually motivates WSL is available
  natively.
- **Everything measured on this machine already works natively on Windows.**

**Recommendation: start Task 6 on native Windows 11**, with a **dedicated conda environment** (never
`yolo_sam_env`), and treat WSL2 as the documented fallback if a specific dependency blocks
(SAM 2's compiled extension, or a decision to enable FlashAttention). Rationale: the least disruptive
technically reliable environment is the one whose critical path is already measured working; WSL2 adds
a second filesystem, a second Python stack and a GPU-passthrough layer for a benefit (FlashAttention)
that this workload does not need.

Note that a **local accelerator is installed and active** on this machine (Steam++ / Watt Toolkit,
`# Steam++ Start` block in the Windows `hosts` file, 210 entries, plus a listener on `0.0.0.0:443`).
This affects only the DSH research tools' view of the network, not Task 6: ordinary HTTP clients
(Python, git) reach GitHub and Hugging Face through it successfully — verified first-hand. It is
recorded here because it explains why the Task 5.5 research tool reported some hosts unreachable, and
because **weight acquisition for Task 6 should be checked in Stage 0** rather than assumed.

---

## 7. Residual unknowns (do not treat as resolved)

1. Whether **FlashAttention 4** (Blackwell/Hopper) exists as a release, and whether it changes the
   Windows picture. Only the blackholed main branch mentioned Blackwell.
2. Whether `bitsandbytes` ships an `sm_120`-capable Windows wheel — the README's Windows row stops at
   SM75+ recommended; **the sm_120 combination is unverified**.
3. Qwen3-VL's **actual image-budget defaults**, which are absent from the released preprocessor config.
4. SAM 2's Windows build status from a first-party source.
5. Real peak VRAM under training. Every number in §4 is arithmetic, not a measurement; Stage 0–2 of
   Task 6 exist precisely to replace these estimates with measurements.
6. Whether the installed accelerator remains active; if it is disabled, Hugging Face weight downloads
   would need the ModelScope route instead. **This is a Task 6 Stage 0 check.**
