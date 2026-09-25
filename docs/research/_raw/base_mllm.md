# BuildReasonSeg — Base multimodal-LLM verification (read-only web research, ModelScope-first)

**Access date for every source: 2026-09-26.** Target machine: **RTX 5080 Laptop, 16 GB VRAM, 32 GB RAM, Windows 11**. Goal: LoRA SFT of a compact multimodal LLM with an added custom segmentation token (`[SEG]`).

## 0. Method and evidence discipline

| Rule applied | Detail |
|---|---|
| Allowed evidence tiers | (1) **`PRIMARY-VENDOR`** = ModelScope (Alibaba's official first-party model host) JSON API: `License`, `Architectures`, `ModelInfos.safetensor.files[].size/.sha256`, `model_size`, `tensor_type`, `StorageSize`, `ReadMeContent` (the model-card body). (2) **`PRIMARY-PAPER`** = arXiv. (3) `DOCS` = vendor documentation content served from a **non-vendor host** → treated as *secondary*, flagged explicitly. |
| **Disallowed as evidence** | Hugging Face model cards, GitHub READMEs/source, `github.io` documentation, and **all mirrors**. `hf-mirror.com` resolves in this environment but is a **third-party mirror** and is never cited for a licence (or anything else) here. Where a fact exists only on a blackholed host, this report writes **`UNVERIFIED (host unreachable at research time, 2026-09-26)`** and does **not** substitute a mirror or a search snippet. |
| Net effect on this file | **No field in §2–§10 is derived from an HF model card, an HF `config.json`/file list, or a GitHub repository.** Any such field carries the `UNVERIFIED (host unreachable at research time, 2026-09-26)` label. Section §13 lists mirror-derived observations *solely* to direct re-verification and states no claim. |
| Correction notice | An earlier draft of this file (and an earlier digest) listed "HF model cards/configs/file lists" as sources. That was wrong: `huggingface.co` is blackholed to `127.0.0.1` in this environment, so those pages were never fetched first-hand. The file has been rebuilt on the ModelScope official API plus reachable arXiv/vendor pages only. |
| Computed sizes | `GiB = bytes ÷ 1024³`, summed from the **ModelScope file list**; always labelled **computed from file list**, never "officially stated". |
| VRAM arithmetic | No invented numbers. Any footprint reasoning in this report is **my own estimate**, labelled as such; only vendor-stated VRAM sentences are quoted as facts. |

## 1. URL not reachable at research time

**Hosts blackholed to `127.0.0.1` (unfetchable):** `huggingface.co`, `github.com`, `raw.githubusercontent.com`, `github.io` (and `*.github.io` except a few cached ones). `pytorch.org` fails DNS entirely. **Every fact below that lives only on those hosts is `UNVERIFIED (host unreachable at research time, 2026-09-26)`.**

Exact URLs needed but **not fetchable** (recorded so re-verification is mechanical):

| # | Blocked URL needed | Fact it would settle |
|---|---|---|
| B1 | `https://huggingface.co/Qwen/Qwen3-VL-{2B,4B}-Instruct` and `…/Qwen3-VL-4B-Thinking` (model cards) | Qwen3-VL licence *file text*, `min_pixels`/`max_pixels` guidance, LoRA/QLoRA mention, Windows guidance |
| B2 | `…/raw/main/config.json` and `…/preprocessor_config.json` for the three Qwen3-VL repos | image-token budget **defaults**, `max_position_embeddings`, dtype |
| B3 | `https://huggingface.co/api/models/Qwen/…?blobs=true` (file lists) | shard names/sizes as published on HF (ModelScope already supplied equivalent first-party sizes) |
| B4 | `https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/README.md` | the `qwen-research` licence field → **the single licence question that blocks using the 3B** |
| B5 | `…/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE` | licence **text** behind that field |
| B6 | `…/Qwen/Qwen3-VL-4B-Instruct-GGUF`, `…/Qwen/Qwen3-VL-4B-Instruct-FP8`, `…/Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | whether GGUF/FP8/AWQ quantisations are officially published |
| B7 | `…/ByteDance/Sa2VA-Qwen3-VL-4B` (+ `Sa2VA-InternVL3-*`, `Sa2VA-Qwen2_5-VL-*`) | Sa2VA checkpoint existence, sizes, licence, and the `[SEG]`-token recipe |
| B8 | `…/OpenGVLab/InternVL3_5-2B-Instruct`, `…/HuggingFaceTB/SmolVLM2-2.2B-Instruct` | whether those alternatives exist as claimed and under what licence |
| B9 | `https://huggingface.co/docs/transformers/main/en/…` | the documented `add_special_tokens` / `resize_token_embeddings` procedure for adding `[SEG]` |
| B10 | `https://github.com/QwenLM/Qwen3-VL`, `…/QwenLM/Qwen2.5-VL` | code licence, `tune_mm_vision` freezing defaults, LoRA hyper-parameters, memory tables |
| B11 | `https://github.com/Bytedance/Sa2VA`, `…/OpenGVLab/InternVL`, `…/huggingface/smollm` | code licences and training recipes for those families |
| B12 | `https://raw.githubusercontent.com/QwenLM/Qwen2.5-VL/main/README.md` | GitHub-rendered Qwen2.5-VL card (blocked even when `qwenlm.github.io` is cached) |
| B13 | `https://pytorch.org/…` (DNS failure) | first-party PyTorch/CUDA-wheel support statement for Windows and for Blackwell (sm_120) |

**Reachable hosts (used):** `arxiv.org` (abs + HTML full text), `www.modelscope.cn` / `modelscope.cn` (official API), `developer.aliyun.com`, `www.alibabacloud.com`. **Reachable but not used in this pass — available for the follow-ups above:** `qwen.ai`, `qwenlm.github.io`, `seed.bytedance.com`, `docs.nvidia.com`, `developer.nvidia.com`, `openaccess.thecvf.com`, `openreview.net`, `www.semanticscholar.org`, `learn.microsoft.com`, `www.ecva.net`, `isprs-archives.copernicus.org`, `www.mdpi.com`, `ai.meta.com`, `segment-anything.com`. `hf-mirror.com` resolves but is a **third-party mirror** and is not cited.

**Consequence:** for the five seed models, identity / parameter count / dtype / file sizes / context / usage guidance and **4 of 5 licences** are first-party-verified via ModelScope; **Qwen2.5-VL-3B's licence is unresolved**, and the pixel-budget API defaults, QLoRA/bitsandbytes, FP16, Windows support, SAM/SAM2 integration and the concrete `[SEG]`-token recipe are **not** — those are reported as `UNVERIFIED (host unreachable at research time, 2026-09-26)`.

## 2. Seed-list corrections (explicit)

| Seed you gave | Verdict | Exact official ID | Note |
|---|---|---|---|
| `Qwen3-VL-2B-Instruct` | ✅ exists | `Qwen/Qwen3-VL-2B-Instruct` | `Architectures: ["Qwen3VLForConditionalGeneration"]`, `ModelType ["qwen3_vl"]` |
| `Qwen3-VL-4B-Instruct` | ✅ exists | `Qwen/Qwen3-VL-4B-Instruct` | same architecture |
| `Qwen3-VL-4B-Thinking` | ✅ **exists** | `Qwen/Qwen3-VL-4B-Thinking` | identical parameter count and shard sizes to the Instruct 4B; its shipped chat template **forces thinking on** |
| `Qwen2.5-VL-3B-Instruct` | ✅ exists | `Qwen/Qwen2.5-VL-3B-Instruct` | architecture `Qwen2_5_VLForConditionalGeneration`; **ModelScope's `License` field is empty → weight license `UNVERIFIED`** |
| `Qwen2.5-VL-7B-Instruct` | ✅ exists | `Qwen/Qwen2.5-VL-7B-Instruct` | same architecture; `License: apache-2.0` |
| "newer compact 2026 model" | ✅ found, **and it is the strongest candidate** | **`Qwen/Qwen3.5-4B`** (`Architectures ["Qwen3_5ForConditionalGeneration"]`, `ModelType ["qwen3_5"]`) | released 2026 (base_model `Qwen/Qwen3.5-4B-Base`, relation `finetune`); **the ID is `Qwen3.5-4B`, not `Qwen3.5-VL-4B`** — no `Qwen3.5-VL-*` repo is verifiable |

**Corrections to interpretations (not IDs):** (a) **no official model ID `Qwen3.5-VL-4B` exists** — the 2026 compact line is `Qwen3.5-*`; (b) **the Qwen2.5-VL-3B weight licence is not confirmable from a first-party host at all** (ModelScope leaves it blank) — the HF card is reported elsewhere to carry `qwen-research`, but that host was unreachable → treat as a **publication risk until re-checked**; (c) **Qwen3-VL does not document a `min_pixels`/`max_pixels` API on its model card** (see §6) — unlike Qwen2.5-VL, whose card documents it explicitly.

## 3. Identity, parameters, weight format, size, precision — ModelScope (first-party)

| Model | Architecture / ModelType | Base model | Total params (`model_size`) | Weight files (ModelScope file list) | BF16 size **computed from file list** | `tensor_type` | `StorageSize` |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | `Qwen3VLForConditionalGeneration` / `qwen3_vl` | — | **2,127,532,032** | `model.safetensors` 4,255,140,312 B | **3.96 GiB** | `["BF16"]` | 4,266,649,720 |
| `Qwen/Qwen3-VL-4B-Instruct` | `Qwen3VLForConditionalGeneration` / `qwen3_vl` | — | **4,437,815,808** | `model-00001-of-00002` 4,967,229,296 B; `model-00002-of-00002` 3,908,490,048 B | **8.27 GiB** | `["BF16"]` | 8,887,293,488 |
| `Qwen/Qwen3-VL-4B-Thinking` | `Qwen3VLForConditionalGeneration` / `qwen3_vl` | — | **4,437,815,808** | shards 4,967,229,296 + 3,908,490,048 B (identical to the Instruct 4B) | **8.27 GiB** | `["BF16"]` | 8,887,293,488 |
| `Qwen/Qwen2.5-VL-3B-Instruct` | `Qwen2_5_VLForConditionalGeneration` / `qwen2_5_vl` | — | **3,754,622,976** | shards 3,982,649,232 + 3,526,688,744 B | **6.99 GiB** | `["BF16"]` | 7,520,919,692 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `Qwen2_5_VLForConditionalGeneration` / `qwen2_5_vl` | — | **8,292,166,656** | 5 shards: 3,900,233,256 + 3,864,726,320 + 3,864,726,424 + 3,864,733,680 + 1,089,994,880 B | **15.45 GiB** | `["BF16"]` | 16,595,981,359 |
| **`Qwen/Qwen3.5-4B`** (2026) | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | `Qwen/Qwen3.5-4B-Base` (finetune) | **4,659,865,088** | shards 5,329,398,688 + 3,990,429,408 B | **8.68 GiB** | **`["BF16","F32"]`** | `UNVERIFIED` (not read) |
| **`Qwen/Qwen3.5-2B`** (2026) | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | `UNVERIFIED` (not read) | `UNVERIFIED` (exact `model_size` not read) | `UNVERIFIED` (exact file list not read) | **4.24 GiB** (first-party ModelScope probe by the parent agent, 2026-09-26 — "computed from file list" basis not restated here) | `UNVERIFIED` | `UNVERIFIED` |
| `OpenGVLab/InternVL3_5-2B-Instruct` | — | — | `UNVERIFIED (host unreachable at research time, 2026-09-26)` | — | — | — | — |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | — | — | `UNVERIFIED (host unreachable at research time, 2026-09-26)` | — | — | — | — |
| `ByteDance/Sa2VA-Qwen3-VL-4B` | — | — | `UNVERIFIED (host unreachable at research time, 2026-09-26)` — **ModelScope API returns HTTP 404 "record not found" for this repo**, so it is not hosted first-party | — | — | — | — |
| does any "vision vs language parameter" split exist? | — | — | **No**: none of the four Qwen ModelScope payloads exposes a vision/language split (only `model_size` + `Architectures`); the only vendor that publishes such a split publicly is InternVL3.5, whose card is unreachable → `UNVERIFIED` | — | — | — | — |

**Weight format / precision actually verified first-party:** safetensors only, **BF16** (`tensor_type`), plus `F32` tensors in Qwen3.5-4B. **FP8 / AWQ / GPTQ / GGUF publication for these repos is `UNVERIFIED (host unreachable at research time, 2026-09-26)`** — those live in separate HF repos that could not be reached.

## 4. Licences — code vs weights (separate)

| Model | **Weight licence (first-party evidence)** | Code licence |
|---|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | **`apache-2.0`** — ModelScope `License` field = `"apache-2.0"` | `UNVERIFIED (host unreachable at research time, 2026-09-26)` |
| `Qwen/Qwen3-VL-4B-Instruct` | **`apache-2.0`** — ModelScope `License` = `"apache-2.0"` | `UNVERIFIED (host unreachable at research time, 2026-09-26)` |
| `Qwen/Qwen3-VL-4B-Thinking` | **`apache-2.0`** — ModelScope `License` = `"apache-2.0"` | `UNVERIFIED (host unreachable at research time, 2026-09-26)` |
| `Qwen/Qwen2.5-VL-3B-Instruct` | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** — ModelScope returns an **empty `License` field** (and empty `LicenseName`, `LicenseLink`); no reachable source confirms Apache-2.0. The only signal pointing to a **research-only** licence comes from an **unreachable** source (the HF model card) and is therefore not admissible here | `UNVERIFIED (host unreachable at research time, 2026-09-26)` |
| `Qwen/Qwen2.5-VL-7B-Instruct` | **`apache-2.0`** — ModelScope `License` = `"apache-2.0"` | `UNVERIFIED (host unreachable at research time, 2026-09-26)` |
| `Qwen/Qwen3.5-2B` | **`apache-2.0`** — ModelScope `License` = `"apache-2.0"` (first-party probe, 2026-09-26) | `UNVERIFIED (host unreachable at research time, 2026-09-26)` |
| `Qwen/Qwen3.5-4B` | **`apache-2.0`** — ModelScope `License` = `"apache-2.0"` | `UNVERIFIED (host unreachable at research time, 2026-09-26)` |
| `ByteDance/Sa2VA-*` (all four probed IDs) | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** — **404 on ModelScope** ("record not found"); the HF-hosted checkpoint licence is on a blackholed host | `UNVERIFIED (host unreachable at research time, 2026-09-26)` |
| InternVL3.5 / SmolVLM2 | `UNVERIFIED (host unreachable at research time, 2026-09-26)` | `UNVERIFIED (host unreachable at research time, 2026-09-26)` |

**Caveat that must travel with these values:** this is **ModelScope's `License` metadata field**, i.e. a first-party platform declaration — it is **not** a re-read of the repository's `LICENSE` file, whose text could not be fetched. For any publication or redistribution decision, the licence *text* must still be read from a reachable host.

## 5. Runtime requirements, precision, quantization

| Model | Minimum `transformers` (first-party card text) | FlashAttention-2 | Without FA2 | BF16 | FP16 | QLoRA / bitsandbytes 4-bit & 8-bit |
|---|---|---|---|---|---|---|
| Qwen3-VL 2B / 4B-Instruct / 4B-Thinking | Card: *"The code of Qwen3-VL has been in the latest Hugging Face transformers and we advise you to build from source: `pip install git+https://github.com/huggingface/transformers` — `# pip install transformers==4.57.0 # currently, V4.57.0 is not released`"* → **no released minimum version is stated; source build required at card-writing time** | **Optional / recommended**: *"We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios"* (`attn_implementation="flash_attention_2"`, shown commented-out = non-default) | No failure documented; default `torch_dtype="auto"` path is the documented default | ✅ (first-party: `tensor_type BF16`; examples use `dtype=torch.bfloat16`) | `UNVERIFIED (host unreachable at research time, 2026-09-26)` | `UNVERIFIED` — no QLoRA/bitsandbytes statement anywhere in the first-party card text |
| Qwen2.5-VL 3B / 7B | Card: *"the code … has been in the latest HF transformers; build from source `pip install git+https://github.com/huggingface/transformers accelerate`, or you might encounter `KeyError: 'qwen2_5_vl'`"* → **no version number stated** | **Optional / recommended** (same wording, commented-out example) | No failure documented | ✅ (`tensor_type BF16`) | `UNVERIFIED` | `UNVERIFIED` |
| `Qwen/Qwen3.5-4B` | Card: *"The latest `transformers` is required for Qwen3.5: `pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"`"* → **no version number**; vLLM/SGLang **main-branch builds** required | **Not mentioned on the card** → `UNVERIFIED` | `UNVERIFIED` | ✅ (`tensor_type ["BF16","F32"]`, `dtype` BF16) | `UNVERIFIED` | `UNVERIFIED` |
| InternVL3.5 / SmolVLM2 / Sa2VA | `UNVERIFIED (host unreachable at research time, 2026-09-26)` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` |

Only **third-party doc rendering** claims an actual number here (`transformers>=4.57.0` for Qwen3-VL) — see §6 and §13.

## 6. Context length and image-token budget

| Model | Context (first-party card text) | `min_pixels` / `max_pixels` documented? | Defaults |
|---|---|---|---|
| Qwen3-VL 2B / 4B / 4B-Thinking | **"Native 256K context, expandable to 1M; handles books and hours-long video with full recall and second-level indexing"** — identical sentence in all three ModelScope cards; corroborated by arXiv 2511.21631 (*"natively supports interleaved contexts of up to 256K tokens"*). YaRN/1M configuration details: **`UNVERIFIED`** (not in the card text; arXiv abstract does not cover it) | **No.** The first-party card shows `processor.apply_chat_template(...)` and standard `AutoProcessor.from_pretrained(...)` only — **no `min_pixels`, no `max_pixels`, no `size[...]` guidance anywhere** | `UNVERIFIED (host unreachable at research time, 2026-09-26)` — the image-processor defaults live in `preprocessor_config.json` on HF |
| Qwen2.5-VL 3B / 7B | Card: **"The current `config.json` is set for context length up to 32,768 tokens"**; beyond that, YaRN with `mrope_section [16,24,24]`, `factor 4`, `original_max_position_embeddings 32768`, **explicitly "not recommended" because "this method has a significant impact on the performance of temporal and spatial localization tasks"**; for long video `max_position_embeddings` may be raised directly "such as 64k" | **Yes.** Card: *"The default range for the number of visual tokens per image in the model is **4-16384**. You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280"*; also `min_pixels = 256*28*28`, `max_pixels = 1280*28*28`, and exact `resized_height`/`resized_width` "rounded to the nearest multiple of 28" | Token range **4–16384** documented first-party; the numeric `min_pixels`/`max_pixels` defaults are in `preprocessor_config.json` → `UNVERIFIED (host unreachable at research time, 2026-09-26)` |
| `Qwen/Qwen3.5-4B` | Card: **"Context Length: 262,144 natively and extensible up to 1,010,000 tokens"**; serving text: *"default context length of 262,144 tokens … we advise maintaining a context length of at least 128K tokens to preserve thinking capabilities"*; YaRN recipe given (`rope_parameters`: `mrope_section [11,11,10]`, `rope_type yarn`, `rope_theta 10000000`, `partial_rotary_factor 0.25`, `factor 4.0`, `original_max_position_embeddings 262144`, vLLM `--max-model-len 1010000`) | **No** image-budget API documented on the card; only video sampling (`mm_processor_kwargs {"fps": 2, "do_sample_frames": True}`, default `fps=2`) | image-budget defaults `UNVERIFIED` |
| InternVL3.5 / SmolVLM2 / Sa2VA | `UNVERIFIED (host unreachable at research time, 2026-09-26)` | `UNVERIFIED` | `UNVERIFIED` |

**Direct answer to the "same budget control as Qwen2.5-VL?" question: Not verifiable from first-party sources.** Qwen2.5-VL documents `min_pixels`/`max_pixels` explicitly on its official card; **no Qwen3-VL or Qwen3.5 model card mentions them at all**. The claim that Qwen3-VL uses `size['longest_edge']`/`size['shortest_edge']` as equivalent controls appears only in a **third-party documentation rendering** (§13) — treat as `UNVERIFIED` and test empirically on the target machine before designing the data pipeline around it.

## 7. Fine-tuning, the `[SEG]` token, and SAM integration

| Question | Finding |
|---|---|
| Recommended image-resolution handling (Qwen2.5-VL, first-party) | Native resolution by default; optionally cap the visual-token budget with `min_pixels`/`max_pixels` (card example: 256–1280 tokens) or force exact `resized_height`/`resized_width` (multiples of 28) |
| Recommended image-resolution handling (Qwen3-VL / Qwen3.5) | **`UNVERIFIED (host unreachable at research time, 2026-09-26)`** — no first-party resolution guidance in the reachable card text |
| Is the visual tower frozen in the official SFT recipe? | `UNVERIFIED (host unreachable at research time, 2026-09-26)` — the `tune_mm_vision` flags live in the QwenLM/Qwen3-VL repository |
| LoRA / QLoRA recipe (rank, alpha, LR, memory) | `UNVERIFIED (host unreachable at research time, 2026-09-26)`; QLoRA/bitsandbytes is **not mentioned** in any reachable first-party card |
| Can the hidden state of an arbitrary special token be read out (needed for `[SEG]`)? | **Architecturally plausible, and paper-supported in principle** — these are standard `…ForConditionalGeneration` decoders (first-party `Architectures` fields) that expose `output_hidden_states`; the Sa2VA paper states that the LLM itself generates the **instruction tokens** that drive the segmenter; and **Qwen3-VL-4B-Thinking's shipped chat template proves arbitrary special tokens can be placed inside the assistant stream** (it injects `<think>` / `</think>` around generated content). **A concrete `[SEG]`-token readout recipe is `UNVERIFIED (host unreachable at research time, 2026-09-26)`.** |
| Officially documented way to add a special token + resize the embedding table | **`UNVERIFIED (host unreachable at research time, 2026-09-26)`** — no Qwen card or reachable doc documents `add_special_tokens`/`resize_token_embeddings`. Note the practical constraint visible first-party: Qwen3-VL/Qwen3.5 tie the LM head to a **fixed padded vocabulary** (`Qwen3.5-4B` card: *"Token Embedding: 248320 (Padded) … LM Output: 248320 (Tied to token embedding)"*), so any added token must be handled together with the tied embedding. |
| A second custom region token (e.g. `[SEG_REASON]` + `[SEG]`) | Not documented anywhere reachable → `UNVERIFIED`; the tied/padded-vocab constraint above is the one first-party fact that matters. |
| Official/well-documented SAM or SAM2 integration for this family — **what the reachable paper states** | **Stated by the paper** (arXiv:2501.04001, *Sa2VA: Marrying SAM2 with MLLM for Dense Grounded Understanding of Images and Videos*, accepted by IEEE TPAMI 2026): "Sa2VA combines **SAM-2**, a foundation video segmentation model, with MLLM … and unifies text, image, and video into a shared LLM token space. Using the LLM, Sa2VA generates **instruction tokens that guide SAM-2 in producing precise masks**"; supporting referring segmentation and conversation for images **and** video with "minimal single-stage instruction tuning"; "Sa2VA can be easily extended into various MLLMs, including **Qwen-VL and Intern-VL**"; contributes the Ref-SAV dataset (72k+ object expressions). Code/models are said to be released at `github.com/Bytedance/Sa2VA` (**unreachable**) |
| Official/well-documented SAM or SAM2 integration for this family — **what cannot be verified** | `UNVERIFIED` — the reachable abstract does **not** name the concrete segmentation token, its token id, the hidden-state projection, or any Qwen3-VL-specific checkpoint, and all four probed `ByteDance/Sa2VA-*` IDs are **404 on ModelScope**. Checkpoint existence, file sizes and licence: `UNVERIFIED / DO NOT USE UNTIL RESOLVED`. |
| Thinking-edition output format (relevant to ending in a mask token) | **First-party, and it is a real complication.** `Qwen/Qwen3-VL-4B-Thinking`'s shipped chat template splits assistant content into `reasoning_content` and answer, and on `add_generation_prompt` it **appends `<|im_start|>assistant\n<think>\n`** — i.e. a reasoning block is opened for you, so the mask token necessarily lands *after* a reasoning trace. Its card's recommended VL decoding budget is `out_seq_length=40960` (vs 16384 for the 2B/4B Instruct cards). `Qwen/Qwen3.5-4B` behaves the same way but **documents the escape hatch**: thinking is on by default, and `chat_template_kwargs: {"enable_thinking": False}` makes the template emit an **empty** `<think>\n\n</think>\n\n` block so the answer follows immediately. |

## 8. Windows-native support / platform guidance

| Model | First-party statement | WSL2 recommendation |
|---|---|---|
| Qwen3-VL 2B/4B (+Thinking) | **No Windows statement.** The card's only platform-relevant text is the source-build instruction and generic CUDA usage; the documented target stack (FA2 `attn_implementation`, vLLM/SGLang) is Linux-oriented but this is not stated | `UNVERIFIED` |
| Qwen2.5-VL 3B/7B | **Explicit, first-party:** *"If you are not using Linux, you might not be able to install `decord` from PyPI. In that case, you can use `pip install qwen-vl-utils` which will fall back to using torchvision for video processing."* → the only documented Windows-relevant blocker/fix for these models is the video back-end | `UNVERIFIED` |
| `Qwen/Qwen3.5-4B` | **No Windows statement.** Documented serving paths all assume Linux/GPU servers (SGLang main branch, vLLM **nightly** wheels, KTransformers, `transformers serve`) | `UNVERIFIED` |
| InternVL3.5 / SmolVLM2 / Sa2VA | `UNVERIFIED (host unreachable at research time, 2026-09-26)` | `UNVERIFIED` |

**Project-relevant read (my estimate, not a documented claim):** the reachable first-party evidence gives no Windows guarantee for any candidate; the concrete known friction points are the Linux-oriented serving/training stacks and the video back-end. Whether the RTX 5080 Laptop (Blackwell, sm_120) is supported by a prebuilt FlashAttention-2 wheel is **`UNVERIFIED`** — no reachable first-party page states supported GPU architectures.

## 9. VRAM — documented statements only

| Model | Vendor-stated VRAM | Verdict |
|---|---|---|
| Qwen3-VL 2B / 4B / 4B-Thinking | none in the ModelScope card text | **NOT_DOCUMENTED** |
| Qwen2.5-VL 3B / 7B | none | **NOT_DOCUMENTED** |
| `Qwen/Qwen3.5-4B` | none; only *"If you encounter out-of-memory (OOM) errors, consider reducing the context window"* | **NOT_DOCUMENTED** |
| InternVL3.5 / SmolVLM2 / Sa2VA | `UNVERIFIED (host unreachable at research time, 2026-09-26)` | **NOT_DOCUMENTED** |

**My own estimates (arithmetic, NOT vendor statements, to be treated as hypotheses to measure):** BF16 weights are 3.96 GiB (2B-VL), 6.99 GiB (2.5-VL-3B), 8.27 GiB (4B-VL), 8.68 GiB (Qwen3.5-4B) and 15.45 GiB (2.5-VL-7B). With 16 GB total, the 4B/8.68 GiB and 2B/3.96 GiB classes leave plausible room for LoRA adapters + optimizer + activations only with small rank, gradient checkpointing and short sequences; the 7B (15.45 GiB of weights alone) is very likely infeasible for training on this device **as an estimate, not a documented limit**. A `[SEG]` projector/decoder head adds additional (unknown, unmeasured) memory.

## 10. Comparative table

| Model | Params | Weight licence (first-party) | Context | BF16 size (computed) | Suitability for adding a 2nd custom region token | Fit for a 16 GB LoRA fine-tune |
|---|---|---|---|---|---|---|
| **`Qwen/Qwen3.5-4B`** (2026) | 4,659,865,088 | **`apache-2.0`** | **262,144 → 1,010,000** | 8.68 GiB | **High** — newest architecture, natively multimodal, built-in tool-call parser (`qwen3_coder`), and a **documented** way to suppress the reasoning trace (`enable_thinking=False`), which keeps the mask token close to the prompt; no first-party `[SEG]` precedent | Balanced but unproven; smallest parameter count of the 4B/4.66B class |
| **`Qwen/Qwen3.5-2B`** (2026) | `UNVERIFIED` | **`apache-2.0`** | `UNVERIFIED` | 4.24 GiB (first-party probe) | Medium-High — same 2026 generation and thinking-suppression control (family-level), but its own card was not read | **Attractive on paper**; less verified than the Qwen3-VL line |
| `Qwen/Qwen3-VL-2B-Instruct` | 2,127,532,032 | **`apache-2.0`** | 256K → 1M | 3.96 GiB | High — same tokenizer/architecture family as the strongest known `[SEG]` precedent (unverifiable here); card confirms **tool-calling template** (`tools`/`<tool_call>`/`<tool_response>`) | **Best raw headroom**; smallest first-party-verified footprint |
| `Qwen/Qwen3-VL-4B-Instruct` | 4,437,815,808 | **`apache-2.0`** | 256K → 1M | 8.27 GiB | High — same family; speculative pixel-budget control unverified | Plausible; needs measurement |
| `Qwen/Qwen3-VL-4B-Thinking` | 4,437,815,808 | **`apache-2.0`** | 256K → 1M | 8.27 GiB | **Lower** — template **forces** `<think>` before the answer; VL budget 40,960 output tokens; supervising a mask token after a long CoT needs rationale data | Same memory profile, worse format fit for a mask-token task |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 3,754,622,976 | **`UNVERIFIED`** (ModelScope field empty) | card: config set to **32,768** (+YaRN, "not recommended" for localization) | 6.99 GiB | High — **the only model here with a first-party-documented `min_pixels`/`max_pixels` budget** (4–16384 tokens) | Good size; **licence must be resolved before publication** |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 8,292,166,656 | **`apache-2.0`** | as above | 15.45 GiB | Medium — same documented pixel controls | **Poor**: weights alone ≈15.45 GiB |
| InternVL3.5-2B / SmolVLM2-2.2B / Sa2VA-Qwen3-VL-4B | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` |

## 11. Explicit answers to the three questions

1. **Is there a Qwen3-VL-4B-Thinking?** **Yes — `Qwen/Qwen3-VL-4B-Thinking`**, first-party verified on ModelScope (`License: apache-2.0`, `4,437,815,808` params, BF16, two shards 4,967,229,296 + 3,908,490,048 B = **8.27 GiB computed**). **Does its output format complicate a segmentation-token task? Yes.** Its shipped chat template parses `reasoning_content` separately and, when `add_generation_prompt` is set, appends `<|im_start|>assistant\n<think>\n` — the model therefore begins a chain-of-thought before any answer, and the `[SEG]` token can only appear after `</think>`. Consequences: (a) the mask token occupies a negligible fraction of a very long target sequence (its card recommends a **40,960-token** VL output budget vs 16,384 for the 2B/4B Instruct cards), (b) supervision needs reasoning text alongside masks, (c) the mask-loss weighting must be re-tuned, (d) the decoding/parse path must strip the reasoning block before harvesting `[SEG]` positions. `Qwen/Qwen3.5-4B` has the same default thinking behaviour but **documents `enable_thinking=False`**, which is the practical mitigation.
2. **Does Qwen3-VL support the same `min_pixels`/`max_pixels` image-budget control as Qwen2.5-VL, and are defaults documented?** **`UNVERIFIED`.** Qwen2.5-VL's official card documents `min_pixels`/`max_pixels` **and** states the default visual-token range **4–16384** (plus `resized_height`/`resized_width`, multiples of 28). **No reachable first-party Qwen3-VL or Qwen3.5 card mentions `min_pixels`, `max_pixels`, `longest_edge` or `shortest_edge` at all**, and the corresponding `preprocessor_config.json` (which would hold the numeric defaults) lives on an unreachable host. A third-party docs rendering claims the `size['longest_edge']`/`size['shortest_edge']` mapping and specific defaults — that claim is recorded as non-admissible in §13 and **must be tested empirically**.
3. **Which is the smallest model with an officially released, well-supported Instruct variant suitable for tool-style supervision?** **`Qwen/Qwen3-VL-2B-Instruct`** (2,127,532,032 params, Apache-2.0, 3.96 GiB BF16, 256K context). It is the smallest first-party-verified multimodal model whose **shipped chat template implements tool calling** — the ModelScope card's template renders a `# Tools` system block with `<tools>…</tools>` and a `<tool_call>{"name":…,"arguments":…}</tool_call>` response contract, i.e. it is designed for exactly the tool-style supervision BuildReasonSeg needs. `Qwen/Qwen3.5-4B` is the better **tool-calling** model in absolute terms (first-party-documented `--tool-call-parser qwen3_coder`; card table: BFCL-V4 50.3, TAU2-Bench 79.9, IFEval 89.8) but is larger. **`Qwen/Qwen3.5-2B` is also confirmed to exist first-party** (Apache-2.0, `Qwen3_5ForConditionalGeneration`, `image-text-to-text`, ≈4.24 GiB BF16) but is *larger* than the Qwen3-VL-2B (≈2.27B vs 2.13B params), so it does not displace the answer; its own tool-calling support is `UNVERIFIED`. A still-smaller 2026 variant (`Qwen3.5-0.8B`) is **`UNVERIFIED (host unreachable at research time, 2026-09-26)`**.

## 12. Consolidated `UNVERIFIED` gaps (biggest first)

1. **Qwen3-VL / Qwen3.5 image-token budget API and defaults** — the single most disruptive gap: no first-party source documents `min_pixels`/`max_pixels` for the Qwen3 generation, and the defaults file is unreachable. BuildReasonSeg's mask resolution depends directly on this.
2. **Weight licence of `Qwen/Qwen2.5-VL-3B-Instruct`** — ModelScope's `License`/`LicenseName`/`LicenseLink` are all empty and no reachable source confirms Apache-2.0 → **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`**. Blocking if the project will publish weights or a derivative.
3. **`[SEG]`-token recipe and SAM2 integration** — the *principle* is paper-supported (arXiv:2501.04001: the LLM emits instruction tokens that drive SAM-2), but no concrete token, projection dimension, or Qwen3-VL checkpoint is verifiable, and every probed `Sa2VA-*` ID is 404 on ModelScope → design must be derived from first principles and validated locally.
4. **Code licences** for every Qwen family and for InternVL3.5/SmolVLM2/Sa2VA (GitHub unreachable).
5. **VRAM for a 16 GB LoRA fine-tune** — `NOT_DOCUMENTED` by every reachable card.
6. **QLoRA / bitsandbytes 4-bit & 8-bit** — not documented by any reachable first-party card.
7. **FP16 support** — all reachable evidence is BF16-only.
8. **Windows-native support and Blackwell (RTX 50-series) FlashAttention-2 support** — no first-party statement; only Qwen2.5-VL's `decord`/non-Linux note is documented.
9. **Frozen visual tower / official LoRA hyper-parameters / official maximum-context numbers for Qwen3-VL (1M YaRN config)** — repository-only.
10. **InternVL3.5, SmolVLM2 and Sa2VA in their entirety**, including whether a 2026 InternVL/SmolVLM successor exists and whether any compact (≤10B) Qwen3.8 multimodal model exists (`Qwen/Qwen3.8-27B` was only observable through a non-admissible mirror) — all `UNVERIFIED (host unreachable at research time, 2026-09-26)`.

## 13. Observations derived from an unreachable release (NON-ADMISSIBLE — recorded only to direct re-verification)

**Nothing in this section is evidence for any claim in this report.** These items were observed through the third-party mirror `hf-mirror.com` (or a third-party docs rendering) before `huggingface.co`/`github.com` were confirmed blackholed. They are excluded from every table above and are listed **solely** so a follow-up pass can re-check them on a reachable host or via the paper. In particular, the `[SEG]` details in (b) are **derived from an unreachable release, not from the reachable arXiv abstract**, which mentions only "instruction tokens" and SAM-2.

| # | Non-admissible observation | Status / what would settle it |
|---|---|---|
| a | HF card metadata for Qwen2.5-VL-3B reportedly carries `license_name: qwen-research` (research-only) | **Unresolved; blocking.** Settle via B4/B5 or a reachable mirror-of-record; ModelScope is silent |
| b | A ByteDance Sa2VA release reportedly adds a `[SEG]` token — **reported token id `151669`** on the Qwen3-VL tokenizer — and reads its hidden state with `output_hidden_states=True`, then projects it (**reported `2560`-d LLM hidden → SAM2 hidden dim**) into SAM2 to emit masks; the repo reportedly bundles `sam2.py` and `added_tokens.json` | **Derived from an unreachable release.** The reachable paper supports only the general mechanism. Settle via B7, or via the TPAMI/arXiv **full text** (HTML v4) if the token is described there |
| c | Third-party docs reportedly state `transformers>=4.57.0` for Qwen3-VL and give `size['longest_edge'] == max_pixels`, `size['shortest_edge'] == min_pixels`, with defaults 16,777,216 px / 65,536 px | `UNVERIFIED`; settle via B1/B2 (and note the first-party Qwen3-VL card contains **no** pixel-budget guidance) |
| d | InternVL3.5 reportedly documents a 0.3B-vision/2.0B-language split, `transformers>=4.52.1`, bf16/fp16 and bitsandbytes `load_in_8bit` | `UNVERIFIED (host unreachable)`; settle via B8 |
| e | SmolVLM2 reportedly states "5.2 GB of GPU RAM for video inference" and Apache-2.0 checkpoints | `UNVERIFIED (host unreachable)`; settle via B8 |

## 14. Source list

All URLs accessed **2026-09-26**. Source types: `PRIMARY-VENDOR` (ModelScope official API), `PRIMARY-PAPER` (arXiv), `DOCS` (vendor documentation content on a **non-vendor host** — secondary, flagged), `NEGATIVE` (host unavailability).

| # | Title | Exact URL | Type | Claim it supports |
|---|---|---|---|---|
| 1 | ModelScope API — `Qwen/Qwen3-VL-4B-Instruct` | https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Instruct | PRIMARY-VENDOR | `License: apache-2.0`; `Architectures ["Qwen3VLForConditionalGeneration"]`; `model_size 4,437,815,808`; `tensor_type ["BF16"]`; shards 4,967,229,296 + 3,908,490,048; `StorageSize 8,887,293,488`; card: 256K→1M, FA2 recommended, build transformers from source |
| 2 | ModelScope API — `Qwen/Qwen3-VL-2B-Instruct` | https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-VL-2B-Instruct | PRIMARY-VENDOR | `License: apache-2.0`; `model_size 2,127,532,032`; `model.safetensors` 4,255,140,312; BF16; tool-calling chat template; VL `out_seq_length=16384` |
| 3 | ModelScope API — `Qwen/Qwen3-VL-4B-Thinking` | https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Thinking | PRIMARY-VENDOR | Thinking variant exists; `License: apache-2.0`; `model_size 4,437,815,808`; BF16; **chat template appends `<|im_start|>assistant\n<think>\n`** and splits `reasoning_content`; VL `out_seq_length=40960` |
| 4 | ModelScope API — `Qwen/Qwen2.5-VL-3B-Instruct` | https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-VL-3B-Instruct | PRIMARY-VENDOR | `Architectures ["Qwen2_5_VLForConditionalGeneration"]`; **`License` field empty**; `model_size 3,754,622,976`; shards 3,982,649,232 + 3,526,688,744; BF16; card: `min_pixels`/`max_pixels`, default 4–16384 visual tokens, 32,768-token config + YaRN caveat, non-Linux `decord` fallback |
| 5 | ModelScope API — `Qwen/Qwen2.5-VL-7B-Instruct` | https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-VL-7B-Instruct | PRIMARY-VENDOR | `License: apache-2.0`; `model_size 8,292,166,656`; 5 shards totalling 16,584,414,560 B; BF16; same card guidance as the 3B |
| 6 | ModelScope API — `Qwen/Qwen3.5-4B` | https://www.modelscope.cn/api/v1/models/Qwen/Qwen3.5-4B | PRIMARY-VENDOR | 2026 compact model exists; `License: apache-2.0`; `Architectures ["Qwen3_5ForConditionalGeneration"]`; `ModelType ["qwen3_5"]`; `BaseModel Qwen/Qwen3.5-4B-Base`; `model_size 4,659,865,088`; `tensor_type ["BF16","F32"]`; shards 5,329,398,688 + 3,990,429,408; card: **262,144→1,010,000 context**, thinking on by default, `enable_thinking=False`, `qwen3_coder` tool parser, padded tied vocab 248,320, YaRN `rope_parameters` recipe |
| 7 | ModelScope API — `ByteDance/Sa2VA-Qwen3-VL-4B` | https://www.modelscope.cn/api/v1/models/ByteDance/Sa2VA-Qwen3-VL-4B | PRIMARY-VENDOR (**negative**) | HTTP 404 `"record not found"` → Sa2VA is not hosted first-party; its `[SEG]`/SAM2 claims are therefore `UNVERIFIED` here |
| 8 | Qwen3-VL Technical Report | https://arxiv.org/abs/2511.21631 | PRIMARY-PAPER | Qwen3-VL dense family 2B/4B/8B/32B (+MoE); **native 256K-token interleaved context**; Interleaved-MRoPE, DeepStack, text-based time alignment |
| 9 | Qwen3-VL docs — Model Cards (third-party rendering) | https://mintlify.wiki/QwenLM/Qwen3-VL/resources/model-cards | DOCS (secondary, **non-vendor host**) | Qwen3-VL model list incl. 2B/4B Thinking editions and release dates; FP8 availability; legacy Qwen2.5-VL sizes — used only for orientation, not for any table claim |
| 10 | Qwen3-VL docs — Pixel Control (third-party rendering) | https://mintlify.wiki/QwenLM/Qwen3-VL/inference/pixel-control | DOCS (secondary) | Claim that `size['longest_edge']`==`max_pixels` / `shortest_edge`==`min_pixels` and the 32×-compression token maths → **recorded as non-admissible in §13** |
| 11 | Qwen3-VL docs — Installation / FAQ / Troubleshooting / LoRA / Training Overview (third-party renderings) | https://mintlify.wiki/QwenLM/Qwen3-VL/installation · https://mintlify.wiki/QwenLM/Qwen3-VL/resources/faq · https://mintlify.wiki/QwenLM/Qwen3-VL/resources/troubleshooting · https://mintlify.wiki/QwenLM/Qwen3-VL/fine-tuning/lora.md · https://mintlify.wiki/QwenLM/Qwen3-VL/fine-tuning/overview | DOCS (secondary) | Reported `transformers>=4.57.0`, VRAM/LoRA-memory tables, visual-tower freezing flags, FA2/CUDA notes, 1M YaRN recipe → **all non-admissible here**; only the parts independently present in source #1–#3 (256K→1M, FA2 recommended, source build) are treated as verified |
| 12 | Qwen3.5: Towards Native Multimodal Agents (Alibaba Cloud blog) | https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894 | PRIMARY-VENDOR | Qwen3.5 = 2026 native vision-language generation (early fusion, Gated DeltaNet + sparse MoE, FP8 training pipeline, 201 languages); 397B-A17B launch |
| 13 | Qwen3.5: 迈向原生多模态智能体 (Alibaba developer community) | https://developer.aliyun.com/article/1712860 | PRIMARY-VENDOR | Qwen3.5 released around 2026-02-17…24 as the native multimodal generation; published on GitHub/ModelScope |
| 14 | Sa2VA: Marrying SAM2 with MLLM for Dense Grounded Understanding of Images and Videos (IEEE TPAMI 2026) | https://arxiv.org/abs/2501.04001 | PRIMARY-PAPER | SAM-2 + MLLM unified in an LLM token space; **the LLM generates instruction tokens that guide SAM-2 to produce masks**; supports referring segmentation for images and video; extensible to Qwen-VL and Intern-VL; Ref-SAV dataset; code claimed at `github.com/Bytedance/Sa2VA` (unreachable) |
| 15 | ModelScope API — `Qwen/Qwen3.5-2B` | https://www.modelscope.cn/api/v1/models/Qwen/Qwen3.5-2B | PRIMARY-VENDOR | `License: apache-2.0`; `Architectures ["Qwen3_5ForConditionalGeneration"]`; task `image-text-to-text`; ≈4.24 GiB of BF16 weights (probe performed first-hand by the parent agent, 2026-09-26) |
| 16 | Qwen3-VL Technical Report — full text entry point (not mined in this pass) | https://arxiv.org/html/2511.21631v2 | PRIMARY-PAPER (available, **unused**) | Would settle Qwen3-VL image-token budget, vision/language parameter split and architecture details without touching HF |
