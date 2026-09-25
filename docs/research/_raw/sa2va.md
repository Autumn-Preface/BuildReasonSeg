# Sa2VA & SAMTok — raw research notes (for BuildReasonSeg)

Access date for every source: **2026-09-26**.
Scope: Sa2VA paper/repo/checkpoints, SAMTok, SAM2 variant, training recipe, task surface, inference requirements, `[SEG]` hidden-state access, adding a second special token, Qwen3-VL adaptation maturity, and an effort/risk estimate for a 16 GB-VRAM laptop MVP.

---

## 0. Evidence provenance & verification status (**read first**)

This session's network blocked several first-party hosts. Evidence tiers used below:

| Tier | Meaning | How to read it |
|---|---|---|
| **[P1]** | **First-party, fetched successfully at research time** (`arxiv.org`, `openaccess.thecvf.com`). | Treat as confirmed. |
| **[P2]** | **Official content, but read through a third-party intermediary** (raw-content proxy in front of the official GitHub repo; third-party mirror of the Hugging Face API/model cards). | Facts (IDs, sizes, config values, code behaviour) usable as *strong indications*; **licence values from this tier are NOT admissible evidence** and are reported as `UNVERIFIED / DO NOT USE UNTIL RESOLVED`. |
| **[X]** | **Host unreachable at research time** (DNS-blackholed or fetch failure). | `UNVERIFIED`. |
| **[I]** | My inference / estimate / judgement. | Not a measured result. |

Host status observed from **this** session (2026-09-26):

| Host | Status here | Notes |
|---|---|---|
| `arxiv.org` (abs, `/html/<id>vN` full text) | **REACHABLE** | Used for Sa2VA v4 full text and both abstracts |
| `openaccess.thecvf.com` | **REACHABLE** | Used for the SAMTok CVPR 2026 published version |
| `lxtgh.github.io`, `zhouyiks.github.io` (project pages) | **REACHABLE here** | Parent agent reports `*.github.io` blackholed in its own session; in mine both fetches returned HTTP 200 → recorded as reachable **at my research time** only |
| `hf-mirror.com` (third-party HF mirror) | reachable | **Third-party mirror — never cited for licence** |
| `ghproxy.net` (raw proxy over `raw.githubusercontent.com`) | reachable | **Third-party intermediary — provenance for repo facts, not for licence** |
| `www.modelscope.cn` | reachable | API probed for Sa2VA → **404 (not found)** |
| `github.com`, `raw.githubusercontent.com`, `api.github.com`, `huggingface.co`, `hf.co`, `www.huggingface.co`, `cdn.jsdelivr.net`, `data.jsdelivr.com`, `r.jina.ai`, `ai.meta.com` | **UNREACHABLE / failed** | Every official repo, model card, dataset card and `LICENSE` URL lives here → see §15 |
| `seed.bytedance.com` | No Sa2VA/SAMTok first-party page located by search | ByteDance Seed first-party confirmation: **UNVERIFIED** |

**Consequence that dominates this document:** the Sa2VA/SAMTok model *release* (checkpoint IDs, file lists, sizes, dtypes, licence) could **not** be verified on any first-party host. All release-level statements below are tier **[P2]** and every checkpoint licence is `UNVERIFIED / DO NOT USE UNTIL RESOLVED`. The *science* (architecture, tokenizer design, training stages, frozen/trainable modules, task surface) is tier **[P1]** from the arXiv/CVF full texts.

---

## 1. Papers (first-hand, [P1])

| Item | Fact | Source |
|---|---|---|
| Sa2VA title | "Sa2VA: Marrying SAM2 with MLLM for Dense Grounded Understanding of Images and Videos" | [arXiv:2501.04001](https://arxiv.org/abs/2501.04001) |
| Sa2VA authors | Haobo Yuan, Xiangtai Li, Tao Zhang, Yueyi Sun, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, Ming-Hsuan Yang | [arXiv abs](https://arxiv.org/abs/2501.04001) |
| Sa2VA affiliations | UC Merced, ByteDance Seed, Wuhan University, Peking University | [arXiv HTML v4](https://arxiv.org/html/2501.04001v4) |
| Sa2VA status | Comments field: "Accepted by IEEE TPAMI"; journal ref "IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026"; DOI `10.1109/TPAMI.2026.3720379` | [arXiv abs](https://arxiv.org/abs/2501.04001) |
| Sa2VA dates | v1 **7 Jan 2025**; v2 13 Feb 2025; v3 3 Nov 2025; v4 (current) **24 Aug 2026** | [arXiv abs](https://arxiv.org/abs/2501.04001) |
| Sa2VA paper text licence | arXiv.org perpetual non-exclusive licence | [arXiv abs](https://arxiv.org/abs/2501.04001) |
| Sa2VA core idea (paper wording) | Combines SAM-2 with an MLLM, unifies text/image/video in a shared LLM token space; LLM generates "instruction tokens" that guide SAM-2 to produce masks; minimal **single-stage** instruction tuning; extendable to Qwen-VL and Intern-VL families | [arXiv abs](https://arxiv.org/abs/2501.04001) |
| Sa2VA dataset contribution | **Ref-SAV**: auto-labelled, >72k object expressions in complex video scenes; 2k manually validated video objects as benchmark | [arXiv abs](https://arxiv.org/abs/2501.04001) |
| Sa2VA capability table | Columns: Image, Video, Visual Prompts, RES, Ref-VOS, **Inter-VOS**, GCG, Image-Chat, Video-Chat, Video Caption, Interactive Caption, End-to-End Training — Sa2VA claimed to cover the full spectrum | [arXiv HTML v4](https://arxiv.org/html/2501.04001v4) (read up to this session's fetch cap) |
| Sa2VA paper-stated encoder freezing | "we adopt a decoupled design in which SAM-2's decoder and memory module are frozen" (v1 text) | [ar5iv v1 full text](https://ar5iv.labs.arxiv.org/html/2501.04001) |
| Sa2VA paper states GPUs/hours/epochs (§Implementation Details) | **UNVERIFIED** — this section is past this session's fetch-truncation point on both arXiv HTML renders; the PDF is not readable through this harness | — |
| SAMTok title / venue | "SAMTok: Representing Any Mask with Two Words" — **CVPR 2026**, pp. 37852–37863 | [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_SAMTok_Representing_Any_Mask_with_Two_Words_CVPR_2026_paper.html) |
| SAMTok authors | Yikang Zhou, Tao Zhang, Dengxian Gong, Yuanzheng Wu, Ye Tian, Haochen Wang, Haobo Yuan, Jiacong Wang, Lu Qi, Hao Fei, Shunping Ji, Anran Wang, Zhuochen Wang, Yujing Wang, Cheng Chen, Xiangtai Li | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_SAMTok_Representing_Any_Mask_with_Two_Words_CVPR_2026_paper.html) |
| SAMTok dates | v1 **22 Jan 2026**, v2 14 Jun 2026; arXiv comment "CVPR 2026 Highlight" ("Highlight" itself is stated in the arXiv comment, not on the CVF page) | [arXiv:2601.16093](https://arxiv.org/abs/2601.16093) |
| SAMTok method (paper) | Discrete mask tokenizer → exactly **two** special tokens per region mask, high-fidelity reconstruction; builds on SAM2; **trained on 209M masks** with a mask encoder + residual vector quantizer; 5M SAMTok-formatted samples; enables pixel-wise ability via plain next-token prediction + RL, no architecture change or specialised loss; textual answer-matching reward for GRPO; results on region captioning, region VQA, grounded conversation, referring segmentation, scene-graph parsing, multi-round interactive segmentation | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_SAMTok_Representing_Any_Mask_with_Two_Words_CVPR_2026_paper.html), [arXiv abs](https://arxiv.org/abs/2601.16093) |
| SAMTok affiliations | Wuhan University, ByteDance, NUS | [arXiv abs](https://arxiv.org/abs/2601.16093) |

---

## 2. Official repository identity, structure and code licence

All repo-derived facts in this document come from **official repo file content served through a raw-content proxy** — tier **[P2]**. The canonical URLs (`github.com/...`, `raw.githubusercontent.com/...`) were unreachable (§15).

| Item | Fact | Tier | Source (fetched URL) |
|---|---|---|---|
| Official repo | `github.com/bytedance/Sa2VA`, now branded "Pixel LLMs: Pixel-Level Grounded Understanding for Multimodal LLMs", a monorepo of `projects/sa2va`, `projects/vrt_sa2va`, `projects/samtok`, `projects/sasasa2va`, `projects/pixel_sail` | [P2] | [raw proxy README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/README.md) |
| Mapping to papers | Repo declares "Sa2VA (T-PAMI-26), SAMTok (CVPR-26), VRT (Arxiv-25), SaSaSa2VA (1st solution LSVOS)" | [P2] | [raw proxy README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/README.md) |
| **Code licence** | `LICENSE` content served through the proxy is the **Apache License 2.0** text. The canonical `https://github.com/bytedance/Sa2VA/blob/main/LICENSE` and `https://raw.githubusercontent.com/bytedance/Sa2VA/main/LICENSE` were **unreachable**, so per policy: **code licence = `UNVERIFIED / DO NOT USE UNTIL RESOLVED`** (indication: Apache-2.0, not admissible) | [P2] → licence **[X]** | [raw proxy LICENSE](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/LICENSE) (official URL unreachable) |
| Training code public? | Yes — `bash tools/dist.sh train projects/sa2va/configs/<cfg>.py <n_gpus>`; per-backbone configs; `tools/convert_to_hf.py`, `tools/convert_to_pth.py`; fine-tune guide | [P2] | [projects/sa2va/README.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md) |
| Env | `uv` + `pyproject.toml`/`uv.lock` under `projects/sa2va`; commands run from repo root | [P2] | same |
| Stale links | Legacy `Sa2VA-8B` card and the project page still point to `github.com/magic-research/Sa2VA` → repo moved orgs | [P1]/[P2] | [project page](https://lxtgh.github.io/project/sa2va), [8B card via mirror](https://hf-mirror.com/ByteDance/Sa2VA-8B/raw/main/README.md) |

---

## 3. SAMTok vs Sa2VA — relationship

**Answer: two separate contributions in one monorepo. SAMTok is *not* the tokenizer inside the shipped Sa2VA checkpoints.**

| Question | Answer | Tier | Source |
|---|---|---|---|
| Is SAMTok a distinct contribution? | Yes: separate paper (CVPR 2026), separate project dir `projects/samtok`, separate model collection (`zhouyik/*`), listed separately in the monorepo README | [P1] venue + [P2] repo layout | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_SAMTok_Representing_Any_Mask_with_Two_Words_CVPR_2026_paper.html), [raw proxy root README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/README.md) |
| Is SAMTok "the tokenizer inside Sa2VA"? | **No.** Sa2VA emits one `[SEG]` token whose **continuous** last-layer hidden state is projected into SAM2 prompt space; the shipped Sa2VA chat module contains no mask encoder, no VQ codebook and no `<\|mt_*\|>` tokens | [P2] code | [modeling_sa2va_qwen.py (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/modeling_sa2va_qwen.py) |
| What SAMTok adds | A *discrete* 2-token mask language (`<\|mt_start\|><\|mt_XXXX\|><\|mt_YYYY\|><\|mt_end\|>`) plus a `VQ_SAM2` mask tokenizer (SAM2 image encoder + residual VQ, `mask_tokenizer_256x2.pth`, codebook 256×2) so that **any** base MLLM gains masks by next-token prediction / RL with no architectural change | [P1] paper + [P2] repo/card | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_SAMTok_Representing_Any_Mask_with_Two_Words_CVPR_2026_paper.html), [raw proxy samtok README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/samtok/README.md) |
| Shared lineage | Same monorepo, overlapping ByteDance/WHU authorship, both built on SAM2 | [P2] | [raw proxy root README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/README.md) |
| SAMTok declared release completeness | Repo checkboxes mark weights / tokenizer code / training instructions / eval / demo / gradio / RL codes as released; 13 checkpoints listed (Qwen3-VL-8B/4B-SAMTok, Qwen2.5-VL-7B/3B-SAMTok-co, PLM-1B-SAMTok-co, Qwen3-VL-4B-SAMTok-dam, GCG/GRES RL+FT variants) | [P2] | [raw proxy samtok README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/samtok/README.md) |
| SAMTok checkpoint licence | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` (HF card unreachable; mirror metadata is not admissible) | [X] | canonical `huggingface.co/zhouyik/...` unreachable → §15 |

---

## 4. Checkpoint inventory

### 4.1 Status of this inventory (read before using)

| Point | Statement |
|---|---|
| Provenance | Every row below was assembled from **third-party mirror metadata of the official Hugging Face API** (`hf-mirror.com/api/models/...`) plus HF search listings. **Tier [P2].** |
| Verification | **The Sa2VA release could not be verified first-hand.** `huggingface.co` is blackholed here; the parent's probe of ModelScope returned 404 for `ByteDance/Sa2VA-Qwen3-VL-4B`, `Sa2VA-Qwen2.5-VL-3B`, `Sa2VA-Qwen2.5-VL-4B`, `Sa2VA-InternVL3-2B`; my own probe of `ByteDance/Sa2VA-Qwen3-VL-2B` also returned **404** (`{"Code":10010205001,"Message":"获取模型信息失败，信息：record not found"}`). No `seed.bytedance.com` Sa2VA page was found. |
| Therefore | **Existence and exact IDs: [P2]-supported only (unverified). Sizes/dtypes/shard counts: [P2]. Licence of every checkpoint: `UNVERIFIED / DO NOT USE UNTIL RESOLVED`.** Mirror metadata *displayed* `apache-2.0` for all of them, but that value is **not admissible licence evidence** and is not offered as such. |
| Separately | "What the paper states" (tier [P1]) is in §1/§7; "what the release contains" (tier [P2], unverified) is here. They are deliberately not merged. |

### 4.2 Candidate released Sa2VA checkpoints (tier [P2] — unverified)

Canonical IDs, should you verify them yourself on a reachable host: `https://huggingface.co/<ID>`.

| Exact HF model ID (canonical form) | Base MLLM (mirror card/table) | Total params (mirror metadata) | Dtype observed (mirror) | Σ weights ≈ | Shards | **Licence** |
|---|---|---|---|---|---|---|
| `ByteDance/Sa2VA-1B` | InternVL2.5-1B (LLM Qwen2.5-0.5B-Instruct) | 1,163,665,714 (F32 859.7M + BF16 304.0M) | F32 + BF16 | 7.55 GB | 1 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-4B` | InternVL2.5-4B (LLM Qwen2.5-3B-Instruct) | 3,941,809,714 (F32 3.638B + BF16 304.0M) | F32 + BF16 | 15.16 GB | 4 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-8B` | InternVL2.5-8B (LLM internlm2_5-7b-chat) | 8,317,666,866 (F32 8.014B + BF16 304.0M) | F32 + BF16 | 32.66 GB | 7 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-26B` | InternVL2.5-26B (LLM internlm2_5-20b-chat) | 25,778,005,938 (F32 20.242B + BF16 5.536B) | F32 + BF16 | 92.04 GB | 20 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-InternVL3-2B` | InternVL3-2B (LLM Qwen2.5-1.5B) | 2,316,157,234 (F32 2.012B + BF16 304.0M) | F32 + BF16 | 8.67 GB | 2 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-InternVL3-8B` | InternVL3-8B (LLM Qwen2.5-7B) | 8,182,606,130 (F32 7.879B + BF16 304.0M) | F32 + BF16 | 32.13 GB | 7 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-InternVL3-14B` | InternVL3-14B (LLM Qwen2.5-14B) | 15,369,268,530 (F32 15.065B + BF16 304.0M) | F32 + BF16 | 60.88 GB | 13 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-Qwen2_5-VL-3B` | Qwen2.5-VL-3B-Instruct | 4,293,849,394 (F32) | F32 | 17.19 GB | 4 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-Qwen2_5-VL-7B` | Qwen2.5-VL-7B-Instruct | 8,527,538,994 (F32) | F32 | 34.12 GB | 7 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-Qwen3-VL-2B` | Qwen3-VL-2B-Instruct | 2,666,774,834 (F32) | F32 | 10.68 GB | 3 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-Qwen3-VL-4B` | Qwen3-VL-4B-Instruct | 5,057,072,434 (F32) | F32 | 20.24 GB (20,228,290,760 B) | 5 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-Qwen3-VL-4B-SAM3` | Qwen3-VL-4B-Instruct + SAM3 grounding encoder | 5,306,227,586 (F32) | F32 | 21.24 GB | 5 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ByteDance/Sa2VA-LLaVA-1.5-7B` | LLaVA-1.5-7B (CLIP-ViT-L-336 + Vicuna-7B) | 7,305,220,402 (F32) | F32 | 29.22 GB | 6 | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |

Mirror-API evidence URLs (all tier [P2], canonical IDs above):
[1B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-1B) · [4B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-4B) · [8B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-8B) · [26B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-26B) · [InternVL3-2B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-InternVL3-2B) · [InternVL3-8B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-InternVL3-8B) · [InternVL3-14B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-InternVL3-14B) · [Qwen2.5-VL-3B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen2_5-VL-3B) · [Qwen2.5-VL-7B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen2_5-VL-7B) · [Qwen3-VL-2B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-2B) · [Qwen3-VL-4B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-4B) · [Qwen3-VL-4B-SAM3](https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-4B-SAM3) · [LLaVA-1.5-7B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-LLaVA-1.5-7B) · [author search](https://hf-mirror.com/api/models?author=ByteDance&search=Sa2VA&limit=100).

Additional [P2] observations about the release (unverified, listed because they matter operationally):

| Observation | Detail |
|---|---|
| Legacy names are InternVL-based | `Sa2VA-1B/4B/8B/26B` map to InternVL2.5 backbones in the repo model-zoo table — not Qwen |
| Qwen-family coverage as claimed | Qwen2.5-VL-3B, Qwen2.5-VL-7B, Qwen3-VL-2B, Qwen3-VL-4B (+ Qwen3-VL-4B-SAM3) |
| Format | safetensors only in the mirror file lists; each repo also ships `modeling_sa2va_qwen.py` / `modeling_sa2va_chat.py` and `sam2.py` (loads via `trust_remote_code`) |
| FP32 by default | `config.json` `"dtype": "float32"`; mirror dtype metadata F32 → ~2× the disk of a bf16 release; cards instruct `torch_dtype=torch.bfloat16` at load |
| Mirror metadata bug (operational) | Qwen-based repos carry `base_model: OpenGVLab/InternVL3-8B`, contradicting the repo table → do not trust `base_model` tags for these IDs |
| Legacy duplicate org | `Dense-World/Sa2VA-*` uploads (Dec-2024/Jan-2025) mirror the same models; treated as same-team legacy, not current model zoo |

### 4.3 Unofficial derivatives seen in mirror search (not releases; tier [P2], licences `UNVERIFIED / DO NOT USE UNTIL RESOLVED`)

| ID | What it is | Licence |
|---|---|---|
| `nicehero/Sa2VA-Qwen3-VL-4B-fp8` | community FP8 (F8_E4M3) re-quantisation of `ByteDance/Sa2VA-Qwen3-VL-4B`; 5.057B params; ≈5.07 GB | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `HarborYuan/R-Sa2VA-Qwen3VL-4B-SFT`, `...-RL` | VRT-project fine-tunes of Sa2VA-Qwen3-VL-4B (SFT / SFT+RL) | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `kumuji/Sa2VA-i-1B/4B/8B/26B` | third-party "Sa2VA-i" fine-tunes (LSVOS 2025 MeViS report, arXiv 2509.19082) | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |
| `ly17/sa2va-*` | third-party medical/vessel fine-tunes (2026) | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` |

---

## 5. SAM2 / SAM2.1 variant used, and whether SAM2 weights are redistributed

| Question | Answer | Tier | Source |
|---|---|---|---|
| SAM2 variant for Sa2VA (release-side evidence) | SAM 2.0 **Hiera-L**: repo README instructs downloading `sam2_hiera_large.pt` and links `facebook/sam2-hiera-large`; no `sam2.1_*` reference appears in the Sa2VA configs/READMEs. Input resolution **1024** | [P2] | [projects/sa2va/README.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md), [sa2va.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/sa2va.py) |
| SAM2 variant, paper-side | Paper states Sa2VA combines **SAM-2** (Meta's foundation video segmentation model) with an MLLM; the specific 2.0-vs-2.1 checkpoint is a release detail, not stated in the abstract | [P1] | [arXiv abs](https://arxiv.org/abs/2501.04001) |
| Architecture actually instantiated in code | Hiera image encoder `embed_dim=144, num_heads=2, stages=[2,6,36,4], global_att_blocks=[23,33,43]` (= Hiera-L), FPN neck `d_model=256, backbone_channel_list=[1152,576,288,144]`, 4-layer memory attention, memory encoder, SAM prompt encoder + `MaskDecoder(num_multimask_outputs=3, TwoWayTransformer depth 2, 8 heads)`; **pure PyTorch, no repo-local CUDA kernels** | [P2] | [sam2.py (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/sam2.py) |
| SAMTok's variant | **SAM 2.1 Hiera-L** (`sam2.1_hiera_large.pt`) inside `VQ_SAM2` — deliberately different from Sa2VA's 2.0 | [P2] | [raw proxy samtok README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/samtok/README.md) |
| SAM3 variant | `facebookresearch/sam3` tracker vendored under `third_parts/sam3`; native input **1008**, `sam3.pt` expected at `pretrained/sam3/sam3.pt` (or `SA2VA_SAM3_PATH`) | [P2] | [sa2va_qwen3_8b_sam3.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_8b_sam3.py) |
| Are SAM2 weights redistributed with Sa2VA, or fetched separately? | **Redistributed inside the Sa2VA safetensors.** The checkpoint's weight index contains `grounding_encoder.sam2_model.{image_encoder,memory_attention,memory_encoder,sam_mask_decoder}.*` keys (Qwen3-VL-4B: `total_size` 20,228,290,760 B), and the model repo's file list contains **no** `sam2*.pt`. A separate download is needed only for from-scratch training (initialisation) and for SAMTok inference | [P2] | [model.safetensors.index.json (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/model.safetensors.index.json), [repo README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md) |
| SAM2 magnitude inside the checkpoint | In the InternVL2.5/InternVL3 repos the non-F32 (BF16) component is exactly **304,012,288 params** ≈ the SAM2 stack + `text_hidden_fcs`; Qwen repos are all-F32 so it is not separable from mirror metadata. **[I]** on the "= SAM2 stack" attribution | [P2] + [I] | mirror API responses §4.2 |
| SAM2 weight licence | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` — `ai.meta.com` did not serve a usable page in this session and the SAM2 model card on HF is unreachable. *(Mirror metadata indicated apache-2.0; not admissible.)* | [X] | §15 |
| SAM3 weight licence | `UNVERIFIED / DO NOT USE UNTIL RESOLVED` — mirror metadata indicated `license: other` **and gated**; first-party confirmation impossible here. Flagged as the highest licensing risk in the family | [X] | §15 |

---

## 6. Training recipe

Provenance: this section is **release/repo-side, tier [P2]**. The paper's own §Implementation Details remains **UNVERIFIED** (fetch truncation, §1). Do not attribute these numbers to the paper.

| Aspect | Fact | Tier | Source |
|---|---|---|---|
| Stages | **One** stage: one `TrainLoop` over a concatenated dataset, `max_epochs = 1`; paper abstract says "minimal single-stage instruction tuning" ([P1] for the phrase) | [P2] | [sa2va_in30_8b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_in30_8b.py), [arXiv abs](https://arxiv.org/abs/2501.04001) |
| Backbone recipes present | InternVL2.5 (`sa2va_in30_*`), InternVL3, Qwen2.5-VL (`configs/sa2va_qwenvl25/`), Qwen3-VL (`configs/sa2va_qwenvl3/`: `sa2va_qwen3_4b.py`, `sa2va_qwen3_8b_sam3.py`) | [P2] | [projects/sa2va/README.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md), [sa2va_qwen3_4b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py) |
| Frozen / trainable (InternVL3-8B recipe) | `freeze_llm=True`, `freeze_visual_encoder=True`, `frozen_sam2_decoder=False`; LLM LoRA `r=256, alpha=512, dropout=0.05, modules_to_save=["embed_tokens","lm_head"]` | [P2] | [sa2va_in30_8b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_in30_8b.py) |
| Frozen / trainable (Qwen3-VL-4B recipe) | same flags; LLM LoRA `r=128, alpha=256`, `modules_to_save=['lm_head','embed_tokens']` | [P2] | [sa2va_qwen3_4b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py) |
| Is LoRA used? | Yes — LoRA on the **LLM only**; vision encoder frozen; `text_hidden_fcs` and SAM2 mask decoder trained | [P2] | [sa2va.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/sa2va.py), [qwen3vl.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/mllm/qwen3vl.py) |
| Which SAM2 parts train | Whole grounding encoder frozen, then **only `sam2_model.sam_mask_decoder` unfrozen**; the saved state dict keeps only `grounding_encoder.sam2_model.sam_mask_decoder*`, `text_hidden_fcs.*` and the MLLM | [P2] | [sa2va.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/sa2va.py) |
| SAMTok tokenizer frozen/trained? | Not applicable to Sa2VA: `[SEG]` is a plain added special token; `text_hidden_fcs` (in_dim→out_dim MLP with ReLU) is trained | [P2] | [modeling_sa2va_qwen.py (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/modeling_sa2va_qwen.py), [sa2va.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/sa2va.py) |
| Datasets in the single stage | RefCOCO / RefCOCO+ / RefCOCOg (repeats 5/5/4); LLaVA-665k; ReVOS (10) / MeViS (4) / RefYTVOS (4) / Ref-SAV (4, requires Meta SA-V); Chat-UniVi VideoQA (5 frames); GCG: RefCOCOg-GCG (10), GranDf (100), Flickr30k (10), OpenPSG (10); Osprey-724k visual prompting — **commented out in the Qwen3-VL recipe** ("remove vp due to not supported yet") | [P2] | [sa2va_qwen3_4b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py) |
| Batching / optimiser | InternVL3-8B: bs 2/device × accum 4 ("on 32 gpus"); Qwen3-VL-4B: bs 1 × accum 8 → comment states effective batch **128 = 1×8×16 GPUs**; SAM3-8B: bs 4 × 1 × 32 GPUs = 128. lr 4e-5, AdamW, cosine + 5% warmup, bf16 AMP, `max_length=8192` | [P2] | [sa2va_in30_8b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_in30_8b.py), [sa2va_qwen3_4b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py) |
| GPUs (repo docs) | "we suggest using at least 8 A100 GPUs"; training command takes a GPU count (example: 8) | [P2] | [projects/sa2va/README.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md) |
| GPU-hours / wall clock | **UNVERIFIED** (not documented in any reachable source) | [X] | — |
| Official fine-tune path | `tools/convert_to_pth.py <hf_model_path> --arch-type {internvl,qwen}` → edit `configs/sa2va_finetune.py` → `bash tools/dist.sh train ... 8`; example dataset + `Sa2VAFinetuneDataset`; guide `projects/sa2va/docs/finetune.md`; example data repo referenced as `bitersun/Sa2VA-finetune-example` | [P2] | [finetune.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/docs/finetune.md) |
| Fine-tune recipe specifics | LoRA `r=128, alpha=256`; `freeze_llm=True`, `freeze_visual_encoder=True`, `frozen_sam2_decoder=False`; `repeats=100`; bs 2 × accum 16 ("on 8 gpus"); `arch_type='qwen'` + `Qwen2_5_VLProcessor`/`Qwen3VLProcessor` variants documented inline | [P2] | [sa2va_finetune.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_finetune.py) |
| Downstream adaptation precedent | VRT (same repo, Sa2VA lead author) fine-tunes Sa2VA-Qwen3-VL-4B with LoRA `r=128`, deliberately **without** `modules_to_save` ("we do not need to finetune the embedding layer and lm_head"), initialised from a converted `Sa2VA-Qwen3-VL-4B_converted.pth`, 8 GPUs; ships R-Sa2VA SFT/RL models and VRT-Bench results | [P2] | [vrt config](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/vrt_sa2va/configs_sa2va/vrt_sa2va_4b_qwen3_sft.py), [vrt README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/vrt_sa2va/README.md) |

---

## 7. Task surface

| Task | Officially supported? | Tier | Source |
|---|---|---|---|
| Image referring segmentation (RES, RefCOCO/+/g) | **Yes** — training data + `sa2va_eval_refcoco.py` + metrics | [P1] capability table + [P2] eval script | [arXiv HTML v4](https://arxiv.org/html/2501.04001v4), [run_all_evals.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/evaluation/run_all_evals.py) |
| Image reasoning segmentation | Claimed as an RES/GCG capability; **no dedicated reasoning-segmentation benchmark or dataset in the repo eval set** | [P2] | [projects/sa2va/README.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md) |
| Video referring VOS (ReVOS, MeViS, RefYTVOS, Ref-SAV, DAVIS) | **Yes** — training data + `sa2va_eval_ref_vos.py` (datasets `DAVIS`, `MEVIS_U`) | [P2] | [run_all_evals.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/evaluation/run_all_evals.py) |
| Grounded conversation generation (GCG) | **Yes** — GCG training data + `sa2va_eval_gcg.py` + `metrics_gcg.py` | [P2] | [run_all_evals.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/evaluation/run_all_evals.py) |
| Image / video chat & QA | **Yes** — LLaVA-665k + Chat-UniVi VideoQA in training; QA benchmark eval via an external `sa2va_eval` (VLMEvalKit fork) | [P2] | [projects/sa2va/README.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md) |
| Interactive / visual-prompt segmentation (mask prompt in) | **Yes at inference** — `predict_forward(..., mask_prompts=...)` in the quick-start; training-side VP data **disabled** in the Qwen3-VL recipe | [P2] | [Qwen3-VL-4B card (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/README.md), [sa2va_qwen3_4b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py) |
| Multi-round interactive segmentation | Supported in the **SAMTok** line (a paper result), not in Sa2VA's own eval set | [P1] | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_SAMTok_Representing_Any_Mask_with_Two_Words_CVPR_2026_paper.html) |
| **In-context / one-shot segmentation** (reference mask → new mask) | **UNVERIFIED** — no such task, dataset, config or eval found. Note: the paper's "one-shot" wording refers to *single-stage instruction tuning*, not one-shot segmentation | [P1] wording / absence of evidence | [arXiv abs](https://arxiv.org/abs/2501.04001) |

---

## 8. Inference requirements

| Item | Fact | Tier | Source |
|---|---|---|---|
| Transformers version | Env extras in the repo: `latest → transformers==4.57.1` (Qwen3-VL, Qwen2.5-VL, InternVL3); `legacy → transformers==4.49.0` (InternVL2.5 or earlier). A legacy inference-only requirements file pins `transformers==4.42.3`. Mirror `config.json` for Sa2VA-Qwen3-VL-4B records `transformers_version: 4.57.0` | [P2] | [pyproject.toml](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/pyproject.toml), [demo/requirements.txt](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/demo/requirements.txt), [config.json (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/config.json) |
| FlashAttention required or optional? | Officially a **hard pinned dependency** (`flash-attn==2.7.3`; bare `flash_attn` in the demo requirements). The Qwen3VL training wrapper hardcodes `attn_implementation="flash_attention_2"`. Quick-start snippets pass `use_flash_attn=True` | [P2] | [pyproject.toml](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/pyproject.toml), [qwen3vl.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/mllm/qwen3vl.py) |
| Non-FlashAttention fallback | **UNVERIFIED.** The chat module declares `_supports_flash_attn_2 = True` and accepts a `use_flash_attn` kwarg that appears unused in the file — an **[I]** reading, not documented behaviour | [P2]/[I] | [modeling_sa2va_qwen.py (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/modeling_sa2va_qwen.py) |
| Compiled CUDA ops | `flash-attn` is a compiled CUDA extension built at install time (pyproject sets `no-build-isolation-package = ["flash-attn"]`, `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`). The Sa2VA/SAM2 model code itself is pure PyTorch. Mask decoding calls `torch.autocast(device_type="cuda", …)` → the inference path is **CUDA-only** | [P2] | [pyproject.toml](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/pyproject.toml), [sam2.py (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/sam2.py) |
| Torch / Python pins | `torch>=2.6.0`; `requires-python = ">=3.11,<3.12"`; `peft==0.17.1`, `xtuner[deepspeed]==0.1.23`, `timm==1.0.17`, `decord2==1.0.0` | [P2] | [pyproject.toml](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/pyproject.toml) |
| Documented VRAM (inference) | **UNVERIFIED** — no VRAM figure anywhere reachable | [X] | — |
| Recommended GPU (training) | "at least 8 A100 GPUs"; recipes assume 8/16/32-GPU runs | [P2] | [projects/sa2va/README.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md) |
| Documented Windows support | **None.** Configuration evidence points to unsupported: `torch`/`torchvision` are routed to the **CPU-only** PyTorch index when `sys_platform != 'linux'`; `flash-attn` is Linux/CUDA-oriented; training `env_cfg` uses `mp_start_method='fork'` (unavailable on Windows). Empirical Windows behaviour: **UNVERIFIED**; treat as unsupported until tested | [P2] | [pyproject.toml](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/pyproject.toml), [sa2va_qwen3_4b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py) |
| Resolution knobs that drive memory | SAM2 branch: `DirectResize(target_length=1024)` (1008 for the SAM3 variant). MLLM branch: `min_pixels = 512*28*28`, `max_pixels = 2048*28*28` (≤ ~1.6M px ⇒ large activation spikes) | [P2] | [modeling_sa2va_qwen.py (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/modeling_sa2va_qwen.py), [sa2va.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/sa2va.py) |

---

## 9. Is the `[SEG]` hidden state exposed to a downstream module?

| Question | Answer | Tier | Source |
|---|---|---|---|
| Can you obtain `hidden([SEG])` from the implementation? | **Yes, at code level.** Inference path: `generate(..., output_hidden_states=True)` → per-step last-layer hidden states → `get_seg_hidden_states(last_hidden_states, sequences[0][:-1], seg_id=self.seg_token_idx)` selects the rows whose token id is `[SEG]` → `self.text_hidden_fcs(...)` projects them → `SAM2.language_embd_inference(...)`. Both `get_seg_hidden_states` (module-level function) and `text_hidden_fcs` (an `nn.Sequential`) are ordinary callable symbols | [P2] | [modeling_sa2va_qwen.py (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/modeling_sa2va_qwen.py) |
| Training-side equivalent | `seg_token_mask = input_ids == self.seg_token_idx`; `hidden_states = self.text_hidden_fcs(output.hidden_states[-1])`; `pred_embeddings = hidden_states[seg_token_mask]` → the **projected `[SEG]` embedding is the interface to SAM2** | [P2] | [sa2va.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/sa2va.py) |
| Public API return value | `predict_forward` returns only `{'prediction': str, 'prediction_masks': list}` — the `[SEG]` hidden state is **not** returned. A downstream module must re-extract it (re-implement the ~15-line routine) or call `model.model.generate(..., output_hidden_states=True)` itself. No documented hook/callback exists. **[I]** on the "must re-extract" characterisation | [P2]/[I] | [modeling_sa2va_qwen.py (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/modeling_sa2va_qwen.py) |
| Paper-side wording | Repo README states plainly: "its hidden state is projected into SAM-2's prompt space, which decodes the corresponding mask(s)" | [P2] | [projects/sa2va/README.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md) |
| Precedent that downstream modules consume it | VRT (same repo/authors) builds **object-level grounded reasoning on Sa2VA** with its own data loader, configs and eval, and publishes R-Sa2VA-Qwen3VL-4B checkpoints | [P2] | [vrt README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/vrt_sa2va/README.md) |
| SAM3 variant | Keeps `text_hidden_fcs` and the same `Sa2VAChatModelQwen` interface, but swaps `sam2.py` for `sam3.py` + `sam3pkg_*` modules | [P2] | [mirror file list](https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-4B-SAM3) |

---

## 10. Adding a second special token (e.g. `[REF]`)

| Question | Answer | Tier | Source |
|---|---|---|---|
| Is an add-token procedure officially implemented? | **Yes** — the same routine that creates `[SEG]`: `add_special_tokens()` does `tokenizer.add_tokens(special_tokens, special_tokens=True)` and, if any token was added, `model.resize_token_embeddings(len(tokenizer))` (Qwen3VL wrapper) / `model.language_model.resize_token_embeddings(len(tokenizer))` (InternVL wrapper) | [P2] | [qwen3vl.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/mllm/qwen3vl.py), [internvl.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/mllm/internvl.py) |
| Is it config-driven rather than a code edit? | **Yes** — configs declare `special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']`; `Sa2VAModel.__init__` consumes `special_tokens` and calls `_add_special_tokens` (default `['[SEG]']`) | [P2] | [sa2va.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/sa2va.py), [sa2va_qwen3_4b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py) |
| Is "add `[REF]` to a released checkpoint **without retraining**" documented? | **No — UNVERIFIED as a documented procedure.** The mechanism is documented; a no-retrain usage is not. **[I]:** adding the token mechanically is trivial (offline tokenizer edit + `resize_token_embeddings`), but the new embedding row is random, so any *behaviour* keyed on `[REF]` needs at least a short LoRA/embedding fine-tune | [P2] mechanism / [I] judgement | [qwen3vl.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/mllm/qwen3vl.py) |
| Supporting facts | `[SEG]` sits at the end of the added-token list (mirror `added_tokens.json` shows `[SEG]: 151669`, with text `vocab_size = 151674`) → appending tokens is the established pattern; fine-tuning from a converted released checkpoint is a supported flow (`tools/convert_to_pth.py` + `pretrained_pth=`, as used by VRT) | [P2] | [added_tokens.json (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/added_tokens.json), [config.json (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/config.json), [finetune.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/docs/finetune.md) |
| Caveat | Adding a token changes `vocab_size` and requires re-saving the checkpoint, so the released repos cannot be used as-is with `[REF]` | [P2] | [config.json (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/config.json) |
| Expected-use warning | "Expected use" / prohibited-use statements could not be read on any first-party model card (HF unreachable) → **UNVERIFIED**; assume none | [X] | §15 |

---

## 11. Qwen3-VL adaptation: does it exist, and how mature?

| Evidence | Detail | Tier | Source |
|---|---|---|---|
| Official-looking Qwen3-VL checkpoints | `ByteDance/Sa2VA-Qwen3-VL-4B` (mirror `createdAt` 2025-10-21) and `ByteDance/Sa2VA-Qwen3-VL-2B` (2025-11-27), plus `ByteDance/Sa2VA-Qwen3-VL-4B-SAM3` (2026-06-10) | [P2] unverified | [mirror API 4B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-4B), [2B](https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-2B), [SAM3](https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-4B-SAM3) |
| Reported metrics (mirror card) | Qwen3-VL-4B: MME 1660/655, MMBench 86.3, RefCOCO 81.7, RefCOCO+ 77.4, RefCOCOg 80.0, MeVIS 57.1, DAVIS 75.9; Qwen3-VL-2B: 1541/520, 79.0, 80.2, 75.1, 78.5, 53.9, 74.8 — **numbers not independently verified** | [P2] | [2B card (mirror)](https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-2B/raw/main/README.md) |
| Training code for Qwen3-VL | `projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py` with `.env` overrides (`SA2VA_QWEN3VL_PATH`, `SA2VA_BS`, `SA2VA_ACCUM`, `SA2VA_DATA_ROOT`, `SA2VA_INCLUDE_REFSAV`), a dedicated `Qwen3VL` MLLM wrapper using `Qwen3VLProcessor`, and a Qwen3-VL-8B + SAM3 config | [P2] | [sa2va_qwen3_4b.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py), [qwen3vl.py](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/mllm/qwen3vl.py) |
| Repo-level commitment | Root README: Sa2VA "Supports InternVL2.5/3 and Qwen2.5-VL/Qwen3-VL backbones"; env extra `latest` covers "Qwen3-VL, Qwen2.5-VL, InternVL3" | [P2] | [root README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/README.md), [projects/sa2va/README.md](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md) |
| Newest line is Qwen3-VL-based | SAM3 variant uses Qwen3-VL-4B (2026-06-10); VRT / R-Sa2VA uses Qwen3-VL-4B (Dec 2025) | [P2] | [projects/sa2va/README.md news](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md), [vrt README](https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/vrt_sa2va/README.md) |
| Community activity | FP8 requant (`nicehero/Sa2VA-Qwen3-VL-4B-fp8`, 2025-11-18), merges, spaces — **third-party, licence-UNVERIFIED** | [P2] | [mirror search](https://hf-mirror.com/api/models?search=Sa2VA&limit=100) |
| **Maturity verdict [I]** | **"Yes, a Qwen3-VL-based Sa2VA appears to exist — as an unverified release."** Weights/config/wrapper reportedly exist and post-date the Qwen2.5-VL line; the newest official variants migrate to Qwen3-VL. But: none of it is first-party verifiable from this session; the VP (visual-prompt) dataset group is disabled in the Qwen3-VL recipe; Qwen3-VL repos carry wrong `base_model` metadata; the only public full recipe on Qwen3-VL-4B documents a 16-GPU run; no independent third-party reproduction was found. Treat as **research-grade**: usable for a research MVP after you verify the checkpoint on a reachable host, not as a frozen production dependency | [I] | synthesises the rows above |

---

## 12. Licence picture (code vs weights vs data) — all first-party checks failed

| Component | Canonical licence URL needed | Status |
|---|---|---|
| Sa2VA/SAMTok/`Pixel-LLMs` **code** | `https://github.com/bytedance/Sa2VA/blob/main/LICENSE` | **Code licence = `UNVERIFIED / DO NOT USE UNTIL RESOLVED`.** (Apache-2.0 text was *observed through a raw-content proxy*; canonical URL unreachable, so not admissible) |
| All 13 `ByteDance/Sa2VA-*` **weights** | e.g. `https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-4B` | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** (mirror metadata displayed apache-2.0; not admissible) |
| SAMTok weights (`zhouyik/*`) | `https://huggingface.co/zhouyik/Qwen3-VL-8B-SAMTok` | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** |
| SAM2 / SAM2.1 weights (redistributed inside Sa2VA; used by SAMTok) | `https://huggingface.co/facebook/sam2-hiera-large`, `.../sam2.1-hiera-large`; Meta project page | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** (`ai.meta.com` unusable; HF unreachable) |
| SAM3 weights (Sa2VA-Qwen3-VL-4B-SAM3) | `https://huggingface.co/facebook/sam3` | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** — mirror indicated `license: other` + gated; treat as the highest licensing risk in the family |
| **Base-model** weights of each Sa2VA variant (separate licence layer!) | `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`, `OpenGVLab/InternVL3-8B`, `OpenGVLab/InternVL2_5-4B`, `llava-hf/llava-1.5-7b-hf` | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** for all. Mirror metadata *indicated* apache-2.0 / `qwen-research` (Qwen2.5-VL) / MIT (InternVL2.5) / `llama2` (LLaVA-1.5) — **not admissible**; the **different-base-licence-per-variant risk is real and must be re-checked first-hand** |
| Third-party derivatives (`nicehero/...-fp8`, `HarborYuan/R-Sa2VA-*`, `kumuji/Sa2VA-i-*`, `ly17/*`) | their own cards | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** |
| Training data `Dense-World/Sa2VA-Training` | `https://huggingface.co/datasets/Dense-World/Sa2VA-Training` | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** (mirror card showed **no licence field**) |
| Eval data `Dense-World/Sa2VA-Eval` (Ref-SAV eval) | `https://huggingface.co/datasets/Dense-World/Sa2VA-Eval` | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** (no licence field mirrored) |
| Ref-SAV dataset itself | — | **UNVERIFIED** (no card/licence located; repo describes it as auto-labelled from Meta SA-V) |
| Meta SA-V (`sam_v_full`) | Meta dataset page | **UNVERIFIED** first-hand; repo states it must be obtained from Meta under **their** licence and is not bundled |
| Third-party data inside the recipe (RefCOCO/+/g, LLaVA-665k, ReVOS, MeViS, RefYTVOS, Chat-UniVi, GranDf, Flickr30k, OpenPSG, Osprey-724k) | each upstream | each retains its own licence; the bundle declares none → per-component first-party verification required |
| VRT data (`HarborYuan/VisualReasoningTracer`, `VRT-Eval`) | HF datasets | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** |

---

## 13. Engineering effort / risk for a local MVP on ONE 16 GB-VRAM laptop GPU

> **All of §13 is [I] my judgement/estimate**, anchored to the [P1]/[P2] facts cited above. No benchmark was run; nothing here is a measured result. Licence gates in §12 apply on top.

### 13.1 Sizing (from [P2] mirror metadata — indicative only)

| Candidate | Reported weight footprint | [I] 16 GB feasibility |
|---|---|---|
| `Sa2VA-Qwen3-VL-2B` | 10.68 GB (F32) | Load in bf16 (~5.3 GB [I]) → the most headroom of any candidate |
| `Sa2VA-Qwen3-VL-4B` | 20.24 GB (F32) | Does **not** fit as shipped; bf16 downcast (~10.1 GB [I]) + activations for 1024-px SAM2 input and ≤1.6M-px MLLM input → tight, OOM plausible without lowering resolution |
| `nicehero/Sa2VA-Qwen3-VL-4B-fp8` | ≈5.07 GB | Fits easily — but third-party, licence-UNVERIFIED quantisation |
| `Sa2VA-Qwen2_5-VL-3B` | 17.19 GB (F32) | Same as 4B, plus a non-Apache base-model licence to resolve |
| 7B/8B/14B/26B/LLaVA variants | 29–92 GB | Out of scope for 16 GB |

### 13.2 (a) Adapt an existing Sa2VA checkpoint + add a second special token

| Dimension | Documented basis | [I] effort / risk |
|---|---|---|
| Code maturity | Monorepo with official training + eval + conversion scripts, fine-tune guide, per-backbone configs, a Qwen3-VL config, plus a worked downstream example (VRT) | **Low** integration risk — you start from a working `[SEG]` pipeline |
| Training code | Public (xtuner/mmengine + PEFT LoRA); VRT's config is a near-template for "new task on top of Sa2VA" | **Low–medium** |
| Adding `[REF]` | `special_tokens` + `add_tokens` + `resize_token_embeddings` already implemented | **Low** mechanically; **[I]** semantics still need data + a short LoRA/embedding fine-tune |
| Stages / data scale | Single stage; full reproduction needs 8–32 GPUs and the (licence-unresolved) 418 GB training bundle + separate Meta SA-V; the *fine-tune* path is explicitly small (RefCOCO example, `repeats=100`) | **Low** for a small in-house adaptation set; **very high** if you attempt the published recipe |
| 16 GB budget | No documented VRAM figures; official recipes assume ≥8 A100 | **[I] estimate:** inference of the 2B (bf16) is realistic; *training* even a LoRA adapter for a 4B MLLM + 1024-px SAM2 mask decoder on 16 GB is borderline (gradient checkpointing, bs 1, LoRA on a layer subset, possibly offload). The 2B checkpoint is the realistic 16 GB training target |
| Residual risks | Unverifiable release/licences, HF metadata bugs, monorepo path churn, no Windows path documented (`flash-attn` build, `fork`), CUDA-hardcoded autocast, no VRAM budget | **Medium** — environment/wiring, not science |

### 13.3 (b) Build a LISA-style `[SEG]` pipeline yourself on a base Qwen3-VL/Qwen2.5-VL + SAM2

| Dimension | Documented basis | [I] effort / risk |
|---|---|---|
| What you must build | `[SEG]` token; LLM-hidden→SAM2-prompt projection head; mask/dice loss path; mask-aware data pipeline; eval harness. Sa2VA's own `sa2va.py` shows the core is ~a projection MLP + `hidden_states[seg_mask]` + a SAM2 decoder forward + Dice/CE losses → not exotic | **Medium** (weeks for one competent engineer, [I]) |
| Code maturity | LISA (CVPR 2024, arXiv:2308.00692) is the canonical published recipe; you re-derive rather than reuse a maintained monorepo | **Medium–high** — no upstream fixes; Qwen3-VL-specific plumbing (processor, M-RoPE, deepstack visual indexes) is yours |
| Training code | Your own; Sa2VA's structure can be imitated (its code is *indicated* Apache-2.0 but currently licence-unverified — do not copy until resolved) | **Medium** |
| Stages / data | Single stage if you copy Sa2VA's structure; the real cost is data: RefCOCO/+/g + GCG + video sets, and the Sa2VA training bundle has **no declared licence** → re-source per dataset | **High** on data licensing/plumbing |
| 16 GB budget | Same wall as (a), and you own every OOM | **Medium–high** |
| Licensing | Fully yours; base checkpoints are the standard Qwen3-VL + SAM2 pair (still to be verified first-hand) | **Low** once verified |
| Upside | Total control of the `[REF]`-style token, projection and training signal; no dependency on an unverifiable release | **[I]** main argument for (b) |

### 13.4 [I] Verdict for BuildReasonSeg on one 16 GB laptop GPU

1. **No 16 GB *training* run at published scale is possible either way** — the recipe is a 128-effective-batch, 16–32-GPU, ~418 GB-data run ([P2], §6).
2. **(a) is the lower-risk MVP path *if* the MVP is "get grounded masks + `[SEG]` spatial embeddings flowing locally"**: start from the smallest Qwen3-VL Sa2VA candidate (2B, ~5.3 GB bf16 [I]) or a bf16 4B (~10.1 GB [I]) for inference, consume `[SEG]` hidden states by re-using the extraction routine (§9), then add `[REF]` via the documented `special_tokens` mechanism and fine-tune with LoRA on a small in-house set (VRT's config is a proven template). Hazards: Windows/CUDA toolchain, FP32-shipped weights, no VRAM budget, and **unverified release + licences**.
3. **(b) is lower-risk for licensing and control, higher-risk for schedule** — it is also the only path that gives you a fully self-owned `[REF]`/`[SEG]` design without inheriting a monorepo whose paths and metadata already shifted once.
4. **Hybrid [I] recommendation:** use Sa2VA as *reference implementation + (once verified) weights*, but write the token/projection/training code yourself against a base Qwen3-VL checkpoint — i.e. take (a)'s head start on runtime, (b)'s ownership of the training path. If the requirement is "mask as discrete language token" rather than "one continuous `[SEG]` embedding", evaluate the SAMTok line instead.
5. **Do not** start from the Qwen2.5-VL variants or the SAM3 variant before their base/SAM3 licences are checked first-hand.

---

## 14. Explicit UNVERIFIED list

| # | Item | Status |
|---|---|---|
| 1 | Sa2VA release: exact IDs, file lists, sizes, dtypes, shard counts (mirror-only evidence) | **UNVERIFIED** (tier [P2]) — verify on a reachable first-party host |
| 2 | Licence of *every* Sa2VA/SAMTok/SAM2/SAM3/base-model/dataset artefact | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** |
| 3 | Repo code licence (canonical GitHub `LICENSE` unreachable) | **`UNVERIFIED / DO NOT USE UNTIL RESOLVED`** (Apache-2.0 text observed via proxy) |
| 4 | Sa2VA paper §Implementation Details (paper-stated GPUs/epochs/hours) | **UNVERIFIED** (fetch truncation; PDF not readable here) |
| 5 | Documented inference VRAM requirement / recommended inference GPU | **UNVERIFIED** |
| 6 | Windows support (CUDA) — no documentation; config evidence points to unsupported | **UNVERIFIED** (empirically untested) |
| 7 | Non-FlashAttention inference fallback | **UNVERIFIED** |
| 8 | Per-checkpoint licences of the 13 SAMTok models (only the list itself was seen) | **UNVERIFIED** |
| 9 | ByteDance Seed first-party project/model page for Sa2VA/SAMTok | **not found** (`seed.bytedance.com`) |
| 10 | ModelScope mirror of any Sa2VA artefact | **not found — 404** for 5 probed IDs (4 by parent, 1 by me) |
| 11 | One-shot / in-context segmentation support | **UNVERIFIED** (no doc/benchmark) |
| 12 | `[REF]`-style second token without retraining | **UNVERIFIED** (mechanism documented; procedure not) |
| 13 | "Expected use" / prohibited-use statements on any Sa2VA card | **UNVERIFIED** (cards unreachable) |
| 14 | Reported Sa2VA metrics (RefCOCO/MeVIS/DAVIS/MME/MMBench) | **UNVERIFIED** (mirror cards only) |

---

## 15. URL not reachable at research time (2026-09-26)

Every URL below is a first-party source I needed and could **not** fetch. Per policy: the corresponding licence/fact is `UNVERIFIED / DO NOT USE UNTIL RESOLVED`, and I did **not** substitute a snippet, mirror, blog or aggregator value.

| Needed source | URL | Error observed |
|---|---|---|
| Official Sa2VA repo (root) | https://github.com/bytedance/Sa2VA | hostname resolves to non-public IP (blackholed) |
| Official Sa2VA repo (legacy org) | https://github.com/magic-research/Sa2VA | same |
| Repo `LICENSE` | https://github.com/bytedance/Sa2VA/blob/main/LICENSE | same |
| Repo raw `LICENSE` | https://raw.githubusercontent.com/bytedance/Sa2VA/main/LICENSE | same |
| Repo raw READMEs (root, `projects/sa2va`, `projects/samtok`, `projects/vrt_sa2va`) | https://raw.githubusercontent.com/bytedance/Sa2VA/main/README.md (+ project paths) | same |
| Repo raw configs/models/scripts | https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/… , `…/models/…`, `…/evaluation/…`, `…/docs/finetune.md`, `…/pyproject.toml` | same |
| GitHub API (repo metadata, file tree) | https://api.github.com/repos/bytedance/Sa2VA , `…/git/trees/main?recursive=1` | same |
| HF model cards / files — all 13 Sa2VA checkpoints | https://huggingface.co/ByteDance/Sa2VA-1B , `-4B`, `-8B`, `-26B`, `-InternVL3-2B`, `-InternVL3-8B`, `-InternVL3-14B`, `-Qwen2_5-VL-3B`, `-Qwen2_5-VL-7B`, `-Qwen3-VL-2B`, `-Qwen3-VL-4B`, `-Qwen3-VL-4B-SAM3`, `-LLaVA-1.5-7B` | hostname resolves to non-public IP |
| HF API endpoints for those models (`/api/models/…`) | https://huggingface.co/api/models/ByteDance/Sa2VA-Qwen3-VL-4B (etc.) | same |
| HF collection | https://huggingface.co/collections/ByteDance/sa2va-model-zoo …-677e3084d71b5f108d00e093 | same |
| SAM2 model cards | https://huggingface.co/facebook/sam2-hiera-large , https://huggingface.co/facebook/sam2.1-hiera-large | same |
| SAM3 model card + licence | https://huggingface.co/facebook/sam3 | same (mirror indicated gated / `license: other`) |
| SAM2/SAM3 first-party project page | https://ai.meta.com/sam2/ | fetch failed |
| Base-model cards | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct , `…/Qwen/Qwen2.5-VL-3B-Instruct`, `…/OpenGVLab/InternVL3-8B`, `…/OpenGVLab/InternVL2_5-4B`, `…/llava-hf/llava-1.5-7b-hf` | same |
| Dataset cards | https://huggingface.co/datasets/Dense-World/Sa2VA-Training , `…/Dense-World/Sa2VA-Eval`, `…/bitersun/Sa2VA-finetune-example`, `…/HarborYuan/VisualReasoningTracer`, `…/HarborYuan/VRT-Eval` | same |
| SAMTok checkpoint cards | https://huggingface.co/zhouyik/Qwen3-VL-8B-SAMTok (and the other 12) | same |
| Community derivative cards | https://huggingface.co/nicehero/Sa2VA-Qwen3-VL-4B-fp8 , `…/HarborYuan/R-Sa2VA-Qwen3VL-4B-SFT`, `…/HarborYuan/R-Sa2VA-Qwen3VL-4B-RL` | same |
| `hf.co` short links used in paper/repo | https://hf.co/collections/ByteDance/sa2va-model-zoo | cross-origin redirect refused / host blackholed |
| ByteDance Seed first-party site (no Sa2VA page found) | https://seed.bytedance.com/ | no matching project page located by search |
| ModelScope (no Sa2VA copy found) | https://www.modelscope.cn/api/v1/models/ByteDance/Sa2VA-Qwen3-VL-2B | **HTTP 404** `{"Code":10010205001,…record not found}` — Sa2VA absent; parent agent got the same 404 for 4 other IDs |
| CDN/proxy alternatives attempted | cdn.jsdelivr.net, data.jsdelivr.com, raw.githack.com, r.jina.ai, www.huggingface.co | fetch failed / 403 / not usable |

**Intermediaries that did work (tier [P2], non-admissible for licence):** `ghproxy.net` (raw content of `raw.githubusercontent.com`) and `hf-mirror.com` (HF API + card mirror). Real project pages `lxtgh.github.io/project/sa2va` and `zhouyiks.github.io/projects/SAMTok/` fetched successfully in this session.

---

## 16. Source list (deduplicated)

Legend: **[P1]** first-party fetched; **[P2]** official content via third-party intermediary (facts indicative, licences inadmissible); **[X]** unreachable; **[S]** third-party/derivative.

| Title | Exact URL | Type | Accessed | Claim it supports |
|---|---|---|---|---|
| Sa2VA arXiv abstract (v1 2025-01-07 → v4 2026-08-24; TPAMI 2026; DOI 10.1109/TPAMI.2026.3720379) | https://arxiv.org/abs/2501.04001 | [P1] official paper | 2026-09-26 | title, authors, venue/status, dates, Ref-SAV 72k/2k, "single-stage", SAM-2 + MLLM design, arXiv licence |
| Sa2VA full text (arXiv HTML v4) | https://arxiv.org/html/2501.04001v4 | [P1] official paper | 2026-09-26 | affiliation list, capability Table 1 (RES/Ref-VOS/Inter-VOS/GCG/…), model-zoo link |
| Sa2VA full text (ar5iv v1 render) | https://ar5iv.labs.arxiv.org/html/2501.04001 | [P1] official paper render | 2026-09-26 | "SAM-2's decoder and memory module are frozen" (paper wording), capability table |
| SAMTok CVPR 2026 published version (pp. 37852–37863) | https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_SAMTok_Representing_Any_Mask_with_Two_Words_CVPR_2026_paper.html | [P1] published paper | 2026-09-26 | SAMTok venue, authors, 2-token tokenizer, 209M masks, task list, RL reward |
| SAMTok arXiv abstract | https://arxiv.org/abs/2601.16093 | [P1] official paper | 2026-09-26 | dates, "CVPR 2026 Highlight" comment, affiliation line |
| Sa2VA project page | https://lxtgh.github.io/project/sa2va | [P1] official project page | 2026-09-26 | `[SEG]`→SAM-2 description, task list, stale `magic-research` link, CCA-SA-4.0 site licence |
| SAMTok project page | https://zhouyiks.github.io/projects/SAMTok/ | [P1] official project page | 2026-09-26 | SAMTok architecture figure/description, training-code link, citation |
| Sa2VA monorepo root README | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/README.md | [P2] official repo file via proxy | 2026-09-26 | monorepo projects, backbone support, uv env |
| Sa2VA project README | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/README.md | [P2] same | 2026-09-26 | model-zoo table, news dates, `[SEG]` mechanism, prerequisites, dataset layout, "at least 8 A100", eval/convert/fine-tune commands |
| Repo LICENSE | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/LICENSE | [P2] same (licence inadmissible) | 2026-09-26 | Apache-2.0 *text observed*; canonical URL unreachable → licence stays UNVERIFIED |
| `projects/sa2va/pyproject.toml` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/pyproject.toml | [P2] same | 2026-09-26 | transformers 4.57.1/4.49.0 extras, `flash-attn==2.7.3`, `torch>=2.6.0`, py3.11, non-Linux CPU torch index |
| `projects/sa2va/demo/requirements.txt` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/demo/requirements.txt | [P2] same | 2026-09-26 | legacy inference pins (torch 2.3.1 / transformers 4.42.3 / flash_attn) |
| `projects/sa2va/docs/finetune.md` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/docs/finetune.md | [P2] same | 2026-09-26 | official fine-tune procedure, data format, hf→pth conversion, 8-GPU command |
| `configs/sa2va_in30_8b.py` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_in30_8b.py | [P2] same | 2026-09-26 | one-stage recipe, freeze flags, LoRA r=256, `special_tokens`, dataset list, batching |
| `configs/sa2va_qwenvl3/sa2va_qwen3_4b.py` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_4b.py | [P2] same | 2026-09-26 | Qwen3-VL training recipe, effective batch 128 = 1×8×16 GPUs, VP data disabled, `.env` overrides |
| `configs/sa2va_qwenvl3/sa2va_qwen3_8b_sam3.py` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_qwenvl3/sa2va_qwen3_8b_sam3.py | [P2] same | 2026-09-26 | SAM3 grounding encoder, input 1008, 32-GPU batch recipe, `sam3.pt` requirement |
| `configs/sa2va_finetune.py` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/configs/sa2va_finetune.py | [P2] same | 2026-09-26 | fine-tune LoRA r=128, `Qwen3VLProcessor` support, `pretrained_pth` flow |
| `projects/sa2va/models/sa2va.py` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/sa2va.py | [P2] same | 2026-09-26 | `[SEG]` hidden-state extraction, `text_hidden_fcs`, SAM2 mask-decoder-only trainable, saved state-dict scope |
| `projects/sa2va/models/mllm/qwen3vl.py` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/mllm/qwen3vl.py | [P2] same | 2026-09-26 | `add_tokens` + `resize_token_embeddings`, hardcoded `flash_attention_2`, LoRA target selection |
| `projects/sa2va/models/mllm/internvl.py` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/models/mllm/internvl.py | [P2] same | 2026-09-26 | same add-token mechanism for InternVL; visual-prompt embedding path |
| `projects/sa2va/evaluation/run_all_evals.py` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/sa2va/evaluation/run_all_evals.py | [P2] same | 2026-09-26 | official eval surface = RefCOCO/+/g, GCG, RefVOS (DAVIS, MEVIS_U) |
| `projects/samtok/README.md` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/samtok/README.md | [P2] same | 2026-09-26 | SAMTok model zoo, release checkboxes, `VQ_SAM2` + `sam2.1_hiera_large.pt`, mask-token API |
| `projects/vrt_sa2va/README.md` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/vrt_sa2va/README.md | [P2] same | 2026-09-26 | VRT built on Sa2VA, R-Sa2VA Qwen3VL-4B SFT/RL, 8-GPU fine-tune command, VRT-Bench results |
| `configs_sa2va/vrt_sa2va_4b_qwen3_sft.py` | https://ghproxy.net/https://raw.githubusercontent.com/bytedance/Sa2VA/main/projects/vrt_sa2va/configs_sa2va/vrt_sa2va_4b_qwen3_sft.py | [P2] same | 2026-09-26 | downstream fine-tune template (LoRA r=128, converted Sa2VA-Qwen3-VL-4B init) |
| HF mirror API — `ByteDance/Sa2VA-Qwen3-VL-4B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-4B | [P2] mirror of official HF API | 2026-09-26 | unverified params 5.057B/F32, 5 shards, ~20.24 GB, `base_model` metadata bug |
| HF mirror API — `…/Sa2VA-Qwen3-VL-2B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-2B | [P2] same | 2026-09-26 | unverified 2.667B, ~10.68 GB, createdAt 2025-11-27 |
| HF mirror API — `…/Sa2VA-Qwen2_5-VL-3B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen2_5-VL-3B | [P2] same | 2026-09-26 | unverified 4.294B, ~17.19 GB |
| HF mirror API — `…/Sa2VA-Qwen2_5-VL-7B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen2_5-VL-7B | [P2] same | 2026-09-26 | unverified 8.528B, ~34.12 GB |
| HF mirror API — `…/Sa2VA-1B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-1B | [P2] same | 2026-09-26 | unverified 1.164B (F32+BF16 304M), ~7.55 GB, InternVL2.5 base |
| HF mirror API — `…/Sa2VA-4B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-4B | [P2] same | 2026-09-26 | unverified 3.942B, ~15.16 GB |
| HF mirror API — `…/Sa2VA-8B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-8B | [P2] same | 2026-09-26 | unverified 8.318B, ~32.66 GB |
| HF mirror API — `…/Sa2VA-26B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-26B | [P2] same | 2026-09-26 | unverified 25.778B, ~92.04 GB |
| HF mirror API — `…/Sa2VA-InternVL3-2B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-InternVL3-2B | [P2] same | 2026-09-26 | unverified 2.316B, ~8.67 GB |
| HF mirror API — `…/Sa2VA-InternVL3-8B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-InternVL3-8B | [P2] same | 2026-09-26 | unverified 8.183B, ~32.13 GB |
| HF mirror API — `…/Sa2VA-InternVL3-14B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-InternVL3-14B | [P2] same | 2026-09-26 | unverified 15.369B, ~60.88 GB |
| HF mirror API — `…/Sa2VA-Qwen3-VL-4B-SAM3` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-Qwen3-VL-4B-SAM3 | [P2] same | 2026-09-26 | unverified 5.306B, ~21.24 GB, SAM3 file set |
| HF mirror API — `…/Sa2VA-LLaVA-1.5-7B` | https://hf-mirror.com/api/models/ByteDance/Sa2VA-LLaVA-1.5-7B | [P2] same | 2026-09-26 | unverified 7.305B, ~29.22 GB, `Sa2VAChatModelLlava` |
| HF mirror search (author=ByteDance, q=Sa2VA) | https://hf-mirror.com/api/models?author=ByteDance&search=Sa2VA&limit=100 | [P2] same | 2026-09-26 | the 13-repo inventory (existence, unverified) |
| HF mirror search (q=Sa2VA, any author) | https://hf-mirror.com/api/models?search=Sa2VA&limit=100 | [P2]/[S] same + derivatives | 2026-09-26 | `Dense-World/*` legacy uploads, fp8 quant, R-Sa2VA, Sa2VA-i, vessel fine-tunes |
| HF mirror collection "Sa2VA Model Zoo" | https://hf-mirror.com/collections/ByteDance/sa2va-model-zoo | [P2] mirror of official collection | 2026-09-26 | official model-zoo membership, collection updated 2025-11-27 |
| Sa2VA-Qwen3-VL-4B card (mirror) | https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/README.md | [P2] mirror of official card | 2026-09-26 | family table, quick-start API, `use_flash_attn=True`, `torch_dtype=bfloat16` |
| Sa2VA-Qwen3-VL-2B card (mirror) | https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-2B/raw/main/README.md | [P2] same | 2026-09-26 | Qwen3-VL-2B metrics + family table |
| Sa2VA-8B card (mirror) | https://hf-mirror.com/ByteDance/Sa2VA-8B/raw/main/README.md | [P2] same | 2026-09-26 | InternVL2.5 family table, legacy repo link |
| `config.json` (mirror) | https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/config.json | [P2] mirror of official file | 2026-09-26 | `dtype float32`, `vocab_size 151674`, Qwen3-VL text/vision config, `transformers_version 4.57.0` |
| `added_tokens.json` (mirror) | https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/added_tokens.json | [P2] same | 2026-09-26 | `[SEG]` = 151669 → tokens appended |
| `model.safetensors.index.json` (mirror) | https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/model.safetensors.index.json | [P2] same | 2026-09-26 | SAM2 tensors redistributed in-checkpoint (`grounding_encoder.sam2_model.*`), `total_size` 20,228,290,760 B |
| `modeling_sa2va_qwen.py` (mirror) | https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/modeling_sa2va_qwen.py | [P2] same | 2026-09-26 | `[SEG]` hidden-state extraction, `text_hidden_fcs`, SAM2 prompt injection, min/max pixels, absence of SAMTok tokens |
| `sam2.py` (mirror) | https://hf-mirror.com/ByteDance/Sa2VA-Qwen3-VL-4B/raw/main/sam2.py | [P2] same | 2026-09-26 | SAM2 Hiera-L architecture, pure-PyTorch, CUDA-only autocast inference |
| HF mirror API — `facebook/sam2-hiera-large` | https://hf-mirror.com/api/models/facebook/sam2-hiera-large | [P2] mirror of official API | 2026-09-26 | SAM2.0 Hiera-L existence + 224.43M params (**licence inadmissible**) |
| HF mirror API — `facebook/sam2.1-hiera-large` | https://hf-mirror.com/api/models/facebook/sam2.1-hiera-large | [P2] same | 2026-09-26 | SAM2.1 Hiera-L existence (used by SAMTok) |
| HF mirror API — `facebook/sam3` | https://hf-mirror.com/api/models/facebook/sam3 | [P2] same | 2026-09-26 | SAM3 marked `license: other` + gated (mirror only → risk flag) |
| HF mirror API — `Qwen/Qwen3-VL-4B-Instruct` | https://hf-mirror.com/api/models/Qwen/Qwen3-VL-4B-Instruct | [P2] same | 2026-09-26 | base-model identity/size (licence inadmissible) |
| HF mirror API — `Qwen/Qwen2.5-VL-3B-Instruct` | https://hf-mirror.com/api/models/Qwen/Qwen2.5-VL-3B-Instruct | [P2] same | 2026-09-26 | base-model identity (mirror indicated `qwen-research` — inadmissible) |
| HF mirror API — `OpenGVLab/InternVL3-8B` | https://hf-mirror.com/api/models/OpenGVLab/InternVL3-8B | [P2] same | 2026-09-26 | base-model identity/size |
| HF mirror API — `OpenGVLab/InternVL2_5-4B` | https://hf-mirror.com/api/models/OpenGVLab/InternVL2_5-4B | [P2] same | 2026-09-26 | base-model identity (mirror indicated MIT — inadmissible) |
| HF mirror API — `llava-hf/llava-1.5-7b-hf` | https://hf-mirror.com/api/models/llava-hf/llava-1.5-7b-hf | [P2] same | 2026-09-26 | LLaVA-1.5 base identity (mirror indicated `llama2` — inadmissible) |
| HF mirror API — dataset `Dense-World/Sa2VA-Training` | https://hf-mirror.com/api/datasets/Dense-World/Sa2VA-Training | [P2] same | 2026-09-26 | training bundle composition, 418.8 GB, **no licence field mirrored** |
| HF mirror API — dataset `Dense-World/Sa2VA-Eval` | https://hf-mirror.com/api/datasets/Dense-World/Sa2VA-Eval | [P2] same | 2026-09-26 | Ref-SAV eval set, **no licence field mirrored** |
| HF mirror API — `zhouyik/Qwen3-VL-8B-SAMTok` | https://hf-mirror.com/api/models/zhouyik/Qwen3-VL-8B-SAMTok | [P2]/[S] same | 2026-09-26 | SAMTok checkpoint composition (`sam2.1_hiera_large.pt`, `mask_tokenizer_256x2.pth`) |
| HF mirror API — `nicehero/Sa2VA-Qwen3-VL-4B-fp8` | https://hf-mirror.com/api/models/nicehero/Sa2VA-Qwen3-VL-4B-fp8 | [P2]/[S] same | 2026-09-26 | community FP8 quantisation, ~5.07 GB, no licence |
| HF mirror API — `HarborYuan/R-Sa2VA-Qwen3VL-4B-SFT` | https://hf-mirror.com/api/models/HarborYuan/R-Sa2VA-Qwen3VL-4B-SFT | [P2]/[S] same | 2026-09-26 | VRT downstream derivative, no licence |
| ModelScope API probe — `ByteDance/Sa2VA-Qwen3-VL-2B` | https://www.modelscope.cn/api/v1/models/ByteDance/Sa2VA-Qwen3-VL-2B | [P1] first-party host, negative result | 2026-09-26 | **404 record not found** → Sa2VA not mirrored on the reachable first-party host |
| ModelScope model search page | https://www.modelscope.cn/models?name=Sa2VA | [P1] first-party host, negative result | 2026-09-26 | no Sa2VA listing surfaced (JS-rendered shell) |
| LISA (CVPR 2024) — publisher page | https://ieeexplore.ieee.org/document/10658574 | [P1] publisher metadata (paper also at arXiv:2308.00692) | 2026-09-26 | LISA as the canonical LISA-style `[SEG]` reference used in §13 |

**Source count:** 60 distinct URLs = 9 first-party fetched successfully ([P1]) + 1 first-party negative probe (ModelScope) + 46 official-content-via-intermediary ([P2], licence-inadmissible) + 3 community/derivative ([S]) + the unreachable set listed in §15 (26 canonical first-party URLs, counted once each; not re-listed here).
