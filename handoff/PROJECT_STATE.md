# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 5.5._

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

The block above is machine-checked against
`evaluation/build_spatial_reason_artifact_index.json` by
`scripts/check_artifact_consistency.py`. Do not hand-edit numbers anywhere else.

## Identity

| Field | Value |
|---|---|
| Project codename | **BuildReasonSeg** |
| Repository | `Autumn-Preface/BuildReasonSeg` (branch `main`) |
| Legacy evidence (read-only) | `../WHU_Building_Segment/` |
| Reasoning dataset | **BuildSpatialReason v0.1.1** (frozen, audited, verdict PASS) |
| Legacy dataset version | **v0.1** (frozen, superseded — never use for training decisions) |
| Semantic visibility policy | **1.0** (`spatial_reasoning/semantic_policy.py`) |
| Independent audit oracle | **1.0** (`spatial_reasoning/semantic_oracle.py`) |
| MVP stack | **ADR-012** — Qwen3-VL-4B-Instruct + SAM 2.1 hiera-large + BF16 LoRA `[SEG]` |

## Completed tasks

| Task | Scope | Result |
|---|---|---|
| 1 | Project foundation + baseline freeze | done |
| 2 | Polygon → building component representation | done (4,038 maps, 36,926 components, 0 conflicts) |
| 3A | Geometry statistics + relation engine + thresholds | done |
| 3B | Relation semantics correction + freeze | done |
| Naming | `SpatialReasoningSeg` → `BuildReasonSeg` | done |
| 4 | BuildSpatialReason-v0.1 dataset generator | done (32,284 records) |
| 5 | v0.1 validator + semantic quality audit | done → `FAIL_REQUIRES_REVISION` |
| 5B | v0.1.1 corrective regeneration + acceptance audit | done → `PASS` |
| 5C | Acceptance hardening + artifact consistency | done → `PASS` |
| 5.5 | **External research, model-stack verification, MVP design freeze** | **done → COMPLETE (ADR-012)** |

## MVP stack (ADR-012)

| Item | Choice |
|---|---|
| Base MLLM | `Qwen/Qwen3-VL-4B-Instruct` (apache-2.0 verified), Stages 0–2 on the 2B sibling |
| Mask decoder | `facebook/sam2.1-hiera-large` (Apache-2.0, LICENSE read), encoder + memory frozen |
| Pathway | LISA-style `[SEG]` for the MVP; `[REF] + [SEG]` from Stage 3 |
| Trainable | LoRA r=16 on `q,k,v,o,gate,up,down`, `[SEG]`/`[REF]` embeddings, projection MLP, SAM2 mask decoder |
| Frozen | vision tower, LLM base weights, SAM2 image encoder + memory |
| Precision | BF16 LoRA; 4-bit QLoRA is the OOM fallback (`sm_120` + bnb on Windows is `PROVISIONAL`) |
| Environment | native Windows 11, **new dedicated conda env**; SDPA (no FlashAttention); WSL2 is the fallback |
| Images | native 512×512, 256 visual tokens/tile, pixel budget set explicitly |
| External training data | **none** |

Excluded on evidence: `Qwen2.5-VL-3B-Instruct` (weight licence unverifiable),
`Qwen2.5-VL-7B` BF16 (exceeds VRAM), Sa2VA-Qwen3-VL-4B as a fine-tuning target (18.84 GiB),
`Qwen3-VL-4B-Thinking` for the MVP.

## Measured environment facts

RTX 5080 Laptop GPU, compute capability **(12, 0)** = `sm_120`, **15.89 GiB** VRAM;
`torch 2.13.0+cu132` with `sm_120` in `get_arch_list()` and `cuda.is_available() == True` — all measured
locally. This closes the Blackwell risk **natively on Windows**.

## Novelty position (Task 5.5 audit)

`[SEG]` is ubiquitous; a `[REF]` reference token is **already published** (SegLLM arXiv 2410.18923;
PSALM ECCV 2024); instruction-tuned building analysis is published prior art (ISPRS Annals XI-2-2026).
**Retained contributions:** geometry-verifiable relation supervision; an explicit **Spatial Relation
Encoder over the model's own predicted geometry**; a **relation-level Spatial Consistency Loss**.
**Most threatening work:** SegLLM.

Dataset position: **no public remote-sensing dataset was verified to combine instance-level building
masks with multi-hop spatial instructions** (28 datasets surveyed) — an evidenced absence, phrased as
such.

## Current blockers

**None.** Non-blocking risks carried forward:

1. A local accelerator (Steam++ / Watt Toolkit) blackholes `github.com` / `huggingface.co` in the hosts
   file while proxying them; **weight acquisition must be confirmed at Task 6 Stage 0** (ModelScope is
   the documented alternative for Qwen).
2. `bitsandbytes` + `sm_120` + Windows is undocumented → the QLoRA fallback is `PROVISIONAL`.
3. All VRAM figures are estimates until Stage 0–2 measure them.
4. Sa2VA repository code licence unresolved (used as a recipe, not a dependency).
5. 22 of 28 external datasets remain licence-`UNVERIFIED` → not used for training.
6. `scene_level_split_leakage = unverified` in v0.1.1.

## Recommended next task

**ChatGPT review of the Task 5.5 research package**, then **Task 6 (MVP implementation)** as a separate
task, starting from `evaluation/task5_5_stack_decision.json` and
`docs/research/task5_5_mvp_decision.md`. Task 6 must pass the entry criteria in §2.2 of that document
before writing implementation code.

Full detail: `handoff/FROM_DSH.md`, `docs/research/task5_5_*.md`,
`evaluation/task5_5_stack_decision.json`.
