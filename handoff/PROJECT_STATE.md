# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 6A._

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
| Reasoning dataset | **BuildSpatialReason v0.1.1** (frozen, audited, PASS) |
| Legacy dataset version | **v0.1** (frozen, superseded — never use for training) |
| MVP environment | **`.conda/buildreasonseg-mvp`** (conda, Python 3.11.16, PyTorch 2.13.0+cu132) |
| MVP stack (measured) | **Qwen3-VL-2B-Instruct + SAM2.1 Hiera Base+ + `[SEG]`** (ADR-013) |
| Design stack (unmeasured) | Qwen3-VL-4B-Instruct + SAM 2.1 hiera-large (ADR-012) |

## Completed tasks

| Task | Scope | Result |
|---|---|---|
| 1 | Project foundation + baseline freeze | done |
| 2 | Polygon → building component representation | done (4,038 maps, 36,926 components) |
| 3A | Geometry statistics + relation engine + thresholds | done |
| 3B | Relation semantics correction + freeze | done |
| Naming | `SpatialReasoningSeg` → `BuildReasonSeg` | done |
| 4 | BuildSpatialReason-v0.1 dataset generator | done (32,284 records) |
| 5 | v0.1 validator + semantic quality audit | done → `FAIL_REQUIRES_REVISION` |
| 5B | v0.1.1 corrective regeneration + acceptance audit | done → `PASS` |
| 5C | Acceptance hardening + artifact consistency | done → `PASS` |
| 5.5 | External research, model-stack verification, MVP design freeze | done → ADR-012 |
| 6A | **Native-Windows env bootstrap + 2B `[SEG]` MVP smoke/overfit** | **done → `PASS`** |

## Task 6A measured results

| Metric | Value |
|---|---|
| Environment created (`conda create --prefix`) | ✅ `.conda/buildreasonseg-mvp`, existing envs untouched |
| Qwen3-VL-2B + SAM2.1 Base+ load | ✅ 4.53 GiB allocated / 5.10 GiB reserved |
| Visual tokens per 512×512 tile (measured) | **256** |
| `[SEG]` token id | **151,669** (single token, tying preserved) |
| Trainable parameters | 24,010,309 of 2,227,632,642 (1.08 %) |
| Stage 1 (2-sample forward/backward) | ✅ all 9 checks; peak **7.25 GiB** |
| Stage 2 (20-sample overfit, 2,000 steps) | ✅ min **0.9366** / mean **0.9807** training mIoU |
| Same-image paired instruction dependence | ✅ **10 / 10** |
| Free-generation `[SEG]` emission | ✅ **20 / 20** |
| Stage 2 peak VRAM | 5.99 GiB allocated / 8.76 GiB reserved |
| Tests | **156 / 156 passed**, exit 0 |
| Verdict | **`PASS`** |

## Measured limitations (carry into Task 6B)

1. **The assistant reasoning text is not learned** (assistant token accuracy ≈ 0). The smoke recipe is
   mask-dominant, so the model emits `[SEG]` and the correct mask follows, but the trace itself is not
   usable. This must be fixed before any MLLM narrative claim.
2. **No generalisation**: 0/4 `[SEG]` emission on unseen test records — expected from a 20-sample overfit.
3. **Small-target IoU ceiling**: sub-300 px targets reach ~0.87 under strict nearest upsampling, because
   SAM2's decoder output is 256×256. The primary metric uses SAM2's own bilinear post-processing.
4. **Network mediation**: a local accelerator (Steam++ / Watt Toolkit) terminates TLS for some hosts; a
   process-local certifi fix is in `buildreasonseg_mvp/local_env.py`. If the accelerator is disabled the
   ModelScope route is the fallback for Qwen weights.

## Current blockers

**None.**

## Recommended next task

**ChatGPT review of the pushed Task 6A implementation and measurements**, then **Task 6B** as a separate
task, in this order: (1) make the language trace real, (2) scale from 20 samples to a real mini-train
using the same code path, (3) add `[REF]`, (4) only then the Spatial Relation Encoder and the Spatial
Consistency Loss.

Full detail: `handoff/FROM_DSH.md`, `docs/task6a_mvp_smoke.md`,
`evaluation/task6a_smoke_report.json`, `docs/architecture_decisions.md` (ADR-013).
