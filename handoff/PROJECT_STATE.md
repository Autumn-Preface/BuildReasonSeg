# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 6B._

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
| MVP environment | **`.conda/buildreasonseg-mvp`** (conda `--prefix`, Python 3.11.16, PyTorch 2.13.0+cu132) |
| MVP stack (measured) | **Qwen3-VL-2B-Instruct + SAM2.1 Hiera Base+ + `[SEG]`** (ADR-013) |
| Design stack (unmeasured) | Qwen3-VL-4B-Instruct + SAM 2.1 hiera-large (ADR-012) |
| Network posture | offline (`HF_HUB_OFFLINE=1`); no accelerator, no hosts edit, no cert-store edit, no insecure TLS flag |

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
| 6A | Native-Windows env bootstrap + 2B `[SEG]` MVP smoke/overfit | done → `PASS` |
| 6B | **Network cleanup + 2B real mini-train + first generalization audit** | **done → `FAIL_REQUIRES_DEBUG`** |

## Task 6B measured results

Headline = the pre-declared section 12 recipe, 480 train / 120 val / 20 paired val images, 2400-step cap,
no test-split contact. Full detail: `docs/task6b_minitrain.md`, ADR-014,
`evaluation/task6b_validation.json`.

| Metric | Fresh baseline | Task 6B headline | Task 6B adjusted recipe |
|---|---|---|---|
| valid `[SEG]` emission (120 unseen val) | 0 / 120 | **120 / 120** | 120 / 120 |
| **strict end-to-end val mIoU** | 0.0000 | **0.1102** | 0.0950 |
| conditional mIoU | n/a | 0.1102 | 0.0950 |
| teacher-forced mIoU | 0.0063 | 0.1102 | 0.0950 |
| val LM CE / perplexity | 3.4576 / 31.74 | **0.00200 / 1.0020** | 5.1e-06 / 1.000 |
| reasoning-token accuracy | 0.4522 | **1.000** | 1.000 |
| operation-chain accuracy | 0.000 | **1.000** | 1.000 |
| **paired instruction dependence** | — | **0 / 20** | **0 / 20** |
| L1 / L2 / nontrivial-L3 strict mIoU | 0 / 0 / 0 | 0.1446 / 0.1082 / **0.0778** | — |
| peak VRAM (Phase B) | — | 6.17 GiB alloc / 7.13 GiB reserved | 6.17 / 7.13 |
| Verdict | — | **`FAIL_REQUIRES_DEBUG`** | did not help |

## What Task 6B changed in the project's understanding

1. **The language pathway generalises, the mask pathway does not.** On 120 unseen records the model emits
   exactly one `[SEG]` every time with 1.000 reasoning exact match and 1.000 operation-chain accuracy,
   while strict end-to-end mIoU is only 0.1102 (against 0.98 on the Task 6A 20-sample overfit).
2. **The failure is a collapsed prompt interface.** Two instructions on the same image give masks with
   IoU 0.9999 because the 256-d sparse prompt is near-constant (cosine 0.999995 within an image, 1.000000
   across images). The `[SEG]` hidden state itself carries little referent information (cosine 0.9952 for
   two instructions on one image vs 0.9996 across different images). Measured by
   `scripts/task6b_diagnose_prompt.py`, recorded in `evaluation/task6b_prompt_diagnosis_*.json`.
3. **Neither loss weighting nor the mask learning rate fixes it.** The single permitted section 12
   adjustment (Task 6A's mask-dominant weights + 3.3× mask LR) produced 0.0950, i.e. slightly worse.
4. **`reasoning_zh` in this dataset is a template.** The whole 480-record training mini-set has 21 distinct
   reasoning strings; all 20 distinct val strings occur in train. Language metrics here cannot support a
   reasoning claim.
5. **The network workaround is gone but Hugging Face is not reachable.** `huggingface.co` does not resolve
   on this network at all; Task 6B ran offline from cache. GitHub HTTPS works, git-smart-HTTP is
   intermittent (`github.com` sometimes resolves to a black-holed address).

## Measured limitations (carry into Task 6C)

1. **Instruction conditioning of the mask is absent** (paired probe 0/20). This is the blocking defect.
2. **Mask quality is not usable** anywhere: every level/family stratum is between 0.08 and 0.20 strict
   end-to-end mIoU.
3. **The gradient budget is dominated by the SAM2 mask decoder** (norm 114.3 vs 13.2 projection and 1.0
   LoRA at the first Phase-B step, all clipped to 1.0), so the two places a referent-specific prompt could
   come from receive ≈11 % and <1 % of the clipped gradient.
4. **Language quality cannot be assessed from this dataset** because the target text is templated.
5. **Small-target IoU ceiling** from Task 6A is unchanged: the SAM2 decoder emits 256×256 logits; the
   primary metric uses SAM2's own bilinear post-processing.
6. **`feature_cache_images: 160` < 480 training images**, so the LRU thrashes and features are recomputed
   most epochs (wall-clock cost only).
7. **`training.deterministic: true` is a dead flag.** Phase B is not bit-reproducible across processes
   (epoch-1 mIoU 0.1008 vs 0.0932 on two executions of the identical recipe; best-epoch 0.11018644 vs
   0.11017864), so the section 17 +0.10 bar is met marginally rather than comfortably. Recorded, not
   changed: enforcing determinism would alter an already-recorded recipe.

## Current blockers

**None for the next task to start.** Task 6B's segmentation failure is understood and measured, not
mysterious, so Task 6C can be specified directly against it. It is **not** a blocker in the sense of a
broken environment or missing data.

## Recommended next task

**ChatGPT review of the pushed Task 6B results**, then **Task 6C focused on the `[SEG]`→prompt interface**,
in this order:

1. stop the projection collapsing (prompt normalisation, a point-inside-the-mask auxiliary supervision
   derived from ground-truth geometry used as *supervision only*, or more than one prompt channel);
2. fix the gradient budget (per-group clipping, or lower decoder LR with higher LoRA LR) — this is a
   different intervention from the section 12 adjustment that was already tried and failed;
3. raise `feature_cache_images` to 480;
4. keep the paired instruction-dependence probe as the primary gate;
5. only then consider 4B, and only as a scale measurement, never as the fix for the collapse.

Full detail: `handoff/FROM_DSH.md`, `docs/task6b_minitrain.md`,
`evaluation/task6b_validation.json`, `evaluation/task6b_training_report.json`,
`docs/architecture_decisions.md` (ADR-014).
