# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 6C._

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
| Network posture | offline (`HF_HUB_OFFLINE=1`); Watt Toolkit stopped during training; no hosts edit, no cert-store edit, no insecure TLS flag |
| Reproducibility | **strict deterministic mode, cross-process bit reproducible** (Task 6C) |

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
| 6B | Network cleanup + 2B real mini-train + first generalization audit | done → `FAIL_REQUIRES_DEBUG` |
| Watt | Standalone Watt Toolkit lifecycle test (3 rounds) | done → `FULL_AUTO_OK` |
| 6C | **Paired counterfactual training × neutral SAM prompt 2×2 ablation** | **done → `EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`** |

## Task 6C measured results

Valid, bit-reproducible 2×2 on the frozen ADR-013 stack. Same recipe, same seed, same validation material;
only the training subset (U/P) and the SAM bridge (C/L) differ. Full detail:
`docs/task6c_prompt_ablation.md`, ADR-015, `evaluation/task6c_comparison.json`.

| Arm | strict e2e mIoU | Dice | emission | paired | mean margin | projected effective rank | top-1 variance |
|---|---|---|---|---|---|---|---|
| `U_C` (Task 6B baseline) | 0.10516 | 0.17360 | 120/120 | **0/20** | −0.000001 | 1.516 | 0.9023 |
| `U_L` (language bridge) | 0.10872 | 0.17846 | 120/120 | **0/20** | −0.003530 | 1.491 | 0.9087 |
| `P_C` (paired data) | 0.10604 | 0.17851 | 120/120 | **0/20** | +0.000007 | **4.100** | **0.5892** |
| `P_L` (both) | 0.09284 | 0.15626 | 120/120 | **0/20** | −0.000003 | **3.371** | **0.6430** |

## What Task 6C changed in the project's understanding

1. **Neither of Task 6B's two confounds is the cause.** All four arms are 0/20 on the paired unseen probe
   with mean margins within ±0.0036 of zero. Paired counterfactual training, the removal of the fixed
   centre point, and their combination all fail to create instruction-conditional mask selection.
2. **But the factors are not inert.** Paired training materially changes the prompt representation
   (effective rank 1.50 → 3.74, top-1 variance 0.905 → 0.616) without changing the masks. The projection
   uses more dimensions; two instructions on one image still produce the same mask.
3. **The fixed positive centre point is not the culprit.** Its norm is 11.39 against a projected language
   norm of 266.8 — a **23.4×** ratio in the language vector's favour — and removing it (the `L` bridge)
   changes little. "The centre point overrides the language" is not supported by the measurement.
4. **The collapse is directional and low-rank, not constant.** Same-image projected cosine > 0.9999 with
   top-1 variance 0.59–0.91 and a non-zero effective rank. Task 6B's looser wording is corrected.
5. **The remaining problem is after the prompt**, i.e. in how the `[SEG]` hidden state is formed and how
   SAM's decoder turns a prompt direction into a region. Prompt normalisation, a multi-token prompt,
   auxiliary point supervision or `[REF]` are justified now, and were not before.
6. **Determinism is real now.** `training.deterministic` is consumed, strict deterministic algorithms are
   active with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and two independent processes produce identical losses,
   gradients and post-step parameters. The `U_C` arm was run before and after the cache fix and is
   bit-identical, which also proves the fix value-preserving.

## Correctness fixes carried forward (do not regress)

1. Determinism (above).
2. Per-arm checkpoint directories come from `cfg["paths"]["checkpoints"]` — no module-level constant.
3. The operation-chain lookup is built from the **train split only**; validation text never extends it.
4. The SAM2 CPU feature cache holds all 480 training images inside the budget by sharing the two
   image-independent tensors once (7.508 GiB instead of 11.25 GiB), and it never mutates stored entries.

## Measured limitations (carry into Task 6D)

1. **Instruction conditioning of the mask is still absent** (0/20 in all four arms). This remains the
   blocking defect for the MVP.
2. **Mask quality is not usable**: strict end-to-end mIoU 0.093–0.109, L3 nontrivial 0.076–0.088.
3. **Language metrics are template metrics**: `reasoning_zh` has 21 distinct values in the whole training
   mini-set, so exact match and operation-chain accuracy cannot support a reasoning claim.
4. **More prompt diversity did not become mask diversity**: effective rank 3.4–4.1 still yields
   IoU(pred_A, pred_B) 0.999.
5. **The best mIoU arm is not the most diverse arm** (`U_L` 0.1087 vs `P_L` 0.0928), so mIoU alone remains
   a misleading selection signal.

## Current blockers

**None for the next task to start.** The failure is characterised, the two candidate causes are excluded by
a valid controlled experiment, and the remaining search space is now narrow and explicit.

## Recommended next task

**ChatGPT review of the pushed Task 6C results**, then **Task 6D aimed after the prompt**:

1. make the `[SEG]` hidden state instruction-conditional (auxiliary point-inside-the-mask supervision from
   ground-truth geometry used *as supervision only*, or a second explicitly negative prompt slot);
2. only then test prompt normalisation / whitening and a multi-token prompt, which Task 6C has now earned
   the right to try;
3. keep the paired instruction-dependence probe as the primary gate and keep the four correctness fixes;
4. do not scale to 4B and do not add `[REF]` on the strength of anything measured so far.

Full detail: `handoff/FROM_DSH.md`, `docs/task6c_prompt_ablation.md`,
`evaluation/task6c_comparison.json`, `docs/architecture_decisions.md` (ADR-014 amendment, ADR-015).
