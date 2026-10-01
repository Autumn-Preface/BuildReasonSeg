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

# FROM_DSH — Task 7H Report: Development Architecture Freeze + Formal Experiment Protocol

_This file holds the Task 7H report. The Task 7G report is preserved in git history at commit `22401fe`;
Task 7F at `c59c6d3`; Task 7E at `f9c6e87`; Task 7D at `86e4f4c`; Task 7C at `632c9c0`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full freeze document: `docs/task7h_development_architecture_freeze.md` · formal protocol:
`docs/task7h_formal_experiment_protocol.md`. **Task 7H performed no training and read no test record.**

## 1. Verdict

**`DEVELOPMENT_ARCHITECTURE_FROZEN`** — all ten section-22 conditions hold:

| # | Condition | Result |
|---|---|---|
| 1 | required checkpoint hashes match | ✓ YOLO `ef852b58…61f474` · Task 7C parser `c1505736…d58d9a` · Task 6O N-B3 `7556e4a4…c7d6ab` · Task 6Z Z-B3 `74f308e1…fc0f0bc` · Task 7D D-B1 `6df31909…21a89c0` |
| 2 | active dataset identity | ✓ BuildSpatialReason **v0.2** / WHU-EA-NativeVector **v1.0** / `scene_disjoint_v1` |
| 3 | no rejected selector/ranker in the chain | ✓ all six rejected modules recorded as frozen negative evidence and excluded |
| 4 | D-B1 role correct | ✓ `preferred L3 target-decoder architecture candidate`; Z-B3 = frozen baseline/ablation |
| 5 | parser limitation recorded | ✓ canonical 1.0 / macro F1 1.0 / 240/240 / 40/40 vs fixed24 5/24, compact 0/8, stress 0.6667 |
| 6 | reference limitation recorded | ✓ L-01 unresolved practical bottleneck |
| 7 | formal protocol complete | ✓ four L3 programs (train 1344 = 323/347/338/336, val 936), D-B1 schedule copied from `task7d_training.json`, seeds 20261001/2/3, complete metric set |
| 8 | test lock LOCKED | ✓ `test_execution_authorized: false` |
| 9 | no training occurred | ✓ |
| 10 | no test accessed | ✓ test file neither read nor hashed |

No other verdict applies: no frozen asset is missing, all hashes match, and the protocol is consistent.

## 2. Recorded Task 7G result (frozen, not modified)

`LARGEST_SELECTOR_NOT_LEARNABLE`. Internal tile-disjoint holdout: G-I0 mean selected reference IoU
`0.5199913587`, oracle-best top-1 `0.4887892377`, mean gap `0.2789719922`; G-I1 mean selected IoU
`0.6198877726`, median `0.7895902547`, Pr(≥0.50) `0.7309417040`, oracle top-1 `0.6591928251`, mean gap
`0.1790755783`, gain `+0.0998964138`. Gate: gain ≥ +0.08 **PASS** · mean IoU ≥ 0.62 **FAIL** (0.6198878) ·
top-1 ≥ 0.55 **PASS** · gap ≤ 0.14 **FAIL** (0.1790756).

Consequences: the external E-HoldoutL3 stage **did not run**; the Task 7G selector is **not adopted**; **no
scene-disjoint result may be claimed**; reference intervention stops.

**Correct interpretation.** The learned selector showed a meaningful **internal** improvement but did not clear
the predeclared internal learnability gate. Calling it a scene-disjoint failure would be scientifically
incorrect because the external stage was never executed. The development system keeps the deterministic
selector — not because it is best in principle, but because **no learned replacement passed the frozen adoption
protocol**.

## 3. Frozen development architecture `BuildReasonSeg-DevFreeze-2026-10`

```text
controlled instruction
→ Task 7C Qwen3-VL-2B text-only ProgramHead (20 canonical programs)
→ U-C1 YOLO26m-seg proposals (imgsz 640, conf 0.05, max_det 300, default NMS, no TTA, no tiling)
→ deterministic largest reference selector (max predicted mask area → higher confidence → lower index)
→ GeometricRelationField v0.2 (P_dir) + NearestBoundaryField v0.1 (P_near)
→ Task 7D D-B1 target decoder
→ target mask
```

This is **not** a new checkpoint, **not** the final paper model, **not** an unrestricted natural-language
system and **not** an end-to-end solved system. The front end is a **canonical-program / controlled-language
development interface**. For L3 both fields are used separately and the product `W = clamp(P_dir * P_near)` is
used only for D-B1 prototype weighting (Task 7D). The directional L2 path is frozen to **Task 6O N-B3** (N-B2/N-B4
must not be substituted). nearest-only stays **`experimental/limited`**.

## 4. Frozen limitations

| ID | Limitation | Status |
|---|---|---|
| L-01 | Reference selection (F-R0 0.245405 → F-R1 0.364909, gain +0.119504 = 85.3 % of the gap; Task 7G not adopted) | **unresolved practical bottleneck** |
| L-02 | Proposal coverage (coverage@0.50 = 0.846039, coverage gain +0.054316, 103 uncovered records) | **secondary unresolved bottleneck** |
| L-03 | Proposal-mask geometry (covered gain +0.004253) | **not a major current bottleneck** |
| L-04 | Free-form L3 language (canonical 1.0 vs fixed24 5/24, compact 0/8, stress 0.6667) | **controlled-language interface only** |
| L-05 | nearest-only (Task 6Y B2 0.2864, paired 6/20, `NEAREST_FIELD_NO_MEANINGFUL_GAIN`) | **not validated as standalone final capability** |
| L-06 | unseen-city/domain generalization | **not established** (raster/scene separation only) |

## 5. Formal protocol and test lock (frozen, not executed)

`evaluation/task7h_formal_experiment_protocol.json` + `docs/task7h_formal_experiment_protocol.md`: train for
gradients · val for checkpoint selection/early stopping · test only after all choices are frozen. The Task 6M
J4-v2 test access is disclosed, so future reporting may only say **`final frozen-architecture test evaluation`**
— never "untouched test". D-B1 formal retraining uses fresh weights with the schedule copied verbatim from
`evaluation/task7d_training.json` (AdamW, lr 3e-4, wd 1e-4, batch 8, ≤25 epochs, patience 5, bf16 AMP, val-mIoU
selection) and seeds **20261001 / 20261002 / 20261003**, each selected independently. Practical predicted
reference and oracle-reference diagnostics are reported separately. Z-B3 is B-L3-0 (baseline/ablation), D-B1 is
B-L3-1 (main candidate); the historical Task 6Z/7D controls stay ablations.

Test lock: `status LOCKED`, `architecture_head D-B1`, `reference_policy "U-C1 deterministic largest"`,
`parser_role "controlled-language/canonical interface"`, `test_execution_authorized false`,
`unlock_condition "ChatGPT audit after formal train/val completion"`. Task 7H did not unlock it.

## 6. Claim registry

C1 `SUPPORTED` · C2 `SUPPORTED_WITH_LIMITATION` · C3 `SUPPORTED` · C4 `NOT_SUPPORTED` · C5 `NOT_SUPPORTED` ·
C6 `SUPPORTED_WITH_LIMITATION` · C7 `NOT_SUPPORTED` · C8 `NOT_SUPPORTED` · C9 `NOT_SUPPORTED`. No novelty or
"first" claim appears anywhere in the registry or the docs.

## 7. Tests, storage, git

`python -m pytest tests/ -q` → **1420 passed, 1 skipped** (Task 7G ended at 1377 passed / 1 skipped; no prior
passing test was reduced). `tests/test_task7h_development_freeze.py` adds the 43 Part-I checks.

Not committed: model checkpoints, proposal/feature caches, imagery/vectors, local generated model data,
`.conda`. Committed: freeze/protocol JSON, docs, audit scripts, tests, handoff. No checkpoint was created and
no production CLI default was changed. Nothing was downloaded or installed. Watt was **not needed** in Task 7H
(the pre-existing instance is transport-only, not owned by this project, left running).

## 8. Interpretation boundary

DSH audited and froze only. No model was trained; no test record was read, hashed or evaluated; no threshold,
field, loss, model structure or data split was changed; the Task 7G selector was not run on E-Holdout after its
STOP; no Task 7I implementation detail beyond the frozen protocol was chosen; D-B1 is not called the final paper
model; unrestricted natural language, end-to-end success, novelty, "first" and unseen-city generalization are
not claimed; the test split is never called untouched.

## 9. Recommended next step (exact wording required by Part L)

等待 ChatGPT 审核 Task 7H 的开发版架构冻结与正式实验协议；在审核通过前不启动正式三种子训练，不解锁 test。

## 10. STOP

Task 7H stops here: Task 7I formal training is not started, the test lock is not unlocked, no test is run, no
further selector is trained, YOLO/parser/D-B1 are not retrained and the architecture is not modified. Waiting
for the ChatGPT audit.
