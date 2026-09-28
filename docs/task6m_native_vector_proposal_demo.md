# Task 6M — Native-Vector Proposal Model + Structured Demo Gate

> Task: `handoff/TO_DSH.md` · Repository: `BuildReasonSeg` · Predecessor: Task 6L `VECTOR_DATASET_MIGRATION_PASS`
> **Verdict: `PROPOSAL_MODEL_NEEDS_IMPROVEMENT`** · Tests: `tests/test_task6m_native_vector_proposal.py`
> Evidence: `evaluation/task6m_*.json`

Task 6M trains a modern instance-segmentation proposal model on `WHU-EA-NativeVector v1.0`, evaluates
the native-vector version of the Task 6J structured chain under `scene_disjoint_v1`, and delivers a
CMD-runnable structured Demo path. It is a **proposal-quality + end-to-end inference** task, not the
final `[REF]`/SRE/SCL paper architecture.

**Outcome in one paragraph.** Every engineering component works: the derived export is valid
(tile-union IoU 0.9988, zero malformed labels, every tiny building retained), the environment and
weights are official and hashed, the M0 smoke passes all six gates, the 20-program parser reaches
1.0000 accuracy on v0.2, the deterministic executor reproduces the frozen Task 6J oracle result
exactly, the single graded test run completes with **no ground truth in inference**, and the CMD Demo
CLI answers real images without any annotation file. The **proposal model itself is not good enough
yet**: trained for 18 of the configured 80 epochs under this session's wall-clock budget, it reaches
recall@0.50 = 0.633 (gate 0.92) because recall collapses on small buildings (0.727 ≥ 1000 px → 0.002
< 50 px), so J1-v2/J4-v2 miss their gates. The verdict is therefore
**`PROPOSAL_MODEL_NEEDS_IMPROVEMENT`** — a training-budget + small-object result, not an architecture
or annotation result.

## 1. Frozen inputs and the paired-evaluation gap

`WHU-EA-NativeVector v1.0`, `BuildSpatialReason v0.2`, `scene_disjoint_v1`, the frozen Task 3B
relation semantics, the 20 canonical programs and every Task 6J/6K/6K.1/6L artifact are treated as
frozen; v0.1.1 and v0.2 were not rewritten.

Task 6L reported `paired_counterfactual_availability: null`. Task 6M closes that gap **before any
training** with four deterministic evaluation packs plus a manifest that freezes the construction
policy, the seeds and the pack hashes:

| Pack | Content | Construction |
|---|---|---|
| `task6m_val_fixed120.json` | 120 records | stratified over (level, program family), one record per tile, all 20 programs represented |
| `task6m_val_paired20.json` | 20 pairs | same tile, same reference instances, same level, **different native targets** |
| `task6m_test_fixed120.json` | 120 records | same policy (seed offset), frozen before tuning |
| `task6m_test_paired20.json` | 20 pairs | same policy, frozen before tuning |

`task6m_eval_pack_manifest.json` records `frozen_before_training: true` and the SHA256 of every pack;
the test packs are frozen before any threshold tuning, and no test metric was inspected before the
frozen inference configuration existed.

## 2. Proposal framework and provenance

* Primary model: **Ultralytics YOLO26m-seg** (`yolo26m-seg.pt`), the current released family with
  official instance-segmentation support; no framework bake-off was run.
* Package: **official released `ultralytics==8.4.164` from PyPI** into a new project-local env
  `.conda/buildreasonseg-proposal` (cloned from the Task 6A env). The historical editable
  `yolo_sam_env` / local Ultralytics checkout was **not** used for training or inference.
* Weights: `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt`
  (54,750,385 bytes, SHA256 recorded in `task6m_environment_manifest.json`), plus `yolo26s-seg.pt`
  (23,467,933 bytes) for the smoke test only. No other download was made.
* GPU: NVIDIA GeForce RTX 5080 Laptop (16.3 GB); measured peak training VRAM **10.5–11.2 GB**.
* **License note:** the Ultralytics open-source stack and its pre-trained models are distributed under
  **AGPL-3.0**, with enterprise licensing available for proprietary/commercial use. This repository's
  LICENSE is **not** changed by Task 6M and **no commercial-licensing claim** is made.

## 3. Derived training export and its audit

`artifacts/task6m_yolo_native/` (gitignored) is a **derived** Ultralytics export of
`scene_disjoint_v1`:

* all **17,388** tiles exported, including 11,909 empty ones with valid **empty** label files;
* all **41,186** native instances kept — no `<50` filter, no connected-component conversion;
* every source image **hardlinked** (17,388 hardlinks, 0 copies) from the read-only archive; nothing
  was moved or modified;
* holes cannot be expressed in Ultralytics polygon TXT, so the export uses the required exterior-ring
  form and **records** the affected hole instances; the canonical GT keeps its holes.

Fidelity audit (`task6m_training_export_audit.json`), GT = canonical masks, decode = Ultralytics'
own convention (`clip(round(polygon * size), 0, size-1)`):

| Metric | Value |
|---|---|
| mean tile-union IoU | **0.99880** (gate ≥ 0.995 ✓) |
| mean per-instance IoU (≥ 9 px) | **0.99587** (gate ≥ 0.995 ✓) |
| malformed labels | **0** ✓ |
| non-hole instances missing | **0** ✓ |
| tiny instances recovered | **1,235 / 1,235** |
| hole instances affected (recorded) | 6 |
| sub-pixel polygons repaired to a 1-px box (documented) | 172 |
| verdict | **`EXPORT_VALID`** |

The sub-pixel repair exists because a polygon that reaches the last row/column loses that edge in
Ultralytics' decode: rather than silently dropping ~170 one-pixel buildings, the export grows their
box **inward** until the decoded polygon is non-degenerate, and lists every repair.

## 4. M0 smoke

`task6m_smoke.json` records the pre-training gates: the official checkpoint loads, a 2-image
forward/backward pass completes, a deterministic 600-tile subset trains for 2 epochs, mask decoding
validates against the canonical evaluator, empty images are handled, and tiny labels enter training.

## 5. M1 full training

`task6m_training_summary.json` (config: COCO-pretrained `yolo26m-seg.pt`, imgsz 640, max 80 epochs,
patience 15, fixed seed 20260812, AMP on, deterministic, batch 16 chosen by the memory probe,
4 workers, test split never used).

| | Value |
|---|---|
| memory probe | batch 16 OK at 10.52 GB peak (largest stable batch; no imgsz reduction was ever applied) |
| epochs completed | **18** of the 80-epoch cap (best epoch **14**) |
| best metric | `metrics/mAP50-95(M)` = **0.36392** at epoch 14 |
| mean epoch time | ~313 s (train + full 3,618-image validation) |
| peak VRAM | 11.2 GB |
| stop | session **wall-clock budget** after the last completed epoch; patience never triggered |
| checkpoints | `best.pt` / `last.pt` (155 MB each), SHA256 recorded in the summary |

**Honest limitation:** the configured 80-epoch cap was not reached. The run was bounded by the
session's wall-clock budget, not by convergence, and the numbers below are therefore a *lower bound*
on what this configuration can reach. The evaluation protocol (packs, thresholds, gates) is exactly
the one specified, so the verdict is directly comparable to a longer run.

## 6. Proposal metrics (validation only)

`task6m_proposal_val.json` — GT is the canonical native masks, never the training TXT.

| Metric | Value |
|---|---|
| recall @ IoU 0.25 / 0.50 / 0.75 | **0.7772 / 0.6333 / 0.3404** |
| mask precision / recall (union) | 0.7189 / 0.7574 (Dice 0.7376) |
| validator mask mAP50 / mAP50-95 | **0.6819 / 0.3637** (box 0.6960 / 0.3991) |
| proposals per tile | 6.38 |
| empty-tile false-proposal rate | 0.1264 (234 of 1,851 empty tiles) |
| tiny / border-truncated / dense-tile recall @0.50 | **0.0023** / 0.5461 / 0.6322 |
| size breakdown (recall @0.50) | large (≥1000 px) **0.7269** · medium (200–1000) 0.5299 · small (50–200) **0.1252** · tiny (<50) **0.0023** |

The size breakdown is the single most important diagnostic: recall falls monotonically with object
size, and sub-50 px buildings are essentially undetected — the model, not the executor, is the limit.

## 7. Frozen inference configuration

`task6m_inference_config_frozen.json` — selected on **validation only** from the declared grid
(confidence ∈ {0.05, 0.10, 0.25} × max_det ∈ {100, 300}), ranked by (1) target recall@0.50,
(2) oracle-program structured performance, (3) proposal burden. Frozen: **conf 0.10, max_det 100**,
`frozen_before_test: true`, `test_metrics_inspected_before_freezing: false`. The three confidence
levels trade recall for burden as expected (0.05 → 0.596 recall at 9.5 proposals/tile, 0.10 → 0.633
at 6.4, 0.25 → 0.607 at 3.7), so 0.10 wins the declared hierarchy.

## 8. J1-v2 (validation): oracle program + predicted proposals

`task6m_j1v2_val.json` — full val (9,111 records), val fixed120 and val paired20.

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| overall recall@0.50 | ≥ 0.92 | **0.6333** | FAIL |
| tiny recall@0.50 | ≥ 0.60 | **0.0023** | FAIL |
| fixed120 mIoU | ≥ 0.50 | **0.2801** | FAIL |
| paired own-vs-cross pass | ≥ 14/20 | **4/20** | FAIL |
| fixed120 abstentions | ≤ 20/120 | **35** | FAIL |

Full val: mIoU **0.2973**, Dice 0.3413, mIoU among answered 0.4110, abstention rate **0.2768**
(2,522/9,111; reasons `nearest_relation_invalid` 16, `filter_multi_candidate` 9,
`no_eligible_candidates` 6 in fixed120). By level: L1 0.3247 (6,460 records), L2 0.2025, L3 0.2812.
Target breakdown: tiny 0.0053, border-truncated 0.2496, dense tiles 0.2411. Paired own-vs-cross:
mean own IoU 0.2640 vs mean cross IoU 0.0244 — the executor *does* distinguish the two targets
(an 11× margin), it simply does not have a proposal good enough to score.

## 9. Program parser on v0.2

`task6m_parser_v02.json` — **`PARSER_READY`**:

* the frozen Task 6J J2 checkpoint's ProgramHead loads, but **626 LoRA/token keys are reported missing
  by `load_state_dict(strict=False)`** in this environment, so its v0.2 val accuracy is only **0.515**
  (confusions are between related direction programs, e.g. `topmost`→`leftmost`);
* the spec's fallback therefore applied: the **same 2B text-only head was retrained on v0.2 train
  only** (5 epochs, final train loss 0.0000), and reached **accuracy 1.0000 / macro F1 1.0000 on the
  full 9,111-record v0.2 val** (fixed120 1.0000), checkpoint
  `artifacts/checkpoints/task6m/program_parser_v02_best.pt` (210 MB, SHA256 recorded);
* no 4B upgrade, no image tokens, no `query_type` leakage.

## 10. J4-v2 (test): the graded single run

`task6m_j4v2_test.json` — run **once** after both freezes, on full test (6,219 records), test fixed120
and test paired20. Chain: instruction → Qwen2B ProgramHead → canonical program; image → YOLO26m-seg
proposals; program + **predicted** geometry → deterministic executor → selected proposal mask. GT
scored the result and never entered inference.

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| fixed120 mIoU | ≥ 0.40 | **0.2575** | FAIL |
| paired own-vs-cross pass | ≥ 14/20 | **5/20** | FAIL |

Parser accuracy on test records: **1.0000**. Full test: mIoU **0.2914**, Dice 0.2949-equivalent per
record set, by level L1 0.3278 / L2 0.1854 / L3 0.2490, abstentions 866 + 642 + 256.

## 11. Demo CLI

`task6m_demo_cli_audit.json` — gate **passed**: the documented command was executed as a subprocess on
**12 real images** (no annotation files available) with prompts drawn from v0.2; all 12 parsed to the
expected program (the graded runs use the same closed 20-way parser), 9 produced a selected mask +
overlay and 3 abstained with an explicit reason, and `any_ground_truth_used: false`. An unrelated
instruction ("Write a poem about the sea.") cannot be silently mapped: the CLI records the parsed
program, and the executor answers or abstains auditable.

```text
python predict_structured.py ^
  --image path\to\image.tif ^
  --prompt "分割面积最大的建筑物右侧最近的建筑物" ^
  --proposal-checkpoint artifacts\checkpoints\task6m\runs\m1_yolo26m_seg\weights\best.pt ^
  --parser-checkpoint artifacts\checkpoints\task6m\program_parser_v02_best.pt ^
  --out-dir outputs\demo
```

Outputs: parsed canonical program, compact id-free zh/en reasoning trace, proposal count, selected
proposal geometry, selected-mask PNG, overlay PNG, JSON result and an explicit abstention reason.

## 12. Verdict and failure attribution

**`PROPOSAL_MODEL_NEEDS_IMPROVEMENT`** (`task6m_verdict.json`), failed gates
`j1v2_gate_passed`, `j4v2_fixed120_miou_ge_0_40`, `j4v2_paired_pass_ge_14`; the export, smoke, parser
and Demo-CLI gates all pass.

Attribution (`task6m_error_attribution.json`):

1. **Proposal recall on small objects is the binding constraint.** Recall@0.50 is 0.727 for buildings
   ≥ 1000 px but 0.125 for 50–200 px and 0.002 for < 50 px; tiny-instance recall is 225× below the
   gate. The model was trained for **18 of the configured 80 epochs** (best epoch 14, mean 313 s per
   epoch) and was stopped by the session's wall-clock budget, not by convergence — the training curve
   was still improving (mask mAP50 0.559 → 0.682 between the last recorded epochs).
2. **The executor is not the problem.** With oracle candidates it reproduces the frozen Task 6J J0
   result exactly (mIoU 1.0000), and in the paired test its own-vs-cross margin is 11×.
3. **The parser is not the problem.** 1.0000 accuracy on v0.2 val and on the test records.
4. **The annotation and split are not the problem.** The export is `EXPORT_VALID` at IoU 0.9988 and
   `scene_disjoint_v1` is leakage-free.
5. **Not a reason to switch frameworks or scale up.** No YOLO26l/x run, no framework change, no 4B
   upgrade was performed or is implied by this evidence; the same configuration simply needs to train
   to convergence (and the tiny-object gap may additionally need higher resolution, tiling or an
   explicit small-object configuration).

## 13. Reproduce

```text
python scripts/task6m_freeze_eval_packs.py                 # section 2 packs
python scripts/task6m_environment.py                       # environment manifest
python scripts/task6m_download_weights.py                  # authorized weights + hashes
python scripts/task6m_build_export.py                      # derived export + fidelity gate
python scripts/task6m_smoke.py                             # M0
python scripts/task6m_train.py --epochs 80 --batch 16      # M1
python scripts/task6m_training_summary.py                  # finalize M1 metrics
python scripts/task6m_proposal_eval.py --checkpoint <best.pt> --sweep   # metrics + freeze
python scripts/task6m_j1v2.py                              # J1-v2 (val)
python scripts/task6m_parser_eval.py                       # parser on v0.2
python scripts/task6m_j4v2.py                              # J4-v2 (test, once)
python scripts/task6m_demo_audit.py --count 12             # CMD CLI audit
python scripts/task6m_error_attribution.py                 # attribution + verdict
```

Every checkpoint, the derived export and the caches stay under the gitignored `artifacts/` tree;
only code, configs, tests, small JSON artifacts and docs are committed.
