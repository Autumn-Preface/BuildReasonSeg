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

# FROM_DSH — Task 6M Report: Native-Vector Proposal Model + Structured Demo Gate

_This file holds the Task 6M report; the Task 6L report is preserved in git history and in
`docs/task6l_vector_dataset_migration.md`._

Full design notes: `docs/task6m_native_vector_proposal_demo.md`.

## 1. Verdict

**`PROPOSAL_MODEL_NEEDS_IMPROVEMENT`** — failed gates: `j1v2_gate_passed`,
`j4v2_fixed120_miou_ge_0_40`, `j4v2_paired_pass_ge_14`. Passing gates: export valid, M0 smoke,
parser ready, Demo-CLI gate.

Everything except the proposal model's quality is delivered and verified: the derived export is valid
(tile-union IoU 0.9988, 0 malformed, 0 missing, all 1,235 tiny instances retained, 6 hole instances
recorded), the parser reaches 1.0000 accuracy on v0.2, the deterministic executor reproduces the
frozen Task 6J oracle behaviour exactly, J4-v2 ran **once** on test with **no ground truth in
inference**, and the CMD Demo CLI answers real imagery with no annotation files. The proposal model
trained for **18 of 80 configured epochs** (wall-clock bound) and its small-object recall is the
binding limit.

## 2. Fixed Eval Packs (Task 6L's paired-evaluation gap)

`task6m_eval_pack_manifest.json` freezes four packs before any training: val/test `fixed120`
(stratified over level × program family, one record per tile, **all 20 programs present**) and
val/test `paired20` (same tile, same reference instances, same level, **different native targets**),
with SHA256 for each and `frozen_before_training: true`. The test packs predate the frozen inference
configuration and were never inspected during tuning.

## 3. YOLO26 Provenance

* `ultralytics==8.4.164` from **PyPI** (official release) into the new project-local env
  `.conda/buildreasonseg-proposal` (cloned from the Task 6A env; `.conda/` gitignored). The historical
  editable `yolo_sam_env` / local Ultralytics checkout was **not** used for training or inference.
* Weights: `yolo26m-seg.pt` (54,750,385 B, SHA256 `16b636f04e8fb6a3…`) and `yolo26s-seg.pt`
  (23,467,933 B, SHA256 `3da1d83e31caec96…`), both from
  `github.com/ultralytics/assets/releases/download/v8.4.0/`. No other download.
* **License:** the Ultralytics open-source stack and pre-trained models are **AGPL-3.0** (enterprise
  licensing available for proprietary/commercial use). The repository LICENSE is **not** changed and
  **no commercial-licensing claim** is made.

## 4. Export Audit

`artifacts/task6m_yolo_native/` (gitignored), derived from `scene_disjoint_v1`: **17,388 tiles**,
**41,186 polygons**, 11,909 valid empty label files, **all 17,388 source images hardlinked** (0 copies,
nothing moved). Fidelity vs canonical masks: mean tile-union IoU **0.99880**, per-instance IoU
(≥ 9 px) **0.99587**, **0 malformed**, **0 non-hole instances missing**, **1,235/1,235 tiny instances
recovered**, 6 hole instances recorded as an export-only limitation, 172 sub-pixel polygons repaired
inward and listed. Verdict **`EXPORT_VALID`**.

## 5. Environment

`.conda/buildreasonseg-proposal`: Python 3.11.16, torch 2.13.0+cu132, ultralytics 8.4.164,
opencv 5.0.0.93, transformers 5.17.0, RTX 5080 Laptop 16.3 GB. Recorded in
`task6m_environment_manifest.json` with the clone recipe, the package table, the GPU and the license
note. No modification to base/jupyter/yolo_sam_env; no package installed outside this env.

## 6. Smoke (M0)

`task6m_smoke.json` — all six gates pass: checkpoint loads, 2-image forward/backward succeeds, a
deterministic 600-tile subset trains 2 epochs, decoded masks are 512×512 and scored with the canonical
evaluator, empty tiles (454 in the subset) are handled, and all 29 tiny instances in the subset are
decodable from the export. No architecture conclusion is drawn from M0.

## 7. Full Training (M1)

`task6m_training_summary.json` — YOLO26m-seg from the COCO checkpoint, imgsz 640, batch **16** (chosen
by the memory probe, 10.52 GB peak; imgsz was never reduced), 4 workers, seed 20260812, AMP,
deterministic, best+last checkpoints, test split never used.

* **epochs completed 18 / 80** (best epoch **14**, `metrics/mAP50-95(M)` **0.36392**);
* mean **313 s** per epoch (train + full 3,618-image validation), peak VRAM 11.2 GB;
* **stop: session wall-clock budget** after the last completed epoch — patience 15 never triggered and
  the curve was still improving (mask mAP50 0.559 → 0.682 across the last recorded epochs);
* checkpoints `best.pt` / `last.pt`, 155 MB each, SHA256 recorded.

This is the honest limitation of the task: the configured cap was not reached, so every metric below
is a **lower bound** for this configuration.

## 8. Proposal Metrics (validation)

GT = canonical native masks. recall@0.25/0.50/0.75 = **0.7772 / 0.6333 / 0.3404**; mask
precision/recall (union) 0.7189 / 0.7574, Dice 0.7376; validator **mask mAP50 0.6819 / mAP50-95
0.3637**; **6.38 proposals/tile**; empty-tile false-proposal rate 0.1264; tiny / border / dense
recall@0.50 = **0.0023** / 0.5461 / 0.6322. Size breakdown (recall@0.50): large 0.7269, medium 0.5299,
small **0.1252**, tiny **0.0023**.

## 9. Frozen Inference Config

`task6m_inference_config_frozen.json`: **conf 0.10, max_det 100**, selected on validation only from
the declared grid (conf ∈ {0.05, 0.10, 0.25} × max_det ∈ {100, 300}) by the declared hierarchy
(target recall@0.50 → oracle-program structured performance → proposal burden);
`frozen_before_test: true`, `test_metrics_inspected_before_freezing: false`.

## 10. J1-v2 (validation)

Full val 9,111 records + fixed120 + paired20: recall@0.50 0.6333 (gate 0.92), tiny recall 0.0023
(gate 0.60), fixed120 mIoU **0.2801** (gate 0.50), paired **4/20** (gate 14), abstentions **35**
(gate ≤ 20) → all five gates FAIL. Full-val mIoU 0.2973 (0.4110 among answered), abstention rate
0.2768, by level L1 0.3247 / L2 0.2025 / L3 0.2812; paired mean own IoU 0.2640 vs cross 0.0244.

## 11. Program Parser

**`PARSER_READY`.** The frozen Task 6J J2 checkpoint loads its ProgramHead but **626 LoRA/token keys
are missing** in this environment (v0.2 val accuracy 0.515, confusions among related direction
programs), which is exactly the spec's retrain condition; the **same 2B text-only head was retrained
on v0.2 train only** (5 epochs, final loss 0.0000) and reaches **1.0000 accuracy / 1.0000 macro F1 on
the full 9,111-record v0.2 val** (fixed120 1.0000). No 4B upgrade, no image tokens, no `query_type`
leakage.

## 12. J4-v2 (test, single graded run)

`task6m_j4v2_test.json`, executed **once** after both freezes, on full test (6,219 records), test
fixed120 and test paired20, with **no ground truth in inference**: parser accuracy **1.0000**; fixed120
mIoU **0.2575** (gate 0.40) FAIL; paired **5/20** (gate 14) FAIL; full-test mIoU 0.2914 (L1 0.3278 /
L2 0.1854 / L3 0.2490).

## 13. Failure Attribution

`task6m_error_attribution.json`: the binding constraint is **proposal recall on small objects**
(0.727 ≥ 1000 px → 0.125 for 50–200 px → 0.002 for < 50 px) amplified by an **18-of-80-epoch**
training budget. The executor is not implicated (oracle candidates reproduce the frozen Task 6J J0
result exactly, mIoU 1.0000; paired own-vs-cross margin 11×), nor is the parser (1.0000), nor the
annotation/split (`EXPORT_VALID`, zero leakage). No YOLO26l/x run, no framework switch and no 4B
upgrade was performed or is implied; the same configuration simply needs to train to convergence.

## 14. Demo CLI

`predict_structured.py` runs from CMD with the documented arguments and **no annotation files**:
`task6m_demo_cli_audit.json` audits **12 real images** (all 12 parsed to the expected program, 9
produced a selected mask + overlay, 3 abstained with an explicit reason, `any_ground_truth_used:
false`) → gate **passed**. Outputs: parsed canonical program, id-free zh/en reasoning trace, proposal
count, selected geometry, selected-mask PNG, overlay PNG, JSON result, abstention reason. An unrelated
instruction cannot be silently mapped to an unrelated program.

## 15. License Note

AGPL-3.0 for the Ultralytics open-source stack and its pre-trained models, with enterprise licensing
for proprietary/commercial use. This repository's LICENSE is unchanged and no commercial-licensing
claim is made. Raw WHU data remains external (no explicit redistribution licence is held by this
project).

## 16. Tests

`python -m pytest tests/ -q` → **547 passed, 1 skipped** (518 before Task 6M + 30 new checks; the
skipped one is the Ultralytics eval-mode determinism check, which requires the proposal env, and is
covered there by `task6m_smoke.json` and the frozen-config rerun).
`tests/test_task6m_native_vector_proposal.py`
covers the 28 required checks: canonical dataset / v0.2 / v0.1.1 unchanged, no source mutation, export
preserves all non-hole instances, valid empty labels, no `<50` filter, hole loss recorded only in the
derived export, correct `scene_disjoint_v1`, zero feature leakage, packs frozen before tuning, no test
metrics before the frozen config, GT never repairs proposals, parser text-only, no `query_type` in the
parser input, no GT in J4, executor receives predicted geometry only, pairs same-image/different
target, all programs represented, CLI works without annotations, eval-mode determinism, unsupported
program → explicit failure, model hashes recorded, no large weights staged, no old editable
Ultralytics fork, no 4B, no `[REF]`/SRE/SCL, no GUI, and artifact presence.

## 17. Git / Watt

Commit `feat: train native-vector proposal model for structured demo` plus a `docs:` handoff commit.
Not committed: `.conda`, pretrained/trained weights, the derived image export, caches, source imagery
and vector data. Watt (pre-existing) was used for the authorized downloads only and was left running
per the ownership rules; no hosts, certificate, proxy or TLS setting was modified — `requests` could
not verify this network path, so the documented fallback downloaded through the OS trust store.

## 18. Recommended Next Step

Train the **same** configuration to convergence before drawing any architecture conclusion: resume
`yolo26m-seg` from the current `best.pt` (or rerun with the 80-epoch cap) so the curve — mask mAP50
0.682 at epoch 14 and still rising — is allowed to saturate, and re-run the frozen protocol
(`task6m_proposal_eval.py --sweep` → `task6m_j1v2.py` → `task6m_j4v2.py`, with the test run repeated
only after a new freeze). If tiny-object recall remains the limit after convergence, address it with
a **small-object strategy** (higher imgsz, tiled/windowed inference, or a documented small-object
configuration) rather than with a bigger backbone. Per the STOP section, nothing was retrained
further, no framework was switched, no 4B upgrade, no `[REF]`/SRE/SCL, no extra dataset download and
no GUI was built; waiting for ChatGPT review.
