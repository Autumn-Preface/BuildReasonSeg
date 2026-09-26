# TO_DSH — Task 6B: Network Cleanup + 2B Real Mini-Train & First Generalization Audit

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Goal: keep the proven Task 6A architecture fixed, remove the Watt Toolkit TLS workaround, then run the first **real generalization experiment** on a deterministic 2B `[SEG]` mini-train:
>
> `Qwen3-VL-2B-Instruct -> [SEG] hidden -> Projection MLP -> SAM2.1 Hiera Base+ -> target mask`
>
> This task must answer:
>
> 1. Does the project work with a normal/direct TLS/Hugging Face path after Watt Toolkit is stopped?
> 2. Can the 2B architecture generalize beyond the 20-sample overfit set?
> 3. Can the model learn the **reasoning_zh** trace and emit exactly one `[SEG]` on unseen validation images?
> 4. What is the first honest end-to-end validation performance by L1 / L2 / nontrivial L3?
>
> Do **not** use Qwen3-VL-4B, `[REF]`, Spatial Relation Encoder, Spatial Consistency Loss, or full-dataset training in Task 6B.

## 0. User-facing language

All narrative text visible in the DSH web/chat UI must be **Chinese**.

`handoff/FROM_DSH.md` and `handoff/PROJECT_STATE.md` may remain English.

## 1. Authorization and safety boundary

Use only the existing project-local Conda environment:

`C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp`

Do not create another environment. Do not modify Conda base, `yolo_sam_env`, system Python, Windows hosts, certificate store, registry, PATH, drivers, CUDA toolkit, WSL, or system proxy settings.

Allowed writes are limited to the BuildReasonSeg repo plus ordinary runtime temp files.

Normal lightweight access to Hugging Face/GitHub is allowed. The 2B and SAM2 Base+ assets are already cached; do not intentionally re-download large weights.

Do not download 4B, external datasets, or begin `[REF]`, Spatial Relation Encoder, Spatial Consistency Loss, or full-dataset training.

## 2. Frozen architecture from Task 6A

Keep fixed:

- `Qwen/Qwen3-VL-2B-Instruct`
- BF16
- frozen Qwen vision tower
- frozen Qwen base language weights
- trainable text-only LoRA
- selectively trainable `[SEG]`
- `facebook/sam2.1-hiera-base-plus`
- frozen SAM2 image encoder / prompt encoder / memory modules
- trainable SAM2 mask decoder
- trainable project-owned Projection MLP
- no `[REF]`

Bridge:

`[SEG] hidden -> Projection MLP -> 256-d sparse prompt -> vanilla SAM2 mask decoder`

No GT point/box/centroid/bbox/component id/reference mask/relation geometry may enter inference.

Keep separate image branches:

- Qwen: source 512×512 -> 256 visual tokens measured in 6A
- SAM2: same source -> official transform -> 1024×1024 internal tensor

## 3. User state

The user has manually stopped acceleration in Watt Toolkit / Steam++ and exited Watt Toolkit.

Do not restart it.

# PART A — Network / TLS cleanup

## 4. Inspect hosts and local listeners — READ ONLY

Record:
- relevant hosts entries for `github.com`, `raw.githubusercontent.com`, `huggingface.co`
- whether any local listener remains on the previous proxy ports, especially 443
- DNS resolution for those hosts

Do not edit hosts.

If those hosts still resolve to localhost via hosts and direct connectivity fails, stop and report the exact residual state. Do not modify system network configuration yourself.

Create:

`evaluation/task6b_network_cleanup.json`

## 5. Restore standard certifi

Task 6A created `cacert.pem.orig` inside the dedicated Conda environment.

If it exists:
1. hash current `cacert.pem`
2. hash `cacert.pem.orig`
3. restore the original backup exactly
4. verify restored `cacert.pem` hash equals the backup
5. record all hashes

Do not append Windows ROOT/CA certificates after restoration.

If the backup does not exist, report that and inspect whether the current bundle still contains the Task 6A additions.

## 6. Remove automatic Watt-specific trust mutation

Refactor `buildreasonseg_mvp/local_env.py` so normal runtime does not mutate certifi automatically.

Preferred:
- default = standard certifi only
- old Windows-root merge retained only as an explicit legacy opt-in helper, or removed entirely

Add a regression test proving ordinary runtime initialization does not write to certifi.

Do not erase Task 6A historical documentation that the workaround once existed.

## 7. Direct connectivity validation

Using the dedicated Conda environment, verify standard access through:
- `huggingface_hub` metadata for `Qwen/Qwen3-VL-2B-Instruct`
- a tiny Hugging Face file request such as `config.json`
- GitHub HTTPS metadata
- `raw.githubusercontent.com`
- `git ls-remote https://github.com/facebookresearch/sam2 HEAD`
- Python `urllib`
- `httpx` / `huggingface_hub`

Acceptance:
- no `CERTIFICATE_VERIFY_FAILED`
- no Watt/SteamTools certificate injection
- no insecure flags
- standard Hub access works

Re-test a small SAM2 Hugging Face metadata/file request and record whether the old `X-Repo-Commit` / `LocalEntryNotFoundError` problem is gone. Do not redownload the 323 MB checkpoint just to test this.

If direct network access still fails, do not use `verify=False`, insecure TLS flags, or system edits.

# PART B — Deterministic real mini-train

## 8. Primary run must start fresh

Do not initialize the headline Task 6B run from the 20-sample Task 6A overfit checkpoint.

Start from:
- clean base Qwen3-VL-2B
- fresh LoRA
- fresh `[SEG]` trainable token state
- fresh Projection MLP
- original/pretrained SAM2 Base+ mask decoder state

Task 6A checkpoints may be used only for regression comparison.

## 9. Deterministic subsets

Create:

`evaluation/task6b_subset_ids.json`

### Train mini-set: 480 records

Use train split only:
- 160 L1
- 160 L2
- 160 nontrivial L3

Balance query families as evenly as feasible.

L1: cover all 6 query types.

L2: include both reference->nearest and reference->direction, largest/smallest refs, left/right/above/below where available.

L3: nontrivial only, balance directions.

Prefer one sample per source image. If exact balancing requires duplicates, minimize and report them.

### Validation mini-set: 120 records

Use val split only:
- 40 L1
- 40 L2
- 40 nontrivial L3

Balance query families as evenly as feasible.

### Paired validation probe

Select 20 val images × 2 instructions = 40 records where:
- same image
- different instructions/query types
- different targets

May overlap the 120 validation set, but list separately.

Never use test for tuning.

## 10. Language target

Input:
- image
- `instruction_zh`

Assistant target:

`reasoning_zh + " [SEG]" + EOS`

Requirements:
- exactly one `[SEG]`
- EOS present
- no component IDs
- no `reasoning_steps` as model input
- no GT geometry as model input

## 11. Language metrics are first-class

Track:
- assistant LM CE
- perplexity
- supervised assistant token accuracy
- reasoning-token accuracy excluding `[SEG]` and EOS
- `[SEG]` token accuracy at expected position
- free-generation valid `[SEG]` count
- normalized exact match of generated `reasoning_zh` before `[SEG]`
- normalized character-level similarity
- operation-chain accuracy by L1/L2/L3

Because reasoning is deterministic/template-generated, normalized exact match is meaningful.

Do not use an external LLM judge.

## 12. Two-phase training

### Phase A — language-format warm-up

Train only:
- text LoRA
- `[SEG]` token

Freeze:
- Projection MLP
- SAM2 mask decoder
- SAM2 image encoder
- Qwen vision tower

Use LM CE only.

Suggested:
- 1 epoch over fixed 480 records
- batch 1
- gradient accumulation if useful
- BF16
- warmup/cosine schedule

After Phase A, evaluate the fixed 120 val records:
- LM CE/perplexity
- reasoning token accuracy
- normalized exact match
- free-generation `[SEG]` emission

### Phase B — joint segmentation training

Continue from Phase A.

Train:
- text LoRA
- `[SEG]`
- Projection MLP
- SAM2 mask decoder

Keep all base/vision encoders frozen.

Initial loss:

`L_total = 2.0*L_lm_ce + 2.0*L_mask_bce + 1.0*L_mask_dice`

Run at least one full Phase-B epoch before changing weights.

One bounded recipe adjustment is allowed only after the first complete documented epoch if one objective clearly fails. Record original and adjusted results. No broad hyperparameter search.

Suggested maximum:
- Phase A: 1 epoch
- Phase B: up to 5 epochs
- <= 2400–3000 Phase-B optimizer steps
- early stopping with documented patience

## 13. Validation must separate teacher-forced and free-generation

### Teacher-forced validation

Use expected `reasoning_zh + [SEG]` to obtain `[SEG]` hidden.

Report:
- mIoU
- Dice
- per-level metrics

### Free-generation validation — PRIMARY

Input only image + instruction.

Generate:
- reasoning + `[SEG]`
- re-forward generated sequence
- locate generated `[SEG]`
- decode SAM2 mask

If zero or multiple `[SEG]`, mark format failure.

Strict end-to-end aggregate:
- invalid/missing `[SEG]` => IoU=0 and Dice=0

Report:
1. strict end-to-end mIoU/Dice
2. conditional mIoU/Dice among valid emissions
3. valid `[SEG]` emission rate

Headline Task 6B metric = strict end-to-end result.

## 14. Validation breakdowns

For the 120 val records report:
- overall
- L1
- L2
- nontrivial L3

Where sample size permits:
- extreme/size
- direction
- nearest
- multi-hop direction->nearest

For each:
- count
- valid `[SEG]` emission
- strict end-to-end mIoU
- conditional mIoU
- Dice
- reasoning exact match
- operation-chain accuracy

## 15. Paired instruction-dependence probe

For each of 20 unseen val image pairs:

`IoU(pred_A, GT_A) > IoU(pred_A, GT_B)`

and

`IoU(pred_B, GT_B) > IoU(pred_B, GT_A)`

If either instruction fails valid `[SEG]` emission, the pair fails.

Report:
- pair passes /20
- mean own-target IoU
- mean cross-target IoU

## 16. Pre-training baseline

Before Phase A, evaluate the fresh architecture on the same fixed 120 val records.

At minimum:
- free-gen valid `[SEG]` rate
- strict end-to-end mIoU
- teacher-forced mask mIoU if meaningful
- reasoning exact match/token accuracy

Final report must show improvement over this baseline.

## 17. Verdict policy

Allowed:
- `PASS`
- `PASS_WITH_WARNINGS`
- `FAIL_REQUIRES_DEBUG`

Minimum for PASS:
1. network cleanup has no insecure workaround
2. no NaN/Inf/OOM
3. val valid `[SEG]` emission >= 90%
4. strict end-to-end val mIoU improves by >= 0.10 absolute over fresh baseline
5. teacher-forced val mIoU is meaningfully above baseline
6. reasoning-token accuracy improves substantially
7. operation-chain accuracy >= 70%
8. paired validation instruction-dependence >= 14/20
9. no GT geometry leakage
10. no test-set tuning

Use PASS_WITH_WARNINGS if segmentation generalization works but language quality remains weak.

Use FAIL_REQUIRES_DEBUG if free-generation still mostly fails, strict end-to-end mIoU does not improve, pair generalization fails badly, mask collapse occurs, or inference-validity is violated.

## 18. Optional SAM2 feature cache

Because the SAM2 image encoder is frozen, a local ignored cache is allowed.

Rules:
- only detached SAM2 image/high-res features
- BF16/FP16 where safe
- under `local_cache/` or `artifacts/`
- <= 8 GB total
- verify cached vs uncached decoder equivalence on multiple samples

Do not cache Qwen hidden states.

## 19. Checkpoints

Save ignored local checkpoints:
- `phaseA_last`
- `best_mask`
- `best_language`
- `best_joint`
- `last`

Do not duplicate base model weights.

Define `best_joint` before Phase B starts using this lexicographic rule:
1. valid `[SEG]` emission rate
2. strict end-to-end val mIoU
3. operation-chain accuracy
4. lower LM CE

Do not change this rule after seeing results.

Create:

`evaluation/task6b_checkpoint_manifest.json`

with checkpoint role/path/hash/bytes/epoch/step/val metrics/base model revisions.

## 20. Visualization pack

Create `evaluation/task6b_samples/` with about 12–16 compact panels:
- successful L1
- successful L2
- successful nontrivial L3
- hard/failure examples
- at least 4 paired-image examples

Each panel:
- source image
- instruction
- generated reasoning
- `[SEG]` validity
- GT mask
- prediction if available
- IoU
- level/query type

Add contact sheet.

## 21. Tests

Extend tests for:
1. normal runtime does not mutate certifi
2. Watt-specific helper not auto-called
3. subset determinism
4. exactly 480 train / 120 val
5. val comes from val only
6. no test data in tuning
7. assistant target ends `[SEG]` then EOS
8. Phase A optimizer excludes projection/SAM decoder
9. Phase B includes them
10. Qwen vision tower remains frozen
11. teacher-forced and free-generation paths are separate
12. strict metric scores generation failure as zero
13. paired probe compares each prediction against both GT masks
14. no GT geometry in inference
15. no `[REF]` in Task 6B

Run:

`python -m pytest tests/ -q`

Report collected/passed/failed/skipped/exit code.

Ordinary pytest must not force large downloads.

## 22. Required artifacts

Create:
- `evaluation/task6b_network_cleanup.json`
- `evaluation/task6b_subset_ids.json`
- `evaluation/task6b_baseline.json`
- `evaluation/task6b_training_report.json`
- `evaluation/task6b_validation.json`
- `evaluation/task6b_checkpoint_manifest.json`
- `docs/task6b_minitrain.md`

`task6b_training_report.json` must include:
- config
- package/model revisions
- Phase A metrics
- every Phase B epoch
- loss components
- learning rates
- gradient norms
- peak VRAM
- runtime
- recipe adjustments

`task6b_validation.json` must include:
- baseline
- final checkpoint
- teacher-forced metrics
- free-generation strict metrics
- conditional metrics
- emission
- language metrics
- per-level breakdown
- paired probe
- exact checkpoint hash

## 23. Documentation / ADR

Update:
- `README.md`
- `docs/architecture_decisions.md`
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

Add a narrow Task 6B measurement ADR only if warranted.

Record:
- whether Watt workaround was removed
- direct TLS status
- mini-train validation behavior
- whether reasoning language is genuinely learned
- whether 2B has enough headroom to justify a later 4B test

Do not freeze `[REF]` yet.

## 24. Git hygiene

Before commit:
1. `git status --short`
2. full tests
3. `python scripts/check_artifact_consistency.py`
4. no model weights staged
5. no `.conda/`
6. no caches/features/checkpoint bytes
7. no BuildSpatialReason JSONL changed
8. no test-set tuning
9. inspect report/visualization sizes

Recommended commit:

`train: validate 2B SEG mini-generalization`

Push:

`git push origin main`

## 25. Handoff

`handoff/FROM_DSH.md` sections:
1. Verdict
2. Network / Watt Cleanup
3. Conda Environment Integrity
4. Direct Hugging Face / GitHub Validation
5. Frozen Architecture
6. Train / Validation Subsets
7. Fresh Baseline
8. Phase A Language Warm-up
9. Phase B Joint Mini-Train
10. Teacher-Forced Validation
11. Free-Generation End-to-End Validation
12. Language / Reasoning Metrics
13. L1 / L2 / L3 Breakdown
14. Paired Instruction Probe
15. VRAM / Runtime
16. Checkpoints
17. Tests
18. Negative Results
19. Git Commit / Push
20. Recommendation for Task 6C

May remain English.

## 26. Final DSH web/chat response — Chinese only

Report concisely:
- Task 6B verdict
- whether Watt-specific TLS workaround was removed
- whether standard Hugging Face/GitHub access works
- train/val sizes
- final free-gen valid `[SEG]` emission
- strict end-to-end val mIoU
- conditional mIoU
- reasoning-token accuracy
- operation-chain accuracy
- paired val result /20
- peak VRAM
- total training time
- commit hash
- push success/failure
- blocker if any

## 27. STOP

After Task 6B, stop.

Do not:
- download/train 4B
- add `[REF]`
- implement Spatial Relation Encoder
- implement Spatial Consistency Loss
- train full 15,592 samples
- use external datasets

Wait for ChatGPT review to decide whether Task 6C is:
- 4B `[SEG]` scale-up
- further 2B recipe correction
- or `[REF]` introduction.
