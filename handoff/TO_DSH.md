# TO_DSH — Task 6C: Paired Counterfactual Training × Neutral SAM Prompt Ablation

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: diagnose and fix the Task 6B instruction-conditioning failure **before** any 4B scale-up or `[REF]` work.
>
> Core question:
>
> Did Task 6B fail because the training subset never forced two different instructions on the same image to select two different targets, because the SAM bridge injected a fixed positive centre point, or because both effects interact?
>
> Task 6C is a controlled **2 × 2 ablation** on the existing 2B stack.
>
> No 4B. No `[REF]`. No Spatial Relation Encoder. No Spatial Consistency Loss. No full-dataset training.

## 0. User-facing language

All narrative text shown in the DSH web/chat UI must be **Chinese**.

Commands, paths, model IDs, raw logs, code identifiers and metric names may remain English.

`handoff/FROM_DSH.md` and `handoff/PROJECT_STATE.md` may remain English.

# PART A — Fixed project / environment rules

## 1. Reuse the existing Conda environment

Use exactly:

`C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp`

Do not create a new environment.

Do not modify Conda `base`, `yolo_sam_env`, system Python, drivers, CUDA toolkit, WSL, registry, system PATH.

Task 6C should run from already cached model assets.

## 2. Watt Toolkit network policy — `FULL_AUTO_OK`

The standalone lifecycle test is accepted.

Current user settings:
- `TrayIcon = false`
- `MinimizeOnStartup = false`
- `ProgramStartupRunProxy = true`

### Start

Use the proven Microsoft Store launch path:

```powershell
Start-Process explorer.exe -ArgumentList 'shell:AppsFolder\4651ED44255E.47979655102CE_k6txddmbb6c52!App'
```

Then verify the Watt main window appears, `Steam++.Accelerator.exe` becomes active, ports 443/80 are owned by the accelerator, Watt's hosts block appears, and the required network operation succeeds.

### Stop

Do **not** use `.NET CloseMainWindow()` / bare `WM_CLOSE`.

Use the tested real window-close action:

```text
PostMessage(
    hWnd of window "Watt Toolkit",
    WM_SYSCOMMAND = 0x0112,
    SC_CLOSE      = 0xF060,
    0
)
```

After stop verify:
- no Watt / Steam++ process remains;
- ports 443/80 and previously swept proxy ports are free;
- Steam++ hosts block is gone;
- hosts returns to baseline.

Never use `taskkill /F`, `Stop-Process -Force`, direct hosts edits, certificate-store edits, or TLS `verify=False`.

### Important TLS rule

While Watt is active, Git/WinHTTP works, but Python `certifi` still fails against Watt's TLS interception.

Do not reintroduce the old certificate-injection workaround.

Task 6C itself should run **offline** from `local_cache/`.

Use Watt only for a networking step that actually needs it, especially final `git push`.

# PART B — Correctness fixes before the experiment

## 3. Task 6B causal conclusion is provisional

Task 6B measured:
- strict e2e val mIoU ≈ 0.11018;
- paired validation = 0 / 20;
- projected prompt cosine ≈ 0.999952 on same-image/different-instruction pairs.

But the statement "`[SEG]→prompt` is the only bottleneck and the mask decoder is fine" is not yet fully established because Task 6B had two confounds:

1. its 480 training records came from 480 distinct images — no same-image / different-instruction counterfactual training;
2. the SAM bridge used a fixed **positive centre point** `(0.5, 0.5)` and then added the projected language vector to that point embedding.

Amend ADR-014:
- preserve every Task 6B measurement;
- mark the causal diagnosis as **PROVISIONAL pending Task 6C ablation**;
- explicitly record these two confounds.

Do not erase Task 6B results.

## 4. Fix reproducibility / bookkeeping defects first

### 4.1 Determinism flag must become real

Implement actual deterministic setup for `training.deterministic`.

At minimum:
- Python `random` seed;
- NumPy seed;
- `torch.manual_seed`;
- `torch.cuda.manual_seed_all`;
- `torch.backends.cudnn.benchmark = False`;
- `torch.backends.cudnn.deterministic = True`;
- `torch.use_deterministic_algorithms(True)` if the complete path supports it.

Run a tiny 2-sample forward/backward deterministic smoke twice.

If strict deterministic algorithms fail because an operation has no deterministic implementation:
- record the exact operation/error;
- use `torch.use_deterministic_algorithms(True, warn_only=True)` as fallback;
- state clearly that bit reproducibility is not guaranteed.

Do not silently claim determinism.

### 4.2 Checkpoint directory must follow config

Remove Task 6B's hard-coded checkpoint-directory behavior.

Every arm must have a distinct config-driven directory:

```text
artifacts/checkpoints/task6c/U_C/
artifacts/checkpoints/task6c/U_L/
artifacts/checkpoints/task6c/P_C/
artifacts/checkpoints/task6c/P_L/
```

No arm may overwrite another. Add regression tests.

### 4.3 Evaluation lookup must not depend on val text

Do not build operation-chain scoring rules from `train + val`.

Build the lookup from:
- frozen template definitions if practical; otherwise
- **train split only**.

Validation annotations may provide the expected query type for scoring, but may not extend the accepted wording/operation lookup.

### 4.4 Feature cache

Increase SAM2 CPU feature cache to cover the whole 480-record train subset when memory budget permits.

Target:
- cache up to 480 images;
- total cache <= 8 GB;
- record actual RAM footprint.

This is a speed change only.

# PART C — Two experimental factors

## 5. Factor 1: training-sample structure

### `U` — Unpaired / unique-image training

Use the exact Task 6B training subset:
- 480 records;
- 480 unique images;
- 160 L1;
- 160 L2;
- 160 nontrivial L3.

Do not change its sample IDs.

### `P` — Paired counterfactual training

Create a new deterministic 480-record subset:
- **240 unique images × 2 instructions per image**;
- every pair must have different target component IDs;
- every pair must have different query types;
- no trivial L3.

Construct pairs so record-level level counts are exactly:
- 160 L1
- 160 L2
- 160 nontrivial L3

Preferred exact pair composition:
- 80 images: one L1 + one L2;
- 80 images: one L1 + one nontrivial L3;
- 80 images: one L2 + one nontrivial L3.

This yields exactly 160 records per level.

Within those constraints balance L1 query types, L2 families/references/directions and L3 directions as evenly as practical.

Selection must be deterministic and independent of model results.

If exact 80/80/80 is impossible, prove why from the dataset and choose the nearest feasible deterministic composition; do not silently relax it.

Create:

`evaluation/task6c_subset_ids.json`

containing both U and P subsets and their audits.

## 6. Factor 2: SAM sparse-prompt bridge

### `C` — Current centre-positive bridge

Preserve Task 6B:

```text
fixed point at normalized (0.5, 0.5), label = positive
    -> SAM prompt encoder point embedding
    -> add projected [SEG] vector
    -> SAM mask decoder
```

This is a valid baseline but must no longer be described as "content-free" or "neutral". It is a real positive spatial prior.

### `L` — Language-only sparse token

Implement a second bridge with **no point prompt at all**:

```python
sparse_prompt_embeddings = projected[:, None, :]
```

subject only to dtype/device/shape expected by vanilla SAM2.

Continue to use the official SAM prompt encoder for:
- image positional encoding;
- no-mask dense embedding.

Do not create point coordinates, point labels, boxes, or mask prompts.

No GT geometry enters inference.

Keep `multimask_output=False`, same SAM2 image/high-res features, same decoder.

### Do not add other bridge changes in Task 6C

Do not simultaneously add:
- prompt normalization;
- whitening;
- learned prompt-type tokens;
- multiple sparse tokens;
- `[REF]`;
- geometry supervision;
- point-inside auxiliary loss.

# PART D — The 2×2 experiment

## 7. Four arms

Run exactly:

| Arm | Train subset | SAM bridge |
|---|---|---|
| `U_C` | unpaired 480 unique images | centre-positive |
| `U_L` | unpaired 480 unique images | language-only |
| `P_C` | 240 paired images × 2 | centre-positive |
| `P_L` | 240 paired images × 2 | language-only |

All four share:
- same Qwen3-VL-2B base snapshot;
- same SAM2.1 Base+ checkpoint;
- same tokenizer/vocab and `[SEG]`;
- same initialization seed;
- same optimizer recipe;
- same loss weights;
- same epoch counts;
- same validation set;
- same paired validation probe;
- same metric code.

The two factors above must be the only intended experimental differences.

## 8. Initialization equality

Every arm starts from clean base, not Task 6A/6B checkpoint.

Before training:
- fresh LoRA;
- fresh `[SEG]` trainable token adapter;
- fresh Projection MLP;
- original pretrained SAM2 mask decoder.

Use the same random seed.

Compute SHA256/fingerprint over all **initial trainable parameter tensors** before first optimizer step.

All four arms should have the same initial-trainable-state fingerprint.

If not, stop and fix before training.

## 9. Training recipe — fixed, no tuning

Use Task 6B headline recipe, no adjusted-run tuning.

### Phase A
- 1 epoch;
- LoRA + `[SEG]` only;
- LM CE only.

### Phase B
- maximum **3 epochs**;
- train LoRA + `[SEG]` + Projection MLP + SAM2 mask decoder;
- base/vision encoders frozen.

Loss:

```text
L_total =
    2.0 * L_lm_ce
  + 2.0 * L_mask_bce
  + 1.0 * L_mask_dice
```

Use same LR groups as corrected Task 6B headline implementation.

No recipe adjustment in Task 6C. This is causal diagnosis, not hyperparameter search.

# PART E — Validation

## 10. Reuse exact Task 6B validation material

Use exactly:
- 120-record validation subset: 40 L1 + 40 L2 + 40 nontrivial L3;
- 20 same-image/different-instruction validation pairs.

Do not select a new validation subset. Do not use test.

## 11. End-to-end validation

For each arm:

### Free generation — primary

Input only image + instruction.

Then:
- generate reasoning + `[SEG]`;
- require exactly one `[SEG]`;
- re-forward generated sequence;
- read generated `[SEG]` hidden;
- project;
- decode mask.

Invalid `[SEG]` generation scores IoU/Dice = 0.

Report:
- valid `[SEG]` emission;
- strict e2e mIoU;
- strict Dice;
- conditional mIoU;
- L1/L2/L3 breakdown;
- query-family breakdown.

### Teacher-forced

Report same mask metrics with expected reasoning prefix. Diagnostic only.

## 12. Paired instruction-dependence remains primary gate

For the same 20 unseen validation pairs:

```text
IoU(pred_A, GT_A) > IoU(pred_A, GT_B)
IoU(pred_B, GT_B) > IoU(pred_B, GT_A)
```

Both directions must pass.

If either member fails valid `[SEG]`, pair fails.

For every arm report:
- passed /20;
- mean own-target IoU;
- mean cross-target IoU;
- mean own-minus-cross margin;
- median own-minus-cross margin;
- mean IoU(pred_A, pred_B).

# PART F — Prompt-pathway diagnosis

## 13. Diagnose all four arms identically

Measure on at least:
- 10 same-image/different-instruction pairs;
- 10 different-image references.

At each stage:

### `[SEG]` hidden, 2048-d
Report cosine similarity, L2 distance, vector norm.

### Projection output, 256-d
Report cosine similarity, L2 distance, vector norm, per-dimension std over diagnosis set, mean absolute activation.

### Final sparse prompt actually passed to SAM
Report cosine similarity, L2 distance, norm.

For `C`, also report:
- norm of point-derived prompt embedding before language addition;
- norm of projected language vector;
- ratio `||projected|| / ||point_embedding||`.

For `L`, final sparse prompt should equal projected language token apart from dtype conversion.

### Dataset-level collapse diagnostics

Over a fixed diagnosis set compute:
- centered covariance or SVD of projected vectors;
- top 10 singular values;
- variance explained by top 1/top 5/top 10;
- effective-rank metric.

Do not call a representation "constant" solely from cosine if L2/norm variation is substantial.

Use measured terminology such as directional collapse, low-rank collapse, or near-constant direction.

### Mask output

For paired samples report IoU(pred_A, pred_B).

# PART G — Interpretation rules

## 14. Pre-declare causal interpretation

- If `P_*` >> `U_*` regardless of bridge: evidence favors **missing counterfactual paired training**.
- If `*_L` >> `*_C` regardless of sampling: evidence favors the **fixed centre-positive prompt**.
- If only `P_L` works: evidence favors a strong **interaction**.
- If none work: problem is deeper in `[SEG]` representation/projection/SAM conditioning; only then consider later normalization, multi-token prompt, auxiliary supervision, or `[REF]`.
- If all work similarly: Task 6B failure was likely dominated by stochastic/implementation defects; verify reproducibility before architecture claims.

## 15. Task 6C verdict

Use one of:
- `EXPERIMENT_COMPLETE_FIX_FOUND`
- `EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`
- `EXPERIMENT_COMPLETE_NO_FIX`
- `FAIL_EXPERIMENT_INVALID`

`EXPERIMENT_COMPLETE_FIX_FOUND` requires at least one arm:
- paired validation >= **14/20**;
- mean own-minus-cross margin > **0.05**;
- valid `[SEG]` emission >= 90%;
- strict e2e mIoU >= **0.11**;
- no GT leakage.

`EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`: no arm clears full gate, but at least one intervention materially improves paired probe or prompt diversity over `U_C`.

`EXPERIMENT_COMPLETE_NO_FIX`: all four remain essentially collapsed / paired probe remains very poor.

`FAIL_EXPERIMENT_INVALID`: unequal initialization, overwritten checkpoints, invalid subset, severe nondeterminism, leakage, or mismatched code paths make comparison invalid.

# PART H — Language-metric caution

## 16. Do not claim reasoning from template exact-match

Preserve Task 6B caveat.

Reasoning exact match = template/format performance.

Operation-chain accuracy is useful but still not sufficient evidence of visual reasoning.

**Mask target selection on paired unseen images is the main reasoning evidence.**

Use wording like:
> language-format / template mapping transfers

not:
> the MLLM has learned spatial reasoning

unless mask/relation behavior supports it.

# PART I — Tests and artifacts

## 17. Tests

Add/regress tests for:
1. `training.deterministic` is consumed;
2. deterministic 2-sample smoke repeated twice;
3. checkpoint path comes from config;
4. four arms have unique checkpoint dirs;
5. trainable-state fingerprint equality;
6. U subset matches Task 6B exactly;
7. P subset = 480 records / 240 images / 2 records per image;
8. every P pair has different targets;
9. every P pair has different query types;
10. P level counts = 160/160/160 unless feasibility exception is proven;
11. no trivial L3 in P;
12. val subset identical to Task 6B;
13. paired val identical to Task 6B;
14. operation lookup built without val text;
15. centre bridge remains Task 6B behavior;
16. language-only bridge contains no point/box/mask prompt;
17. language-only sparse tensor shape correct;
18. no GT geometry in inference;
19. no `[REF]`;
20. no 4B.

Run:
`python -m pytest tests/ -q`

Report collected/passed/failed/skipped/exit code.

Ordinary pytest must not download model weights.

## 18. Required artifacts

Create:

```text
evaluation/task6c_subset_ids.json
evaluation/task6c_determinism.json
evaluation/task6c_initialization.json

evaluation/task6c_U_C.json
evaluation/task6c_U_L.json
evaluation/task6c_P_C.json
evaluation/task6c_P_L.json

evaluation/task6c_prompt_U_C.json
evaluation/task6c_prompt_U_L.json
evaluation/task6c_prompt_P_C.json
evaluation/task6c_prompt_P_L.json

evaluation/task6c_comparison.json
evaluation/task6c_checkpoint_manifest.json

docs/task6c_prompt_ablation.md
```

`task6c_comparison.json` must contain:
- exact factor definitions;
- initialization fingerprints;
- subset hashes;
- all primary metrics;
- paired metrics;
- prompt diagnostics;
- deterministic status;
- winner/no-winner;
- causal interpretation under §14;
- final Task 6C verdict.

## 19. Visualizations

Create compact `evaluation/task6c_samples/`.

For at least 4 fixed paired val images show four-arm comparisons:
- source image;
- instruction A/B;
- GT A/B;
- U_C predictions;
- U_L predictions;
- P_C predictions;
- P_L predictions.

Optionally add one small PCA/singular-value diagnostic figure.

Do not overproduce large images.

# PART J — Documentation / ADR

## 20. ADR update

Update ADR-014:
- preserve Task 6B measurements;
- mark causal diagnosis provisional until Task 6C;
- record unique-image training confound and fixed positive-centre prompt confound.

After Task 6C add ADR-015 containing only what the 2×2 experiment establishes.

Do not claim `[REF]` novelty, Spatial Relation Encoder success, 4B superiority, or reasoning success beyond evidence.

# PART K — Git / Watt finish workflow

## 21. Training/evaluation: Watt OFF

Keep Watt closed during local training/evaluation.

Use cached Qwen/SAM2 and offline HF/Transformers mode if needed.

## 22. Final push: Watt may be automated

After tests, artifacts, and local commit are complete:

If normal `git push` is unavailable/unreliable, DSH is authorized to:

1. start Watt automatically using the proven launch command;
2. verify acceleration active;
3. `git push origin main`;
4. close Watt with `WM_SYSCOMMAND / SC_CLOSE`;
5. verify processes/listeners/hosts returned to baseline.

Do not leave Watt running after push unless shutdown verification fails.

If shutdown verification fails:
- do not force-kill;
- tell the user immediately.

Recommended commit:

`experiment: isolate prompt conditioning failure`

## 23. Git hygiene

Before commit:
- no model weights staged;
- no checkpoints staged;
- no `.conda/`;
- no `local_cache/`;
- no feature-cache tensors;
- no BuildSpatialReason JSONL changed;
- no test data used;
- no Watt settings/system files committed.

Run:
- full pytest;
- `python scripts/check_artifact_consistency.py`.

# PART L — Handoff

## 24. `handoff/FROM_DSH.md`

Include:
1. Verdict
2. Correctness Fixes
3. Determinism Status
4. Subset Construction
5. Four Experimental Arms
6. Initialization Equality
7. U_C Result
8. U_L Result
9. P_C Result
10. P_L Result
11. Paired Validation Comparison
12. Prompt Representation Diagnostics
13. 2×2 Causal Interpretation
14. Language Metric Caveat
15. Runtime / VRAM
16. Tests
17. Git / Watt Push Lifecycle
18. Negative Results
19. Recommendation for Task 6D

## 25. Final DSH web/chat response — Chinese only

Report concisely:
- Task 6C verdict;
- whether deterministic mode is genuinely active;
- P subset exact composition;
- four-arm strict e2e mIoU;
- four-arm paired pass /20;
- four-arm own-minus-cross margins;
- four-arm projected-prompt cosine/effective-rank summary;
- which factor helped most;
- whether P_L or another arm clears fix gate;
- peak VRAM;
- total runtime;
- commit hash;
- push success;
- whether Watt auto-start/auto-close succeeded for push.

Do not paste full report.

# 26. STOP

After Task 6C: **STOP.**

Do not download/run 4B, add `[REF]`, implement Spatial Relation Encoder, implement Spatial Consistency Loss, train full 15,592 records, or add external datasets.

Wait for ChatGPT review.
