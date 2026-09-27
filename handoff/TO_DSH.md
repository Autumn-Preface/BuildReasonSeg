# TO_DSH — Task 6D.1: Corrective G0 Rerun + `[SEG]` Spatial-Decodability Audit

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: audit and correct Task 6D before any new architecture is introduced.
>
> Task 6D produced a valuable oracle result, but its causal conclusion
>
> > "`[SEG]` hidden state carries no usable target location"
>
> is **not yet accepted as frozen**, because the G0 training script contains a learning-rate scheduler defect and the grounding head begins with LayerNorm, which may itself remove a decodable signal.
>
> This task must correct those confounds first.
>
> No 4B. No `[REF]`. No Spatial Relation Encoder. No Spatial Consistency Loss. No dataset change. No G1 unless the corrected G0 gate passes.

## 0. User-facing language

All narrative DSH UI/chat output must be **Chinese**.

Code, paths, raw metric names and logs may remain English.

# PART A — Preserve the valid Task 6D evidence

## 1. Oracle result is accepted and must not be rerun unless a correctness issue is discovered

Freeze:

- Oracle Point: mIoU `0.4876`, paired `18/20`;
- Oracle Box: mIoU `0.7506`, paired `20/20`;
- geometry choice: **BOX**.

These results establish:

> Given correct target geometry, the current SAM2 prompt encoder + decoder can select and segment the intended building well.

Do not reinterpret this as proof that every SAM failure is solved; it is specifically an upper-bound diagnostic for the current validation material.

# PART B — Correct the scheduler defect

## 2. Task 6D G0 scheduler is currently wrong

Current `scripts/task6d_train.py` creates:

```python
steps_per_epoch = len(train_samples)
optimizer, scheduler, groups = make_optimizer(runtime, steps_per_epoch)
```

and then reuses that scheduler for **2 epochs**.

With the configured cosine schedule:

```yaml
warmup_steps: 20
lr_schedule: cosine
```

the scheduler reaches its terminal factor at approximately step 480, i.e. the end of epoch 1, and epoch 2 then runs with the learning rate at/near zero.

This means the recorded "2 epochs" are not a valid two-epoch optimization budget.

### Required correction

For a normal multi-epoch run:

```python
total_optimizer_steps = epochs * steps_per_epoch
```

The scheduler must be constructed with the actual total number of optimizer steps.

If a smoke/max-step mode truncates the run, calculate the true total steps from the actual loop, not from nominal epochs.

### Required LR audit

Every training report must record LR for each optimizer group at:

- first optimizer step;
- after warmup;
- end of epoch 1;
- start of epoch 2;
- final optimizer step.

Add a regression test proving for a 2×480 G0 run:

- LR is non-zero at the beginning of epoch 2;
- cosine reaches its terminal value only at the end of the full 960-step schedule.

Do not silently rewrite Task 6D history. Mark the original G0 result as:

`VALID_MEASUREMENT_WITH_SCHEDULER_CONFOUND`

until this task resolves it.

# PART C — Exact corrective rerun first

## 3. G0-R: corrected-scheduler replication

Before changing the architecture, rerun **exactly the Task 6D G0 model**:

```text
[SEG] hidden
→ LayerNorm
→ Linear(2048,512)
→ GELU
→ Linear(512,4)
→ sigmoid/canonical box
```

Keep exactly:

- Task 6C paired P subset: 480 records;
- 2 epochs;
- Qwen3-VL-2B;
- text-only LoRA;
- `[SEG]`;
- BOX;
- lambda_ground = 5.0;
- `2.0*LM CE + 5.0*SmoothL1`;
- same optimizer LRs;
- strict determinism;
- same validation set and paired probe;
- G0 only (SAM mask decoder frozen);
- Task 6C.7 visual cache.

The **only intended change** is the scheduler horizon bug fix.

Report:

- full LR traces;
- train raw grounding loss;
- train box IoU;
- train predicted-box spread;
- validation box IoU;
- geometry paired /20;
- emission;
- hidden/head representation diagnostics.

Use the same original G0 gate:

- emission >= 90%;
- geometry paired >= 14/20;
- mean GT-box IoU >= 0.35.

### If G0-R passes

Verdict:

`SCHEDULER_FIX_RECOVERS_GROUNDING`

Then G1 may run under the original Task 6D rules with the corrected scheduler.

### If G0-R fails

Do **not** immediately invent a new architecture.

Proceed to the decodability audit below.

# PART D — `[SEG]` spatial-decodability audit

## 4. Why this is required

High cosine similarity does **not** prove the target location is absent.

A small target-dependent component can exist in a high-dimensional vector while cosine remains near 1.

Likewise, failure of one MLP readout does not prove information-theoretic absence.

Task 6D also prepended:

```text
LayerNorm(2048)
```

before the grounding MLP.

LayerNorm removes per-sample mean/scale information. If spatial information is partly encoded through those statistics, Task 6D's own head can destroy it.

Therefore explicitly test whether geometry is decodable from the raw hidden state.

## 5. Freeze representation source

Use one clearly identified frozen representation source.

Preferred:

- the Task 6C `P_C` checkpoint before Task 6D grounding training, if the local checkpoint is available and its hash can be verified.

If not available:

- use the clean Task 6D initialization reconstructed from the same seed/base/LoRA state and document the exact source.

Do not use a checkpoint whose representation has already been altered by the failed grounding run unless explicitly labelled as a separate diagnostic.

Extract teacher-forced `[SEG]` hidden vectors for:

- all 480 P training records;
- fixed 120 validation records.

Also record image id, query type/template id, target box and target component.

No test split.

Save only compact feature artifacts if reasonable; large tensors may stay local/gitignored with hashes/manifests committed.

# PART E — Four frozen probes

## 6. Probe A — raw linear readout

Freeze Qwen completely.

Train only:

```text
Linear(2048, 4)
```

from the raw `[SEG]` hidden to normalized GT box coordinates.

Use a simple deterministic optimizer and enough steps to test decodability, not paper performance.

Primary question:

> Can a linear map recover target geometry from the frozen raw hidden representation?

Report train and val:

- SmoothL1;
- box IoU;
- center-inside rate;
- geometry paired /20.

## 7. Probe B — raw MLP readout, NO LayerNorm

Freeze Qwen completely.

Train:

```text
Linear(2048,512)
→ GELU
→ Linear(512,4)
→ sigmoid/canonicalization
```

No LayerNorm.

Use enough iterations to test:
- 20-sample overfit;
- then 480-sample train fit;
- then validation.

The 20-sample overfit is important:

If a 1M-parameter MLP cannot fit 20 frozen hidden vectors to 20 boxes, inspect implementation/data before drawing representation conclusions.

Report:
- 20-sample train box IoU;
- 480-sample train box IoU;
- val box IoU;
- paired /20.

## 8. Probe C — current LayerNorm MLP

Freeze Qwen and train exactly the Task 6D head:

```text
LayerNorm
→ Linear
→ GELU
→ Linear
```

Use the same probe protocol as B.

This isolates whether LayerNorm itself destroys useful signal.

## 9. Probe D — representation-statistics ablation

Measure whether GT geometry correlates with information discarded by LayerNorm:

For each raw `[SEG]` hidden:
- vector mean;
- vector standard deviation;
- L2 norm.

Report basic correlation / predictive usefulness with box center/size.

Do not overinterpret correlation as causation.

# PART F — Stronger representation controls

## 10. Four-condition similarity control, expanded

Repeat:

- same image / different instruction;
- different image / same template;
- different image / different template.

But report more than cosine:

- cosine;
- L2;
- raw norm;
- centered cosine;
- LayerNorm-output cosine;
- within-template variance;
- within-image variance;
- between-target box distance.

If practical, use PCA/SVD on raw `[SEG]` hiddens and report:
- effective rank;
- top-1/top-5/top-10 explained variance.

The purpose is not to prove reasoning, but to determine what information the token actually carries.

# PART G — Interpretation matrix

## 11. Predeclare conclusions

### Case A — corrected G0-R passes

Task 6D failure was materially caused by the scheduler defect.

Do not redesign representation yet.

### Case B — raw probes decode geometry, LayerNorm probe fails

Conclusion:

`LAYER_NORM_READOUT_CONFOUND`

The `[SEG]` state contains usable location information, but Task 6D's head destroyed/suppressed it.

Next architecture should use the successful raw readout.

### Case C — raw MLP can fit train/20-sample but not validation

Conclusion:

`LOCATION_SIGNAL_MEMORIZABLE_NOT_GENERALIZABLE`

The representation contains sample-specific information but not a robust spatial code.

This justifies changing the representation objective, but not saying "no location exists".

### Case D — even raw 1M MLP cannot overfit 20 frozen hidden vectors

Conclusion:

`READOUT_OR_FEATURE_IDENTITY_BUG_SUSPECTED`

Do not make an information-content claim until implementation is audited.

### Case E — raw MLP easily overfits 20 and 480 but val remains collapsed

Conclusion:

`SPATIAL_REPRESENTATION_DOES_NOT_GENERALIZE`

This is strong evidence that current `[SEG]` features are unsuitable as a generalizable geometry code.

### Case F — raw probes cannot meaningfully fit even 480 after implementation is verified

Then and only then may the project freeze the stronger statement:

> Current `[SEG]` representation contains no practically decodable target geometry under this training setup.

Use `PRACTICALLY_DECODABLE`, not information-theoretic "contains no information".

# PART H — If representation truly fails: one next-method recommendation only

## 12. Do not implement it automatically

If Case E/F is established, recommend exactly one of these for the next task, with evidence:

### Option 1 — explicit coordinate/grid tokens

Make target location a first-class supervised sequence target, e.g. quantized spatial tokens / coordinate tokens, rather than expecting the generic `[SEG]` hidden state to spontaneously encode geometry.

### Option 2 — target-aware query token before reasoning completion

Introduce a dedicated trainable grounding/query token whose hidden state is directly supervised for box geometry.

### Option 3 — `[REF]`/relation mechanism

Only if the evidence specifically shows multi-hop relation grounding needs a reference object.

Do not implement 4B as the first response.

Do not implement multiple options in the same task.

# PART I — Dataset policy

## 13. WHU decision remains open but unchanged in this corrective task

Do not change dataset.

Task 6D oracle box = 0.7506 / paired 20/20 is evidence that the current validation examples are sufficient to test this architecture pathway.

However, do not declare WHU permanently adequate.

Record separately:
- pseudo-instance ambiguity;
- touching/merged buildings;
- border truncation;
- tiny-target prevalence.

Dataset migration remains allowed later if architecture results show it is beneficial.

# PART J — Required artifacts

## 14. Create

```text
evaluation/task6d1_scheduler_audit.json
evaluation/task6d1_g0_corrected.json
evaluation/task6d1_hidden_extract_manifest.json
evaluation/task6d1_probe_linear.json
evaluation/task6d1_probe_raw_mlp.json
evaluation/task6d1_probe_layernorm_mlp.json
evaluation/task6d1_representation_stats.json
evaluation/task6d1_decodability_summary.json
docs/task6d1_corrective_grounding_audit.md
```

If G0-R passes and G1 is allowed:

```text
evaluation/task6d1_g1.json
evaluation/task6d1_paired_probe.json
```

Large extracted hidden tensors stay local/gitignored; commit their:
- shape;
- dtype;
- sample-id hash;
- SHA256 if serialized.

# PART K — Tests

## 15. Required tests

Add/regress tests for:

1. scheduler total steps equals actual optimizer-step budget;
2. epoch-2 initial LR is non-zero for the standard 2×480 G0;
3. terminal cosine LR occurs only at final full-run step;
4. smoke/max-step mode uses its actual step budget;
5. G0-R differs from original Task 6D only by scheduler correction;
6. frozen probe extraction does not update Qwen;
7. probe A uses raw hidden with no LayerNorm;
8. probe B has no LayerNorm;
9. probe C exactly includes LayerNorm;
10. probe labels are GT boxes used only in probe supervision;
11. no test split;
12. strict determinism;
13. no 4B / `[REF]` / SRE / SCL;
14. no dataset change;
15. no G1 unless corrected G0 gate passes.

Run:

`python -m pytest tests/ -q`

# PART L — Documentation correction

## 16. Amend Task 6D causal wording

Do not erase Task 6D numbers.

Update project state / ADR / handoff language from:

> "`[SEG]` hidden carries no target location"

to:

> "Task 6D's first grounding readout collapsed; the original result is confounded by a scheduler-horizon defect and by the use of LayerNorm before the readout. Spatial decodability is under corrective audit."

Only restore a stronger conclusion if Task 6D.1 supports it.

The oracle conclusion may remain frozen.

# PART M — Git / Watt

## 17. Git hygiene

No weights/checkpoints/hidden-tensor dumps/local caches.

Recommended commit:

`fix: audit spatial grounding representation`

Ignore UU.

Use normal Watt ownership rules for final push.

# PART N — Final DSH UI response

## 18. Chinese-only final summary

Report:

- Task 6D.1 verdict;
- scheduler bug confirmation and exact LR behavior before/after;
- corrected G0 paired /20 and box IoU;
- Probe A train/val;
- Probe B 20-sample overfit, 480-train, val;
- Probe C corresponding results;
- whether LayerNorm is a confound;
- raw hidden similarity/effective-rank summary;
- the accepted causal conclusion;
- whether G1 ran;
- the single recommended next architecture direction;
- tests;
- commit/push;
- Watt handling.

# 19. STOP

After Task 6D.1:

**STOP.**

Do not automatically:
- change dataset;
- scale to 4B;
- add `[REF]`;
- add Spatial Relation Encoder;
- add Spatial Consistency Loss;
- build GUI;
- run full training.

Wait for ChatGPT review.
