# CURRENT_TASK - TASK8D_ABOVE_TARGET_CHAIN_DIAGNOSIS_V1

Status:
READY_FOR_SUPERVISOR_AUDIT

Decision owner: ChatGPT Supervisor
Executor: CODEX
Next gate: CHATGPT_TASK8D_ABOVE_CHAIN_DIAGNOSIS_REMOTE_AUDIT

## Supervisor task authorization - verbatim

TASK AUTHORIZATION

Task ID:
TASK8D_ABOVE_TARGET_CHAIN_DIAGNOSIS_V1

Model:
GPT-6.1 Sol

Reasoning:
极高

Mode:
DIAGNOSIS ONLY — NO REPAIR

Accepted predecessor:
TASK8C_FINAL_DEMO_V1 = ACCEPT

Accepted predecessor branch:
eval/task8c-final-demo-v1

Accepted remote HEAD:
764a805ef7d9651bfffd27dfaeb5808e8c21df1f

Required new branch:
diag/task8d-above-target-chain-v1

Next gate:
CHATGPT_TASK8D_ABOVE_CHAIN_DIAGNOSIS_REMOTE_AUDIT


# 1. Purpose

This task performs a post-Final-Demo scientific diagnosis of the cleanest frozen failure:

ABOVE case

sample_id:
buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314

program:
largest_to_above_to_nearest

canonical GT reference:
tile_instance_id = 4

canonical GT target:
tile_instance_id = 6

Frozen Task 8C result:

runtime = SUCCESS
automatic reference proposal id = 5

selected-reference best GT instance = 4
reference IoU with canonical GT reference = 0.9097432024169184
reference identity match = true

predicted target best-overlap GT instance = 7
predicted target best GT IoU = 0.48158096699923253

canonical target IoU = 0.07762201453790239
canonical target Dice = 0.14406167188629246

semantic status:
TARGET_IDENTITY_MISMATCH


This case is selected because:

Reference identity is already correct,
but Target identity is wrong.

Therefore it allows diagnosis of the downstream:

Reference
→ geometric relation representation
→ nearest representation
→ field composition / competition
→ visual prototype similarity
→ decoder
→ final dense mask

without REF-01 identity failure as the primary confounder.


# 2. Scientific correction / frozen understanding

Do NOT describe the current RC1 target stage as:

"select one target proposal"

That is not the current frozen implementation.

The actual RC1 core chain is:

reference mask
→ P_dir
→ P_near
→ W = clamp(P_dir * P_near)
→ A_fixed = W / sum(W)
→ projected SAM2 visual features F
→ q = Σ A_i F_i
→ C = cosine(F_i, q)
→ D-B1 decoder
→ logits
→ bilinear upsample
→ logits > 0
→ dense target mask

Relevant frozen source:

delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py

delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/
task7d_global_competition_decoder.py

delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/
task6z_field_composition.py

delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/
geometric_relation_field_v02.py

delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/
nearest_boundary_field.py


# 3. Absolute prohibitions

This milestone MUST NOT:

- rerun Task 8C formal runner
- execute predict.py
- execute DetectorRuntime
- execute Sam2Runtime
- execute Db1Runtime
- execute Qwen / ProgramHead
- load any model checkpoint for inference
- invoke Ultralytics
- invoke Transformers inference
- use --reference-id
- use --inspect-proposals
- retry any Final Demo case
- change any Task 8C artifact
- change any external RC1 formal output
- change product source
- change model weights
- change thresholds
- change ranking
- train anything
- modify governance
- implement any repair

No new model inference of any kind is authorized.

This is an offline forensic / scientific diagnosis only.


# 4. Git preflight

Read in canonical order:

1. AGENTS.md
2. governance/PROJECT_STATE.yaml
3. governance/DECISIONS.md
4. handoff/CURRENT_TASK.md
5. handoff/EXECUTOR_STATE.yaml

Then:

git fetch

Verify:

origin/eval/task8c-final-demo-v1
=
764a805ef7d9651bfffd27dfaeb5808e8c21df1f

Create:

diag/task8d-above-target-chain-v1

from that exact accepted HEAD.

Unknown branch/HEAD/diff:
inspect then STOP.

No reset --hard / rebase / amend / stash / clean / force push.


# 5. Frozen evidence lock

Task 8C evidence is READ ONLY.

Read:

evaluation/task8c_final_demo_v1.json

Locate the ABOVE frozen case and verify all recorded artifact identities.

The external frozen run root is expected to be:

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\
inference\output\1008

Required frozen files include at least:

diagnostics/result.json
diagnostics/proposals.json
diagnostics/reference_context_mask.png
diagnostics/maps.npz
masks/1008_mask.png
overlays/1008_overlay.png

Before analysis:

- verify each file against the SHA256 identity recorded in Task 8C;
- verify runtime_results.json remains:
  SHA256 =
  2fcbb705acf1ceeb28fc9609bfda869150e7b800b2bea0d9fba9a723c12ff8ca
- verify Task 8C evidence files are not modified.

If any frozen identity differs:
STOP.


# 6. Primary diagnostic question

Answer:

At what earliest observable stage does the canonical target
GT instance 6
lose to the eventual wrong building
GT instance 7?

The diagnosis must follow:

Reference geometry
→ P_dir
→ P_near
→ W
→ A
→ C
→ logits
→ probability
→ final mask


# 7. Ground truth

Use the exact same frozen WHU native-vector truth used by Task 8C.

Do not regenerate annotations.

Canonical:

reference = instance 4
target = instance 6

Frozen observed wrong best-overlap target:
instance 7

Verify these identities before analysis.

Also reconstruct all GT building-instance masks in tile 1008,
because candidate-level rankings must be reported for the full relevant set,
not only instance 6 and 7.


# 8. Context mapping

Read frozen:

result.json -> reasoning_context.origin / size

Map native GT masks into the exact frozen reasoning context.

Verify:

- canonical ref 4 is represented;
- canonical target 6 is represented;
- wrong instance 7 is represented;
- no mapping/resizing mismatch;
- final Task8C target mask remains byte-identical.

No repainting or manual mask editing.


# 9. Frozen map analysis

Load ONLY the already-saved:

diagnostics/maps.npz

It must contain:

P_dir
P_near
W
A
C
logits
probability

Verify shapes and finite values.

Do not recompute these seven frozen automatic-reference maps.

They are the primary evidence.


# 10. Candidate-mask aggregation rule

For 512x512 maps:

use exact native GT masks directly.

For 64x64 maps:

downsample each binary GT mask with area interpolation to a fractional occupancy mask.

Do NOT threshold the fractional mask.

For a 64x64 map X and GT instance occupancy M:

mass(X,M) =
sum(X * M)

mean(X,M) =
sum(X * M) / sum(M)

For A specifically:

candidate_attention_mass =
sum(A * M)

because A is the frozen global competition distribution.

For W:

candidate_W_mass =
sum(W * M)

For C:

candidate_C_mean =
sum(C * M) / sum(M)

For 512 logits/probability:

candidate_logit_mean =
mean(logits over exact GT mask)

candidate_probability_mean =
mean(probability over exact GT mask)

candidate_positive_logit_fraction =
fraction of exact GT pixels where logits > 0

For final predicted binary mask:

compute IoU with every GT instance.

No new acceptance threshold may be introduced.


# 11. Required per-instance table

For every GT building instance inside the reasoning context, record:

instance_id
GT area
relation validity using the existing canonical relation implementation
canonical boundary distance to reference if the existing frozen implementation exposes it

P_dir mass / mean
P_near mass / mean
W mass / mean
A attention mass
C mean
logit mean
probability mean
positive-logit fraction
final predicted-mask IoU

Important:

For canonical relation validity and boundary distance,
reuse the existing canonical repository implementation.

Do NOT invent a new geometry predicate or distance formula.

Locate and record the exact function/module used.

If no authoritative canonical implementation can be located:
record that field as NOT_ESTABLISHED rather than inventing one.


# 12. Target 6 vs wrong instance 7 comparison

Produce an explicit stage table:

stage
score_target6
score_wrong7
delta_6_minus_7
which_instance_is_favoured

At minimum:

P_dir
P_near
W mass
A mass
C mean
logit mean
probability mean
positive-logit fraction
final mask IoU

No threshold.

"favoured" simply means numerical comparison under the predeclared statistic.


# 13. Rank trajectory

For every relevant GT instance, rank the appropriate stage statistic.

Report the ranks of:

canonical target 6
wrong instance 7

for:

W mass
A mass
C mean
logit mean
probability mean
final mask IoU

Also report the top-5 instance ids at every stage.

Do not change scoring statistics after viewing the result.


# 14. First-divergence rule

Determine:

FIRST_OBSERVED_TARGET_DIVERGENCE_STAGE

using the fixed stage order:

W
→ A
→ C
→ logits
→ probability
→ final mask

The first stage where instance 7 is favoured over canonical instance 6
under the predeclared statistic
is the first observed divergence.

If neither 6 nor 7 is the main competitor at that stage,
report that fact and the actual top instance.

Do not force a root-cause label.


# 15. Reference-shape counterfactual — deterministic fields only

This is NOT model inference.

Use canonical GT reference mask instance 4 and the exact frozen deterministic field code to recompute ONLY:

P_dir_GTref
P_near_GTref
W_GTref
A_GTref = W_GTref / sum(W_GTref)

No SAM2.
No q.
No C.
No decoder.
No checkpoint.

Compare automatic-reference fields versus canonical-GT-reference fields.

For target 6 and wrong 7 report:

W mass
A mass
delta target6-minus-wrong7

Also report:

field difference statistics
and rank changes.

Purpose:

test whether the ~0.91-but-not-perfect automatic reference mask
is sufficient to explain the downstream geometric preference.

Interpretation must remain conservative:

- if GT-reference fields restore target6 preference,
  reference-shape residual is a plausible contributor;
- if GT-reference fields still favour wrong7,
  reference-shape residual cannot be the sole explanation;
- otherwise:
  MIXED / NOT RESOLVED.

Do not call this a causal proof.


# 16. Frozen D-B1 mechanism audit

Using source inspection only, document exactly:

W
A_fixed
q
C
decoder inputs
output threshold

For D-B1 specifically establish from source whether:

- A is learned or deterministic;
- q is derived from A-weighted projected visual features;
- C is cosine similarity;
- target proposals are or are not used;
- graph construction is or is not used;
- the final mask is selected as an instance or emitted densely;
- the final hard runtime validity gate checks target identity, nearest identity,
  or only structural/directional properties.

Every conclusion must cite exact source path + line/function in the report.


# 17. Runtime guard audit

Inspect:

buildreasonseg/runtime/context.py
buildreasonseg/runtime/pipeline.py

and the frozen ABOVE result's:

directional_guard

Answer:

What exactly must a wrong target satisfy in order for runtime to still return SUCCESS?

Specifically distinguish:

- direction availability guard;
- final centroid direction check;
- nearest-target identity correctness;
- GT identity correctness.

Do not infer beyond the source.


# 18. Required visual evidence

Create:

evaluation/task8d_above_target_chain_diagnosis_v1/

with at least:

01_reference_and_gt.png
02_P_dir.png
03_P_near.png
04_W_and_A.png
05_C.png
06_logits_probability.png
07_final_mask_vs_gt.png
08_stage_trajectory.png
contact_sheet.png

All diagnostic maps must show boundaries for:

GT reference 4
GT target 6
wrong instance 7

using a clear legend.

Do not alter underlying values for display.

Heatmaps may normalize only for visualization;
raw quantitative calculations must use frozen arrays.


# 19. Required diagnosis JSON

Create:

evaluation/task8d_above_target_chain_diagnosis_v1.json

Required top-level fields:

task_id
status
accepted_predecessor
task8c_head
task8c_runtime_identity
above_artifact_identity_verification

source_paths
source_function_trace

sample_lock
reference_lock
target_lock
wrong_instance_lock

reasoning_context

frozen_maps_identity
frozen_map_shapes

all_gt_instance_stage_table
target6_vs_wrong7
rank_trajectory
first_observed_target_divergence_stage

canonical_reference_field_counterfactual

runtime_guard_audit
db1_mechanism_audit

model_calls
detector_calls
predict_calls
checkpoint_loads

external_writes
task8c_artifacts_unchanged

diagnostic_figures

conclusions
limitations


# 20. Required scientific report

Create:

docs/task8d_above_target_chain_diagnosis_v1.md

Required structure:

1. Question being diagnosed
2. Why ABOVE is the clean case
3. Frozen evidence / no-rerun guarantee
4. Actual target mechanism in RC1
5. Reference geometry audit
6. P_dir / P_near analysis
7. W / A analysis
8. C prototype-similarity analysis
9. Decoder logits / probability analysis
10. Final mask analysis
11. Target-6 vs wrong-7 trajectory
12. Canonical-reference deterministic counterfactual
13. First observed divergence
14. Runtime SUCCESS guard explanation
15. Evidence-supported diagnosis
16. What remains unresolved
17. Possible future research directions — hypotheses only, NO IMPLEMENTATION
18. Questions worth discussing with the advisor


# 21. Diagnosis conclusion vocabulary

Do NOT use:

"fixed"
"solved"
"root cause proven"

unless logically established, which is not expected here.

Prefer:

FIRST_OBSERVED_DIVERGENCE
PRIMARY_EVIDENCE_SUPPORTS
PLAUSIBLE_CONTRIBUTOR
NOT_SOLE_EXPLANATION
MIXED
UNRESOLVED


# 22. No-repair rule

Even if the cause appears obvious:

DO NOT:

change P_dir parameters
change P_near sigma
change W
change A normalization
change prototype
change D-B1
change threshold
add proposal selection
add graph logic
add a target-instance selector
train a model
change the architecture

This task ends at diagnosis.


# 23. Dedicated tests

Create:

tests/test_task8d_above_target_chain_diagnosis.py

Tests must prove at minimum:

- correct Task8C above sample lock;
- frozen artifact identity verification;
- no predict.py;
- no DetectorRuntime;
- no Sam2Runtime;
- no Db1Runtime;
- no torch.load checkpoint;
- no Ultralytics;
- no Transformers;
- no subprocess model execution;
- no external writes;
- Task8C files opened read-only;
- 64x64 fractional area mapping is deterministic;
- candidate mass/mean formulas;
- target6-vs-7 delta calculation;
- ranking deterministic;
- first-divergence ordering fixed;
- no numeric pass threshold;
- canonical-reference counterfactual contains deterministic fields only;
- report/JSON consistency.

All tests must PASS before final handoff.


# 24. Allowed repository changes

Only:

scripts/task8d_above_target_chain_diagnosis.py
tests/test_task8d_above_target_chain_diagnosis.py
docs/task8d_above_target_chain_diagnosis_v1.md
evaluation/task8d_above_target_chain_diagnosis_v1.json
evaluation/task8d_above_target_chain_diagnosis_v1/*.png
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml

No product source changes.
No governance changes.
No Task8C evidence changes.


# 25. External writes

NONE.

All external RC1 files are READ ONLY.

Diagnostic outputs belong only in repository evaluation/.


# 26. Completion state

On successful completion:

CURRENT_TASK Status:
READY_FOR_SUPERVISOR_AUDIT

EXECUTOR_STATE status:
READY_FOR_SUPERVISOR_AUDIT

next_gate:
CHATGPT_TASK8D_ABOVE_CHAIN_DIAGNOSIS_REMOTE_AUDIT

Commit + push all evidence.

Then STOP.

Do not begin a repair task.


# 27. Final executor summary

Report objectively:

- first observed divergence stage;
- target6 vs wrong7 trajectory;
- whether canonical-reference field counterfactual changes the result;
- whether the failure is already visible before C;
- whether C introduces/reinforces the error;
- whether decoder logits introduce/reinforce the error;
- why runtime structural SUCCESS permits the semantic error;
- unresolved questions.

Do not prescribe or implement the next architecture.

## Executor completion checkpoint

Status: READY_FOR_SUPERVISOR_AUDIT
Next gate: CHATGPT_TASK8D_ABOVE_CHAIN_DIAGNOSIS_REMOTE_AUDIT

Diagnosis only; no repair. First observed divergence: C / C_mean (actual top GT7); W/A favour GT6 over GT7, but their full ranking is topped by reference4. Logits/probability/final IoU continue to favour GT7.
GT-reference deterministic W/A still favour6, ranks6/7 remain2/3; no preference restoration. Interpretation: MIXED / NOT RESOLVED; q/C/decoder counterfactual not evaluated.
Runtime SUCCESS verifies mapped nonemptiness and centroid direction, not nearest/native GT target identity. Causal feature/prototype/decoder contributions remain unresolved.
Current dedicated tests: 41 passed in1.76s / exit0; all9 diagnostic PNGs visually reviewed. Source/config helper135/135; manifest Git-byte identical. External2184 files unchanged in bytes/SHA256/mtime/file set; all Task8C repo evidence/artifacts/TIFFs and native GT unchanged. Grandfathered settings606bytes/SHA256 unchanged.
Model/detector/predict/checkpoint calls0; external writes0; automatic frozen maps recomputed0. No product/governance/Task8C changes. Report, structured evidence and9 figures are persisted in the Task8D authorized paths.
STOP after commit/push. Scientific acceptance remains with ChatGPT Supervisor; no repair task or new architecture is authorized.
