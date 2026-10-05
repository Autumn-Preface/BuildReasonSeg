请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C2 — Selection Scalar Forensics**

# TO_DSH — Task 8B.3-REF01-E3C2: Selection Scalar Forensics

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `b5be1d4e7b02d93f91c3f08c8e90014e590ea21e`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Required script: `C:\D\DeepSeekHarness\E3C2_selection_scalar_forensics.py`
> Required SHA256: `a4e22aabc2439343ee213c1a4941daa9d9deecd385894e362dbb33524c96e869`

# 0. CHATGPT AUDIT DISPOSITION

Task `8B.3-REF01-E3C1` is APPROVED.

Locked record-level replay established:

```text
right:  1 -> 1   correct / stable
left:  14 -> 14  wrong-covered remains
above:  4 -> 5   eligibility blocker resolved
below:  1 -> 1   wrong-covered remains
```

Therefore:

```text
eligibility repair validation = PASS
REF-01 = ACTIVE
dominant remaining blocker = SELECTION / RANKING
remaining cases = left, below
```

Do NOT modify rank yet.

E3C2 is a read-only scalar-forensics task used by ChatGPT to decide whether a simple, threshold-free scalar ranking repair is technically supportable.

# 1. PREDECLARED DIAGNOSTIC RULE FAMILY

The exact script evaluates ONLY these six threshold-free ranking families:

```text
AREA_FIRST
CONFIDENCE_FIRST
BBOX_AREA_FIRST
FILL_RATIO_FIRST
AREA_X_CONF
BBOX_AREA_X_CONF
```

They are diagnostic hypotheses only.

DSH MUST NOT:
- add a rule;
- remove a rule;
- tune a threshold;
- choose a winner;
- recommend implementation.

The task records their behavior on the four locked cases and returns control to ChatGPT.

# 2. REQUIRED FORENSICS

For each locked case compute from the repaired candidate set:

```text
current repaired selected ID
best-IoU ID
best IoU
extent-exception IDs

selected features:
mask_area
confidence
bbox_area
fill_ratio
bbox_extent_ratio
centroid

best-IoU features:
same fields

selected-minus-best deltas

best-IoU Pareto dominators in:
(mask_area, confidence)
(bbox_area, confidence)

for each predeclared rule:
selected ID
rank position of best-IoU proposal
whether selected == best-IoU
```

Also compute:

```text
globally_perfect_rules_on_locked_four
```

This is diagnostic only and MUST NOT automatically become a repair rule.

# 3. EXECUTOR CONTRACT

Allowed ONLY:

1. verify exact branch/head;
2. verify script SHA;
3. run supplied script exactly ONCE;
4. inspect stdout;
5. inspect generated evidence/report/FROM_DSH;
6. verify only four allowed repo paths changed;
7. stage exactly four paths;
8. commit exactly once;
9. push once;
10. STOP.

Forbidden:

- NO detector/model inference;
- NO proposal regeneration;
- NO selector repair implementation;
- NO product/test/manifest/helper edit;
- NO external write;
- NO broad search;
- NO new rank rule;
- NO threshold tuning;
- NO architecture change;
- NO final Demo;
- NO script edit/rebuild/rerun;
- NO pytest/py_compile;
- NO package/environment change;
- NO reset/amend/rebase/stash/clean;
- NO force push;
- NO NEXT execution.

# 4. EXACT SCIENTIFIC INPUTS

Read only:

```text
four locked proposals.json:
1010
1003
1008
1009

evaluation/task8b3_ref01_locked_reference_forensics.json
```

Exact proposal and forensics SHA256 values are embedded in the supplied script.

No result.json is needed.
No product source execution is needed.
No broad discovery is allowed.

# 5. GIT PRE-FLIGHT

Require:

```text
branch = audit/task8b3-ref01-locked-replay-artifacts
HEAD = b5be1d4e7b02d93f91c3f08c8e90014e590ea21e
```

Allowed `git status --short`:
- empty;
- ` M handoff/TO_DSH.md`;
- `M  handoff/TO_DSH.md`.

No other path/status.

# 6. SCRIPT GATE

Place exactly:

```text
C:\D\DeepSeekHarness\E3C2_selection_scalar_forensics.py
```

SHA256 must equal:

```text
a4e22aabc2439343ee213c1a4941daa9d9deecd385894e362dbb33524c96e869
```

Do not modify/reconstruct.

# 7. RUN ONCE

Run exactly once:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\E3C2_selection_scalar_forensics.py
```

Require exit 0.

Require stdout contains:

```text
E3C2_SELECTION_SCALAR_FORENSICS: COMPLETE
NEXT_GATE=CHATGPT_SELECTION_REPAIR_DECISION
```

`GLOBALLY_PERFECT_RULES=` may be `NONE` or a comma-separated subset of the six predeclared rules. Record it exactly; do not interpret it.

# 8. GENERATED OUTPUTS

Required new files:

```text
evaluation/task8b3_ref01_e3c2_selection_scalar_forensics.json
docs/task8b3_ref01_e3c2_selection_scalar_forensics.md
```

Required modified:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Evidence must contain:

```text
task = 8B.3-REF01-E3C2
scope = LOCKED_SELECTION_SCALAR_FORENSICS
detector_model_calls = 0
proposal_regeneration_performed = false
actual_selector_repair_implementation_performed = false
external_write_performed = false
product_source_changed = false
scalar_rule_design_selected = false
ref01_status = ACTIVE
prop01_status = PROP01_OPEN_ENGINEERING_DEFECT
overall_outcome = REF01_SELECTION_SCALAR_FORENSICS_COMPLETE
next_gate = CHATGPT_SELECTION_REPAIR_DECISION
```

# 9. DIFF / STAGING GATE

After successful script run:

```text
git status --porcelain=v1 --untracked-files=all
```

Only these paths may appear:

```text
docs/task8b3_ref01_e3c2_selection_scalar_forensics.md
evaluation/task8b3_ref01_e3c2_selection_scalar_forensics.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Stage exactly those four.

Require `git diff --cached --name-only` contains exactly those four paths.

# 10. COMPLETE COMMIT

If every gate passes:

```text
git commit -m "docs(rc1): characterize remaining reference selection blocker"
```

Exactly one commit.
Push current branch once.
No force push.
STOP.

# 11. STOP PROTOCOL

If any gate fails:
- do not rerun;
- do not self-repair;
- preserve factual outputs;
- record exact failure in FROM_DSH;
- commit once with:
  `docs(rc1): record reference selection forensics stop`
- push once;
- STOP.

# 12. COMPLETE DEFINITION

COMPLETE requires:
- exact head / script SHA;
- one script run;
- six and only six predeclared rank rules;
- no threshold tuning;
- no repair implementation;
- four locked cases characterized;
- exact four output paths;
- zero inference/regeneration/external write/product changes;
- one commit;
- push once;
- STOP.
