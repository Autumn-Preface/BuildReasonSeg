# TO_DSH — Task 8B.3-P1D11A-R3: Correct Policy Lock Identity and Input-Domain Wording

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `e0e0a74760d619db3cdd6be62c4ea42520f802ec`

# 0. ChatGPT audit disposition

P1D11A-R2 is NOT YET APPROVED.

Accepted:
- correct starting parent;
- only authorized files changed;
- pre-edit Git-canonical manifest = 135/135;
- post-edit staged/index manifest = 135/135;
- only README.md and docs/model_card.md manifest identities changed;
- A2 remained `NOT ESTABLISHED`;
- external delivery was not modified;
- no model/test/runtime execution.

Two policy defects require correction.

## Defect A — immutable left candidate ID typo

Authoritative frozen ID from P1D10-R3:

```text
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
```

P1D11A-R2 policy incorrectly wrote:

```text
buildsr_test_1003_3_largest_to_left_of_nearest_f3fcb14e14c3
```

The missing `_to_` must be corrected. No candidate substitution is authorized.

## Defect B — ambiguous input-domain wording remains

README still contains:

```text
RGB 光学遥感影像（正式输入域）
```

and model card contains:

```text
输入域：RGB 光学遥感影像
```

These phrases can still be read as a generalization-domain claim.

They must instead distinguish:

```text
software-readable input modality
vs
verified research/evaluation domain
```

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = e0e0a74760d619db3cdd6be62c4ea42520f802ec
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run any model/detector/inference;
- run pytest/check_setup;
- modify runtime/tests/config/checkpoints/model assets;
- change any frozen metric;
- copy/run/inspect locked candidates;
- replace any locked candidate;
- modify A1/A2/A3/A4/B1/B2;
- modify external delivery;
- run external write sync;
- modify `.gitattributes` or Git config;
- mark PROP-01 closed;
- call A2 proven out-of-domain;
- update main;
- force push.

# 3. Allowed tracked changes

Only:

```text
docs/task8b3_p1d11_supported_domain_policy.md
docs/task8b3_p1d11a_policy_implementation.md
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 4. Pre-edit manifest gate

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
entries = 135
```

Verify all 135 entries against:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<path>
```

Require:

```text
pre-edit Git-canonical manifest = 135/135 PASS
```

Otherwise STOP before edits.

# 5. Correct immutable candidate identity

In:

```text
docs/task8b3_p1d11_supported_domain_policy.md
```

replace only the incorrect left sample ID with exactly:

```text
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
```

Reconfirm all four immutable IDs exactly:

```text
right =
buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91

left =
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3

above =
buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314

below =
buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
```

No other candidate identity is permitted.

# 6. Correct README wording

Modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
```

In `## 输入`, replace:

```text
- **RGB 光学遥感影像**（正式输入域）；支持 ...
```

with wording whose meaning is exactly:

```text
- **软件接受的输入模态：RGB 光学影像**；支持 ...
```

or an equally explicit wording that does NOT use `正式输入域`.

The nearby `### 已验证数据域与 Demo 边界` section remains authoritative for demonstrated domain.

Require the README to contain NO phrase equivalent to:

```text
RGB 光学遥感影像（正式输入域）
```

Do not change runtime behavior documentation.

# 7. Correct model-card wording

Modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

In `## 2. 数据`, replace:

```text
- 输入域：**RGB 光学遥感影像** ...
```

with wording whose meaning is exactly:

```text
- 软件输入模态：**RGB 光学遥感影像** ...
```

or an equally explicit wording separating accepted modality from verified domain.

Keep the following separate positive domain statement:

```text
已验证域：WHU East Asia ...
```

No frozen metric may change.

# 8. Reconfirm A2 wording

Across:
- policy doc;
- README;
- model card;

require:

```text
A2 provenance/domain = NOT ESTABLISHED
A2 = documented persistent non-detection/stress case
A2 != proven out-of-domain
A2 != fixed
```

No contradictory wording allowed.

# 9. Reconfirm test-reuse disclosure

The policy doc must still contain the exact English sentence:

```text
The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.
```

Faithful Chinese equivalent must remain.

# 10. Update implementation report

Update:

```text
docs/task8b3_p1d11a_policy_implementation.md
```

Append/replace with an R3 correction section recording:

```text
left candidate ID typo = CORRECTED
README ambiguous "正式输入域" = REMOVED
model-card generic "输入域" = replaced by software-input-modality wording
A2 wording = unchanged and evidence-bounded
locked candidates = unchanged
```

Correct the next gate to:

```text
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC
```

Rationale:

```text
formal RC1 product/document changes follow canonical -> audit -> controlled external sync
before the next external-delivery diagnostic phase.
```

Also record the subsequent gate, not yet authorized:

```text
AFTER_SYNC = PROP01_LOCKED_DEMO_PROPOSAL_GATE
```

Do NOT execute either.

# 11. Manifest update

Stage only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Compute Git-index canonical bytes/hashes using binary:

```text
git show :delivery_src/BuildReasonSeg_Advisor_RC1/README.md
git show :delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Update exactly the manifest entries:

```text
README.md
docs/model_card.md
```

Do NOT change any other identity or top-level metadata.

Stage source_manifest.json.

# 12. Post-edit manifest gate

Using staged/index bytes for staged files and HEAD blobs for unchanged files, require:

```text
post-edit Git-canonical/index manifest = 135/135 PASS
entry count = 135
missing = 0
duplicates = 0
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

Require exactly two manifest entry identities changed from starting HEAD:

```text
README.md
docs/model_card.md
```

# 13. External boundary

External delivery:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

must remain untouched.

Do NOT write sync.

# 14. Final status

If COMPLETE:

```text
supported-domain policy =
IMPLEMENTED_IN_CANONICAL_RC1_DOCS_CORRECTED

locked candidate identity =
4/4 EXACT MATCH

A2 domain =
NOT ESTABLISHED

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

scientific freeze =
PRESERVED

NEXT =
PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC

AFTER_SYNC =
PROP01_LOCKED_DEMO_PROPOSAL_GATE
```

# 15. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11A-R3
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: e0e0a74760d619db3cdd6be62c4ea42520f802ec
Model/test execution: NONE
Functional runtime files modified: NO
Tests modified: NO
External delivery modified: NO
Pre-edit Git-canonical manifest: 135/135 PASS / FAIL
Post-edit staged/index manifest: 135/135 PASS / FAIL / NOT RUN
Manifest entry count: 135 / other
Manifest entry identities changed: 2 / other
Changed manifest entries: README.md; docs/model_card.md / other
Locked candidate exact identities: 4/4 MATCH / other
Locked left candidate: buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
README ambiguous formal-input-domain phrase: REMOVED / PRESENT
Model-card input wording: SOFTWARE_INPUT_MODALITY / other
A2 domain classification: NOT ESTABLISHED
A2 policy status: DOCUMENTED_PERSISTENT_NON_DETECTION
Supported-domain policy: IMPLEMENTED_IN_CANONICAL_RC1_DOCS_CORRECTED / NOT IMPLEMENTED
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Scientific freeze preserved: YES / NO
External write sync: NOT RUN
Next gate: PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC
After sync: PROP01_LOCKED_DEMO_PROPOSAL_GATE
Next action: Awaiting ChatGPT audit; do not sync external or run locked candidates.
```

# 16. Commit / push

If COMPLETE:

```text
docs(rc1): correct prop01 supported-domain policy
```

If STOP/FAILED:

```text
docs(rc1): record prop01 policy correction stop
```

Push normally.
No force push.

# 17. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model/test/runtime/external changes;
- pre-edit manifest = 135/135;
- left candidate ID exactly corrected;
- all 4 candidate IDs exact;
- README no longer calls RGB modality "正式输入域";
- model card distinguishes software input modality from verified domain;
- A2 remains NOT ESTABLISHED;
- required reuse disclosure remains;
- exactly README/model_card manifest entries update;
- post-edit manifest = 135/135;
- next gate is controlled external policy sync;
- PROP-01 remains OPEN;
- commit/push succeeds;
- STOP.
