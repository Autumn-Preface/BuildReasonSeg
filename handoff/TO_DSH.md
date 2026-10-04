# TO_DSH — Task 8B.3-P1D11M1: Normalize Canonical RC1 Source-Manifest Identity

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `ca0e9f4217f0abfb58bffe36ac297c46206fe016`

# 0. Purpose

P1D11A-R1 correctly STOPPED and established:

```text
manifest entries = 135

match Git-index AND working-tree bytes = 93
match Git-index only                  = 4
match working-tree only               = 38
match neither                         = 0
```

Therefore the current `source_manifest.json` mixes byte conventions.

This task performs ONE infrastructure repair:

```text
normalize all 135 manifest entry identities to Git canonical blob/index bytes
```

It does NOT implement the supported-domain policy yet.

The chosen authoritative identity convention is:

```text
GIT_CANONICAL_BLOB_BYTES
```

Meaning:

- manifest `bytes` and `sha256` describe the Git canonical content of each tracked canonical RC1 source/config file;
- Windows checkout CRLF expansion is NOT part of the manifest identity;
- external-delivery byte equality remains a separate sync/check concern.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = ca0e9f4217f0abfb58bffe36ac297c46206fe016
```

Allowed initial tracked tree:

```text
clean
```

or only:

```text
M handoff/TO_DSH.md
```

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:

- run any detector/model/inference;
- run `predict.py`, Qwen, SAM2, D-B1;
- run pytest/check_setup;
- train/fine-tune/export/download;
- modify runtime source;
- modify tests;
- modify README/model_card or supported-domain wording;
- modify `.gitattributes`;
- change `core.autocrlf` or any Git configuration;
- run repository-wide line-ending normalization;
- run `git add --renormalize`;
- modify model assets/checkpoints/configs;
- modify/copy/run locked Demo candidates;
- modify external delivery;
- run external write sync;
- modify `main`;
- force push.

# 3. Allowed tracked changes

Only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
docs/task8b3_p1d11_manifest_normalization.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other tracked file may change.

# 4. Reproduce the mixed-convention finding

Before modifying the manifest, reproduce the current state across all 135 entries.

For each manifest entry:

```text
relative path
manifest bytes
manifest sha256
Git canonical bytes
Git canonical sha256
working-tree bytes
working-tree sha256
matches Git canonical? YES/NO
matches working tree? YES/NO
```

Git canonical bytes MUST be read as binary from:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<relative path>
```

Do not text-decode before hashing.

Require aggregate exactly:

```text
both = 93
Git-only = 4
working-tree-only = 38
neither = 0
total = 135
```

If these exact counts are not reproduced:

```text
STOP
```

Do not normalize.

# 5. Verify consumer semantics before normalization

Read only:

```text
scripts/sync_advisor_rc1_delivery.py
```

Record whether the sync helper uses manifest fields as follows:

```text
path field used to enumerate sync files = YES / NO
manifest bytes used to validate source/destination = YES / NO
manifest sha256 used to validate source/destination = YES / NO
source-vs-destination actual bytes/hash compared directly = YES / NO
```

Expected from current code:

```text
path = YES
manifest bytes = NO
manifest sha256 = NO
actual source-vs-destination comparison = YES
```

If current code materially disagrees:

```text
STOP
```

because normalization semantics need a new ChatGPT decision.

Also confirm:

```text
source_manifest.json is NOT itself one of the 135 manifest file entries
```

If it is self-listed:

```text
STOP
```

# 6. Freeze identity semantics in manifest metadata

Keep:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
task
source_delivery
canonical_root
copy_policy
```

unchanged.

Add exactly these top-level metadata fields:

```json
"identity_basis": "GIT_CANONICAL_BLOB_BYTES",
"identity_basis_note": "bytes and sha256 are computed from Git canonical blob/index bytes; working-tree line-ending expansion is not part of source identity"
```

Do NOT change the schema string.

Do NOT add per-entry metadata fields.

# 7. Normalize all 135 entry identities

For every entry, preserve exactly:

```text
path
entry order
number of entries
```

Replace its:

```text
bytes
sha256
```

with values computed from the exact binary Git canonical bytes:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<path>
```

Expected effects from the preflight classification:

```text
93 both-match entries:
  identity values remain unchanged

4 Git-only entries:
  identity values remain unchanged

38 working-tree-only entries:
  identity values change to Git canonical identity

path additions/removals:
  0
```

Require:

```text
entry count = 135
changed entry identities = 38
unchanged entry identities = 97
```

If a different number of entry identities changes:

```text
STOP before commit
```

Do not adjust criteria.

# 8. Authoritative post-normalization check

After editing `source_manifest.json`, parse the new manifest.

For all 135 entries, independently read binary canonical bytes from:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<path>
```

and compare to normalized:

```text
bytes
sha256
```

Require:

```text
Git-canonical normalized manifest = 135/135 PASS
missing paths = 0
duplicate paths = 0
entry count = 135
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

Because source files are not modified in this task, `HEAD:<path>` remains authoritative.

# 9. No semantic/source changes gate

Require:

```text
git diff --name-only <starting HEAD>
```

contains only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
docs/task8b3_p1d11_manifest_normalization.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Explicitly confirm:

```text
runtime source modified = NO
tests modified = NO
README modified = NO
model_card modified = NO
external delivery modified = NO
locked candidates modified/run = NO
```

# 10. External-delivery interpretation

Do NOT sync external delivery.

Record:

```text
external source/config files = UNCHANGED
external source_manifest.json = UNCHANGED
canonical source_manifest.json = NORMALIZED
```

State clearly:

```text
The normalized source manifest now describes Git canonical source identity.
It is not a claim that CRLF-expanded external files have identical bytes to each manifest entry.
External source-vs-destination equivalence remains governed by the existing sync helper's direct comparison.
```

Do not run write sync.

A read-only sync check is NOT required for this task.

# 11. Report

Create:

```text
docs/task8b3_p1d11_manifest_normalization.md
```

Required sections:

1. task / starting HEAD
2. reason for normalization
3. reproduced 93 / 4 / 38 / 0 classification
4. sync-helper consumer semantics
5. selected identity convention
6. manifest metadata change
7. exactly 38 normalized entry identities
8. 135/135 Git-canonical post-check
9. no source/runtime/test change
10. external delivery untouched
11. implications for P1D11 supported-domain policy
12. exact next gate.

Do NOT dump all 135 rows into the report.

List the 38 changed paths OR store their path list in the report.
For each changed path, full SHA is optional; aggregate identity check is authoritative.

# 12. Outcome

Choose exactly ONE.

## A — `RC1_SOURCE_MANIFEST_GIT_IDENTITY_NORMALIZED`

Require all gates above pass.

Then:

```text
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION_RETRY
```

## B — `RC1_SOURCE_MANIFEST_NORMALIZATION_BLOCKED`

Use for any failed gate.

Then:

```text
NEXT = RC1_SOURCE_MANIFEST_EVIDENCE_RECOVERY
```

Do NOT execute next gate.

# 13. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11M1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: ca0e9f4217f0abfb58bffe36ac297c46206fe016
Model/test execution: NONE
Functional runtime files modified: NO
Tests modified: NO
External delivery modified: NO
Pre-normalization both/Git-only/disk-only/neither: 93 / 4 / 38 / 0
Manifest entry count: 135
Sync helper uses manifest bytes/hash for source-destination validation: NO
source_manifest self-listed: NO
Identity basis: GIT_CANONICAL_BLOB_BYTES
Changed entry identities: 38
Unchanged entry identities: 97
Post-normalization Git-canonical manifest: 135/135 PASS / FAIL
Missing paths: 0 / other
Duplicate paths: 0 / other
Outcome: <exact enum>
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Supported-domain policy: NOT YET IMPLEMENTED
Locked candidate status: FINAL_METADATA_LOCK
Locked candidate runtime status: NOT YET RUN
Scientific freeze preserved: YES
Next gate: <exact enum>
Report: docs/task8b3_p1d11_manifest_normalization.md
Next action: Awaiting ChatGPT audit; do not implement policy or sync external.
```

# 14. Commit / push

If COMPLETE:

```text
chore(rc1): normalize canonical source manifest identity
```

If STOP/FAILED:

```text
docs(rc1): record source-manifest normalization stop
```

Push current branch normally.
No force push.

# 15. COMPLETE definition

COMPLETE only if:

- exact starting HEAD;
- no model/test execution;
- mixed convention reproduced exactly as 93/4/38/0;
- sync-helper semantics confirmed;
- source_manifest is not self-listed;
- schema remains v1;
- identity basis metadata added;
- exactly 38 entry identities normalized to Git canonical bytes;
- all 135 paths/order preserved;
- post-normalization Git-canonical check = 135/135;
- no runtime/test/README/model_card/external change;
- PROP-01 remains OPEN;
- policy remains not yet implemented;
- next gate recorded but not executed;
- commit/push succeeds;
- STOP.
