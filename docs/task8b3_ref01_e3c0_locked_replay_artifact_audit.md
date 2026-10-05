# Task 8B.3-REF01-E3C0-R1 — Locked Replay Saved-Artifact Audit (Correction)

## 1. Scope and correction

```text
base head = bde13bd15bbab3e455ea1d3ef10b3f6740fc110d
branch    = audit/task8b3-ref01-locked-replay-artifacts
scope     = READ_ONLY_LOCKED_REPLAY_SAVED_ARTIFACT_AUDIT_CORRECTION
detector_model_calls = 0 · proposal regeneration = NO · actual replay = NO · external write = NO · product change = NO
```

The previous audit is discarded in full: it searched forbidden roots and used unapproved enums. Discarded roots:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\artifacts
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\inference
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs
```

## 2. Allowed search roots used by this audit

```text
repo     : evaluation, docs, handoff, scripts
external : C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output · C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\docs · C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 root-level text files only
```

## 3. Candidate artifacts found in the allowed roots

| # | root | file | bytes | sha256 (prefix) |
|---:|---|---|---:|---|
| 1 | output | `parsed_program.json` | 2 | `44136fa355b3678a` |
| 2 | output | `prompt.txt` | 0 | `e3b0c44298fc1c14` |
| 3 | output | `proposals.json` | 21263 | `62865e42b4d92077` |
| 4 | output | `result.json` | 23296 | `d2cef24ea3b15a52` |
| 5 | output | `parsed_program.json` | 2 | `44136fa355b3678a` |
| 6 | output | `prompt.txt` | 0 | `e3b0c44298fc1c14` |
| 7 | output | `proposals.json` | 3631 | `257297112fb2fe35` |
| 8 | output | `result.json` | 4782 | `bf70127106e5798b` |
| 9 | output | `parsed_program.json` | 2 | `44136fa355b3678a` |
| 10 | output | `prompt.txt` | 0 | `e3b0c44298fc1c14` |
| 11 | output | `proposals.json` | 2412 | `7c1a45690c860910` |
| 12 | output | `result.json` | 3503 | `ab75dd17e082c171` |
| 13 | output | `parsed_program.json` | 2 | `44136fa355b3678a` |
| 14 | output | `prompt.txt` | 0 | `e3b0c44298fc1c14` |
| 15 | output | `proposals.json` | 2411 | `2b09041ee86d5e7d` |
| 16 | output | `result.json` | 3502 | `54c964933fca667f` |

## 4. Per-case readiness facts

| relation | tile | complete proposal list | scalar fields complete | exact mask/object material | forensic linkage | replay records |
|---|---|---|---|---|---|---|
| right | 1010 | False | True | False | False | 6 |
| left | 1003 | False | True | False | False | 53 |
| above | 1008 | False | True | False | False | 9 |
| below | 1009 | False | True | False | False | 6 |

## 5. Determinations (frozen enums only)

```text
overall_readiness = LOCKED_REPLAY_ARTIFACTS_PARTIAL
production_object_replay_possible_without_detector = False
record_level_replay_possible_without_detector = False
historical_consistency_mismatches = [{'relation': 'right', 'tile': '1010', 'detail': {'counts_match_ref01_evidence': False, 'stored_counts_match_live': True}}, {'relation': 'left', 'tile': '1003', 'detail': {'counts_match_ref01_evidence': False, 'stored_counts_match_live': True}}, {'relation': 'above', 'tile': '1008', 'detail': {'counts_match_ref01_evidence': False, 'stored_counts_match_live': True}}, {'relation': 'below', 'tile': '1009', 'detail': {'counts_match_ref01_evidence': False, 'stored_counts_match_live': True}}]
overall_outcome = REF01_LOCKED_REPLAY_ARTIFACT_AUDIT_CORRECTED
next_gate = REF01_LOCKED_REPLAY_INPUT_RECOVERY_DESIGN
```

## 6. Explicit non-execution

```text
detector / model inference = NONE · proposal regeneration = NONE · actual replay = NONE
external write = NONE · product / test / manifest / helper modification = NONE
manual visual inspection / candidate replacement = NO / NO · NEXT executed = NO
```
