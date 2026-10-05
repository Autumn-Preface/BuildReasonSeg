# Task 8B.3-REF01-E3C0 — Locked Replay Saved-Artifact Audit

## 1. Task, branch and scope

```text
base branch / head = fix/task8b3-ref01-eligibility-repair-sync / 1197d860b8b1a3503c6cba730eda33ccbff4e1c5
audit branch       = audit/task8b3-ref01-locked-replay-artifacts
scope              = READ_ONLY_SAVED_ARTIFACT_AUDIT
detector/YOLO/SAM/Qwen/MLLM inference = NONE
proposal regeneration = NONE · real replay = NONE · external write = NONE
product / test / manifest / helper modification = NONE
```

Only read-only searches, hashing and schema inventory were performed inside the repository and external delivery roots.

## 2. Limited-root inventory

| root | exists | files | locked tiles seen |
|---|---|---:|---|
| `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\inference` | False | 0 | [] |
| `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\artifacts` | True | 85238 | ['1003', '1008', '1009', '1010'] |
| `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation` | True | 572 | [] |
| `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output` | True | 1543 | ['1003', '1008', '1009', '1010'] |
| `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs` | True | 265 | ['1003'] |

## 3. Per-case saved artifacts

| relation | tile | result.json | proposals.json | GT mask | GT pixels | reference IoU | verdict |
|---|---|---|---|---|---|---|---|
| right | 1010 | True | True | True | 2478 | True | SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU |
| left | 1003 | True | True | True | 3512 | True | SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU |
| above | 1008 | True | True | True | 5013 | True | SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU |
| below | 1009 | True | True | True | 2606 | True | SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU |

Artifact presence detail:

| relation | artifacts |
|---|---|
| right | result.json:yes | proposals.json:yes | prompt.txt:yes | parsed_program.json:yes | global_proposals.png:yes |
| left | result.json:yes | proposals.json:yes | prompt.txt:yes | parsed_program.json:yes | global_proposals.png:yes |
| above | result.json:yes | proposals.json:yes | prompt.txt:yes | parsed_program.json:yes | global_proposals.png:yes |
| below | result.json:yes | proposals.json:yes | prompt.txt:yes | parsed_program.json:yes | global_proposals.png:yes |

## 4. Mechanical replay-readiness determination

```text
per-case verdicts = {"right": "SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU", "left": "SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU", "above": "SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU", "below": "SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU"}
reference-IoU replay ready for all four cases = True
end-to-end replay ready = False
replay gaps (not saved anywhere in the audited roots) = ['language parse output', 'relation-field output', 'SAM2 reference mask', 'D-B1 decoder output', 'final composite mask']
overall outcome = REFERENCE_IOU_REPLAY_READY_END_TO_END_NOT_READY
NEXT = REF01_LOCKED_REPLAY_ARTIFACT_COMPLETION (not executed)
```

The four locked cases therefore have the saved detector result/proposal metadata and the canonical GT reference masks
needed to re-derive the reference-IoU classification **without any new inference**, while a full end-to-end replay is
not yet possible from saved artifacts alone because the language parse, relation-field, SAM2, D-B1 and final-mask
outputs were never persisted for these candidates.

## 5. Explicit non-execution

```text
inference of any kind / proposal regeneration / real replay = NONE / NONE / NONE
external RC1 write = NONE · product / test / manifest / helper modification = NONE
manual visual inspection / candidate replacement = NO / NO · NEXT executed = NO
```
