# Task 8B.3-M1B.2 — External Full Regression Gate

## 1. Task / scope

Validate the synchronized runnable external RC1 delivery by running the complete delivery-oriented pytest suite
exactly once. No code, test, manifest or model edit was authorized; no inference was run.

## 2. Starting HEAD

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD   = 25d879845ef1508228067d4faa45e98266ed6aa5
tracked tree at start = only M handoff/TO_DSH.md
```

## 3. External preflight manifest result

```text
root                        = C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
schema                      = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
entry count                 = 135
path/size/sha256            = 135/135 PASS
mismatches                  = none
```

## 4. External `source_manifest.json` byte-identity result (before pytest)

```text
byte-identical to canonical = True
```

## 5. Full pytest command

```bat
cd /d C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 && C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe -m pytest tests -q
```

## 6. Full pytest invocation count

```text
1
```

## 7. Exact pytest result and exit code

```text
exit code = 0
summary   = 116 passed in 65.12s (0:01:05)
```

## 8. Failed / error node list

```text
none
```

## 9. Post-pytest external manifest result

```text
external manifest-listed files = 135/135 PASS (135 entries)
mismatches                     = none
source_manifest still byte-identical = True
```

Pytest cache files created by the run are not part of the 135-entry manifest and were not deleted.

## 10. Canonical product / tests / manifest modified

```text
NO
```

## 11. External product / tests / manifest manually modified

```text
NO
```

## 12. Real inference

`NOT RUN` — no `predict.py`, no A1–B2, no YOLO/Qwen/SAM2/D-B1 execution.

## 13. PROP-01 / REF-01 / MASK-01

`UNCHANGED / UNCHANGED / UNCHANGED`.

## 14. Gate result

```text
M1B_EXTERNAL_FULL_REGRESSION_PASS
```

## 15. Next action

Awaiting ChatGPT audit; **no B1/B2 run until that audit**.
