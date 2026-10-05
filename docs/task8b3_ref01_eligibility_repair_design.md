

---

## 1. E2 — prescribed repair design `LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1`

```text
base branch / head = fix/task8b3-ref01-eligibility-forensics / 8d86e7b77f9423834a4a15117009c5a1e3f79e5d
task branch        = fix/task8b3-ref01-eligibility-repair-design
prescribed design  = LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1 (pre-decided by the task book; no alternative chosen, modified or compared)
code source        = handoff/TO_DSM.md lines 470-738 (fenced python block, byte-for-byte)
verbatim code sha256 = a39e6f724fe78a10652e29898228fc7795392fca8de38c7eaa02809778a88592
script             = scripts\task8b3_ref01_eligibility_repair_design.py
py_compile exit    = 0
design run exit    = 0
status             = COMPLETE
detector / model calls = 0
product code modified  = NO
```

Design output:

```text
DESIGN_ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
LOCKED_CANDIDATES: 4/4 PASS
RIGHT_FINAL: 1
LEFT_FINAL: 14
ABOVE_FINAL: 5
BELOW_FINAL: 1
ABOVE_BORDER_GIANT_EXCLUDED: PASS
SYNTHETIC_CONTRACT: 8/8 PASS
NEW_NUMERIC_THRESHOLDS: NONE
SMALLEST_FAMILY: UNCHANGED
OUTCOME: REF01_ELIGIBILITY_REPAIR_DESIGN_VALIDATED
NEXT: REF01_ELIGIBILITY_REPAIR_IMPLEMENTATION
DETECTOR_MODEL_CALLS: 0
```

No execution error was raised.

### 1.1 Explicit non-execution

```text
alternative repair designs compared / selected / modified = NONE
product code modification = NONE
threshold change implemented in product code = NONE
manual visual inspection / candidate replacement = NO / NO
NEXT executed = NO
```
