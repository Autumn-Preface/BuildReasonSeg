# Task 8B.3-M1B.3 — Large-Image Proposal Memory Gate (RC1-DEMO-MEM-01)

## 1. Task / scope

Real external `predict.py` validation of the frozen B1/B2 5000×5000 inputs after the controlled M1B.1 sync, to decide
whether `RC1-DEMO-MEM-01` (full-frame proposal bool allocation) is closed. Each image was run **at most once**; no
visual/semantic quality assessment is made here.

## 2. Starting HEAD

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD   = f73b4294667219f093aa6333c3fd5d4a36832814
```

## 3. External preflight

```text
external manifest          = 135/135 PASS (135 entries)
mismatches                 = none
B1.tif                     = exists=True size=[5000, 5000] bytes=75040370
B2.tif                     = exists=True size=[5000, 5000] bytes=75080895
```

Both inputs are 5000×5000 as required by the gate.

## 4. Frozen prompts (byte-identical to the canonical harness `CASES`)

```text
B1: 请找出面积最大的建筑，并分割它右边离它最近的那栋楼。
B2: 最大建筑物的上面，离它最近的那一栋是什么，分割出来
```

## 5. Per-case real-predict results

| case | invocations | exit | status | tile_count | raw proposals | merged proposals | duration | memory signature | E502 |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 1 | 0 | SUCCESS | 169 | 6578 | 3066 | 44.14 s | False | False |
| B2 | 1 | 0 | SUCCESS | 169 | 7864 | 3740 | 34.33 s | False | False |

## 6. Detector / proposal stage

For both images the detector/proposal stage **completed** (`tile_count = 169` planned tiles each, thousands of raw
detections merged into thousands of proposals) and reached a final `SUCCESS` product status — the stage that
previously aborted with the allocation error now runs to completion.

## 7. Memory-failure search

```text
signature "Unable to allocate 23.8 MiB for an array with shape (5000, 5000) and data type bool"
    B1 transcript = NOT FOUND
    B2 transcript = NOT FOUND
MemoryError / "Unable to allocate"
    B1 = NOT FOUND
    B2 = NOT FOUND
E502
    B1 = NOT FOUND
    B2 = NOT FOUND
```

The previous R4B failures were `E502` with the allocation signature above; neither appears in these runs.

## 8. Observed interaction detail

B1 answered one product confirmation prompt; B2 recorded zero answered prompts while still returning `SUCCESS`
(a driver-side chunk-boundary counting artifact — the per-case transcripts are preserved under
`logs/task8b3_m1b3_transcripts/`). This does not affect the memory gate, which is decided by the proposal stage and
the absence of the allocation failure.

## 9. Verdict

```text
MEM01_REAL_GATE_PASS
RC1-DEMO-MEM-01 = CLOSED
```

## 10. Unchanged scope

No runtime, test, manifest, model asset, threshold, tiling, merge policy, Reference policy or SUCCESS-validity code
was modified by this task. `PROP-01`, `REF-01` and `MASK-01` remain `UNCHANGED / UNCHANGED / UNCHANGED`, and no
visual/semantic judgement of the B1/B2 outputs is expressed here.

## 11. Next action

Awaiting ChatGPT audit before any further task; Task 8B.4 / Task 8C not entered.
