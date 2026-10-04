# Task 8B.3-P1D1 — A2 Zero-Proposal Forensics (read-only)

## 1. Task and scope

Read-only forensic audit of the historical `RC1-DEMO-PROP-01` symptom — A2 produced `raw_proposal_count = 0` and
`merged_proposal_count = 0` while every other frozen sample produced non-zero detector output. **No predict, pytest,
`check_setup.py` or model execution was performed**, and no `detector.py`, test, manifest or external-delivery file
was modified.

```text
base branch/commit : main @ 57b368d5647e842d8f31d6d1a9997bf1df3cc0fb
new branch         : fix/task8b3-prop01-a2-zero-proposals
```

## 2. Recovered historical A2 evidence (R4B artifacts)

```text
diagnostics : inference/output/diagnostics/A2/result.json
              status = FAILED · error_code = E401 (整幅影像未检测到任何建筑实例。)
              tile_count = 9 · raw_proposal_count = 0 · merged_proposal_count = 0
              language = FALLBACK_CORRECT, user_confirmation = Y,
              suggested/parsed program = largest_to_left_of_to_nearest (qwen_program_head)
transcript  : logs/task8b3_transcripts/A2.txt (2 059 bytes)
              L55: WARNING NMS time limit 2.050s exceeded
              L59: [E401 NO_BUILDING_DETECTED] …
              occurrences of "WARNING NMS time limit" in the transcript = 1
input       : inference/input/A2.png size=(1024, 1024) mode=RGB bytes=1 677 040
              sha256 = 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
              (identical to the frozen Task 8B.3 A2 hash)
```

## 3. Frozen detector configuration in force

```text
TILE_SIZE = 512          TILE_OVERLAP = 128        TILE_STRIDE = 384
IMGSZ = 640              CONF = 0.05               MAX_DET = 300
DUPLICATE_IOU = 0.50     MERGE_BBOX_EXTENT_RATIO_MAX = 0.20
tile plan for A2 (1024 x 1024) = 3 x 3 = 9 tiles
```

The `result.json` `tile_count = 9` matches the frozen tile plan, so tiling itself executed as designed.

## 4. Static code-path audit — every way `raw_count` can become zero

`detect_global()` accumulates `raw_count += len(self.detect_tile(tile_rgb))`, and `detect_tile()` returns early only
in three situations:

```python
output = []
if not results:                      # (1) ultralytics returned no results object
    return output
result = results[0]
if result.masks is None or result.boxes is None or len(result.boxes) == 0:   # (2) zero boxes / no masks
    return output
```

Otherwise **every** returned mask is appended (`output.append(...)`) and only re-ordered by
`(-confidence, index)`; there is no confidence re-filter, no box-size filter, no class filter and no truncation
inside the wrapper. Therefore:

```text
raw_count = 0  ⇔  all 9 tiles hit path (1) or (2)
              ⇔  the underlying model returned zero boxes above conf=0.05 for every tile of A2
```

`detect_global()` itself has no path that discards a non-empty `detect_tile()` result except an empty compacted mask
(`_compact_mask` returns None only for an all-False mask), which cannot occur when boxes are empty in the first
place.

## 5. Local Ultralytics NMS time-limit semantics (installed source)

```text
ultralytics 8.4.164
.conda/buildreasonseg-mvp/Lib/site-packages/ultralytics/utils/nms.py
  L91 : time_limit = 2.0 + max_time_img * bs      # seconds to quit after
  L162: output[xi] = x[i]                          # this image's NMS result is stored first
  L165: if (time.time() - t) > time_limit:
  L166:     LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
  L167:     break                                   # leaves the *batch* loop
```

Consequences established from the source:

1. the warning is emitted **after** the current image's NMS output has already been written to `output[xi]`;
2. the `break` exits only the per-batch loop, so with one tile per `model.predict()` call (batch size 1) the tile's
   detections are still returned;
3. the observed threshold `2.050 s` corresponds to `2.0 + max_time_img * bs` with `max_time_img * bs ≈ 0.05 s`, i.e. a
   ~50 ms budget — so the warning is a cheap CPU-timing symptom, not a result-dropping mechanism;
4. therefore the single warning in A2's transcript cannot explain `raw_count = 0`, and the eight tiles without any
   warning were equally zero.

## 6. Cross-case comparison (same frozen settings)

| case | tiles | raw | merged | status | error | NMS warning |
|---|---:|---:|---:|---|---|---|
| A1 | 9 | 133 | 52 | SUCCESS | – | not recorded |
| A2 | 9 | **0** | **0** | FAILED | E401 | yes (1 occurrence) |
| A3 | 9 | 7 | 6 | SUCCESS | – | not recorded |
| A4 | 9 | 216 | 77 | SUCCESS | – | not recorded |
| B1 | 169 (post-fix run) | 6578 | 3066 | SUCCESS | – | no |
| B2 | 169 (post-fix run) | 7864 | 3740 | SUCCESS | – | no |

A2 is the only sample with a zero detector output, and it is also the only sample whose transcript contains the NMS
warning — but §5 shows the warning cannot produce a zero output, and A2's zero output covers tiles that never warned.
The A2 input itself is byte-identical to the frozen hash, RGB, 1024×1024, i.e. the same decode path as A1/A3/A4.

## 7. Exclusion reasoning

| candidate conclusion | verdict on available evidence |
|---|---|
| `PROP01_NMS_TIMEOUT_SUSPECT` | **excluded** — NMS stores `output[xi]` before the time check and `break` only leaves the batch loop; with batch size 1 the tile's detections are returned; 8 of 9 zero tiles raised no warning at all |
| `PROP01_WRAPPER_DROP_SUSPECT` | **excluded** — the wrapper appends every returned mask and applies no post-filter; it can only return empty when ultralytics itself reports zero boxes |
| `PROP01_INPUT_OR_TILE_PATH_SUSPECT` | **excluded** — A2 is byte-identical to the frozen hash, RGB 1024×1024 like A1/A3/A4, `tile_count = 9` matches the plan, and B1/B2 later proved the tiled path on far larger inputs |

## 8. Primary conclusion (exactly one, not a confirmed root cause)

```text
PROP01_MODEL_ZERO_DETECTION_SUSPECT
```

Reading: for the frozen A2 input, the detector returned **zero boxes above `conf = 0.05` on all nine tiles** — i.e. a
model-side/content-side zero-detection outcome for this specific image — while the NMS warning is a coincident timing
symptom rather than the cause. This remains a *suspect* classification because no post-NMS confidence distribution or
pre-NMS box count for A2 exists in the frozen artifacts, so "zero detections" versus "all detections at/below the
frozen threshold" cannot be distinguished without one controlled run.

## 9. Recommended next diagnostic gate (exactly one — NOT executed)

```text
CONTROLLED_A2_INSPECT_PROPOSALS_RUN
```

Justification: static evidence already excludes wrapper drops, input/tile-path breaks and NMS-timeout truncation, so
the only remaining discriminator is the model's own output distribution on A2 (how many boxes exist before NMS and at
what maximum confidence). That requires exactly one controlled model run with proposal inspection; it was **not**
executed here, and no code was changed pending ChatGPT's audit.

## 10. Scope statement

No `predict.py`, pytest, `check_setup.py`, Demo or model run occurred; `detector.py`, tests, `source_manifest.json`
and the external delivery were not modified; `RC1-DEMO-PROP-01`, `RC1-DEMO-REF-01` and `RC1-DEMO-MASK-01` remain
open and untouched; Task 8B.4 / Task 8C were not entered.
