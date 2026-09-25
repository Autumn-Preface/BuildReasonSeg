# BuildReasonSeg

**BuildReasonSeg is the working codename of the project on
spatial-reasoning-guided building reasoning segmentation.**

The official Chinese competition title of the project is unchanged; the codename
only names this repository. The project is **not tied to WHU**: WHU is simply the
first source dataset. Further public building / remote-sensing datasets are
expected to be added later. See `docs/architecture_decisions.md` → ADR-009.

Canonical development root for the **proposed method** of this project.

Research direction: **spatial-reasoning-guided building structure reasoning segmentation**
(a task description; "spatial reasoning segmentation" remains a generic technical term).

> This repository is the *only* place where the proposed method is developed.
> The legacy `../WHU_Building_Segment/` directory is kept in place as read-only
> **baseline evidence** and **geometry GT source**. It is never modified from here,
> and its name is deliberately unchanged because it is a historical experiment.

---

## Naming

| Kind | Name | Notes |
|---|---|---|
| Project codename / repo | **BuildReasonSeg** | this repository |
| Future reasoning dataset | **BuildSpatialReason** (`build_spatial_reason`) | **not yet generated** |
| Source dataset (first) | WHU Building Dataset | historical, unchanged |
| Source dataset directory | `datasets/whu/` | source-specific raw/derived data |
| Legacy baseline experiment | `../WHU_Building_Segment/` | historical, unchanged |
| Legacy baseline record | `baseline/yolo_whu/` | historical, unchanged |

"WHU" is retained wherever it denotes a **real historical source or provenance**;
it is not renamed away.

---

## Status

```
[x] YOLO baseline                (frozen record only; see baseline/yolo_whu/)
[ ] BuildSpatialReason           (spatial reasoning annotation dataset)
[ ] Reasoning Segmentation MVP   (MLLM -> [SEG] -> mask decoder)
[ ] Spatial Relation Module      (Spatial Relation Encoder + Feature Fusion)
[ ] Spatial Consistency Loss
[ ] Evaluation & Ablation suite
```

**Nothing below the first line is implemented yet.** This README describes the intended
architecture so that the boundary between done and planned stays explicit. Do not read the
target architecture as a description of existing code.

---

## Development architecture

```
      Legacy YOLO Baseline
      (frozen, ../../WHU_Building_Segment/)
                |
                |  geometry GT  (instance polygons, read-only)
                v
        BuildSpatialReason
        (instruction + relations + target instance + mask)
                |
                v
      Reasoning Segmentation MVP
      Image + Instruction -> MLLM -> [SEG] -> Mask Decoder -> mask
                |
                v
        Spatial-Enhanced Model
        H_seg + H_spatial + F_visual -> Fusion -> Mask Decoder -> mask
```

---

## Baseline line

**YOLOv8m-seg** — building instance segmentation on the WHU Building Dataset.

- **Retained.** Kept in its original location, untouched.
- **Frozen.** A byte-identical record of its config and run output is archived under
  `baseline/yolo_whu/`; the checkpoints are referenced by path + SHA256, not copied.
- **Purpose: comparison / sanity check only.** It provides a performance floor
  (val mask mAP50 = 0.80693 at epoch 100, peak 0.84373 at epoch 44) and confirms that the
  dataset and polygon geometry are usable.

The baseline is **not** a component of the proposed method. The proposed method does not
import YOLO code. See `docs/architecture_decisions.md` → ADR-001.

---

## Proposed method line

The target pipeline, none of which exists yet:

```
Image + Instruction
  -> MLLM
  -> reasoning text + [SEG] token
  -> [SEG] hidden embedding
  -> Spatial Relation Encoder
  -> semantic-spatial-visual fusion
  -> segmentation decoder
  -> target mask
```

Design constraints already fixed (see `docs/architecture_decisions.md`):

- GT geometry is **supervision only** and must never be a model input at inference time (ADR-002).
- The usable spatial relation set is limited by the actual WHU annotation topology (ADR-003).
- Every instruction must be semantically complete: any criterion that decides the target must
  be stated in the instruction text (ADR-004).
- The Spatial Relation Encoder must be inference-valid; its design is deferred to Task 8 (ADR-005).

---

## Layout

| Path | Purpose |
|---|---|
| `baseline/yolo_whu/` | Frozen YOLO baseline record. Reference only. |
| `configs/` | All configuration. No hyperparameters hard-coded in Python. |
| `datasets/whu/` | Dataset documentation and (later) dataset code. **No image copies.** |
| `spatial_reasoning/` | Geometry, relations, thresholds, annotation generation, relation encoder, losses. |
| `models/` | Network definitions (MLLM wrapper, encoders, fusion, decoders). |
| `training/` | Training engine and staged training loops. |
| `evaluation/` | Segmentation metrics, reasoning metrics, ablation harness. |
| `scripts/` | Thin CLI entry points. |
| `tests/` | Unit tests, including leakage regression tests. |
| `docs/` | Architecture decisions, data schema, relation definitions, protocols. |

---

## Data

WHU images are **not copied** into this repository. The legacy dataset stays where it is and
is referenced by a relative path. See `datasets/whu/README.md`.

New annotations derived from that geometry GT are written **only** under this repository.

---

## Not yet implemented

To keep expectations accurate, none of the following exists in this repository yet:

- any MLLM or LLM code;
- any `[SEG]` token or projection module;
- any segmentation decoder;
- any spatial annotation generation;
- any Spatial Relation Encoder;
- any spatial consistency loss;
- any training or evaluation entry point.
