# Baseline: YOLOv8m-seg on WHU Building Dataset

**Status: FROZEN. Read-only reference. Not part of the proposed method.**

This directory is a lightweight *record* of an already-completed experiment. It contains no
model code. Nothing here is trained, modified, or extended by this project.

---

## 1. Role

This model **is**:

- a building **instance segmentation baseline** for the WHU Building Dataset.

This model **is NOT**:

- an MLLM;
- a reasoning segmentation method;
- a Spatial Relation Encoder;
- the proposed method;
- the backbone of any future model;
- an inference-time component of the proposed pipeline.

The proposed method **must not import any YOLO code**. YOLO exists here solely as a
performance floor and a sanity check on the dataset.

See `docs/architecture_decisions.md` → **ADR-001**.

---

## 2. What was trained

| Item | Value |
|---|---|
| Architecture | YOLOv8m-seg (`SegmentationModel`) |
| Pretrained weights | `weights/yolov8m-seg.pt` (COCO-seg, nc=80 → re-headed to nc=1) |
| Dataset | WHU Building Dataset, YOLO polygon format |
| Classes | 1 — `building` |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 |
| Device | `0` |
| Seed | 0 (deterministic) |
| AMP | enabled |
| Frozen layers | none (`freeze: null`) |
| Total training time | 4674.74 s (~78 min) |

Full hyperparameters: `run_record/args.yaml`. Machine-readable record: `manifest.json`.

---

## 3. Existing result (confirmed from `run_record/results.csv`)

These are **validation-split** numbers. 100 epoch rows are recorded.

**Final epoch (100):**

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.81787 | 0.82810 |
| Recall | 0.77094 | 0.77045 |
| mAP50 | 0.81416 | 0.80693 |
| mAP50-95 | 0.49151 | 0.43388 |

**Best recorded epochs (not the final checkpoint):**

| Metric | Value | Epoch |
|---|---|---|
| Mask mAP50 | **0.84373** | 44 |
| Mask mAP50-95 | 0.46655 | 31 |
| Box mAP50-95 | 0.52396 | 52 |

> The mask mAP50 peak (epoch 44) is **higher** than the final epoch value (epoch 100).
> Because `patience` (100) equals `epochs` (100), early stopping never triggered, so the
> retained checkpoint is simply the last epoch rather than the best one on this metric.

---

## 4. Test-set metrics: NONE EXIST

**There are currently no formal test-set metrics for this baseline.**

The 903 prediction images produced during that run are **inference visualizations only**.
They are *not* equivalent to a quantitative test evaluation. No metric was ever computed on
the test split, because `save_json` was `false` in `args.yaml`, so no machine-readable
prediction file was emitted.

Any test-set number used in a paper or report must come from a proper evaluation protocol
that has not yet been run.

---

## 5. Dependency warning

> **The existing YOLO baseline depends on an editable install of `ultralytics` that resolves
> to a mutable source directory OUTSIDE this workspace.**

Consequences:

- The baseline is **not portable** and **not reproducible** from this repository alone.
- The external source directory can change without notice, silently altering behaviour.
- The external fork's own import chain is partially broken (its SAM predictor imports fail).

**Therefore:**

> The new proposed method **must not depend on this mutable external fork.**
> It will declare its own pinned dependencies. See `manifest.json` → `known_issues` → `KI-01`.

---

## 6. Contents

```
baseline/yolo_whu/
├── README.md                  # this file
├── manifest.json              # machine-readable baseline record
├── config/
│   └── building_dataset.yaml  # dataset config (byte-identical to legacy)
├── run_record/
│   ├── args.yaml              # frozen training arguments (byte-identical to legacy)
│   └── results.csv            # per-epoch metrics, 100 rows (byte-identical to legacy)
└── plots/
    ├── results.png            # training curves (byte-identical to legacy)
    └── confusion_matrix_normalized.png
```

**Deliberately NOT copied:** the dataset, the 903 prediction JPEGs, and the checkpoints.
Checkpoints are referenced by path + SHA256 in `manifest.json` instead of being duplicated.

---

## 7. Provenance

> **Confirmation status: `user_confirmed`.**
> The statement below was provided by the project owner. It is **not** the
> result of automatic verification from the checkpoint or the source code.

**User-confirmed:** the baseline is **official pretrained YOLO weights
fine-tuned on a subset of the WHU Building Dataset** — specifically the
`Satellite dataset II (East Asia)` cropped image data and raster labels.

Two parts of that statement were **independently verified** while building this
record, from the artefacts themselves:

| Claim | Verified how |
|---|---|
| The starting weights are official pretrained segmentation weights | `weights/yolov8m-seg.pt` loads as `SegmentationModel` with `nc=80` and the 80 COCO class names, dated 2023-01-04 |
| The data is the WHU Building Dataset `Satellite dataset II (East Asia)` cropped tiles | `scripts/mask_to_yolo.py` lines 11-12 name the source and output roots; the derived YOLO dataset holds 4038 image/label pairs of 512x512 |

The word "subset" is taken from the user statement; the exact proportion of the
WHU release used here was **not** independently confirmed.

Machine-readable form: `manifest.json` -> `provenance`.

---

## 8. Known exception: an absolute path inside the frozen `args.yaml`

`run_record/args.yaml` line 110 contains the following historical value (quoted here for
documentation; this line is an allowed exception to the absolute-path audit):

```yaml
save_dir: C:\D\resources\project\WHU_Building_Segment\runs\segment\logs\whu_building_v1
```

This value was **written by Ultralytics itself** when the training run executed, not authored
by this project. It is retained deliberately, because:

- `args.yaml` is a byte-identical frozen copy of the legacy run record, and editing it would
  destroy its value as evidence;
- it is the only direct proof of which directory the run actually executed from;
- **no program in this repository reads `args.yaml`.** It is a record, not a config.

This is the **only** absolute path that appears as a configuration *value* anywhere in
`BuildReasonSeg/`, and it is confined to this frozen record. Every other absolute path
appears solely inside `manifest.json`, under a key explicitly suffixed
`_historical_absolute`, marked provenance-only. No config or source file in this repository
may depend on any of them.

