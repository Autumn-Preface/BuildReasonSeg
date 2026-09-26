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
| Reasoning dataset | **BuildSpatialReason** (`build_spatial_reason`) | **v0.1.1 generated and validated** |
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
[x] BuildSpatialReason           (v0.1.1 generated + independently audited, verdict PASS)
[x] MVP stack selection          (Task 5.5 research freeze; see docs/research/)
[x] Reasoning Segmentation MVP   (Task 6A: 2B [SEG] pipeline proven end to end, verdict PASS)
[x] MVP generalization audit     (Task 6B: language generalises, masks do not; verdict FAIL_REQUIRES_DEBUG)
[ ] Spatial Relation Module      (Spatial Relation Encoder + Feature Fusion)  -- the lead contribution
[ ] Spatial Consistency Loss
[ ] Evaluation & Ablation suite
```

**The first five lines are done.** The MVP chain `image + instruction -> Qwen3-VL-2B -> reasoning text +
[SEG] -> projection -> SAM2.1 -> mask` is implemented, measured on the target laptop, proven on a
deterministic 20-sample overfit set (`docs/task6a_mvp_smoke.md`, ADR-013) and then audited on a real
480-record mini-train (`docs/task6b_minitrain.md`, ADR-014). The audit splits the result in two:

* the **language pathway generalises** — on 120 unseen validation records the model emits exactly one
  `[SEG]` 120/120 times with 1.000 reasoning exact match and 1.000 operation-chain accuracy;
* the **mask pathway does not** — strict end-to-end validation mIoU is **0.1102**, and the paired
  instruction-dependence probe scores **0/20** because the projection collapses to a near-constant
  256-d prompt, so two different instructions on one image produce masks with IoU 0.9999.

Do not read the target architecture as a description of existing code beyond what is listed here, and do
not read the language metrics as evidence of reasoning: the dataset's `reasoning_zh` is template-generated
(21 distinct strings in the whole 480-record training mini-set).

### MVP environment (Task 6A, reused by Task 6B)

| Item | Value |
|---|---|
| Environment | `.conda/buildreasonseg-mvp` (conda `--prefix`, Python 3.11.16, gitignored) |
| PyTorch | 2.13.0+cu132, RTX 5080 Laptop (sm_120), 15.89 GiB VRAM |
| Models | `Qwen/Qwen3-VL-2B-Instruct` + `facebook/sam2.1-hiera-base-plus` (both in `local_cache/`, gitignored) |
| Trainable | text LoRA (196 modules) + one `[SEG]` row + projection MLP + SAM2 mask decoder = 1.078 % of parameters |
| Offline | Task 6B runs with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`: `huggingface.co` does not resolve on this network after the local accelerator was removed |
| Reproduce | `environment/task6a_requirements_lock.txt`, then `scripts/task6a_download.py`, the four `scripts/task6a_*.py` stage scripts, and `scripts/task6b_*.py` |

### Frozen MVP stack (Task 5.5, ADR-012)

| Item | Choice |
|---|---|
| Base MLLM | `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0), stages 0–2 on the 2B sibling |
| Mask decoder | `facebook/sam2.1-hiera-large` (Apache-2.0), image encoder + memory frozen |
| Pathway | LISA-style `[SEG]` for the MVP; `[REF] + [SEG]` from Stage 3 |
| Trainable | LoRA r=16 on `q,k,v,o,gate,up,down`, `[SEG]`/`[REF]` embeddings, projection MLP, SAM2 mask decoder |
| Environment | native Windows 11, new dedicated conda env, PyTorch SDPA (no FlashAttention); WSL2 is the fallback |
| Images | native 512×512, no upscaling, 256 visual tokens per tile, pixel budget set explicitly |
| External training data | **none** — EarthReason / RefSegRS are reserved for evaluation |

Evidence, alternates and the exact Task 6 entry criteria:
`docs/research/task5_5_mvp_decision.md`, `evaluation/task5_5_stack_decision.json`.

### Contribution framing after the novelty audit

The `[SEG]` token is ubiquitous and a `[REF]` reference token is **already published** (SegLLM, 2024;
PSALM, 2024), and instruction-tuned building analysis is published prior art (ISPRS Annals, 2026). The
retained contributions are therefore: **geometry-verifiable relation supervision**, an **explicit
Spatial Relation Encoder over the model's own predicted region geometry**, and a **relation-level Spatial
Consistency Loss**. See `docs/research/task5_5_novelty_collision.md`.

<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

The block above is machine-checked against
`evaluation/build_spatial_reason_artifact_index.json` by
`scripts/check_artifact_consistency.py`.


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
- any Spatial Relation Encoder;
- any spatial consistency loss;
- any training or evaluation entry point.

The spatial annotation generator does exist (`scripts/build_spatial_reason.py`);
it is the *dataset* line, not the model line.
