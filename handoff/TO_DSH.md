# TO_DSH — Task 6A: Native-Windows Environment Bootstrap + 2B `[SEG]` MVP Smoke/Overfit

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Goal: build the **smallest working BuildReasonSeg segmentation pipeline** on the user's actual laptop:
>
> `Qwen3-VL-2B-Instruct -> [SEG] hidden state -> Projection MLP -> vanilla SAM2.1 Hiera Base+ mask decoder`
>
> and prove it with:
>
> 1. environment/model-load measurements;
> 2. a real 2-sample forward/backward pass;
> 3. a deterministic 20-sample overfit test;
> 4. free-generation inference that emits `[SEG]` and produces masks.
>
> This task deliberately does **NOT** use Qwen3-VL-4B, `[REF]`, Spatial Relation Encoder, Spatial Consistency Loss, or full-dataset training.

---

## 0. User-facing language

All narrative text shown in the DSH web/chat UI must be **Chinese**, including progress updates, package/model download explanations, warnings, failures, and the final summary.

Commands, paths, raw logs, identifiers, model IDs, and package names may remain English.

`handoff/FROM_DSH.md` and `handoff/PROJECT_STATE.md` may remain English.

---

## 1. Full-access boundary and new authorization

DSH is still running with Full Access because the normal workspace-write sandbox cannot start on this Windows host. Full Access remains a **technical workaround**, not unlimited permission.

### Allowed writes

Only inside:

`C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\`

plus ordinary OS/runtime temporary directories if unavoidable.

### Newly authorized for Task 6A

Task 6A explicitly authorizes:

- creation of one new dedicated Python environment;
- package installation **inside that environment only**;
- download of exactly:
  - `Qwen/Qwen3-VL-2B-Instruct`
  - `facebook/sam2.1-hiera-base-plus`
- download/install of the official `facebookresearch/sam2` Python source/package;
- normal Hugging Face / ModelScope metadata and weight access;
- lightweight official dependencies needed by the dedicated environment.

### Still forbidden

Do not:

- modify any existing Python/conda environment;
- modify `yolo_sam_env`;
- modify system Python;
- change Windows registry, PATH, drivers, CUDA toolkit, or system-wide env vars;
- install WSL2;
- install/change NVIDIA drivers;
- install/change a system CUDA toolkit;
- download Qwen3-VL-4B or any other large model;
- download external training datasets;
- modify `../WHU_Building_Segment/`;
- modify frozen BuildSpatialReason JSONL;
- train on the full dataset;
- start `[REF]`, Spatial Relation Encoder, or Spatial Consistency Loss work;
- write outside `BuildReasonSeg` except unavoidable temp files.

If native Windows cannot be made to work within the bounded fallback rules below, stop and report the blocker. Do **not** automatically migrate to WSL2.

---

## 2. Accepted project state entering Task 6A

Treat these as frozen facts:

- BuildSpatialReason-v0.1.1 total: 25,229
- train / val / test: 15,592 / 3,884 / 5,753
- independent semantic oracle: 25,229 / 25,229
- dataset quality verdict: PASS
- artifact consistency: consistent
- JSONL hashes frozen
- `datasets/build_spatial_reason/v0.1/` must not be used

Canonical training source:

`datasets/build_spatial_reason/v0.1.1/train.jsonl`

Do not alter any JSONL.

---

## 3. Task 5.5 review corrections that override ADR-012 details

### 3.1 2B is the only model in Task 6A

Use exactly:

`Qwen/Qwen3-VL-2B-Instruct`

Do not download or instantiate 4B.

Interpretation:
- 2B = `SAFE_LOCAL` candidate
- 4B = `BORDERLINE_LOCAL` until measured later

### 3.2 Use SAM2.1 Hiera Base+ for the proof

Use exactly:

`facebook/sam2.1-hiera-base-plus`

Do not use Hiera Large in Task 6A.

### 3.3 `[SEG]` only

Task 6A contains exactly one new architecture token:

`[SEG]`

No `[REF]`.

### 3.4 Do not train a full embedding matrix

Qwen3-VL uses tied input/output embeddings. Training the full resized embedding matrix is memory-inefficient and dangerous.

Preferred implementation:
- current PEFT `trainable_token_indices=[seg_id]` integrated with LoRA;
- preserve weight tying;
- train only the `[SEG]` token row.

Do **not** put the entire vocabulary embedding matrix into AdamW merely with a gradient hook.

If installed PEFT cannot safely train only selected token indices with this model:
1. do not silently unfreeze the full embedding matrix;
2. implement a small explicit trainable-token adapter/override;
3. prove with a test that normal token rows remain unchanged after an optimizer step.

### 3.5 Qwen and SAM image preprocessing are separate branches

Do not assume “512×512 everywhere”.

The source WHU image is 512×512.

Expected architecture:

```text
same source RGB 512x512
        |
        +--> Qwen3-VL processor branch
        |      measure actual processed shape
        |      measure image_grid_thw
        |      measure actual visual-token count
        |
        +--> SAM2.1 branch
               official SAM2 transform/predictor
               verify internal image size at runtime
               frozen image encoder
```

Do not hard-code a visual-token count without measuring processor output.

---

## 4. Native Windows environment bootstrap

### 4.1 Preflight first — no changes yet

Record:
- `where python`
- Python versions available
- conda/mamba/venv availability
- active env name
- current torch path/version without modifying it
- GPU name
- driver version
- `torch.cuda.is_available()`
- CUDA runtime reported by torch
- compute capability
- `torch.cuda.get_arch_list()`
- total/free VRAM
- free disk space on project drive
- Git clean/dirty state
- relevant cache env vars

Write:

`evaluation/task6a_preflight.json`

### 4.2 Dedicated environment

Create exactly one dedicated environment.

Preferred order:
1. if conda/mamba exists, create a **new environment**, not a clone of `yolo_sam_env`;
2. otherwise create a project-local venv such as `.venv-buildreasonseg-mvp`.

Python:
- prefer 3.11;
- Python 3.10 acceptable if 3.11 unavailable.

Never modify an existing env.

Add local env/cache paths to `.gitignore`.

### 4.3 PyTorch

Use an **official GPU-capable PyTorch build** that works with the RTX 5080 Laptop GPU.

Preferred:
- reproduce the already measured GPU-capable PyTorch/CUDA family if an official install source is available;
- otherwise use the current official stable CUDA wheel supporting this GPU.

Do not continue until the new env independently verifies:

```text
torch.cuda.is_available() == True
compute_capability == (12, 0)
CUDA tensor operations succeed
```

Record exact versions and install source.

Do not modify drivers or system CUDA to solve a package mismatch.

### 4.4 Core packages

Install only what Task 6A needs, including as appropriate:
- transformers (Qwen3-VL capable; compatible major version)
- peft
- accelerate
- safetensors
- huggingface_hub
- qwen-vl-utils
- pillow
- numpy
- opencv-python if required
- pytest
- pyyaml
- tqdm

After success, freeze exact relevant versions into:

`environment/task6a_requirements_lock.txt`

Do not blindly freeze unrelated system packages.

### 4.5 SAM2 installation on native Windows

Use official `facebookresearch/sam2`.

Official SAM2 documentation recommends WSL on Windows, but the CUDA extension is optional. Task 6A must intentionally skip that extension using:

`SAM2_BUILD_CUDA=0`

or the equivalent process-local Windows setting.

Do not attempt to compile the optional CUDA extension in Task 6A.

Record the exact SAM2 source revision used.

---

## 5. Local cache and asset policy

Keep model caches/downloads inside ignored project paths, for example:

```text
local_cache/
  huggingface/
  models/
  sam2_source/
artifacts/
  checkpoints/
  stage_outputs/
```

Requirements:
- all large assets gitignored;
- no model weights staged;
- no environment directory staged;
- no external repository contents committed.

Before commit, explicitly search for `.safetensors`, `.pt`, `.pth`, env folders, caches, and checkpoints.

---

## 6. Required code architecture

Create a small modular MVP package. Names may vary, but responsibilities must stay separate.

Recommended:

```text
buildreasonseg_mvp/
  __init__.py
  data.py
  qwen_seg.py
  sam2_bridge.py
  model.py
  losses.py
  checkpointing.py
  metrics.py

configs/mvp/
  task6a_2b_seg.yaml

scripts/
  task6a_probe.py
  task6a_smoke2.py
  task6a_overfit20.py
  task6a_infer.py

tests/
  test_task6a_data.py
  test_task6a_token.py
  test_task6a_bridge.py
  test_task6a_trainables.py
```

Keep Task 6A independent of legacy YOLO code.

Do not import YOLO code.

---

## 7. Training sample format — Chinese-only for Task 6A

Task 6A is a pipeline proof, not the final language policy.

### Input
- source image;
- `instruction_zh`.

### Assistant target text

`reasoning_zh + " [SEG]"`

Requirements:
- `[SEG]` exactly once;
- `[SEG]` one tokenizer token;
- no component IDs exposed in natural text;
- `reasoning_steps` never fed to model;
- GT component geometry never model input.

Mask supervision comes only from:

```text
component_map_path
+
target_component_id / target_mask.component_id
```

GT mask is supervision/evaluation only, never inference input.

---

## 8. Deterministic 2-sample and 20-sample subsets

Do not choose arbitrary first-N records.

### 8.1 Two-sample smoke pair

Select two records from the **same source image** with:
- different instructions;
- different target component IDs;
- preferably different query types.

Save sample IDs in:

`evaluation/task6a_subset_ids.json`

### 8.2 Twenty-sample overfit set

Construct a deterministic set of approximately:

`10 source images × 2 instructions per image = 20 records`

Requirements:
- all from train;
- no val/test;
- cover L1, L2, and nontrivial L3 where available;
- include direction / nearest / size / extreme families;
- each paired image has two different target masks;
- selection is deterministic and recorded;
- do not optimize subset choice using model results.

Purpose: make image-only memorization insufficient; the model must attend to the instruction.

---

## 9. Qwen3-VL `[SEG]` pathway

### 9.1 Token setup

- add `[SEG]` as an additional special token;
- verify tokenizer round-trip;
- assert literal `[SEG]` maps to exactly one token;
- record `seg_token_id`;
- resize embeddings exactly once;
- verify tied-weight state after resize.

### 9.2 Selective token training

Prefer PEFT LoRA with:

`trainable_token_indices=[seg_token_id]`

Verify:
- only new token is trainable in token adapter;
- full vocab embedding is not made trainable;
- tied output behavior remains valid.

Automated regression:
- snapshot several normal token embeddings;
- run one optimizer step;
- ordinary rows unchanged within strict tolerance;
- `[SEG]` row changes.

### 9.3 LoRA scope

Target language-model layers only.

Desired projections:
- q
- k
- v
- o
- gate
- up
- down

Do **not** attach LoRA to the visual tower merely because names overlap.

Programmatically inspect full module names and use a text-only target set.

After injection assert:

`number of LoRA modules under vision tower == 0`

Record total/trainable/LoRA/token params and LoRA module names.

### 9.4 Hidden-state extraction under teacher forcing

1. build chat input from image + `instruction_zh`;
2. assistant target = `reasoning_zh + " [SEG]"`;
3. mask all user/image/prompt tokens in LM labels;
4. LM CE only on assistant target tokens;
5. request hidden states;
6. locate actual `[SEG]` position from token IDs;
7. extract final-layer hidden state.

Do not hard-code hidden size; read config.

### 9.5 Inference hidden state

For free generation:
1. generate from image + instruction only;
2. verify `[SEG]` emitted;
3. re-run complete generated sequence with `output_hidden_states=True`;
4. locate generated `[SEG]`;
5. decode mask from that hidden state.

Do not use GT reasoning or target to obtain inference hidden state.

---

## 10. Qwen visual preprocessing measurements

For a real WHU 512×512 image record:
- original shape;
- processor pixel tensor shape;
- `image_grid_thw`;
- actual visual-token count;
- total sequence length for at least one L1, one L2, one L3 sample.

Do not state “256 visual tokens” unless measured output confirms it.

Set intended image budget explicitly where the processor API supports it without distorting aspect ratio.

---

## 11. Vanilla SAM2.1 language bridge

Do **not** depend on Sa2VA custom code.

Use official vanilla SAM2.1.

### 11.1 Frozen/trainable SAM modules

Freeze:
- image encoder/backbone;
- memory modules / memory attention;
- prompt encoder parameters.

Train:
- SAM2 mask decoder;
- BuildReasonSeg projection MLP.

### 11.2 Image feature path

Use official SAM2 image preprocessing/predictor behavior.

Image encoder runs under `torch.no_grad()`.

Measure:
- source shape;
- transformed SAM tensor shape;
- `sam_model.image_size`;
- image embedding shape;
- high-res feature shapes.

Verify internal resolution rather than assuming it.

### 11.3 Language-as-sparse-prompt bridge

Implement a minimal project-owned bridge:

```text
Qwen [SEG] hidden
    -> Projection MLP
    -> SAM prompt embedding dimension
    -> one sparse prompt token
    -> vanilla SAM2 mask decoder
```

Use official prompt encoder only to obtain:
- positional encoding;
- no-mask dense embedding.

Inject projected `[SEG]` as the sparse prompt embedding passed to `sam_mask_decoder`.

Requirements:
- projection output dim read from SAM prompt-encoder embed dim;
- no GT point/box/mask prompt supplied at inference;
- no GT centroid/bbox/component id supplied at inference;
- no reference mask;
- `multimask_output=False`;
- use official high-res image features if selected config expects them.

Keep projection separate so a later Spatial Relation Encoder can be inserted.

### 11.4 Mask resolution

For training:
- use SAM mask logits;
- either resize GT mask to logit resolution with nearest-neighbor or postprocess logits to original size;
- document the choice and keep deterministic.

For evaluation/visualization:
- output aligned to original 512×512;
- fixed threshold;
- IoU computed at original resolution.

---

## 12. Losses — provisional smoke config

Initial objective:

```text
L_total =
    1.0 * L_lm_ce
  + 2.0 * L_mask_bce
  + 0.5 * L_mask_dice
```

This is provisional smoke/overfit config, not a paper hyperparameter.

Requirements:
- BCE-with-logits;
- soft Dice + epsilon;
- report components separately;
- no Spatial Consistency Loss;
- no reference-mask loss.

If this prevents the 20-sample sanity overfit, you may adjust mask-loss weights only after recording the initial result. No broad hyperparameter search.

---

## 13. Optimizer/training policy

Suggested:
- batch size 1;
- gradient accumulation as needed;
- BF16 autocast if stable;
- AdamW;
- zero weight decay on trainable special-token representation;
- modest weight decay elsewhere if appropriate;
- gradient clipping;
- `use_cache=False` during training;
- deterministic seed.

No QLoRA in Task 6A unless BF16 2B unexpectedly OOMs. If BF16 2B OOMs, stop and report before adding quantization.

Trainable only:
- text-only LoRA adapters;
- `[SEG]` trainable token representation;
- projection MLP;
- SAM2 mask decoder.

Everything else frozen.

---

## 14. Stage 0 — environment/model-load proof

Load:
- Qwen3-VL-2B-Instruct;
- SAM2.1 Hiera Base+.

No training yet.

Record Qwen:
- RAM before/after;
- GPU allocated/reserved peak;
- dtype;
- embedding tie;
- hidden size;
- vocab before/after `[SEG]`;
- processor outputs on a real WHU image.

Record SAM:
- image size;
- total params;
- mask-decoder params;
- transformed input shape;
- embedding/high-res feature shapes;
- GPU peak after feature extraction.

Then load both together and record combined idle allocated/reserved/free VRAM.

If combined load is unsafe, stop before training.

---

## 15. Stage 1 — real 2-sample forward/backward

Use the fixed same-image/different-target pair.

Perform real forward/backward through:

```text
image + instruction
    -> Qwen teacher-forced reasoning + [SEG]
    -> [SEG] hidden
    -> projection
    -> SAM2 mask decoder
    -> LM + BCE + Dice
```

Success criteria:
- all losses finite;
- projected embedding shape correct;
- predicted mask logits finite and non-constant;
- non-zero finite grads on LoRA, `[SEG]`, projection, SAM2 mask decoder;
- zero trainable params in Qwen visual tower;
- frozen SAM2 encoder has no grads;
- no GT geometry passed as input;
- no OOM;
- true peak VRAM recorded.

Run at least one optimizer step and verify:
- `[SEG]` trainable state changes;
- normal vocab samples unchanged;
- frozen modules unchanged.

Do not proceed to Stage 2 until Stage 1 passes.

---

## 16. Stage 2 — deterministic 20-sample overfit

### 16.1 Goal

Prove the architecture can intentionally memorize paired instruction-mask examples where the same image has different targets under different instructions.

### 16.2 Training bound

Use early stopping.

Suggested bounds:
- max 1000 optimizer steps;
- or ~2–3 hours wall time;
- stop earlier when success is repeatedly met.

Do not run indefinitely.

### 16.3 Primary success criteria

- teacher-forced training mIoU on all 20 >= **0.90**;
- no NaN/Inf;
- no empty/full-image collapse;
- instruction-specific behavior on paired images.

### 16.4 Same-image pair sanity

For each pair A/B:
- prediction under instruction A must overlap GT-A more than GT-B;
- prediction under instruction B must overlap GT-B more than GT-A.

Target at least:

`9 / 10 pairs`

Record every pair result.

### 16.5 Free generation

For all 20:
- image + instruction only;
- generate assistant output;
- count whether `[SEG]` appears exactly once;
- re-forward generated text to obtain `[SEG]` hidden;
- decode mask.

Target:

`20 / 20 [SEG] emission`

If mask overfit succeeds but free-generation emission is imperfect, verdict may be `PASS_WITH_WARNINGS`; report exact count.

If teacher-forced mIoU cannot exceed 0.90 within bounded training, verdict cannot be PASS.

### 16.6 Optional frozen SAM feature cache

Because SAM image encoder is frozen and only ~10 unique images are used, after Stage 1 you may cache detached SAM image/high-res features locally for Stage 2.

If used:
- ignored local path;
- record dtype/shapes;
- verify cached vs uncached decoder outputs on one sample.

Do not cache Qwen hidden states.

---

## 17. Required measurements

Record at minimum:
- exact package versions;
- model revisions/commit hashes where available;
- actual download sizes on disk;
- env disk size;
- Qwen load VRAM;
- SAM load VRAM;
- combined idle VRAM;
- Stage 1 forward peak;
- Stage 1 backward peak;
- Stage 2 peak;
- stage runtimes;
- Qwen processor shape;
- `image_grid_thw`;
- measured visual-token count;
- sequence lengths;
- SAM transformed resolution;
- SAM embedding/high-res shapes;
- `[SEG]` id;
- `[SEG]` hidden shape;
- projection shape;
- mask-logit shape;
- trainable param counts by module.

Clearly separate MEASURED from CONFIGURED values.

---

## 18. Checkpointing

For Stage 2 save ignored local checkpoints:
- `last`
- `best` by training mIoU

Checkpoint only adapter state needed to reproduce the MVP:
- LoRA adapter;
- trainable `[SEG]` token state;
- projection MLP;
- SAM2 mask decoder;
- optimizer/scheduler if resume supported;
- config;
- RNG state if practical.

Do not duplicate base Qwen/SAM weights.

Create committed:

`evaluation/task6a_checkpoint_manifest.json`

with paths, sizes, SHA256, base model IDs/revisions; do not commit checkpoint bytes.

---

## 19. Visual outputs

Create lightweight committed outputs under:

`evaluation/task6a_samples/`

At least:
- both smoke-pair samples;
- 4–6 Stage-2 examples;
- at least two same-image/different-instruction pairs.

Show:
- source image;
- GT mask;
- predicted mask;
- instruction;
- query type;
- IoU.

Keep pack small.

---

## 20. Tests

Add fast tests for at least:

1. v0.1.1 target-mask reconstruction;
2. `[SEG]` is one token;
3. only assistant target is labeled for LM CE;
4. `reasoning_steps` / component IDs are not model inputs;
5. selective `[SEG]` training leaves normal vocab rows unchanged;
6. LoRA targets no visual-tower module;
7. SAM bridge dimensions;
8. SAM image encoder frozen;
9. projection + mask decoder trainable;
10. no `[REF]` path in Task 6A;
11. inference consumes no GT geometry;
12. paired subset has different targets per same image.

Also preserve a Stage-1 real-model integration result, but do not force model downloads from ordinary pytest.

Run the full lightweight test suite and report:
- exact command;
- collected;
- passed;
- failed;
- skipped;
- exit code.

---

## 21. Required reports

Create:

```text
evaluation/task6a_preflight.json
evaluation/task6a_environment.json
evaluation/task6a_subset_ids.json
evaluation/task6a_smoke_report.json
evaluation/task6a_checkpoint_manifest.json
docs/task6a_mvp_smoke.md
```

`task6a_smoke_report.json` must include machine-readable sections for:
- environment;
- model revisions;
- preprocessing;
- trainable params;
- Stage 0;
- Stage 1;
- Stage 2;
- pairwise instruction dependence;
- free-generation `[SEG]` emission;
- VRAM;
- timings;
- checkpoint hashes;
- final verdict;
- blockers/warnings.

Allowed verdicts:
- `PASS`
- `PASS_WITH_WARNINGS`
- `FAIL_REQUIRES_DEBUG`

Model load alone is never PASS.

---

## 22. ADR update

Do not rewrite Task 5.5 broadly.

Append a measurement note to ADR-012 or create narrow ADR-013 recording:
- 2B + Base+ measured proof stack;
- 4B still unmeasured/borderline;
- `[SEG]` selectively trainable;
- separate Qwen/SAM preprocessing branches;
- SAM2 optional CUDA extension intentionally disabled on native Windows;
- Task 6A measurements supersede overlapping Task 5.5 estimates.

Only mark ADR-013 accepted if Stage 1 succeeds.

---

## 23. Git hygiene

Before commit:
1. `git status --short`
2. confirm no environment dir staged
3. confirm no HF cache staged
4. confirm no `.safetensors`, `.pt`, `.pth`, model weights, checkpoint bytes, or downloaded SAM repo staged
5. confirm no BuildSpatialReason JSONL changed
6. run `python scripts/check_artifact_consistency.py`
7. inspect all staged files

Commit only source/config/tests/small reports/docs/lightweight visualizations/handoff state/`.gitignore`.

Recommended commit:

`feat: prove 2B SEG MVP pipeline`

Then:

`git push origin main`

If push fails, record exact blocker.

---

## 24. Handoff

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

English is allowed.

`FROM_DSH.md` should include:
1. Verdict
2. Environment Created
3. Packages / Model Revisions
4. Stage 0 Measurements
5. Qwen Preprocessing
6. SAM2 Preprocessing
7. `[SEG]` Token / PEFT Verification
8. LoRA Scope Verification
9. Stage 1 Forward/Backward
10. Stage 2 Overfit
11. Same-Image Pair Test
12. Free-Generation Test
13. VRAM / Runtime
14. Checkpoint Manifest
15. Tests
16. Git Status / Push
17. Known Problems
18. Recommendation for Task 6B

---

## 25. Final DSH web/chat response — Chinese only

The final visible response must concisely report:
- Task 6A verdict;
- dedicated env created or not;
- 2B + SAM2 Base+ load success;
- measured Stage-1 peak VRAM;
- 2-sample backward pass/fail;
- final 20-sample training mIoU;
- same-image pair pass count;
- `[SEG]` free-generation emission count;
- Stage-2 peak VRAM;
- commit hash;
- push success/failure;
- any blocker requiring user action.

Do not paste the full technical report into the UI.

---

## 26. Stop condition

After Task 6A:

**STOP.**

Do not:
- download Qwen3-VL-4B;
- add `[REF]`;
- implement Spatial Relation Encoder;
- implement Spatial Consistency Loss;
- start full-dataset training;
- modify external datasets.

Wait for ChatGPT to review the pushed Task 6A implementation and measured results.
