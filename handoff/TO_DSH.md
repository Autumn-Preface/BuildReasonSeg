# TO_DSH — Task 5.5: External Research, Model-Stack Verification & BuildReasonSeg-MVP Design Freeze

> Status: ACTIVE
> Repository: `BuildReasonSeg`
> Goal: use current public evidence to select a technically defensible, license-compatible, hardware-feasible model stack for the first **BuildReasonSeg-MVP**, while mapping the project's novelty against 2024–2026 reasoning-segmentation work.
> This is a research/architecture-selection task, not a model-training task.

## 0. User-facing language

All DSH web/chat narrative visible to the user must be in Chinese:
- progress updates;
- research summaries;
- warnings;
- permission/escalation explanations;
- final recommendation;
- blockers.

Commands, paths, code identifiers, model IDs, paper titles, raw logs, and field names may remain English.

`handoff/FROM_DSH.md` and `handoff/PROJECT_STATE.md` may remain English.

## 1. Full-access and network boundary

DSH currently has Full Access only because the normal workspace-write sandbox cannot start correctly on this Windows host. Full Access is a technical workaround, not permission to perform unrelated actions.

Allowed writes:
- only inside `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\`
- ordinary runtime temp directories when technically required.

Task 5.5 explicitly authorizes read-only public web research. Allowed sources:
- official GitHub repositories;
- official project pages;
- arXiv;
- publisher/CVF/ISPRS/IEEE pages;
- Hugging Face official model/dataset cards and file metadata;
- official vendor documentation;
- official license files.

Allowed network actions are limited to retrieving lightweight public metadata, HTML/text, JSON, README/LICENSE/config information, paper abstracts/pages, and repository metadata.

Forbidden:
- model-weight downloads;
- dataset downloads;
- cloning large external repositories;
- package installation;
- system/registry/PATH/WSL/CUDA/driver changes;
- local model inference or training;
- modifying `../WHU_Building_Segment/`;
- writes outside `BuildReasonSeg`;
- cloud resource creation/costs;
- starting Task 6 implementation.

If a source requires authentication or a large download, mark it unavailable/unverified.

## 2. Accepted dataset state

Treat BuildSpatialReason-v0.1.1 as frozen and accepted:

- total 25,229
- train 15,592 / val 3,884 / test 5,753
- L1 17,275 / L2 5,036 / L3 2,918
- independent semantic oracle 25,229 / 25,229 target match
- semantic violations 0
- reasoning ID leakage 0
- artifact consistency gate consistent
- scene-level split leakage unverified

Canonical evidence:
- `evaluation/build_spatial_reason_v0.1.1_quality.json`
- `evaluation/build_spatial_reason_artifact_index.json`

Do not use v0.1 for future training decisions.

## 3. Preflight housekeeping

Before research, fix the two non-blocking artifact-hygiene issues identified after Task 5C.

### 3.1 Separate generation-time provenance from current-source provenance

Task 5C modified `scripts/build_spatial_reason.py` after v0.1.1 JSONL had already been generated, then refreshed its manifest hash. This mixes:
- source actually used to generate the frozen JSONL;
- source currently present in the repository.

Use Git history and Task 5B generation commit:

`a9e5bd69cbc331f79763dd401d5c5f68d2b5f780`

to recover generation-time source bytes/hashes.

Preferred schema:

```text
generation_source:
  commit: <actual generation commit>
  file_sha256: { ... generation-time files ... }

current_source:
  commit: <current commit before Task 5.5 changes>
  file_sha256: { ... current generation-relevant files ... }
```

Requirements:
- generation-time hashes must describe code that actually produced the frozen 25,229 JSONL records;
- current-source hashes may track later maintenance code;
- generation-time hashes must never be silently refreshed;
- include `semantic_policy.py` in the generation-time set if it existed and affected generation;
- derive hashes from Git object contents when necessary;
- update consistency checks to enforce the distinction.

Do not change JSONL.

### 3.2 Remove nested `PENDING_CONSISTENCY_GATE`

The top-level quality verdict is `PASS`, but the embedded consistency result may still contain:

`quality_verdict: PENDING_CONSISTENCY_GATE`

because of execution order.

Fix final artifact generation/finalization so:
- top-level verdict is the actual final verdict;
- embedded consistency references the same final verdict;
- live consistency check remains `consistent`.

Do not regenerate JSONL.

## 4. Evidence policy

Record the actual research/access date in every research report.

Evidence priority:
1. official repo/model/dataset card;
2. official paper/publisher/project page;
3. official docs;
4. secondary sources only if primary sources do not answer the question.

For every important factual claim record:
- source title;
- exact URL;
- source type;
- accessed date;
- supported claim.

License claims must distinguish:
- code license;
- model-weight license;
- dataset license.

If unclear, mark:

`UNVERIFIED / DO NOT USE UNTIL RESOLVED`

Do not use forum comments or third-party mirrors as authoritative license evidence.

## 5. Required literature/system survey

At minimum verify these lines of work.

Foundational/general:
- LISA / LISA++
- GSVA
- PixelLM
- GLaMM or another closely related pixel-grounded MLLM if technically relevant

Pixel-level MLLM integration:
- Sa2VA
- SAMTok if relevant

Remote-sensing reasoning segmentation:
- SegEarth-R1
- EarthReason
- Think2Seg-RS
- FIRM (Fine-Grained Intra-Token Representation of Masks)
- any clearly newer/highly relevant 2026 work discovered during research

These are seed candidates, not assumed facts.

For each work extract:
- exact title/authors/venue/status;
- official repo/project/checkpoint links;
- code license;
- model-weight license;
- dataset license where relevant;
- base MLLM;
- vision encoder;
- decoder/foundation segmenter;
- special tokens or structured prompts;
- how language features become masks;
- reasoning type: textual/latent/structured/RL/decoupled;
- frozen vs trainable segmentation modules;
- image/video support;
- remote-sensing relevance;
- source-code availability/maturity;
- training complexity;
- dependency/runtime constraints;
- relevance to BuildReasonSeg;
- novelty-collision risk.

## 6. Novelty-collision audit

Current proposed causal innovation chain:
1. BuildSpatialReason geometry-verifiable spatial reasoning supervision;
2. reference-guided explicit spatial reasoning, potentially `[REF] + [SEG]`;
3. semantic–spatial–visual fusion;
4. Spatial Relation Encoder / explicit spatial prior;
5. Spatial Consistency Loss;
6. building-group spatial reasoning segmentation.

Do not assume these are novel.

Create a matrix with rows:
- geometry-verifiable spatial reasoning dataset;
- `[REF]`-style reference representation;
- `[SEG]` target token;
- reference mask -> geometry/spatial prior;
- explicit Spatial Relation Encoder;
- semantic–spatial–visual fusion;
- spatial consistency loss;
- multi-hop spatial reasoning supervision;
- building/remote-sensing specialization.

Columns should include the major works above.

Classify overlap:
- DIRECT_OVERLAP
- PARTIAL_OVERLAP
- ORTHOGONAL
- NO_EVIDENCE
- UNVERIFIED

Then state which BuildReasonSeg claims should be:
- retained;
- narrowed;
- reframed;
- postponed;
- dropped.

Do not use “first”, “novel”, “state of the art”, etc. without strong primary evidence.

## 7. Base MLLM candidate verification

Required seed candidates:
- Qwen3-VL-2B-Instruct
- Qwen3-VL-4B-Instruct
- Qwen3-VL-4B-Thinking if relevant
- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- any newer compact official multimodal model that is clearly more suitable

Also inspect existing pixel-level variants/integrations:
- Sa2VA Qwen3-VL variants
- Sa2VA Qwen2.5-VL variants
- Think2Seg-RS 3B if available
- FIRM default/released backbone

For each verify:
- exact official model ID;
- parameter count;
- weight format;
- Transformers/framework compatibility;
- minimum/known framework version if documented;
- BF16/FP16 support;
- FlashAttention requirement/optionality;
- LoRA/QLoRA evidence;
- code license;
- weight license;
- official checkpoint disk size;
- train/freeze behavior of visual tower in typical SFT;
- access to special-token hidden states;
- ease of adding `[SEG]` / `[REF]`;
- Windows-native feasibility;
- WSL2/Linux requirement or preference;
- existing SAM/SAM2 integration.

## 8. Segmentation pathway candidates

Compare:

### Route A — LISA-style direct `[SEG]`
`MLLM hidden([SEG]) -> projection MLP -> segmentation decoder -> mask`

Assess original SAM vs SAM2/SAM2.1, train/freeze choices, and future `[REF] + [SEG]` compatibility.

### Route B — Sa2VA-style MLLM + SAM2
Assess whether adapting an existing Sa2VA Qwen3-VL backbone is lower risk than building integration from scratch.

### Route C — Think2Seg-RS-style decoupled prompting
`MLLM -> structured point/box/geometric prompts -> frozen SAM2`

Assess implementation simplicity, 16 GB feasibility, value as baseline, divergence from the Challenge Cup narrative, and novelty collision.

### Route D — SegEarth-R1-style learned mask decoder
Assess practicality for first local MVP.

### Route E — FIRM/tokenized mask representation
Assess relevance for small/adjacent remote-sensing objects, engineering complexity, and whether it belongs in the final model or related work.

Do not implement any route.

## 9. `[REF] + [SEG]` feasibility

Evaluate:

```text
Image + Instruction
        ↓
      MLLM
        ↓
reasoning text + [REF] + [SEG]
        ↓
[REF] hidden -> predicted reference mask / reference representation
        ↓
inference-available geometry / spatial prior
        ↓
[SEG] hidden + spatial prior + visual features
        ↓
target mask decoder
```

Critical rule:
**Ground-truth geometry must never become an inference-time input.**

Assess three inference-valid reference/spatial-prior sources:
1. predicted `[REF]` mask;
2. predicted point/box/proposal representation;
3. learned latent spatial map without explicit reference mask.

For each assess:
- supervision;
- differentiability;
- inference availability;
- implementation complexity;
- failure modes;
- VRAM impact;
- literature overlap;
- MVP vs final-model suitability.

The MVP need not implement the full Spatial Relation Encoder yet, but must not block it later.

## 10. Hardware feasibility

User hardware:
- Windows 11 64-bit
- Intel Core Ultra 9 275HX
- 32 GB DDR5-5600
- RTX 5080 Laptop GPU, 16 GB VRAM
- 1 TB NVMe
- 250–300 GB can be freed
- preferred dataset storage < 200 GB
- formal run preferably < 7 days
- local preferred; cloud acceptable only if high value

For each serious candidate classify:
- SAFE_LOCAL
- BORDERLINE_LOCAL
- NOT_RECOMMENDED_LOCAL

Estimate separately:
1. inference VRAM;
2. LoRA training;
3. QLoRA training;
4. decoder training;
5. image-resolution impact;
6. activation/checkpointing impact.

State assumptions:
- precision/quantization;
- optimizer;
- batch size;
- gradient accumulation;
- vision tower frozen or trainable;
- SAM/SAM2 frozen or trainable;
- image token/max-pixel budget;
- context length.

Distinguish:
- parameter memory;
- optimizer memory;
- gradients;
- activations;
- CUDA/runtime overhead.

Do not present estimates as measured results.

## 11. Windows vs WSL2/Linux decision

Research whether Task 6 should use:
- native Windows;
- WSL2 Ubuntu;
- another local Linux environment;
- cloud only if truly necessary.

For each candidate stack record:
- documented OS assumptions;
- compiled CUDA ops;
- DeepSpeed dependence;
- FlashAttention dependence;
- SAM2 build-extension behavior;
- native Windows blockers.

Prefer the least disruptive technically reliable environment.

Do not install anything in Task 5.5.

## 12. External dataset survey

BuildSpatialReason-v0.1.1 remains primary.

Survey external datasets only for:
- pretraining/warm-up;
- comparison;
- robustness;
- domain transfer;
- paper baselines.

At minimum:
- EarthReason
- ReasonSeg / LISA++ data where available
- remote-sensing datasets used by SegEarth-R1, Think2Seg-RS, FIRM, Sa2VA
- LaSeRS
- DRSeg
- RISBench
- RRSIS-D if relevant

For each verify:
- official source;
- size;
- annotation type;
- reasoning vs referring;
- image domain;
- mask availability;
- splits;
- license;
- approximate storage;
- whether legal/technical use is appropriate.

If license unclear, do not recommend merging it into training.

Do not download datasets.

## 13. Evaluation protocol planning

Propose paper-ready evaluation.

Segmentation:
- mIoU / gIoU / cIoU where appropriate;
- Dice/F1 if useful;
- boundary metric only if justified.

Reasoning:
- target selection accuracy;
- relation satisfaction rate;
- reference selection accuracy;
- multi-hop success;
- nontrivial Level-3 primary metric;
- trivial Level-3 separate reporting.

Spatial consistency:
- direction consistency;
- size consistency;
- nearest consistency.

Baselines:
- historical YOLOv8m-seg-WHU;
- non-reasoning segmentation baseline;
- one LISA/Sa2VA-style reasoning baseline if feasible;
- one remote-sensing reasoning-segmentation comparison if reproducible.

Do not compare metrics across different datasets as if directly equivalent.

## 14. Final decision matrix

Include at least:
- Qwen3-VL-2B + SAM2/SAM2.1
- Qwen3-VL-4B + SAM2/SAM2.1
- Sa2VA-Qwen3-VL-2B adaptation
- Sa2VA-Qwen3-VL-4B adaptation
- Think2Seg-RS-style decoupled route
- another candidate if research strongly supports it

Compare on:
- 16 GB feasibility;
- implementation effort;
- Windows/WSL complexity;
- license clarity;
- code maturity;
- special-token flexibility;
- `[REF] + [SEG]` compatibility;
- remote-sensing relevance;
- inference speed;
- paper extensibility;
- novelty collision risk.

Do not choose via naive numeric total alone.

Output:
1. Primary MVP stack
2. Fallback stack
3. Baseline-only stack
4. Not recommended now

## 15. Exact decisions required

Answer these exact questions:
1. What exact MLLM/checkpoint should Task 6 start from?
2. What exact segmentation model/decoder should Task 6 use?
3. Should MVP use `[SEG]` only, `[REF] + [SEG]`, or structured prompts?
4. Which modules are frozen vs trainable?
5. LoRA, QLoRA, or another PEFT method?
6. First image resolution/max-pixel policy?
7. Local OS/environment?
8. Smallest staged experiment proving the pipeline works?
9. Any external dataset before/after BuildSpatialReason?
10. What path stays open for final Spatial Relation Encoder and Spatial Consistency Loss?
11. What literature result most threatens current novelty?
12. How should the Challenge Cup technical narrative change?

The result must be concrete enough for Task 6 implementation without repeating this survey.

## 16. Task 6 staged plan — design only

Prepare, but do not execute:

```text
Stage 0: environment + model load smoke test
Stage 1: 2-sample forward/backward
Stage 2: overfit 20 samples
Stage 3: 200–500 sample mini-train
Stage 4: full train
Stage 5: validation + demo
```

For each specify:
- runtime scale;
- success criteria;
- failure criteria;
- checkpoint policy;
- VRAM safety;
- logging.

Future training code must support:
- best/last checkpoints;
- periodic saves;
- resume;
- mixed precision;
- gradient accumulation;
- gradient clipping;
- NaN detection;
- OOM recovery guidance;
- validation;
- early stopping where appropriate.

No implementation now.

## 17. Deliverables

Create at least:

```text
docs/research/task5_5_sources.md
docs/research/task5_5_literature_matrix.md
docs/research/task5_5_model_stack.md
docs/research/task5_5_hardware_feasibility.md
docs/research/task5_5_novelty_collision.md
docs/research/task5_5_mvp_decision.md
evaluation/task5_5_stack_decision.json
```

Optional:

```text
docs/research/task5_5_external_datasets.md
docs/research/task5_5_evaluation_plan.md
```

Update:
- `docs/architecture_decisions.md`
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`
- `README.md`

Only freeze a new ADR if evidence is strong enough. If critical evidence remains unresolved, mark the ADR `PROVISIONAL`.

## 18. Machine-readable stack decision

`evaluation/task5_5_stack_decision.json` should include:
- research date;
- primary stack;
- fallback stack;
- baseline-only stack(s);
- MLLM model IDs;
- decoder/model IDs;
- licenses + verification status;
- environment recommendation;
- precision/quantization recommendation;
- trainable/frozen modules;
- first image/max-pixel policy;
- external datasets approved/rejected/unverified;
- key novelty-overlap flags;
- unresolved risks;
- Task 6 entry criteria;
- source URLs.

Label estimates as estimates.

## 19. Source-quality checks

Before finalizing:
- verify every URL reachable at research time;
- remove duplicate sources;
- prefer canonical GitHub/HF/arXiv links;
- ensure every license claim has a source;
- distinguish paper date from repo update date;
- distinguish code release from weight release;
- distinguish remote-sensing reasoning segmentation from ordinary referring segmentation;
- explicitly verify the official Think2Seg-RS source;
- explicitly inspect current 2026 work, not only 2023–2024 baselines.

## 20. No implementation/downloads

Do not:
- add model integration code;
- create training scripts;
- install environments;
- download weights;
- download datasets;
- test inference;
- allocate cloud GPUs.

Small repository scripts for provenance/consistency housekeeping are allowed.

## 21. Git commit/push

ChatGPT will review the research package directly from GitHub.

Before staging:
1. `git status --short`
2. inspect all changes
3. ensure no weights, dataset archives, external repos, caches, or large files are staged
4. verify frozen JSONL hashes unchanged
5. verify no unauthorized out-of-repo writes

Recommended commit:

`research: select BuildReasonSeg MVP stack`

Then:

`git push origin main`

If push fails, record the blocker.

## 22. Handoff

Write `handoff/FROM_DSH.md` (English allowed).

Suggested sections:
1. Verdict / Research Confidence
2. Preflight Provenance Fix
3. Sources Reviewed
4. Literature Landscape
5. Novelty Collision
6. Base MLLM Comparison
7. Segmentation Pathways
8. `[REF] + [SEG]` Feasibility
9. Hardware Feasibility
10. Windows vs WSL2/Linux
11. External Dataset Decisions
12. Evaluation Plan
13. Primary MVP Stack
14. Fallback / Baselines
15. Task 6 Staged Plan
16. Remaining Risks
17. Files Created / Modified
18. Git Commit / Push
19. Ready for Task 6?

Update `handoff/PROJECT_STATE.md` concisely.

## 23. Final DSH web/chat response — Chinese only

The visible final response must concisely report:
- Task 5.5 complete/incomplete;
- selected primary MVP stack;
- selected fallback stack;
- recommended execution environment;
- whether 16 GB local training appears feasible;
- largest unresolved risk;
- commit hash;
- push success/failure.

Do not paste the full research report into the UI.

## 24. Stop

After Task 5.5:

STOP.

Do not begin Task 6 implementation, environment installation, model download, dataset download, or training.

Wait for ChatGPT to review the pushed research package.
