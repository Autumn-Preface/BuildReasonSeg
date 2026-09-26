# Task 6B — Network cleanup + 2B `[SEG]` real mini-train and first generalization audit

**Date:** 2026-09-26
**Actor:** DSH
**Task:** Task 6B (`handoff/TO_DSH.md`)
**Verdict:** `FAIL_REQUIRES_DEBUG` — language pathway solved and `[SEG]` emission generalises, but the
mask pathway does **not** generalise: the paired instruction-dependence probe scores **0/20** because the
projection collapses to a near-constant prompt.

Task 6B had to answer four questions. The answers are:

| # | Question | Answer |
|---|---|---|
| 1 | Does the project work with a normal/direct TLS path after the local accelerator was stopped? | **Partly.** The Watt-specific workaround is gone and nothing insecure is used, but `huggingface.co` does not resolve on this network at all, so Task 6B runs fully offline from the cache. GitHub HTTPS works; git-smart-HTTP is intermittent. See §1. |
| 2 | Can the 2B architecture generalise beyond the 20-sample overfit? | **Only for the language pathway.** Format, reasoning text and `[SEG]` emission generalise perfectly to unseen images (120/120). Mask quality does not: strict end-to-end val mIoU is **0.1102**, against 0.98 on the 20-sample overfit. |
| 3 | Can it learn `reasoning_zh` and emit exactly one `[SEG]` on unseen validation images? | **Yes**, but the task is easier than it looks, because the dataset's `reasoning_zh` is template-generated: the whole 480-record mini-train contains only **21 distinct reasoning strings**. The model reproduces the reasoning exactly on 120/120 unseen records. |
| 4 | First honest end-to-end validation performance by L1 / L2 / nontrivial L3? | L1 **0.1446**, L2 **0.1082**, nontrivial L3 **0.0778** strict end-to-end mIoU at 100 % emission in every stratum. |

The decisive measurement is not the mIoU number but the paired probe and the prompt diagnosis in §10:
two different instructions on the same image produce masks with IoU **0.9999**, because the 256-d sparse
prompt the projection emits has cosine similarity **0.999995** between them (and **1.000000** between
*different images*). The mask decoder is being handed essentially the same prompt every time, so it
cannot possibly answer the instruction.

---

## 1. Network / TLS cleanup (Part A)

Full record: `evaluation/task6b_network_cleanup.json`.

| Check | Result |
|---|---|
| hosts entries for `github.com` / `raw.githubusercontent.com` / `huggingface.co` / `github.io` | **none** (`workaround_present: false`, 0 blackhole lines, 0 accelerator markers) |
| listener on 443 or on the accelerator's proxy ports | **none** (`any_previous_proxy_port_open: false`) |
| certifi bundle | restored to the standard 121-certificate bundle; **no accelerator root**; sha256 `9cc2a774b5198dcff14d9be1e66091f538975d867ce029a96bce15a55dfd730f` |
| automatic trust mutation in project code | removed (`buildreasonseg_mvp/local_env.py` no longer edits certifi on import; the legacy helper is opt-in and a no-op without `confirm=True`) |
| insecure TLS flags anywhere | **none** (`insecure_flags_used: false`) |
| direct HTTPS `api.github.com`, `raw.githubusercontent.com`, `pypi.org` | HTTP 200 with standard certificate validation (both `urllib` and `httpx`) |
| direct HTTPS `huggingface.co` | **unreachable** — `[Errno 11001] getaddrinfo failed`; the host does not resolve on this network |
| git smart-HTTP to GitHub | **intermittent** |
| cached assets usable offline | **yes** — processor + SAM2 load offline in ≈30 s |

### The `github.com` address problem, measured rather than guessed

`github.com` resolves to `20.205.243.166`, and a plain TCP connect to that address on port 443
black-holes (~21 s timeout) while other GitHub edge addresses answer immediately
(`140.82.113.4` 284 ms, `20.205.243.168` 90 ms, `185.199.108.133` 76 ms). Five consecutive
`git ls-remote` attempts failed, and minutes later the same command succeeded with no change to the
machine. This is a network-level property of the current connection, **not** a leftover of the
accelerator: the hosts file is clean, no proxy is listening, and no certificate is being injected.

The recorded fallback is per-command only:

```
git -c http.curloptResolve=github.com:443:140.82.113.4 <command>
```

It changes which address the connection is opened to and nothing else: TLS still verifies the real
`github.com` hostname against the standard bundle (`git_resolve_override_persisted: false`). Nothing
was written to the hosts file, the git config, the certificate store, the registry, `PATH`, drivers or
the system proxy.

### Hugging Face

`huggingface.co` does not resolve here at all, so no Hub request is attempted and the failure mode
Task 6A saw (`X-Repo-Commit`, `LocalEntryNotFoundError` from the accelerator's TLS proxy) is simply
not reproducible. Task 6B therefore runs with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` against the
already-cached Qwen3-VL-2B snapshot (`89644892e4d85e24eaac8bacfd4f463576704203`) and SAM2.1 Base+
weights. The four questions Task 6B asks can all be answered offline; downloading new weights could not.

## 2. Environment integrity

Conda only, as required: `.conda/buildreasonseg-mvp` created with `conda create --prefix`, Python 3.11.16,
torch 2.13.0+cu132, transformers 5.17.0, peft 0.21.0, SAM2 1.0 at revision
`2b90b9f5ceec907a1c18123530e92e794ad901a4`, scipy 1.17.1. `.conda/` is gitignored. `base` and
`yolo_sam_env` were never touched; no `venv` was created.

## 3. Frozen architecture

Unchanged from ADR-013 and verified trainable-set-by-trainable-set:

| Component | Task 6B state |
|---|---|
| Qwen3-VL-2B-Instruct BF16 | base LLM frozen; **vision tower frozen in both phases** (`qwen_visual_trainable: []`) |
| text LoRA r=16, α=32 | trainable, 196 modules, 17,432,576 parameters |
| `[SEG]` row (id 151669) | trainable, one tied row shared by input embedding and `lm_head`, 2,048 parameters |
| Projection MLP | **frozen in Phase A**, trainable in Phase B |
| SAM2.1 Base+ image/prompt encoders and memory | frozen |
| SAM2 mask decoder | **frozen in Phase A**, trainable in Phase B |
| total trainable | **24,010,309 / 2,227,632,642 = 1.078 %** |

No `[REF]`. No Qwen3-VL-4B. No Spatial Relation Encoder or Spatial Consistency Loss. No external data.

## 4. Train / validation subsets (frozen before any training)

Full record: `evaluation/task6b_subset_ids.json`. Selected by round-robin over per-level buckets ordered
by `sample_id`, so the choice does not depend on file order or on any model behaviour. The test split is
**never** touched (`test_split_used: false`).

| Subset | Size | Composition | Unique images | Image reuse |
|---|---|---|---|---|
| train | 480 | 160 L1 + 160 L2 + 160 nontrivial L3 | 480 | 0 |
| val | 120 | 40 L1 + 40 L2 + 40 nontrivial L3 | 120 | 0 |
| paired probe | 40 records | 20 val images × 2 instructions, different targets and (20/20) different query types | 20 | — |

No trivial L3 anywhere; train and val records come from disjoint splits (`train` / `val`), verified by
test.

## 5. Fresh pre-training baseline

Measured on the same 120 val records before Phase A, from the clean ADR-013 architecture (no Task 6A
checkpoint is ever used as an initialisation):

| Metric | Baseline |
|---|---|
| free-generation valid `[SEG]` emission | **0/120 = 0.000** |
| strict end-to-end mIoU / Dice | **0.000 / 0.000** |
| conditional mIoU | n/a (no valid emission) |
| operation-chain accuracy | 0.000 |
| teacher-forced mIoU / Dice | 0.00626 / 0.01004 |
| teacher-forced LM CE / perplexity | 3.4576 / 31.74 |
| teacher-forced reasoning-token accuracy | 0.4522 |
| teacher-forced `[SEG]` token accuracy | **0.000** |

The untrained model never emits `[SEG]` at all, and even when teacher-forced it never puts probability
on `[SEG]`. Its mask is whatever a random projection produces from an untrained `[SEG]` hidden state.

## 6. Phase A — language-format warm-up

Text LoRA + `[SEG]` only; projection MLP and SAM2 mask decoder frozen; LM CE only; 1 epoch = 480 steps;
no SAM2 work at all in this phase.

| | Baseline | After Phase A |
|---|---|---|
| training LM CE | — | 0.0003 at step 40 → ~0.0000 by step 120 |
| val LM CE | 3.4576 | **0.2299** |
| val reasoning-token accuracy | 0.4522 | high |
| valid `[SEG]` emission | 0.000 | **1.000 (120/120)** |
| operation-chain accuracy | 0.000 | 0.667 |
| strict end-to-end mIoU | 0.000 | 0.00708 |

Phase A already solves the format and most of the language: the model goes from never emitting `[SEG]`
to emitting exactly one on every unseen record. The mask stays at chance because both the projection and
the mask decoder are still frozen, which is what Phase A is designed to show.

## 7. Phase B — joint segmentation training

Continue from Phase A; train text LoRA, `[SEG]`, projection MLP and SAM2 mask decoder; base and vision
encoders stay frozen. Pre-declared section 12 objective
`L = 2.0·L_lm_ce + 2.0·L_mask_bce + 1.0·L_mask_dice`, 5 epochs / 2400-step cap, early-stop patience 2,
supervision at the original 512×512 resolution. The `best_joint` selection rule was declared before
Phase B and never changed:

```
1. valid [SEG] emission rate   2. strict end-to-end val mIoU
3. operation-chain accuracy    4. LM CE
```

### Headline run, epoch by epoch

| Epoch | Phase-B steps | valid `[SEG]` | strict e2e mIoU | chain acc. | val LM CE |
|---|---|---|---|---|---|
| 1 | 480 | 1.000 | 0.0932 | 0.667 | 0.2101 |
| 2 | 960 | 1.000 | 0.1101 | 0.867 | 0.0148 |
| **3** | **1440** | **1.000** | **0.1102** | **1.000** | 0.0020 |
| 4 | 1920 | 1.000 | 0.0991 | 1.000 | 0.000003 |
| 5 | 2400 | 1.000 | 0.0992 | 1.000 | 0.000002 |

Early stopping fired after two epochs without `best_joint` improvement; the best checkpoint is at
**step 1440 (epoch 3)**. No NaN, no Inf, no OOM, no collapsed prediction in any epoch.

The training mask loss never descended: `mask_dice` stayed in 0.65–1.00 and `mask_bce` in 0.01–0.05 for
all 1920 steps. That signature — very low BCE with near-maximal Dice on small targets — is what an
almost-all-background prediction costs, and it is the first evidence that the mask branch is stuck.

### The one permitted bounded recipe adjustment, and its outcome

Section 12 allows exactly one bounded adjustment after the first complete documented epoch if one
objective clearly fails. It did: language solved (LM CE 0.00041, emission 120/120, chain 1.00) while
segmentation stalled. The adjustment was recorded and made once:

| | original (headline) | adjusted |
|---|---|---|
| loss weights | `lm_ce 2.0 / mask_bce 2.0 / mask_dice 1.0` | `lm_ce 1.5 / mask_bce 4.0 / mask_dice 3.0` (the Task 6A mask-dominant weighting) |
| Phase-B `decoder_lr` | 3e-4 | 1e-3 |

Everything else was identical. Result: **the adjustment did not help.**

| Metric | original recipe | adjusted recipe |
|---|---|---|
| strict end-to-end val mIoU | **0.11018** | **0.09498** |
| conditional mIoU | 0.11018 | 0.09498 |
| valid `[SEG]` emission | 1.000 | 1.000 |
| operation-chain accuracy | 1.000 | 1.000 |
| val LM CE | 0.00200 | 5.1e-06 |
| paired instruction dependence | **0/20** | **0/20** |
| best epoch / step | 3 / 1440 | 3 / 1440 |
| Phase-B epochs run | 5 of 5 | 5 of 5 |

Per-epoch adjusted mIoU: 0.0903, 0.0948, 0.0950, 0.0874, 0.0776 — flat, then falling. The mask Dice
loss stayed in the same 0.65–1.00 band throughout. So the failure is **not** explained by loss weighting
or by the mask-pathway learning rate, which is why the diagnosis in §10 matters.

Both runs are on record: the headline (original) recipe's artifacts are the canonical
`evaluation/task6b_*.json`, the adjustment is kept in full as
`evaluation/task6b_*_adjusted_recipe.json`, and the first execution of the headline recipe is
re-measured with the corrected metrics in `evaluation/task6b_revalidation_original_recipe.json`.

## 8. Teacher-forced validation

Given the expected `reasoning_zh + [SEG]`, on the 120 unseen val records, headline checkpoint:

| Metric | Baseline | Headline |
|---|---|---|
| mIoU / Dice | 0.00626 / 0.01004 | **0.11018 / 0.17874** |
| LM CE / perplexity | 3.4576 / 31.74 | **0.00200 / 1.0020** |
| assistant-token accuracy | 0.4405 | **1.000** |
| reasoning-token accuracy | 0.4522 | **1.000** |
| `[SEG]` token accuracy | 0.000 | **1.000** |
| greedy-decoded reasoning exact match | 0.000 | **1.000** |
| greedy-decoded operation-chain accuracy | 0.000 | **1.000** |

The teacher-forced mIoU equals the free-generation mIoU **exactly** (0.11017863780842695). That is a
result, not a coincidence: with a perfectly reproduced prefix, the mask is identical to the
free-generation mask, which says the mask error has nothing to do with generation quality.

## 9. Free-generation end-to-end validation (headline metric)

Input is only the image and the instruction. The generated sequence is re-forwarded, the generated
`[SEG]` is located, and its hidden state drives the mask. Zero or multiple `[SEG]` would score
IoU = 0, Dice = 0.

| Metric | Baseline | Headline |
|---|---|---|
| valid `[SEG]` emission | 0/120 (0.000) | **120/120 (1.000)** |
| **strict end-to-end mIoU** | 0.000 | **0.11018** |
| strict end-to-end Dice | 0.000 | 0.17874 |
| conditional mIoU / Dice (among valid) | n/a | 0.11018 / 0.17874 |
| collapsed samples | 0 | 0 |
| generated reasoning exact match | 0.000 | **1.000** |
| generated reasoning char similarity | 0.2464 | **1.000** |
| generated operation-chain accuracy | 0.000 | **1.000** |
| mean generated tokens | 53.3 | 35.5 |

The improvement against the baseline is **+0.11018 absolute**, which clears section 17's +0.10 bar by
0.010. Conditional mIoU equals strict mIoU because emission is 120/120: there is no format failure to
hide behind, and none to blame either.

### Breakdown by level and by query family

| Stratum | strict e2e mIoU | emission | conditional mIoU |
|---|---|---|---|
| L1 (single-object extreme/size) | **0.1446** | 1.000 | 0.1446 |
| L2 (two-hop) | 0.1082 | 1.000 | 0.1082 |
| L3 nontrivial (trivial-selection excluded) | **0.0778** | 1.000 | 0.0778 |

| Family | strict e2e mIoU |
|---|---|
| `extreme` (leftmost/rightmost/topmost/bottommost) | **0.1496** |
| `size` (largest/smallest) | 0.1341 |
| `direction` (`largest|smallest` `_to_` direction) | 0.0878 |
| `nearest` (`…_to_nearest`) | **0.1898** |
| `multi_hop_direction_to_nearest` | **0.0778** |

Difficulty increases with the number of hops, as expected. Every stratum is between 0.08 and 0.19, i.e.
the model is never close to a usable mask anywhere.

## 10. The paired instruction probe — the result that decides the verdict

20 unseen val images, each with two instructions that have **different targets** and a different query
type. A pair passes only if each prediction overlaps its **own** ground truth more than the other
target's.

| | Baseline | Headline | Adjusted |
|---|---|---|---|
| pairs passed | — | **0/20** | **0/20** |
| mean own-target IoU | — | 0.13885 | 0.10489 |
| mean cross-target IoU | — | 0.13873 | 0.10490 |
| pairs with both emissions valid | — | 20/20 | 20/20 |

Own-target and cross-target IoU agree to four decimals. The model is not choosing the *wrong* object —
it is producing nearly the *same* mask for both instructions. Examples from the recorded probe: on image
`1_2103` both instructions give IoU `0.5297312140464783` against target A, identical to 16 digits.

### Prompt-pathway diagnosis — where the collapse is

`evaluation/task6b_prompt_diagnosis_original_recipe.json` and
`evaluation/task6b_prompt_diagnosis_adjusted_recipe.json` measure the signal at each stage of
`[SEG] hidden → Projection MLP → 256-d sparse prompt → SAM2 mask decoder`, teacher-forced so that
generation quality cannot confound the result, with the same measurement repeated across **different
images** for scale:

| Stage | headline: same image, different instruction | headline: different images | adjusted: same image / different images |
|---|---|---|---|
| `[SEG]` hidden cosine (2,048-d) | 0.9656 | 0.9975 | 0.9952 / 0.9996 |
| projected prompt cosine (256-d) | **0.999952** | **0.999997** | **0.999995** / **1.000000** |
| decoded mask IoU (a vs b) | **0.9970** | — | **0.9999** |

Reading, stage by stage:

1. **`[SEG]` hidden state**: two different instructions on the same image are almost as similar to each
   other (0.9656) as two completely different images are (0.9975). The `[SEG]` position carries very
   little instruction-specific or even image-specific information — it is dominated by "the assistant
   turn ends here".
2. **Projection MLP**: the 256-d sparse prompt is a **near-constant vector**. Its direction differs by
   5e-5 between two different instructions and by 3e-6 between two different images. This is a collapsed
   projection.
3. **Mask decoder**: consequently the decoded masks differ by IoU 0.9970. The decoder is fine; it is
   being asked to segment the same thing every time.

So the mask pathway fails at the *interface*, before the mask decoder: the model never learns to place a
referent-specific point. The failure is upstream of the mask decoder even though the mask decoder has
the most trainable capacity.

Gradient evidence from the first Phase-B step points the same way: gradient norms were
`sam_mask_decoder 114.3`, `projection 13.2`, `seg_token 1.57`, `lora 1.04`, and the total (115.1) was
clipped to 1.0. The mask decoder therefore consumes ≈99 % of the clipped gradient budget while the
projection — the only place a referent-specific prompt can come from — receives ≈11 % of it, and the
LoRA adapters (the only place instruction routing can be learned) receive under 1 %.

## 11. Language metrics, and the templating that limits what they mean

Section 11 requires language metrics to be first-class; they are, in
`evaluation/task6b_validation.json` and per record in the same file. But the honest caveat matters more
than the numbers:

| | count | distinct `reasoning_zh` | distinct `template_id` |
|---|---|---|---|
| train mini-set (480) | 480 | **21** | 39 |
| val mini-set (120) | 120 | **20** | 37 |

and all 20 val reasoning strings also occur in the training mini-set. L1 has 6 distinct reasoning
strings for 160 records; L3 has 4 for 160. So "the model learned the reasoning" means "the model maps
each query type onto one of a handful of fixed Chinese sentences", which is also why training LM CE
collapses to ~0 within 40 optimizer steps. This is a property of the dataset, not of the model, and it
means:

* reasoning-token accuracy and reasoning exact match are **not** evidence of reasoning;
* operation-chain accuracy is the only language metric here that carries information, because it checks
  the *chain* rather than the wording;
* the interesting comparison is the mask, not the text.

## 12. Cost

Headline run on the RTX 5080 Laptop (15.894 GiB):

| | Value |
|---|---|
| Phase A | 480 steps, 218.9 s |
| Phase B | 5 epochs × 480 steps = 2400 steps; 423.6 s for epoch 1 (feature-cache warm-up), then 311.8–317.3 s per epoch |
| optimisation time (Phase A + Phase B) | 1898.5 s ≈ 31.6 min |
| teacher-forced validation (120 records) | 18.5 s per pass |
| free-generation validation (120 records) | 375.0 s per pass |
| peak VRAM, Phase A | 6.06 GiB allocated / 7.10 GiB reserved |
| peak VRAM, Phase B | 6.17 GiB allocated / 7.13 GiB reserved |
| overall peak | 5.37 GiB allocated / 7.13 GiB reserved of 15.894 GiB |
| SAM2 feature cache | LRU, CPU-resident, 160 images ≈ 1.7 GB — inside the section 18 cap of 8 GB |

The `feature_cache_images: 160` setting is smaller than the 480 training images, so the LRU thrashes and
features are recomputed for most records each epoch. That costs wall-clock (not correctness) and is worth
raising in Task 6C.

## 13. Defects found and fixed during Task 6B

Recorded because each one changed a number that had already been reported.

1. **Teacher-forced token accuracy was indexed one position early.** `teacher_forced_validation` scored
   `lm_logits[i - 1]` against `labels[i]`, but the labels are already shifted (`labels[i]` holds token
   `i + 1`) and `lm_logits[i]` is the distribution over token `i + 1`. Reported accuracy was therefore
   ≈3 % while the same records had LM CE 0.00041 (perplexity 1.0004) — a contradiction that exposed the
   bug. Fixed; the affected numbers were re-measured on the already-trained checkpoint with
   `scripts/task6b_revalidate.py` instead of being silently restated. Baseline reasoning-token accuracy
   went 0.4522 (was reported as 0.0160) and the trained value is 1.000.
2. **The teacher-forced path reported language quality as zero by construction.** `language_summary`
   hard-coded `reasoning_exact_match: False`, `operation_chain_correct: False` and
   `reasoning_char_similarity: 0.0` for teacher-forced records, which is why the first report showed
   `reasoning_exact_match 0.0` next to `generation_reasoning_exact_match 1.0`. The teacher-forced path
   now greedy-decodes its own distribution and measures the text it would have emitted.
3. **`optimizer.*.token_lr` was silently unused.** `trainable_parameter_groups` gave the `[SEG]` row
   `decoder_lr`. In the headline recipe both are 3e-4, so the headline result is unaffected; in the
   adjusted recipe the `[SEG]` row therefore trained at 1e-3 rather than the configured 3e-4. Fixed by
   adding an explicit `token_lr` argument (defaulting to `decoder_lr`, so existing callers are unchanged);
   the runs were not re-executed, because the headline recipe is unaffected and section 12 forbids
   further recipe changes.
4. **`scripts/task6b_train.py` uses a module-level `CHECKPOINT_DIR`, not `paths.checkpoints` from the
   config.** The adjusted run therefore overwrote the headline run's checkpoints in
   `artifacts/checkpoints/task6b/`. The headline recipe was re-run deterministically to regenerate a
   checkpoint whose bytes match the recorded manifest; the reproducibility check is in
   `task6b_training_report.json → recipe_adjustment.reproducibility_of_the_headline_recipe`.
5. **A test asserted the wrong label convention** (`labels[-2] == seg and labels[-1] == eos`).
   Because the labels are pre-shifted, the last position is `-100`; the test now checks
   `labels[-3] == seg`, `labels[-2] == eos`, `labels[-1] == -100`.
6. **`training.deterministic: true` is a dead flag — recorded, deliberately not changed.**
   `buildreasonseg_mvp/runtime.py` calls `set_seed` and never `torch.use_deterministic_algorithms` or
   `torch.backends.cudnn.deterministic`, so sampled CUDA kernels in the SAM2 path are not reproducible
   step-for-step. Re-running the identical headline recipe reproduced the fresh baseline and Phase A
   **bit-identically** (same LM CE 0.22990993372364604, same mIoU 0.007083570592059386) but not Phase B:
   epoch-1 mIoU was 0.1008 in the first execution and 0.0932 in the re-run, and the best-epoch mIoU was
   0.11018644 versus 0.11017864. Enforcing determinism would change the numerics of an already-recorded
   recipe, so it was left alone and is recommended as the first change in Task 6C. The practical
   consequence is that the section 17 +0.10 bar is **marginal**: it is met by both executions
   (+0.11019 and +0.11018), but the epoch-to-epoch spread is larger than the margin.

## 14. Verdict against the section 17 gate

| # | Minimum for `PASS` | Result |
|---|---|---|
| 1 | network cleanup has no insecure workaround | **met** — no hosts edit, no cert-store edit, no proxy, no insecure flag; TLS verifies normally |
| 2 | no NaN/Inf/OOM | **met** — clean in every epoch of both runs |
| 3 | val valid `[SEG]` emission ≥ 90 % | **met** — 120/120 = 100 % |
| 4 | strict e2e val mIoU improves ≥ 0.10 absolute over fresh baseline | **met by the headline recipe** 0.000 → 0.11018 (+0.11018); **not met by the adjusted recipe** (0.09498) |
| 5 | teacher-forced val mIoU meaningfully above baseline | **met** — 0.00626 → 0.11019 (17.6×) |
| 6 | reasoning-token accuracy improves substantially | **met** — 0.4522 → 1.000 |
| 7 | operation-chain accuracy ≥ 70 % | **met** — 1.000 (120/120) |
| 8 | paired instruction-dependence ≥ 14/20 | **NOT MET — 0/20** |
| 9 | no GT geometry leakage | **met** — ground truth is used only for scoring and for drawing panels |
| 10 | no test-set tuning | **met** — the test split was never read |

Nine of ten minimums are met and criterion 8 fails as badly as it can. Section 17 says to use
`FAIL_REQUIRES_DEBUG` when "pair generalization fails badly", and that is the case here, so the verdict
is **`FAIL_REQUIRES_DEBUG`** rather than `PASS_WITH_WARNINGS`. The warnings branch is for weak language
quality; here the language is solved and the segmentation is what fails. Calling this a pass because
nine boxes are ticked would hide the only result that matters.

## 15. What Task 6B establishes, and what it does not

**Establishes**

* The Task 6A pipeline works after the local accelerator is removed, with no insecure workaround, and
  runs fully offline from cache.
* The format and language half of the task generalises to unseen images: 120/120 valid `[SEG]`,
  1.000 reasoning exact match and 1.000 operation-chain accuracy on records the model never saw.
* The segmentation half does **not** generalise at this scale: strict end-to-end val mIoU 0.1102 with
  the mask essentially independent of the instruction.
* The mechanism of that failure is measured, not guessed: the `[SEG]` hidden state carries little
  referent information and the projection emits a near-constant 256-d prompt (§10).
* Neither mask-dominant loss weighting nor a 3.3× mask-pathway learning rate fixes it.

**Does not establish**

* That 2B cannot do the task — only that this recipe, at this budget, converges to a collapsed prompt.
  A 4B scale-up is not justified by this result as a *fix* for the collapse; the collapse is at the
  `[SEG]`-to-prompt interface and would plausibly reproduce.
* Any language *reasoning* claim: the `reasoning_zh` target has 21 distinct values in the whole
  training mini-set.
* Anything about the test split, `[REF]`, or the Spatial Relation Encoder.

## 16. Recommendation for Task 6C

The measured failure is a **prompt-interface** failure, so the next step should target it rather than
model scale:

1. **Stop the projection collapsing.** Nothing in the objective rewards a prompt that varies with the
   instruction. Candidates: normalise or whiten the 256-d prompt; supervise the prompt against the
   target's interior (a point-inside-the-mask auxiliary loss, which uses only ground-truth geometry that
   is already available as supervision); or give SAM2 more than one prompt channel.
2. **Fix the gradient budget.** The mask decoder takes ≈99 % of the clipped gradient while the LoRA
   adapters take under 1 %. Per-group clipping, or a lower decoder LR with a higher LoRA LR, is worth
   measuring — this is a *different* intervention from the section 12 adjustment that was tried, which
   changed only the global weights and the decoder LR.
3. **Raise `feature_cache_images` to 480** so the LRU stops thrashing; it is a pure wall-clock cost.
4. **Do not read the language metrics as reasoning.** If a reasoning claim is wanted, the dataset's
   template collapse (21 distinct strings / 480 records) has to be addressed first.
5. The paired probe should stay the *primary* gate. It is the only metric that detected the real failure;
   mIoU alone looked like a modest success.

A 4B run is defensible only as a scale measurement once the prompt interface is fixed, not as the fix.

## 17. Reproduce

```bash
# Part A
python scripts/task6b_network_cleanup.py
# subsets (frozen before training)
python scripts/task6b_select_subsets.py
# headline 2B mini-train (~30 min optimisation on an RTX 5080 Laptop)
python scripts/task6b_train.py --config configs/mvp/task6b_2b_minitrain.yaml
# the single permitted bounded adjustment
python scripts/task6b_train.py --config configs/mvp/task6b_2b_minitrain_adjusted.yaml
# re-measure an existing checkpoint with the corrected metrics
python scripts/task6b_revalidate.py --checkpoint artifacts/checkpoints/task6b/best_joint.pt --tag original_recipe
# prompt-pathway diagnosis
python scripts/task6b_diagnose_prompt.py --config configs/mvp/task6b_2b_minitrain.yaml
# merge both runs into the final artifact set
python scripts/task6b_finalize_report.py
# visualization pack
python scripts/task6b_visualize.py --config configs/mvp/task6b_2b_minitrain.yaml
```

Artifacts: `evaluation/task6b_network_cleanup.json`, `evaluation/task6b_subset_ids.json`,
`evaluation/task6b_baseline.json`, `evaluation/task6b_training_report.json`,
`evaluation/task6b_validation.json`, `evaluation/task6b_checkpoint_manifest.json`,
`evaluation/task6b_prompt_diagnosis_original_recipe.json`, `evaluation/task6b_samples/`.
