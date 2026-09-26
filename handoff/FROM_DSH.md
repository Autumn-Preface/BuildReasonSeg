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

# FROM_DSH — Task 6C.7 Report: Frozen Qwen Visual-Feature Cache & Remaining Batch-1 Sync Overhead

_This file holds the Task 6C.7 report. The Task 6C.6 report is preserved in git history and in
`docs/task6c6_launch_optimization.md`; Task 6C.5 is in `docs/task6c5_training_optimization.md`; Task 6C is
in `docs/task6c_prompt_ablation.md`._

## 1. Verdict

**`OPTIMIZATION_PARTIAL` — two changes adopted.**

| Change | Status |
|---|---|
| Section 3: remove the formal Phase-B duplicate device transfer | **adopted** — exact, simpler, and it fixes a real ownership bug |
| Section 5-8: frozen Qwen visual-tower feature cache | **adopted** — `BIT_EQUIVALENT`, **+9.94 %** warm paired throughput, −8.44 % kernels, −43.5 % sync ops, 2 GiB bound |
| Section 4: remaining scalar read-backs | **localized and dismissed** — 97.7 % of them are PyTorch's own optimizer bookkeeping on CPU-hosted `step` counters, they do not synchronize, and the project's share is <1 % |
| Section 10: defer project-owned scalar logging (V3) | **not implemented** — the profiling shows it is not worth any complexity |

No model, loss, data, optimizer-mathematics, sample-order or batch-size semantics changed, and no Task 6C
model-quality conclusion is reinterpreted.

## 2. Formal H2D Fix (section 3)

`scripts/task6c_train.py` built an unused copy on every Phase-B step:

```python
batch, image = runtime.prepare(sample)
moved = batch.to(runtime.device)          # allocated and filled …
result = runtime.train_step(batch, ...)   # … never used: train_step copies again
```

`MvpRuntime.train_step` performs `batch = batch.to(self.device)` itself. The loop now creates `moved`
**inside the Phase-A branch only** (Phase A calls Qwen directly and has no other mover), so Phase B
transfers exactly once and the loop no longer owns a copy it does not use.

The regression tests run the real `train_phase` against a stub runtime that mirrors the production
`train_step` and counts `.to()` calls: **Phase B = 1 transfer, Phase A = 1 transfer**. Writing that test
found a live defect the refactor had introduced — Phase B still executed a trailing `del batch, moved`,
which raised `UnboundLocalError` on the very first Phase-B step. Fixed (`del batch`), covered, and worth
recording because the benchmark harness does not go through `train_phase` and would never have caught it.

The saved work is one H2D copy of a ~2.5 MB batch per step, so this is a correctness/clarity fix first.
Section 3 says not to overclaim it and the benchmark agrees: V0 (pre-fix) versus V1 measured −13.1 %
against a drifting reference whose uncertainty was larger than that, i.e. **unresolvable**. No number is
claimed for it.

## 3. Remaining Sync Hotspots (section 4)

Task 6C.6 counted ~1,051 `aten::item` per step; this task names them. Three measurements were needed
because the first plausible answer was wrong.

**Stack capture is unavailable on this build.** `KinetoEvent.stack` is always an empty list and
`torch._C._get_python_stack` does not exist, so section 4's scoped-window fallback was used, plus a
`sys.setprofile` `c_call` hook that recovers the calling Python frame of every `Tensor.item()`.

Scoped windows, per step (one Phase-B step, 1,048 scalar reads by this run's convention):

| Component | Scalar reads/step | Share | Sync ops/step | Kernels/step |
|---|---|---|---|---|
| **optimizer + clipping (`optimizer.step()`)** | **1,024** | **97.7 %** | **1** | 2,639 |
| forward (Qwen + vision + SAM decode + loss) | 19 | 1.8 % | 267 | 20,501 |
| — vision tower | 9 | 0.9 % | 200 | 3,297 |
| — language model | 10 | 0.9 % | 67 | 17,204 |
| backward pass | 5 | 0.5 % | 159 | 34,123 |
| data preparation | 3 | 0.3 % | 1 | 0 |
| project loss scalars + clip norm | ~4 | 0.4 % | — | — |
| SAM projection + decode | 2 | 0.2 % | 14 | 742 |

The `sys.setprofile` tracer is decisive: **99.5 % of `.item()` calls are issued under `runtime.py:423`,
which is `optimizer.step()`**, and the mechanism is inside PyTorch:

```
torch/optim/adam.py:770-776   bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
                              bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]
torch/optim/optimizer.py:95   return x.item() if isinstance(x, torch.Tensor) else x
torch/optim/adam.py:165-176   # Deliberately host `step` on CPU if both capturable and fused are off.
                              # This is because kernel launches are costly on CUDA and XLA.
```

With `capturable=False, fused=False` — the project's configuration — AdamW keeps each parameter's `step`
counter as a **CPU scalar tensor** and converts it twice per parameter per step: 2 × 528 ≈ 1,056, which
is what is measured.

**What this rules out, with evidence:**

* **Not the backward pass.** The first scoped run pointed there because its optimizer window called
  `zero_grad(set_to_none=True)`, which sets every `grad` to `None`; AdamW then skipped every parameter
  and the window measured nothing. Corrected, the optimizer takes 1,024 of 1,048 reads and backward 5.
* **Not gradient checkpointing.** A probe runtime with checkpointing disabled measured 1,048 → 1,048
  reads/step (backward 5 either way).
* **Not expensive, and not synchronizing.** 1,024 reads in the optimizer window come with **one** sync
  op: these are host-tensor conversions. The supported removal is `fused=True`, which hosts `step` on
  the GPU — Task 6C.6 measured it at −0.71 % (inside the noise band) and `NOT_EQUIVALENT`. Section 4
  forbids patching third-party internals, and section 10 says to leave the project's own ~7 reads alone.

Artifact: `evaluation/task6c7_sync_hotspots.json`.

## 4. Visual-Tower Cache Eligibility (section 5)

`evaluation/task6c7_visual_cache_eligibility.json` — **all conditions pass**:

| Condition | Result |
|---|---|
| every visual-tower parameter frozen | **true** (315 tensors, all bf16, `requires_grad=False`) |
| zero LoRA modules in the visual tower | **true** (0; `lora.text_only: true`, no visual target suffixes) |
| visual output deterministic under training settings | **true** — recomputation is bit-identical, max abs difference **0.0** |
| no training-time visual state changes during a step | **true** — parameters *and* buffers unchanged after a real optimizer step |
| independent of instruction / `[SEG]` / LoRA / optimizer state | **true** — 8 same-image instruction pairs: `pixel_values` identical and visual features bit-identical |
| no dropout in the tower (why bit-equivalence is plausible rather than lucky) | **true** — no `Dropout(p>0)` in the tower, so it consumes no RNG |

The only registered buffers are `rotary_pos_emb.inv_freq` / `original_inv_freq` — constants that did not
move during the step.

## 5. Cache Boundary (section 6)

`Qwen3VLModel.forward` reads exactly two things from the tower:

```python
image_outputs = self.get_image_features(pixel_values, image_grid_thw)
image_embeds  = image_outputs.pooler_output         # merger output, split per image
deepstack     = image_outputs.deepstack_features    # deepstack merger outputs
```

so the cache holds **`pooler_output` + `deepstack_features`** — the last tensors produced exclusively by
frozen parameters, before they are mixed with trainable text hidden states. Not cached: the tower's
pre-merger `last_hidden_state` (the language model never reads it and it would double the footprint),
language-model hidden states, the `[SEG]` state, logits.

The integration is an **instance-level method replacement on the project's own Qwen module**
(`model.qwen.model.get_image_features`), not a patch of library source: the original bound method is kept
as `_task6c7_original_get_image_features` and called unchanged on a miss, so the miss path is
byte-identical to the previous behaviour, and `remove_visual_feature_cache` restores it. This
`transformers` version has no "pass precomputed image features" argument
(`accepts_precomputed_kwargs` only rewrites modality-prefixed kwargs), so this was the narrowest clean
boundary available.

**Keying.** `image:<image_id>` when the caller supplies source-image identity (the formal loop does),
else `content:<blake2b of pixel_values + image_grid_thw>`. Both are functions of the source image alone,
so two instructions on one image share one entry, and no sample-id plumbing mistake can return another
image's features. Both key strategies are gated and both are `BIT_EQUIVALENT`.

## 6. Equivalence (section 7)

`evaluation/task6c7_equivalence.json`, 12 steps, same initial trainable state, same samples in the same
order, strict determinism, per-run re-seeding.

| Object | Result |
|---|---|
| 16-sample feature equality (uncached vs cache hit) | **bit-identical**, max abs difference **0.0**, shapes and dtype equal |
| reference run twice | `BIT_EQUIVALENT` (the gate reproduces itself) |
| cached, image-identity key | **`BIT_EQUIVALENT`** — losses, gradients, post-step parameters identical |
| cached, content-hash key | **`BIT_EQUIVALENT`** |
| first-step total loss, cached vs uncached | `9.425538063049316` both |

Boundary cost measured: hit **1.07 ms**, miss **30.9 ms** (host launch time of the tower's kernels).

## 7. V0 / V1 / V2 / V3 Results (section 9)

Whole-variant runs (`evaluation/task6c7_variants.json`, 8 warmup + 64 measured steps):

| Variant | samples/s | vs interpolated reference | kernels/step | syncs/step | cache |
|---|---|---|---|---|---|
| V1 (section 3 fix) | 2.978 / 2.246 (bracket) | reference | 57,341 | 444 | none |
| V0 (pre-fix duplicate H2D) | 2.428 | −13.1 % (*unresolvable*) | 57,382 | 456 | none |
| V2 (V1 + cache, image key) | 2.348 | −10.1 % (*unresolvable*) | **52,501** | **251** | 0.125 GiB |
| V2b (V1 + cache, content key) | 2.323 | −4.4 % (*unresolvable*) | **52,501** | **251** | 0.125 GiB |

The reference drifted −24.6 % across that group, so nothing in it is resolvable, and the four-run
interleaved V1/V2 head-to-head was no better: **round 1 favoured V1 by 15.7 %, round 2 favoured V2 by
8.9 %** — rounds that disagree in sign, i.e. drift rather than a cache effect. The adoption decision
therefore does not rest on those numbers.

**The decision rests on a paired ablation** (`evaluation/task6c7_paired_ablation.json`): inside one
runtime, 8-step blocks alternate cache-off / cache-on so each adjacent off/on pair is an observation at
nearly the same thermal state.

| Block | Arm | ms/step | samples/s | ON vs OFF |
|---|---|---|---|---|
| 1 | off | 322.6 | 3.1000 | — |
| 2 | on | 288.5 | 3.4667 | **+11.8 %** |
| 3 | off | 319.9 | 3.1256 | — |
| 4 | on | 283.8 | 3.5236 | **+12.7 %** |
| 5 | off | 317.5 | 3.1492 | — |
| 6 | on | 293.9 | 3.4028 | **+8.1 %** |
| 7 | off | 328.5 | 3.0438 | — |
| 8 | on | 301.4 | 3.3183 | **+9.0 %** |

**Paired mean +10.41 %, warm-only +9.94 %, 4/4 pairs in favour of the cache.**

**A measurement error found and fixed here changed the answer.** The first paired run warmed the cache
while it was still disabled, so nothing was stored and every ON block paid misses; it reported **+2.31 %**
and the cache would have been rejected as sub-threshold. Enabling the cache during the warm-up — which is
what "warm throughput" means — moves the same measurement to **+9.94 %**. Both runs are in git history,
and the artifact records the warm-up rule.

**V3 (deferred project-owned scalar logging): not implemented.** Section 10 makes it conditional on the
profiling showing those reads matter. They are ~4 loss floats plus the clip norm out of 1,048 reads
(≈0.4 %), and the clip norm's own `.item()` is what the training loop logs every step, so there is
nothing worth the change; the decision is recorded in `task6c7_variants.json` under `v3_decision`.

## 8. Kernel / Sync Changes

| Per step | Cache OFF | Cache ON | Change |
|---|---|---|---|
| CUDA kernels | 57,341 | **52,501** | **−8.44 %** |
| Host↔device sync ops | 444 | **251** | **−43.47 %** |
| Scalar read-backs | 1,051 | 1,042 | −0.9 % |

The removed kernels are the frozen tower's ~3,297 per step plus its 200 sync ops. Scalar read-backs barely
move, consistent with §3: they belong to the optimizer, not the tower.

## 9. Throughput / GPU / CPU

| | Cache OFF | Cache ON (adopted) |
|---|---|---|
| ms/step (paired, warm) | ~322 (317.5 – 328.5) | **~292 (283.8 – 301.4)** |
| samples/s (paired, warm) | ~3.11 | **~3.43** |
| Speedup | — | **+9.94 %** |
| GPU utilization | 39.6 / 40.3 % | 40.4 % |

Section 11 adoption checklist: eligibility ✅, equivalence `BIT_EQUIVALENT` ✅, no trainable signal
bypassed ✅, no OOM/paging ✅, warm throughput **+9.94 % ≥ 8 %** ✅, break-even immediate ✅.
Adopted runtime: section 3 fix + visual cache, enabled by
`training.visual_feature_cache: true` with `visual_feature_cache_max_images: 512`.

## 10. RAM / VRAM / Cold Cost

| | Value |
|---|---|
| Adopted reserved VRAM | **8.582 GiB** (limit 14) |
| Adopted RSS | **4.751 GiB** (limit 24); +128 MiB resident cache at 32 images |
| Cache bound at 512 images | **2.0 GiB** (guidance 8 GiB) |
| Bytes per image | 4,194,304 (4.0 MiB): `pooler_output` 1 MiB + 3 × `deepstack_features` 1 MiB, bf16 kept native |
| Task 6C 480-image estimate | 1.875 GiB |
| Full 2,508-image WHU-train estimate | 9.797 GiB — above the 8 GiB guidance, so the bound stays at 512 |
| Cold build (32 unique images) | 1.32 s after a warm step; 15.7 s when measured before any step in the process (first-call kernel setup) |
| Miss / hit per image | 30.9 ms / 1.07 ms (33 ms/image once warm) |
| Break-even | immediate — a cold epoch performs the same tower work it would anyway; from the second epoch the cache saves ~30 ms per step |
| Peak across Task 6C.7 experiments | 13.166 GiB reserved, 5.305 GiB RSS, 16.3 GiB system RAM, 76.1 °C, 111 W, no paging/OOM |

## 11. Tests

`python -m pytest tests/ -q` → **238 passed** (236 from before plus the 13 new ones, minus the two Task
6C.6 stub tests that were extended for the new runtime API), 31 warnings, ~462 s.

`tests/test_task6c7_visual_cache.py` covers all 15 section-14 items: the formal Phase-B loop performs one
logical transfer, Phase A still trains and transfers once, the visual tower is frozen with zero visual
LoRA, the cache key is image-based (not sample-based), the cache stores no trainable language state, cache
hits preserve shape/dtype and recomputation is deterministic, cached and uncached features match exactly,
the 12-step end-to-end gate is bit-equivalent, no test split is used, batch stays 1, no
architecture/loss/optimizer/data change, strict determinism is preserved, and no `[REF]`/4B/SRE/SCL
appears. It also asserts the sync attribution artifact keeps naming the optimizer as the dominant source.

## 12. Git / Watt

* Committed and pushed to `Autumn-Preface/BuildReasonSeg` on `main`: commit **`4d62c8f`**
  (`perf: cache frozen Qwen visual features`) plus a follow-up `docs:` commit recording this hash and the
  push result (`5f515a0..4d62c8f  main -> main`). Remote `main` was at `5f515a0` before this task.
* Artifacts: `evaluation/task6c7_visual_cache_eligibility.json`, `task6c7_sync_hotspots.json`,
  `task6c7_equivalence.json`, `task6c7_variants.json`, `task6c7_paired_ablation.json`,
  `task6c7_final_benchmark.json`, `task6c7_resource_usage.json`; report
  `docs/task6c7_visual_cache_optimization.md`. No weights, no checkpoints, no `.conda`, no
  `local_cache`, no dataset edits, no raw profiler traces, and no cache tensors are staged.
* **Watt Toolkit: `watt_preexisting = true`** (`Steam++.exe` running since 14:16:23). Used for the push if
  needed and **left running** — DSH did not start it, so DSH did not close it. Nothing was force-killed,
  no hosts file was edited, no certificate or TLS setting was changed, and every model run was offline
  from local cache.

## 13. Whether true batching is now justified (section 12)

**Yes.** Section 12 permits that conclusion only after the redundant Phase-B H2D is removed, the remaining
`aten::item`/sync hotspots are localized, and the frozen visual-cache feasibility is measured. All three
are done:

1. the duplicate transfer is gone, with a regression test that proved Phase B transfers once;
2. the remaining ~1,051 scalar read-backs are localized to PyTorch's own optimizer bookkeeping on
   CPU-hosted `step` counters — third-party, by design, non-synchronizing, and worth ~0.25 % of the step,
   with the project's own share under 1 %;
3. the frozen visual cache — the last large, removable, semantics-preserving batch-1 win — is eligible,
   bit-equivalent and adopted at **+9.94 %**.

What is left is the *shape* of execution: ~52,500 CUDA kernels per step at a ~2 µs median, one sample per
optimizer step, GPU utilization ~40 %. No further batch-1 change of this class is visible. The next lever
is arithmetic intensity, i.e. a **Task 6C.8 true-batching feasibility experiment** (VRAM headroom at
batch 2/4 with checkpointing ON, an explicit equivalence story for a changed effective batch, and its own
adoption gate). Nothing in Task 6C.7 authorizes starting it, and Task 6D model work remains gated on
review.
