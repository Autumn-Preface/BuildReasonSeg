# Task 6C.7 — Frozen Qwen visual-feature cache and remaining batch-1 sync overhead

**Verdict: `OPTIMIZATION_PARTIAL` (two changes adopted).**

1. **The formal Phase-B duplicate H2D transfer is removed** — exact, and it makes the loop correct
   about which component owns the transfer.
2. **The frozen Qwen visual-tower feature cache is adopted**: `BIT_EQUIVALENT` over a 12-step gate,
   **+9.94 %** warm throughput in a paired ablation (4/4 pairs, +8.1 % … +12.7 %), **−8.44 %** CUDA
   kernels and **−43.5 %** host↔device sync ops per step, at 4 MiB per image and a 2 GiB default bound.
3. **The remaining ~1,051 scalar read-backs per step are localized and dismissed**: they are PyTorch's
   own per-parameter optimizer bookkeeping on CPU-hosted `step` counters, they do not synchronize the
   device, and the project's own share is under 1 %. No action, by section 10.

This is performance engineering. Nothing here changes the model, the loss, the data, the optimizer
mathematics or the sample order, and it does not reinterpret any Task 6C result.

## 1. Verdict

| | Value |
|---|---|
| Phase-B duplicate H2D | **removed** (section 3); exact; effect small and not overclaimed |
| Remaining scalar read-backs | **localized** to `torch.optim` bias-correction bookkeeping (97.7 %), cheap, third-party |
| Frozen visual cache | eligibility **passes**, equivalence **`BIT_EQUIVALENT`**, **adopted** |
| Warm paired speedup | **+9.94 %** (all-pairs +10.41 %, 4/4 pairs in favour) |
| Adopted runtime footprint | 8.582 GiB reserved VRAM, 4.751 GiB RSS + 128 MiB cache (32 images) |
| Adopted runtime | section 3 fix + visual cache, `training.visual_feature_cache: true` |

## 2. The formal Phase-B duplicate transfer (section 3)

`scripts/task6c_train.py` used to do this on every Phase-B step:

```python
batch, image = runtime.prepare(sample)
moved = batch.to(runtime.device)      # created …
...
result = runtime.train_step(batch, ...)  # … and never used: train_step copies again
```

`MvpRuntime.train_step` performs `batch = batch.to(self.device)` itself, so the first copy was
allocated, filled and discarded. The loop now creates `moved` **inside the Phase-A branch only**,
because Phase A calls Qwen directly and has no other mover. Phase B hands the CPU batch to
`train_step`, which moves it exactly once.

* Regression test: `tests/test_task6c7_visual_cache.py` runs the real `train_phase` against a stub
  runtime whose `train_step` mirrors the real one and counts `.to()` calls — Phase B performs
  **exactly one** logical transfer, Phase A still performs exactly one.
* The saved work is one H2D copy of the batch (~2.5 MB of `pixel_values` plus ids) per step, so this is
  a correctness-and-clarity fix first and a performance fix second. Section 3 says not to overclaim it,
  and the benchmark reports it against the reference drift rather than asserting a number.

## 3. Where the remaining ~1,051 scalar read-backs come from (section 4)

Task 6C.6 counted ~1,051 `aten::item` per step without attributing them. Three independent
measurements were needed, because two plausible answers turned out to be wrong.

**Python stack capture is unavailable on this build.** `KinetoEvent.stack` is always empty and
`torch._C._get_python_stack` does not exist, so section 4's scoped-window fallback was used, plus a
`sys.setprofile` `c_call` hook that recovers the calling frame of every `Tensor.item()` call.

Scoped windows (per step, one full Phase-B step = 1,048 scalar reads by that run's convention):

| Component | Scalar reads / step | Share | Sync ops / step | Kernels / step |
|---|---|---|---|---|
| **optimizer + clipping (`optimizer.step()`)** | **1,024** | **97.7 %** | **1** | 2,639 |
| forward (Qwen + vision + SAM decode + loss) | 19 | 1.8 % | 267 | 20,501 |
| — of which vision tower | 9 | 0.9 % | 200 | 3,297 |
| — of which language model | 10 | 0.9 % | 67 | 17,204 |
| backward pass | 5 | 0.5 % | 159 | 34,123 |
| data preparation (processor + tokenizer) | 3 | 0.3 % | 1 | 0 |
| project loss scalars + clip norm | ~4 | 0.4 % | — | — |
| SAM projection + mask decode | 2 | 0.2 % | 14 | 742 |

The optimizer window issues **1,024 scalar read-backs while synchronizing the device once** — the
signature of reading *CPU* scalars, which the source confirms.

The `sys.setprofile` tracer names the call site directly: **99.5 % of `.item()` calls are issued under
`runtime.py:423 → optimizer.step()`**, and the mechanism is in PyTorch itself —

```
torch/optim/adam.py:770-776   bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
                              bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]
torch/optim/optimizer.py:95   return x.item() if isinstance(x, torch.Tensor) else x
torch/optim/adam.py:165-176   # Deliberately host `step` on CPU if both capturable and fused are off.
                              # This is because kernel launches are costly on CUDA and XLA.
```

With `capturable=False, fused=False` — the project's configuration — AdamW keeps each parameter's
`step` counter as a **CPU scalar tensor** and converts it to a Python number **twice per parameter per
step**: 2 × 528 trainable tensors ≈ 1,056, which is what is measured.

Three consequences, all measured:

* **It is not the backward pass.** The first scoped run suggested backward because its optimizer window
  was wrong: `zero_grad(set_to_none=True)` set every `grad` to `None`, AdamW then skipped every
  parameter, and the window measured nothing. With the window corrected, the optimizer takes 1,024 of
  1,048 reads (97.7 %) and backward takes 5 (0.5 %).
* **It is not gradient checkpointing.** Disabling checkpointing for the same step left the count
  unchanged (1,048 → 1,048 reads/step, backward 5 both ways).
* **It does not synchronize the device, and it is cheap.** 1,024 reads in the optimizer window come with
  **one** sync op, so the counters are being read on the host. The supported way to remove them is
  `fused=True` (which hosts `step` on the GPU), and Task 6C.6 measured that at −0.71 % (inside the noise
  band) and `NOT_EQUIVALENT`. Section 4 forbids patching third-party internals, and the project's own
  reads are under 1 % of the total, so section 10's answer is to leave them alone.
  Artifact: `evaluation/task6c7_sync_hotspots.json`.

## 4. Visual-tower cache eligibility (section 5)

`evaluation/task6c7_visual_cache_eligibility.json` — **all conditions pass**:

| Condition | Result |
|---|---|
| every visual-tower parameter frozen | **true** (315 tensors, all `requires_grad=False`, all bf16) |
| zero LoRA modules in the visual tower | **true** (0; the LoRA config is `text_only: true`) |
| visual output deterministic under training settings | **true** — recomputation is bit-identical, max abs difference **0.0** |
| no training-time visual state changes during a step | **true** — parameters *and* registered buffers unchanged |
| independent of the instruction text / `[SEG]` / LoRA / optimizer state | **true** — 8 same-image instruction pairs: `pixel_values` identical **and** visual features bit-identical |
| no dropout in the tower (why bit-equivalence is plausible, not lucky) | **true** — no `Dropout(p>0)` anywhere in the tower |

The only registered buffers are `rotary_pos_emb.inv_freq` / `original_inv_freq`, which are constants and
did not move during a real optimizer step.

## 5. Cache boundary (section 6)

Inspecting `transformers/models/qwen3_vl/modeling_qwen3_vl.py`, `Qwen3VLModel.forward` reads exactly two
things from the visual tower:

```python
image_outputs = self.get_image_features(pixel_values, image_grid_thw)
image_embeds  = image_outputs.pooler_output         # merger output, split per image
deepstack     = image_outputs.deepstack_features    # deepstack merger outputs
```

so the cache stores **`pooler_output` + `deepstack_features`** and nothing else — the last tensors
produced exclusively by frozen parameters, before they are mixed with trainable text hidden states.
Not cached: the vision tower's pre-merger `last_hidden_state` (the language model never reads it, and it
would double the footprint), language-model hidden states, the `[SEG]` state and logits.

The wrapper is an **instance-level method replacement on the project's own Qwen module**
(`model.qwen.model.get_image_features`), not a patch of library source: the original bound method is
kept as `_task6c7_original_get_image_features` and called unchanged on a miss, so the miss path is
byte-identical to today's behaviour, and `remove_visual_feature_cache` restores it. There is no official
"pass precomputed image features" argument in this `transformers` version (`accepts_precomputed_kwargs`
only rewrites modality-prefixed kwargs), so this was the cleanest available boundary.

**Cache key.** `image:<image_id>` when the caller supplies the source image identity, else
`content:<blake2b of pixel_values + image_grid_thw>`. Both are functions of the source image alone, so
two instructions on one image share one entry (verified: the 8 pairs above produce identical pixels),
and no sample-id plumbing mistake can return another image's features.

**Footprint** (measured, bf16 kept native):

| | Value |
|---|---|
| `pooler_output` | `[256, 2048]` bf16 |
| `deepstack_features` | 3 × `[256, 2048]` bf16 |
| bytes per image | **4,194,304 (4.0 MiB)** |
| Task 6C 480-image estimate | 1.875 GiB |
| full 2,508-image WHU-train estimate | 9.797 GiB |
| default bound | `max_images=512` → **2.0 GiB** |

The working set that matters here is 240 unique images (Task 6C `P` subset) or 480 (`U`), both inside the
default bound; a full 2,508-image cache would need 9.8 GiB and is **not** enabled by default, which is
the explicit justification section 8 asks for. The cache is CPU-resident, LRU-bounded, disabled by
default and never pinned.

## 6. Equivalence (section 7)

`evaluation/task6c7_equivalence.json` — 12 steps, same initial trainable state, same samples in the same
order, strict determinism, **per-run re-seeding** (the Task 6C.5 lesson: LoRA dropout consumes the global
RNG).

| Object | Result |
|---|---|
| 16-sample feature equality: uncached vs cache hit | **bit-identical**, max abs difference **0.0**, same shapes, same dtype (`torch.bfloat16`) |
| Reference run twice (control) | `BIT_EQUIVALENT` — the gate reproduces itself |
| Cached, keyed by source-image identity | **`BIT_EQUIVALENT`** (losses, gradients, post-step parameters all identical) |
| Cached, keyed by pixel content hash | **`BIT_EQUIVALENT`** |
| First-step total loss, cached vs uncached | `9.425538063049316` both |

Measured cost of the boundary itself: a cache **hit costs 1.07 ms**, a **miss (the real tower) 30.9 ms**
— and the miss figure is host launch time for the tower's kernels, which is why the end-to-end effect is
~10 % of a 320 ms step rather than ~9× larger. Two key strategies are gated because the image-identity
key depends on plumbing while the content hash does not; both are bit-exact, so the cheaper one
(image identity) is what the runtime uses, with the content hash as the automatic fallback.

## 7. V0 / V1 / V2 results and the measurement problem (section 9)

`evaluation/task6c7_variants.json` (whole-variant runs) and `evaluation/task6c7_paired_ablation.json`
(fine-grained paired ablation).

Whole-variant runs, 8 warmup + 64 measured steps, bracketed references with interpolated comparison:

| Variant | samples/s | vs interpolated reference | kernels/step | syncs/step |
|---|---|---|---|---|
| V1 (section 3 fix) | 2.978 / 2.246 (first / last) | reference | 57,341 | 444 |
| V0 (pre-fix duplicate H2D) | 2.428 | −13.1 % (*unresolvable*) | 57,382 | 456 |
| V2 (V1 + visual cache, image key) | 2.348 | −10.1 % (*unresolvable*) | **52,501** | **251** |
| V2b (same, content key) | 2.323 | −4.4 % (*unresolvable*) | **52,501** | **251** |

The reference drifted **−24.6 %** across that group (first run 2.978, last 2.246), so nothing in it is
resolvable — and the four-run interleaved V1/V2 head-to-head was no better: **round 1 favoured V1 by
15.7 %, round 2 favoured V2 by 8.9 %**, i.e. rounds that disagree in sign. That is drift, not a cache
effect, and it is why the adoption decision does not rest on those numbers.

**Paired ablation.** Inside one runtime, 8-step blocks alternate cache-off / cache-on, so each adjacent
off/on pair is an observation taken at nearly the same thermal state. The vision tower has no dropout,
so toggling the cache cannot perturb the RNG, and both arms train identically.

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

**One measurement error was found and fixed here, and it changed the answer.** The first paired run
warmed the cache while it was still disabled, so nothing was stored and every ON block paid misses; it
reported **+2.31 %** and the cache would have been rejected. Enabling the cache during the warm-up
(which is what "warm throughput" means) moves the same measurement to **+9.94 %**. Both runs are in git
history and the artifact records the warm-up rule; this is exactly the kind of self-inflicted error that
section 11's gates exist to catch.

## 8. Kernel and sync changes

Measured per step, same runtime, same samples, one profiled block per arm:

| | Cache OFF | Cache ON | Change |
|---|---|---|---|
| CUDA kernels | 57,341 | **52,501** | **−8.44 %** |
| Host↔device sync ops | 444 | **251** | **−43.47 %** |
| Scalar read-backs (`aten::item` pairs) | 1,051 | 1,042 | −0.9 % |

The removed kernels are the frozen tower's ~3,297 per step plus its 200 sync ops; the scalar read-back
count barely moves, which is consistent with §3's finding that those come from the optimizer, not the
vision tower.

## 9. Throughput / GPU / CPU

| | Cache OFF | Cache ON (adopted) |
|---|---|---|
| ms/step (paired, warm) | ~322 (317.5 – 328.5) | **~292 (283.8 – 301.4)** |
| samples/s (paired, warm) | ~3.11 | **~3.43** |
| Speedup | — | **+9.94 %** |
| GPU utilization | 39.6 / 40.3 % (whole-variant runs) | 40.4 % |

Adoption follows section 11: eligibility ✅, equivalence ✅ `BIT_EQUIVALENT`, no trainable signal
bypassed ✅, no OOM ✅, warm throughput **+9.94 % ≥ 8 %** ✅, break-even immediate ✅.

## 10. RAM / VRAM / cold cost

| | Value |
|---|---|
| Adopted runtime reserved VRAM | **8.582 GiB** (limit 14) |
| Adopted runtime RSS | **4.751 GiB** (limit 24) |
| Cache resident (32 benchmark images) | 128 MiB |
| RSS + cache | 4.876 GiB, well inside the budget |
| Cache bound at `max_images=512` | **2.0 GiB** (guidance 8 GiB) |
| Bytes per image | 4,194,304 (4.0 MiB): `pooler_output` 1 MiB + 3 × `deepstack_features` 1 MiB, bf16 kept native |
| Cold build, 32 unique images | **1.32 s** measured after a warm step (15.7 s when measured before any step in the process, i.e. first-call kernel setup) |
| Cold build, 240-image Task 6C `P` subset | ≈ 10–12 s equivalent work (measured 33 ms/image once warm) |
| Break-even | immediate: the cache performs the same tower work a cold epoch would, then saves ~30 ms per step from the second epoch onward |
| Peak across all Task 6C.7 experiments | 13.166 GiB reserved, 5.305 GiB RSS, 76.1 °C, 111 W, no paging |

The cache is CPU-resident and LRU-bounded, is never pinned, and a full 2,508-image WHU-train cache would
need 9.8 GiB — above the 8 GiB guidance — so the bound stays at 512 images, which covers both the
240-image `P` subset and the 480-image frozen training subset.

## 11. Tests

`python -m pytest tests/ -q` → see `handoff/FROM_DSH.md` §11 for the recorded result.

`tests/test_task6c7_visual_cache.py` covers the 15 section-14 items, and it caught one real defect while
being written: Phase B still executed a trailing `del batch, moved` after the fix, raising
`UnboundLocalError` on the first Phase-B step. The regression tests now run the formal `train_phase`
against a stub runtime and assert exactly one logical device transfer in **both** phases.

## 12. Reproduce

```bash
python scripts/task6c7_visual_cache.py     # eligibility, boundary, footprint
python scripts/task6c7_profile.py          # sync attribution (scopes + item call-site tracer)
python scripts/task6c7_equivalence.py      # 16-sample feature equality + 12-step gate
python scripts/task6c7_benchmark.py --group variants
python scripts/task6c7_benchmark.py --group final
python scripts/task6c7_benchmark.py --group paired   # the measurement the decision uses
python scripts/task6c7_summarize.py        # decisions + resource accounting
```

`PYTHONUTF8=1` is still required to read `torch.compile` errors on this machine, and the sync tracer
uses `sys.setprofile`, which slows execution — it is a diagnostic, never a timing measurement.

## 13. Is true batching now the next lever? (section 12)

Yes, and this task is what makes that statement earned rather than assumed. All three closing conditions
are met: the redundant Phase-B H2D is gone, the remaining `aten::item`/sync traffic is localized (to
third-party optimizer bookkeeping that does not synchronize) and the frozen visual cache — the last
large, removable, semantics-preserving win at batch 1 — is measured, gated and adopted at +9.94 %.

What remains is the shape of the execution: ~52,500 CUDA kernels per step at a ~2 µs median, one sample
per optimizer step, GPU utilization ~40 %. No further batch-1 change of this kind is visible; raising
arithmetic intensity per launch is the next lever, and it is a new experiment (Task 6C.8 feasibility:
VRAM headroom at batch 2/4 with checkpointing ON, an explicit equivalence story for a changed effective
batch, and its own adoption gate), not a performance tweak. Nothing in this task authorizes it.
