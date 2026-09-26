# Task 6C — Paired counterfactual training × neutral SAM prompt ablation

**Question.** Did Task 6B fail because its training subset never forced two different instructions on
the *same* image to select two different targets, because the SAM bridge injected a fixed positive centre
point, or because both effects interact?

**Design.** A controlled 2×2 on the frozen ADR-013 stack. No 4B, no `[REF]`, no Spatial Relation
Encoder, no Spatial Consistency Loss, no full-dataset training, and no recipe tuning.

| Arm | Training subset | SAM sparse-prompt bridge |
|---|---|---|
| `U_C` | unpaired: Task 6B's exact 480 records over **480 unique images** | **C** centre-positive: fixed point `(0.5, 0.5)`, label positive, then the projected `[SEG]` vector is added |
| `U_L` | same as `U_C` | **L** language-only: `sparse_prompt_embeddings = projected[:, None, :]`, no point / box / mask prompt |
| `P_C` | paired counterfactual: **240 images × 2 instructions** | **C** |
| `P_L` | same as `P_C` | **L** |

Everything else is shared: base snapshot (`Qwen/Qwen3-VL-2B-Instruct`, revision
`89644892e4d85e24eaac8bacfd4f463576704203`), SAM2.1 Base+ checkpoint, tokenizer and `[SEG]` id, seed
20260926, optimizer recipe, loss weights, epoch counts, the 120-record validation subset, the 20 paired
validation pairs, and the metric code.

## Correctness fixes made before the experiment

Task 6B left four defects that would have made a four-arm comparison meaningless. All four are fixed and
tested (`tests/test_task6c_fixes.py`, `tests/test_task6c_subsets.py`, `tests/test_task6c_bridge.py`).

| Fix | What was wrong | What it is now |
|---|---|---|
| **Determinism** | `training.deterministic: true` was never consumed; Phase B was not reproducible across processes | `build_runtime` calls `enable_determinism`, which seeds `random`/NumPy/torch/CUDA, sets `cudnn.benchmark=False`, `cudnn.deterministic=True` and `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and reports each setting individually |
| **Checkpoint directory** | `scripts/task6b_train.py` used a module-level constant, so the adjusted run overwrote the headline run's checkpoints | the directory is `cfg["paths"]["checkpoints"] / <ARM>`, giving four distinct directories; the constant is gone and a regression test asserts uniqueness |
| **Evaluation lookup** | the operation-chain lookup was built from `train + val` text | built from the **train split only**; val annotations may supply the expected query type but cannot extend the accepted wording |
| **Feature cache** | 160 images for 480 training images, so the LRU thrashed every epoch | 480 images, CPU-resident, with the actual RAM footprint recorded in every arm report |

## Frozen subsets

`evaluation/task6c_subset_ids.json`.

* **U** — Task 6B's exact 480 sample ids, unchanged (160 L1 + 160 L2 + 160 nontrivial L3, 480 unique
  images). The file's `source_sample_ids_sha256` records that Task 6B is the source.
* **P** — 240 unique images × 2 instructions, built deterministically from the train split. Composition
  80 × (L1+L2) + 80 × (L1+L3nt) + 80 × (L2+L3nt) = exactly 160 L1 + 160 L2 + 160 L3. Every pair has two
  different target components and two different query types; no trivial L3. Selection orders candidates by
  query type, image id and sample id and, at each of the 240 slots, takes the candidate whose query-type
  usage is currently the smallest, so the query types stay balanced without any model input.

The preferred 80/80/80 composition was **feasible exactly** — the train split offers 1454 eligible images
for L1+L2, 754 for L1+L3, and 618 for L2+L3 — so no relaxation was needed.

The validation material is reused verbatim from Task 6B: the same 120 records (40/40/40) and the same 20
paired images. `test` is never read.

## Recipe (fixed, no tuning)

Phase A: 1 epoch, LoRA + `[SEG]` only, LM CE only. Phase B: ≤ 3 epochs (1440 steps), training LoRA +
`[SEG]` + Projection MLP + SAM2 mask decoder with base and vision encoders frozen. Objective
`L = 2.0·L_lm_ce + 2.0·L_mask_bce + 1.0·L_mask_dice`, LoRA `1e-4`, token `3e-4`, decoder `3e-4`, warmup
20, cosine. No per-epoch model selection, so no tuning can creep in.

## Initialization equality

Every arm starts from a clean base (fresh LoRA, fresh `[SEG]` adapter, fresh Projection MLP, original SAM2
mask decoder). Before the first optimizer step each arm hashes every trainable tensor;
`evaluation/task6c_initialization.json` records the digests and
`tests/test_task6c_fixes.py::test_arm_fingerprints_are_equal` fails if they differ.

## Diagnostics (identical code path for all four arms)

* **Free generation (primary)** — image + instruction only; require exactly one `[SEG]`; re-forward the
  generated sequence; read the generated `[SEG]` hidden state; project; decode. Invalid generation scores
  IoU = Dice = 0.
* **Teacher-forced** — the expected reasoning prefix is supplied; diagnostic only.
* **Paired instruction dependence (primary gate)** — for the same 20 unseen val pairs, both
  `IoU(pred_A, GT_A) > IoU(pred_A, GT_B)` and `IoU(pred_B, GT_B) > IoU(pred_B, GT_A)`. A pair fails if
  either member fails to emit exactly one `[SEG]`.
* **Prompt pathway** — `[SEG]` hidden (2048), projection output (256) and the sparse prompt actually given
  to SAM, each reported as cosine / L2 / norm (plus per-dimension std and mean absolute activation for the
  projection), with 10 same-image pairs and 10 cross-image references for scale, and a centred-SVD
  spectrum (top-10 singular values, variance explained by the top 1/5/10, effective rank). A
  representation is never called "constant" on cosine alone.

## Pre-declared decision rules

`EXPERIMENT_COMPLETE_FIX_FOUND` needs at least one arm with paired ≥ 14/20, mean own-minus-cross margin
> 0.05, valid `[SEG]` emission ≥ 90 %, strict end-to-end mIoU ≥ 0.11 and no ground-truth leakage.
`EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT` needs no arm to clear that, but at least one non-baseline arm to
improve materially over `U_C` (paired passed +≥3, or mean margin +≥0.02, or same-image projected cosine
−≥0.02, or effective rank +≥1.0, or top-1 variance −≥0.05). `EXPERIMENT_COMPLETE_NO_FIX` is everything
else. `FAIL_EXPERIMENT_INVALID` covers unequal initialization, shared/overwritten checkpoints, an invalid
subset, non-bit-reproducible runs, leakage or mismatched code paths. The causal reading follows section 14
in the order: P vs U, then L vs C, then only-`P_L`, then none, then all-similar.

## Language-metric caveat (unchanged from Task 6B)

The dataset's `reasoning_zh` is template-generated (21 distinct strings in the whole training mini-set), so
reasoning exact match measures template/format mapping, not reasoning. Operation-chain accuracy is better
but still not sufficient. **The main evidence about reasoning here is mask target selection on paired
unseen images.** Wording like "language-format / template mapping transfers" is used deliberately.

## Results

All four arms trained the same number of steps (Phase A 480, Phase B 1440) from the same
initialization. The experiment is **valid**: identical initial trainable fingerprints
(`97daa58a4cff0ae904512d79a7ae50ec4b96abbd36e25c2b37f95d8956416acf`, 528 tensors, 24,010,309
parameters), four distinct checkpoint directories, valid U/P subsets, no test-split contact, and
cross-process bit reproducibility in strict deterministic mode.

### Primary (free generation on 120 unseen records)

| Arm | Bridge | Emission | strict e2e mIoU | strict Dice | teacher-forced mIoU | op-chain acc. | val LM CE |
|---|---|---|---|---|---|---|---|
| `U_C` | centre | 1.000 | 0.10516 | 0.17360 | 0.10516 | 0.983 | 0.00335 |
| `U_L` | language | 1.000 | **0.10872** | 0.17846 | 0.10873 | 0.808 | 0.04900 |
| `P_C` | centre | 1.000 | 0.10604 | 0.17851 | 0.10604 | 1.000 | 0.00000 |
| `P_L` | language | 1.000 | **0.09284** | 0.15626 | 0.09284 | 1.000 | 0.00000 |

Every arm emits exactly one `[SEG]` on all 120 records. No arm reaches the 0.11 strict-mIoU bar. The
L1/L2/L3 spread is flat in all four arms:

| Arm | L1 | L2 | nontrivial L3 |
|---|---|---|---|
| `U_C` | 0.1231 | 0.1117 | 0.0807 |
| `U_L` | 0.1213 | 0.1171 | 0.0878 |
| `P_C` | 0.1144 | 0.1227 | 0.0810 |
| `P_L` | 0.1005 | 0.1018 | 0.0763 |

### Paired instruction dependence — the primary gate

| Arm | passed | own IoU | cross IoU | mean margin | median margin | mean IoU(pred_A, pred_B) |
|---|---|---|---|---|---|---|
| `U_C` | **0/20** | 0.13149 | 0.13149 | −0.000001 | 0.000000 | 0.99971 |
| `U_L` | **0/20** | 0.12833 | 0.13186 | −0.003530 | 0.000000 | 0.97903 |
| `P_C` | **0/20** | 0.13475 | 0.13474 | +0.000007 | 0.000000 | 0.99945 |
| `P_L` | **0/20** | 0.12374 | 0.12375 | −0.000003 | 0.000000 | 0.99972 |

**No arm passes a single pair, and every mean own-minus-cross margin is within ±0.0036 of zero** against a
required > 0.05. Paired counterfactual training and the language-only bridge, alone or together, do not
create instruction-conditional mask selection at this budget.

### Prompt-pathway diagnostics (10 same-image pairs, 10 cross-image references)

| Arm | `[SEG]` hidden cos (same img) | projected cos (same img) | projected cos (diff img) | sparse-prompt cos | proj. top-1 var. | proj. top-5 var. | proj. effective rank |
|---|---|---|---|---|---|---|---|
| `U_C` | 0.99238 | 0.999997 | 1.000000 | 0.999997 | 0.9023 | 0.9959 | 1.516 |
| `U_L` | 0.98976 | 0.999909 | 0.999991 | 0.999909 | 0.9087 | 0.9952 | 1.491 |
| `P_C` | 0.99719 | 0.999996 | — | 0.999996 | **0.5892** | 0.9109 | **4.100** |
| `P_L` | 0.99898 | 0.999999 | — | 0.999999 | **0.6430** | 0.9512 | **3.371** |

* **The collapse is directional, and it is measured, not assumed.** Every arm's projected prompt has a
  same-image cosine above 0.9999 and a top-1 variance share of 0.59–0.91, so the language vector is
  dominated by one direction. The correct description is *directional collapse / near-constant direction*
  with a low but non-zero effective rank — not "a constant", because the norms and the L2 distances do
  vary.
* **Paired training materially changes the geometry.** Effective rank rises from 1.50 (U arms) to 3.74
  (P arms) and top-1 variance falls from 0.905 to 0.616. This clears the pre-declared diversity
  threshold, which is why the verdict is a partial improvement rather than "no fix".
* **Removing the fixed centre point barely helps.** The L-vs-C diversity difference is not material
  (effective rank 2.43 vs 2.81; top-1 0.776 vs 0.746), though it does make the decoded masks slightly more
  variable (`U_L` prediction-to-prediction IoU 0.979 vs `U_C` 0.9997).
* **The centre point is not drowning the language vector.** For the `C` arms the point embedding norm is
  11.39 and the projected language norm is 266.8, a ratio of **23.4×** (`P_C`: 18.3×). So the fixed point
  contributes ~4–5 % of the sparse-prompt magnitude, which is the opposite of what "the fixed centre point
  dominates" would predict — and it fits the measurement that removing it changes little.
* For the `L` arms the sparse prompt equals the projected token exactly (`equals_projected=True`), and its
  shape is `(1, 1, 256)`; for the `C` arms SAM's prompt encoder appends its own padding slot, giving
  `(1, 2, 256)`. That shape difference is stated rather than smoothed over.

### Section 14 causal reading

Pre-declared thresholds: a factor is materially better when the mean paired passed count improves by ≥ 3
or the mean margin by ≥ 0.05.

| Comparison | Paired-passed effect | Margin effect | Diversity effect |
|---|---|---|---|
| P vs U (sampling) | 0.0 vs 0.0 | +0.000002 vs −0.001765 | **material** (rank +2.23, top-1 −0.289) |
| L vs C (bridge) | 0.0 vs 0.0 | −0.001766 vs +0.000003 | not material |

Neither factor moves the paired probe, and the factors *do* move the representation. That is the
`deeper_representation_problem` branch: the failure is not fixed by paired counterfactual sampling or by
removing the fixed centre point. Paired training makes the projection use more of its 256 dimensions
without making it instruction-conditional, and the decoded masks stay almost identical between two
different instructions. The remaining problem is therefore after the prompt — in how the `[SEG]` hidden
state is formed and how SAM's decoder converts a prompt direction into a region — rather than in the
sampling scheme or the point prior.

### Verdict

**`EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`** — no arm clears the full fix gate (all four are 0/20 on the
paired probe with near-zero margins, and no arm reaches 0.11 strict mIoU), but two arms materially improve
the prompt representation over the `U_C` baseline: `P_C` (effective rank 1.52 → 4.10, top-1 variance
0.902 → 0.589) and `P_L` (effective rank 1.52 → 3.37, top-1 variance 0.902 → 0.643).

## Cost

| | Value |
|---|---|
| Optimisation time per arm | 1141–1256 s (Phase A ≈ 168 s, Phase B ≈ 1000 s) |
| Peak VRAM | 5.37 GiB allocated / **8.42 GiB reserved** of 15.894 GiB, identical in all four arms |
| SAM2 CPU feature cache | 480 images for the U arms (**7.508 GiB**: 16 MiB per image plus 8 MiB of shared constants), 240 images for the P arms (5.805 GiB) — inside the 8 GiB budget in GiB terms; in decimal units 8.06 GB, which is stated rather than rounded down |
| Determinism | strict `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, two-process bit reproducibility confirmed |

## Reproduce

```bash
python scripts/task6c_select_subsets.py     # frozen U and P subsets
python scripts/task6c_determinism.py        # two-process determinism evidence
python scripts/task6c_train.py --arm U_C    # then U_L, P_C, P_L
python scripts/task6c_compare.py            # four-arm table, causal reading, verdict
python scripts/task6c_visualize.py          # four-arm side-by-side panels
```
