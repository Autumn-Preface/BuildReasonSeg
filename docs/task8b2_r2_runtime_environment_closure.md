# Task 8B.2-R2 — Close RC1 Runtime Dependency and Readiness Gap

## 1. Task

`Task 8B.2-R2 — Close RC1 Runtime Dependency and Readiness Gap`

## 2. ChatGPT Frozen Decision

- required Ultralytics runtime: **`ultralytics==8.4.164`** (frozen U-C1 detector dependency);
- designated environment: `.conda/buildreasonseg-mvp` (`…\buildreasonseg-mvp\python.exe`, Python 3.11.16);
- no algorithm changes (no ProgramHead/Qwen prompt/detector/Reference/SAM2/D-B1/threshold/checkpoint edits);
- continuation of the **same A1** Demo smoke with the same image and the same natural-language prompt.

## 3. Initial Runtime State (Phase B)

| item | value |
|---|---|
| Python | 3.11.16 |
| executable | `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe` |
| pip | 26.2.1 (`…buildreasonseg-mvp\Lib\site-packages\pip`) |
| torch | 2.13.0+cu132 |
| CUDA | True |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| transformers | 5.17.0 |
| numpy | 2.4.6 |
| OpenCV | 5.0.0 |
| scipy | 1.17.1 |
| torchvision | 0.28.0+cu132 |
| Ultralytics | **missing** — `pip show ultralytics` → `WARNING: Package(s) not found`; `import ultralytics` → `ModuleNotFoundError` |

Old false-positive state of `RC1-ENV-01`: not re-run in R2 (Phase D pins first). It is documented in the Task 8B.2
STOP record (`logs/task8b2_stop_report.md`) and in the Task 8B.2-R1.1 state: `check_setup.py` reported READY while
`ultralytics` was not importable. After the R2 fix the two regression tests below prove missing/wrong versions make
the checker NOT READY.

## 4. pip Dry-run (Phase C)

Command: `ENV_PYTHON -m pip install --dry-run ultralytics==8.4.164` (exit 0; full output in
`logs/task8b2_r2_phaseC_dryrun.txt`).

Proposed packages (all newly installed; none replaces an existing package):

```text
cloudpickle-3.1.2 contourpy-1.3.3 cycler-0.12.1 fonttools-4.66.1 kiwisolver-1.5.1 matplotlib-3.11.2
nvidia-ml-py-13.615.71 polars-1.44.2 polars-runtime-32-1.44.2 pyparsing-3.3.3 python-dateutil-2.9.0.post0
six-1.17.0 ultralytics-8.4.164 ultralytics-platform-0.1.78 ultralytics-thop-2.2.2
```

**No protected core package (torch, torchvision, torchaudio, transformers, numpy, opencv-python,
opencv-python-headless, scipy) was proposed for upgrade, downgrade, uninstall or replacement**, and pip reported no
unsatisfied/conflicting resolution.

## 5. Dependency Specification (Phase D)

- `environment.yml`: `ultralytics>=8.3` → **`ultralytics==8.4.164`**; the nearby comment now states this is the
  frozen RC1 U-C1 detector runtime requirement.
- `requirements.txt`: `ultralytics>=8.3` → **`ultralytics==8.4.164`**.
- No other dependency version changed in either file.

## 6. RC1-ENV-01 Fix (Phase E)

- **Old problem**: readiness validation never imported the U-C1 detector runtime, so `READY` could be printed while
  `import ultralytics` was impossible.
- **New live check**: `REQUIRED_ULTRALYTICS = "8.4.164"` plus `check_runtime_dependencies(report, package)`, called
  from `run_checks()` after `check_model_package()` and before readiness is computed.
- **Missing dependency** → `[MISSING] Ultralytics runtime — U-C1 detector requires ultralytics==8.4.164` and NOT READY.
- **Wrong version** → `[INVALID] Ultralytics runtime — runtime=<actual>, required=8.4.164` and NOT READY.
- **Exact match** → `[OK] Ultralytics runtime — runtime=8.4.164; model provenance=8.4.164`.
- The live version always comes from the imported module; a runtime that differs from the model provenance is NOT READY.
- Transformers: real `import transformers`; missing → `[MISSING] Transformers runtime` and NOT READY; otherwise the
  live version is reported (with provenance shown separately). No exact Transformers version is enforced in R2.

## 7. Runtime vs Model Provenance

| concept | source | value |
|---|---|---|
| live runtime torch | imported module | 2.13.0+cu132 |
| live runtime ultralytics | imported module | 8.4.164 |
| live runtime transformers | imported module | 5.17.0 |
| model provenance (metadata.json `framework`) | `model/buildreasonseg_advisor/metadata.json` | pytorch `2.13.0+cu132 …`, ultralytics `8.4.164`, transformers `5.17.0`, python `3.11` |

`metadata.json` is **provenance**, not proof that a package is installed: the checker now reports the live runtime
separately and never presents a metadata value as an installation.

## 8. Portability Wording (Phase E4)

New wording:

```text
[OK] source/assets portability — delivery 内未发现 workspace 绝对路径依赖；runtime environment 需单独配置
```

The scan algorithm is unchanged. The wording now claims only source/assets path portability; the Python/Conda
runtime environment remains separately configured and is **not** bundled into the delivery. `项目可整体移动` no
longer appears anywhere in `check_setup.py`.

## 9. Installation Before/After (Phase H)

| package | before | after | changed? |
|---|---|---|---|
| torch | 2.13.0+cu132 | 2.13.0+cu132 | no |
| torchvision | 0.28.0+cu132 | 0.28.0+cu132 | no |
| transformers | 5.17.0 | 5.17.0 | no |
| numpy | 2.4.6 | 2.4.6 | no |
| OpenCV | 5.0.0 | 5.0.0 | no |
| scipy | 1.17.1 | 1.17.1 | no |
| ultralytics | missing | **8.4.164** | expected |

`pip check` → `No broken requirements found.` (exit 0).

## 10. Canonical/Delivery Sync (Phases G, I, J)

- canonical manifest: **135 entries** (unchanged count; only `bytes`/`sha256` of the four authorized files updated:
  `environment.yml`, `requirements.txt`, `check_setup.py`, `tests/test_setup_checker.py`).
- canonical sync tests: `pytest tests/test_sync_advisor_rc1_delivery.py -q` → **23 passed**.
- pre-sync read-only check: exactly the four expected mismatches, `checked=135 match=131 missing=0 mismatch=4`.
- one normal sync: `copied=135 verified=135 failures=0`.
- post-sync read-only check: `checked=135 match=135 missing=0 mismatch=0` (exit 0). No external model asset deleted.

### 10.1 Post-review correction inside R2 (recorded for transparency)

The first external run of `tests/test_setup_checker.py` produced 10 failures with
`TypeError: argument of type 'NoneType' is not iterable`. Root cause: after `check_setup.py` imports
`ultralytics`/`transformers`, the child process writes UTF-8 to the pipe while the test helper decoded with the
machine locale (GBK), so the reader thread raised `UnicodeDecodeError` and `stdout` became `None`. The fix pins the
helper decoding to `encoding="utf-8", errors="replace"` in the authorized test file only; the manifest entry for that
file was refreshed and the delivery re-synced. No runtime or algorithm behaviour was involved.

## 11. Validation

| Check | Result |
|---|---|
| dry-run safety gate | PASS |
| ultralytics import 8.4.164 | PASS |
| pip check | PASS |
| protected packages unchanged | PASS |
| missing-ultralytics regression | PASS |
| wrong-version regression | PASS |
| true check_setup READY | PASS (`BuildReasonSeg environment: READY`, exit 0) |
| portability wording | PASS (`source/assets portability`; no `项目可整体移动`) |
| canonical sync tests | PASS (23 passed) |
| external setup tests | PASS (12 passed) |
| external full suite | PASS (109 passed) |
| repository full suite | PASS (1535 passed; the previously skipped test now runs because the dependency is installed) |
| A1 Qwen parse | PASS (`largest -> right_of -> nearest`) |
| A1 Y-path full inference | PASS (`Result : SUCCESS`) |
| A1 N-path early abort | PASS (`E901 USER_ABORTED`, exit 90, no detector/core, no new artifacts) |

## 12. A1 Positive Path (Phase M)

Unchanged prompt (verbatim):

```text
找出最大的建筑，然后把它右边离它最近的那栋分割出来
```

- initial parse: `largest -> right_of -> nearest` (`largest_to_right_of_to_nearest`);
- command confirmation: exactly one `Y`;
- U-C1 detector executed (no `ModuleNotFoundError`);
- Reference resolution executed (Reference ID `48`);
- SAM2 executed; D-B1 executed;
- final status: `Result : SUCCESS` (exit 0);
- generated artifacts:

```text
mask        = inference/output/masks/A1_mask.png
overlay     = inference/output/overlays/A1_overlay.png
diagnostics = inference/output/diagnostics/A1/
```

This is execution closure only: no visual semantic scoring was performed and no result was used to tune anything.

## 13. A1 Negative Path (Phase N)

- initial parse: `largest -> right_of -> nearest` (`largest_to_right_of_to_nearest`) — unchanged;
- `N` at the confirmation → existing user-abort behaviour `[E901 USER_ABORTED]`, exit code 90;
- U-C1 detector not loaded/run; SAM2 not run; D-B1 not run (no such markers in the transcript);
- no new final mask/overlay: output counts were 96 masks / 96 overlays before and after the N run.

## 14. RC1 Final Environment State

`READY`

## 15. Remaining Issues

- Windows locale decoding nuance: subprocess-based checks must decode child output explicitly (UTF-8) because the
  new runtime imports switch the child stream encoding; handled in the RC1 setup tests.
- `RC1-ENV-01` is fixed for the live-import path; other third-party runtime requirements discovered in future
  (e.g. a different detector family) would need the same explicit live-runtime validation.
- No other factual remaining issue was observed in R2.

## 16. Scientific Scope

> Task 8B.2-R2 did not modify training, checkpoints, detector thresholds, tiling/merge policy, ProgramHead semantics, Qwen prompt template, Reference policy, SAM2, relation fields, D-B1, datasets, final-test results or scientific claims. It only closed the RC1 runtime dependency/readiness gap and continued the already-authorized A1 Demo smoke.
