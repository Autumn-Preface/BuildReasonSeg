# Task 8B.3 — Fixed Six-Image Automatic Demo Suite

> Status: **PARTIAL** (Task 8B.3-R1). Deterministic driver implemented; driver-test gate not green, therefore the
> real six-case suite was **not** started (Task 8B.3-R1 §10 forbids running it before both test commands pass).

## 1. Task

`Task 8B.3 — Fixed Six-Image Automatic Demo Suite` (executed under **Task 8B.3-R1**, the deterministic-driver task).

## 2. Preconditions (Phase A)

| item | value |
|---|---|
| validation branch | `eval/task8b3-six-image-demo-suite` @ `c45ecbec7fd293c454ccced22310db32c1542be4` |
| audited RC1 commit | `c45ecbec7fd293c454ccced22310db32c1542be4` |
| local `main` before/after | `6c2b915…` → **`c45ecbec…`** |
| `origin/main` before/after | `6c2b915…` → **`c45ecbec…`** (pushed this task) |
| working tree | only `M handoff/TO_DSH.md` (the task book) |

Deviation recorded for transparency: `git switch main` aborts while the task book edit is unstaged, and
stash/reset/discard is forbidden; `main` was therefore fast-forwarded **in place** with
`git branch -f main audit/task8b2-rc1-runtime-closure` after verifying `main` is an ancestor (identical to
`merge --ff-only`, no working-tree change), then pushed. No merge/rebase of new work occurred.

## 3. Fixed Suite

A1 → A2 → A3 → A4 → B1 → B2, Automatic Mode only, real CLI with `--confirm-command`, prompts frozen by the task
book and never altered or retried; no `--reference-id`, no `--inspect-proposals`, no Assisted Mode.

## 4. Input Identity (Phase G of the earlier attempt, unchanged)

| file | bytes | width | height | mode | channels | sha256 |
|---|---:|---:|---:|---|---:|---|
| A1.png | 1607301 | 1024 | 1024 | RGB | 3 | 8a4b459d65773a7dfb0ffcc509c26b5d3a7cea23cd759cdadf94cd46be84c227 |
| A2.png | 1677040 | 1024 | 1024 | RGB | 3 | 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f |
| A3.png | 1775940 | 1024 | 1024 | RGB | 3 | f3cd05870385bd7f978b345b703d407cca757d909fc7be64011449ca711b19fd |
| A4.png | 1529883 | 1024 | 1024 | RGB | 3 | a3962ed18467997366de7b070d7ae39b3ac9916ca01497014ad5c03b434775a3 |
| B1.tif | 75040370 | 5000 | 5000 | RGB | 3 | 8b68c9e2fe3438b511e482998f34f772a870b99020eb5e7fc1b83b08b4828e31 |
| B2.tif | 75080895 | 5000 | 5000 | RGB | 3 | c91663edd7abfaa3d1f198a9903ce28b5d8c99f80028fc24bd883e285b62319a |

## 5. Deterministic Driver (Phase B/C/D/E)

`scripts/task8b3_interactive_suite.py` (validation harness, never shipped to the RC1 delivery):

- `subprocess.Popen(..., cwd=EXTERNAL_RC1_ROOT, stdin=PIPE, stdout=PIPE, stderr=STDOUT, text=False, bufsize=0)`;
- `child_env = os.environ.copy()` plus exactly `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1`, `PYTHONUTF8=1`
  (plus the offline flags already used by the delivery); no CUDA/model/device variables set;
- binary incremental read loop + `codecs.getincrementaldecoder("utf-8")(errors="replace")`; **no** `readline()`,
  **no** `for line in pipe`, **no** `communicate(input=...)`;
- exact prompt substrings detected without requiring a newline, each answered at most once:
  `是否按此理解执行？ [Y/N]: `, `是否使用建议指令继续？ [Y/N]: `, `是否进入有限兼容模式？ [Y/N]: `;
- Y/N strictly by frozen program comparison: direct parse == expected → `Y` (`DIRECT_CORRECT`), otherwise `N`
  (`LANGUAGE_ERROR_SUPPORTED_WRONG`); suggestion display mapped only in the harness → equal → `Y`
  (`FALLBACK_CORRECT`) else `N` (`FALLBACK_WRONG`); fallback prompt → `N`
  (`LANGUAGE_RUNTIME_ERROR_OR_FALLBACK_REQUEST`); initial program extracted only with
  `\[解析\].*?\(([^()\r\n]+)\)`;
- per-case wall-clock timeout 15 min → terminate, 10 s wait, kill if alive, `DRIVER_TIMEOUT`, transcript kept,
  `check_setup.py` re-run, continue only if READY.

## 6. Driver Tests (Phase F)

`tests/test_task8b3_interactive_suite.py` — fake child scripts only (no models/images):

```text
python -m pytest tests/test_task8b3_interactive_suite.py -q
11 passed, 2 failed
```

Failures (both are defects of the new harness test file itself, not of the driver or the RC1 product):

1. `test_correct_direct_program_sends_y` — the fake-child helper still defaults to the A1 program instead of the
   per-test expected program, so the helper answers `N`;
2. `test_decision_path_has_no_readline` — the forbidden-construct scan matches the driver's own docstring text
   describing the prohibition.

Because this command must PASS before the real suite (Task 8B.3-R1 §10), the real suite was **not** started.

Repository suite: **NOT RUN** (same gate).

## 7. Per-Case Results

| ID | language | runtime | note |
|---|---|---|---|
| A1 | NOT RUN | NOT RUN | suite not started (driver-test gate) |
| A2 | NOT RUN | NOT RUN | " |
| A3 | NOT RUN | NOT RUN | " |
| A4 | NOT RUN | NOT RUN | " |
| B1 | NOT RUN | NOT RUN | " |
| B2 | NOT RUN | NOT RUN | " |

## 8. Language Resolution Summary

`direct_correct`, `fallback_correct`, `language_failed` — **NOT RUN** (no case executed).

## 9. Automatic Runtime Summary

`runtime_success_pending_visual_review`, `runtime_failed` — **NOT RUN** (no case executed).

## 10. Large-Image B1/B2

**NOT RUN** (tiling/merge/full-size mapping evidence not produced).

## 11. Review Pack

**NOT CREATED** (no outputs to review).

## 12. Visual Verdict = `PENDING CHATGPT/USER REVIEW`

No mask/overlay was produced, so no visual question can be answered yet. DSH makes no semantic visual verdict.

## 13. R2 Handoff Clarification

> R2 intentionally installed `ultralytics==8.4.164`; pip also installed its previously missing transitive/runtime
> dependencies listed in the R2 dry-run. No protected core package version changed. The old R2 handoff sentence
> "no package other than ultralytics added" was overly narrow; the detailed R2 audit report was authoritative and
> already listed the added dependencies.

Documentation clarification only — not an R2 runtime defect.

## 14. No-Tuning Statement

- no prompt retry and no wording change;
- no source edit (no RC1 runtime/`predict.py`/ProgramHead/Qwen-suggestion/detector/Reference/SAM2/D-B1 change);
- no threshold/config change; no checkpoint change;
- no Assisted Mode / `--reference-id` / `--inspect-proposals`;
- no training; no final-test access; no download; no package installation.

## 15. Output-Layout Requirement (Phase L)

```text
Each predict run should own one independent directory under inference/output,
with subdirectories diagnostics/, masks/, overlays/.
```

Status: **ACCEPTED / DEFERRED TO SEPARATE DELIVERY ITERATION** (reserved as Task 8B.4). Not implemented here.

## 16. Next Gate

> Awaiting ChatGPT/user visual review. Do not automatically run Assisted Mode, additional directions, additional
> images or Task 8C.

Also awaiting resolution of the two harness-test defects above (both confined to
`tests/test_task8b3_interactive_suite.py`) before any real six-case run.
