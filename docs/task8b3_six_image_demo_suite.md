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

## Task 8B.3-R2 — Driver Audit Closure and Formal Suite

### R2.1 ChatGPT R1 audit findings

1. **Test defect A** — `_run_fake(... expected=...)` ignored its `expected` argument and hard-coded A1's program;
2. **Test defect B** — `test_decision_path_has_no_readline` scanned raw source text and falsely matched the
   driver's explanatory docstring;
3. **Driver timeout defect** — blocking `os.read(...)` in the main control loop could block past the 15-minute
   deadline when the child stayed alive without emitting bytes;
4. **Environment-scope deviation** — the harness added `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.

### R2.2 Exact driver fixes (harness only)

- `child_environment()` now copies `os.environ` and overrides **exactly** `PYTHONIOENCODING=utf-8`,
  `PYTHONUNBUFFERED=1`, `PYTHONUTF8=1` — the two offline flags are no longer created or overridden by the harness;
- the interaction core was rewritten around a **daemon reader thread + `queue.Queue`**: the thread performs the
  blocking binary reads and pushes raw chunks plus one EOF sentinel, the main thread consumes with
  `queue.get(timeout=0.10)`, decodes incrementally, detects the exact prompts, makes every Y/N decision and owns the
  `time.monotonic()` deadline — so the 900 s timeout fires even if the child prints zero bytes
  (terminate → 10 s wait → kill → bounded drain → `DRIVER_TIMEOUT`);
- a reusable `_run_interactive_process(...)` helper now carries that logic for both the six frozen cases and the
  harness tests;
- frozen constants, prompts, regexes, `DISPLAY_TO_PROGRAM`, `CASES` and `CASE_TIMEOUT_SECONDS = 900` were not
  changed; no product/runtime/delivery file was touched.

### R2.3 Dedicated test gate (Phase G) — did not reach 100 % PASS

```text
python -m pytest tests/test_task8b3_interactive_suite.py -q
12 passed, 3 failed
```

Failing harness tests:

```text
- test_repeated_prompt_is_answered_only_once
- test_no_executable_readline_or_communicate_in_driver
- test_suite_cases_and_prompts_frozen
```

All three are defects **of the new harness test file itself**, not of the driver or the RC1 product:

- `test_repeated_prompt_is_answered_only_once` — the fake child asks the same prompt twice and therefore blocks
  forever on the second question, which the driver (correctly) never answers; the test's expectation that the run
  completes is wrong.
- `test_no_executable_readline_or_communicate_in_driver` — the AST-based scan still reports a match; the driver
  itself only calls `stream.read(...)` and `subprocess.run(...)`.
- `test_suite_cases_and_prompts_frozen` — the exact-tuple comparison against the driver's `CASES` does not match as
  written (the A3 entry in the driver is built with a `str.replace` expression rather than a literal).

Per Task 8B.3-R2 §9 the formal suite was therefore **not** started, and no further patch was applied after this gate.

### R2.4 Other gates

| gate | result |
|---|---|
| repository suite (`pytest tests/ -q`) | NOT RUN (§10 requires the dedicated gate to pass first) |
| external `check_setup.py` | READY (live `ultralytics==8.4.164`; verified in R1 and unchanged since) |
| six input hashes | unchanged vs R1 (A1–A4 1024×1024 RGB PNG; B1/B2 5000×5000 RGB TIFF) |
| FORMAL_SUITE_FREEZE | NOT RECORDED (gate not reached) |

### R2.5 Formal suite (Phase K)

`NOT RUN ONCE` — no sample was executed; A1–B2 remain `NOT RUN` for language and runtime.

### R2.6 Review pack (Phase M)

NOT CREATED (no outputs produced).

### R2.7 Visual verdict

`PENDING CHATGPT/USER REVIEW` — no mask/overlay exists to review.

### R2.8 No-tuning statement

No prompt was changed or retried; no RC1 product/runtime/`delivery_src`/`predict.py`/ProgramHead/Qwen-suggestion/
detector/Reference/SAM2/D-B1/threshold/config/checkpoint was modified; no Assisted Mode, `--reference-id` or
`--inspect-proposals` was used; no training, download, package installation or final-test access occurred.

### R2.9 Output-layout requirement

`ACCEPTED / DEFERRED TO TASK 8B.4` — still not implemented.

### R2.10 Exact STOP reason

Dedicated harness-test gate not green (`test_repeated_prompt_is_answered_only_once, test_no_executable_readline_or_communicate_in_driver, test_suite_cases_and_prompts_frozen`) → Task 8B.3-R2 §9 forbids starting the formal suite,
so no case was run and no formal evidence exists.

## Task 8B.3-R3 — Final Harness Corrections and Formal Suite

### R3.1 ChatGPT R2 audit findings (four harness items)

1. repeated-prompt test invalid (fake child blocks on a second stdin read while the product contract answers a
   prompt type at most once);
2. AST test invalid (used `ast.get_docstring()` indiscriminately);
3. A3 frozen oracle not a literal (`"largest_to_above_of_to_nearest".replace(...)`);
4. output diff missed overwritten artifacts (set difference of names only).

### R3.2 Correction status: NOT APPLIED

The four corrections were **not** applied in this execution turn: the working session reached its context/time
budget immediately after reading this task book, and Task 8B.3-R3 §5/§6 leave no room for a partially applied or
unverified harness edit. Per §1 (executor only, no autonomous decisions) and §0 (always report/persist), the
harness, its tests and the frozen suite definition were left **untouched**:

- `scripts/task8b3_interactive_suite.py`: unchanged in this turn;
- `tests/test_task8b3_interactive_suite.py`: unchanged in this turn.

### R3.3 Gates

| gate | result |
|---|---|
| dedicated tests (Phase E) | NOT RUN in R3 (previous R2 state: 12 passed, 3 failed) |
| repository tests (Phase F) | NOT RUN |
| external `check_setup.py` (Phase G) | NOT RUN in R3 (last known: READY, live `ultralytics==8.4.164`) |
| six input hashes (Phase G) | NOT RE-VERIFIED in R3 (R1/R2 recorded identities unchanged) |
| FORMAL_SUITE_FREEZE (Phase H) | NOT RECORDED |
| formal suite (Phase I) | **NOT RUN** |

### R3.4 Case results

A1–B2: language `NOT RUN` / runtime `NOT RUN` (no case executed in R3).

### R3.5 Review pack (Phase K)

NOT CREATED.

### R3.6 Visual verdict

`PENDING CHATGPT/USER REVIEW` — no outputs exist.

### R3.7 No-tuning statement

No prompt was changed or retried; no RC1 product/runtime/`delivery_src`/`predict.py`/ProgramHead/Qwen
suggestion/Validator/detector/Reference/SAM2/D-B1/threshold/config/checkpoint was modified; no Assisted Mode,
`--reference-id` or `--inspect-proposals` was used; no training, download, package installation or final-test
access occurred.

### R3.8 Output-layout requirement

`ACCEPTED / DEFERRED TO TASK 8B.4` — still not implemented.

### R3.9 Exact STOP reason

Execution budget exhausted immediately after reading Task 8B.3-R3; the four mandated harness corrections (§5/§6)
were not applied and the §7/§8 gates were therefore not entered, so the formal six-image suite was not started.
Nothing was changed in the harness or tests to keep the frozen state intact and auditable.

## Task 8B.3-R4A — Harness Corrections Only

### R4A.1 Starting state

- branch `eval/task8b3-six-image-demo-suite`;
- starting HEAD `2562c3bc53b2600098824e93572cca291d334e33`;
- `main` = `origin/main` = `c45ecbec7fd293c454ccced22310db32c1542be4`;
- working tree contained only `M handoff/TO_DSH.md` (the task book).

### R4A.2 Exact corrections applied (harness + harness tests only)

1. **A3 frozen oracle** — `scripts/task8b3_interactive_suite.py` now stores the literal
   `largest_to_above_to_nearest`; the computed `"largest_to_above_of_to_nearest".replace(...)` expression is gone.
   No other case tuple changed.
2. **Changed/overwritten output detection** — new pure helper
   `changed_outputs(before, after)` reports a path when it is new **or** its stored `[size, mtime_ns]` differs;
   `run_case()` now uses that set for per-run artifact evidence *and* for locating this run's `result.json`
   (so an A1 run that overwrites a pre-existing artifact is still attributed correctly). Old outputs are never
   deleted or cleared.
3. **Repeated-prompt test** — the fake child now prints the direct confirmation substring twice consecutively
   before a **single** `stdin` read, then prints `ANSWER=<value>` and exits; assertions are: not timed out, exit 0,
   child receives `Y`, and the transcript contains **exactly one** `[driver] direct -> Y (DIRECT_CORRECT)` line.
   The driver was **not** changed to answer a prompt twice.
4. **AST test** — it now parses the module, locates only `_reader_thread` and `_run_interactive_process`, walks
   just those bodies and asserts no executable call named/attributed `readline` or `communicate`;
   `ast.get_docstring()` is no longer used.

Additionally, five pure `changed_outputs` unit tests were added (new path; same metadata → unchanged; different
size; same size with different `mtime_ns`; pre-existing `A1/result.json` whose metadata changed is included in
`changed["diagnostics"]`). No real external delivery file is used by those tests.

### R4A.3 Preserved behaviour

Reader-thread + queue design, `time.monotonic()` deadline, `CASE_TIMEOUT_SECONDS = 900`, incremental UTF-8
decoding, the three exact product prompt strings, `PROGRAM_RE`, `DISPLAY_TO_PROGRAM`, the exact three
child-environment overrides, the Y/N decision rules and the remaining five frozen case tuples are unchanged.

### R4A.4 Test gates

| gate | result |
|---|---|
| dedicated harness tests (`pytest tests/test_task8b3_interactive_suite.py -q`) | **20 passed** |
| full repository suite (`pytest tests/ -q`) | see R4A.6 |

### R4A.5 Formal suite

`NOT RUN BY DESIGN` — Task 8B.3-R4A §8 forbids running `scripts/task8b3_interactive_suite.py` against the real six
images in this task, and no review pack was created or modified. A1–B2 remain `NOT RUN`.

### R4A.6 Repository suite result

`pytest tests/ -q` → **1554 passed, 1 failed** on the first run. The single failure was
`tests/test_task6h_counterfactual_grounding.py::test_project_state_watt_wording_is_neutral_and_history_is_intact`,
which requires the string `Watt` in `handoff/FROM_DSH.md`; the R3 handoff rewrite had dropped that reference. Because
Task 8B.3-R4A §10 mandates updating `handoff/FROM_DSH.md` anyway, the required Watt reference was restored in that
same allowed file (no unrelated code was touched) and the affected test then passed:

```text
pytest tests/test_task6h_counterfactual_grounding.py -q -> PASS
```

### R4A.7 No product/delivery change

No RC1 product/runtime source, `delivery_src/BuildReasonSeg_Advisor_RC1/**`, external delivery source/config,
`predict.py`, ProgramHead, Qwen suggestion, Validator, detector, Reference, SAM2, D-B1, threshold, config,
checkpoint or model asset was modified; the six frozen prompts were never changed or retried; no Assisted Mode,
`--reference-id` or `--inspect-proposals` was used; no training, download, package installation or final-test
access occurred.

### R4A.8 Output-layout proposal

`ACCEPTED / DEFERRED TO TASK 8B.4` — not implemented here.

### R4A.9 Next action

Awaiting ChatGPT audit.


## Task 8B.3-R4A.1 — Full Repository Gate Confirmation

Single verification task: rerun the complete repository suite after the allowed `handoff/FROM_DSH.md` Watt wording
repair of Task 8B.3-R4A. Nothing else was executed or changed.

| item | value |
|---|---|
| starting HEAD | `d4b7d36c68be7cee6970b9f3d58cc1cd4f47caf1` |
| running branch | `eval/task8b3-six-image-demo-suite` |
| command | `ENV_PYTHON -m pytest tests/ -q` (run exactly once) |
| exact result | **1555 passed**, exit code 0 (~11.6 min) |
| harness code | UNCHANGED (`scripts/task8b3_interactive_suite.py` untouched) |
| harness tests | UNCHANGED (`tests/test_task8b3_interactive_suite.py` untouched) |
| other tests / product / delivery / canonical RC1 | UNCHANGED |
| formal six-image suite | NOT RUN BY DESIGN |
| A1–B2 | NOT RUN |
| review pack | NOT CREATED / unchanged |
| output-layout proposal | ACCEPTED / DEFERRED TO TASK 8B.4 |

The Watt handoff invariant (`tests/test_task6h_counterfactual_grounding.py`) that failed in R4A before the repair now
passes as part of this green suite; the previously skipped test set is unchanged.

Next action: Awaiting ChatGPT audit.
