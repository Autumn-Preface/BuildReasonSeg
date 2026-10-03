# TO_DSH — Task 8B.2-R2: Close RC1 Runtime Dependency and Readiness Gap

> Status: ACTIVE  
> Role boundary: ChatGPT decides; DSH executes only.  
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`  
> Branch: `audit/task8b2-rc1-runtime-closure`  
> Required starting HEAD: `7e885e176f23f301156d32f1800c5719165c8322`  
> Canonical RC1 source: `delivery_src/BuildReasonSeg_Advisor_RC1`  
> External runnable RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`  
> Runtime Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe`  
> Predecessor: Task 8B.2-R1.1 — canonical source policy cleanup accepted by ChatGPT.

---

# 0. Executor-only rule

You are the executor only.

ChatGPT has already frozen the diagnosis, required dependency version, files to edit, test procedure, A1 command, acceptance criteria and STOP conditions.

Do not make independent technical decisions.

Do not:
- choose another Python/Conda environment;
- install a different Ultralytics version;
- upgrade/downgrade torch, torchvision, transformers, numpy or OpenCV;
- modify ProgramHead, Qwen prompt, fallback UX or parser semantics;
- modify detector weights, detector thresholds, tiling, merge logic or reference selection;
- modify SAM2, relation fields, D-B1, checkpoints, model architecture or losses;
- retrain anything;
- access final test data;
- download a new model;
- alter A1 prompt to rescue a result;
- run A2/A3/A4/B1/B2;
- start Task 8C or external-image testing;
- merge to `main`;
- force-push;
- solve any unexpected model/inference failure by editing algorithm code.

If this task reaches a STOP condition, stop and report the exact evidence. Do not invent a workaround.

All user-facing DSH output must be Chinese. Code identifiers and commands remain English.

---

# 1. ChatGPT frozen audit decision

The following facts are already accepted and must not be re-decided:

1. External RC1 `buildreasonseg/runtime/detector.py` requires:

```python
from ultralytics import YOLO
```

2. The required frozen U-C1 runtime version is:

```text
ultralytics==8.4.164
```

3. The designated runtime environment currently has:
- Python 3.11.16;
- torch 2.13.0+cu132;
- CUDA available;
- NVIDIA GeForce RTX 5080 Laptop GPU;
- transformers 5.17.0;
- no importable Ultralytics before R2.

4. The old `check_setup.py` can report `READY` without validating the required Ultralytics runtime import. This defect remains:

```text
RC1-ENV-01
```

5. `environment.yml` currently declares:

```text
ultralytics>=8.3
```

6. `requirements.txt` currently declares:

```text
ultralytics>=8.3
```

7. `metadata.json` records model provenance:

```text
ultralytics = 8.4.164
transformers = 5.17.0
pytorch = 2.13.0+cu132 ...
```

8. Task 8B.2-R2 must:
- pin the RC1 Ultralytics runtime requirement to exactly 8.4.164;
- install exactly that package into the designated runtime environment if the dry-run safety gate passes;
- make `check_setup.py` validate the live Ultralytics runtime;
- clearly distinguish live runtime information from model provenance;
- correct the misleading portability wording;
- sync canonical source to the external RC1;
- run readiness regression tests;
- continue the **same A1** test with the same natural-language command;
- not perform new algorithm development.

---

# 2. Allowed canonical source edits

Only these RC1 canonical files may be edited for runtime functionality:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/environment.yml
delivery_src/BuildReasonSeg_Advisor_RC1/requirements.txt
delivery_src/BuildReasonSeg_Advisor_RC1/check_setup.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_setup_checker.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Do NOT edit `predict.py` in this task.

Reason: after the runtime is installed at the exact provenance version, the observed `ultralytics = 8.4.164` line is no longer false. Runtime/provenance distinction will be made explicitly by `check_setup.py`, without changing Demo inference UX.

Allowed repository documentation/handoff edits:

```text
docs/task8b2_r2_runtime_environment_closure.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Do not modify `handoff/PROJECT_STATE.md`.

No other path may change.

---

# 3. Phase A — Git safety gate

Run:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git branch --show-current
git rev-parse HEAD
git status --short
git remote -v
```

Continue only if:

```text
branch = audit/task8b2-rc1-runtime-closure
HEAD   = 7e885e176f23f301156d32f1800c5719165c8322
```

Allowed starting working tree:
- only `M handoff/TO_DSH.md`, if this R2 task book has been written there; or
- completely clean.

Any additional tracked/untracked file -> STOP.

Do not reset, stash, clean, discard, delete or switch branch.

STOP report:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b2_r2_stop_report.md
```

---

# 4. Phase B — Runtime precondition and reproducibility record

Use the exact interpreter below for every environment command:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe
```

For readability below:

```text
ENV_PYTHON = C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe
```

Do not substitute `python` from PATH.

Run and record:

```bat
ENV_PYTHON --version
ENV_PYTHON -c "import sys; print(sys.executable)"
ENV_PYTHON -m pip --version
ENV_PYTHON -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
ENV_PYTHON -c "import transformers; print(transformers.__version__)"
ENV_PYTHON -c "import numpy; print(numpy.__version__)"
ENV_PYTHON -c "import cv2; print(cv2.__version__)"
ENV_PYTHON -m pip show ultralytics
ENV_PYTHON -c "import ultralytics; print(ultralytics.__version__)"
```

The last two are expected to fail before installation.

## B1. Continue only if all are true

- Python major/minor = 3.11;
- `sys.executable` exactly belongs to `buildreasonseg-mvp`;
- torch = `2.13.0+cu132`;
- CUDA = True;
- GPU contains `NVIDIA GeForce RTX 5080 Laptop GPU`;
- transformers = `5.17.0`;
- Ultralytics is not importable / pip show cannot find it.

Record baseline versions of:
- torch;
- torchvision if importable;
- transformers;
- numpy;
- cv2 / opencv;
- scipy.

If Ultralytics is already importable before R2 -> STOP.
If any other frozen fact above differs -> STOP.

Do not repair the environment.

---

# 5. Phase C — pip dry-run safety gate

Run exactly:

```bat
ENV_PYTHON -m pip install --dry-run ultralytics==8.4.164
```

Save the complete dry-run output to the R2 report evidence notes.

## C1. Allowed dry-run effects

The dry-run may install:
- `ultralytics==8.4.164`;
- currently missing direct/transitive dependencies required by that package.

## C2. Mandatory STOP if dry-run proposes changing any currently installed core package

STOP if dry-run proposes upgrade, downgrade, uninstall or replacement of any of:

```text
torch
torchvision
torchaudio
transformers
numpy
opencv-python
opencv-python-headless
scipy
```

Do not install.
Do not choose another Ultralytics version.
Do not manually resolve.

Also STOP if pip reports an unsatisfied/conflicting resolution.

If the dry-run passes this gate, continue.

---

# 6. Phase D — Canonical dependency pin

Edit:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/environment.yml
```

Change only:

```text
ultralytics>=8.3
```

to:

```text
ultralytics==8.4.164
```

Also adjust the nearby explanatory comment only if necessary so it says this is the **frozen RC1 U-C1 runtime requirement**, not merely an approximate compatible range.

Do not change any other dependency version.

Edit:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/requirements.txt
```

Change only:

```text
ultralytics>=8.3
```

to:

```text
ultralytics==8.4.164
```

Do not change any other dependency line.

---

# 7. Phase E — Fix `check_setup.py`

Edit only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/check_setup.py
```

## E1. Add frozen constant

Add:

```python
REQUIRED_ULTRALYTICS = "8.4.164"
```

## E2. Add a live runtime dependency check

Add a function with equivalent semantics to:

```python
def check_runtime_dependencies(report: Report, package: ModelPackage | None) -> None:
    ...
```

It must verify the **live Python environment**, not read a version from metadata and pretend it is installed.

### Ultralytics rules

Attempt a real:

```python
import ultralytics
```

If import fails:

```text
[MISSING] Ultralytics runtime — U-C1 detector requires ultralytics==8.4.164
```

and `report.ready` must become false.

If import succeeds but:

```text
ultralytics.__version__ != "8.4.164"
```

output semantic equivalent of:

```text
[INVALID] Ultralytics runtime — runtime=<actual>, required=8.4.164
```

and `report.ready` must become false.

If exact, output semantic equivalent of:

```text
[OK] Ultralytics runtime — runtime=8.4.164; model provenance=8.4.164
```

The current live runtime must be obtained from the imported module.

If runtime != model provenance, the checker must be NOT READY.

### Transformers rules

Attempt a real:

```python
import transformers
```

If import fails:
- report `[MISSING] Transformers runtime`;
- NOT READY.

If import succeeds:
- report actual `transformers.__version__`;
- if model metadata has provenance, show it separately in the detail.

Do **not** enforce an exact Transformers version in R2.
Do **not** change the Transformers dependency specification.

## E3. Call order

`run_checks()` must call the new runtime dependency check after `check_model_package()` returns the package and before final readiness is computed.

## E4. Portability wording

Do not change the portability scan algorithm.

Replace the success wording:

```text
[OK] portability — 无 workspace 依赖；项目可整体移动
```

with wording semantically equivalent to:

```text
[OK] source/assets portability — delivery 内未发现 workspace 绝对路径依赖；runtime environment 需单独配置
```

Do not claim the Conda/Python environment is bundled into the delivery.

---

# 8. Phase F — Setup-checker regression tests

Modify only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_setup_checker.py
```

Preserve all existing tests.

Add at least these tests.

## F1. Exact real runtime version

On the actual RC1 runtime after installation, the real-project setup check must contain:

```text
Ultralytics runtime
8.4.164
BuildReasonSeg environment: READY
```

## F2. Missing Ultralytics regression

Without uninstalling or changing site-packages:

- import `check_setup.py` as a module in-process;
- monkeypatch Python import behavior so an attempt to import `ultralytics` raises `ModuleNotFoundError`;
- call only the new runtime dependency check with a fresh `Report`;
- verify:
  - `report.ready == False`;
  - a `[MISSING] Ultralytics runtime` line exists.

Do not uninstall the real package for this test.

## F3. Wrong Ultralytics version regression

Without modifying the real installation:

- monkeypatch the import result with a fake module whose `__version__` is not `8.4.164`;
- call the new runtime dependency check;
- verify:
  - `report.ready == False`;
  - the report contains `[INVALID] Ultralytics runtime`;
  - required version `8.4.164` is visible.

## F4. Portability wording regression

Verify setup output:
- contains `source/assets portability` or the exact equivalent implemented;
- does not contain `项目可整体移动`.

## F5. Runtime/provenance distinction

Verify the good runtime result shows:
- actual runtime Ultralytics version;
- model provenance version as a separate labeled concept.

Do not add tests that require final-test data, internet, training or new model assets.

---

# 9. Phase G — Refresh canonical source manifest

Because canonical source files changed, update:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

The manifest must remain at exactly:

```text
135 entries
```

Do not add or remove a manifest path in R2.

Only update `bytes` and `sha256` for canonical files whose contents actually changed:

```text
environment.yml
requirements.txt
check_setup.py
tests/test_setup_checker.py
```

If any other manifest-listed file changed -> STOP.

For all 135 entries verify:
- path exists;
- `bytes` equals actual size;
- `sha256` equals actual SHA256;
- entries remain lexicographically sorted;
- no duplicates;
- no absolute path;
- no `..`.

If entry count differs from 135 -> STOP.

---

# 10. Phase H — Install exact Ultralytics runtime

Only after Phase C dry-run passed.

Run exactly:

```bat
ENV_PYTHON -m pip install ultralytics==8.4.164
```

Then run:

```bat
ENV_PYTHON -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
ENV_PYTHON -m pip show ultralytics
ENV_PYTHON -m pip check
```

Re-record:

```bat
ENV_PYTHON -c "import torch; print(torch.__version__)"
ENV_PYTHON -c "import torchvision; print(torchvision.__version__)"
ENV_PYTHON -c "import transformers; print(transformers.__version__)"
ENV_PYTHON -c "import numpy; print(numpy.__version__)"
ENV_PYTHON -c "import cv2; print(cv2.__version__)"
ENV_PYTHON -c "import scipy; print(scipy.__version__)"
```

## H1. Required outcome

- Ultralytics = exactly `8.4.164`;
- `pip check` succeeds;
- all pre-existing recorded core versions from Phase B remain unchanged:
  - torch;
  - torchvision;
  - transformers;
  - numpy;
  - cv2/OpenCV;
  - scipy.

If any core version changed -> STOP immediately.
Do not attempt to roll back automatically.
Record exact before/after values for ChatGPT.

---

# 11. Phase I — Canonical tests before delivery sync

From repository root run:

```bat
ENV_PYTHON -m pytest tests/test_sync_advisor_rc1_delivery.py -q
```

Must PASS, including the 135-entry policy checks from R1.1.

If it fails because manifest hashes/sizes are stale, correct only the four expected changed manifest entries from Phase G and rerun.

Any other failure -> STOP.

---

# 12. Phase J — Synchronize canonical source to external RC1

First perform read-only check:

```bat
ENV_PYTHON scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

Because R2 intentionally changed four manifest-listed canonical files, exactly those files are expected to appear as MISMATCH before sync.

Required pre-sync check:
- no MISSING;
- mismatch set is exactly:

```text
environment.yml
requirements.txt
check_setup.py
tests/test_setup_checker.py
```

If any other mismatch exists -> STOP.

Then run the normal sync exactly once:

```bat
ENV_PYTHON scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Then rerun:

```bat
ENV_PYTHON scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

Required final result:

```text
checked=135
match=135
missing=0
mismatch=0
```

No external model asset may be deleted.
The sync helper must remain unchanged.

---

# 13. Phase K — True readiness verification

Enter external RC1:

```bat
cd /d C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Run with exact interpreter:

```bat
ENV_PYTHON check_setup.py
```

Required output semantics:

```text
[OK] Python
[OK] PyTorch — 2.13.0+cu132
[OK] CUDA
[OK] GPU — NVIDIA GeForce RTX 5080 Laptop GPU
[OK] Ultralytics runtime — runtime=8.4.164; model provenance=8.4.164
[OK] Transformers runtime — runtime=5.17.0; model provenance=5.17.0
...
[OK] source/assets portability — ... runtime environment ...
BuildReasonSeg environment: READY
```

Exact punctuation may differ; semantic content may not.

The old success phrase:

```text
项目可整体移动
```

must not appear.

If setup is NOT READY -> STOP.
Do not change algorithm/runtime code beyond the authorized source files.

---

# 14. Phase L — External RC1 tests

Run:

```bat
ENV_PYTHON -m pytest tests/test_setup_checker.py -q
```

Must PASS.

Then run the full external RC1 test suite:

```bat
ENV_PYTHON -m pytest tests -q
```

Must PASS with no reduction caused by R2.

Then return to repository root and run:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
ENV_PYTHON -m pytest tests/ -q
```

The repository suite must PASS.

Do not modify unrelated tests to obtain green status.

Any unrelated failure -> STOP and report.

---

# 15. Phase M — Same A1 positive path

This is continuation of the already-authorized user Demo test, not a new research experiment.

Do not use another image.
Do not change the prompt.
Do not change thresholds/config/model.

Confirm this file exists:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\A1.png
```

If missing -> STOP.

Record a pre-run listing/timestamp snapshot of:
- `inference/output/masks`
- `inference/output/overlays`
- `inference/output/diagnostics`

Run exactly from external RC1:

```bat
(echo Y)| ENV_PYTHON predict.py ^
  --image inference/input/A1.png ^
  --prompt "找出最大的建筑，然后把它右边离它最近的那栋分割出来" ^
  --confirm-command
```

If piping does not satisfy the existing Y/N input on Windows, use an equivalent deterministic stdin method that supplies exactly one `Y`. Do not edit code to automate confirmation.

## M1. Required language result

The initial Qwen / ProgramHead parse must be:

```text
largest -> right_of -> nearest
largest_to_right_of_to_nearest
```

This is a regression confirmation only.

If language parse differs -> STOP.
Do not change prompt or parser.

## M2. Required runtime result

After Y:
- U-C1 detector must actually execute;
- no `ModuleNotFoundError: ultralytics`;
- Reference resolution must execute;
- SAM2 must execute;
- D-B1 must execute;
- final result must reach `SUCCESS`.

Required new artifacts:
- mask;
- overlay;
- diagnostics/result data.

Record exact newly generated paths.

If any stage fails:
- STOP;
- do not alter detector/SAM2/D-B1/Reference/thresholds;
- record stage and exact error/log path for ChatGPT.

Do not inspect the image result to tune anything in R2.
R2 only verifies that the pipeline executes after environment closure.

---

# 16. Phase N — Same A1 negative confirmation path

Record another pre-run output snapshot.

Run the same A1 and same prompt:

```bat
(echo N)| ENV_PYTHON predict.py ^
  --image inference/input/A1.png ^
  --prompt "找出最大的建筑，然后把它右边离它最近的那栋分割出来" ^
  --confirm-command
```

Supply exactly one `N`.

Required:
- initial parse remains `largest_to_right_of_to_nearest`;
- user rejection exits through the existing user-abort behavior;
- U-C1 detector must not load/run;
- SAM2 must not run;
- D-B1 must not run;
- no new final mask/overlay may be generated by this N run.

Do not change the existing error protocol.

If the N run executes detector/core or creates final mask/overlay -> STOP.

---

# 17. Phase O — Git diff safety gate

Return to repository:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git status --short
git diff --check
git diff
```

Allowed changed repository paths only:

```text
handoff/TO_DSH.md
delivery_src/BuildReasonSeg_Advisor_RC1/environment.yml
delivery_src/BuildReasonSeg_Advisor_RC1/requirements.txt
delivery_src/BuildReasonSeg_Advisor_RC1/check_setup.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_setup_checker.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
docs/task8b2_r2_runtime_environment_closure.md
handoff/FROM_DSH.md
```

No other repository path may change.

Do not commit:
- A1 image;
- masks/overlays/diagnostics;
- logs;
- Conda environment;
- installed packages;
- model assets;
- checkpoints.

Any other repository change -> STOP.

---

# 18. Phase P — Required audit report

Create:

```text
docs/task8b2_r2_runtime_environment_closure.md
```

Required sections:

## 1. Task
`Task 8B.2-R2 — Close RC1 Runtime Dependency and Readiness Gap`

## 2. ChatGPT Frozen Decision
Record:
- required Ultralytics = 8.4.164;
- designated environment;
- no algorithm changes;
- same A1 continuation.

## 3. Initial Runtime State
Record:
- Python;
- executable;
- pip;
- torch;
- CUDA;
- GPU;
- transformers;
- numpy;
- OpenCV;
- scipy;
- Ultralytics missing evidence;
- old check_setup false-positive state if reproduced.

## 4. pip Dry-run
Record:
- exact command;
- packages proposed;
- explicit statement that no protected core package replacement was proposed.

## 5. Dependency Specification
Record:
- `environment.yml`: `ultralytics==8.4.164`;
- `requirements.txt`: `ultralytics==8.4.164`;
- no other dependency version changed.

## 6. RC1-ENV-01 Fix
Describe:
- old problem;
- new live import/version check;
- missing-dependency behavior;
- wrong-version behavior;
- readiness result.

## 7. Runtime vs Model Provenance
Record actual live:
- torch;
- ultralytics;
- transformers.

Record metadata provenance separately.

State explicitly that metadata is provenance, not proof of an installed package.

## 8. Portability Wording
Record new wording and state:
- source/assets path portability verified;
- Python/Conda runtime remains separately configured.

## 9. Installation Before/After

| package | before | after | changed? |
|---|---|---|---|
| torch | | | |
| torchvision | | | |
| transformers | | | |
| numpy | | | |
| OpenCV | | | |
| scipy | | | |
| ultralytics | missing | 8.4.164 | expected |

## 10. Canonical/Delivery Sync
Record:
- manifest count = 135;
- exact four pre-sync mismatches;
- post-sync 135/135.

## 11. Validation

| Check | Result |
|---|---|
| dry-run safety gate | PASS/FAIL |
| ultralytics import 8.4.164 | PASS/FAIL |
| pip check | PASS/FAIL |
| protected packages unchanged | PASS/FAIL |
| missing-ultralytics regression | PASS/FAIL |
| wrong-version regression | PASS/FAIL |
| true check_setup READY | PASS/FAIL |
| portability wording | PASS/FAIL |
| canonical sync tests | PASS/FAIL |
| external setup tests | PASS/FAIL |
| external full suite | PASS/FAIL |
| repository full suite | PASS/FAIL |
| A1 Qwen parse | PASS/FAIL |
| A1 Y-path full inference | PASS/FAIL |
| A1 N-path early abort | PASS/FAIL |

## 12. A1 Positive Path
Record:
- exact unchanged prompt;
- parse;
- detector reached;
- Reference reached;
- SAM2 reached;
- D-B1 reached;
- final status;
- generated mask/overlay/diagnostics paths.

Do not claim visual semantic correctness beyond what was actually inspected.
This task is execution closure, not Demo-quality scoring.

## 13. A1 Negative Path
Record:
- parse;
- N rejection behavior;
- detector/core not executed;
- no new final mask/overlay.

## 14. RC1 Final Environment State
Use only:
- `READY` if all R2 runtime gates pass;
- `NOT READY` otherwise.

## 15. Remaining Issues
Only factual remaining issues.
Do not propose a new algorithm.

## 16. Scientific Scope
Include:

> Task 8B.2-R2 did not modify training, checkpoints, detector thresholds, tiling/merge policy, ProgramHead semantics, Qwen prompt template, Reference policy, SAM2, relation fields, D-B1, datasets, final-test results or scientific claims. It only closed the RC1 runtime dependency/readiness gap and continued the already-authorized A1 Demo smoke.

---

# 19. Phase Q — Update handoff

Update:

```text
handoff/FROM_DSH.md
```

Preserve the existing required `ARTIFACT-FACTS` block verbatim.

Below it write current engineering handoff:

- Task = `8B.2-R2`;
- branch;
- starting HEAD;
- Ultralytics runtime;
- check_setup;
- external sync;
- external test suite;
- repository test suite;
- A1 parse;
- A1 Y-path;
- A1 N-path;
- report path;
- RC1 status;
- next action:

```text
Awaiting ChatGPT audit. Do not start Task 8C or additional Demo images.
```

Do not modify `handoff/PROJECT_STATE.md`.

---

# 20. Phase R — Commit

Run:

```bat
git status --short
git diff --check
```

Stage allowed files individually.

Do not use:

```text
git add .
git add -A
```

Verify:

```bat
git diff --cached --name-only
```

Commit exactly:

```text
fix(rc1): close runtime dependency readiness gap
```

After commit:

```bat
git status --short
git show --stat --oneline HEAD
```

Working tree must be clean.

If not clean -> STOP and do not push.

---

# 21. Phase S — Push

Push:

```bat
git push origin audit/task8b2-rc1-runtime-closure
```

No force push.

Record:

```bat
git rev-parse HEAD
git status --short
```

---

# 22. Completion gate

Declare `TASK 8B.2-R2 COMPLETE` only if all are true:

1. correct branch/head used;
2. pre-R2 environment exactly matched frozen state;
3. pip dry-run passed protected-package gate;
4. only `ultralytics==8.4.164` was intentionally added for the missing runtime dependency;
5. protected core package versions remained unchanged;
6. `pip check` passed;
7. environment.yml exact pin is present;
8. requirements.txt exact pin is present;
9. check_setup performs a real Ultralytics import;
10. missing Ultralytics -> NOT READY regression passes;
11. wrong Ultralytics version -> NOT READY regression passes;
12. live runtime/provenance distinction is explicit;
13. portability wording no longer claims bundled runtime portability;
14. source manifest remains exactly 135 and valid;
15. external delivery final sync check is 135/135;
16. `check_setup.py` reports true READY;
17. external RC1 tests pass;
18. repository tests pass;
19. same A1 parse is exactly `largest_to_right_of_to_nearest`;
20. A1 Y path reaches actual final SUCCESS with mask/overlay/diagnostics;
21. A1 N path exits before detector/core and creates no final mask/overlay;
22. no algorithm/research/checkpoint/final-test change occurred;
23. report completed;
24. handoff completed with ARTIFACT-FACTS preserved;
25. exact commit created;
26. push succeeded;
27. working tree clean;
28. DSH stops.

If any item fails:

```text
TASK 8B.2-R2 PARTIAL
```

Do not enter Task 8C.

---

# 23. Final DSH reply

Output only:

```text
TASK 8B.2-R2 COMPLETE / PARTIAL

Branch:
audit/task8b2-rc1-runtime-closure

Commit:
<sha or NONE>

Push:
PASS / NOT DONE

Ultralytics:
before = MISSING / other
after  = 8.4.164 / other

Protected core packages unchanged:
PASS / FAIL

pip check:
PASS / FAIL / NOT RUN

Canonical manifest:
135 / other

External delivery sync:
135/135 PASS / FAIL / NOT RUN

check_setup:
READY / NOT READY / NOT RUN

Missing-Ultralytics regression:
PASS / FAIL / NOT RUN

Wrong-version regression:
PASS / FAIL / NOT RUN

External RC1 tests:
<result>

Repository tests:
<result>

A1 parse:
largest_to_right_of_to_nearest / other / NOT RUN

A1 Y-path:
PASS / FAIL / NOT RUN

A1 Y outputs:
mask = <path or NONE>
overlay = <path or NONE>
diagnostics = <path or NONE>

A1 N-path:
PASS / FAIL / NOT RUN

RC1-ENV-01:
FIXED / NOT FIXED

RC1:
READY / NOT READY

Report:
docs/task8b2_r2_runtime_environment_closure.md / NOT CREATED

Handoff:
handoff/FROM_DSH.md / NOT UPDATED

STOP reason:
<none or exact reason>

等待 ChatGPT 审核；不得进入 Task 8C、A2/A3/A4/B1/B2 或任何新研发。
```

Then stop.
