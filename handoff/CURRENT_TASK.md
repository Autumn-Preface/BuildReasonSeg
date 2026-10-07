# CURRENT_TASK - TASK8E_USER_FACING_DEMO_PACKAGE_V1

Status:
IN_PROGRESS

Decision owner: ChatGPT Supervisor
Executor: CODEX
Next gate: CHATGPT_TASK8E_USER_DEMO_REMOTE_AND_LOCAL_DELIVERY_AUDIT

## Supervisor task authorization - verbatim

TASK AUTHORIZATION

Task ID:
TASK8E_USER_FACING_DEMO_PACKAGE_V1

Model:
GPT-6.1 Sol

Reasoning:
极高

Mode:
USER-FACING DELIVERY / NO SCIENTIFIC REPAIR

Accepted scientific predecessor:
TASK8D_ABOVE_TARGET_CHAIN_DIAGNOSIS_V1 = ACCEPT

Accepted predecessor branch:
diag/task8d-above-target-chain-v1

Accepted remote HEAD:
b569dcefa26af9788a9ec75eb52cbb8317305bea

Required new branch:
delivery/task8e-user-demo-v1

External delivery destination:
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V1

Next gate:
CHATGPT_TASK8E_USER_DEMO_REMOTE_AND_LOCAL_DELIVERY_AUDIT


# 1. Goal

Create a clean user-facing BuildReasonSeg Demo package.

This is NOT:

- a development handoff;
- a research audit package;
- a debugging interface;
- a new algorithm;
- a repair of Task8C/8D failures.

The package is intended for a normal local user who wants to:

1. choose an RGB aerial/satellite image;
2. enter a supported spatial-language instruction;
3. confirm the system interpretation;
4. run automatic inference;
5. view the resulting overlay and mask;
6. locate saved results.

The external package must be delivered at:

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V1


# 2. Frozen scientific behavior

The user Demo MUST reuse the accepted RC1 scientific runtime unchanged.

Scientific chain remains:

Qwen / ProgramHead
→ validator / optional Qwen suggestion
→ YOLO global proposals
→ automatic largest Reference selection
→ 512 reasoning context
→ P_dir
→ P_near
→ D-B1
→ dense mask

Do NOT change:

- detector
- RGB/BGR contract
- proposal merge
- reference selector
- eligibility
- relation fields
- nearest field
- W / A / q / C
- decoder
- checkpoint
- threshold
- final structural guards
- model assets
- program grammar
- supported semantic set

This task may improve presentation and interaction only.


# 3. User-facing semantic scope

Expose only the four accepted automatic tasks:

最大建筑 → 左侧 → 最近建筑
最大建筑 → 右侧 → 最近建筑
最大建筑 → 上方 → 最近建筑
最大建筑 → 下方 → 最近建筑

Natural-language text input MUST still go through the existing:

Qwen / ProgramHead → Validator

flow.

Do NOT create a keyword/regex shortcut that bypasses Qwen.

Recommended example buttons / quick-fill controls are allowed,
but they only fill the text box.

They must not directly set the program id.


# 4. No diagnostic user modes

The user-facing UI MUST NOT expose:

--reference-id
--inspect-proposals
assisted reference mode
manual proposal selection
threshold controls
ranking controls
model/checkpoint controls
debug map selection
training/evaluation controls

Automatic reference mode only.


# 5. GUI preference

First perform a read-only environment check:

<accepted Python> -c "import tkinter; print('TK_OK')"

If tkinter works:

implement a Windows Tkinter GUI.

Do NOT install any dependency.

If tkinter is unavailable:

implement a clean interactive console launcher instead and record:

GUI_UNAVAILABLE_EXISTING_ENV

Do not install PyQt, Gradio, Streamlit, Flask, customtkinter, tkinterdnd or any other package.

Preferred GUI is strongly recommended if available.


# 6. Required GUI workflow

The main screen should contain at minimum:

- title:
  BuildReasonSeg 建筑空间推理分割 Demo

- image path display
- “选择图片” button
- natural-language prompt text box
- four example / quick-fill buttons
- “开始分析” button
- status/progress display
- original-image preview
- result-overlay preview
- result summary
- “查看 Mask”
- “打开结果文件夹”
- “重新选择图片” or equivalent

Optional:
a small “查看使用说明” button.


# 7. User interaction semantics

Normal flow:

SELECT IMAGE
→ ENTER PROMPT
→ LANGUAGE PARSE
→ USER CONFIRMATION
→ INFERENCE
→ RESULT

If Qwen directly produces one of the four supported programs:

show a user-facing interpretation such as:

系统理解：
参考对象：最大建筑
方向关系：上方
目标关系：最近建筑

Then ask:

是否按此理解继续？

Only confirmation proceeds to inference.

If initial parse is unsupported but existing Qwen suggestion returns
a Validator-approved supported program:

show:

原始指令未能直接映射到当前支持的四类任务。

系统建议理解为：
“……”

Then ask the user to accept or reject.

Never auto-accept a suggestion.

If Qwen is unavailable or runtime fails:

preserve the existing fallback contract.

Use a user-facing confirmation such as:

语言模型当前不可用。
是否使用有限兼容模式继续？

Do not silently enter fallback.


# 8. User-facing status text

Do NOT expose false scientific success.

When runtime returns status SUCCESS, display wording equivalent to:

推理流程已完成。
结果已生成，请结合原图人工核验目标是否正确。

Optionally show:

运行状态：流程完成
语义正确性：未自动验证

Do NOT display:

识别正确
分割正确
AI 已成功找到正确建筑

unless ground truth exists, which this Demo does not use.


# 9. Friendly progress states

During execution show coarse user-facing states such as:

正在理解指令……
正在检测建筑……
正在确定参考建筑……
正在进行空间推理……
正在生成分割结果……

Do not invent precise progress percentages unless real progress is available.

The GUI must remain responsive during heavy inference.

Use a background worker/thread and communicate UI updates safely.

One Run button press must produce at most one inference attempt.

Disable duplicate Run actions while running.


# 10. Error handling

Translate common RC1 failures into concise Chinese user messages.

Examples:

unsupported image
no buildings detected
no valid reference
reasoning-context limitation
empty target mask
language parse unsupported
model/environment unavailable

Retain the original error code in a secondary/detail field.

Do not hide the error.

Do not automatically retry.


# 11. Result presentation

On runtime SUCCESS, show at minimum:

- original image
- overlay
- prompt
- interpreted supported task
- runtime status
- elapsed time if available

Provide buttons:

查看 Mask
打开结果文件夹

The result UI should not require the user to navigate `_engine/inference/output` manually.


# 12. Clean user result directory

Create at Demo root:

results/

For each completed run create a clean user result directory, for example:

results/
  20261007_183500_1008/
    overlay.png
    mask.png
    result_summary.json

The exact collision-safe naming may differ but must be deterministic/collision-safe.

For successful runs copy ONLY user-facing artifacts from the frozen engine run:

mask
overlay
a sanitized user summary JSON

Do not mutate the engine artifacts.

Do not pretend the copied summary is the original scientific result.json.

Suggested summary fields:

demo_version
timestamp
input_image
prompt
interpreted_program
language_mode
runtime_status
validity_scope
semantic_status
elapsed_seconds
mask_file
overlay_file
engine_run_root

Failure runs may save a small failure summary but must not create fake mask/overlay.


# 13. Internal engine layout

The user-facing delivery root should stay clean.

Preferred:

BuildReasonSeg_Demo_V1/
├─ 启动BuildReasonSeg Demo.bat
├─ BuildReasonSeg_Demo.py
├─ 使用说明.md
├─ 版本说明.txt
├─ results/
└─ _engine/
   ├─ buildreasonseg/
   ├─ configs/
   ├─ model/
   ├─ inference/
   ├─ logs/
   └─ other strictly required runtime files

The accepted RC1 engine may be copied/assembled inside `_engine`.

Do NOT expose development handoff material.


# 14. Explicitly forbidden in external user Demo

The final external Demo package MUST NOT contain:

handoff/
governance/
evaluation/
.git/
Git metadata
Task8C reports
Task8D reports
Supervisor task books
Codex executor states
historical research diagnostics
historical inference/output
developer test suites
research workspace source paths
private temporary files

Also do not include development-only:

train.py
evaluate.py
prepare_dataset.py

unless static dependency analysis proves one is unexpectedly required by runtime.

They are not user Demo features.


# 15. Clean runtime state

The new Demo package must start with:

empty results/
clean internal inference/output/
clean logs/

Do NOT copy the 1543+ historical RC1 outputs.

Do NOT copy Task8C formal outputs.

Do NOT copy A1/A2/A3/A4/B1/B2 historical diagnostics.

The existing accepted:

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1

must remain byte-for-byte unchanged.


# 16. Model assets

The new user Demo may copy the accepted model/runtime assets from:

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1

where needed.

Copied model asset SHA256 must match the accepted RC1 identities.

Do not use links/junctions/symlinks to the old RC1.

The new Demo should own its required asset files.

This supports folder-level movability.


# 17. Folder portability goal

Target:

PATH_PORTABLE_ON_CONFIGURED_MACHINE

Meaning:

- no code path may depend on the absolute location of BuildReasonSeg_Demo_V1;
- all Demo/engine/model/result paths resolve relative to the Demo root;
- the whole Demo folder may be moved to another directory on the same configured PC;
- no path into the research workspace is required for scientific assets/source.

Cross-machine zero-configuration portability is NOT required in V1 because Python/CUDA/runtime dependencies are not bundled.

Record this distinction clearly:

folder/source/model portability:
SUPPORTED

runtime environment portability:
NOT_BUNDLED_V1


# 18. Python runtime discovery

Do not bundle or copy the existing conda environment in this task.

The launcher should discover Python in this order or an equivalent deterministic order:

1. environment variable BUILDREASONSEG_PYTHON if set;
2. optional `<demo_root>\runtime\python.exe` if it exists in the future;
3. the currently validated BuildReasonSeg Python environment on this PC;
4. compatible `python` on PATH.

Before launch, perform a lightweight import/version check for the required runtime.

At minimum verify:

Python
torch
ultralytics == accepted frozen version
transformers
numpy
Pillow
opencv
scipy

If no compatible runtime is found:

show a clear Chinese message and STOP.

No installation.


# 19. Offline/cache isolation

The launcher/app must create writable cache directories relative to the Demo root or TEMP and explicitly configure:

YOLO_CONFIG_DIR
MPLCONFIGDIR
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONDONTWRITEBYTECODE=1
PYTHONUTF8=1
PYTHONIOENCODING=utf-8

Do not allow Ultralytics to create a new settings file in the Demo root unexpectedly.

Prefer:

<demo_root>\_runtime_cache\yolo
<demo_root>\_runtime_cache\matplotlib

or an equivalent private cache subtree.

The cache subtree is runtime-generated and not a scientific artifact.


# 20. Assembly reproducibility

In the research repository create an assembly/build script, for example:

scripts/build_task8e_user_demo.py

It must:

- construct the new package from accepted source/assets;
- create clean runtime directories;
- reject forbidden development content;
- verify copied asset hashes;
- emit a package manifest;
- never modify the source RC1.

Do not manually assemble an undocumented external folder only.


# 21. Repository-side source

Keep user-Demo source in a clear authorized repository location, for example:

demo/user_demo_v1/

or another concise equivalent.

It should contain:

GUI/launcher source
user README template
version metadata

Do not copy multi-GB model assets into Git.

The assembly script copies heavy assets only into the external package.


# 22. User documentation

External root must contain:

使用说明.md

Written for a non-developer.

Required sections:

1. Demo 是做什么的
2. 当前支持什么类型的图片
3. 当前支持的四类指令
4. 如何启动
5. 如何选择图片
6. 如何输入指令
7. 如何确认系统理解
8. 如何查看结果
9. Mask 和 Overlay 分别是什么
10. results 文件夹在哪里
11. 常见错误及处理
12. 当前能力边界
13. “流程完成”不等于“语义结果一定正确”
14. 如何关闭程序

Do not discuss Task IDs, governance, handoff, Codex or Supervisor in user docs.


# 23. Version file

External:

版本说明.txt

Keep it short.

Example content:

BuildReasonSeg Demo V1
User-facing local demonstration package

Core:
BuildReasonSeg Advisor RC1

Supported task:
largest building → direction → nearest building

Supported directions:
left / right / above / below

Runtime semantic result:
not automatically ground-truth validated


# 24. Demo UI must not disclose developer clutter by default

Do not dump:

raw proposals
full Python traceback
maps.npz
C/W/A internals
Qwen top5
checkpoint SHA
Git SHA
Task IDs

into the main UI.

A concise expandable technical error detail may show:

error code
short detail

Raw engine logs stay internal.


# 25. User-facing image handling

Use the same accepted image contract as RC1.

File picker filter:

PNG
JPG/JPEG
TIF/TIFF

Do not add new image-format support.

Preview resizing is presentation-only and must preserve aspect ratio.

Never modify the original user image.


# 26. Core call contract

Prefer importing/reusing the existing accepted runtime rather than shell-parsing CLI stdout.

UI-specific orchestration may wrap:

ProgramHeadRuntime
Validator
existing suggestion flow
PredictRuntime
PipelineRequest
predict_one

but must not duplicate/rewrite scientific algorithms.

If CLI helper functions are safely reusable, reuse them.

Any UI implementation of language confirmation/suggestion must preserve the same decision semantics:

direct supported → user confirmation
unsupported → Qwen suggestion → Validator → user confirmation
fallback → explicit user confirmation

No automatic Y.


# 27. Thread/process safety

The GUI must not freeze while heavy inference runs.

Use exactly one background worker for one inference request.

No parallel model runs.

Prevent double-click duplicate requests.

Do not terminate a live inference by destructive process kill.

On application close during inference:

either disable close until safe
or show a warning that current inference should finish.

Choose a conservative implementation.


# 28. Testing

Create repository tests for the user Demo.

At minimum verify:

- no handoff/governance/evaluation/test artifacts enter external package;
- assembly never modifies accepted RC1;
- heavy model assets are copied byte-identically;
- no absolute Demo-root dependency;
- runtime Python discovery deterministic;
- unsupported runtime produces friendly failure;
- four quick-fill examples only populate text;
- quick-fill does not bypass Qwen;
- user confirmation required before inference;
- Qwen suggestion confirmation required;
- fallback confirmation required;
- no reference-id;
- no inspect-proposals;
- no threshold/ranking/model UI;
- one click → at most one inference;
- duplicate Run disabled while busy;
- SUCCESS wording does not claim semantic correctness;
- failure creates no fake mask;
- results copy does not modify engine artifacts;
- clean package starts with empty results/output/logs;
- image preview is display-only;
- user docs contain no Task/Codex/Supervisor/handoff language.

Use mocks/fakes for GUI/backend tests where appropriate.

Tests must not perform model inference unless explicitly authorized below.


# 29. Real inference during Task8E

Do NOT rerun any Task8C locked Final Demo sample.

Do NOT use:

1003.tif
1008.tif
1009.tif
1010.tif

for Task8E smoke.

A real model inference is NOT required for the initial package build if UI/backend integration can be validated safely with mocks plus accepted RC1 runtime evidence.

If a real package smoke is genuinely needed:

select one deterministic NON-Task8C image already available locally;
record why it was selected;
perform at most ONE inference;
do not tune/retry;
do not use the outcome as scientific evidence.

If no suitable safe sample exists:
skip real inference and record:
REAL_MODEL_SMOKE_DEFERRED_TO_USER_ACCEPTANCE_TEST

Do not create a new research claim.


# 30. Folder-move smoke

Because source/model path portability is desired:

the package must not contain hardcoded Demo-root paths.

Perform a practical path-portability test if it can be done without duplicating all model bytes again.

Acceptable examples:

- assemble under a staging folder then rename/move the whole package to final destination and run self-check;
- or build at a temporary delivery name, verify, rename to final, verify again.

Do not use symlinks/junctions.

Success criterion:

the final application resolves `_engine`, models and results from its own location.

Do not claim cross-PC portability.


# 31. Self-check

Provide a user-accessible but simple self-check mechanism.

Could be:

“环境检查” button

or:

BuildReasonSeg_Demo.py --self-check

It should return user-facing:

环境检查：通过

or a clear failure explanation.

It must not require development files.


# 32. External delivery acceptance

At final external path verify:

- launcher exists
- app starts
- user documentation exists
- no forbidden developer directories/files
- model assets verify
- clean results initially
- clean engine output initially
- cache dirs private
- source paths relative
- existing Advisor_RC1 unchanged
- no Task8C/Task8D artifact modified


# 33. Required repository report

Create:

docs/task8e_user_demo_delivery_v1.md

This is a developer/audit report in the repository only.

It must NOT be copied into the user Demo.

Include:

scope
package tree
GUI/console choice
runtime discovery
portability status
asset provenance
tests
external package manifest
forbidden-content audit
existing RC1 preservation
known limitations
exact user workflow
final external path


# 34. Evidence JSON

Create:

evaluation/task8e_user_demo_delivery_v1.json

Include:

task_id
status
accepted_predecessor
branch/head
external_demo_path
gui_mode
python_runtime_discovery
package_manifest
model_asset_identity
source_asset_identity
forbidden_content_audit
portability_result
tests
real_model_smoke
advisor_rc1_unchanged
task8c_unchanged
task8d_unchanged
user_docs
user_workflow
known_limitations


# 35. Allowed repository changes

Only user-demo related files, for example:

demo/user_demo_v1/**
scripts/build_task8e_user_demo.py
tests/test_task8e_user_demo.py
docs/task8e_user_demo_delivery_v1.md
evaluation/task8e_user_demo_delivery_v1.json
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml

No product scientific source changes.
No governance changes.


# 36. External writes

Authorized:

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V1\**

and a temporary sibling staging folder used solely for package assembly.

Forbidden:

modifying
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\**

except read-only access.

Do not delete any pre-existing delivery.


# 37. Completion state

On success:

CURRENT_TASK Status:
READY_FOR_SUPERVISOR_AUDIT

EXECUTOR_STATE status:
READY_FOR_SUPERVISOR_AUDIT

next_gate:
CHATGPT_TASK8E_USER_DEMO_REMOTE_AND_LOCAL_DELIVERY_AUDIT

Commit + push repository evidence.

Then STOP.

Do not start Task8F or scientific repair.


# 38. Final executor summary

Report:

- exact external Demo path;
- GUI or console mode;
- whether folder is movable on the configured machine;
- what remains machine-specific;
- exact startup action;
- exact six-step user workflow;
- where results appear;
- how semantic uncertainty is communicated;
- package size;
- tests;
- whether real model smoke was performed;
- confirmation that old Advisor_RC1 and Task8C/8D evidence are unchanged.