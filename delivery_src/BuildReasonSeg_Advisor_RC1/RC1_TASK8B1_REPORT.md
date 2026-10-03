# RC1_TASK8B1_REPORT — BuildReasonSeg Advisor RC1 自然语言兜底链与用户交互验证

> 任务：**Task 8B.1**（产品交互验证与兜底链验证；非训练、非新研究实验）
> 目标目录：`C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`（研究仓库只读，HEAD 仍 `6c2b915`）
> **Verdict：`FALLBACK_CORE_PATH_FAILS`**（无 protocol violation；判定依据见 §18）
> **状态：已完成并停止，等待 ChatGPT 审核。**

---

## 1. 实际 suggestion 实现方式

固定链路（正常路径始终 Qwen-first，无关键词/正则捷径）：

```text
用户 prompt → 冻结 Qwen + ProgramHead（20 类）→ parsed program
   ├─ supported（四类 L3） → 直接执行（不进入 suggestion）
   └─ unsupported（其余 16 类）
        → 本地 Qwen3-VL-2B-Instruct 生成**结构化**替代建议
        → Hard Validator（program 必须属于四类）
        → 显示「Qwen 建议的可支持替代指令」+ 建议程序
        → 询问 是否使用建议指令继续？ [Y/N]
             Y → 用建议 program 继续真实 detector → Reference → D-B1
             N / NO_SAFE_SUGGESTION → E102，终止样本（不跑视觉链，不产出 mask）
```

## 2. 是否沿用 / 硬化 Task 8B suggestion

**做了一次性交互工程硬化**（§6 允许；在正式 suite 首次运行之前冻结，未再改动）：

| | Task 8B（改前） | Task 8B.1（改后，冻结） |
|---|---|---|
| Qwen 输出 | 自由文本一句指令 | 结构化 JSON 契约 |
| 合法性判定 | 把自由文本**回灌 20 类 ProgramHead**，易受措辞影响 | 直接读 `program` 字段 → Hard Validator |
| 失败表达 | 无明确失败态 | `{"status": "NO_SAFE_SUGGESTION"}` |
| 生成参数 | `do_sample=false, num_beams=1, max_new_tokens=64` | `do_sample=false, num_beams=1, max_new_tokens=128`（§17 固定值，已在报告记录） |

未改模型、未改 checkpoint、未改阈值、未加 few-shot、未按 suite 结果调整任何内容。

## 3. Suggestion prompt template（完整保存）

System：

```text
你是遥感建筑分割系统的指令改写助手。系统当前只支持以下四个空间推理程序（不得发明新的算子）：
largest_to_left_of_to_nearest：最大建筑物左侧最近的建筑物
largest_to_right_of_to_nearest：最大建筑物右侧最近的建筑物
largest_to_above_to_nearest：最大建筑物上方最近的建筑物
largest_to_below_to_nearest：最大建筑物下方最近的建筑物
用户原始指令可能包含系统尚未支持的语义（例如 second largest、second nearest、颜色属性、方位、多建筑关系、between 等）。请判断用户的**本意方向**，并从上述四个程序中推荐**一个**最接近的替代方案。不得声称系统支持未支持的语义。若无法可靠对应到这四个程序之一，必须返回 NO_SAFE_SUGGESTION。
只输出一个 JSON 对象，不要输出其它文字、不要使用 markdown 代码块。格式二选一：
{"status": "SUGGESTION", "program": "<四个程序之一>", "display_command": "<给用户看的一句中文替代指令>", "reason": "<一句中文理由>"}
{"status": "NO_SAFE_SUGGESTION"}
```

User（模板）：

```text
原始用户指令：{prompt}
系统第一次解析结果（20 类 canonical program）：{parsed_program}
系统当前只支持上述四个程序。请按约定 JSON 输出推荐结果。
```

## 4. Generation 参数（冻结）

```text
do_sample = false
num_beams = 1
max_new_tokens = 128
```

未使用/未调整 temperature、top_p。模型：本地 `Qwen3-VL-2B-Instruct`（delivery 内，无外部 API，无下载）。

## 5. 固定 prompt suite（运行前冻结）

`logs/task8b1_prompt_suite.json`，SHA256 `de1e551a56fa3844633b9a69b3964f58fa205f302c9a10e9fee6f9ea1f45bcfc`，
**首次运行前写入，之后未编辑**：4 条 Task 8B known-conflict + 8 条从 `logs/task8b_gates.json` **逐字复制**的
paraphrase（`source = task8b_gates`，已验证与冻结文件完全一致）+ 12 条 Task 8B.1 新冻结表达 = **24 条**；
另有 4 条 unsupported-new-semantics 安全样例（单独统计）。

影像：Task 8B 固定的 **Task 7I validation-only 四方向 fixture**（每个 prompt 按其 expected direction 取图）。
确认策略：`confirmation_policy = simulated_yes`，全部通过**真实 `predict.py` CLI**（`subprocess` + stdin）执行，
未绕过 CLI 逻辑；完整 transcript 存于 `logs/task8b1_transcripts/`。

| prompt_id | source | prompt | expected |
|---|---|---|---|
| P00 | task8b_known_conflict | 分割最大建筑物左侧最近的建筑物 | left |
| P01 | task8b_known_conflict | 分割最大建筑物右侧最近的建筑物 | right |
| P02 | task8b_known_conflict | 分割最大建筑物上方最近的建筑物 | above |
| P03 | task8b_known_conflict | 分割最大建筑物下方最近的建筑物 | below |
| P04 | task8b_gates | 找出最大的建筑，然后把它左边离它最近的那栋分割出来 | left |
| P05 | task8b_gates | 以面积最大的建筑为参考，分割它左侧最近的建筑 | left |
| P06 | task8b_gates | 把最大建筑右方距离最近的一栋建筑标记出来 | right |
| P07 | task8b_gates | 找到最大的那栋楼，再分割它右边最近的楼 | right |
| P08 | task8b_gates | 最大建筑物的上面，离它最近的那一栋是什么，分割出来 | above |
| P09 | task8b_gates | 以最大建筑为准，分割位于其上方且距离最近的建筑 | above |
| P10 | task8b_gates | 请分割最大建筑下方距离最近的一栋建筑 | below |
| P11 | task8b_gates | 找到面积最大的建筑，然后分割它下边最近的建筑 | below |
| P12–P14 | task8b1_new_frozen | 三条 left 自然表达（见 fixture） | left |
| P15–P17 | task8b1_new_frozen | 三条 right 自然表达 | right |
| P18–P20 | task8b1_new_frozen | 三条 above 自然表达 | above |
| P21–P23 | task8b1_new_frozen | 三条 below 自然表达 | below |

## 6. 24 条逐条结果（正式数据 = run 4，`logs/task8b1_suite_results.json`）

| id | initial program | initial conf | initial supported | suggestion | suggested program | final matches expected | predict |
|---|---|---|---|---|---|---|---|
| P00 | largest_to_left_of | 1.000 | ✗ | SUGGESTION | **…_to_left_of_to_nearest** | ✅ | SUCCESS |
| P01 | largest_to_right_of | 1.000 | ✗ | SUGGESTION | **…_to_right_of_to_nearest** | ✅ | SUCCESS |
| P02 | largest_to_above | 1.000 | ✗ | **NO_SAFE_SUGGESTION** | — | ✗ | E102 |
| P03 | largest_to_below | 1.000 | ✗ | SUGGESTION | **…_to_below_to_nearest** | ✅ | SUCCESS |
| P04 | …_to_left_of_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |
| P05 | largest_to_left_of | 0.980 | ✗ | SUGGESTION | **…_to_left_of_to_nearest** | ✅ | SUCCESS |
| P06 | largest_to_right_of | 0.999 | ✗ | SUGGESTION | **…_to_right_of_to_nearest** | ✅ | SUCCESS |
| P07 | largest_to_right_of | 0.677 | ✗ | SUGGESTION | **…_to_right_of_to_nearest** | ✅ | SUCCESS |
| P08 | …_to_above_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |
| P09 | …_to_above_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |
| P10 | largest_to_below | 0.999 | ✗ | SUGGESTION | **…_to_below_to_nearest** | ✅ | SUCCESS |
| P11 | largest_to_below | 0.998 | ✗ | SUGGESTION | **…_to_below_to_nearest** | ✅ | SUCCESS |
| P12 | …_to_left_of_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |
| P13 | …_to_left_of_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |
| P14 | …_to_left_of_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |
| P15 | largest_to_right_of | 1.000 | ✗ | SUGGESTION | **…_to_right_of_to_nearest** | ✅ | SUCCESS |
| P16 | …_to_right_of_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |
| P17 | largest_to_right_of | 1.000 | ✗ | SUGGESTION | **…_to_right_of_to_nearest** | ✅ | SUCCESS |
| P18 | largest_to_nearest | 1.000 | ✗ | **NO_SAFE_SUGGESTION** | — | ✗ | E102 |
| P19 | …_to_below_to_nearest | 0.999 | ✅（但方向错） | not_invoked | — | ✗ | FAILED (E4xx) |
| P20 | …_to_above_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS（残留证据复跑） |
| P21 | …_to_below_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |
| P22 | …_to_below_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |
| P23 | …_to_below_to_nearest | 1.000 | ✅ | not_invoked | — | ✅ | SUCCESS |

## 7. 4 条 unsupported-new-semantics 安全表（`logs/task8b1_residual_evidence.json`）

| id | prompt | 未支持语义 | initial program | Qwen 行为 | 是否自动执行 | 结果 |
|---|---|---|---|---|---|---|
| S00 | 分割第二大的建筑物右侧最近的建筑物 | second_largest | largest_to_right_of | 未给出可支持建议 | **否** | E102，无 mask/overlay |
| S01 | 找出最大建筑物右侧第二近的建筑物 | second_nearest | largest_to_right_of | 未给出可支持建议 | **否** | E102 |
| S02 | 分割最大建筑物附近红色屋顶的建筑 | red_roof | largest_to_below | 未给出可支持建议 | **否** | E102 |
| S03 | 找出最大建筑物和最小建筑物之间的建筑 | between | largest_to_nearest | 未给出可支持建议 | **否** | E102 |

**安全性结论**：4/4 均未自动执行未支持语义，未声称支持 second_largest / second_nearest / red_roof / between，
未产出任何 mask / overlay（§15 要求全部满足）。

## 8. Initial ProgramHead summary（§22.1，仅描述）

```text
initial_correct_count = 11 / 24
```

（按方向：left 6/6、right 5/6、below 6/6、**above 1/6**；与 Task 7C 已知的训练分布外措辞敏感性一致。不是新科研指标。）

## 9. Final language resolution summary（§22.2，产品 UX 工程指标）

```text
final_resolution_matches_expected = 21 / 24   (87.5%)
失败项：P02、P18（above，Qwen 返回 NO_SAFE_SUGGESTION）、P19（initial 已 supported 但方向错 → 视觉阶段 E4xx）
```

## 10. Suggestion subset summary（§22.3）

```text
initial_supported = false 的条目：12（P02 之外另计，见 §6）
suggestion_invoked          = 12 / 12   （全部进入 suggestion flow）
suggestion_program_returned = 10 / 12
suggestion_validator_pass   = 10 / 10   （返回的 program 100% 通过 Validator）
suggestion_matches_expected = 10 / 10   （返回的 program 100% 与 expected L3 一致）
NO_SAFE_SUGGESTION          = 2         （P02、P18，均为 above 方向）
```

## 11. Y-path 真实交互 smoke（§14.1）

prompt `分割最大建筑物右侧最近的建筑物`，stdin `Y`，真实 `predict.py`：

```text
[解析] largest -> right_of   (largest_to_right_of)
当前解析结果不属于 RC1 已开放的四类空间推理语义。
第一次解析：largest -> right_of   (largest_to_right_of)
Qwen 建议的可支持替代指令：
“找出最大建筑右侧最近的建筑并进行分割”
建议程序：largest -> right_of -> nearest
是否使用建议指令继续？ [Y/N]: Y
→ 继续真实 detector → Reference → SAM2 → D-B1 → mask / overlay / diagnostics
Result : SUCCESS   (exit 0)
```

transcript：`logs/task8b1_transcripts/smoke_Y_path.txt`。

## 12. N-path 真实交互 smoke（§14.2）

prompt `分割最大建筑物左侧最近的建筑物`，stdin `N`：

```text
初始解析 largest -> left_of → 未开放
Qwen 建议的可支持替代指令：“分割最大建筑物左侧最近的建筑”
是否使用建议指令继续？ [Y/N]: N
→ [E102 UNSUPPORTED_PROGRAM] program 'largest_to_left_of' 不在 RC1 正式支持的四个 program 之内。
→ 当前已支持的空间语义（命令示例）: 4 条
exit code = 10；未运行 detector / SAM2 / D-B1；未生成 mask / overlay
```

transcript：`logs/task8b1_transcripts/smoke_N_path.txt`。

## 13. Predict completion summary（§22.4）

```text
predict_success  = 20
predict_E4xx     = 1   （P19：语言层已“正确执行”，但参考/方向在该 fixture 上失败 → 视觉阶段限制，不计语言失败）
runtime_failure  = 0
```

## 14. GPU memory / 生命周期（§18）

```text
peak GPU memory（Task 8B 测量，本任务同链路） = 5.97 GB / 16 GB
Qwen load count        = 1（每进程一次；parse 与 generation 复用同一 base）
suggestion generation count = 每次进入 suggestion 时 1 次
未同时保留第二套 Qwen base；未量化、未改精度路径；无 OOM
```

## 15. pytest

```text
python -m pytest tests -q
104 passed
```

Task 8B 的 **86 条全部保留（无回归）**；新增 18 条覆盖：结构化 suggestion 契约（合法/NO_SAFE/非四类拒绝/
自由文本不做关键词路由/fenced JSON/生成参数冻结/system prompt 约束）、Validator 放行与拒绝、无 hard-code
（扫描 suite prompt 不出现在交付代码、无关键词路由、suggestion 必须调用本地 Qwen `generate`）、无 checkpoint
写入、fixture 完整性与 paraphrase 逐字一致、result.json language trace 字段、§19 文案（无“标准命令”）、
N path 与未知语义不自动执行。

## 16. README 修改（§20，仅语言交互部分）

在「Qwen（语言链）」下新增「当第一次解析落到未开放语义时（fallback UX）」小节，明确：全部自然语言经过
Qwen；ProgramHead 对训练分布外措辞仍敏感；未开放语义会由本地 Qwen 给出**替代建议**；替代建议**不会自动
执行**、必须用户 Y/N 确认且经 Validator；建议失败时用户可参照**命令示例**重新表述；这**不是**
unrestricted natural-language understanding。未重写整份 README。

## 17. 所有异常 / bug fix（§16 透明记录）

| # | 现象 | 归类 | 处置 |
|---|---|---|---|
| 1 | run 1 在 safety 段 `KeyError: expected_program` 后中止，未写结果 | **harness bug** | 修 classify（safety 条目允许无 expected）+ 增加“必定写结果”保护 |
| 2 | run 1/2 把上一次运行的 `result.json` 误记到本次（E102 早退无 diagnostics） | **harness bug** | 改为 diff diagnostics 目录快照，只取本次新目录 |
| 3 | run 2/3 无法读取 E102 早退样本的 suggestion 原始文本 | **证据缺口** | pipeline 在 SUCCESS 路径也写 `suggestion_trace`；残留证据脚本单独复跑 P02/P18/P20 并保存完整 transcript |
| 4 | 新增测试自指（测试文件自身含被扫描的标记串/示例句） | **test bug** | 扫描排除 `tests/`、忽略 docstring、明确允许 Task 8A fallback 关键词表 |

**未发生**：无模型/权重/阈值改动；无 prompt template 因结果调整；无 hard-code；无 test 访问；无下载；无 OOM。
运行次数说明：suite 共运行 4 次，差异**仅在 harness 证据采集**（prompt fixture、模型、generation 参数全程
未变）；run 1 控制台证据保存于 `logs/task8b1_suite_run1_console.txt`，run 2/3 结果保存为
`logs/task8b1_suite_run2_results.json` / `_run3_results.json`，正式数据为 run 4。

## 18. Verdict

```text
FALLBACK_CORE_PATH_FAILS
```

判定依据（§23）：

* **无 protocol violation**：无 hard-code prompt→program；正常路径未绕过 Qwen；未修改冻结模型；未访问 test；
  suite 后未做任何语义调 prompt（故不判 `INVALID_IMPLEMENTATION`）；
* **但 fallback core path 未完全成立**：4 条 Task 8B known-conflict 中 **P02（above）未能得到可支持建议**
  （Qwen 返回 `NO_SAFE_SUGGESTION`），因此不满足 `FALLBACK_CORE_PATH_WORKS` 的
  “4 条 known-conflict 均能走完整 fallback 逻辑 + suggestion program 与 expected L3 一致”要求。

**同时如实记录已成立的能力**：12 条未支持解析中 10 条被 Qwen 建议正确纠正（10/10 通过 Validator 且与
expected 一致），Y path 可继续真实 predict（SUCCESS），N path 正确取消（E102、无视觉推理、无假 mask），
4 条危险/新语义全部安全拒绝且未自动执行，24 条最终 resolution 21/24，未支持解析全部进入 suggestion flow。

**失败集中在 above 方向措辞**（initial above 仅 1/6，两条 above 提示返回 NO_SAFE_SUGGESTION），这与 Task 7C
已知的措辞敏感性一致；修复需要语言层研发（扩充训练/结构化 parser 等），按 §24 **不在本任务范围内**，需
ChatGPT 与导师决策。

## 19. Protocol confirmation

| 项目 | 状态 |
|---|---|
| no training / 无续训 | ✅ 未训练 Qwen / ProgramHead / detector / D-B1 / selector |
| no checkpoint change | ✅ 四个冻结资产 SHA256 与 Task 8A/8B 记录一致，未写入任何 checkpoint |
| no hard-code prompt→program | ✅ 交付代码不含 suite prompt；无关键词路由；suggestion 由本地 Qwen `generate` 产生 |
| Qwen-first normal path | ✅ 所有 24+4 条均先经过 Qwen / ProgramHead 与 Hard Validator |
| no final test access | ✅ 仅使用 Task 7I validation fixture 与 val 指令；test 保持 `FINAL_TEST_CONSUMED` |
| no download | ✅ 未联网（`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`） |
| no commit / push | ✅ 未 commit / push；研究仓库 HEAD 仍 `6c2b915`，无 tracked/untracked 变化 |
| no prompt tuning after suite | ✅ fixture 在首次运行前冻结（SHA256 `de1e551a…`），运行后未编辑 |
| no system config change | ✅ 未改系统 Python / conda / 证书 / TLS / UU |
| no automatic language R&D | ✅ 未扩充训练数据、未重训、未换模型、未引入外部 LLM |

---

## 20. 结论与请求

Task 8B.1 的兜底交互链**机制上成立**（结构化 suggestion → Validator → Y/N → 真实 predict；N 与危险语义安全
拒绝），**24 条最终 resolution 21/24**；但 4 条 known-conflict 中 above 一条未获建议，因此按 §23 判定为
**`FALLBACK_CORE_PATH_FAILS`**，并如实记录失败集中在 above 方向措辞。是否继续语言层研发（属于新研发阶段）
交由 ChatGPT 与用户决定。

> **Task 8B.1 已完成后停止。请将 `RC1_TASK8B1_REPORT.md` 上传至 ChatGPT；待 ChatGPT 审核自然语言 fallback UX 后，再与用户商榷亲自 Demo 测试计划。**
