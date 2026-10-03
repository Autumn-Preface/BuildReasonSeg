# RC1_BUILD_REPORT — Task 8A: BuildReasonSeg Advisor RC1 本地交付工程骨架与模型包契约

> 任务：**Task 8A**（本地交付工程建设；不是新算法研发、不是新实验、不是 GitHub 发布任务）
> 研发基线：Task 7J 完成，研究状态 commit `6c2b915dbc64acdeb099d194005d74c7180c95fa`
> 报告生成时间：Task 8A 结束时 · **状态：完成并停止（未开始 Task 8B）**

---

## 1. 实际创建的 RC1 根目录

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

- 交付项目总大小：**4.39 GB**（其中模型资产约 4.36 GB）；
- 研发源仓库 `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg` 只读使用：**HEAD 未变**
  （`6c2b915`），无代码改动、无未跟踪文件；唯一改动是 ChatGPT 写入的 `handoff/TO_DSH.md`（Task 8A 任务书），
  按任务要求**未 commit / 未 push**。

## 2. 完整目录树（交付项目）

```text
BuildReasonSeg_Advisor_RC1/
├─ predict.py                    # CLI 契约完成，执行边界 = NOT_IMPLEMENTED_IN_TASK_8A
├─ train.py                      # 同上
├─ prepare_dataset.py            # 同上
├─ evaluate.py                   # 同上
├─ check_setup.py                # 真正可运行（Task 8A 要求）
├─ README.md  VERSION  environment.yml  requirements.txt  setup_env.bat
├─ RC1_BUILD_REPORT.md           # 本文件
├─ buildreasonseg/
│  ├─ __init__.py  paths.py  errors.py
│  ├─ cli/{__init__.py, common.py}
│  ├─ models/{__init__.py, package.py, checkpoints.py}
│  ├─ language/{__init__.py, registry.py, validator.py, frontend.py, suggestion.py, fallback.py}
│  ├─ inference/{__init__.py, contract.py}
│  ├─ training/{__init__.py, contract.py}
│  ├─ data/{__init__.py, contract.py}
│  ├─ diagnostics/{__init__.py, contract.py}
│  └─ utils/{__init__.py, hashing.py, logging.py, device.py}
├─ configs/{inference.yaml, train.yaml}
├─ model/
│  ├─ buildreasonseg_advisor/{decoder.pt, detector.pt, model.yaml, metadata.json, metrics.json}
│  └─ components/
│     ├─ sam2/{sam2.1_hiera_base_plus.pt, sam2.1_hiera_b+.yaml}
│     └─ program_head/{program_parser_l3_rehearsal_v1.pt, Qwen3-VL-2B-Instruct/ (10 files)}
├─ datasets/README.md            # 空用户数据目录（raw/ prepared/ 约定）
├─ inference/{README.md, input/, output/{masks,overlays,diagnostics}}
├─ runs/{train/, eval/}
├─ logs/{task8a_build_step1.json, task8a_portability_smoke.json, build_scripts/}
├─ docs/{command_grammar.md, dataset_format.md, model_card.md}
└─ tests/{conftest.py, test_paths_and_package.py, test_model_package.py, test_cli_contract.py,
          test_language_contract.py, test_setup_checker.py}
```

## 3. Copied assets（source / target / SHA256 / bytes）

| role | source（研发仓库，只读） | target（交付） | SHA256 | bytes | 复制后校验 |
|---|---|---|---|---|---|
| decoder | `artifacts/checkpoints/task7i/D-B1/20261003/best.pt` | `model/buildreasonseg_advisor/decoder.pt` | `9187b133ee4c71ca2750d421d6149bdba1812eabc8c9faacde193036c0db8586` | 1,117,495 | **一致** |
| detector | `artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt` | `model/buildreasonseg_advisor/detector.pt` | `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474` | 54,480,241 | **一致** |
| sam2_checkpoint | `local_cache/models/sam2.1_hiera_base_plus.pt` | `model/components/sam2/sam2.1_hiera_base_plus.pt` | `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` | 323,606,802 | **一致** |
| sam2_config | `.conda/buildreasonseg-proposal/Lib/site-packages/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml` | `model/components/sam2/sam2.1_hiera_b+.yaml` | `ef47e14197a65c1fd542e1862270ee851c2b15e571b2c58ceb0329d15be33769` | 3,766 | **一致** |
| program_head | `artifacts/checkpoints/task7c/program_parser_l3_rehearsal_v1.pt` | `model/components/program_head/program_parser_l3_rehearsal_v1.pt` | `c150573613c421098f55776a4b2a26e1536b806dd5c210f716715fd2ded58d9a` | 70,090,713 | **一致** |
| qwen_base | `local_cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203` | `model/components/program_head/Qwen3-VL-2B-Instruct/` | 10 个文件逐个校验（chat_template.json / config.json / generation_config.json / merges.txt / **model.safetensors** / preprocessor_config.json / tokenizer.json / tokenizer_config.json / video_preprocessor_config.json / vocab.json） | 4,266,640,306 | **全部一致** |

复制清单与逐文件校验记录：`logs/task8a_build_step1.json`（`all_match: true`）。

## 4. 默认 decoder 校验

- architecture **D-B1**，seed **20261003**，`selected_epoch = 8`，`trainable_parameters = 278081`；
- SHA256 **与任务书给定值完全一致**（见 §3 表）；
- 选择依据：**Task 7I validation**（936 条 oracle-reference v0.2 L3 val，三个 D-B1 seed 中 0.387688 最高）
  → 写入 `model.yaml: decoder.selected_by = task7i_validation` 与 `metadata.json
  checkpoint_selection_provenance`（`test_used_for_selection: false`）；
- **未**根据 Task 7J test 选择 seed；**未**做三 seed ensemble；**未**重新导出/重新保存；**未**使用 `last.pt`。

## 5. Detector authoritative source 校验（唯一确定）

- 权威来源：`scripts/task6u_common.py::PROPOSAL_CHECKPOINT`（Task 6U 冻结的 **U-C1** proposal 配置），
  即 `artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt`；
- SHA256 `ef852b58…61f474` 与 Task 7H/7I/7J 记录一致；
- family **YOLO26m-seg**，推理参数 **imgsz 640 / conf 0.05 / max_det 300 / NMS default / TTA false**；
- `tiling: false` 仅描述冻结 proposal model 本体配置（大图 orchestration 属外层工程，已注明）；
- 说明：仓库中另存在 `task6m1/source_epoch18_snapshot/…/best.pt`、`task6m/runs/…`、`task6j/j2_best.pt`、
  `WHU_Building_Segment/…` 等其它 `best.pt` 字符串引用，但**只有 U-C1 配置所指向的 checkpoint** 被 Task 7F/7G/7I/7J
  作为冻结 proposal model 使用，且其 hash 与任务书给定值一致 → **唯一确定，无歧义**。

## 6. SAM2 authoritative source（唯一确定）

- 代码权威来源：`buildreasonseg_mvp/task6n_relation_decoder.py::SAM2_CHECKPOINT`；
- 路径 `local_cache/models/sam2.1_hiera_base_plus.pt`，SHA256 `a2345aed…c004c5`，323,606,802 bytes；
- 全盘扫描确认该 SAM2 权重**只有一个候选**（未发现其它 `sam2*.pt` / `*hiera*.pt`）；
- family **SAM2.1 Hiera Base+**，**未**升级 Large；config 取自当前实际使用的 `sam2` 包内
  `configs/sam2.1/sam2.1_hiera_b+.yaml`（已复制进交付包）。

## 7. Qwen / ProgramHead authoritative source（唯一确定）

- base model：**Qwen3-VL-2B-Instruct**（`configs/mvp/task6j_program_parser.yaml` 的
  `models.qwen_model_id` + `paths.hf_cache`），本地快照 **唯一**：
  `…/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`（10 文件 4.27 GB，含 tokenizer/config/safetensors）；
- ProgramHead：`artifacts/checkpoints/task7c/program_parser_l3_rehearsal_v1.pt`
  （Task 7C controlled-language 20-class ProgramHead），SHA256 `c1505736…d58d9a`；
- **未**更换 4B/7B/14B；**未**下载；**未**重训 parser；
- `model.yaml: language.mode = qwen-first` 且 `unrestricted_natural_language_verified: false`。

## 8. `check_setup.py` 运行结果（真实可运行）

```text
[OK] Python — 3.11.16
[OK] OS — Windows 10 (AMD64)
[OK] PyTorch — 2.13.0+cu132
[OK] CUDA — 可用 (torch 13.2)
[OK] GPU — NVIDIA GeForce RTX 5080 Laptop GPU
[OK] delivery root 可写
[OK] required directories — 16 项齐备
[OK] writable paths — 输出 / 运行 / 日志目录均可写
[OK] model package — …\model\buildreasonseg_advisor
[OK] model.yaml — decoder=D-B1 seed=20261003
[OK] metadata.json — 6 个资产记录
[OK] D-B1 decoder — sha256=9187b133ee4c71ca… bytes=1117495
[OK] U-C1 detector — sha256=ef852b5801e6bdf9… bytes=54480241
[OK] SAM2 assets — sam2.1_hiera_base_plus.pt sha256=a2345aede8715ab1…
[OK] Qwen / ProgramHead assets — ProgramHead sha256=c150573613c42109… + 3 base files
[OK] portability — 无 workspace 依赖；项目可整体移动

BuildReasonSeg environment: READY
```

退出码 **0**。

## 9. Portability smoke test 结果（§17）

- 将整个 `BuildReasonSeg_Advisor_RC1` 复制到
  `C:\D\DeepSeekHarness\delivery\_portability_smoke\BuildReasonSeg_Advisor_RC1`（**未修改原 RC1**）；
- 在复制目录运行 `python check_setup.py` → **`BuildReasonSeg environment: READY`（exit 0）**；
- 路径解析（复制后）：`project_root` = 复制路径；`inference_config / model_dir / decoder / detector /
  sam2 / program_head / qwen` **全部 true**；
- 可写性：`inference/output/{masks,overlays,diagnostics}`、`runs/{train,eval}`、`logs` **全部可写**；
- workspace 依赖审计：**0 处**违规（未使用 junction / symlink）；
- 测试完成后**已删除**临时副本（`_portability_smoke` 不再存在）；
- 记录：`logs/task8a_portability_smoke.json`（`passed: true`）。

## 10. 单元测试结果

在交付项目根目录执行（使用研发环境 Python，**只读**使用，未安装任何东西）：

```text
python -m pytest tests -q
61 passed
```

覆盖 §19.1–19.5：路径与包（root 解析 / 迁移后 config+model 路径 / 无 workspace 指向 / 无 symlink /
包导入 / 支持 program 常量）、model package（decoder 与 detector 的 hash+bytes / model.yaml 解析 /
metadata 字段完整 / metrics 区分 7I 与 7J 且区分 oracle 与 predicted / fallback 需用户确认 /
指定坏模型不静默切换 / 默认模型缺失即硬错误 / 组件完整）、CLI（五个入口 `--help` / `--image` 与
`--input-dir` 互斥 / 正常推理必须 `--prompt` / `--inspect-proposals` 不解析语言 / train 与
prepare_dataset 枚举 / evaluate `test` 仅显式可达 / 默认值逐项核对 / 有效请求返回
`NOT_IMPLEMENTED_IN_TASK_8A` 且非零退出 / error registry 完整）、语言契约（Qwen-first 标记 /
支持列表恰为四个 L3 program / 建议必须通过 validator / 建议必须 Y/N / fallback 不能成为正常路径 /
fallback banner）、setup checker（好 fixture → READY；缺 decoder、hash 不一致、缺 Qwen 组件、缺 SAM2、
model.yaml 非法 → 均非零）。

## 11. 未完成项（Task 8A 范围之外，按任务书留给后续任务）

1. `predict.py` 真实推理：大图 Global Detection → Merge → Global Reference → Local D-B1；
2. Qwen / ProgramHead 真实推理集成（接口已冻结，调用点返回 `NOT_IMPLEMENTED_IN_TASK_8A`）；
3. `prepare_dataset.py` 的 COCO / instance-mask / vector adapter 完整实现；
4. `train.py` / `evaluate.py` 的真实训练与评估实现；
5. 大图 tile size / overlap / merge threshold / context 参数（**故意未冻结**）；
6. GUI / Web / PPT / 阶段报告 / GitHub 源码版与网盘资产版（按 §1.2 / §21 不做）。

## 12. STOP / ambiguity 报告

- **未触发任何 STOP 条件**：D-B1 hash 与任务书一致；U-C1 detector、SAM2、Qwen/ProgramHead 均可**唯一确定**；
  复制未破坏任何研究 artifact；交付可在不依赖 workspace 的情况下运行；无需新下载；无需重训；
  未发现 Task 7I/7J 冻结事实与任务书冲突。
- 记录两处**已解决的非阻塞歧义**（供审计参考）：
  1. detector：仓库存在多个 `best.pt` 字符串引用，按“U-C1 配置 + 任务书给定 hash”双重判定唯一；
  2. SAM2 config：`SAM2_CONFIG_NAME` 是 **sam2 包内**的相对配置名（仓库内无该文件），交付时从当前实际使用的
     包路径复制并记录 SHA256。

## 13. 确认（§22 第 13 项）

| 项目 | 状态 |
|---|---|
| no training | ✅ 未训练任何模型（含 selector / detector / decoder / parser） |
| no test run | ✅ 未访问、未重跑 final test（test 状态保持 `FINAL_TEST_CONSUMED`） |
| no GitHub push | ✅ 未 commit / push 任何内容（研究仓库 HEAD 仍为 `6c2b915`） |
| no new downloads | ✅ 未下载任何模型 / 依赖 |
| no system config change | ✅ 未修改系统 Python / conda base / 研发 `.conda` / 证书 / TLS / UU 加速器 |
| no workspace dependency | ✅ 交付运行不依赖 `workspace\`；迁移 smoke test 通过 |

---

## 14. 结论

**Task 8A 已完成后停止。等待 ChatGPT 审核 RC1 Foundation / Package Contract；不得自动进入完整 predict、train、dataset adapter 或 Demo 实现。**
