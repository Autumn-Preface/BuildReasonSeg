# Legacy-Cleanup 清理前方案

日期：2026-10-19。用户明确授权历史清理；本文件和逐文件白名单必须先提交推送，之后才删除。
起点：`97ee74c5ffb4a883584f7a994d75ccc5a7bdd4cf`，远端成果分支与本地一致，起始工作区干净。
独立分支：`codex/legacy-cleanup-20261019`。不修改治理、算法、训练配置、阈值、指标或 Git 历史。

## 清单与口径

扫描 `C:\D\DeepSeekHarness`（含 workspace/delivery），以及由进程/Appx/卸载注册信息核实的 DSH、Codex 安装根和实际使用的 Codex 工具链根。
`disk_inventory.csv` 是递归逻辑大小，各层不能相加；`files_before.csv.gz` 是逐文件清单。DSH 根逻辑大小 64,713,247,198 bytes（60.2689 GiB），BuildReasonSeg 44,387,910,417 bytes（41.3395 GiB）。
完整扫描有一处访问拒绝：另一项目 `workspace/system-access-risk/.pytest_cache`；提权只读复核仍拒绝，该目录大小未知、原样保留。其余可读文件均列入，不能把总数解释为该不可读目录也已测量。
硬链接逐文件核验：17,388 张 Task6M 训练影像与原始 WHU tile index/source_image_ref 精确对应；硬链接路径减少不等于释放同量磁盘。
实际预计释放量使用 Windows FILE_STANDARD_INFO AllocationSize 和真实文件 ID/link count，保留其他链接的文件不计物理释放。
白名单 71,578 文件，逻辑移除 44,155,344,750 bytes；预计释放实际分配空间 31,762,702,968 bytes（29.5813 GiB）。报告及小型档案新增占用另计。

## 授权删除类别

|代码|依据|文件数|逻辑 bytes|
|---|---|---:|---:|
|C01|Retired Demo deployment; all non-model bytes preserved in verified local ZIP.|410|57574728|
|C02|Retired Demo model duplicate; single complete Advisor_RC1 retained.|36|9431900776|
|C03|Redundant Qwen cached model/tokenizer copy; pinned revision and complete RC1 retained.|10|4266640306|
|C04|Redundant SAM2 pretrained checkpoint; byte-identical RC1 copy retained.|1|323606802|
|C05|Rebuildable frozen-SAM2 per-tile feature cache; configs/task6n_oracle_relation_field.yaml and original RGB/weights retained.|2363|9911442816|
|C06|Derived YOLO training image export; exact retained original verified (usually NTFS hardlink).|17388|12400660232|
|C07|Redundant historical baseline dataset; exact full-tree peer outside cleanup scope retained.|8078|3220368896|
|C08|Retired original Python clone; proposal environment retained intact, no active MVP process at final gate.|42619|3995489323|
|C09|Exact duplicate historical checkpoint; unique stage/ablation state retained, no model selection.|3|395047617|
|C10|Rebuildable Python bytecode / pytest cache.|413|13143782|
|C11|Rebuildable pip HTTP/wheel download cache; installed proposal environment retained.|257|139469472|

## 删除前保护

- 1500 个起始 Git 跟踪文件已建立 SHA256 索引，只有本任务 handoff 与新报告允许改变。Git connectivity 检查 exit 0；两个既有 dangling blob 保留，不做 gc/prune。
- RC1 完整目录及其模型、历史输入/输出/日志全部保留；逐文件哈希在 preserved_hash_index.csv。Qwen/SAM2/D-B1/ProgramHead/YOLO 具可靠保留副本后才移除额外副本。
- 两套退役 Demo 的全部非模型文件已保存到 artifacts/legacy_archive/20261019 的两个 ZIP（277/133 entries），逐 entry SHA256 与原始文件一致。档案约 53.3 MiB，不包含冗余大模型，不上传 GitHub；归档路径/hash/恢复映射纳入 Git。
- 保留全部唯一阶段/消融 checkpoint，包括 Task6A best_run5、最终六个 Task7I 权重。仅删除三个精确重复的历史 checkpoint；不根据精度、文件名或大小重新选择模型。其余唯一历史状态尚无逐字节恢复来源，保留待审。
- 关键数据集 split/index/schema、GT、模型配置、原始 WHU 数据、基线日志/曲线/权重和中期成果包均保留。旧基线 workspace 数据集 8078 文件与 C:\D\resources\project 中完整副本逐一 hash 一致；该外部副本仅只读核查，不删除。
- `.conda/buildreasonseg-proposal` 整体保留。退役的 mvp 环境为旧原始 clone，包版本清单已保存；删除前再次确认没有进程运行其中。审计改用 proposal Python。
- 正在运行的 Codex、Node、Git、安装依赖、认证/登录/个人配置不清理。DSH 安装根未发现可独立确认的废弃缓存，整体保留。其他课程项目不处理。

## 执行约束

只从 planned_deletions.csv.gz 白名单逐文件 unlink；逐路径 resolve 校验位于 C:\D\DeepSeekHarness，排除 .git、跟踪文件、保留哈希索引和 symlink/reparse。删除前统一复核大小/mtime/file ID；任何未知 drift 停止。
不使用 reset/rebase/amend/stash/clean/force push；不清空系统 TEMP，不删除程序依赖，不改变原始硬链接目标的属性或字节。仅移除白名单涉及目录中已空的目录，不递归扩大范围。
每个实际删除项写 cleanup_manifest.csv；原始理由/恢复路径和哈希证明见压缩白名单/duplicate_hash_proof/native_image_export_proof。锁定或拒绝访问的文件保留记录，不强行改权限。
完成后复核 Git 跟踪/治理/中期成果/RC1/数据/关键权重 SHA256，核实软件进程与保留 Python/Git可用，统计空间前后与真实卷空闲变化。无模型加载、推理、训练、阈值调整。

## 删除前安全检查补记

首个执行检查在任何 unlink 之前拦截 Typer 包内 `.agents/skills/typer/SKILL.md`（6709 bytes）。未删除历史文件。保留该文件并从白名单排除，额外纳入保留哈希索引；没有削弱安全检查。退役 MVP 环境将残留这份包内技能说明及其父目录，不构成可运行 Python 环境。
