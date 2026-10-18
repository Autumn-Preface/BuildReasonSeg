# Legacy-Cleanup 工作报告

任务：LEGACY_CLEANUP_20261019。执行结果：COMPLETE；等待用户查阅报告，不再主动进行旧架构研究。

|统计范围/指标|清理前|清理后或释放量|
|---|---:|---:|
|C:\D\DeepSeekHarness 可读文件逻辑大小|64,713,247,198 bytes（60.2689 GiB）|20,643,668,113 bytes（19.2259 GiB）|
|其中 BuildReasonSeg 仓库逻辑大小|44,387,910,417 bytes（41.3395 GiB）|13,028,175,732 bytes（12.1334 GiB）|
|删除文件逻辑大小|—|44,155,344,750 bytes（41.1229 GiB）|
|已释放文件实际分配空间（扣除保留硬链接）|—|31,762,702,968 bytes（29.5813 GiB）|
|删除阶段 C: 可用空间实测增加|—|31,533,355,008 bytes（29.3677 GiB）|

已删除 **71,578 个白名单文件**，失败 0；清单先提交推送，再执行删除。目录逻辑大小不是物理占用：17,388 张影像中大量路径为硬链接，移除路径不会再次释放已被原始数据使用的块。实际分配空间采用 Windows FILE_STANDARD_INFO AllocationSize/file ID/link count 核算；卷空闲增量含并发系统活动及日志开销，不能与逻辑大小混用。报告/ZIP/Git 新增占用纳入清理后快照。

## 已完成的清理

- 退役 Demo V1/V2 两套重复部署，全部非模型文件（源码、配置、全部结果和运行记录）保存为逐 entry hash 校验的本地 ZIP，模型由完整 RC1 保留副本恢复。
- 移除额外 Qwen/SAM2 副本、三个字节完全一致的历史 checkpoint 副本；保留所有唯一权重状态，没有重新挑选模型。
- 移除 Task6N 冻结 SAM2 特征缓存、Task6M 派生训练影像导出路径、与外部可靠副本逐文件一致的旧基线数据集副本。
- 退役未运行的 mvp Python clone，保留 proposal 环境和依赖版本清单；清理源树 Python/pytest 缓存与 pip 下载缓存。

## 完整性与限制

完整性检查 PASS：1498 个既有跟踪文件（排除两份授权 handoff）、6443 条保留资产记录、六个冻结 Task7I 权重、完整 RC1、中期成果包及来源、关键数据划分/GT 均通过；135/135 source/config 与 source_manifest 保持一致。未改科学结论、算法、阈值、指标或治理，项目模型加载/推理/训练均为 0。

Codex/Node/Git 活动安装和依赖、认证/登录/个人配置均未清理。软件运行检查见 final_verification.md；DSH 未运行，验证了程序仍存在和文件元数据未变，未宣称进行了完整启动测试。其他课程项目整体保留。

唯一历史 checkpoint 没有可靠逐字节恢复来源，故保留待审，不因 best/last 名称或体积直接删除。另有其他项目 `.pytest_cache` 一处拒绝访问，提权只读仍不可读，未处理也未计入已测得占用。旧环境保留一份 Typer 自带 `.agents` 技能说明（6709 bytes），首个安全检查在删除前即拦截并缩小白名单，没有削弱检查。

## 交付导航

- `disk_inventory.csv` / `largest_files_before.csv`：清理前目录和大文件；完整逐文件清单为 `files_before.csv.gz`。
- `cleanup_plan.md` / `planned_deletions.csv.gz`：删除前方案与逐文件白名单。
- `cleanup_manifest.csv`：实际删除文件；reason_id 释义见 `reason_codes.json`，原始路径/哈希/恢复映射见两个压缩 proof 表。
- `preserved_assets.md` / `preserved_hash_index.csv`：关键保留位置与 SHA256。
- `disk_usage_before_after.md`：逐根空间统计、硬链接及实际释放口径。
- `recovery_notes.md`：恢复路径和明确不可逐字节恢复的缓存/环境。
- `final_verification.md` / `integrity_result.json`：最终核验与边界。

分支：`codex/legacy-cleanup-20261019`。起点 `97ee74c5ffb4a883584f7a994d75ccc5a7bdd4cf`；有效删除前 checkpoint `ae708695ac8dc5a344c10b1f941fa6ce76f019d8` 已推送。最终本地/远端 HEAD 在提交推送后实际核对并向用户报告，不制造自引用 SHA。
