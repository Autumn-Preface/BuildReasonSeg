"""Produce factual closure documents from saved inventory and integrity results."""
import csv,json,shutil,subprocess,time
from pathlib import Path
from audit_cleanup import REPO,OUT,ROOTS

def j(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def md(name,s):(OUT/name).write_text(s.strip()+'\n',encoding='utf-8')
def size(n):return f'{n:,} bytes（{n/2**30:.4f} GiB）'

r=j('execution_result.json');v=j('integrity_result.json');assert v['pass']
before=j('inventory_before.json');after=j('inventory_after.json')
manifest=list(csv.DictReader((OUT/'cleanup_manifest.csv').open(encoding='utf-8-sig')))
preserved=list(csv.DictReader((OUT/'preserved_hash_index.csv').open(encoding='utf-8-sig')))
archives=j('archives.json');categories=j('plan_summary.json')['categories']
dshtotal_before=before['roots'][str(ROOTS[0])]['logical_bytes']
dshtotal_after=after['roots'][str(ROOTS[0])]['logical_bytes']
directories=list(csv.DictReader((OUT/'disk_inventory_after.csv').open(encoding='utf-8-sig')))
repoafter=next(int(x['bytes']) for x in directories if x['path']==str(REPO))
net_volume=r['free_after_bytes']-r['free_before_bytes']
md('README.md',f'''
# Legacy-Cleanup 工作报告

任务：LEGACY_CLEANUP_20261019。执行结果：COMPLETE；等待用户查阅报告，不再主动进行旧架构研究。

|统计范围/指标|清理前|清理后或释放量|
|---|---:|---:|
|C:\\D\\DeepSeekHarness 可读文件逻辑大小|{size(dshtotal_before)}|{size(dshtotal_after)}|
|其中 BuildReasonSeg 仓库逻辑大小|{size(44387910417)}|{size(repoafter)}|
|删除文件逻辑大小|—|{size(r['deleted_logical_bytes'])}|
|已释放文件实际分配空间（扣除保留硬链接）|—|{size(r['released_file_allocation_bytes'])}|
|删除阶段 C: 可用空间实测增加|—|{size(net_volume)}|

已删除 **{r['deleted_files']:,} 个白名单文件**，失败 {r['failed_files']}；清单先提交推送，再执行删除。目录逻辑大小不是物理占用：17,388 张影像中大量路径为硬链接，移除路径不会再次释放已被原始数据使用的块。实际分配空间采用 Windows FILE_STANDARD_INFO AllocationSize/file ID/link count 核算；卷空闲增量含并发系统活动及日志开销，不能与逻辑大小混用。报告/ZIP/Git 新增占用纳入清理后快照。

## 已完成的清理

- 退役 Demo V1/V2 两套重复部署，全部非模型文件（源码、配置、全部结果和运行记录）保存为逐 entry hash 校验的本地 ZIP，模型由完整 RC1 保留副本恢复。
- 移除额外 Qwen/SAM2 副本、三个字节完全一致的历史 checkpoint 副本；保留所有唯一权重状态，没有重新挑选模型。
- 移除 Task6N 冻结 SAM2 特征缓存、Task6M 派生训练影像导出路径、与外部可靠副本逐文件一致的旧基线数据集副本。
- 退役未运行的 mvp Python clone，保留 proposal 环境和依赖版本清单；清理源树 Python/pytest 缓存与 pip 下载缓存。

## 完整性与限制

完整性检查 PASS：{v['tracked_hash_checks']} 个既有跟踪文件（排除两份授权 handoff）、{v['protected_hash_records']} 条保留资产记录、六个冻结 Task7I 权重、完整 RC1、中期成果包及来源、关键数据划分/GT 均通过；135/135 source/config 与 source_manifest 保持一致。未改科学结论、算法、阈值、指标或治理，项目模型加载/推理/训练均为 0。

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
''')
rows=['# 清理前后空间对比','','单位均明确区分；目录递归逻辑大小不可将父子层重复相加。清单时间见 inventory_before/after.json。','',
'|互不重叠的扫描根|清理前逻辑 bytes|清理后逻辑 bytes|逻辑差值|','|---|---:|---:|---:|']
for root,b in before['roots'].items():
    a=after['roots'][root]
    rows.append(f'|{root}|{b["logical_bytes"]:,}|{a["logical_bytes"]:,}|{b["logical_bytes"]-a["logical_bytes"]:,}|')
bt=sum(x['logical_bytes'] for x in before['roots'].values());at=sum(x['logical_bytes'] for x in after['roots'].values())
rows += [f'|以上扫描根总逻辑量|{bt:,}|{at:,}|{bt-at:,}|','',
f'- DSH 根目录逻辑减少：{size(dshtotal_before-dshtotal_after)}。',
f'- 白名单文件逻辑移除：{size(r["deleted_logical_bytes"])}。',
f'- 去除保留硬链接后的文件实际分配释放：**{size(r["released_file_allocation_bytes"])}**。',
f'- 删除阶段卷空闲：{size(r["free_before_bytes"])} → {size(r["free_after_bytes"])}，实测净增 {size(net_volume)}。',
f'- 两份必要的小型 Demo 非模型归档合计 {sum(x["bytes"] for x in archives):,} bytes；报告/Git 文件也产生新增占用，故目录差值小于删除文件总量。',
'- 卷空闲受其他活动、目录/Git/NTFS 元数据影响；文件实际分配核算按完整删除的唯一 NTFS file ID 计一次，其他链接仍保留时计 0。',
'- 初次候选核算使用 GetCompressedFileSizeW，它在本机未提供所需分配口径；正式方案提交前已改为 FILE_STANDARD_INFO，所有最终数值采用后者。',
'- `.pytest_cache` 不可读目录大小未知；可读总量不包括它，未声称精确测量整个 C: 或所有个人数据。',
'- after 快照完成后还会新增少量报告/Git 提交数据；上述是带时间点的可复核快照，而非持续不变的磁盘读数。']
md('disk_usage_before_after.md','\n'.join(rows))
ar='\n'.join(f'- `{a["path"]}`：{a["entries"]} 项，{a["bytes"]:,} bytes，SHA256 `{a["sha256"]}`。' for a in archives)
md('preserved_assets.md',f'''
# 关键保留资产

已校验 {v['protected_hash_records']} 条 SHA256 记录，逐路径/字节数/hash 见 preserved_hash_index.csv。初始 1500 个 Git 跟踪文件除 handoff 外的 1498 个字节保持一致，完整历史没有重写。

|资产|保留位置|保留依据|
|---|---|---|
|核心源码、治理、历史科研报告/指标/图像|当前 Git 工作树及 origin 各历史分支|全部原内容保留；不更新冻结结论|
|中期成果图片及来源证明|reports/midterm_evidence_review/|12 项成果清单、10 原图副本、2 基于记录绘图、原始 JSON/CSV/配置证据和审查结果均保留|
|一套完整旧系统/模型资产|C:\\D\\DeepSeekHarness\\delivery\\BuildReasonSeg_Advisor_RC1|全目录哈希保持不变，含 Qwen/SAM2/ProgramHead/YOLO/D-B1 及历史 inputs/outputs/logs|
|Task7I 六个冻结最终权重|artifacts/checkpoints/task7i/{{Z-B3,D-B1}}/{{20261001,20261002,20261003}}/best.pt|六项与既有 frozen manifest 一致，不做权重选择|
|各阶段/消融唯一历史权重|artifacts/checkpoints/；旧 WHU baseline weights|SHA256 不同的状态保留待审；包括 Task6A best_run5；仅三个相同 hash 副本移除|
|关键数据/标注/split/GT/索引|datasets/ 与 artifacts/whu_native_vector/ 及原始 WHU 根|关键记录哈希和未授权变更对比通过；未改 GT/alignment|
|原始 WHU 数据（范围外，只读）|C:\\D\\resources\\Satellite dataset Ⅱ (East Asia)|17,388 个原始图像 file ID、大小和 mtime 与被移除导出链接前一致；不写任何原始文件属性或字节|
|旧 WHU 基线源码、训练记录和权重|C:\\D\\DeepSeekHarness\\workspace\\project\\WHU_Building_Segment|代码、配置、曲线、评价、best/last 均保留；数据副本有可靠外部整树保留副本|
|旧基线数据集保留副本（范围外，只读）|C:\\D\\resources\\project\\WHU_Building_Segment\\dataset\\WHU_YOLO_dataset|8078 项 SHA256 精确对应，删除后再次核验|
|保留 Python 环境|.conda/buildreasonseg-proposal|整个环境未变，版本清单 environment_versions.json；无真实模型导入|
|Codex/ChatGPT/DSH 安装、运行依赖、个人配置|实际安装根见 registered_appx.json/registered_desktop_apps.json|未删除任何安装文件或身份资料；C:\\D\\DeepSeek Harness 与 C:\\D\\DeepSeekHarness 是不同目录|

## Demo 实现与全部非模型运行证据

{ar}

ZIP 为本地研究保留档案（Git 忽略的 artifacts 内），不将大模型/垃圾副本上传 GitHub。源码/构建器/技术文档和中期精选来源记录已在 GitHub；全部外部结果另在上述小型档案中。归档逐 entry SHA256 已在删除前校验，删除后整包 SHA256/CRC/数量再次通过。

旧报告中的原始 Demo 路径保留历史含义，不伪装为当前仍可直接访问；恢复映射在 duplicate_hash_proof.csv.gz。唯一未退役的 MVP 目录内容为 Typer 随包 `.agents/skills/typer/SKILL.md`，6709 bytes，保持 hash；没有可执行 MVP Python。
''')
md('recovery_notes.md',r'''
# 恢复说明

默认不恢复、不重新部署旧 Demo、不启动推理。旧研究已结束，以下只为必要追溯保留。

1. **代码/治理/科研结论与成果附件**：保留在 Git 当前树、原始分支和完整历史中。起始成果分支 `codex/midterm-evidence-review-20261017` 未改写。所有新增报告在独立清理分支。
2. **Demo V1/V2**：从 artifacts/legacy_archive/20261019 相应 ZIP 解压至新的审阅目录；先对照 archives.json 校验 hash，再按 duplicate_hash_proof.csv.gz 中原路径映射，从 RC1 的对应 model 文件复制被省略的模型。archive entry 与被删除文件一一对应，不能用新预测替代历史结果。若未来需要运行，机器 Python locator 应与保留 proposal 环境核对；本次没有改档案或启动 Demo。
3. **Qwen 本地缓存**：完整 RC1 的 `model/components/program_head/Qwen3-VL-2B-Instruct/` 十个文件对应原 HF snapshot revision `89644892e4d85e24eaac8bacfd4f463576704203`。缓存 refs/树记录仍留存，按 proof 逐文件复制回原 snapshot 并验 hash 即可，不必联网更换版本。
4. **SAM2 与三个历史重复 checkpoint**：duplicate_hash_proof.csv.gz 给出逐字节一致的保留副本路径。只恢复同一 hash，不做新模型选择。
5. **旧基线数据副本**：从 C:\D\resources\project\WHU_Building_Segment\dataset\WHU_YOLO_dataset 按对应相对路径恢复；8078 项 hash 已验证。此保留副本位于清理范围外，未移动/修改。
6. **Task6M YOLO 影像导出**：源 tile index、data.yaml、labels 和原始 TIFF 都保留。按 source_image_ref 恢复相同链接或副本即可。历史 scripts/task6m_build_export.py 自带删除整个导出目录的行为，**本次未执行它，日后不得为了恢复影像而盲目执行破坏性整树重建**。不要编辑仍有多个链接的原始影像。
7. **Task6N feature cache**：是冻结 SAM2.1 Hiera Base+ 对原始 RGB 的 float32 memoization，配置/encoder 源码/精确 pretrained hash/原始数据保留。需要时可重新计算，耗时/显存和跨运行逐位一致性不承诺；本次没有重新编码或推理。
8. **旧 MVP 环境**：environment_versions.json 和 environment/task6a_requirements_lock.txt 保留依赖信息；proposal 环境整套保留。被删除 clone 不承诺逐字节恢复，不能把版本清单当环境镜像；不得重新安装/破坏正在运行的 Codex。
9. **pip/bytecode/pytest cache**：按既有版本下载或正常使用时自然重建，不包含唯一科研结论；本次未运行模型或重试实验。

两个 Demo ZIP 只保留在本机，没有把重复大模型上传 GitHub。若本机存储未来丢失，ZIP 中未另行提交的全部原始运行细节不能仅从 GitHub 完整恢复；精选中期原图和来源记录已经在 Git。请将这约 53.3 MiB 保留档案纳入日常研究备份。本次不擅自访问外部备份服务。

所有唯一历史模型状态均保留，没有删除不可恢复的唯一训练权重。无法确认价值或来源的项目文件、认证与个人设置、不相关课程项目和不可读目录不处理。
''')
sw=j('software_verification.json')
md('final_verification.md',f'''
# 最终完整性与软件检查

结果：PASS（执行者自检，保留用户审查）。依据 integrity_result.json、hash_gate_after.json、execution_result.json、software_verification.json。

|检查|结果|
|---|---|
|删除前 Git branch/local HEAD/remote HEAD|独立分支、清单 checkpoint 已推送；执行时与 upstream 同 SHA|
|执行安全检查|所有 {r['deleted_files']:,} 路径先 resolve、file ID/size/mtime、跟踪/保留排除；仅白名单 unlink|
|实际删除与失败|{r['deleted_files']:,} 删除，{r['failed_files']} 失败；没有额外删除|
|起始跟踪文件|1498/1498 SHA256 一致（两份 handoff 授权更新）|
|关键资产|{v['protected_hash_records']} 条保留记录 PASS；{v['unique_file_hash_checks']} 个唯一文件 post hash 检查|
|冻结最终权重|6/6 SHA256 一致；唯一历史权重保留，无模型加载/挑选|
|外部完整 RC1|整目录保留 hash；历史 input/output/logs 与既有 grandfathered Ultralytics 配置全部不变|
|canonical source/config|135/135 SHA256 一致；external source_manifest 与 canonical byte/hash 一致|
|中期检查成果及原始来源|全部原字节保留|
|原始数据与 split/GT|关键哈希一致；17,388 原始 tile 的身份/大小/mtime 一致；8078 项外部保留副本 hash 一致|
|两套退役 Demo|原部署目录已移除；410 个非模型文件保留在 2 个完整 CRC/hash 校验 ZIP|
|其他路径变更|unexpected missing/changed/additions = 0（仅允许报告/归档/Git/handoff 的授权变动）|
|安装/认证/其他课程项目|未删除，已扫描安装根文件路径/大小/mtime 对比无异常；拒绝访问目录保留|
|Codex|活动进程仍存在；当前工具会话持续可用，未停机/重装/清理其依赖|
|Git / 保留 Python|Git version、proposal Python stdlib 与已安装包元数据读取通过；不加载真实模型|
|DSH|安装程序存在且元数据未变；未运行，未进行完整 GUI 启动测试|
|治理 / 源码 / 科学结论|无修改|
|真实模型推理 / 训练 / 算法优化|0 / 0 / 0|

清理前 git fsck --connectivity-only exit 0，两项 dangling blob 属既有状态，未清理 Git 数据。清理后同项结果见 git_fsck_after.txt，exit code {sw['git_fsck_exit']}。

已知保留：唯一历史 checkpoint、完整 RC1/proposal 环境、新方向可能复用的原始数据/通用 pretrained 权重、全部科研记录；未把全部 best/last 当同一文件。旧 MVP 残留 6709-byte Typer 随包 skill，是删除前 guard 保留项。其他项目不可读 pytest cache 未测量、未清理。

本次没有测试模型科学性能，也未重跑历史 suite/Final Demo。旧目录清理不改变历史评价结果；旧报告中的历史原路径与本次恢复索引应结合阅读。

最终 handoff 使用 AGENTS 规定的 NONE/AWAITING_SUPERVISOR neutral idle，branch/head/last_checkpoint 为 null。实际最终 HEAD 在 push 后从 Git 核对并向用户报告，不写自引用 SHA。停止旧架构主动研究。
''')
(REPO/'handoff/CURRENT_TASK.md').write_text('''# CURRENT_TASK — NONE

Task ID: NONE
Status: AWAITING_SUPERVISOR

Previous task: LEGACY_CLEANUP_20261019 — COMPLETE
Report: reports/legacy_cleanup/20261019/README.md

Inventory, explicit deletion plan, hashes and recovery records were committed/pushed
before cleanup. Allowlisted redundant files were removed; preservation and space
checks passed. No product/governance/scientific changes, training or inference.

Idle Git metadata: branch = null; head = null. Actual Git branch/HEAD remain authoritative.
Await user/Supervisor report review or a new explicit task.
Do not proactively resume legacy architecture research, Demo, inference or development.
''',encoding='utf-8')
(REPO/'handoff/EXECUTOR_STATE.yaml').write_text('''schema_version: 1
task_id: NONE
status: AWAITING_SUPERVISOR
executor: CODEX
branch: null
head: null
last_checkpoint:
  commit: null
  pushed: null
previous_task: LEGACY_CLEANUP_20261019
previous_task_status: COMPLETE
report_directory: reports/legacy_cleanup/20261019
current_step: IDLE_AFTER_LEGACY_CLEANUP
next_action: Await explicit user task; no autonomous legacy architecture research.
model_inference_calls: 0
training_calls: 0
governance_changes: false
scientific_changes: false
integrity_verification: PASS
''',encoding='utf-8')
print('Closure reports generated; idle handoff ready.')
