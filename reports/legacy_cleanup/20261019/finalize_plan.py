"""Finalize explicit pre-delete gate, compressed per-file evidence and handoff."""
import collections,csv,gzip,json
from pathlib import Path
from audit_cleanup import OUT,REPO,write_json
from prepare_plan import physical

rows=list(csv.DictReader((OUT/'planned_deletions.csv').open(encoding='utf-8-sig',newline='')))
reasons={reason:'C%02d'%(i+1) for i,reason in enumerate(dict.fromkeys(r['reason'] for r in rows))}
for i,r in enumerate(rows):
    r['reason_id']=reasons[r['reason']]
    r['allocated_bytes']=physical(Path(r['path']))
    if i%15000==0: print('Native allocation',i,'/',len(rows),flush=True)
groups=collections.defaultdict(list)
for r in rows: groups[(r['dev'],r['inode'])].append(r)
released=sum(int(v[0]['allocated_bytes']) for v in groups.values() if len(v)==int(v[0]['nlink']))
with gzip.open(OUT/'planned_deletions.csv.gz','wt',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Remove only this task's superseded, generated untracked serialization, not historical files.
(OUT/'planned_deletions.csv').unlink()
for name in ('duplicate_hash_proof.csv','native_image_export_proof.csv','hash_cache.json'):
    p=OUT/name
    with p.open('rb') as f,gzip.open(str(p)+'.gz','wb') as z:
        import shutil
        shutil.copyfileobj(f,z)
    p.unlink()
summary=json.loads((OUT/'plan_summary.json').read_text(encoding='utf-8'))
summary['expected_released_allocated_bytes']=released
summary['physical_accounting']='FILE_STANDARD_INFO.AllocationSize; unique file ID only when all its hardlinks are planned for deletion.'
write_json('plan_summary.json',summary)
write_json('reason_codes.json',{code:reason for reason,code in reasons.items()})
lines=['# Legacy-Cleanup 清理前方案','',
'日期：2026-10-19。用户明确授权历史清理；本文件和逐文件白名单必须先提交推送，之后才删除。',
'起点：`97ee74c5ffb4a883584f7a994d75ccc5a7bdd4cf`，远端成果分支与本地一致，起始工作区干净。',
'独立分支：`codex/legacy-cleanup-20261019`。不修改治理、算法、训练配置、阈值、指标或 Git 历史。','',
'## 清单与口径','',
'扫描 `C:\\D\\DeepSeekHarness`（含 workspace/delivery），以及由进程/Appx/卸载注册信息核实的 DSH、Codex 安装根和实际使用的 Codex 工具链根。',
'`disk_inventory.csv` 是递归逻辑大小，各层不能相加；`files_before.csv.gz` 是逐文件清单。DSH 根逻辑大小 64,713,247,198 bytes（60.2689 GiB），BuildReasonSeg 44,387,910,417 bytes（41.3395 GiB）。',
'完整扫描有一处访问拒绝：另一项目 `workspace/system-access-risk/.pytest_cache`；提权只读复核仍拒绝，该目录大小未知、原样保留。其余可读文件均列入，不能把总数解释为该不可读目录也已测量。',
'硬链接逐文件核验：17,388 张 Task6M 训练影像与原始 WHU tile index/source_image_ref 精确对应；硬链接路径减少不等于释放同量磁盘。',
'实际预计释放量使用 Windows FILE_STANDARD_INFO AllocationSize 和真实文件 ID/link count，保留其他链接的文件不计物理释放。',
f'白名单 {len(rows):,} 文件，逻辑移除 {summary["logical_bytes"]:,} bytes；预计释放实际分配空间 {released:,} bytes（{released/2**30:.4f} GiB）。报告及小型档案新增占用另计。','',
'## 授权删除类别','',
'|代码|依据|文件数|逻辑 bytes|','|---|---|---:|---:|']
for reason,v in summary['categories'].items():
    lines.append(f'|{reasons[reason]}|{reason}|{v["files"]}|{v["logical_bytes"]}|')
lines += ['',
'## 删除前保护','',
'- 1500 个起始 Git 跟踪文件已建立 SHA256 索引，只有本任务 handoff 与新报告允许改变。Git connectivity 检查 exit 0；两个既有 dangling blob 保留，不做 gc/prune。',
'- RC1 完整目录及其模型、历史输入/输出/日志全部保留；逐文件哈希在 preserved_hash_index.csv。Qwen/SAM2/D-B1/ProgramHead/YOLO 具可靠保留副本后才移除额外副本。',
'- 两套退役 Demo 的全部非模型文件已保存到 artifacts/legacy_archive/20261019 的两个 ZIP（277/133 entries），逐 entry SHA256 与原始文件一致。档案约 53.3 MiB，不包含冗余大模型，不上传 GitHub；归档路径/hash/恢复映射纳入 Git。',
'- 保留全部唯一阶段/消融 checkpoint，包括 Task6A best_run5、最终六个 Task7I 权重。仅删除三个精确重复的历史 checkpoint；不根据精度、文件名或大小重新选择模型。其余唯一历史状态尚无逐字节恢复来源，保留待审。',
'- 关键数据集 split/index/schema、GT、模型配置、原始 WHU 数据、基线日志/曲线/权重和中期成果包均保留。旧基线 workspace 数据集 8078 文件与 C:\\D\\resources\\project 中完整副本逐一 hash 一致；该外部副本仅只读核查，不删除。',
'- `.conda/buildreasonseg-proposal` 整体保留。退役的 mvp 环境为旧原始 clone，包版本清单已保存；删除前再次确认没有进程运行其中。审计改用 proposal Python。',
'- 正在运行的 Codex、Node、Git、安装依赖、认证/登录/个人配置不清理。DSH 安装根未发现可独立确认的废弃缓存，整体保留。其他课程项目不处理。','',
'## 执行约束','',
'只从 planned_deletions.csv.gz 白名单逐文件 unlink；逐路径 resolve 校验位于 C:\\D\\DeepSeekHarness，排除 .git、跟踪文件、保留哈希索引和 symlink/reparse。删除前统一复核大小/mtime/file ID；任何未知 drift 停止。',
'不使用 reset/rebase/amend/stash/clean/force push；不清空系统 TEMP，不删除程序依赖，不改变原始硬链接目标的属性或字节。仅移除白名单涉及目录中已空的目录，不递归扩大范围。',
'每个实际删除项写 cleanup_manifest.csv；原始理由/恢复路径和哈希证明见压缩白名单/duplicate_hash_proof/native_image_export_proof。锁定或拒绝访问的文件保留记录，不强行改权限。',
'完成后复核 Git 跟踪/治理/中期成果/RC1/数据/关键权重 SHA256，核实软件进程与保留 Python/Git可用，统计空间前后与真实卷空闲变化。无模型加载、推理、训练、阈值调整。','']
(OUT/'cleanup_plan.md').write_text('\n'.join(lines),encoding='utf-8')
task='''# CURRENT_TASK — LEGACY_CLEANUP_20261019

Status: IN_PROGRESS
Authority: User authorization dated 2026-10-19, Legacy-Cleanup 工作区清理与交接任务书.
Branch: codex/legacy-cleanup-20261019
Starting HEAD: 97ee74c5ffb4a883584f7a994d75ccc5a7bdd4cf

Scope: inventory actual project/deployment/software roots; remove confirmed redundant
deployments, exact duplicates and rebuildable caches via reviewed per-file allowlist.
Preserve source, Git history, governance, frozen conclusions, unique final weights,
dataset/GT/splits, research records, midterm evidence and authentication/running tools.
No model inference/training/algorithm work; no reset/rebase/amend/stash/clean/history rewrite.

Plan and evidence: reports/legacy_cleanup/20261019/cleanup_plan.md
Checkpoint: inventory, hashes and verified small Demo archives ready before deletion.
Next: commit/push plan, perform allowlisted cleanup, verify integrity and space,
commit/push closure, then end legacy architecture work and enter neutral idle handoff.
'''
(REPO/'handoff/CURRENT_TASK.md').write_text(task,encoding='utf-8')
(REPO/'handoff/EXECUTOR_STATE.yaml').write_text('''schema_version: 1
task_id: LEGACY_CLEANUP_20261019
status: IN_PROGRESS
executor: CODEX
branch: codex/legacy-cleanup-20261019
head: 97ee74c5ffb4a883584f7a994d75ccc5a7bdd4cf
current_step: INVENTORY_AND_PRESERVATION_READY_BEFORE_DELETE
report_directory: reports/legacy_cleanup/20261019
next_action: Push inventory checkpoint, then execute explicit allowlist and verify.
last_checkpoint:
  commit: null
  pushed: false
model_inference_calls: 0
training_calls: 0
governance_changes: false
scientific_changes: false
deletions_started: false
''',encoding='utf-8')
print('FINAL_PLAN_READY',released,flush=True)
