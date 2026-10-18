"""Preserve a bundled .agents skill caught by the pre-unlink safety guard."""
import collections,csv,gzip,json
from pathlib import Path
from audit_cleanup import OUT,REPO,digest,write_csv,write_json
with gzip.open(OUT/'planned_deletions.csv.gz','rt',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
excluded=[r for r in rows if any(x in Path(r['path']).parts for x in ('.git','.agents','.codex'))]
assert len(excluded)==1 and not (OUT/'cleanup_manifest.csv').exists() and not (OUT/'execution_start.json').exists()
remain=[r for r in rows if r not in excluded]
with gzip.open(OUT/'planned_deletions.csv.gz','wt',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(remain)
with (OUT/'preserved_hash_index.csv').open('a',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f)
    for r in excluded:w.writerow([r['path'],r['bytes'],digest(r['path']),'preserved_bundled_agents_skill',''])
summary=json.loads((OUT/'plan_summary.json').read_text(encoding='utf-8'))
summary['files']=len(remain)
summary['logical_bytes']-=sum(int(r['bytes']) for r in excluded)
summary['expected_released_allocated_bytes']-=sum(int(r['allocated_bytes']) for r in excluded)
for r in excluded:
    summary['categories'][r['reason']]['files']-=1
    summary['categories'][r['reason']]['logical_bytes']-=int(r['bytes'])
summary['preserved_hash_records']+=len(excluded)
write_json('plan_summary.json',summary)
write_json('execution_attempt1_guard.json',dict(checkpoint='1fd354d164a9a6f5a896aec4654d5ee13b9d2fec',
    outcome='SAFETY_GUARD_BEFORE_FIRST_DELETE',deleted_files=0,execution_start_file_absent=True,
    excluded_paths=excluded,disposition='Preserve file, narrow allowlist, recommit/push; guard not weakened.'))
p=OUT/'cleanup_plan.md';s=p.read_text(encoding='utf-8')
s=s.replace('71,579','71,578').replace('44,155,351,459','44,155,344,750').replace('31,762,711,160','31,762,702,968')
s=s.replace('|42620|3995496032|','|42619|3995489323|')
s+='\n## 删除前安全检查补记\n\n首个执行检查在任何 unlink 之前拦截 Typer 包内 `.agents/skills/typer/SKILL.md`（6709 bytes）。未删除历史文件。保留该文件并从白名单排除，额外纳入保留哈希索引；没有削弱安全检查。退役 MVP 环境将残留这份包内技能说明及其父目录，不构成可运行 Python 环境。\n'
p.write_text(s,encoding='utf-8')
p=REPO/'handoff/EXECUTOR_STATE.yaml';s=p.read_text(encoding='utf-8').replace('head: 97ee74c5ffb4a883584f7a994d75ccc5a7bdd4cf','head: 1fd354d164a9a6f5a896aec4654d5ee13b9d2fec').replace('current_step: INVENTORY_AND_PRESERVATION_READY_BEFORE_DELETE','current_step: SAFETY_GUARD_EXCLUSION_READY_BEFORE_DELETE').replace('  commit: null','  commit: 1fd354d164a9a6f5a896aec4654d5ee13b9d2fec').replace('  pushed: false','  pushed: true')
s+='first_execution_guard: Bundled Typer .agents skill excluded; zero historical deletions.\n'
p.write_text(s,encoding='utf-8')
print('Restricted plan:',len(remain),summary['expected_released_allocated_bytes'])
