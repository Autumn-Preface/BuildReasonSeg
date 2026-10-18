"""Complete the journaled cleanup with bounded empty-directory removals only."""
import collections,csv,gzip,json,os,shutil,time
from pathlib import Path
from audit_cleanup import REPO,OUT,ROOTS,write_json

assert not (OUT/'execution_result.json').exists()
with gzip.open(OUT/'planned_deletions.csv.gz','rt',encoding='utf-8-sig',newline='') as f:plan=list(csv.DictReader(f))
with (OUT/'cleanup_manifest.csv').open(encoding='utf-8-sig',newline='') as f:actual=list(csv.DictReader(f))
assert len(actual)==len(plan) and {r['path'] for r in actual}=={r['path'] for r in plan}
assert all(r['status']=='DELETED' and not Path(r['path']).exists() for r in actual)
roots={ROOTS[0]/'delivery/BuildReasonSeg_Demo_V1',ROOTS[0]/'delivery/BuildReasonSeg_Demo_V2',
       ROOTS[0]/'workspace/project/WHU_Building_Segment/dataset/WHU_YOLO_dataset',
       REPO/'.conda/buildreasonseg-mvp',REPO/'artifacts/task6n/features',REPO/'artifacts/task6m_yolo_native/images',
       REPO/'local_cache/pip',REPO/'local_cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203'}
for r in plan:
    p=Path(r['path'])
    for label in ('__pycache__','.pytest_cache'):
        if label in p.parts:roots.add(Path(*p.parts[:p.parts.index(label)+1]))
removed=0
for root in roots:
    resolved=root.resolve()
    assert ROOTS[0] in resolved.parents and not any(x in resolved.parts for x in ('.git','.agents','.codex'))
    if not root.exists():continue
    # No file deletion, symlink following, chmod, or recursive rmtree.
    for parent,dirs,files in os.walk(root,topdown=False,followlinks=False):
        p=Path(parent)
        if p.is_symlink() or p.stat().st_file_attributes&0x400:continue
        assert p==root or root in p.parents
        try:p.rmdir();removed+=1
        except OSError:pass
groups=collections.defaultdict(list)
for r in plan:groups[(r['dev'],r['inode'])].append(r)
release=sum(int(g[0]['allocated_bytes']) for g in groups.values() if len(g)==int(g[0]['nlink']))
start=json.loads((OUT/'execution_start.json').read_text(encoding='utf-8'))
result=dict(finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'),checkpoint=start['checkpoint'],
    planned_files=len(plan),deleted_files=len(actual),failed_files=0,removed_empty_directories=removed,
    deleted_logical_bytes=sum(int(r['bytes']) for r in actual),released_file_allocation_bytes=release,
    physical_method='FILE_STANDARD_INFO allocation; only completely unlinked file IDs counted.',
    free_before_bytes=start['disk_free_bytes'],free_after_bytes=shutil.disk_usage('C:\\').free,
    no_source_or_governance_or_inference_writes=True,mechanics_correction='Own process stopped only after all unlinks; bounded empty-directory-only finish.')
write_json('execution_result.json',result)
write_json('deletion_failures.json',[])
print(json.dumps(result,ensure_ascii=False),flush=True)
