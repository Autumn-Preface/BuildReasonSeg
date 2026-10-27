"""Verify exact deletions and retained file/env identities. Never trains or downloads."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from inventory import ROOT,OUT,scan


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
deleted=[json.loads(x) for x in (OUT/'deletion_log.jsonl').read_text(encoding='utf-8-sig').splitlines()]
assert len(deleted)==plan['total_files'] and {r['path'] for r in deleted}=={r['path'] for r in plan['deletions']}
assert all(not Path(r['path']).exists() for r in deleted)
retained=json.loads(gzip.decompress((OUT/'retained_files_before.json.gz').read_bytes()))
allowed_handoff={str(ROOT/'handoff/CURRENT_TASK.md').lower(),str(ROOT/'handoff/EXECUTOR_STATE.yaml').lower()}
checked=0
for row in retained:
    path=Path(row['path'])
    if str(path).lower() in allowed_handoff:continue
    assert path.is_file() and path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],row['path']
    checked+=1
expected=json.loads(gzip.decompress((OUT/'environment_files_before.json.gz').read_bytes()))
actual=[]
for env in (ROOT/'.conda').iterdir():
    if env.is_dir():
        files,links,errors=scan(env);assert not errors;actual.extend(files)
key=lambda r:r['path']
assert sorted(expected,key=key)==sorted(actual,key=key),'Environment file identity metadata changed'
for row in json.loads((OUT/'environment_critical_before.json').read_text(encoding='utf-8')):
    assert digest(Path(row['path']))==row['sha256']
manifest=json.loads((Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Demo_V3')/'_engine/package_manifest.json').read_text(encoding='utf-8-sig'))
for row in manifest['files']:
    p=Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Demo_V3')/row['path']
    assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],row['path']
changed=subprocess.check_output(['git','diff','--name-only','4dcabff660de82a0b8f585da6c2c22dea9bafc9f'],cwd=ROOT,text=True).splitlines()
assert all(p in ['handoff/CURRENT_TASK.md','handoff/EXECUTOR_STATE.yaml'] or p.startswith('reports/storage_cleanup/20261027/') for p in changed),changed
before=json.loads((OUT/'disk_immediately_before.json').read_text(encoding='utf-8-sig'))
after=json.loads((OUT/'disk_immediately_after.json').read_text(encoding='utf-8-sig'))
result={'deleted_files':len(deleted),'deleted_logical_bytes':sum(r['bytes'] for r in deleted),'retained_hash_records_checked':checked,
    'intentional_handoff_changes_excluded':2,'environment_size_mtime_file_set_unchanged':len(expected),'environment_critical_sha256_pass':True,
    'v3_manifest_files_verified':len(manifest['files']),'old_reports_and_scientific_assets_unchanged':True,
    'before_free_bytes':before['free_bytes'],'after_free_bytes':after['free_bytes'],'net_volume_free_increase_bytes':after['free_bytes']-before['free_bytes'],
    'space_caveat':'Logical removed bytes and net live volume change differ due to allocation and other processes; do not equate them',
    'no_training_or_dataset_download':True}
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
