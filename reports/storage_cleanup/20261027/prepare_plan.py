"""Prepare exact cleanup allowlist and retained identities. Does not delete anything."""
import csv
import gzip
import hashlib
import json
import shutil
import stat
import subprocess
from pathlib import Path
from inventory import ROOT,OUT,DELIVERY,scan


def identity(path):
    s=path.stat()
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':digest,'hard_links':s.st_nlink}


def safe(path):
    path=path.absolute()
    if not path.is_relative_to(ROOT) or path==ROOT:raise ValueError('outside project')
    current=path
    while current!=ROOT:
        if current.exists() and current.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:raise ValueError('reparse '+str(current))
        current=current.parent
    return path


def main():
    plan=OUT/'plan.json'
    if plan.exists():raise FileExistsError('Do not overwrite prior cleanup plan')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    tracked={str((ROOT/p).absolute()).lower() for p in tracked if p}
    candidates=[];dirs=[]
    def add_file(path,category,reason,retained_copy=None):
        path=safe(path)
        if str(path).lower() in tracked:raise ValueError('tracked target')
        row={**identity(path),'category':category,'reason':reason}
        if row['hard_links']!=1:raise ValueError('hardlinked delete target')
        if retained_copy:
            retained=identity(retained_copy)
            assert row['sha256']==retained['sha256'] and row['bytes']==retained['bytes']
            row['retained_copy']=retained
        candidates.append(row)
    def add_tree(relative,category,reason):
        path=safe(ROOT/relative)
        if not path.exists():return
        files,links,errors=scan(path)
        if links or errors:raise ValueError('unreadable or linked target: '+str(path))
        for row in files:add_file(Path(row['path']),category,reason)
        dirs.extend(str(p) for p in path.rglob('*') if p.is_dir())
        dirs.append(str(path))
    for name in ['demo_r1_57_final','demo_r1_final_2334','demo_r1_fake_evidence',
                 'damage_p1a_tests_04','damage_p1a_tests_05','damage_p1a_tests_06']:
        add_tree('.tmp/'+name,'synthetic_test_temp','Completed fixed-fixture tests only; source and final pass/evidence already committed; no learned weights')
    add_tree('artifacts/damage_p1a/pip_cache','project_install_cache','Only task-private pip HTTP/self-check cache; installed package and exact environment retained')
    source=ROOT/'artifacts/damage_p1a/source'
    for path in sorted(source.glob('*.html')):
        add_file(path,'parsed_directory_cache','Official folder metadata already represented in committed archive inventory/source_access; HTML transport cache has no image or annotation bytes')
    samples=ROOT/'artifacts/damage_p1a/duplicate_samples'
    for old,keep in [('val_SP0_818.JPG','train_SP1_813.JPG'),('val_SP0_1102.JPG','train_SP1_1091.JPG')]:
        add_file(samples/old,'exact_duplicate_image','Exact same encoded bytes retained under train name, SHA verified; original val path/hash retained in research manifest',samples/keep)
    retained=[]
    def keep(path,reason):
        files,links,errors=scan(path) if path.is_dir() else ([{'path':str(path)}],[],[])
        if links or errors:raise ValueError('protected root not fully inspectable')
        for row in files:retained.append({**identity(Path(row['path'])),'reason':reason})
    keep(ROOT/'artifacts/checkpoints','Historical scientific weights and experiment records; not cache')
    keep(ROOT/'datasets','Datasets, annotations, GT and split evidence including historical P0 samples')
    keep(DELIVERY/'BuildReasonSeg_Advisor_RC1','Frozen complete RC1 scientific delivery')
    v3=DELIVERY/'BuildReasonSeg_Demo_V3'
    for path in v3.iterdir():
        if path.name=='_runtime_cache':continue
        if path.name=='_engine':
            for child in path.iterdir():
                if child.name!='logs':keep(child,'V3 engine/config/model/output retained')
        else:keep(path,'V3 user app, results and launcher retained')
    keep(ROOT/'artifacts/legacy_archive/20261019','Two unique non-model historical Demo evidence archives, not redundant runtime packages')
    keep(ROOT/'artifacts/damage_p1a/raw_labels','Complete original labels; live HEAD gives HTML/zero length, same-version recovery not confirmed; not auto-deleted')
    keep(ROOT/'artifacts/damage_p1a/source/README.md','Dated source/license evidence')
    keep(ROOT/'artifacts/damage_p1a/source/LICENSE','Dated source/license evidence')
    keep(ROOT/'artifacts/damage_p1a/protected_baseline.json','Historical protection audit baseline, not disposable data')
    keep(ROOT/'.tmp/demo_r1_raw_test_attempts','Only full raw engineering failure logs; Git contains excerpts/hash, preserve provenance')
    # All existing tracked source/reports before this maintenance remain fixed, except designated handoff.
    for path in tracked:
        p=Path(path)
        if p.is_file() and not p.is_relative_to(OUT) and p.name not in ('CURRENT_TASK.md','EXECUTOR_STATE.yaml'):
            retained.append({**identity(p),'reason':'Existing Git-tracked research/source record'})
    # No environment deletion is authorized; snapshot entire trees by size/mtime, plus critical SHA samples.
    environments=[];critical=[]
    for env in (ROOT/'.conda').iterdir():
        if not env.is_dir():continue
        files,links,errors=scan(env)
        if errors:raise ValueError('environment cannot be inventoried')
        environments.extend(files)
        for rel in ('python.exe','Lib/site-packages/torch/__init__.py','Lib/site-packages/torch/lib/torch_cuda.dll',
                    'Lib/site-packages/numpy/__init__.py','Lib/site-packages/pycocotools/mask.py'):
            p=env/rel
            if p.exists():critical.append(identity(p))
    with gzip.open(OUT/'retained_files_before.json.gz','wt',encoding='utf-8') as f:json.dump(retained,f)
    with gzip.open(OUT/'environment_files_before.json.gz','wt',encoding='utf-8') as f:json.dump(environments,f)
    (OUT/'environment_critical_before.json').write_text(json.dumps(critical,indent=2),encoding='utf-8')
    totals={}
    for row in candidates:
        x=totals.setdefault(row['category'],{'files':0,'logical_bytes':0});x['files']+=1;x['logical_bytes']+=row['bytes']
    result={'scope_root':str(ROOT),'task':'LOCAL-STORAGE-CLEANUP-20261027','deletions':candidates,
        'empty_directories_after_files':sorted(set(dirs),key=lambda x:len(Path(x).parts),reverse=True),
        'categories':totals,'total_files':len(candidates),'total_logical_bytes':sum(x['bytes'] for x in candidates),
        'retained_hash_records':len(retained),'environment_file_records':len(environments),
        'before_plan_free_bytes':shutil.disk_usage(ROOT).free,'source_recovery_caveat':'HEAD checks do not confirm old labels recoverable; original label ZIPs retained',
        'no_legacy_demo_runtime_dirs':not any((DELIVERY/name).exists() for name in ['BuildReasonSeg_Demo_V1','BuildReasonSeg_Demo_V2'])}
    plan.write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (OUT/'cleanup_list.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256','category','reason']);writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in candidates)
    (OUT/'retention_summary.json').write_text(json.dumps({'retained_hash_records':len(retained),
        'environment_records':len(environments),'category_totals':totals,'planned_logical_bytes':result['total_logical_bytes']},indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['total_files','total_logical_bytes','categories','retained_hash_records','environment_file_records']},indent=2))


if __name__=='__main__':main()
