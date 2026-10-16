"""Read-only evidence audit, copying and plotting only. Never imports a model runtime."""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ROOTS=[REPO, REPO.parent/'WHU_Building_Segment',Path('C:/D/resources/project/WHU_Building_Segment'),
       Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Advisor_RC1'),Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Demo_V1'),
       Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Demo_V2')]
SKIP={'.git','.conda','local_cache','weights','model','models','node_modules','__pycache__','.pytest_cache','midterm_evidence_review'}
EXT={'.png','.jpg','.jpeg','.svg','.webp'}

def identity(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
    return {'bytes':path.stat().st_size,'sha256':digest.hexdigest()}

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args]).decode('utf-8').strip()

def inventory():
    from PIL import Image
    if (OUT/'audit/baseline.json').exists():raise FileExistsError('Preserve prior baseline; no overwrite')
    (OUT/'audit').mkdir(parents=True,exist_ok=True)
    tracked=git('ls-files','-z').split('\0')
    protected=[{'path':r,**identity(REPO/r)} for r in tracked if r and r not in ('handoff/CURRENT_TASK.md','handoff/EXECUTOR_STATE.yaml')]
    metadata=[];images=[];summaries=[]
    for root in ROOTS:
        counts=collections.Counter(); errors=[]
        if not root.is_dir():summaries.append({'root':str(root),'exists':False});continue
        for current,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d not in SKIP]
            for name in files:
                path=Path(current)/name;stat=path.stat();rel=path.relative_to(root).as_posix()
                metadata.append({'root':str(root),'path':rel,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
                if path.suffix.lower() not in EXT:continue
                group='/'.join(rel.split('/')[:2]);counts[group]+=1
                row={'root':str(root),'path':rel,'bytes':stat.st_size,'width':'','height':'','format':'','read_error':''}
                try:
                    if path.suffix.lower()=='.svg':row['format']='SVG'
                    else:
                        with Image.open(path) as im:row.update(width=im.width,height=im.height,format=im.format)
                except Exception as error:row['read_error']=type(error).__name__;errors.append(str(path))
                images.append(row)
        summaries.append({'root':str(root),'exists':True,'image_count':sum(counts.values()),'groups':dict(counts),'image_header_errors':errors})
    for name,rows in [('image_inventory.csv.gz',images),('existing_files_metadata.csv.gz',metadata)]:
        with gzip.open(OUT/'audit'/name,'wt',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    refs={};remote_images={}
    for line in git('for-each-ref','--format=%(refname) %(objectname)','refs/remotes/origin').splitlines():
        ref,head=line.split(' ');refs[ref]=head
        for path in git('ls-tree','-r','--name-only',head).splitlines():
            if Path(path).suffix.lower() in EXT:remote_images.setdefault(path,[]).append(ref)
    dump(OUT/'audit/github_inventory.json',{'repository':'https://github.com/Autumn-Preface/BuildReasonSeg','fetch_succeeded':True,'starting_head':git('rev-parse','HEAD'),
            'default_main_head':git('rev-parse','origin/main'),'refs':refs,'unique_image_paths':len(remote_images),'image_paths_refs':remote_images})
    dump(OUT/'audit/baseline.json',{'audit_date':'2026-10-17','timezone':'Asia/Shanghai','branch':git('branch','--show-current'),
            'starting_head':git('rev-parse','HEAD'),'tracked_protected':protected,'scope':summaries,
            'metadata_files':len(metadata),'image_files':len(images),'skipped_directory_names':sorted(SKIP),
            'limits':'Filename/header inventory is exhaustive within listed roots/exclusions; content review is selective. No whole-disk/private-document scan. Large weights/environment/model cache excluded from image traversal; no model APIs invoked.'})
    print(json.dumps({'scope':summaries,'images':len(images),'protected_tracked_files':len(protected)},ensure_ascii=False))

def verify():
    baseline=json.loads((OUT/'audit/baseline.json').read_text(encoding='utf-8'));changes=[]
    for row in baseline['tracked_protected']:
        path=REPO/row['path']
        if not path.is_file() or identity(path)!={k:row[k] for k in ('bytes','sha256')}:changes.append(str(path))
    with gzip.open(OUT/'audit/existing_files_metadata.csv.gz','rt',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            path=Path(row['root'])/row['path']
            if path in (REPO/'handoff/CURRENT_TASK.md',REPO/'handoff/EXECUTOR_STATE.yaml'):continue
            if not path.is_file() or path.stat().st_size!=int(row['bytes']) or path.stat().st_mtime_ns!=int(row['mtime_ns']):changes.append(str(path))
    selected=json.loads((OUT/'audit/selection_records.json').read_text(encoding='utf-8'))
    for row in selected:
        if identity(row['original_path'])!=row['identity'] or identity(OUT/row['copy_path'])!=row['identity']:changes.append(row['original_path'])
    result={'pass':not changes,'changed_existing_paths':sorted(set(changes)),'protected_tracked_files':len(baseline['tracked_protected']),
            'metadata_checked':baseline['metadata_files'],'selected_originals_sha256_checked':len(selected),'model_inference_calls':0,'training_calls':0,
            'historical_results_logs_configs_unchanged':not changes,'only_allowed_existing_changes':['handoff/CURRENT_TASK.md','handoff/EXECUTOR_STATE.yaml']}
    dump(OUT/'audit/integrity_result.json',result);print(json.dumps(result,ensure_ascii=False));return 0 if result['pass'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.inventory:inventory()
    if a.verify:raise SystemExit(verify())
