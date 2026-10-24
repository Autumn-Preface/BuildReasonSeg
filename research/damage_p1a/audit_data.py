"""Numeric-label audit only: does not guess category names or admit training."""
from __future__ import annotations
import collections
import concurrent.futures
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from . import audit_sources as source

ROOT=source.ROOT
OUT=source.OUT
RAW=ROOT/'artifacts/damage_p1a/raw_labels'


def budget():
    data=json.loads((OUT/'official_archive_inventory.json').read_text(encoding='utf-8'))
    archives=sum(a['archive_bytes'] for s in data['splits'].values() for a in s['archives'].values())
    unpacked=sum(a['uncompressed_bytes'] for s in data['splits'].values() for a in s['archives'].values())
    free=shutil.disk_usage(ROOT).free
    # Measured old environment 3,992,649,226 logical bytes; leave 8 GiB for its independent copy/cache.
    plan={'all_archives':archives,'all_uncompressed':unpacked,'new_env_and_cache_upper':8*1024**3,
          'training_outputs_upper':4*1024**3,'temporary_upper':2*1024**3}
    reserve=50*1024**3
    result={'free_at_budget_bytes':free,'full_plan':plan,'minimum_system_free_bytes':reserve,
            'full_plan_plus_reserve_bytes':sum(plan.values())+reserve,
            'fits':free>=sum(plan.values())+reserve,'current_acquisition':'LABEL_ARCHIVES_ONLY_UNTIL_MAPPING_GATE',
            'full_resolution_mask_cache':False,'test_usage':'annotation inventory only; no predictions or tuning',
            'dense_probability_archive':False,'old_assets_cleanup':False}
    source.dump('storage_budget_exact.json',result)
    if not result['fits']:raise OSError('Full plan fails 50 GiB reserve; no download')
    return data


def download(split, info):
    path=RAW/(split+'_labels.zip')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=info['archive_bytes']:raise ValueError('Existing archive incomplete; no overwrite')
        return path
    url='https://drive.usercontent.google.com/download?id='+info['drive_id']+'&export=download&confirm=t'
    temporary=path.with_suffix('.zip.part')
    if temporary.exists():raise FileExistsError('Preserve previous partial download for review')
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DAMAGE-P1A-label-admission'}),timeout=60) as response, temporary.open('xb') as f:
        total=0
        while block:=response.read(1024**2):
            total+=len(block)
            if total>info['archive_bytes']:raise ValueError('Download exceeds budget')
            f.write(block)
    if total!=info['archive_bytes']:raise ValueError('Archive size differs from prior budget')
    temporary.rename(path)
    return path


def main():
    inventories=budget()
    result={'class_names':'UNCONFIRMED_NO_FILENAME_MAPPING','splits':{}}
    manifests=[]
    name_sets={}
    crc_groups=collections.defaultdict(list)
    for split, inv in inventories['splits'].items():
        images=inv['archives']['images.zip']['files']
        name_sets[split]={Path(f['name']).stem for f in images}
        for f in images:crc_groups[(f['bytes'],f['crc32'])].append((split,f['name']))
        info=inv['archives']['labels.zip']
        path=download(split,info)
        stats={'images':len(images),'labels':0,'empty_labels':0,'class_instances':collections.Counter(),
               'class_images':collections.Counter(),'invalid_lines':[],'polygons':0,'coordinate_min':1.0,'coordinate_max':0.0,
               'images_with_multiple_numeric_classes':0,'archive_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
        with zipfile.ZipFile(path) as z:
            label_names={Path(i.filename).stem for i in z.infolist() if i.filename.endswith('.txt')}
            stats['missing_labels']=sorted(name_sets[split]-label_names)
            stats['orphan_labels']=sorted(label_names-name_sets[split])
            for i in z.infolist():
                if i.is_dir():continue
                content=z.read(i)
                manifests.append({'split':split,'path':i.filename,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
                if not i.filename.endswith('.txt'):continue
                stats['labels']+=1
                present=set()
                lines=content.decode('utf-8-sig').splitlines()
                if not any(x.strip() for x in lines):stats['empty_labels']+=1
                for number,line in enumerate(lines,1):
                    if not line.strip():continue
                    try:
                        tokens=line.split();class_id=int(tokens[0]);points=[float(x) for x in tokens[1:]]
                        if str(class_id)!=tokens[0] or class_id<0 or len(points)<6 or len(points)%2:raise ValueError('ID/point count')
                        if not all(0<=x<=1 for x in points):raise ValueError('out-of-range/nonfinite coordinate')
                        stats['coordinate_min']=min(stats['coordinate_min'],min(points))
                        stats['coordinate_max']=max(stats['coordinate_max'],max(points))
                        stats['class_instances'][class_id]+=1;present.add(class_id);stats['polygons']+=1
                    except Exception as error:stats['invalid_lines'].append({'path':i.filename,'line':number,'error':str(error)})
                stats['class_images'].update(present)
                if len(present)>1:stats['images_with_multiple_numeric_classes']+=1
        result['splits'][split]=stats
        source.dump('label_audit.json',result)
        print(split,stats['class_instances'],stats['class_images'],'invalid',len(stats['invalid_lines']),flush=True)
    result['shared_stems']={a+'__'+b:sorted(name_sets[a]&name_sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    result['archive_crc_size_duplicate_candidates']=[v for v in crc_groups.values() if len(v)>1]
    result['duplicate_scope']='CRC32+size screening only; exact decoded-pixel hash/near-duplicate audit NOT_COMPLETED'
    source.dump('label_audit.json',result)
    source.dump('label_file_manifest.json',manifests)


if __name__=='__main__':main()
