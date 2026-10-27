"""Read-only bounded project inventory. No deletion or training in this script."""
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
DELIVERY=Path('C:/D/DeepSeekHarness/delivery')


def scan(root):
    files=[];links=[];errors=[]
    def error(e):errors.append(str(e))
    if not root.exists():return files,links,errors
    for parent,dirs,names in os.walk(root,followlinks=False,onerror=error):
        for name in list(dirs):
            path=Path(parent)/name
            s=path.lstat()
            if s.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                links.append(str(path));dirs.remove(name)
        for name in names:
            p=Path(parent)/name
            try:
                s=p.lstat()
                if s.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    links.append(str(p));continue
                files.append({'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'links':s.st_nlink})
            except OSError as e:errors.append(str(e))
    return files,links,errors


def main():
    # Exactly the repository and named project deliveries; never inspect other projects/global caches.
    summaries=[];large=[];details={}
    roots=[p for p in ROOT.iterdir() if p.is_dir()]+[p for p in DELIVERY.iterdir() if p.is_dir() and p.name.startswith('BuildReasonSeg_')]
    for root in roots:
        files,links,errors=scan(root)
        summary={'path':str(root),'files':len(files),'bytes':sum(x['bytes'] for x in files),'reparse_points':links,'errors':errors}
        summaries.append(summary)
        large.extend(x for x in files if x['bytes']>=20*1024**2)
        if root.name in ('.tmp','artifacts','local_cache','datasets','evaluation','logs','runs'):
            groups={}
            for row in files:
                rel=Path(row['path']).relative_to(root)
                key='/'.join(rel.parts[:min(3,len(rel.parts)-1)]) or '.'
                g=groups.setdefault(key,{'files':0,'bytes':0});g['files']+=1;g['bytes']+=row['bytes']
            details[str(root)]=groups
    (OUT/'inventory_before.json').write_text(json.dumps({'roots':summaries,'details':details,
        'disk':{'total':shutil.disk_usage(ROOT).total,'free':shutil.disk_usage(ROOT).free},
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True)},indent=2),encoding='utf-8')
    with (OUT/'largest_files_before.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','mtime_ns','links']);writer.writeheader();writer.writerows(sorted(large,key=lambda x:x['bytes'],reverse=True))
    for row in sorted(summaries,key=lambda x:x['bytes'],reverse=True):print(row['bytes'],row['files'],row['path'])
    for path,groups in details.items():
        if Path(path).name in ('.tmp','artifacts'):
            print('\nGROUPS',path)
            for k,v in sorted(groups.items(),key=lambda x:x[1]['bytes'],reverse=True)[:22]:print(v['bytes'],v['files'],k)


if __name__=='__main__':main()
