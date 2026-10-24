"""Bounded official metadata audit, with a budget before archive range acquisition."""
from __future__ import annotations
import datetime as dt
import hashlib
import html
import io
import json
import re
import shutil
import struct
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/damage_p1a/20261025'
LOCAL = ROOT / 'artifacts/damage_p1a/source'
_history = json.loads((OUT/'source_access.json').read_text(encoding='utf-8')) if (OUT/'source_access.json').exists() else {}
EVENTS = _history.get('events', [])
MAX_TRANSFER = 48 * 1024**2
TRANSFERRED = _history.get('transferred_bytes', 0)


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def get(url, cap=2*1024**2, headers=None, method='GET'):
    global TRANSFERRED
    row = {'at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'url': url, 'method': method}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'BuildReasonSeg-DAMAGE-P1A-admission/1', **(headers or {})}, method=method)
        with urllib.request.urlopen(req, timeout=45) as response:
            row.update(status=response.status, final_url=response.url,
                headers={k:v for k,v in response.headers.items() if k.lower() in ('content-length','content-range','etag','last-modified','content-type')})
            data = response.read(cap+1) if method != 'HEAD' else b''
            TRANSFERRED += len(data)
            if len(data)>cap or TRANSFERRED>MAX_TRANSFER:
                raise ValueError('metadata transfer budget exceeded')
            row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            return data, row
    except Exception as error:
        row['error'] = repr(error)
        raise
    finally:
        EVENTS.append(row)
        dump('source_access.json', {'transferred_bytes':TRANSFERRED,'cap_bytes':MAX_TRANSFER,'events':EVENTS})


def folder(fid):
    data, row = get('https://drive.google.com/drive/folders/'+fid)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL/(fid+'.html')).write_bytes(data)
    rows=[]
    for tag in re.findall(r'<tr\b.*?</tr>', data.decode('utf-8'), re.S):
        ident=re.search(r'data-id="([^"]+)"',tag)
        name=re.search(r'<strong[^>]*>(.*?)</strong>',tag,re.S)
        if ident and name:
            rows.append({'id':ident[1],'name':html.unescape(re.sub('<[^>]*>','',name[1])),
                         'description':html.unescape(re.sub('<[^>]*>',' ',tag))})
    return rows


class RemoteZip(io.RawIOBase):
    def __init__(self, ident):
        self.url='https://drive.usercontent.google.com/download?id='+ident+'&export=download&confirm=t'
        self.pos=0
        _, row=get(self.url, 0, method='HEAD')
        if row['headers'].get('Content-Length'):
            self.size=int(row['headers']['Content-Length'])
        else:
            _,row=get(self.url,1,{'Range':'bytes=0-0'})
            self.size=int(row['headers']['Content-Range'].split('/')[-1])
        self.version_headers=row['headers']
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos=off if whence==0 else self.pos+off if whence==1 else self.size+off
        return self.pos
    def read(self, n=-1):
        n=min(self.size-self.pos, n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        if n>4*1024**2:raise ValueError('range too large')
        data,row=get(self.url,n,{'Range':f'bytes={self.pos}-{self.pos+n-1}'})
        if row['status']!=206 or not row['headers'].get('Content-Range','').startswith(f'bytes {self.pos}-'):
            raise ValueError('Server did not honor range; no full archive fallback')
        self.pos+=len(data)
        return data


def main():
    # Conservative full-plan budget before reading even the archive directories.
    free=shutil.disk_usage(ROOT).free
    planned={'archives':22*1024**3,'extracted':24*1024**3,'new_environment_and_install_cache':16*1024**3,
             'checkpoints_logs_predictions':8*1024**3,'safety_margin':4*1024**3}
    budget={'free_bytes':free,'planning_upper_bounds':planned,'required_plus_min_free':sum(planned.values())+35*1024**3,
        'minimum_system_free_bytes':35*1024**3,'dense_full_resolution_mask_cache':'FORBIDDEN_BY_BUDGET_ON_THE_FLY_ONLY',
        'metadata_transfer_cap':MAX_TRANSFER,'no_legacy_cleanup':True}
    budget['full_plan_fits']=free>=budget['required_plus_min_free']
    dump('storage_budget_initial.json',budget)
    if free<35*1024**3+MAX_TRANSFER:raise OSError('Insufficient metadata audit reserve')
    ref_data,_=get('https://api.github.com/repos/BenyunZhao/CUBIT-InSeg/git/ref/heads/master')
    ref=json.loads(ref_data)
    dump('official_git_ref.json',ref)
    commit=ref['object']['sha']
    tree_data,_=get('https://api.github.com/repos/BenyunZhao/CUBIT-InSeg/git/trees/'+commit+'?recursive=1')
    tree=json.loads(tree_data)
    dump('official_git_tree.json',tree)
    sha=commit
    for name in ('README.md',):
        data,_=get('https://raw.githubusercontent.com/BenyunZhao/CUBIT-InSeg/'+sha+'/'+name)
        LOCAL.mkdir(parents=True,exist_ok=True)
        (LOCAL/name).write_bytes(data)
    roots=folder('1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b')
    result={'official_git_commit':sha,'official_git_tree':tree['sha'],'root_entries':roots,'splits':{}}
    dump('official_archive_inventory.json',result)
    for split in ('train','val','test'):
        item=next(x for x in roots if x['name']==split)
        entries=folder(item['id'])
        result['splits'][split]={'folder':item,'entries':entries,'archives':{}}
        for entry in entries:
            if not entry['name'].endswith('.zip'):continue
            remote=RemoteZip(entry['id'])
            with zipfile.ZipFile(remote) as z:
                files=[{'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':f'{i.CRC:08x}'} for i in z.infolist() if not i.is_dir()]
            result['splits'][split]['archives'][entry['name']]={'drive_id':entry['id'],'archive_bytes':remote.size,
                'version_headers':remote.version_headers,'uncompressed_bytes':sum(i['bytes'] for i in files),'files':files}
            dump('official_archive_inventory.json',result)
        print(split,[(n,a['archive_bytes'],len(a['files'])) for n,a in result['splits'][split]['archives'].items()],flush=True)
    print('Official Git',sha,'metadata bytes',TRANSFERRED,flush=True)


if __name__=='__main__':main()
