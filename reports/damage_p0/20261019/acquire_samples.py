"""Small, bounded public samples; no credentials, no training, no archive extraction to arbitrary paths."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import concurrent.futures, datetime, hashlib, html, io, json, re, struct, urllib.request, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'datasets/damage/local'
LOCAL.mkdir(parents=True,exist_ok=True)
events=[]
transferred=0

def get(url,cap=8*1024*1024,headers=None,method=None):
    global transferred
    req=urllib.request.Request(url,headers={'User-Agent':'Damage-P0 research sample audit',**(headers or {})},method=method)
    row={'url':url,'method':method or 'GET'}
    try:
        with urllib.request.urlopen(req,timeout=30) as r:
            row.update(status=r.status,content_length=r.headers.get('Content-Length'),content_range=r.headers.get('Content-Range'))
            data=r.read(cap+1)
            transferred+=len(data)
            if len(data)>cap:raise ValueError(f'response exceeds cap {cap}; aborted')
            row.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            return data,row
    except Exception as e:
        row['error']=repr(e)
        raise
    finally:events.append(row)

def save(url,path,cap=8*1024*1024):
    data,row=get(url,cap)
    p=LOCAL/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    row['local']=str(p.relative_to(ROOT))
    return data

def folder(fid):
    raw=get('https://drive.google.com/drive/folders/'+fid,2*1024*1024)[0].decode('utf-8')
    rows=[]
    for tag in re.findall(r'<tr\b.*?</tr>',raw,re.S):
        m=re.search(r'data-id="([^"]+)"',tag)
        n=re.search(r'<strong[^>]*>(.*?)</strong>',tag,re.S)
        if m and n:
            rows.append({'id':m[1],'name':html.unescape(re.sub('<[^>]*>','',n[1])),
                         'text':html.unescape(re.sub('<[^>]*>',' ',tag))})
    return rows

class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0;self.bytes=0
        _,row=get(url,0,method='HEAD')
        if row.get('content_length'):
            self.size=int(row['content_length'])
        else:
            _,row=get(url,1,{'Range':'bytes=0-0'})
            self.size=int(row['content_range'].split('/')[-1])
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        if n>8*1024*1024 or self.bytes+n>24*1024*1024:raise ValueError('remote zip range budget exceeded')
        data,row=get(self.url,n,{'Range':f'bytes={self.pos}-{self.pos+n-1}'})
        if row['status']!=206 or not row.get('content_range','').startswith(f'bytes {self.pos}-'):
            raise ValueError('server did not honor byte range; refusing full archive')
        self.pos+=len(data);self.bytes+=len(data)
        return data

def cubit():
    root=folder('1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b')
    train=next(x for x in root if x['name']=='train')
    entries=folder(train['id'])
    inventory={'root':root,'train':entries}
    for d in entries:
        if d['name'].lower() in ['images','labels','image','label']:
            inventory[d['name']]=folder(d['id'])
    (ROOT/'evaluation/damage_p0/cubit_folder_inventory.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False),encoding='utf-8')
    print('CUBIT train entries',[(x['name'],x['id']) for x in entries],flush=True)
    if any(x['name']=='images.zip' for x in entries):
        archives={x['name']:x for x in entries}
        iz=RemoteZip('https://drive.usercontent.google.com/download?id='+archives['images.zip']['id']+'&export=download&confirm=t')
        lz=RemoteZip('https://drive.usercontent.google.com/download?id='+archives['labels.zip']['id']+'&export=download&confirm=t')
        with zipfile.ZipFile(iz) as zi,zipfile.ZipFile(lz) as zl:
            images=sorted(x.filename for x in zi.infolist() if x.filename.lower().endswith(('.jpg','.png','.jpeg')))
            labels={Path(x.filename).stem:x.filename for x in zl.infolist() if x.filename.endswith('.txt')}
            picked=[]
            for im in images:
                if Path(im).stem not in labels:continue
                lab=labels[Path(im).stem]
                for zz,n in [(zi,im),(zl,lab)]:
                    data=zz.read(n);p=LOCAL/'cubit'/Path(n).name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
                    events.append({'archive_entry':n,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'local':str(p.relative_to(ROOT))})
                picked.append(im)
                if len(picked)==2:break
            inventory.update(image_archive_bytes=iz.size,label_archive_bytes=lz.size,
                             image_count=len(images),label_count=len(labels),selected=picked)
        (ROOT/'evaluation/damage_p0/cubit_folder_inventory.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False),encoding='utf-8')
        print('CUBIT sampled',len(picked),flush=True)
        return
    # Only download matching publicly exposed first two images and labels, if available.
    imgs=inventory.get('images',inventory.get('image',[]));labs=inventory.get('labels',inventory.get('label',[]))
    pairs=[]
    for im in imgs:
        stem=Path(im['name']).stem
        lab=next((x for x in labs if Path(x['name']).stem==stem),None)
        if not lab:continue
        for x in [im,lab]:save('https://drive.google.com/uc?export=download&id='+x['id'],'cubit/'+x['name'])
        pairs.append((im['name'],lab['name']))
        if len(pairs)>=2:break
    print('CUBIT pairs',pairs,flush=True)

def deepcrack():
    u='https://raw.githubusercontent.com/yhlleo/DeepCrack/master/dataset/DeepCrack.zip'
    r=RemoteZip(u)
    with zipfile.ZipFile(r) as z:
        entries=[{'name':x.filename,'bytes':x.file_size,'compressed':x.compress_size} for x in z.infolist()]
        (ROOT/'evaluation/damage_p0/deepcrack_archive_index.json').write_text(json.dumps({'archive_bytes':r.size,'entries':entries},indent=2),encoding='utf-8')
        imgs=sorted(x.filename for x in z.infolist() if x.filename.startswith('train_img/') and not x.is_dir())[:3]
        for im in imgs:
            stem=Path(im).stem
            lab=next((x.filename for x in z.infolist() if x.filename.startswith('train_lab/') and Path(x.filename).stem==stem),None)
            if not lab:continue
            for name in [im,lab]:
                data=z.read(name)
                p=LOCAL/'deepcrack'/Path(name).name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
                events.append({'archive_entry':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'local':str(p.relative_to(ROOT))})
    print('DeepCrack sampled',len(imgs),'archive',r.size,'range bytes',r.bytes,flush=True)

def dacl():
    u='https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_devphase.zip'
    r=RemoteZip(u)
    with zipfile.ZipFile(r) as z:
        names=z.namelist()
        # Training examples only; selection by polygon classes, no model output.
        chosen=[]
        for name in sorted(n for n in names if 'annotations/train/' in n and n.endswith('.json'))[:12]:
            data=z.read(name);d=json.loads(data)
            labels={s['label'] for s in d['shapes']}
            if not labels.intersection({'Crack','Spalling'}):continue
            image=next((n for n in names if 'images/train/' in n and Path(n).name==d['imageName']),None)
            if not image:continue
            for n,b in [(name,data),(image,z.read(image))]:
                p=LOCAL/'dacl'/Path(n).name;p.parent.mkdir(exist_ok=True);p.write_bytes(b)
                events.append({'archive_entry':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'local':str(p.relative_to(ROOT))})
            chosen.append({'annotation':name,'labels':sorted(labels)})
            if len(chosen)==2:break
        (ROOT/'evaluation/damage_p0/dacl_sample_selection.json').write_text(json.dumps({'archive_bytes':r.size,'range_bytes':r.bytes,'chosen':chosen},indent=2),encoding='utf-8')
    print('dacl sampled',len(chosen),'archive',r.size,'range bytes',r.bytes,flush=True)

if __name__=='__main__':
    for name,fn in [('cubit',cubit),('deepcrack',deepcrack),('dacl',dacl)]:
        try:fn()
        except Exception as e:events.append({'dataset':name,'error':repr(e)});print(name,repr(e),flush=True)
    p=ROOT/'evaluation/damage_p0/sample_access.json'
    previous=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    p.write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'transferred_bytes':transferred+previous.get('transferred_bytes',0),'events':previous.get('events',[])+events},indent=2,ensure_ascii=False),encoding='utf-8')
