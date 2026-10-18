import sys
sys.stdout.reconfigure(encoding='utf-8')
from acquire_samples import get,save,RemoteZip,ROOT,LOCAL,events
import json,zipfile,collections,datetime
from pathlib import Path

def cubit_spall():
    p=ROOT/'evaluation/damage_p0/cubit_folder_inventory.json'
    inv=json.loads(p.read_text(encoding='utf-8'))
    ids={x['name']:x['id'] for x in inv['train']}
    with zipfile.ZipFile(RemoteZip('https://drive.usercontent.google.com/download?id='+ids['images.zip']+'&export=download&confirm=t')) as zi,zipfile.ZipFile(RemoteZip('https://drive.usercontent.google.com/download?id='+ids['labels.zip']+'&export=download&confirm=t')) as zl:
        imgs=sorted(x.filename for x in zi.infolist() if x.filename.lower().endswith(('.jpg','.jpeg','.png')))
        labs={Path(x.filename).stem:x.filename for x in zl.infolist() if x.filename.endswith('.txt')}
        print('CUBIT prefixes',collections.Counter(Path(x).stem.split('_')[0] for x in imgs).most_common(12),flush=True)
        inv['image_names']=imgs
        candidates=[x for x in imgs if not Path(x).stem.upper().startswith('CR')]
        selected=[]
        for im in candidates[:8]:
            lab=labs.get(Path(im).stem)
            if not lab:continue
            b=zl.read(lab)
            cls={l.split()[0] for l in b.decode().splitlines() if l.strip()}
            if '1' not in cls:continue
            for name,data in [(lab,b),(im,zi.read(im))]:
                q=LOCAL/'cubit'/Path(name).name;q.write_bytes(data)
                import hashlib
                events.append({'archive_entry':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'local':str(q.relative_to(ROOT))})
            selected.append(im)
            if len(selected)>=2:break
        inv['selected']+=selected
        p.write_text(json.dumps(inv,indent=2,ensure_ascii=False),encoding='utf-8')
        print('CUBIT extra selected',selected,flush=True)

def cif():
    rows=json.loads((ROOT/'evaluation/damage_p0/sources/cif_train_rows.txt').read_text(encoding='utf-8'))
    for r in rows['rows'][:2]:
        row=r['row'];save(row['image']['src'],'cif/'+row['file_name'])
        row=dict(row);row.pop('image')
        p=LOCAL/'cif'/(Path(row['file_name']).stem+'.json')
        p.write_text(json.dumps(row,ensure_ascii=False),encoding='utf-8')
    print('CiF pairs 2',flush=True)

for n,fn in [('cubit_extra',cubit_spall),('cif',cif)]:
    try:fn()
    except Exception as e:events.append({'dataset':n,'error':repr(e)});print(n,repr(e),flush=True)
p=ROOT/'evaluation/damage_p0/sample_access.json'
old=json.loads(p.read_text(encoding='utf-8'));old['events']+=events
old['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
old['transferred_bytes']+=sum(x.get('bytes',0) for x in events if 'url' in x)
# Expiring public asset signatures are not needed for reproducibility.
for row in old['events']:
    if '/cached-assets/' in row.get('url',''):row['url']=row['url'].split('?')[0]
p.write_text(json.dumps(old,indent=2,ensure_ascii=False),encoding='utf-8')
