"""Bounded eight-image supplement using existing structural index; no test access."""
import gzip,json,pathlib,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
from scan import dump,sha
base=pathlib.Path(__file__).parent; out=base/'supplement';out.mkdir(exist_ok=False)
source=pathlib.Path('C:/D/resources/CUBIT-InSeg');start=time.perf_counter();cost=0
pool=json.loads((base/'derived_run2/pilot_pool.json').read_text()); used={r['image_id'] for r in pool}
with gzip.open(base/'derived_run2/images.jsonl.gz','rt') as f: rows=[json.loads(l) for l in f]
eligible=[r for r in rows if r['split']=='train' and not r['issues'] and r['image_id'] not in used and 2<=r['instances']<=15]
eligible.sort(key=lambda r:sha(('supplement-v1:'+r['image_id']).encode()))
chosen=[]
for cls in [0,1]: chosen.extend([r for r in eligible if r['class_ids_internal']==[cls]][:4])
with gzip.open(base/'derived_run2/instances.jsonl.gz','rt') as f:
    objs={r['image_id']:[] for r in chosen}
    for l in f:
        o=json.loads(l)
        if o['image_id'] in objs: objs[o['image_id']].append(o)
sheet=Image.new('RGB',(1280,1040),'white')
for i,r in enumerate(chosen,43):
    ip=source/r['image_relpath'];b=ip.read_bytes();cost+=len(b)
    r.update(image_sha256=sha(b),image_hash_status='sha256_current_candidate_only',pilot_order=i,
       target_options=[o['instance_id'] for o in objs[r['image_id']]],candidate_status='pending_visual_review',
       ambiguity_flags=['class_mapping_unconfirmed','annotation_completeness_unknown','building_group_unknown','semantic_reference_unverified'])
    lb=(source/r['label_relpath']).read_bytes();cost+=len(lb);assert sha(lb)==r['label_sha256']
    lines=lb.decode().splitlines()
    with Image.open(ip) as orig:
        orig.load();im=orig.convert('RGB');im.thumbnail((640,480));d=ImageDraw.Draw(im)
        for o in objs[r['image_id']]:
            xy=np.fromstring(lines[o['line_1based']-1],sep=' ')[1:].reshape(-1,2)*np.array(im.size)
            d.line([tuple(p) for p in xy]+[tuple(xy[0])],fill='red',width=1)
            d.text(tuple(xy[0]),str(o['line_1based']),fill='yellow')
        canvas=Image.new('RGB',(640,520),'#202020');canvas.paste(im,(0,30));ImageDraw.Draw(canvas).text((8,8),f'{i} {ip.stem} | DERIVED OVERLAY - NOT GT',fill='white')
        canvas.save(out/f'{i}_{ip.stem}.jpg',quality=85);canvas.thumbnail((320,520));j=i-43;sheet.paste(canvas,((j%4)*320,(j//4)*520))
sheet.save(out/'sheet.jpg',quality=90);dump(out/'pool.json',chosen)
dump(out/'cost.json',dict(elapsed_seconds=time.perf_counter()-start,logical_source_bytes=cost,image_hashes=8,test_reads=0))
