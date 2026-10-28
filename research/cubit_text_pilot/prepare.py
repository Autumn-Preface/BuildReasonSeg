"""Bounded provenance preparation. Only four smoke samples, then authorized ten."""
import argparse, gzip, hashlib, json, math
from pathlib import Path
from PIL import Image
BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
PRIOR = REPO/'reports/cubit_text_prep/20261029'
FIRST = ['CR1_177','CR1_40','SP1_1427','SP0_794']
TEN = ['CR1_177','CR1_40','CR1_140','CR1_663','CR1_1775','SP1_1427','SP1_1133','SP0_70','SP1_436','SP0_794']
def sha(b): return hashlib.sha256(b).hexdigest()
def read_json(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def prepare(source, names):
    source=source.resolve()
    old={r['image_id'].split('/')[-1]:r for r in read_json(PRIOR/'pilot_candidates.json')}
    index={r['instance_id']:r for r in (json.loads(x) for x in gzip.open(PRIOR/'derived_run2/instances.jsonl.gz','rt',encoding='utf8')) if r['image_id'].split('/')[-1] in names and '/train/' in r['image_id']}
    records=[]
    for name in names:
        c=old[name]; assert c['split']=='train' and not c['issues']
        ip=source/c['image_relpath']; lp=source/c['label_relpath']
        assert ip.resolve().is_relative_to(source/'train/images') and lp.resolve().is_relative_to(source/'train/labels')
        ib=ip.read_bytes(); lb=lp.read_bytes()
        assert sha(ib)==c['image_sha256'] and sha(lb)==c['label_sha256'],name+' hash mismatch'
        with Image.open(ip) as im:
            wh=list(im.size); assert im.getexif().get(274,1)==1,'EXIF orientation requires explicit review'
        assert wh==c['width_height']
        instances=[]
        for n,line in enumerate(lb.decode('utf-8-sig').splitlines(),1):
            if not line.strip(): continue
            a=[float(t) for t in line.split()]
            assert len(a)>=7 and len(a)%2==1 and a[0] in (0,1) and all(math.isfinite(x) for x in a) and all(0<=x<=1 for x in a[1:])
            iid=f"{c['image_id']}/L{n:04d}/{sha(line.encode())[:16]}"
            assert index[iid]['line_sha256']==sha(line.encode()) and index[iid]['label_sha256']==sha(lb)
            instances.append(dict(instance_id=iid,line_1based=n,line_sha256=sha(line.encode()),numeric_class_id=int(a[0]),polygon_normalized=list(zip(a[1::2],a[2::2]))))
        assert [r['instance_id'] for r in instances]==c['target_options']
        records.append(dict(key=name,image_id=c['image_id'],official_split='train',image_relpath=c['image_relpath'],label_relpath=c['label_relpath'],image_sha256=sha(ib),label_sha256=sha(lb),width_height=wh,instances=instances,source_metadata=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in (ip,lp)],prior_observation=c['visual_observation_provisional'],risk_flags=c['ambiguity_flags'],class_mapping_status='unconfirmed',valid_domain='unknown',building_group=None,candidate_status='pending_human_review'))
    return dict(schema_version='pilot-source-1',source_root=str(source),prior_commit='4b9cc90c39a9d6375f81445935ad2bc7d3dc4318',prior_candidates_sha256=sha((PRIOR/'pilot_candidates.json').read_bytes()),images=records)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('C:/D/resources/CUBIT-InSeg'));p.add_argument('--stage',choices=['four','ten'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert a.out.resolve().is_relative_to(REPO) and not a.out.exists(),'new project-local output required'
    result=prepare(a.source,FIRST if a.stage=='four' else TEN)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'images':len(result['images']),'all_source_hashes_match':True,'path':str(a.out),'source_writes':0}))
