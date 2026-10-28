"""Read-only CUBIT structural index; no model, network, training or source writes.
Run with existing P1A Python: scan.py --source ... --repo ... --out NEW_DIRECTORY
Output must not exist. Official test is structural metadata only, never selection.
"""
import argparse, collections, gzip, hashlib, json, pathlib, time
import numpy as np
from PIL import Image, ImageDraw

def sha(b): return hashlib.sha256(b).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def geom(line):
    try:
        a=np.fromstring(line,sep=' ')
    except ValueError:
        return {'issue':'invalid_format_or_coordinates'}
    if len(a)!=len(line.split()) or len(a)<7 or len(a)%2!=1 or not np.isfinite(a).all() or a[0] not in (0,1) or np.any((a[1:]<0)|(a[1:]>1)):
        return {'issue':'invalid_format_or_coordinates'}
    p=a[1:].reshape(-1,2); x,y=p.T
    area=abs(float(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))/2
    box=[float(x.min()),float(y.min()),float(x.max()),float(y.max())]
    return dict(class_id_internal=int(a[0]),area_normalized=area,bbox_normalized=box,
                bbox_center=[(box[0]+box[2])/2,(box[1]+box[3])/2],vertices=len(p),issue='zero_area' if area<=1e-14 else None)

def run(source,repo,out):
    start=time.perf_counter(); out.mkdir(parents=True,exist_ok=False)
    audit=repo/'reports/damage_p1a/20261025'
    ap=audit/'label_audit.json'; mp=audit/'label_file_manifest.json'
    a=json.loads(ap.read_text()); old={(r['split'],r['path']):r for r in json.loads(mp.read_text())}
    groups=a['archive_crc_size_duplicate_candidates']; excluded=set(); cross=[]
    for n,g in enumerate(groups):
        for s,p in g: excluded.add((s,pathlib.Path(p).stem))
        if len({s for s,p in g})>1: cross.append(g)
    dump(out/'duplicate_quarantine.json',dict(source=str(ap),source_sha256=sha(ap.read_bytes()),
          evidence='historical CRC32+size candidates, not upgraded to SHA256 proof',
          cross_split_groups=len(cross),all_groups=groups,policy='exclude every listed member from pilot; no split edits'))
    rows=[]; anomalies=[]; counts=collections.Counter(); bytes_read=0; snapshots=[]
    with gzip.open(out/'instances.jsonl.gz','wt',encoding='utf8') as inst:
        for split in ('train','val','test'):
            images={p.stem:p for p in sorted((source/split/'images').iterdir()) if p.is_file()}
            labels={p.stem:p for p in sorted((source/split/'labels').iterdir()) if p.is_file()}
            for stem in sorted(set(images)|set(labels)):
                ip=images.get(stem); lp=labels.get(stem); issues=[]; objs=[]
                ident=f'cubit-v1/{split}/{stem}'
                if not ip or not lp: issues.append('unpaired')
                lb=lp.read_bytes() if lp else b''; bytes_read+=len(lb); lh=sha(lb)
                hist=old.get((split,f'labels/{stem}.txt'))
                if not hist or hist['sha256']!=lh: issues.append('historical_label_hash_mismatch')
                for lineno,line in enumerate(lb.decode('utf-8-sig').splitlines(),1):
                    if not line.strip(): continue
                    o=geom(line); o.update(instance_id=f'{ident}/L{lineno:04d}/{sha(line.encode())[:16]}',image_id=ident,
                        line_1based=lineno,label_sha256=lh,line_sha256=sha(line.encode()))
                    objs.append(o); inst.write(json.dumps(o)+'\n')
                    if o['issue']: anomalies.append(o); issues.append(o['issue'])
                if (split,stem) in excluded: issues.append('historical_duplicate_candidate')
                wh=None
                if ip and split=='train':
                    try:
                        with Image.open(ip) as im: wh=list(im.size)
                    except Exception: issues.append('image_header_error')
                row=dict(image_id=ident,split=split,image_relpath=f'{split}/images/{ip.name}' if ip else None,
                    label_relpath=f'{split}/labels/{lp.name}' if lp else None,label_sha256=lh,
                    image_sha256=None,image_hash_status='not_rehashed',width_height=wh,
                    image_bytes=ip.stat().st_size if ip else None,instances=len(objs),
                    class_ids_internal=sorted({o.get('class_id_internal') for o in objs if 'class_id_internal' in o}),
                    issues=sorted(set(issues)),building_group=None,annotation_valid_domain='unknown',
                    _objects=objs)
                rows.append(row); counts[split]+=1
                for p in (ip,lp):
                    if p: st=p.stat(); snapshots.append([str(p),st.st_size,st.st_mtime_ns])
    # Pilot strata are scheduling criteria, not scientific thresholds or training labels.
    eligible=[r for r in rows if r['split']=='train' and not r['issues'] and r['instances']>0]
    chosen=[]
    for cls in (0,1):
        for lo,hi in ((1,1),(2,5),(6,15),(16,100000)):
            pool=[r for r in eligible if r['class_ids_internal']==[cls] and lo<=r['instances']<=hi]
            pool.sort(key=lambda r:sha(('pilot-v1:'+r['image_id']).encode()))
            chosen.extend(pool[:6])
    preview=out/'preview_non_gt'; preview.mkdir()
    for n,r in enumerate(chosen):
        ip=source/r['image_relpath']; raw=ip.read_bytes(); bytes_read+=len(raw)
        r['image_sha256']=sha(raw); r['image_hash_status']='sha256_current_candidate_only'
        r['pilot_order']=n+1; r['candidate_status']='pending_visual_review'
        r['screening_reason']='train; outside historical duplicate groups; valid polygon geometry; class/instance-count diversity'
        r['ambiguity_flags']=['class_mapping_unconfirmed','annotation_completeness_unknown','building_group_unknown','semantic_reference_unverified']
        with Image.open(ip) as orig:
            orig.load(); im=orig.convert('RGB'); im.thumbnail((640,480)); d=ImageDraw.Draw(im)
            for o in r['_objects']:
                l=o['line_1based']; line=(source/r['label_relpath']).read_text().splitlines()[l-1]
                xy=np.fromstring(line,sep=' ')[1:].reshape(-1,2)*np.array(im.size)
                d.line([tuple(p) for p in xy]+[tuple(xy[0])], fill=(255,60,50),width=1)
                b=o['bbox_normalized'];d.text((b[0]*im.width,b[1]*im.height),str(l),fill=(255,255,0))
            canvas=Image.new('RGB',(640,520),'#202020');canvas.paste(im,(0,30))
            ImageDraw.Draw(canvas).text((8,8),f'{n+1:02} {ip.stem} | DERIVED REVIEW OVERLAY - NOT GT',fill='white')
            canvas.save(preview/f'{n+1:02}_{ip.stem}.jpg',quality=85)
    # Contact sheets: each preview remains inspectable on its own.
    for batch in range((len(chosen)+11)//12):
        sheet=Image.new('RGB',(1280,1560),'white')
        for j,r in enumerate(chosen[batch*12:(batch+1)*12]):
            p=next(preview.glob(f'{r["pilot_order"]:02}_*.jpg'))
            with Image.open(p) as im: im.thumbnail((320,520)); sheet.paste(im,((j%4)*320,(j//4)*520))
        sheet.save(preview/f'sheet_{batch+1}.jpg',quality=90)
    def public(r): return {k:v for k,v in r.items() if k!='_objects'}
    with gzip.open(out/'images.jsonl.gz','wt',encoding='utf8') as f:
        for r in rows: f.write(json.dumps(public(r))+'\n')
    dump(out/'pilot_pool.json',[dict(public(r),target_options=[o['instance_id'] for o in r['_objects']]) for r in chosen])
    dump(out/'anomalies.json',anomalies); dump(out/'source_metadata_before.json',snapshots)
    changes=[]
    for p,size,mt in snapshots:
        st=pathlib.Path(p).stat()
        if (st.st_size,st.st_mtime_ns)!=(size,mt): changes.append(p)
    summary=dict(counts=dict(counts),instances=sum(r['instances'] for r in rows),anomalies=len(anomalies),
        anomaly_counts=dict(collections.Counter(o['issue'] for o in anomalies)),eligible_train=len(eligible),pilot_pool=len(chosen),
        label_hash_mismatches=sum('historical_label_hash_mismatch' in r['issues'] for r in rows),
        cross_duplicate_groups=len(cross),source_metadata_changes=changes,elapsed_seconds=time.perf_counter()-start,
        counted_label_and_candidate_image_bytes_read=bytes_read,candidate_image_hash_count=len(chosen),
        other_image_hash_count=0,test_image_decode_count=0,test_selection_count=0,
        byte_cost_caveat='logical bytes; excludes Pillow header reads, repeated candidate label reads and filesystem/cache IO',
        image_hash_coverage='candidate only; all other image hashes deferred, never fabricated',
        audit_sha256=sha(ap.read_bytes()),historical_manifest_sha256=sha(mp.read_bytes()))
    dump(out/'scan_cost.json',summary); print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
    args=p.parse_args()
    assert not args.out.resolve().is_relative_to(args.source.resolve()),'output must be outside raw dataset'
    run(args.source,args.repo,args.out)
