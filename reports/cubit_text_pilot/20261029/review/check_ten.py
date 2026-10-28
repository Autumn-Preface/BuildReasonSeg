"""Independent source-to-manifest validation for exactly ten authorized train candidates."""
import collections,gzip,hashlib,json,pathlib,time
from decimal import Decimal
from PIL import Image
R=pathlib.Path(__file__).resolve().parents[4];B=R/'research/cubit_text_pilot';OUT=pathlib.Path(__file__).resolve().parent
start=time.perf_counter();m=json.loads((B/'samples.json').read_text(encoding='utf8'));old={r['image_id']:r for r in json.loads((R/'reports/cubit_text_prep/20261029/pilot_candidates.json').read_text(encoding='utf8'))};source=pathlib.Path(m['source_root']);cost=0;rows=[]
sha=lambda b:hashlib.sha256(b).hexdigest()
want={'CR1_177','CR1_40','CR1_140','CR1_663','CR1_1775','SP1_1427','SP1_1133','SP0_70','SP1_436','SP0_794'}
quarantine=json.loads((R/'reports/cubit_text_prep/20261029/derived_run2/duplicate_quarantine.json').read_text(encoding='utf8'))
q={(s,pathlib.Path(p).stem) for g in quarantine['all_groups'] for s,p in g}
for s in m['images']:
 c=old[s['image_id']];ip=source/s['image_relpath'];lp=source/s['label_relpath'];ib=ip.read_bytes();lb=lp.read_bytes();cost+=len(ib)+len(lb);checks={}
 checks['train_only']=s['official_split']=='train' and s['image_id']==f"cubit-v1/train/{s['key']}" and s['key'] in want
 checks['both_raw_hashes_match_previous_and_manifest']=sha(ib)==s['image_sha256']==c['image_sha256'] and sha(lb)==s['label_sha256']==c['label_sha256']
 expected=[];areas=[]
 for n,line in enumerate(lb.decode('utf-8-sig').splitlines(),1):
  if not line.strip():continue
  a=[float(v) for v in line.split()];iid=f"{s['image_id']}/L{n:04d}/{sha(line.encode())[:16]}"
  expected.append(dict(instance_id=iid,line_1based=n,line_sha256=sha(line.encode()),numeric_class_id=int(a[0]),polygon_normalized=[list(p) for p in zip(a[1::2],a[2::2])]))
  d=[Decimal(v) for v in line.split()[1:]];p=list(zip(d[0::2],d[1::2]));areas.append(sum(x*p[(i+1)%len(p)][1]-y*p[(i+1)%len(p)][0] for i,(x,y) in enumerate(p))/2)
 checks['all_line_ids_numeric_classes_and_coordinates_exact']=s['instances']==expected and [x['instance_id'] for x in expected]==c['target_options']
 checks['no_exact_zero_signed_area']=all(a!=0 for a in areas)
 checks['outside_all_historical_crc_candidates']=('train',s['key']) not in q
 checks['risks_unresolved_not_formal']=s['class_mapping_status']=='unconfirmed' and s['valid_domain']=='unknown' and s['building_group'] is None and s['candidate_status']=='pending_human_review' and 'annotation_completeness_unknown' in s['risk_flags']
 with Image.open(ip) as im:checks['original_dimensions_orientation']=list(im.size)==s['width_height']==c['width_height'] and im.getexif().get(274,1)==1
 checks['original_size_mtime_still_match_preparation']=all((pathlib.Path(v['path']).stat().st_size,pathlib.Path(v['path']).stat().st_mtime_ns)==(v['bytes'],v['mtime_ns']) for v in s['source_metadata'])
 rows.append(dict(key=s['key'],checks=checks,instances=len(expected)))
checks={'exact_ten':len(m['images'])==10 and {s['key'] for s in m['images']}==want,'unique_image_hashes':len({s['image_sha256'] for s in m['images']})==10,'prior_commit_expected':m['prior_commit']=='4b9cc90c39a9d6375f81445935ad2bc7d3dc4318','prior_candidate_file_sha_match':m['prior_candidates_sha256']==sha((R/'reports/cubit_text_prep/20261029/pilot_candidates.json').read_bytes()),'four_gate_recorded':json.loads((OUT.parent/'four_gate.json').read_text())['gate']=='PASS_BEFORE_OTHER_SIX_PREPARATION','all_per_image_checks':all(all(r['checks'].values()) for r in rows)}
result=dict(all_pass=all(checks.values()),checks=checks,images=rows,source_bytes_read=cost,elapsed_seconds=time.perf_counter()-start,image_pixel_decodes=0,test_images_accessed=0,scope='Original bytes, header sizes/orientation and polygon provenance only; not human visual acceptance or proof of no scene leakage')
(OUT/'ten_source_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2));assert result['all_pass']
