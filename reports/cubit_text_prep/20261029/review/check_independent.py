"""Independent bounded review of derived index; reads only 13 anomaly labels and pilot images."""
import collections, decimal, gzip, hashlib, importlib.util, json, pathlib, time

BASE=pathlib.Path(__file__).resolve().parents[1]
DATA=BASE/'derived_run2'
SOURCE=pathlib.Path('C:/D/resources/CUBIT-InSeg')
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf8'))
start=time.perf_counter(); checks={}; read_bytes=0
images=[json.loads(x) for x in gzip.open(DATA/'images.jsonl.gz','rt')]
instances=[json.loads(x) for x in gzip.open(DATA/'instances.jsonl.gz','rt')]
pool=load(DATA/'pilot_pool.json'); anomalies=load(DATA/'anomalies.json'); q=load(DATA/'duplicate_quarantine.json')
im={r['image_id']:r for r in images}; byim=collections.defaultdict(list)
for r in instances:byim[r['image_id']].append(r)
checks['counts_match_6996_62187']=len(images)==6996 and len(instances)==62187
checks['unique_ids']=len(im)==len(images) and len({r['instance_id'] for r in instances})==len(instances)
checks['links_and_label_hash']=all(r['image_id'] in im and r['label_sha256']==im[r['image_id']]['label_sha256'] for r in instances)
checks['counts_per_image']=all(len(byim[r['image_id']])==r['instances'] for r in images)
excluded={(s,pathlib.Path(p).stem) for g in q['all_groups'] for s,p in g}
checks['pool_train_only']=all(r['split']=='train' for r in pool)
checks['pool_avoids_duplicate_candidates']=all((r['split'],r['image_id'].split('/')[-1]) not in excluded for r in pool)
checks['pool_geometry_and_identity_issues_empty']=all(not r['issues'] for r in pool)
checks['pool_targets_resolve']=all(set(r['target_options'])=={i['instance_id'] for i in byim[r['image_id']]} for r in pool)
hash_results=[]
for r in pool:
    b=(SOURCE/r['image_relpath']).read_bytes();read_bytes+=len(b)
    hash_results.append(sha(b)==r['image_sha256'])
checks['bounded_pilot_image_hashes_match']=all(hash_results)
exact=[]
for a in anomalies:
    split,stem=a['image_id'].split('/')[-2:]
    b=(SOURCE/split/'labels'/f'{stem}.txt').read_bytes();read_bytes+=len(b)
    line=b.decode('utf-8-sig').splitlines()[a['line_1based']-1]
    v=[decimal.Decimal(x) for x in line.split()[1:]];pts=list(zip(v[::2],v[1::2]))
    twice=sum(x*pts[(j+1)%len(pts)][1]-y*pts[(j+1)%len(pts)][0] for j,(x,y) in enumerate(pts))
    exact.append(dict(instance_id=a['instance_id'],exact_twice_signed_area=str(twice),
        line_sha256_matches=sha(line.encode())==a['line_sha256']))
checks['all_reported_degenerates_exact_zero']=all(decimal.Decimal(r['exact_twice_signed_area'])==0 for r in exact)
checks['anomaly_line_hashes_match']=all(r['line_sha256_matches'] for r in exact)
spec=importlib.util.spec_from_file_location('subject',BASE/'scan.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
try:
    probe=mod.geom('0 0.1 0.1 0.4 0.1 0.2 0.5 junk')
    checks['quarantines_trailing_nonnumeric_token']=probe['issue']=='invalid_format_or_coordinates'
    parser_exception=None
except ValueError as e:
    checks['quarantines_trailing_nonnumeric_token']=False
    parser_exception=str(e)
result=dict(checks=checks,exact_degenerate_checks=exact,candidate_hashes_checked=len(hash_results),
    bounded_source_bytes_read=read_bytes,elapsed_seconds=time.perf_counter()-start,
    parser_exception=parser_exception,
    scope='independent artifact consistency plus bounded source checks; not scientific approval or full image duplicate audit')
(BASE/'review'/'independent_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=True))
