"""Bounded follow-up: final candidate integrity and repaired-parser adversarial checks."""
import collections, gzip, hashlib, importlib.util, json, pathlib, time
B=pathlib.Path(__file__).resolve().parents[1]
S=pathlib.Path('C:/D/resources/CUBIT-InSeg')
def load(p):return json.loads(p.read_text(encoding='utf8'))
def sha(b):return hashlib.sha256(b).hexdigest()
start=time.perf_counter();cost=0;checks={}
final=load(B/'pilot_candidates.json'); old=load(B/'derived_run2/pilot_pool.json');extra=load(B/'supplement/pool.json');reject=load(B/'pilot_rejections.json')
allpool={r['image_id']:r for r in old+extra};oldids={r['image_id'] for r in old}
images={r['image_id']:r for r in (json.loads(x) for x in gzip.open(B/'derived_run2/images.jsonl.gz','rt'))}
instances={r['instance_id']:r for r in (json.loads(x) for x in gzip.open(B/'derived_run2/instances.jsonl.gz','rt'))}
q=load(B/'derived_run2/duplicate_quarantine.json');excluded={(s,pathlib.Path(p).stem) for g in q['all_groups'] for s,p in g}
checks['exactly_31_unique_train']=len(final)==31 and len({r['image_id'] for r in final})==31 and all(r['split']=='train' for r in final)
checks['all_final_avoid_historical_duplicate_candidates']=all((r['split'],r['image_id'].split('/')[-1]) not in excluded for r in final)
checks['all_final_issue_free']=all(not r['issues'] for r in final)
checks['all_final_label_hashes_match_index']=all(r['label_sha256']==images[r['image_id']]['label_sha256'] for r in final)
checks['all_final_source_image_hashes_match_pool']=all(r['image_sha256']==allpool[r['image_id']]['image_sha256'] for r in final)
checks['all_instance_targets_resolve']=all(all(i in instances and instances[i]['image_id']==r['image_id'] and instances[i]['label_sha256']==r['label_sha256'] for i in r['target_options']) for r in final)
checks['focus_is_target_option']=all(r['focus_instance_id'] in r['target_options'] for r in final)
checks['not_human_annotations_no_instructions']=all(r['candidate_status']=='codex_thumbnail_triaged_not_human_annotated' and r['instruction_text'] is None for r in final)
checks['all_previews_exist']=all((B/r['preview_relpath']).is_file() for r in final)
checks['all_50_triage_accounted']=len(allpool)==50 and len(reject)==19 and {r['image_id'] for r in final}.isdisjoint({r['image_id'] for r in reject}) and {r['image_id'] for r in final}|{r['image_id'] for r in reject}==set(allpool)
checks['final_hashes_unique']=len({r['image_sha256'] for r in final})==31
# Original 42 image hashes were all independently matched in prior recorded check.
checks['prior_42_hash_check_pass']=load(B/'review/independent_results.json')['checks']['bounded_pilot_image_hashes_match']
new_hash=[]
for r in final:
    if r['image_id'] not in oldids:
        b=(S/r['image_relpath']).read_bytes();cost+=len(b);new_hash.append(sha(b)==r['image_sha256'])
checks['two_supplement_hashes_match_source']=len(new_hash)==2 and all(new_hash)
spec=importlib.util.spec_from_file_location('subject',B/'scan.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cases={
 'bad_token':('0 0.1 0.1 0.4 0.1 0.2 0.5 junk','invalid_format_or_coordinates'),
 'trailing_numeric_token':('0 0.1 0.1 0.4 0.1 0.2 0.5 7','invalid_format_or_coordinates'),
 'nonfinite':('0 0.1 nan 0.4 0.1 0.2 0.5','invalid_format_or_coordinates'),
 'outside_domain':('0 0.1 0.1 1.4 0.1 0.2 0.5','invalid_format_or_coordinates'),
 'unknown_class':('2 0.1 0.1 0.4 0.1 0.2 0.5','invalid_format_or_coordinates'),
 'collinear_exact_integer':('0 0.1 0.2 0.2 0.2 0.3 0.2','zero_area'),
 'positive_triangle':('0 0.1 0.1 0.4 0.1 0.2 0.5',None)}
probes={k:m.geom(line)['issue']==want for k,(line,want) in cases.items()}
checks['parser_adversarial_7_pass']=all(probes.values())
checks['prior_13_decimal_degenerate_check_pass']=load(B/'review/independent_results.json')['checks']['all_reported_degenerates_exact_zero']
result=dict(checks=checks,parser_probes=probes,all_pass=all(checks.values()),new_image_hashes_checked=len(new_hash),
 source_bytes_read=cost,elapsed_seconds=time.perf_counter()-start,
 source_cost_cumulative_prior_two_rounds_plus_this=272995358+cost,
 opportunities=dict(collections.Counter(k for r in final for k in r['pilot_task_opportunities'])),
 limitation='thumbnail triage remains provisional; no independent scientific or annotation acceptance')
(B/'review/final_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result))
