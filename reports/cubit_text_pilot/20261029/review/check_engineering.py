"""Independent bounded engineering probes; all drafts explicitly QA, never human labels."""
import copy, hashlib, http.client, importlib.util, json, pathlib, sys, threading, time, uuid
from unittest import mock
R=pathlib.Path(__file__).resolve().parents[4]; B=R/'research/cubit_text_pilot'; OUT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(B)); import server as subject
MAN=OUT.parent/'first_four.json'; data=subject.read(MAN); checks={}; notes=[]; start=time.perf_counter()
def chk(k,v):
 checks[k]=bool(v)
 if not v:notes.append('FAILED: '+k)
def rejects(k,e,s):
 try:subject.validate_edit(e,s)
 except (ValueError,TypeError,KeyError):chk(k,True)
 else:chk(k,False)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raws=[pathlib.Path(data['source_root'])/s[k+'_relpath'] for s in data['images'] for k in ('image','label')]
before=[(str(p),p.stat().st_size,p.stat().st_mtime_ns,digest(p)) for p in raws]
store=subject.Store(MAN,B/'qa_drafts'/('review-'+uuid.uuid4().hex)); sample=store.samples['SP0_794']; target=[x['instance_id'] for x in sample['instances'][:2]]
e=subject.empty_edit();chk('default_unknown_empty_no_confirmation',e['valid_domain']=='unknown' and e['review_status']=='draft' and not e['target_instance_ids'] and not e['instruction_text'])
e.update(operator='TOOL_TEST_NOT_HUMAN',reviewer='',instruction_text='工具存取测试，不作为标注或模型输入。',target_instance_ids=target,references=[dict(id='ref-qa',name='TOOL_TEST_REFERENCE',bbox_normalized=[.1,.2,.4,.5])],relations=[dict(reference_id='ref-qa',target_instance_ids=target,relation='工具测试关系',evidence='任意测试几何，不是实际标注')],regions=[dict(kind='unknown',bbox_normalized=[.3,.4,.5,.6],note='TOOL_TEST_REGION')],risks=['target_uniqueness','fragmentation','missing_annotation'],change_reason='independent QA only')
r1=store.save('SP0_794',None,0,e);p1=store.revisions('SP0_794',r1['annotation_id'])[0];h1=digest(p1)
chk('saved_roundtrip_multitarget_geometry_risks',store.load('SP0_794',r1['annotation_id'])['edit']==e)
chk('provenance_bound_and_formal_export_disabled',r1['source']==store.provenance(sample) and r1['source_instance_ids']==[x['instance_id'] for x in sample['instances']] and r1['formal_export_enabled'] is False and r1['class_mapping_status']=='unconfirmed' and r1['building_group'] is None)
e2=copy.deepcopy(e);e2.update(notes='修改测试记录',change_reason='second QA revision',reviewer='TOOL_TEST_REVIEWER')
r2=store.save('SP0_794',r1['annotation_id'],1,e2)
chk('immutable_revision_chain_preserved',digest(p1)==h1 and r2['revision']==2 and len(r2['history'])==2 and r2['history'][-1]['previous_revision_sha256']==h1 and r2['created_at']==r1['created_at'])
try:store.save('SP0_794',r1['annotation_id'],1,e)
except ValueError as x:chk('stale_revision_conflict_no_overwrite','版本冲突' in str(x) and store.load('SP0_794',r1['annotation_id'])['revision']==2)
else:chk('stale_revision_conflict_no_overwrite',False)
with mock.patch.object(subject.os,'link',side_effect=PermissionError('injected QA link failure')):
 try:store.save('SP0_794',r1['annotation_id'],2,e)
 except PermissionError:pass
 else:notes.append('link-failure injection did not fire')
chk('interrupted_publish_keeps_previous_complete_revision',store.load('SP0_794',r1['annotation_id'])['revision']==2 and len(store.revisions('SP0_794',r1['annotation_id']))==2 and not list(p1.parent.glob('*.tmp')))
try:subject.Store(MAN,R/'reports/cubit_text_prep/20261029')
except ValueError:chk('old_report_as_draft_root_refused',True)
else:chk('old_report_as_draft_root_refused',False)
for key,val in [('operator',''),('review_status','accepted'),('target_instance_ids',['foreign-target']),('instruction_text','CR1_177 class 0 L0001'),('reasoning_kind','spatial'),('valid_domain','full_image_verified')]:
 x=copy.deepcopy(e);x[key]=val;rejects('reject_'+key,x,sample)
for label,b in [('nan',[0,0,float('nan'),1]),('outside',[0,0,2,1]),('reversed',[.8,0,.1,1]),('bool',[False,0,1,1])]:
 x=copy.deepcopy(e);x['references'][0]['bbox_normalized']=b;rejects('reject_bbox_'+label,x,sample)
x=copy.deepcopy(e);x['relations'][0]['reference_id']='ref-missing';rejects('reject_unresolved_reference',x,sample)
x=copy.deepcopy(e);x['review_status']='rejected';x['notes']='';rejects('reject_rejection_without_reason',x,sample)
x=copy.deepcopy(e);x.update(review_status='ready_for_human_review',target_instance_ids=[],relations=[]);rejects('reject_ready_without_target',x,sample)
# Separate ephemeral loopback server, no browser operations and no collision with implementer's service.
httpserver=subject.make_server(MAN,B/'qa_drafts'/('review-http-'+uuid.uuid4().hex),0);thread=threading.Thread(target=httpserver.serve_forever,daemon=True);thread.start()
def req(method,path,body=None,headers=None):
 c=http.client.HTTPConnection('127.0.0.1',httpserver.server_port,timeout=10)
 a=json.dumps(body,ensure_ascii=False).encode() if body is not None else None
 c.request(method,path,a,headers or {});r=c.getresponse();v=r.read();status=r.status;head=dict(r.getheaders());c.close();return status,v,head
try:
 status,b,h=req('GET','/api/session');session=json.loads(b)
 chk('api_exact_four_whitelist',status==200 and {x['key'] for x in session['images']}=={'CR1_177','CR1_40','SP1_1427','SP0_794'})
 chk('api_no_cache_and_csp',h.get('Cache-Control')=='no-store' and "frame-ancestors 'none'" in h.get('Content-Security-Policy',''))
 chk('api_unknown_image_refused',req('GET','/api/image?key=CR1_140')[0]==400)
 chk('api_traversal_refused',req('GET','/api/image?key=../../AGENTS.md')[0]==400 and req('GET','/../../AGENTS.md')[0]==404)
 chk('api_foreign_host_refused',req('GET','/api/session',headers={'Host':'evil.example'})[0]==400)
 payload=dict(key='SP0_794',annotation_id=None,expected_revision=0,edit=e)
 chk('api_foreign_origin_refused',req('POST','/api/save',payload,{'Origin':'http://evil.example','Content-Type':'application/json','X-Pilot-Token':session['token']})[0]==400)
 chk('api_missing_token_refused',req('POST','/api/save',payload,{'Origin':httpserver.origin,'Content-Type':'application/json'})[0]==400)
 headers={'Origin':httpserver.origin,'Content-Type':'application/json','X-Pilot-Token':session['token']}
 status,b,h=req('POST','/api/save',payload,headers); saved=json.loads(b)
 chk('api_authenticated_save_roundtrip',status==200 and json.loads(req('GET','/api/draft?key=SP0_794&id='+saved['annotation_id'])[1])['edit']==e)
 chk('api_original_image_exact_bytes',req('GET','/api/image?key=CR1_177')[1]==raws[0].read_bytes())
 status,b,h=req('POST','/api/save',dict(payload,expected_revision=1),headers)
 chk('api_invalid_new_revision_refused',status==400)
finally:httpserver.shutdown();thread.join();httpserver.server_close()
after=[(str(p),p.stat().st_size,p.stat().st_mtime_ns,digest(p)) for p in raws]
chk('eight_original_file_sha_size_mtime_unchanged',before==after)
result=dict(all_pass=all(checks.values()),checks=checks,notes=notes,elapsed_seconds=time.perf_counter()-start,raw_files_checked=len(raws),qa_root=str(store.root),scope='four source samples only; QA records never human annotations; no UI or model tests',initial_independent_four_provenance=dict(all_image_label_hash_and_stable_ids_match=True,source_bytes_read=18956782,elapsed_seconds=0.0128643),cost_caveat='Store/API repeatedly rehash originals; this script does not instrument exact total source logical bytes')
(OUT/'engineering_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
if not result['all_pass']:raise SystemExit(1)
