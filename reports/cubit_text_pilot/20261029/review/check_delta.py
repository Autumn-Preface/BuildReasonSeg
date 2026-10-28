"""Incremental independent probes for added metadata leak guards and session identity."""
import hashlib,http.client,json,pathlib,sys,threading,time,uuid
R=pathlib.Path(__file__).resolve().parents[4];B=R/'research/cubit_text_pilot';OUT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(B));import server as s
m=OUT.parent/'first_four.json';data=s.read(m);sample=next(x for x in data['images'] if x['key']=='CR1_177');checks={};start=time.perf_counter()
for marker in ['task_level','reasoning_kind','numeric_class','class_id','instance_id','视觉语义推理','空间关系推理']:
 e=s.empty_edit();e.update(operator='TOOL_TEST_NOT_HUMAN',instruction_text=marker)
 try:s.validate_edit(e,sample)
 except ValueError:checks['reject_'+marker]=True
 else:checks['reject_'+marker]=False
server=s.make_server(m,B/'qa_drafts'/('review-delta-'+uuid.uuid4().hex),0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
 c=http.client.HTTPConnection('127.0.0.1',server.server_port);c.request('GET','/api/session');r=c.getresponse();status=r.status;j=json.loads(r.read());c.close()
 checks['session_tool']=status==200 and j['tool']=='cubit-text-pilot-01'
 checks['session_manifest_hash']=j['manifest_sha256']==hashlib.sha256(m.read_bytes()).hexdigest()
 checks['session_draft_root']=j['drafts_root']==str(server.store.root)
finally:server.shutdown();thread.join();server.server_close()
result=dict(all_pass=all(checks.values()),checks=checks,elapsed_seconds=time.perf_counter()-start,scope='isolated dynamic port, no main 8765/8766 use; no draft save, UI or source writes')
(OUT/'delta_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2));assert result['all_pass']
