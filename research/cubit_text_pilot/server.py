"""Local-only manual draft workbench. stdlib server; original data read-only."""
import argparse, copy, hashlib, json, math, os, re, secrets, tempfile, threading, uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
STATUSES={'draft','needs_clarification','rejected','ready_for_human_review'}
RISKS={'target_uniqueness','fragmentation','missing_annotation','occlusion','truncation','alignment','valid_domain','spatial_relation','visibility'}
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(b): return hashlib.sha256(b).hexdigest()
def check(condition,message):
    if not condition: raise ValueError(message)
def text(x,limit=8000):
    check(isinstance(x,str) and len(x)<=limit,'文本类型或长度错误'); return x

def validate_edit(e, sample):
    check(isinstance(e,dict),'edit 必须是对象')
    allowed={'operator','reviewer','review_status','instruction_text','task_level','reasoning_kind','target_instance_ids','references','relations','regions','valid_domain','risks','notes','visual_class_provisional','change_reason'}
    check(set(e)==allowed,'草稿字段不完整或含未知字段')
    for k in ('operator','reviewer','instruction_text','notes','visual_class_provisional','change_reason'): text(e[k])
    check(bool(e['operator'].strip()),'请填写真实操作人；工具测试使用 TOOL_TEST_NOT_HUMAN')
    check(e['review_status'] in STATUSES,'不支持正式人工确认状态')
    check(e['task_level'] in ('direct','position','reasoning'),'内部任务层次错误')
    check(e['reasoning_kind'] in ('none','visual_semantic','spatial'),'推理子类错误')
    check((e['task_level']=='reasoning')==(e['reasoning_kind']!='none'),'推理层次与子类不一致')
    check(e['valid_domain'] in ('unknown','target_region_proposed','class_region_proposed','full_image_proposed'),'有效域仅记录建议，不自动确认')
    ids={r['instance_id'] for r in sample['instances']}
    targets=e['target_instance_ids'];check(isinstance(targets,list) and len(set(targets))==len(targets) and all(x in ids for x in targets),'目标必须来自本图原实例')
    check(not re.search(r'cubit-v1|\b(?:CR|SP)[01]_\d+|L\d{4}|(?:class|类别|类目)\s*[:：#]?\s*[01]\b|(?:task_family|task_level|reasoning_kind|numeric_class|class_id|instance_id|direct_target|visual_semantic|spatial_reasoning|视觉语义推理|空间关系推理)',e['instruction_text'],re.I),'指令含内部 ID/类别/任务元数据，请改为自然语言')
    for k in ('references','relations','regions','risks'): check(isinstance(e[k],list) and len(e[k])<=100,'列表过长或类型错误')
    refids=set()
    for r in e['references']:
        check(set(r)=={'id','name','bbox_normalized'},'参照物字段错误');text(r['name'],200)
        check(isinstance(r['id'],str) and re.fullmatch(r'ref-[a-zA-Z0-9-]{1,60}',r['id']) and r['id'] not in refids,'参照物 ID 重复或非法');refids.add(r['id']);bbox(r['bbox_normalized'])
    for r in e['regions']:
        check(set(r)=={'kind','bbox_normalized','note'} and r['kind'] in ('domain_proposed','unknown'),'区域字段错误');text(r['note']);bbox(r['bbox_normalized'])
    for r in e['relations']:
        check(set(r)=={'reference_id','target_instance_ids','relation','evidence'},'关系字段错误')
        check(r['reference_id'] in refids and isinstance(r['target_instance_ids'],list) and r['target_instance_ids'] and all(x in targets for x in r['target_instance_ids']),'关系参照/目标未解析')
        text(r['relation'],200);text(r['evidence']);check(bool(r['relation'].strip()),'关系描述为空')
    check(all(x in RISKS for x in e['risks']) and len(set(e['risks']))==len(e['risks']),'风险类型错误')
    if e['review_status']=='rejected': check(bool(e['notes'].strip()),'退回须记录原因')
    if e['review_status']=='ready_for_human_review': check(bool(targets) and bool(e['instruction_text'].strip()),'送审需要目标集合和指令')
    return copy.deepcopy(e)
def bbox(a):
    check(isinstance(a,list) and len(a)==4 and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in a) and a[0]<a[2] and a[1]<a[3],'区域须为原图归一化非空 bbox')

def empty_edit():
    return dict(operator='',reviewer='',review_status='draft',instruction_text='',task_level='direct',reasoning_kind='none',target_instance_ids=[],references=[],relations=[],regions=[],valid_domain='unknown',risks=[],notes='',visual_class_provisional='',change_reason='')

class Store:
    def __init__(self,manifest,root):
        self.manifest=read(manifest);self.manifest_sha256=sha(Path(manifest).read_bytes());self.source=Path(self.manifest['source_root']).resolve();self.root=Path(root).resolve();self.lock=threading.Lock()
        check(any(self.root.is_relative_to(BASE/name) for name in ('drafts','qa_drafts')) and not self.root.is_relative_to(self.source),'仅允许本工具 drafts/qa_drafts 下保存')
        self.samples={r['key']:r for r in self.manifest['images']};check(len(self.samples) in (4,10),'仅允许四图门禁或十图批次')
        for k,s in self.samples.items():
            check(k in __import__('prepare').TEN and s['official_split']=='train' and s['image_id']==f'cubit-v1/train/{k}','非授权样本')
            self.verify(s)
    def verify(self,s):
        for kind,folder in (('image','images'),('label','labels')):
            p=(self.source/s[kind+'_relpath']).resolve();check(p.is_relative_to(self.source/'train'/folder),'原始路径越界')
            check(sha(p.read_bytes())==s[kind+'_sha256'],'原始文件哈希变化，停止保存/展示')
    def sample(self,k):
        check(k in self.samples,'不在本批候选');return self.samples[k]
    def revisions(self,k,aid):
        self.sample(k);check(isinstance(aid,str) and re.fullmatch('[0-9a-f]{32}',aid),'草稿 ID 非法')
        return sorted((self.root/k/aid).glob('rev-*.json'))
    def load(self,k,aid):
        rows=self.revisions(k,aid);check(bool(rows),'草稿不存在')
        record=read(rows[-1]);check(record['source']==self.provenance(self.sample(k)),'草稿来源版本不一致')
        return record
    def provenance(self,s):
        return {k:s[k] for k in ('image_id','image_sha256','label_sha256','official_split','width_height')}
    def listings(self,k):
        self.sample(k);out=[]
        for p in sorted((self.root/k).glob('*')):
            if p.is_dir() and re.fullmatch('[0-9a-f]{32}',p.name):
                r=self.load(k,p.name);out.append(dict(annotation_id=r['annotation_id'],revision=r['revision'],status=r['edit']['review_status'],updated_at=r['updated_at']))
        return out
    def save(self,k,aid,expected,edit):
        with self.lock:
            s=self.sample(k);self.verify(s);edit=validate_edit(edit,s)
            check(type(expected)==int and expected>=0,'版本号错误')
            if aid:
                previous=self.load(k,aid);check(previous['revision']==expected,'版本冲突，请重载后合并');history=previous['history'][:];created=previous['created_at']
            else:
                check(expected==0,'新草稿版本须为0');aid=uuid.uuid4().hex;history=[];created=None
            now=datetime.now(timezone.utc).isoformat();revision=expected+1
            history.append(dict(revision=revision,at=now,operator=edit['operator'],reviewer=edit['reviewer'],change_reason=edit['change_reason'],review_status=edit['review_status'],previous_revision_sha256=sha(self.revisions(k,aid)[-1].read_bytes()) if expected else None))
            record=dict(schema_version='cubit-text-pilot-draft-1',annotation_id=aid,revision=revision,source=self.provenance(s),source_instance_ids=[x['instance_id'] for x in s['instances']],class_mapping_status='unconfirmed',building_group=None,formal_export_enabled=False,record_kind='manual_review_draft_not_training_sample',created_at=created or now,updated_at=now,edit=edit,history=history)
            folder=self.root/k/aid;folder.mkdir(parents=True,exist_ok=True)
            # Publish a fully flushed revision atomically without replacing any existing revision.
            # A crash before link leaves only a private temp file, ignored by revisions().
            p=folder/f'rev-{revision:06d}.json'
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf8',dir=folder,suffix='.tmp',delete=False) as f:
                temp=Path(f.name);json.dump(record,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
            try: os.link(temp,p)
            finally: temp.unlink()
            return record

class Handler(BaseHTTPRequestHandler):
    def send(self,value,status=200,ctype='application/json; charset=utf-8'):
        b=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'");self.end_headers();self.wfile.write(b)
    def guard(self,post=False):
        check(self.headers.get('Host')==self.server.authority,'仅允许本地标准地址')
        if post:
            check(self.headers.get('Origin')==self.server.origin and self.headers.get('X-Pilot-Token')==self.server.token,'请求来源或会话令牌错误')
            check(self.headers.get('Content-Type')=='application/json','需要 JSON 请求')
    def do_GET(self):
        try:
            self.guard();u=urlparse(self.path);q=parse_qs(u.query);store=self.server.store
            if u.path in ('/','/app.js','/style.css'):
                p=BASE/'web'/({'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[u.path]);self.send(p.read_bytes(),ctype={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'}[p.suffix])
            elif u.path=='/api/session': self.send(dict(tool='cubit-text-pilot-01',manifest_sha256=store.manifest_sha256,drafts_root=str(store.root),token=self.server.token,images=[dict(key=k,instances=len(s['instances'])) for k,s in store.samples.items()]))
            elif u.path=='/api/sample':
                s=store.sample(q['key'][0]);store.verify(s);self.send(dict(sample=s,drafts=store.listings(s['key']),empty_edit=empty_edit()))
            elif u.path=='/api/image':
                s=store.sample(q['key'][0]);store.verify(s);self.send((store.source/s['image_relpath']).read_bytes(),ctype='image/jpeg')
            elif u.path=='/api/draft': self.send(store.load(q['key'][0],q['id'][0]))
            else: self.send({'error':'not found'},404)
        except (ValueError,KeyError,TypeError,FileNotFoundError) as e:self.send({'error':str(e)},400)
    def do_POST(self):
        try:
            self.guard(True);check(self.path=='/api/save','不支持此操作');n=int(self.headers.get('Content-Length','0'));check(0<n<=200000,'请求大小错误');a=json.loads(self.rfile.read(n));check(set(a)=={'key','annotation_id','expected_revision','edit'},'保存请求字段错误');self.send(self.server.store.save(a['key'],a['annotation_id'],a['expected_revision'],a['edit']))
        except (ValueError,KeyError,TypeError,FileNotFoundError) as e:self.send({'error':str(e)},409 if '版本冲突' in str(e) else 400)

def make_server(manifest,root,port):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.store=Store(manifest,root);server.token=secrets.token_urlsafe(32);server.authority=f'127.0.0.1:{server.server_port}';server.origin='http://'+server.authority;return server
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=BASE/'samples.json');p.add_argument('--drafts',type=Path,default=BASE/'drafts');p.add_argument('--port',type=int,default=8765);a=p.parse_args();server=make_server(a.manifest,a.drafts,a.port)
    print(server.origin,flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
