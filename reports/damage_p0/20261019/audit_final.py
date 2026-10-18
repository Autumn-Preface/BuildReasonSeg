"""Read-only legacy hashes, scoped diffs, evidence links, structured records and P0 storage."""
import sys
sys.stdout.reconfigure(encoding='utf-8');sys.dont_write_bytecode=True
import csv,datetime,hashlib,json,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'evaluation/damage_p0'
BASE='0e7ee93c8ca62426e5f79b39fc874f1a2b7e7ee4'
GIT='C:/Users/ROG/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
ALLOWED_FILES={'AGENTS.md','governance/PROJECT_STATE.yaml','governance/DECISIONS.md',
 'governance/DAMAGE_RESEARCH_SCOPE.md','handoff/CURRENT_TASK.md','handoff/EXECUTOR_STATE.yaml'}
PREFIXES=('docs/research/damage_p0/','datasets/damage/','evaluation/damage_p0/','reports/damage_p0/20261019/')
errors=[]
def allowed(p):return p in ALLOWED_FILES or p.startswith(PREFIXES)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output([GIT,*args],cwd=ROOT).decode('utf-8',errors='replace').strip()

changes=git('diff','--name-only',BASE).splitlines()
untracked=git('ls-files','--others','--exclude-standard').splitlines()
bad=[p for p in changes+untracked if not allowed(p)]
errors.extend({'kind':'OUT_OF_SCOPE','path':p} for p in bad)
if git('diff','--diff-filter=D','--name-only',BASE):errors.append({'kind':'TRACKED_DELETION'})
paths={}
with (ROOT/'reports/legacy_cleanup/20261019/preserved_hash_index.csv').open(encoding='utf-8-sig',newline='') as f:
 rows=list(csv.DictReader(f))
 skipped=[]
 for r in rows:
  p=Path(r['path'])
  try:rel=p.relative_to(ROOT).as_posix()
  except ValueError:rel=None
  if rel in ALLOWED_FILES:skipped.append(rel);continue
  if str(p) not in paths:
   actual=digest(p) if p.is_file() else None
   paths[str(p)]=actual
  if paths[str(p)]!=r['sha256']:errors.append({'kind':'PROTECTED_HASH_MISMATCH','path':str(p),'expected':r['sha256'],'actual':paths[str(p)]})
 print('protected hash records',len(rows),'unique',len(paths),'errors',len(errors),flush=True)

documents=list((ROOT/'docs/research/damage_p0').glob('*.md'))+list((ROOT/'reports/damage_p0/20261019').rglob('*.md'))+[ROOT/'datasets/damage/README.md',ROOT/'governance/DAMAGE_RESEARCH_SCOPE.md']
checked_links=0
for p in documents:
 if p.name=='authorization.md':continue
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8-sig')):
  if target.startswith(('http:','https:','#','mailto:')):continue
  target=target.split('#')[0].strip('<>')
  if not target:continue
  checked_links+=1
  if not (p.parent/target).exists():errors.append({'kind':'BROKEN_LINK','file':str(p.relative_to(ROOT)),'target':target})
for p in OUT.glob('*.json'):
 try:json.loads(p.read_text(encoding='utf-8-sig'))
 except Exception as e:errors.append({'kind':'INVALID_JSON','file':p.name,'error':repr(e)})
counts={}
for name in ['dataset_registry.csv','model_registry.csv']:
 with (OUT/name).open(encoding='utf-8',newline='') as f:counts[name]=len(list(csv.DictReader(f)))
records=json.loads((OUT/'sample_checks.json').read_text())['records']
for r in records:
 for key,sha in [('image','image_sha256'),('annotation','annotation_sha256')]:
  if digest(ROOT/r[key])!=r[sha]:errors.append({'kind':'SAMPLE_HASH_MISMATCH','file':r[key]})
 if r['errors']:errors.append({'kind':'SAMPLE_PARSER_ERROR','file':r['image'],'errors':r['errors']})
facts=json.loads((OUT/'python_proposal.json').read_text(encoding='utf-8-sig'))
if facts.get('gpu_smoke',{}).get('passed') is not True:errors.append({'kind':'GPU_SMOKE_NOT_PASS'})
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline':BASE,
 'observed_head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
 'protected_hash_records':len(rows),'protected_unique_files_hashed':len(paths),'allowed_governance_hash_skips':sorted(set(skipped)),
 'changed_paths':changes,'untracked_paths':untracked,'scope_check':'PASS' if not bad else 'FAIL',
 'local_links_checked':checked_links,'registry_rows':counts,'literature_records':len(json.loads((OUT/'literature_registry.json').read_text(encoding='utf-8'))),
 'sample_pairs':len(records),'errors':errors,'passed':not errors,
 'model_inference_calls':0,'training_calls':0,'new_weight_download_bytes':0,
 'meaning':'engineering integrity and evidence consistency only; no scientific acceptance'}
(OUT/'integrity_final.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')

storage=[]
for rel in ['docs/research/damage_p0','datasets/damage','evaluation/damage_p0','reports/damage_p0/20261019']:
 files=[p for p in (ROOT/rel).rglob('*') if p.is_file() and p.name!='storage_final.json']
 storage.append({'path':rel,'files':len(files),'logical_bytes':sum(p.stat().st_size for p in files)})
cache=[]
for rel in ['datasets/damage/local','evaluation/damage_p0/sources']:
 files=[p for p in (ROOT/rel).rglob('*') if p.is_file()]
 cache.append({'path':rel,'files':len(files),'logical_bytes':sum(p.stat().st_size for p in files),
               'action':'retain for pending review; candidate rebuildable cache, no deletion performed'})
disk=shutil.disk_usage(ROOT)
s={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':'logical bytes; not NTFS allocation',
 'scope_excludes':'storage_final.json itself; .git objects; small increments to existing governance/handoff files',
 'directories':storage,'logical_bytes_total':sum(x['logical_bytes'] for x in storage),'local_only_caches':cache,
 'volume_total_bytes':disk.total,'volume_free_bytes':disk.free,'cleanup_performed':False}
(OUT/'storage_final.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'passed':not errors,'errors':errors,'storage_logical_bytes':s['logical_bytes_total'],'free_bytes':disk.free},ensure_ascii=False))
if errors:raise SystemExit(1)
