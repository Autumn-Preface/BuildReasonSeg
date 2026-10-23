"""Independent V3 recovery builder; never invokes/bypasses V1 preservation gates."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import build_task8f_user_demo as v2
from scripts import build_task8e_user_demo as base
OUT=REPO/'reports/demo_r1/20261023'
FINAL=Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Demo_V3')
STAGING=FINAL.with_name('BuildReasonSeg_Demo_V3_demo_r1_staging')
def run(assemble=False):
    blobs=v2.accepted_blobs();plan,changed=v2.plan(blobs)
    historic=json.loads((REPO/'evaluation/task8f_user_demo_live_trace_v1.json').read_text(encoding='utf-8'))
    expected={x['path']:x for x in historic['preflight']['old_rc1_inventory']}
    selected=plan['source_paths']+plan['assets']
    for p in selected:
        base.reject_link(base.SOURCE/p)
        if base.identity(base.SOURCE/p)!={k:expected[p][k] for k in ('bytes','sha256')}:
            raise ValueError('STOP accepted RC1 asset drift: '+p)
        if p in blobs and (base.SOURCE/p).read_bytes()!=blobs[p]:raise ValueError('STOP source differs: '+p)
    for p in (REPO/'demo/user_demo_v2').rglob('*'):
        if p.is_file():
            rel=p.relative_to(REPO).as_posix()
            original=subprocess.check_output(['git','show','e06777b0a0844c5c29325701eb2e819bd193ff04:'+rel],cwd=REPO)
            if original.replace(b'\r\n',b'\n')!=p.read_bytes().replace(b'\r\n',b'\n'):raise ValueError('V2 source drift '+rel)
    facts={'accepted_engine_head':v2.ACCEPTED_HEAD,'historical_v2_head':'e06777b0a0844c5c29325701eb2e819bd193ff04',
           'selected_assets_and_sources_checked':len(selected),'free_bytes':shutil.disk_usage(FINAL.parent).free,
           'rc1_before':base.inventory(base.SOURCE),'source_paths':plan['source_paths'],'asset_paths':plan['assets'],
           'observation_source_differences':plan['observation_source_differences'],
           'recovery_difference':'V1 absent; historical Git source and original RC1 selected-file hashes independently verified; original builders unchanged'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'preflight.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    if not assemble:return facts
    base.safe_destination(base.SOURCE,STAGING,FINAL)
    required=sum(expected[p]['bytes'] for p in selected)
    if facts['free_bytes']<required+(1<<30):raise OSError('STOP insufficient space')
    STAGING.mkdir();records=[]
    def write(rel,data,role):
        p=STAGING/rel;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(data)
        records.append({'path':rel,**base.identity(p),'role':role})
    for rel in selected:
        data=changed.get(rel,(base.SOURCE/rel).read_bytes())
        write('_engine/'+rel,data,'v2_observation_source' if rel in changed else 'accepted_rc1')
        print('verified',rel,flush=True)
    write('_engine/buildreasonseg/runtime/observation.py',changed['buildreasonseg/runtime/observation.py'],'v2_transport')
    app=REPO/'demo/user_demo_v3'
    for p in app.rglob('*'):
        if p.is_file() and 'engine_observation' not in p.parts and p.name!='engine_observation_patch.py' and '__pycache__' not in p.parts:
            write(p.relative_to(app).as_posix(),p.read_bytes(),'v3_ui')
    write('_ui/runtime.json',json.dumps({'validated_python':sys.executable,'runtime_environment_bundled':False}).encode(),'python_locator')
    for name in ('results','_engine/inference/input','_engine/inference/output','_engine/logs','_runtime_cache/yolo','_runtime_cache/matplotlib'):
        (STAGING/name).mkdir(parents=True,exist_ok=True)
    manifest={'demo_version':'V3','accepted_engine_head':v2.ACCEPTED_HEAD,'files':records,'file_count':len(records),'total_bytes':sum(x['bytes'] for x in records),'runtime_environment_bundled':False,'observation_source_differences':plan['observation_source_differences']}
    (STAGING/'_engine/package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=base.forbidden_audit(STAGING)
    if not audit['pass']:raise ValueError(audit)
    base.verify_inventory(base.SOURCE,facts['rc1_before'])
    # No overwrite; same-volume rename only, after complete manifest and protection checks.
    STAGING.rename(FINAL)
    for row in records:
        if base.identity(FINAL/row['path'])!={k:row[k] for k in ('bytes','sha256')}:raise ValueError('Post-copy identity mismatch')
    (OUT/'build.json').write_text(json.dumps({'final':str(FINAL),'files':len(records),'bytes':manifest['total_bytes'],'hash_check':'PASS','forbidden_audit':audit},indent=2),encoding='utf-8')
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assemble',action='store_true');a=p.parse_args();run(a.assemble)
