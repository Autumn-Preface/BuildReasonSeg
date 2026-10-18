"""Integrity-only closure; does not import any model/runtime product modules."""
import csv,gzip,json,os,subprocess,zipfile,time,sys
from pathlib import Path
from audit_cleanup import REPO,OUT,ROOTS,digest,write_json

def load_rows(name):
    if name.endswith('.gz'):
        with gzip.open(OUT/name,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    actual=load_rows('cleanup_manifest.csv')
    deleted={r['path'] for r in actual if r['status']=='DELETED'}
    expected={r['path'] for r in load_rows('planned_deletions.csv.gz')}
    errors=[];checked={};cache={}
    if (OUT/'hashes_after.csv.gz').exists():
        cache={r['path']:r for r in load_rows('hashes_after.csv.gz')}
    def check_file(p,h):
        p=str(p)
        if p not in checked:
            q=Path(p)
            if q.is_file():
                s=q.stat();old=cache.get(p,{})
                checked[p]=old['sha256'] if old and (s.st_size,s.st_mtime_ns)==(int(old['bytes']),int(old['mtime_ns'])) else digest(p)
            else:checked[p]=None
        if checked[p]!=h:errors.append(dict(path=p,kind='SHA256_MISMATCH_OR_MISSING',expected=h,actual=checked[p]))
    tracked=json.loads((OUT/'tracked_before.json').read_text(encoding='utf-8'))
    for r in tracked:
        if r['path'] not in ('handoff/CURRENT_TASK.md','handoff/EXECUTOR_STATE.yaml'):
            check_file(REPO/r['path'],r['sha256'])
    preserved=load_rows('preserved_hash_index.csv')
    for i,r in enumerate(preserved):
        check_file(r['path'],r['sha256'])
        if i%1500==0:print('Protected hashes',i,'/',len(preserved),flush=True)
    peer_checks=0
    for i,r in enumerate(load_rows('duplicate_hash_proof.csv.gz')):
        keep=r['retained_copy']
        if '!/' not in keep:
            check_file(keep,r['sha256']);peer_checks+=1
        if i%2000==0:print('Retained duplicate proofs',i,flush=True)
    plan={r['path']:r for r in load_rows('planned_deletions.csv.gz')}
    images=load_rows('native_image_export_proof.csv.gz')
    for r in images:
        p=Path(r['retained_original']);s=p.stat();old=plan[r['path']]
        if (s.st_size,s.st_ino,s.st_dev,s.st_mtime_ns)!=(int(r['bytes']),int(r['source_inode']),int(r['source_dev']),int(old['mtime_ns'])):
            errors.append(dict(path=str(p),kind='ORIGINAL_TILE_IDENTITY_CHANGED'))
    for a in json.loads((OUT/'archives.json').read_text(encoding='utf-8')):
        with zipfile.ZipFile(a['path']) as z:
            if z.testzip() is not None or len(z.namelist())!=a['entries']:
                errors.append(dict(path=a['path'],kind='ARCHIVE_CRC_OR_COUNT'))
    if '--hashes-only' in sys.argv:
        with gzip.open(OUT/'hashes_after.csv.gz','wt',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','bytes','mtime_ns','sha256']);w.writeheader()
            for p,h in checked.items():
                if h:
                    s=Path(p).stat();w.writerow(dict(path=p,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h))
        write_json('hash_gate_after.json',dict(unique_hashes=len(checked),errors=errors,original_tiles_checked=len(images),pass_=not errors))
        print('HASH_GATE',len(checked),'errors',len(errors),flush=True)
        if errors:raise SystemExit(1)
        return
    result=json.loads((OUT/'execution_result.json').read_text(encoding='utf-8'))
    before={r['path']:r for r in load_rows('files_before.csv.gz')}
    after={r['path']:r for r in load_rows('files_after.csv.gz')}
    unexpected_missing=[p for p in before.keys()-after.keys()-deleted if '\\.git\\' not in p]
    unexpected_changed=[]
    report=str(OUT)+'\\'
    for p in before.keys()&after.keys():
        if '\\.git\\' in p or p.startswith(report) or p in (str(REPO/'handoff/CURRENT_TASK.md'),str(REPO/'handoff/EXECUTOR_STATE.yaml')):continue
        if (before[p]['bytes'],before[p]['mtime_ns'])!=(after[p]['bytes'],after[p]['mtime_ns']):
            unexpected_changed.append(p)
    unexpected_additions=[p for p in after.keys()-before.keys() if not p.startswith(report) and '\\.git\\' not in p
                          and not p.startswith(str(REPO/'artifacts/legacy_archive/20261019')+'\\')]
    # All unique/final accepted model identities, not runtime performance, are checked.
    final6=json.loads((REPO/'reports/midterm_evidence_review/audit/checkpoint_presence_proof.json').read_text(encoding='utf-8'))['records']
    for r in final6:check_file(r['path'],r['sha256'])
    canonical=REPO/'delivery_src/BuildReasonSeg_Advisor_RC1'
    rc1=ROOTS[0]/'delivery/BuildReasonSeg_Advisor_RC1'
    source_manifest=json.loads((canonical/'source_manifest.json').read_text(encoding='utf-8'))
    for r in source_manifest['files']:check_file(rc1/r['path'],r['sha256'])
    source_match=digest(canonical/'source_manifest.json')==digest(rc1/'source_manifest.json')
    assert deleted<=expected
    closure=dict(checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        preexisting_tracked_files=len(tracked),tracked_hash_checks=len(tracked)-2,
        protected_hash_records=len(preserved),unique_file_hash_checks=len(checked),retained_duplicate_proofs=peer_checks,
        original_native_images_checked=len(images),six_frozen_final_checkpoints=len(final6),
        canonical_source_entries=len(source_manifest['files']),external_manifest_byte_identical=source_match,
        unexpected_removed_files=unexpected_missing,unexpected_changed_files=unexpected_changed,
        unexpected_added_files=unexpected_additions,hash_errors=errors,
        all_planned_deletions_accounted=(deleted==expected and result['failed_files']==0),
        no_model_loads_or_inference_or_training=True,
        unmeasured_other_project_cache='PRESERVED_ACCESS_DENIED',
        retired_demo_v1_exists=(ROOTS[0]/'delivery/BuildReasonSeg_Demo_V1').exists(),
        retired_demo_v2_exists=(ROOTS[0]/'delivery/BuildReasonSeg_Demo_V2').exists(),
        retired_mvp_python_exists=(REPO/'.conda/buildreasonseg-mvp/python.exe').exists(),
        bundled_agents_skill_preserved=(REPO/'.conda/buildreasonseg-mvp/Lib/site-packages/typer/.agents/skills/typer/SKILL.md').is_file())
    closure['pass']=not errors and not unexpected_missing and not unexpected_changed and not unexpected_additions and source_match and closure['all_planned_deletions_accounted']
    write_json('integrity_result.json',closure)
    print(json.dumps(closure,ensure_ascii=False),flush=True)
    if not closure['pass']:raise SystemExit(1)

if __name__=='__main__':main()
