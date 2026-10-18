"""Execute only the committed, task-specific per-file cleanup allowlist."""
import argparse,collections,csv,gzip,json,os,shutil,subprocess,sys,time
from pathlib import Path
from audit_cleanup import REPO,OUT,ROOTS,digest,write_json

def run(checkpoint):
    assert Path(sys.executable).resolve()==(REPO/'.conda/buildreasonseg-proposal/python.exe').resolve()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip()==checkpoint
    assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=REPO).strip(), 'Commit/push gate requires clean worktree.'
    assert subprocess.check_output(['git','rev-parse','@{u}'],cwd=REPO).decode().strip()==checkpoint
    processes=json.loads((OUT/'execution_process_gate.json').read_text(encoding='utf-8-sig'))
    assert processes['active_retired_environment_or_demo_processes']==0
    assert not (OUT/'cleanup_manifest.csv').exists(), 'Never replay cleanup automatically.'
    with gzip.open(OUT/'planned_deletions.csv.gz','rt',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    protect={str(Path(r['path']).resolve()).casefold() for r in csv.DictReader((OUT/'preserved_hash_index.csv').open(encoding='utf-8-sig'))}
    tracked={str((REPO/p).resolve()).casefold() for p in subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0') if p}
    actual_allowed=[ROOTS[0]/'delivery/BuildReasonSeg_Demo_V1',ROOTS[0]/'delivery/BuildReasonSeg_Demo_V2',
                    ROOTS[0]/'workspace/project/WHU_Building_Segment/dataset/WHU_YOLO_dataset',REPO]
    # Validate the entire list BEFORE the first unlink. No glob- or size-based deletion.
    for i,r in enumerate(rows):
        p=Path(r['path']); q=p.resolve()
        assert str(q)==str(p) and q.is_absolute() and ROOTS[0] in q.parents
        assert any(q.is_relative_to(root) and q!=root for root in actual_allowed)
        assert not any(x in q.parts for x in ('.git','.agents','.codex'))
        assert str(q).casefold() not in tracked|protect
        assert not q.is_relative_to(REPO/'.conda/buildreasonseg-proposal')
        s=q.stat(follow_symlinks=False)
        assert not s.st_file_attributes & 0x400
        assert (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)==tuple(int(r[k]) for k in ('bytes','mtime_ns','inode','dev')), str(q)
        if i%20000==0: print('Safety precheck',i,'/',len(rows),flush=True)
    # Verified archives remain available before retiring all deployment paths.
    for a in json.loads((OUT/'archives.json').read_text(encoding='utf-8')):
        assert digest(a['path'])==a['sha256']
    groups=collections.defaultdict(list)
    for r in rows: groups[(r['dev'],r['inode'])].append(r)
    free_before=shutil.disk_usage('C:\\').free
    write_json('execution_start.json',dict(checkpoint=checkpoint,allowlist_sha256=digest(OUT/'planned_deletions.csv.gz'),
        started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),disk_free_bytes=free_before,
        all_paths_resolved_and_inside_explicit_scope=True,all_predelete_identities_match=True,
        all_selected_untracked_and_not_protected=True))
    deleted,failed=set(),[]
    fields=['path','bytes','allocated_bytes','reason_id','status','error']
    with (OUT/'cleanup_manifest.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,r in enumerate(rows):
            p=Path(r['path'])
            result={k:r[k] for k in fields[:4]}
            try:
                # Recheck at point of deletion; do not chmod shared originals on permission failure.
                s=p.stat(follow_symlinks=False)
                assert not s.st_file_attributes & 0x400
                assert (s.st_size,s.st_mtime_ns,s.st_ino)==tuple(int(r[k]) for k in ('bytes','mtime_ns','inode'))
                p.unlink()
                deleted.add(str(p));result.update(status='DELETED',error='')
            except Exception as ex:
                result.update(status='PRESERVED_DELETE_FAILED',error=str(ex));failed.append(result)
            w.writerow(result)
            if i%2000==0:
                f.flush();print('Cleanup',i,'/',len(rows),'failures',len(failed),flush=True)
    # rmdir removes only empty known subdirectories; no recursive or cross-shell deletion.
    prune_roots=actual_allowed[:3]+[REPO/'.conda/buildreasonseg-mvp',REPO/'artifacts/task6n/features',
        REPO/'artifacts/task6m_yolo_native/images',REPO/'local_cache/pip',
        REPO/'local_cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203']
    for r in rows:
        p=Path(r['path'])
        for label in ('__pycache__','.pytest_cache'):
            if label in p.parts:
                prune_roots.append(Path(*p.parts[:p.parts.index(label)+1]))
    directories=set()
    for r in rows:
        p=Path(r['path']).parent
        eligible=[x for x in prune_roots if p==x or x in p.parents]
        if not eligible:continue
        root=max(eligible,key=lambda x:len(x.parts))
        while p==root or root in p.parents:
            assert ROOTS[0] in p.resolve().parents and p!=ROOTS[0] and '.git' not in p.parts
            directories.add(p)
            if p==root:break
            p=p.parent
    removed_dirs=0
    for p in sorted(directories,key=lambda x:-len(x.parts)):
        try:p.rmdir();removed_dirs+=1
        except OSError:pass
    released=0
    for g in groups.values():
        if len(g)==int(g[0]['nlink']) and all(r['path'] in deleted for r in g):
            released+=int(g[0]['allocated_bytes'])
    summary=dict(finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'),checkpoint=checkpoint,
       planned_files=len(rows),deleted_files=len(deleted),failed_files=len(failed),removed_empty_directories=removed_dirs,
       deleted_logical_bytes=sum(int(r['bytes']) for r in rows if r['path'] in deleted),
       released_file_allocation_bytes=released,physical_method='FILE_STANDARD_INFO allocation; only completely unlinked file IDs counted.',
       free_before_bytes=free_before,free_after_bytes=shutil.disk_usage('C:\\').free,
       no_source_or_governance_or_inference_writes=True)
    write_json('execution_result.json',summary)
    write_json('deletion_failures.json',failed)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--execute-checkpoint',required=True)
    run(a.parse_args().execute_checkpoint)
