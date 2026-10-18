"""Read-only source checks and explicit deletion plan; does not delete files."""
import collections
import ctypes
import gzip
import csv
import json
import os
from pathlib import Path
import subprocess
import zipfile
from audit_cleanup import REPO, OUT, ROOTS, digest, write_csv, write_json

BASE = ROOTS[0]
RC1 = BASE / 'delivery/BuildReasonSeg_Advisor_RC1'
ARCHIVE = REPO / 'artifacts/legacy_archive/20261019'
RAW = Path(r'C:\D\resources\Satellite dataset Ⅱ (East Asia)')
OLD = BASE / 'workspace/project/WHU_Building_Segment'
PEER = Path(r'C:\D\resources\project\WHU_Building_Segment')
HASHES = {}

def sha(p):
    s = str(p)
    if s not in HASHES:
        HASHES[s] = digest(p)
    return HASHES[s]

def rows_before():
    with gzip.open(OUT/'files_before.csv.gz','rt',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def children(root, rows):
    prefix = str(root) + '\\'
    return [Path(r['path']) for r in rows if r['path'].startswith(prefix)]

def physical(p):
    class Standard(ctypes.Structure):
        _fields_=[('AllocationSize',ctypes.c_longlong),('EndOfFile',ctypes.c_longlong),
                  ('NumberOfLinks',ctypes.c_uint32),('DeletePending',ctypes.c_ubyte),('Directory',ctypes.c_ubyte)]
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    k.CreateFileW.restype=ctypes.c_void_p
    k.GetFileInformationByHandleEx.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]
    k.CloseHandle.argtypes=[ctypes.c_void_p]
    name='\\\\?\\'+str(p.resolve())
    handle=k.CreateFileW(name,0x80,7,None,3,0,None)
    if handle == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    data=Standard()
    try:
        if not k.GetFileInformationByHandleEx(handle,1,ctypes.byref(data),ctypes.sizeof(data)):
            raise ctypes.WinError(ctypes.get_last_error())
        return data.AllocationSize
    finally: k.CloseHandle(handle)

def main():
    rows = rows_before()
    # DirEntry.stat on Windows reports inode=0; no unique-physical claim is made from it.
    inv = json.loads((OUT/'inventory_before.json').read_text(encoding='utf-8'))
    for v in inv['roots'].values():
        v.pop('unique_logical_bytes', None)
        v['physical_bytes'] = None
    inv['physical_accounting'] = 'Full-tree logical bytes only; actual deletion uses os.stat file IDs, link counts and GetCompressedFileSizeW.'
    write_json('inventory_before.json', inv)
    selected, proof, archives, preserved = {}, [], [], []
    def add(p, reason, recovery='', kept=''):
        p = p.resolve()
        assert BASE in p.parents and p != BASE
        assert '.git' not in p.parts
        selected[str(p)] = dict(path=str(p),reason=reason,recovery=recovery,retained_copy=str(kept) if kept else '')
    def duplicate(p, keep, reason):
        assert p.stat().st_size == keep.stat().st_size and sha(p) == sha(keep), (p,keep)
        proof.append(dict(path=str(p),retained_copy=str(keep),bytes=p.stat().st_size,sha256=sha(p)))
        add(p, reason, 'Copy retained byte-identical file back to original path if historical execution is needed.',keep)

    # Model index first: never deserialize / select / evaluate model weights.
    weights = [Path(r['path']) for r in rows if r['path'].startswith(str(BASE)+'\\')
               and '.conda' not in Path(r['path']).parts
               and Path(r['path']).suffix.lower() in ('.pt','.pth','.safetensors','.ckpt','.onnx')
               and int(r['bytes']) > 500000]
    wrecords=[]
    for i,p in enumerate(weights):
        wrecords.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
        if i%20 == 0: print('Weight identity:',i,'/',len(weights),flush=True)
    write_csv('weight_identities_before.csv',wrecords,['path','bytes','sha256'])

    # Archive all non-model Demo files, including source/config, all result records and previews.
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    for name in ('BuildReasonSeg_Demo_V1','BuildReasonSeg_Demo_V2'):
        root=BASE/'delivery'/name
        files=children(root,rows)
        zpath=ARCHIVE/(name+'_non_model.zip')
        assert not zpath.exists(), 'Never overwrite an archive.'
        nonmodels=[p for p in files if not p.is_relative_to(root/'_engine/model')]
        with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in nonmodels:
                z.write(p,p.relative_to(root).as_posix())
        with zipfile.ZipFile(zpath) as z:
            assert z.testzip() is None
            import hashlib
            for p in nonmodels:
                rel=p.relative_to(root).as_posix()
                assert hashlib.sha256(z.read(rel)).hexdigest() == sha(p)
                proof.append(dict(path=str(p),retained_copy=str(zpath)+'!/'+rel,bytes=p.stat().st_size,sha256=sha(p)))
        archives.append(dict(path=str(zpath),bytes=zpath.stat().st_size,sha256=sha(zpath),entries=len(nonmodels)))
        for p in files:
            if p in nonmodels:
                add(p,'Retired Demo deployment; all non-model bytes preserved in verified local ZIP.','Extract ZIP entry at original relative path.',zpath)
            else:
                keep=RC1/'model'/p.relative_to(root/'_engine/model')
                duplicate(p,keep,'Retired Demo model duplicate; single complete Advisor_RC1 retained.')
        print('Archived and checked',name,len(nonmodels),'non-model files',flush=True)

    # HF and SAM2 copies are exact duplicates of the retained complete RC1 assets.
    hf=REPO/'local_cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203'
    for p in children(hf,rows):
        duplicate(p,RC1/'model/components/program_head/Qwen3-VL-2B-Instruct'/p.relative_to(hf),'Redundant Qwen cached model/tokenizer copy; pinned revision and complete RC1 retained.')
    duplicate(REPO/'local_cache/models/sam2.1_hiera_base_plus.pt',RC1/'model/components/sam2/sam2.1_hiera_base_plus.pt','Redundant SAM2 pretrained checkpoint; byte-identical RC1 copy retained.')

    # Feature tensors are memoized frozen SAM2 embeddings, not GT, weights or results.
    for p in children(REPO/'artifacts/task6n/features',rows):
        assert p.suffix == '.npy', str(p)
        add(p,'Rebuildable frozen-SAM2 per-tile feature cache; configs/task6n_oracle_relation_field.yaml and original RGB/weights retained.',
            'Regeneration requires the original frozen encoder/environment and compute; not performed in cleanup.')

    # All native-vector YOLO image links/copies must resolve to the precise original tile index.
    idx=[json.loads(line) for line in (REPO/'datasets/whu_native_vector/v1.0/tiles/index.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id={x['tile_id']:x for x in idx}
    image_proof=[]
    for p in children(REPO/'artifacts/task6m_yolo_native/images',rows):
        origin=RAW/by_id[p.stem]['source_image_ref']
        s,o=p.stat(),origin.stat()
        same=os.path.samefile(p,origin)
        assert s.st_size==o.st_size and (same or sha(p)==sha(origin)), str(p)
        image_proof.append(dict(path=str(p),retained_original=str(origin),bytes=s.st_size,same_file_id=same,
                                source_inode=o.st_ino,source_dev=o.st_dev,source_nlink=o.st_nlink))
        add(p,'Derived YOLO training image export; exact retained original verified (usually NTFS hardlink).',
            'Recreate exact links/copies using retained tile index and source_image_ref; labels/data.yaml remain.',origin)
    write_csv('native_image_export_proof.csv',image_proof,list(image_proof[0]))

    # Previous baseline dataset is a complete byte-identical deployment duplicate, outside-scope peer stays untouched.
    ds=OLD/'dataset/WHU_YOLO_dataset'
    peer_ds=PEER/'dataset/WHU_YOLO_dataset'
    files=children(ds,rows)
    for i,p in enumerate(files):
        duplicate(p,peer_ds/p.relative_to(ds),'Redundant historical baseline dataset; exact full-tree peer outside cleanup scope retained.')
        if i%2000==0: print('Dataset duplicate:',i,'/',len(files),flush=True)
    # Refuse asymmetric dataset tree.
    peer_files={str(p.relative_to(peer_ds)) for p in peer_ds.rglob('*') if p.is_file()}
    assert peer_files=={str(p.relative_to(ds)) for p in files}

    # Preserve one full proposal environment used by accepted external RC1; obsolete original clone is retired.
    env=REPO/'.conda/buildreasonseg-mvp'
    env_records=[]
    for e in (env,REPO/'.conda/buildreasonseg-proposal'):
        code='import importlib.metadata,json,sys;print(json.dumps({"python":sys.version,"packages":sorted([(d.metadata["Name"],d.version) for d in importlib.metadata.distributions()])}))'
        detail=json.loads(subprocess.check_output([str(e/'python.exe'),'-B','-c',code],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}))
        detail['path']=str(e)
        env_records.append(detail)
    write_json('environment_versions.json',env_records)
    for p in children(env,rows):
        add(p,'Retired original Python clone; proposal environment retained intact, no active MVP process at final gate.',
            'Recreate using recorded Python/package versions and tracked environment recipes; binary clone is not archived.')

    # Only exact duplicate historical checkpoints are removed; keep all unique stage/ablation weights pending review.
    groups=collections.defaultdict(list)
    for r in wrecords:
        if str(REPO/'artifacts/checkpoints')+'\\' in r['path']:
            groups[r['sha256']].append(Path(r['path']))
    for h,ps in groups.items():
        if len(ps)<2: continue
        ps.sort(key=lambda p: (not ('best' in p.name.lower()),len(p.parts),str(p)))
        keep=ps[0]
        for p in ps[1:]: duplicate(p,keep,'Exact duplicate historical checkpoint; unique stage/ablation state retained, no model selection.')

    # Source-tree caches only, never live installed toolchains or unrelated projects.
    for r in rows:
        p=Path(r['path'])
        if not p.is_relative_to(REPO): continue
        rel=p.relative_to(REPO)
        if rel.parts[0] in ('.conda','.git','local_cache'): continue
        if '__pycache__' in rel.parts or '.pytest_cache' in rel.parts:
            add(p,'Rebuildable Python bytecode / pytest cache.','Regenerated on ordinary import/test; no training or inference required.')
    for p in children(REPO/'local_cache/pip',rows):
        add(p,'Rebuildable pip HTTP/wheel download cache; installed proposal environment retained.','Re-download pinned packages if needed.')

    tracked={str((REPO/r['path']).resolve()) for r in json.loads((OUT/'tracked_before.json').read_text(encoding='utf-8'))}
    assert not (tracked & selected.keys()), 'Never delete tracked assets.'
    # All unique weights, baseline sources/results, native-vector/split records and midterm pack are protected.
    for r in wrecords:
        if r['path'] not in selected: preserved.append({**r,'kind':'retained_model_or_checkpoint'})
    for r in rows:
        p=Path(r['path'])
        if str(p) in selected: continue
        if p.is_relative_to(REPO/'datasets') or p.is_relative_to(REPO/'reports/midterm_evidence_review'):
            preserved.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),kind='dataset_schema_split_GT_or_midterm_evidence'))
        elif p.is_relative_to(RC1):
            preserved.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),kind='retained_complete_RC1'))
        elif p.is_relative_to(OLD) and p.suffix.lower() not in ('.jpg','.jpeg','.png'):
            preserved.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),kind='legacy_baseline_source_config_log_or_weight'))
    for a in archives:
        preserved.append({**a,'kind':'Demo_non_model_archive'})
    write_csv('preserved_hash_index.csv',preserved,['path','bytes','sha256','kind','entries'])
    write_csv('duplicate_hash_proof.csv',proof,['path','retained_copy','bytes','sha256'])
    write_json('archives.json',archives)

    plans=list(selected.values())
    for i,r in enumerate(plans):
        p=Path(r['path']); s=p.stat()
        r.update(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,dev=s.st_dev,nlink=s.st_nlink,
                 allocated_bytes=physical(p),sha256=HASHES.get(str(p),''))
    write_csv('planned_deletions.csv',plans,list(plans[0]))
    ids=collections.defaultdict(list)
    for r in plans: ids[(r['dev'],r['inode'])].append(r)
    released=sum(v[0]['allocated_bytes'] for v in ids.values() if len(v)==v[0]['nlink'])
    assert all(len(v)<=v[0]['nlink'] for v in ids.values())
    summary=collections.defaultdict(lambda:dict(files=0,logical_bytes=0))
    for r in plans:
        v=summary[r['reason']];v['files']+=1;v['logical_bytes']+=r['bytes']
    write_json('plan_summary.json',dict(files=len(plans),logical_bytes=sum(r['bytes'] for r in plans),
               expected_released_allocated_bytes=released,shared_hardlink_paths=sum(r['nlink']>1 for r in plans),
               categories=summary,preserved_hash_records=len(preserved),all_selected_paths_untracked_or_ignored=True,
               archives=archives,source_model_loads=0))
    write_json('hash_cache.json',HASHES)
    print('PLAN_READY',len(plans),'logical',sum(r['bytes'] for r in plans),'expected allocated release',released,flush=True)

if __name__=='__main__': main()
