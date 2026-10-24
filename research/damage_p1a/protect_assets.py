"""Fresh read-only protected-delivery snapshot and verification; never repairs assets."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/damage_p1a/20261025'
BASE=ROOT/'artifacts/damage_p1a/protected_baseline.json'
PATHS=[Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Advisor_RC1'),
       Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Demo_V3'),
       ROOT/'artifacts/legacy_archive/20261019',ROOT/'datasets/damage/local']


def inventory():
    result=[]
    for root in PATHS:
        if not root.is_dir():raise FileNotFoundError(root)
        for path in sorted(root.rglob('*')):
            if not path.is_file():continue
            stat=path.stat()
            with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            result.append({'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':digest})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');a=p.parse_args()
    if a.snapshot:
        if BASE.exists():raise FileExistsError('Never replace earlier baseline')
        rows=inventory();BASE.parent.mkdir(parents=True,exist_ok=True)
        BASE.write_text(json.dumps(rows,indent=2),encoding='utf-8')
        (OUT/'protected_baseline_summary.json').write_text(json.dumps({'files':len(rows),'logical_bytes':sum(r['bytes'] for r in rows),
            'inventory_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'roots':[str(p) for p in PATHS],
            'capture_scope':'fresh P1A checkpoint before any training; task code never writes these roots'},indent=2),encoding='utf-8')
        print('Protected snapshot',len(rows));return
    expected=json.loads(BASE.read_text(encoding='utf-8'))
    now=inventory()
    assert expected==now,'Protected external assets changed; STOP without restoring'
    changed=subprocess.check_output(['git','diff','--name-only','a2f43aa2f5379920b9db6452dcc18b80481cdb76'],cwd=ROOT,text=True).splitlines()
    protected=[x for x in changed if x.startswith(('delivery_src/','demo/','docs/research/damage_p0/','reports/damage_p0/',
        'reports/demo_r1/','evaluation/damage_p0/'))]
    assert not protected,protected
    result={'protected_files':len(now),'fresh_hash_mtime_size_match':True,'protected_git_paths_changed':protected,
        'old_environment_change_scope':'only read for offline clone; sampled binary copies have distinct file IDs and equal SHA',
        'real_damage_training_steps':0,'formal_test_predictions_read':False}
    (OUT/'protection_final.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
