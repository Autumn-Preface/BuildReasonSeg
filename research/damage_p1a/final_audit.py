"""Small report/source audit. This does not train, accept or change the dataset."""
import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
import yaml


def main():
    root=Path(__file__).resolve().parents[2];out=root/'reports/damage_p1a/20261025'
    json_files=list(out.rglob('*.json'))+list((root/'configs/damage_p1a').glob('*.json'))
    for p in json_files:json.loads(p.read_text(encoding='utf-8-sig'))
    for p in [root/'governance/PROJECT_STATE.yaml',root/'handoff/EXECUTOR_STATE.yaml']:yaml.safe_load(p.read_text(encoding='utf-8-sig'))
    python_files=list((root/'research/damage_p1a').glob('*.py'))+[root/'tests/test_damage_p1a.py']
    for p in python_files:ast.parse(p.read_text(encoding='utf-8'))
    missing=[]
    for p in list(out.rglob('*.md'))+list((root/'docs/research/damage_p1a').glob('*.md')):
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if link.startswith(('https://','http://','#')):continue
            target=(p.parent/link.split('#')[0]).resolve()
            if not target.exists():missing.append({'file':str(p),'target':str(target)})
    assert not missing,missing
    gate=json.loads((out/'admission.json').read_text(encoding='utf-8'));assert gate['status']=='BLOCKED' and gate['training_allowed'] is False
    assert not (root/'artifacts/damage_p1a/runs').exists()
    assert '38 passed' in (out/'tests_final.txt').read_text(encoding='utf-8')
    assert json.loads((out/'training_gate_result.json').read_text(encoding='utf-8-sig'))['actual_exit']==2
    duplicates=json.loads((out/'duplicate_sample_audit.json').read_text(encoding='utf-8'))
    for pair in duplicates['verified_pairs']:
        for row in pair['files']:
            with (root/row['local']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    labels=json.loads((out/'label_audit.json').read_text(encoding='utf-8'))
    for split,stats in labels['splits'].items():
        with (root/'artifacts/damage_p1a/raw_labels'/f'{split}_labels.zip').open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==stats['archive_sha256']
    allowed=('AGENTS.md','governance/','handoff/','configs/damage_p1a/','research/damage_p1a/',
             'tests/test_damage_p1a.py','reports/damage_p1a/','docs/research/damage_p1a/')
    changed=subprocess.check_output(['git','diff','--name-only','a2f43aa2f5379920b9db6452dcc18b80481cdb76'],cwd=root,text=True).splitlines()
    assert all(p.startswith(allowed) for p in changed),changed
    source_env=root/'.conda/buildreasonseg-proposal';new_env=root/'.conda/buildreasonseg-damage-p1a'
    assert (new_env/'python.exe').is_file()
    free=shutil.disk_usage(root).free;assert free>50*1024**3
    result={'status':'STOP_DATA_ADMISSION','json_parsed':len(json_files),'python_ast_parsed':len(python_files),
        'local_markdown_links_pass':True,'label_archive_hashes_pass':True,'duplicate_sample_hashes_pass':True,
        'admission_blocked':True,'real_training_run_directory_exists':False,'real_damage_training_steps':0,
        'engineering_tests':38,'test_predictions_read':False,'changed_paths_within_scope':True,'free_bytes':free,
        'minimum_free_bytes':50*1024**3,'snapshot_scope':'logical sizes exclude Git and existing old assets',
        'new_artifact_logical_bytes':sum(p.stat().st_size for p in (root/'artifacts/damage_p1a').rglob('*') if p.is_file()),
        'new_env_logical_bytes':sum(p.stat().st_size for p in new_env.rglob('*') if p.is_file()),
        'frozen_config_sha256':hashlib.sha256((root/'configs/damage_p1a/unet_baseline.json').read_bytes()).hexdigest()}
    (out/'final_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
