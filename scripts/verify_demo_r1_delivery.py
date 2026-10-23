"""Read-only final evidence audit; never rebuilds or writes protected packages."""
import hashlib
import json
import queue
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import build_task8e_user_demo as base
from scripts import build_task8f_user_demo as v2


def main():
    report = REPO / 'reports/demo_r1/20261023'
    package = Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Demo_V3')
    before = json.loads((report / 'preflight.json').read_text(encoding='utf-8'))
    base.verify_inventory(base.SOURCE, before['rc1_before'])
    manifest = json.loads((package / '_engine/package_manifest.json').read_text(encoding='utf-8-sig'))
    for row in manifest['files']:
        assert base.identity(package / row['path']) == {k: row[k] for k in ('bytes', 'sha256')}, row['path']
        if row['role'] == 'v3_ui':
            assert (package / row['path']).read_bytes() == (REPO / 'demo/user_demo_v3' / row['path']).read_bytes(), row['path']
    _, changed = v2.plan(v2.accepted_blobs())
    for name, data in changed.items():
        assert (package / '_engine' / name).read_bytes() == data, name
    archive_expected = {
        'BuildReasonSeg_Demo_V1_non_model.zip': '6d117df063913488328ae9ffa22b1d2f218cd4e8f0ac76acf1d1269c01cc111e',
        'BuildReasonSeg_Demo_V2_non_model.zip': '38b7188237d15b95011cccdc80f99a5d8bce051e8d9b0d11a457b968dd3171b4'}
    archives = {}
    for name, digest in archive_expected.items():
        found = list((REPO / 'artifacts/legacy_archive/20261019').rglob(name))
        assert len(found) == 1, found
        archives[name] = base.identity(found[0])
        assert archives[name]['sha256'] == digest
    sys.path.insert(0, str(package))
    from _ui.trace import reopen_trace
    run = package / 'results/20261023_230642_1034'
    trace = run / 'trace'
    identities = base.inventory(run)
    events = queue.Queue()
    saved = reopen_trace(trace / 'trace_manifest.json', package, events)
    assert identities == base.inventory(run), 'Replay modified saved files'
    records = [json.loads(x) for x in (trace / 'stage_events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {e['run_id'] for e in records} == {run.name}
    assert {e['stage_id'] for e in records if e['state'] == 'COMPLETED'} == set(range(1, 10))
    # Keep real evidence separate from the fake E404 presentation window.
    real = report / 'real_run'
    real.mkdir(exist_ok=True)
    for name in ('trace_manifest.json', 'stage_events.jsonl'):
        (real / name).write_bytes((trace / name).read_bytes())
    (real / 'result_summary.json').write_bytes((run / 'result_summary.json').read_bytes())
    for name in ('mask.png', 'overlay.png'):
        (real / name).write_bytes((run / name).read_bytes())
    audit = base.forbidden_audit(package)
    assert audit['pass'], audit
    diff = subprocess.check_output(['git', 'diff', '--name-only', '0ef69a2ed6f80251dc14760bab81053a8468954e'], cwd=REPO, text=True).splitlines()
    protected = [p for p in diff if p.startswith(('delivery_src/', 'reports/damage_p0/', 'research/', 'data/', 'evaluation/', 'demo/user_demo_v1/', 'demo/user_demo_v2/'))]
    assert not protected, protected
    result = {'rc1_inventory_unchanged': True, 'rc1_records': len(before['rc1_before']),
        'package_hashes_pass': True, 'package_files': len(manifest['files']), 'package_bytes': manifest['total_bytes'],
        'package_ui_matches_repository': True, 'derived_engine_exact_v2_observer_patch': True,
        'archives': archives, 'forbidden_audit': audit, 'protected_git_paths_changed': protected,
        'read_only_replay_pass': True, 'replay_queued_events': events.qsize(),
        'real_run_id': run.name, 'real_events': len(records), 'real_all_nine_completed': True,
        'real_output_files': {p: base.identity(run / p) for p in ('mask.png', 'overlay.png')},
        'real_rc1_vs_v3_paired_equivalence': 'NOT_PERFORMED', 'semantic_correctness': 'NOT_EVALUATED'}
    (report / 'final_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
