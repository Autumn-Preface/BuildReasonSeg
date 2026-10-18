"""Task-local inventory / preservation audit. No model imports or execution."""
import argparse
import collections
import csv
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ROOTS = [
    Path(r'C:\D\DeepSeekHarness'),
    Path(r'C:\D\DeepSeek Harness'),
    Path(r'C:\Users\ROG\AppData\Local\OpenAI\Codex'),
    Path(r'C:\Program Files\WindowsApps\OpenAI.Codex_26.1013.4894.0_x64__2p2nqsd0c76g0'),
    Path(r'C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime'),
]

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(name, rows, fields):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def inventory(tag):
    rows, errors, links = [], [], []
    for root in ROOTS:
        stack = [root]
        while stack:
            parent = stack.pop()
            try:
                with os.scandir(parent) as entries:
                    for e in entries:
                        p = Path(e.path)
                        try:
                            st = e.stat(follow_symlinks=False)
                            if st.st_file_attributes & 0x400:
                                links.append(str(p))
                                continue
                            if e.is_dir(follow_symlinks=False):
                                stack.append(p)
                            elif e.is_file(follow_symlinks=False):
                                rows.append(dict(root=str(root), path=str(p), bytes=st.st_size,
                                                 mtime_ns=st.st_mtime_ns, dev=st.st_dev,
                                                 inode=st.st_ino, nlink=st.st_nlink))
                        except OSError as ex:
                            errors.append(dict(path=str(p), error=str(ex)))
            except OSError as ex:
                errors.append(dict(path=str(parent), error=str(ex)))
    rows.sort(key=lambda r: r['path'].casefold())
    with gzip.open(OUT / f'files_{tag}.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    directories = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        root = Path(r['root'])
        p = Path(r['path']).parent
        while p == root or root in p.parents:
            directories[str(p)][0] += r['bytes']
            directories[str(p)][1] += 1
            if p == root:
                break
            p = p.parent
    drows = [dict(path=p, bytes=v[0], files=v[1], gib=round(v[0]/2**30, 6),
                  accounting='recursive_logical_not_additive') for p, v in directories.items()]
    drows.sort(key=lambda x: -x['bytes'])
    write_csv('disk_inventory.csv' if tag == 'before' else 'disk_inventory_after.csv',
              drows, ['path','bytes','files','gib','accounting'])
    write_csv(f'largest_files_{tag}.csv', sorted(rows, key=lambda x: -x['bytes'])[:300], list(rows[0]))
    totals = {}
    for root in ROOTS:
        rr = [r for r in rows if r['root'] == str(root)]
        totals[str(root)] = dict(files=len(rr), logical_bytes=sum(r['bytes'] for r in rr),
                                physical_bytes=None)
    write_json(f'inventory_{tag}.json', dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               roots=totals, errors=errors, skipped_reparse_points=links,
               disk_free_bytes=shutil.disk_usage('C:\\').free))
    print(json.dumps(dict(tag=tag, totals=totals, errors=len(errors), links=len(links)), ensure_ascii=False))

def baseline():
    tracked = subprocess.check_output(['git','ls-files','-z'], cwd=REPO).decode().split('\0')
    records = [dict(path=p, bytes=(REPO/p).stat().st_size, sha256=digest(REPO/p)) for p in tracked if p]
    write_json('tracked_before.json', records)
    write_json('git_start.json', dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO).decode().strip(),
               head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip(),
               status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=REPO).decode(),
               tracked_count=len(records)))
    print('Tracked baseline:', len(records))

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('action', choices=['before','after','baseline'])
    args = a.parse_args()
    baseline() if args.action == 'baseline' else inventory(args.action)
