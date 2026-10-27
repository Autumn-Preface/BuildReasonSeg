"""HEAD-only availability check; never read/download dataset response bodies."""
import concurrent.futures
import datetime
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
old=json.loads((ROOT/'reports/damage_p1a/20261025/official_archive_inventory.json').read_text(encoding='utf-8'))


def head(item):
    split,archive=item
    url='https://drive.usercontent.google.com/download?id='+archive['drive_id']+'&export=download&confirm=t'
    row={'split':split,'url':url,'method':'HEAD','response_body_bytes_read':0,'expected_bytes':archive['archive_bytes']}
    try:
        req=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'BuildReasonSeg-storage-maintenance-HEAD-only'})
        with urllib.request.urlopen(req,timeout=20) as response:
            row.update(status=response.status,headers=dict(response.headers))
        length=int(row['headers'].get('Content-Length','0'))
        row['available_same_size']=row['status']==200 and length==archive['archive_bytes'] and 'text/html' not in row['headers'].get('Content-Type','')
    except Exception as error:row.update(error=repr(error),available_same_size=False)
    return row


items=[(split,info['archives']['labels.zip']) for split,info in old['splits'].items()]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(head,items))
(OUT/'source_header_checks.json').write_text(json.dumps({'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'body_bytes_downloaded':0,'scope':'HEAD only, not archive hash/recovery proof','checks':rows},indent=2),encoding='utf-8')
print(json.dumps([{k:r[k] for k in ('split','available_same_size')} for r in rows],indent=2))
