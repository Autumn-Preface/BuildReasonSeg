"""Verify a small deterministic set of CRC duplicate candidates by full-file SHA."""
import collections
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
import numpy as np
from . import audit_sources as source


def main():
    import zipfile
    data=json.loads((source.OUT/'label_audit.json').read_text(encoding='utf-8'))
    inv=json.loads((source.OUT/'official_archive_inventory.json').read_text(encoding='utf-8'))
    groups=data['archive_crc_size_duplicate_candidates']
    cross=[g for g in groups if len({s for s,n in g})>1]
    # Only two train/val pairs; do not inspect test appearance or run predictions.
    candidates=[g for g in cross if {'train','val'}<={s for s,n in g}]
    chosen=[]
    for g in candidates[:2]:
        chosen.append([next((s,n) for s,n in g if s==split) for split in ('train','val')])
    local=source.ROOT/'artifacts/damage_p1a/duplicate_samples'
    local.mkdir(parents=True,exist_ok=True)
    archives={}
    result={'crc_size_candidate_groups':len(groups),'cross_split_candidate_groups':len(cross),
        'candidate_pair_counts':dict(collections.Counter('__'.join(sorted({s for s,n in g})) for g in cross)),
        'selection':'first two archive-order train/val candidate groups; not representative; no test image appearance inspected',
        'verified_pairs':[]}
    for pair in chosen:
        rows=[]
        for split,name in pair:
            if split not in archives:
                archives[split]=zipfile.ZipFile(source.RemoteZip(inv['splits'][split]['archives']['images.zip']['drive_id']))
            content=archives[split].read(name)
            path=local/(split+'_'+Path(name).name)
            path.write_bytes(content)
            with Image.open(io.BytesIO(content)) as im:
                rgb=np.array(im.convert('RGB'))
                size=list(im.size)
            rows.append({'split':split,'archive_path':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),
                'decoded_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'size_wh':size,
                'local':path.relative_to(source.ROOT).as_posix()})
        result['verified_pairs'].append({'files':rows,'encoded_identical':rows[0]['sha256']==rows[1]['sha256'],
                                        'decoded_identical':rows[0]['decoded_rgb_sha256']==rows[1]['decoded_rgb_sha256']})
        source.dump('duplicate_sample_audit.json',result)
    for z in archives.values():z.close()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
