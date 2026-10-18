"""Bounded primary-source capture. No model/dataset archives are downloaded."""
import concurrent.futures, datetime, hashlib, json, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'evaluation/damage_p0/sources'
OUT.mkdir(parents=True,exist_ok=True)
SOURCES={
 'cubit_readme':'https://raw.githubusercontent.com/BenyunZhao/CUBIT-InSeg/master/README.md',
 'cubit_tree':'https://api.github.com/repos/BenyunZhao/CUBIT-InSeg/git/trees/master?recursive=1',
 'cubit_drive':'https://drive.google.com/drive/folders/1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b?usp=sharing',
 'cubit_baidu':'https://pan.baidu.com/s/1OV9-K3d-a0i5Sv5coa9EgQ',
 'dacl_readme':'https://raw.githubusercontent.com/phiyodr/dacl10k-toolkit/master/README.md',
 'dacl_tree':'https://api.github.com/repos/phiyodr/dacl10k-toolkit/git/trees/master?recursive=1',
 's2ds_readme':'https://raw.githubusercontent.com/ben-z-original/s2ds/main/README.md',
 'deepcrack_readme':'https://raw.githubusercontent.com/yhlleo/DeepCrack/master/README.md',
 'deepcrack_tree':'https://api.github.com/repos/yhlleo/DeepCrack/git/trees/master?recursive=1',
 'omnicrack_readme':'https://raw.githubusercontent.com/ben-z-original/omnicrack30k/main/README.md',
 'sam2_readme':'https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md',
 'sam2_setup':'https://raw.githubusercontent.com/facebookresearch/sam2/main/setup.py',
 'sam3_readme':'https://raw.githubusercontent.com/facebookresearch/sam3/main/README.md',
 'sam3_license':'https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE',
 'sam3_pyproject':'https://raw.githubusercontent.com/facebookresearch/sam3/main/pyproject.toml',
 'segformer_readme':'https://raw.githubusercontent.com/NVlabs/SegFormer/master/README.md',
 'segformer_license':'https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE',
 'unet_official':'https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/',
 'unet_impl':'https://raw.githubusercontent.com/milesial/Pytorch-UNet/master/README.md',
 'unet_license':'https://raw.githubusercontent.com/milesial/Pytorch-UNet/master/LICENSE',
 'yolo11':'https://docs.ultralytics.com/models/yolo11/',
 'yolo_seg':'https://docs.ultralytics.com/tasks/segment/',
 'yolo_license':'https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE',
 'ctcrack_readme':'https://raw.githubusercontent.com/HqiTao/CT-crackseg/main/README.md',
 'omnicrack_paper':'https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.pdf',
 'dacl_paper':'https://openaccess.thecvf.com/content/WACV2024/papers/Flotzinger_dacl10k_Benchmark_for_Semantic_Bridge_Damage_Segmentation_WACV_2024_paper.pdf',
 'efficient_paper':'https://openaccess.thecvf.com/content/WACV2025/papers/Zim_EfficientCrackNet_A_Lightweight_Model_for_Crack_Segmentation_WACV_2025_paper.pdf',
 's2ds_paper':'https://ben-z-original.github.io/2022_Benz_DetectionStructuralDefects.pdf',
}

def get(item):
    key,url=item
    row={'id':key,'url':url,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    suffix='.pdf' if '.pdf' in url else '.txt'
    p=OUT/(key+suffix)
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Damage-P0 research audit'})
        with urllib.request.urlopen(req,timeout=35) as r:
            row.update(status=r.status,content_length=r.headers.get('Content-Length'),final_url=r.url)
            data=r.read(8*1024*1024+1)
            if len(data)>8*1024*1024: raise ValueError('8 MiB per source cap exceeded')
        p.write_bytes(data)
        row.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),local=str(p.relative_to(ROOT)))
    except Exception as e: row['error']=repr(e)
    return row

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(get,SOURCES.items()))
    (OUT.parent/'source_access.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    for r in rows: print(r['id'],r.get('status'),r.get('bytes'),r.get('error',''))
