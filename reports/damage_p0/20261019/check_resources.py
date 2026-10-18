"""Metadata and HEAD checks only, no model weights downloaded."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from fetch_sources import get,OUT,ROOT
import concurrent.futures,json,urllib.request,datetime

sources={
 'ctcrack_correct_paper':'https://arxiv.org/abs/2302.11728',
 'ctcrack_fulltext':'https://arxiv.org/pdf/2302.11728',
 'ctcrack_license':'https://raw.githubusercontent.com/HqiTao/CT-crackseg/main/LICENSE',
 'ctcrack_requirements':'https://raw.githubusercontent.com/HqiTao/CT-crackseg/main/requirements.txt',
 'cracksam_details':'https://raw.githubusercontent.com/KG-TSI-Civil/CrackSAM/main/CrackSAM/README.md',
 'cracksam_requirements':'https://raw.githubusercontent.com/KG-TSI-Civil/CrackSAM/main/CrackSAM/requirements.txt',
 'cracksam_abs':'https://arxiv.org/abs/2312.04233',
 'crack_edge_abs':'https://arxiv.org/abs/2412.07205',
 'deepcrack_paper':'https://raw.githubusercontent.com/yhlleo/DeepCrack/master/paper/DeepCrack-Neurocomputing-2019.pdf',
 'cubit_page':'https://www.sciencedirect.com/science/article/pii/S092658052600180X',
 'sam3_weight_info':'https://huggingface.co/api/models/facebook/sam3?blobs=true',
 'sam2_weight_card':'https://huggingface.co/facebook/sam2.1-hiera-tiny/raw/main/README.md',
}
repos=['BenyunZhao/CUBIT-InSeg','phiyodr/dacl10k-toolkit','ben-z-original/s2ds',
       'yhlleo/DeepCrack','KG-TSI-Civil/CrackSAM','HqiTao/CT-crackseg',
       'spiderforest/MixerCSeg','facebookresearch/sam2','facebookresearch/sam3',
       'NVlabs/SegFormer','milesial/Pytorch-UNet','ultralytics/ultralytics']
sources.update({r.replace('/','_')+'_commit':'https://api.github.com/repos/'+r+'/commits?per_page=1' for r in repos})
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(get,sources.items()))
p=OUT.parent/'source_access.json';old=json.loads(p.read_text(encoding='utf-8'));old+=rows
for r in old:
    if r['id']=='ctcrack_paper':r['excluded_reason']='ID title mismatch discovered; correct arXiv is 2302.11728, do not cite 2302.11707'
p.write_text(json.dumps(old,indent=2,ensure_ascii=False),encoding='utf-8')
pins=[]
for repo in repos:
    f=OUT/(repo.replace('/','_')+'_commit.txt')
    if f.exists():
        d=json.loads(f.read_text(encoding='utf-8'))[0]
        pins.append({'repo':repo,'sha':d['sha'],'commit_url':d['html_url'],'commit_date':d['commit']['committer']['date']})
(OUT.parent/'resource_pins.json').write_text(json.dumps(pins,indent=2),encoding='utf-8')
heads=[]
for name,url in [
 ('sam2.1_hiera_tiny','https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'),
 ('yolo11s-seg','https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt'),
 ('sam3','https://huggingface.co/facebook/sam3/resolve/main/sam3.pt')]:
    row={'name':name,'url':url,'method':'HEAD'}
    try:
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=25) as r:
            row.update(status=r.status,content_length=r.headers.get('Content-Length'),downloaded_weight_bytes=0)
    except Exception as e:row['error']=repr(e)
    heads.append(row)
(OUT.parent/'weight_access.json').write_text(json.dumps(heads,indent=2),encoding='utf-8')
for r in rows:print(r['id'],r.get('status'),r.get('error',''))
print('weight heads',heads)
