import sys
sys.stdout.reconfigure(encoding='utf-8')
from fetch_sources import get, OUT, ROOT
import concurrent.futures,json,subprocess

sources={
 'cif_readme':'https://huggingface.co/datasets/ibm-research/cif-dataset/raw/main/README.md',
 'cif_paper':'https://arxiv.org/html/2605.18413v1',
 'cif_info':'https://datasets-server.huggingface.co/info?dataset=ibm-research%2Fcif-dataset&config=default',
 'cif_train_rows':'https://datasets-server.huggingface.co/rows?dataset=ibm-research%2Fcif-dataset&config=default&split=train_full&offset=0&length=3',
 'cracksam_readme':'https://raw.githubusercontent.com/KG-TSI-Civil/CrackSAM/main/README.md',
 'cracksam_tree':'https://api.github.com/repos/KG-TSI-Civil/CrackSAM/git/trees/main?recursive=1',
 'cracksam_paper':'https://arxiv.org/html/2312.04233v2',
 'mixer_readme':'https://raw.githubusercontent.com/spiderforest/MixerCSeg/main/README.md',
 'mixer_license':'https://raw.githubusercontent.com/spiderforest/MixerCSeg/main/LICENSE',
 'mixer_requirements':'https://raw.githubusercontent.com/spiderforest/MixerCSeg/main/requirements.txt',
 'mixer_paper':'https://arxiv.org/html/2603.01361v1',
 'cldice':'https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html',
 'boundary_loss':'https://proceedings.mlr.press/v102/kervadec19a.html',
 'crack_edge_sam':'https://arxiv.org/html/2412.07205v1',
 'ctcrack_paper':'https://arxiv.org/abs/2302.11707',
 'segformer_weights':'https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/raw/main/README.md',
}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(get,sources.items()))
p=OUT.parent/'source_access.json'
rows=json.loads(p.read_text(encoding='utf-8'))+rows
p.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
for r in rows[-len(sources):]:print(r['id'],r.get('status'),r.get('bytes'),r.get('error',''))
# Decode WSL native UTF-16 output without changing/starting a distribution.
wsl={}
for args in [['--version'],['--status'],['--list','--verbose']]:
 r=subprocess.run(['C:/Windows/System32/wsl.exe',*args],capture_output=True,timeout=15)
 wsl[' '.join(args)]={'returncode':r.returncode,'stdout':r.stdout.decode('utf-16-le',errors='replace'),'stderr':r.stderr.decode('utf-16-le',errors='replace')}
(OUT.parent/'wsl_facts.json').write_text(json.dumps(wsl,indent=2,ensure_ascii=False),encoding='utf-8')
