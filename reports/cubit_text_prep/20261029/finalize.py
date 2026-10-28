"""Persist bounded visual triage decisions; no raw file access."""
import json,pathlib,csv
from scan import dump
b=pathlib.Path(__file__).parent
pool=json.loads((b/'derived_run2/pilot_pool.json').read_text())+json.loads((b/'supplement/pool.json').read_text())
reject={3:2,4:2,5:2,6:1,28:25,29:27,33:32,34:31,38:31,39:31,40:31,41:37,42:31,44:15,45:10,46:21,47:35,48:37,50:49}
observations={1:'墙面单条线状损伤；近景',2:'窗框下方单条线状损伤；可检查窗框参照',7:'管线旁分叉线状损伤',8:'管线与叶影附近线状损伤；阴影干扰',9:'墙角和横梁附近两段线状损伤',10:'多分支线状损伤；需定义完整目标集合',11:'空调下方线状损伤；遮挡与线缆干扰',12:'窗与管线旁竖向线状损伤',13:'网状线状损伤；碎片实例歧义',14:'梁下交叉线状损伤',15:'管道旁多条斜向线状损伤',16:'屋顶梁边缘损伤；截断',17:'梁底与灯具附近多条线状损伤',18:'裸墙多段损伤；位置比较',19:'网状损伤；内部实例过碎',20:'网状损伤；上下/左右需唯一性复核',21:'多分支损伤；替代解释较多',22:'竖向主线与下部分支；部件完整性待定',23:'水平主线与支线；碎片化',24:'开口上沿网状损伤；边缘截断',25:'梁下单处表面缺损；近景',26:'梁底长条缺损；局部遮挡',27:'梁下单处缺损；钢筋可见',30:'梁底细长缺损；目标边界需原图复核',31:'梁柱附近多处缺损；上下参照可检查',32:'梁底多处缺损；位置可比较',35:'墙面缺损周围存在未圈细线；有效域未知',36:'露筋缺损多实例；边界与集合需核对',37:'墙角及排水管旁多处缺损',43:'屋面梁与墙角附近线状损伤；远景',49:'梁底三处缺损；可比较左右位置'}
rel={2:'窗框',7:'管线',8:'管线',9:'墙角/梁',11:'空调',12:'窗框/管线',14:'梁',15:'管道',17:'灯具/梁',24:'开口',31:'梁柱',37:'排水管/墙角',43:'屋面梁/墙角'}
final=[];rejected=[]
for r in pool:
    n=r['pilot_order']
    if n in reject:
        rejected.append(dict(image_id=r['image_id'],pilot_order=n,reason='bounded_visual_same_scene_or_shifted_view',similar_to_order=reject[n]));continue
    r['candidate_status']='codex_thumbnail_triaged_not_human_annotated'
    r['visual_observation_provisional']=observations[n]
    r['pilot_task_opportunities']=['direct_target']+(['position_reference'] if r['instances']>1 else [])+(['spatial_or_visual_semantic_review'] if n in rel else [])
    r['reference_to_verify']=rel.get(n)
    r['focus_instance_id']=r['target_options'][0]
    r['focus_policy']='first original polygon row only; reviewer may choose another; no accepted target yet'
    r['ambiguity_flags']+=['thumbnail_only_review','same_building_possible','reference_not_annotated']
    r['instruction_text']=None
    r['preview_relpath']=(f'derived_run2/preview_non_gt/{n:02}_{r["image_id"].split("/")[-1]}.jpg' if n<=42 else f'supplement/{n}_{r["image_id"].split("/")[-1]}.jpg')
    final.append(r)
assert len(final)==31
dump(b/'pilot_candidates.json',final);dump(b/'pilot_rejections.json',rejected)
with (b/'pilot_candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['order','image_id','focus_instance_id','all_instance_ids','opportunities','observation','ambiguity','preview'])
    for r in final:w.writerow([r['pilot_order'],r['image_id'],r['focus_instance_id'],';'.join(r['target_options']),';'.join(r['pilot_task_opportunities']),r['visual_observation_provisional'],';'.join(r['ambiguity_flags']),r['preview_relpath']])
html=['<!doctype html><meta charset="utf-8"><title>CUBIT 31图候选审查</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto}article{border-top:1px solid #aaa;padding:16px}img{max-width:640px}code{overflow-wrap:anywhere}</style><h1>31图试标注候选（尚未人工标注）</h1><p>全部为train。预览为非GT衍生叠加图；编号是原标签行号。类别及有效域未知，参照物和目标需原图复核。官方test不在此页面。</p>']
for r in final:html.append(f'<article><h2>{r["pilot_order"]}. {r["image_id"]}</h2><img src="{r["preview_relpath"]}"><p>{r["visual_observation_provisional"]}</p><p>机会：{", ".join(r["pilot_task_opportunities"])}</p><p>待复核焦点：<code>{r["focus_instance_id"]}</code></p><details><summary>全部实例及歧义</summary><p>{"<br>".join(r["target_options"])}</p><p>{", ".join(r["ambiguity_flags"])}</p></details></article>')
(b/'review_gallery.html').write_text('\n'.join(html),encoding='utf8')
print(json.dumps(dict(final=len(final),rejected=len(rejected),opportunities={k:sum(k in r['pilot_task_opportunities'] for r in final) for k in ['direct_target','position_reference','spatial_or_visual_semantic_review']})))
