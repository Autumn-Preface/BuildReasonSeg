"""Build the review pack from unchanged saved artifacts and recorded numbers only."""
from __future__ import annotations
import csv
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE))
from review_audit import identity,dump,git
BASE='e06777b0a0844c5c29325701eb2e819bd193ff04'
DEMO=Path('C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Demo_V2')
RUN=DEMO/'results/20261012_155017_1034'

def read(relative):return json.loads((REPO/relative).read_text(encoding='utf-8'))
def rel(path):return Path(path).relative_to(HERE).as_posix()
def copy_new(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Preserve existing review output: '+str(target))
    with Path(source).open('rb') as reader,target.open('xb') as writer:shutil.copyfileobj(reader,writer)
    if identity(source)!=identity(target):raise ValueError('Copied bytes do not match source')

def records():
    return [
      ('E01','YOLO历史训练原图','baseline/yolo_whu/plots/results.png','历史baseline训练',
       '100轮训练/验证损失与Box/Mask指标曲线。','baseline/yolo_whu/run_record/results.csv; baseline/yolo_whu/run_record/args.yaml; baseline/yolo_whu/manifest.json',
       '历史YOLOv8m-seg建筑实例分割baseline的100轮训练与验证记录；Mask验证mAP50最终0.80693，最高记录0.84373（第44轮）。',
       '成果附件原图；PPT优先用E11重绘摘要','验证集而非测试集；不是YOLO26m或MLLM结果。sem_loss全零，不能称新增语义监督；不得选择权重或将峰值当最终精度。',
       '原图与两个本地legacy根目录results.png均须hash一致；数值逐项对照CSV。'),
      ('E02','原生矢量与伪实例GT审查','evaluation/task6k1_samples/1_0_panel.png','数据标注与质量审查',
       '同一train/1_0样本的RGB、raster label、native vector实例、pseudo实例及差异。','evaluation/task6k1_samples/index.json; evaluation/task6k1_tile_mapping.json; scripts/task6k1_samples.py',
       '同一样本1_0从左至右为遥感RGB、栅格标签、原生矢量实例、历史伪实例、二者差异；记录中native实例4个、pseudo实例3个。',
       '申报书数据准备；PPT“数据与质量控制”','是GT来源比较，不是模型预测；伪实例由二值连通域形成，不能充当所有独立建筑真值。图内无列标题，应配本文图注。',
       'index.json split=train/stem=1_0及五列定义；生成脚本按同一stem拼图；tile_mapping有采样像素身份审查。'),
      ('E03','矢量GT缓存一致性原图','evaluation/task6l_samples/1_0_overlay.png','BuildSpatialReason v0.2数据迁移',
       '同一1_0的RGB、canonical GT绿色前景、native vector重新栅格化蓝色前景。','evaluation/task6l_verdict.json; evaluation/task6l_vector_dataset_integrity.json; scripts/task6l_validate.py',
       '原生矢量数据迁移质量控制示例：RGB、缓存标签前景（绿）与源矢量核对结果（蓝）；既有审查记录40个样本标签一致。',
       '成果附件数据质量；PPT数据页备选','绿/蓝都是GT或GT核对，绝不能写成预测与GT达到IoU=1。17,388是切片索引数，包含11,909空切片，不等于17,388条推理问答。',
       '生成脚本同一record.tile_id=1_0拼接rgb/cached/fresh；既有verdict引用min_iou=1/40，不重跑栅格化。'),
      ('E04','正式Demo目标错误对照','evaluation/task8c_final_demo_v1/above_review.png','真实运行失败分析',
       '同一样本1008原RGB、auto/GT reference边界、saved overlay与GT target边界及后验指标。','evaluation/task8c_final_demo_v1.json; docs/task8c_final_demo_v1.md; scripts/task8c_final_demo_evaluate.py',
       '正式定性Demo的above失败案例：参考建筑接近GT，但最终目标身份不匹配（target IoU=0.077622）；结构运行SUCCESS不能代表选对最近目标。',
       'PPT“问题与下一步”；附件失败分析','必须明确失败；四例已有审查均无完整chain identity match。此前已消费测试样本的历史定性复用，不是新测试或成功率证明。',
       '同一frozen sample_id/test1008；审查图hash匹配Task8C既有artifact记录，三面板由同一个case RGB/已保存输出构成。'),
      ('E05','空间权重与attention诊断','evaluation/task8d_above_target_chain_diagnosis_v1/04_W_and_A.png','空间关系场与离线诊断',
       '已保存W/A空间场以及GT reference4、target6和错误实例7的同context边界。','evaluation/task8d_above_target_chain_diagnosis_v1.json; docs/task8d_above_target_chain_diagnosis_v1.md',
       'above失败案例的空间推理中间量：W与A在同一context坐标下展示，并标出GT参考、GT目标及错误重叠实例，用于检查链路偏差。',
       'PPT方法实现/失败诊断；附件','是实际保存中间量的后验展示，不证明最终选对目标，也不证明某一模块造成错误。两个图颜色分别缩放，不可直接按颜色大小比较W与A。',
       'Task8D产物hash与既有记录一致，沿用Task8C above样本；不重新求场、不重跑模型。'),
      ('E06','真实原型界面空载截图','docs/task8f_final_results.png','原型系统开发',
       '原生Tk Demo V2的结果/推理过程平级页、图片选择、输入、快捷按钮、环境检查与结果控件。','docs/task8f_user_demo_live_trace_v1.md; evaluation/task8f_user_demo_live_trace_v1.json; demo/user_demo_v2/BuildReasonSeg_Demo.py',
       '已实现的本地交互原型界面：支持图片选择、空间指令输入、结果与推理过程两页；本图为环境检查通过后的空载界面。',
       'PPT“原型系统”；申报书工程进展','真实窗口截图但此帧未选图、未推理；不能称真实模型运行完成截图。与10月12日运行记录属于不同时间，不拼接冒充同一运行。',
       'Task8F自有窗口截图/源代码及既有身份记录；排除docs/task8f_fake_*.png。'),
      ('E07','真实运行自动Reference快照',RUN/'trace/stage_03_event_010_1.png','用户Demo实际运行中间量',
       '10月12日1034左侧指令运行的全部候选及黄色边界Reference ID9。',RUN/'result_summary.json',
       '2026年10月12日保存的1034左侧最近建筑查询过程：实例候选保留，程序自动Reference为9，黄色边界标出其区域。',
       'PPT原型流程；附件运行记录','是实际候选和自动选择的展示，不代表Reference已通过GT身份验证；没有手工reference override。此页快照不是整个GUI截图。',
       '同run的trace manifest/event sequence10及PNG SHA；输入RGB像素hash匹配事件2，engine run root一致。'),
      ('E08','真实运行四种空间场快照',RUN/'trace/stage_06_event_018_1.png','用户Demo实际空间推理',
       '同一次1034左侧查询的P_dir/P_near/W/A热力图、原始范围和显示范围。',RUN/'trace/stage_events.jsonl',
       '同次真实Demo记录的方向场、邻近场、组合权重与归一化attention；黄色轮廓为Reference，颜色映射仅用于展示。',
       'PPT方法实现/系统过程；附件','不能把颜色当作目标正确概率；A与其他场的颜色范围不同。场存在不等于空间语义已正确执行。',
       '同run stage6 COMPLETED sequence18，四种array snapshot摘要和对应PNG hash；不根据输入另行重算。'),
      ('E09','真实运行D-B1候选快照',RUN/'trace/stage_08_event_022_1.png','用户Demo实际分割中间量',
       '同run的probability、二值候选Mask和context overlay，原图明示未通过最终结构检查。',RUN/'trace/stage_events.jsonl',
       'D-B1已产生候选Mask及probability的保存快照；该阶段尚未通过最终结构检查，更未验证目标实例身份。',
       '成果附件；PPT原型页备选','必须保留“候选/未经最终结构验证”含义；多处建筑响应不能称已精确选中唯一最近目标。白色区域不是损伤区域。',
       '同run stage8 COMPLETED sequence22；文件hash与事件日志一致，未生成新的模型输出。'),
      ('E10','真实运行最终结构输出',RUN/'overlay.png','用户Demo保存结果',
       '1034左侧指令的已保存最终overlay，建筑区域显示为红色。',RUN/'result_summary.json',
       '同次真实Demo保存的建筑Mask叠加图；日志记载结构流程完成，语义正确性NOT_EVALUATED，需人工检查目标是否符合指令。',
       '成果附件；PPT与E07配对作为系统输出记录','没有对应本次GT目标审核；不能称分割精度高、目标选择正确或损伤识别成功。结构SUCCESS与语义成功严格区分。',
       '与result_summary指向的engine run root之overlay逐字节hash一致；与E07/E08/E09共享同一run ID与输入。')]

def build():
    from PIL import Image,ImageDraw,ImageFont
    import numpy as np
    if (HERE/'audit/selection_records.json').exists():raise FileExistsError('Selection already exists; preserve it')
    rows=[];selection=[];read_records=[]
    for ident,title,source,phase,content,log,caption,location,limit,pair in records():
        source=Path(source);source=source if source.is_absolute() else REPO/source
        original=identity(source);target=HERE/'selected'/(ident+'_'+source.name);copy_new(source,target)
        with Image.open(source) as im:size=f'{im.width}×{im.height}'
        link=''
        if source.is_relative_to(REPO):
            relative=source.relative_to(REPO).as_posix();blob=subprocess_bytes(BASE,relative)
            if blob!=source.read_bytes():raise ValueError('Existing tracked image differs from base')
            link='https://github.com/Autumn-Preface/BuildReasonSeg/blob/'+BASE+'/'+relative
        selection.append({'id':ident,'original_path':str(source),'copy_path':rel(target),'identity':original,'visual_review':'ORIGINAL_IMAGE_INSPECTED','github_source':link})
        rows.append({'编号':ident,'图片名称':title,'原始文件路径':str(source),'对应实验阶段':phase,'图像尺寸':size,'图片展示的实际内容':content,
                     '对应实验日志':('仅有界面验收记录，无该帧模型推理日志：' if ident=='E06' else '是：')+str(log),'建议中文图注':caption,'推荐使用位置':location,'解释限制':limit,'素材类型':'历史原始图片逐字节副本',
                     '副本路径':rel(target),'SHA256':original['sha256'],'同样本或来源核查':pair,'GitHub证据链接':link})
    # All saved previews and their single-run linkage are checked, not visually best-case screened.
    trace=json.loads((RUN/'trace/trace_manifest.json').read_text(encoding='utf-8'))
    events=[json.loads(line) for line in (RUN/'trace/stage_events.jsonl').read_text(encoding='utf-8').splitlines()]
    summary=json.loads((RUN/'result_summary.json').read_text(encoding='utf-8'))
    assert trace['run_id']==RUN.name and trace['event_count']==len(events)==24
    previews=0
    for seq,event in enumerate(events,1):
        assert event['run_id']==RUN.name and event['sequence']==seq
        for preview in event['previews']:
            assert identity(RUN/'trace'/preview['path'])=={k:preview[k] for k in ('bytes','sha256')};previews+=1
    rgb_event=next(e for e in events if e['stage_id']==2 and e['state']=='COMPLETED')
    input_path=Path(summary['input_image'])
    with Image.open(input_path) as im:rgb=np.asarray(im.convert('RGB'))
    import hashlib
    actual_rgb_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
    assert actual_rgb_hash==rgb_event['metadata']['array_snapshot_identities']['rgb']['sha256']
    engine_root=DEMO/summary['engine_run_root'];engine_result=json.loads((engine_root/'diagnostics/result.json').read_text(encoding='utf-8'))
    assert summary['engine_run_root']==trace['engine_run_root']
    assert identity(RUN/'overlay.png')==identity(Path(engine_result['output_paths']['overlay']))
    assert identity(RUN/'mask.png')==identity(Path(engine_result['output_paths']['mask']))
    dump(HERE/'audit/demo_provenance.json',{'run_id':RUN.name,'timestamp':summary['timestamp'],'source_summary':identity(RUN/'result_summary.json'),'trace_status':trace['status'],
            'events':len(events),'previews_checked':previews,'event_png_hashes_match':True,'input_path':str(input_path),'input_identity':identity(input_path),
            'input_rgb_pixel_hash_matches_stage2':True,'input_rgb_sha256':actual_rgb_hash,'same_engine_run_root':True,'overlay_mask_copy_hash_match':True,
            'runtime_status':summary['runtime_status'],'semantic_status':summary['semantic_status'],'gt_verified':False,'new_inference_calls':0,
            'selection_rule':'The two existing V2 runs were both inventoried; first chronological run is used for all four related snapshots, not chosen by visual accuracy.','other_existing_run':'20261012_155256_1034'})
    # Snapshot indispensable external records; existing tracked sources have immutable Git links and hashes.
    for source,name in [(RUN/'result_summary.json','demo_result_summary.json'),(RUN/'trace/trace_manifest.json','demo_trace_manifest.json'),
                        (RUN/'trace/stage_events.jsonl','demo_stage_events.jsonl'),(engine_root/'diagnostics/result.json','demo_engine_result.json')]:
        target=HERE/'source_records'/name;copy_new(source,target);read_records.append({'original_path':str(source),'copy_path':rel(target),**identity(source)})
    # CSV-backed baseline and all three pre-recorded formal seeds, never selecting a weight or rerunning test.
    source_csv=REPO/'baseline/yolo_whu/run_record/results.csv'
    with source_csv.open(encoding='utf-8-sig',newline='') as f:history=list(csv.DictReader(f))
    assert len(history)==100 and [int(r['epoch']) for r in history]==list(range(1,101))
    assert all(float(r['train/sem_loss'])==float(r['val/sem_loss'])==0 for r in history)
    baseline=read('baseline/yolo_whu/manifest.json');final=history[-1]
    assert float(final['metrics/mAP50(M)'])==baseline['metrics']['final_epoch']['mask_mAP50']
    curve=REPO/'baseline/yolo_whu/plots/results.png'
    for legacy in [REPO.parent/'WHU_Building_Segment',Path('C:/D/resources/project/WHU_Building_Segment')]:
        for relative,tracked in [('results.png',curve),('results.csv',source_csv),('args.yaml',REPO/'baseline/yolo_whu/run_record/args.yaml')]:
            assert identity(legacy/'runs/segment/logs/whu_building_v1'/relative)==identity(tracked)
    comparison=read('evaluation/task7j_final_comparison.json');population=read('evaluation/task7j_test_population_manifest.json');pair_population=read('evaluation/task7j_test_pair_manifest.json')
    metric_rows=[]
    for mode in ('oracle','practical'):
        for model in ('z_b3','d_b1'):
            for metric in (('miou','pair_pass_rate') if mode=='oracle' else ('strict_miou','pair_pass_rate')):
                item=comparison[mode][model][metric]
                assert len(item['values'])==3 and math.isclose(statistics.mean(item['values']),item['mean'],abs_tol=1e-12)
                assert math.isclose(statistics.stdev(item['values']),item['std_ddof1'],abs_tol=1e-12)
                metric_rows.append({'reference_mode':mode,'model':model,'metric':metric,'seed_20261001':item['values'][0],'seed_20261002':item['values'][1],
                                    'seed_20261003':item['values'][2],'recorded_mean':item['mean'],'recorded_sample_sd':item['std_ddof1']})
    (HERE/'derived').mkdir(exist_ok=True)
    with (HERE/'derived/formal_metrics_source.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
    copy_new(source_csv,HERE/'source_records/yolo_training_results.csv')
    copy_new(REPO/'baseline/yolo_whu/run_record/args.yaml',HERE/'source_records/yolo_training_args.yaml')
    copy_new(REPO/'evaluation/task7j_final_comparison.json',HERE/'source_records/task7j_final_comparison.json')
    cache=Path(tempfile.gettempdir())/'midterm-evidence-mpl-20261017';cache.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'axes.unicode_minus':False,'font.family':font.get_name(),'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    x=[int(r['epoch']) for r in history]
    axes[0].plot(x,[float(r['train/seg_loss']) for r in history],label='train/seg_loss',color='#2375b8')
    axes[0].plot(x,[float(r['val/seg_loss']) for r in history],label='val/seg_loss',color='#cc5930')
    axes[0].set(title='历史 YOLOv8m-seg：分割损失',xlabel='Epoch',ylabel='原记录 loss');axes[0].legend()
    ys=[float(r['metrics/mAP50(M)']) for r in history];axes[1].plot(x,ys,color='#2375b8')
    best=max(range(len(ys)),key=ys.__getitem__)
    axes[1].scatter([x[best],100],[ys[best],ys[-1]],color=['#2c885b','#cc5930'])
    axes[1].annotate(f'最高记录 epoch{x[best]}: {ys[best]:.5f}',(x[best],ys[best]),xytext=(20,0.92),arrowprops={'arrowstyle':'->'})
    axes[1].annotate(f'最终 epoch100: {ys[-1]:.5f}',(100,ys[-1]),xytext=(50,0.67),arrowprops={'arrowstyle':'->'})
    axes[1].set(title='验证集 Mask mAP50（不是测试精度）',xlabel='Epoch',ylabel='Mask mAP50',ylim=(0,1))
    fig.suptitle('根据已有实验记录重新绘制｜100轮真实CSV，不重新训练或选择权重',fontproperties=font,fontsize=15)
    curve_path=HERE/'derived/E11_yolo_record_redrawn.png';fig.savefig(curve_path,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,(mode,metric,title) in zip(axes.flat,[('oracle','miou','GT Reference｜mIoU'),('practical','strict_miou','预测 Reference｜strict mIoU'),
                                             ('oracle','pair_pass_rate','GT Reference｜成对指令 pass rate'),('practical','pair_pass_rate','预测 Reference｜成对指令 pass rate')]):
        data=[comparison[mode][model][metric] for model in ('z_b3','d_b1')]
        means=[d['mean'] for d in data];sd=[d['std_ddof1'] for d in data]
        ax.bar(['Z-B3','D-B1'],means,yerr=sd,capsize=5,color=['#2375b8','#da793c'],width=.55)
        for i,y in enumerate(means):ax.text(i,y+sd[i]+.025,f'{y:.4f} ± {sd[i]:.4f}',ha='center',fontsize=11)
        ax.set(title=title,ylim=(0,1),ylabel='已有记录 mean ± sample SD');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('根据已有实验记录重新绘制｜冻结架构历史测试，3 seeds全部呈现\n736条canonical L3记录；paired set 274对；不是外立面损伤或自然语言端到端准确率',fontproperties=font,fontsize=15)
    metrics_path=HERE/'derived/E12_formal_metrics_redrawn.png';fig.savefig(metrics_path,dpi=160);plt.close(fig)
    for ident,title,path,source,caption,limits in [
        ('E11','历史训练CSV重绘摘要',curve_path,source_csv,'根据已有实验记录重新绘制：历史YOLOv8m-seg的train/val分割损失及验证Mask mAP50；显示中后期验证损失回升和最终指标低于峰值。','只重绘原CSV，不用曲线推断未记录精度，不改变checkpoint或阈值。'),
        ('E12','冻结测试对照重绘',metrics_path,REPO/'evaluation/task7j_final_comparison.json','根据已有实验记录重新绘制：三seed的Z-B3与D-B1在GT/预测Reference两种设置下的mIoU和成对指令通过率，误差线为样本标准差。','描述性历史结果；test已消费，无重测、选模型、显著性结论。D-B1的GT-reference mIoU增加，实际reference增量很小且pair pass rate更低，不能称全面提升。')]:
        with Image.open(path) as im:size=f'{im.width}×{im.height}'
        rows.append({'编号':ident,'图片名称':title,'原始文件路径':str(source),'对应实验阶段':'已有记录重绘','图像尺寸':size,'图片展示的实际内容':caption,
                     '对应实验日志':'是：'+str(source),'建议中文图注':caption,'推荐使用位置':'PPT“训练与客观评价”；附件','解释限制':limits,
                     '素材类型':'根据已有实验记录重新绘制','副本路径':rel(path),'SHA256':identity(path)['sha256'],
                     '同样本或来源核查':'逐列读取原文件；三seed均值及sample SD与记录核对，未运行推理。','GitHub证据链接':'https://github.com/Autumn-Preface/BuildReasonSeg/blob/'+BASE+'/'+source.relative_to(REPO).as_posix()})
    with (HERE/'evidence_manifest.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(HERE/'audit/selection_records.json',selection)
    dump(HERE/'audit/copied_record_identities.json',read_records)
    # Record everything read for claims, not a claim to have manually inspected all 28k images.
    paths=['baseline/yolo_whu/README.md','baseline/yolo_whu/manifest.json','baseline/yolo_whu/run_record/args.yaml','baseline/yolo_whu/run_record/results.csv',
           'evaluation/build_spatial_reason_v0.1.1_quality.json','docs/build_spatial_reason_v0.1.1_quality_audit.md','evaluation/task6k1_samples/index.json',
           'evaluation/task6k1_tile_mapping.json','scripts/task6k1_samples.py','evaluation/task6l_vector_dataset_integrity.json','evaluation/task6l_verdict.json',
           'scripts/task6l_validate.py','evaluation/task6d_checkpoint_manifest.json','docs/task6d_spatial_grounding_bridge.md','scripts/task6d_evaluate.py',
           'evaluation/task7i_training_db1.json','evaluation/task7i_training_zb3.json','evaluation/task7j_final_comparison.json','evaluation/task7j_test_population_manifest.json',
           'evaluation/task7j_test_pair_manifest.json','evaluation/task7j_verdict.json','docs/task7j_final_test.md','evaluation/task8c_final_demo_v1.json',
           'docs/task8c_final_demo_v1.md','evaluation/task8d_above_target_chain_diagnosis_v1.json','docs/task8d_above_target_chain_diagnosis_v1.md',
           'evaluation/task8f_user_demo_live_trace_v1.json','docs/task8f_user_demo_live_trace_v1.md']
    read_records.extend({'original_path':str(REPO/p),'github_source':'https://github.com/Autumn-Preface/BuildReasonSeg/blob/'+BASE+'/'+p,**identity(REPO/p)} for p in paths)
    dump(HERE/'audit/read_records.json',read_records)
    dump(HERE/'audit/claims_checked.json',{'baseline_rows':len(history),'baseline_sem_loss_all_zero':True,'baseline_final_mask_map50':ys[-1],
              'baseline_recorded_peak_map50':ys[best],'baseline_peak_epoch':x[best],'two_legacy_baseline_copies_match':True,'formal_population':population['records'],
              'formal_reference_modes':['oracle','practical'],'formal_seed_values':[20261001,20261002,20261003],'metric_cells':metric_rows,
              'selected_originals':10,'redrawn_figures':2,'new_model_calls':0,'new_training_calls':0,'no_source_figures_modified':True})
    contact(rows)
    print(json.dumps({'original_copies':10,'redraws':2,'manifest_rows':len(rows),'trace_previews_checked':previews,'model_calls':0},ensure_ascii=False))

def subprocess_bytes(base,path):
    import subprocess
    return subprocess.check_output(['git','-C',str(REPO),'show',base+':'+path])

def contact(rows):
    from PIL import Image,ImageDraw,ImageFont
    from textwrap import wrap
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    width,height=660,490;sheet=Image.new('RGB',(3*width,4*height+92),'#eef2f6');draw=ImageDraw.Draw(sheet)
    draw.text((24,16),'BuildReasonSeg 中期检查素材总览｜10张历史原图副本 + 2张已有记录重绘',font=font,fill='#133856')
    draw.text((24,50),'仅证明此前遥感建筑研究进展；不是外立面损伤成果。缩略图用于定位，阅读细节请打开原图。',font=small,fill='#804022')
    for i,row in enumerate(rows):
        x=(i%3)*width;y=92+(i//3)*height
        draw.rounded_rectangle((x+9,y+7,x+width-9,y+height-8),radius=10,fill='white',outline='#ccd5df')
        draw.text((x+22,y+18),row['编号']+'  '+row['图片名称'],font=font,fill='#133856')
        with Image.open(HERE/row['副本路径']) as im:
            image=im.convert('RGB');image.thumbnail((width-44,355))
        sheet.paste(image,(x+22+(width-44-image.width)//2,y+61+(355-image.height)//2))
        label='原始副本' if row['编号'] not in ('E11','E12') else '根据已有实验记录重新绘制'
        draw.text((x+22,y+423),label+'｜'+row['图像尺寸'],font=small,fill='#445c70')
        warning={'E02':'GT对照，不是模型预测','E03':'两侧都是GT，不是预测IoU=1','E04':'失败案例：目标身份不匹配','E05':'诊断场，不证明最终正确','E06':'真实空载UI；此帧未推理','E07':'实际候选/Reference，未做GT验证','E08':'实际空间场；颜色不是正确率','E09':'中间候选，未经最终结构验证','E10':'结构SUCCESS，语义NOT_EVALUATED','E11':'验证集曲线；不是测试精度','E12':'完整两种Reference设置；非全面提升'}.get(row['编号'],'历史baseline验证记录')
        draw.text((x+22,y+454),warning,font=small,fill='#9a4428')
    sheet.save(HERE/'contact_sheet.png')

if __name__=='__main__':build()
