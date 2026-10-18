# Damage-P0 模型与开源资源调查

核查2026-10-19。具体资源见 [model_registry.csv](../../../evaluation/damage_p0/model_registry.csv)、
[源码pin](../../../evaluation/damage_p0/resource_pins.json)、[权重HEAD核查](../../../evaluation/damage_p0/weight_access.json)。
代码存在、权重入口存在、下载成功、任务适配成功是四种证据，本阶段没有下载任何新模型权重。
“未知”不等于禁止或开放；缺条款的权重不得在P1前当作已准入资源。

|具体候选|实现/权重及许可证分别核查|依赖与输入|自动性、训练/评价入口及适配成本|
|---|---|---|---|
|U-Net（第三方milesial PyTorch实现）|[原论文/原作者Caffe](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)；[PyTorch第三方](https://github.com/milesial/Pytorch-UNet) GPL-3.0。第三方Carvana预训练链接存在，权重专门条款未核，不使用；原作者权重也不是损伤权重|README Python≥3.6/torch≥1.13，非当前全兼容声明；建议P1隔离现代torch，原生512²patch|RGB全自动；train.py/predict.py；无独立完整损伤评估器，须P1统一评估。二输出sigmoid等适配须登记；从头训练对照，低适配成本|
|SegFormer-B0|[官方NVlabs](https://github.com/NVlabs/SegFormer)，NVIDIA Source Code License、非商业限制；官方ImageNet encoder/分割权重链接公开。[NVIDIA HF B0卡](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)license=other，卡由HF团队写，**不能当权重已获Apache许可**|官方mmseg0.13/torch1.7.1/cu10.1、mmcv1.2.7/timm0.3.2；配置512²；这些老轮子不适用于当前GPU原样安装|RGB自动；tools/train.py/test.py。P1可提议现代可验证实现迁移，必须记录差异与权重条件，不冒称原版零修改复现；中适配成本|
|SAM2.1 Hiera-tiny|[Meta官方](https://github.com/facebookresearch/sam2)，代码/训练/权重Apache2，另有cc_torch BSD组件；tiny公开HEAD200，156008466 bytes|Python≥3.10、torch≥2.5.1、torchvision≥0.20.1；通常1024²编码；可选CUDA编译；官方Windows推荐WSL|支持点/框/mask，**原生不支持文本**；AMG可自动给候选mask，但不自动辨认哪个是crack/spalling。ImagePredictor、AMG、training/README，官方训练入口；自动损伤系统还需定位/分类，额外成本必须计入|
|SAM3 image（不把3.1视频更新混入同一个版本）|[Meta官方](https://github.com/facebookresearch/sam3)，代码与权重适用自定义SAM License；HF访问需接受条件，匿名HEAD401；未请求账号/下载。README已发布3.1，另记版本不自动替换候选|Python≥3.12、torch≥2.7、CUDA≥12.6；848M参数；图像/文本/视觉接口；输入processor配置需pin；无本机显存实测|真实文本API Sam3Processor.set_text_prompt；也有视觉提示。需固定词表/输入条件，归GroupC；未验证“crack/spalling”域效果，不能列为A无额外条件基线。README_TRAIN/评估代码存在声明，未执行|
|YOLO11s-seg|[Ultralytics官方模型表](https://docs.ultralytics.com/models/yolo11/)，[分割入口](https://docs.ultralytics.com/tasks/segment/)；代码AGPL3/可选企业条款；代码/官方模型使用以[官方许可说明](https://www.ultralytics.com/license)分别确认。权重HEAD200、20669228 bytes，未下载|官方PyPI包Python≥3.8（具体pin另核），现代torch；标准640²；当前proposal环境8.4.164仅作已安装事实，未加载|RGB自动实例分割；model.train/val/predict或yolo segment train/val/predict。语义比较将同类实例取并集；mask AP另报。不能因是YOLO沿用旧RC1模型或旧权重|
|CT-CrackSeg|[作者官方](https://github.com/HqiTao/CT-crackseg)，代码MIT；Crack500/DeepCrack checkpoint链接公开，权重单独条款未明确，不认定自动继承MIT|requirements torch1.10+cu111/torchvision0.11.1、numpy1.21.5；Python版本未明确；训练尺寸以pin配置核验（本阶段未锁定）|RGB裂缝语义；train_crack.py/evaluate_crack.py；改到现代CUDA及双类输出需验证；适配中等，优先作为专用裂缝诊断替补|
|CrackSAM adapter/LoRA|[作者官方](https://github.com/KG-TSI-Civil/CrackSAM) README声明代码MIT；目录存在3种delta checkpoint；原SAM权重Apache2，**delta权重条款未单独核定**；没有下载delta|作者例448²、batch8、ViT-H、adapter32；依赖numpy1.22.4等，Python/torch未在requirements钉死。例配置不是16GiB适用承诺|train.py/test.py，可见SAMed衍生实现。需检查实际forward有无GT点/框或默认内部prompt才决定A/B，不依据名称归组；高显存/迁移成本，不作为P1首轮必跑|
|MixerCSeg（2026）|[作者官方](https://github.com/spiderforest/MixerCSeg)，MIT代码；作者发布四库checkpoint链接，权重独立条款未确认；第三方VMamba内核许可另核|作者Python3.10/torch2.1+cu118、numpy1.23，selective_scan自编译、triton/mmcv；RTX5080原样兼容未验证|自动裂缝；main.py/test.py/eval/evaluate.py；需源码内改路径，适配高。现代库兼容和checkpoint许可解决后，才可作为领域扩展候选|
|EfficientCrackNet|[WACV2025原论文](https://openaccess.thecvf.com/content/WACV2025/papers/Zim_EfficientCrackNet_A_Lightweight_Model_for_Crack_Segmentation_WACV_2025_paper.pdf)可读；本次未确认作者代码/权重，均unknown|依赖、训练入口、实际显存unknown；论文表2参数0.26M、0.483GFLOPs，不等于显存实测|轻量领域参考，暂不列可直接复现P1执行项；不拿第三方同名项目充官方|

## 推荐的有限 P1 路线（建议，不是批准）

首轮三条Group A：**U-Net、SegFormer-B0、YOLO11s-seg**，分别代表局部细节恢复、层次Transformer、
实例到语义端到端路径。资源/实现适配需在P1授权和许可复核后完成。
第四条是 **SAM2.1-tiny提示诊断**，与A分表：同一分割器分别接受人工、自动和Oracle位置。
自动框可复用已训练YOLO11s-seg产生的boxes，避免再训练一个检测器；成本计为整个系统，
其instance masks与SAM细化结果在同一定位条件下对照。该建议不确定最终主干。

若Supervisor希望首轮包含裂缝专用方法，可在许可/兼容核验后以CT-CrackSeg替换或追加，
同时明确其仅评价crack；不默默增加到全部候选。CrackSAM/MixerCSeg保留为第二轮专用方向；
SAM3预留GroupC，当前Python3.11旧环境不满足其最低Python要求，不能原地升级。

公平性限度：U-Net从头训练与ImageNet/COCO预训练不同；结果是“可获得资源条件下系统能力”
比较，不能宣称纯架构优越。架构因果比较另设从头训练或匹配监督的受控轨，预算另审。
