# Damage-P0 本机环境核查

实测2026-10-19（Asia/Shanghai）。[audit_environment.ps1](../../../reports/damage_p0/20261019/audit_environment.ps1)
读取系统信息；[probe_python.py](../../../reports/damage_p0/20261019/probe_python.py)仅读已安装包元数据和
一次64×64 CUDA矩阵乘。未安装/升级/卸载依赖，未加载分割模型、未训练、未压力测试。

## 实际检测

|项目|本次事实|证据/含义|
|---|---|---|
|CPU|Intel Core Ultra 9 275HX，24核/24逻辑线程|environment_facts.json / Win32_Processor；不是沿用旧报告|
|内存|33,690,263,552 bytes，约31.38 GiB；当时OS可见空闲约17.11 GiB|Win32_ComputerSystem/OperatingSystem；不是可持续占用承诺|
|GPU|NVIDIA GeForce RTX 5080 Laptop GPU；nvidia-smi 16303 MiB；torch17066033152 bytes；compute capability12.0|驱动与torch口径略异，均约16GiB，不当成完整可用训练显存|
|驱动/运行时|NVIDIA617.42；nvidia-smi显示CUDA支持13.4；PyTorch自带CUDA13.2|驱动上限不等于安装了CUDA Toolkit；PATH未找到nvcc，不能据此断言全盘没装Toolkit|
|Windows|Windows11家庭中文版10.0.26300；WSL输出26300.9711|本机命令；GPU使用WDDM|
|卷|C: size991902896128、审计快照可用154617380864 bytes（约144.0GiB）|项目路径C:\D实际仍在C:，不存在独立D:空间池；最终快照见storage_final.json|
|WSL|3.0.3.0，kernel6.18.54.1-1；默认WSL2；只列docker-desktop、Stopped、version2|wsl_facts.json使用UTF-16正确解码。未启动Docker发行版，未验证WSL内GPU/PyTorch；没有可用Ubuntu研究环境的证据|
|Conda|25.11.0；可列base、proposal、access-risk、jupyter、yolo_sam_env|仅base/proposal运行本次包探针，另外三环境仅登记存在，不冒称可训练；未创建新环境|
|Python/torch|base Python3.13.9，无torch；proposal Python3.11.16、torch2.13.0+cu132、torchvision0.28.0+cu132|python_base.json / python_proposal.json；这些是当前机器实测，不是推荐所有候选采用此版本|
|图像/深度学习依赖|proposal numpy2.4.6、Pillow12.3.0、opencv5.0.0.93、scipy1.17.1、transformers5.17.0、ultralytics8.4.164、SAM-2元数据1.0、PyYAML6.0.3；scikit-image未装|只有元数据，不代表所有包导入/扩展兼容。base另有scikit-image0.25.2；检查脚本使用bundled Python/Pillow/numpy/pypdf|
|GPU轻量检查|cuda_available=true，64² ones@ones元素64，总和262144，与期望一致；bf16_supported=true|小型算子成功；不推论任何模型推理精度、训练稳定性、吞吐或显存峰值|
|Git/网络|PATH Git缺HTTPS helper；bundled Git fetch/push成功；原始站点HTTP核查详source_access.json|未改全局Git配置。只对外部公开地址发GET/HEAD，不保存凭据。dacl403/SAM3权重401仍是资源限制|
|缓存配置|HF_HOME/HUGGINGFACE_HUB_CACHE/TORCH_HOME/PIP_CACHE_DIR/CONDA_PKGS_DIRS本进程未设置|不等于缓存目录不存在；未递归清理默认用户缓存。P0源缓存按本任务目录单份保存|

原始证据：[environment_facts.json](../../../evaluation/damage_p0/environment_facts.json)、
[wsl_facts.json](../../../evaluation/damage_p0/wsl_facts.json)、
[python_base.json](../../../evaluation/damage_p0/python_base.json)、
[python_proposal.json](../../../evaluation/damage_p0/python_proposal.json)。最初PS对WSL输出解码错误，
保留原始输出，并以Python UTF-16版本为解释依据；不是WSL损坏。

## 资源建议（全部是规划估算，未经模型测试）

下表显存区间是工程预算范围，按模型规模、输入像素、激活/优化器状态粗估，不是官方最低要求。
单卡空闲显存受桌面进程影响，P1必须独立环境做≤几步的授权试跑后确定batch，不能预先保证。

|路线|起始输入/微批建议|显存规划估算|Windows/WSL、精度与限制|
|---|---|---|---|
|U-Net|512²，microbatch2，effective batch8|约2–6GiB训练|纯torch原生Windows较可行；AMP FP16/BF16需逐模型验证；512→1024激活大致随像素增约4倍|
|SegFormer-B0|512²，microbatch2，effective8|约3–8GiB训练|现代实现Windows可能可行，原官方旧mmcv/cu10.1不原样安装；WSL2新环境兼容性待测|
|YOLO11s-seg|640²，microbatch2起，effective8（若实现支持）|约4–10GiB训练|Windows官方路径可候选；AMP需本机验证；1024²+全分辨率mask可能显著增加激活|
|SAM2.1 tiny|1024²编码，推理1图；微调预算另审|约3–8GiB图像推理/轻量适配规划，训练峰值unknown|官方推荐Windows用WSL；当前只有Docker发行版，不能宣称ready。无nvcc时可选后处理扩展不一定可编译；不强行装旧CUDA|
|SAM3|1图推理先可行性核查；不建议本机完整训练|848M参数，单FP32参数约3.39GB；Adam全参持久状态粗计约13.57GB，**尚未含激活/工作区**，16GiB全参训练不宜规划|需新Python≥3.12环境、许可访问；无本机模型实测；推理显存unknown，不能由参数字节推出能运行|
|CT-CrackSeg/MixerCSeg/CrackSAM|先固定代码/依赖/权重条件；不预许batch|实际峰值unknown；ViT-H和自定义CUDA迁移有额外风险|官方旧torch/cu版本、selective_scan/triton可能不支持本机架构，P1需要单独兼容性工程。不得破坏旧环境来迁就|

先行存储规划：新环境/轮子5–15GiB，代码/小checkpoint5–10GiB，数据原始+派生预计30–70GiB，
运行/预测5–15GiB，合计约45–110GiB（估算，非下载授权）。现有144GiB余量不足以
同时复制CiF约123GB全资源和其派生缓存。优先按CUBIT具体版本测量，禁止模型/数据多套缓存。
建议保持50GiB系统余量、Damage活动资产软预算60GiB；超出先提出具体新预算给Supervisor。
本阶段未实施系统设置、驱动、WSL安装和环境重构。
