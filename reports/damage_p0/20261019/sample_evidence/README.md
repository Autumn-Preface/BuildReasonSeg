# 人工标注抽样证据

每图依次是原图总览、真实标注叠加、原生256×256裁剪及同位置叠加。没有模型预测。
全尺寸原图、原始标注单份保存在Git忽略目录；SHA256、尺寸、坐标/掩码检查见
[sample_checks.json](../../../../evaluation/damage_p0/sample_checks.json)。9张均经执行者可视检查，
结果详见 [dataset_survey](../../../../docs/research/damage_p0/dataset_survey.md)。
小样本、定向取样不能代表全库完整性，也不能据此证明无漏标或无划分泄漏。

署名与使用条件：

- CUBIT图像/标注：Benyun Zhao, Jihan Zhang, Yijun Huang, Xi Chen, Ben M. Chen，2026，
  [原始发布](https://github.com/BenyunZhao/CUBIT-InSeg)，[论文](https://doi.org/10.1016/j.autcon.2026.106980)，
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本项目添加缩略、裁剪与彩色GT叠加，未改原始GT。
- DeepCrack图像/标注：Yahui Liu, Jian Yao, Xiaohu Lu, Renping Xie, Li Li，2019，
  [原始发布/限制](https://github.com/yhlleo/DeepCrack)，[论文](https://doi.org/10.1016/j.neucom.2019.01.036)。
  仅用于非商业研究/教育诊断；此处为带标注叠加的小型研究图，原始数据版权与限制继续适用。
- CiF图像/标注：Nicola Farronato, Niccolo Avogaro, Thomas Frick, Mattia Rigotti, Rizwan Ullah Khan,
  Michele Magno, Konrad Schindler, Cristiano Malossi, Florian Scheidegger，2026，
  [数据](https://huggingface.co/datasets/ibm-research/cif-dataset)，[论文](https://arxiv.org/abs/2605.18413)，
  CDLA Permissive 2.0（以官方数据卡为准）。本项目添加裁剪/GT叠加，不改变标签。

证据文件：

- [CUBIT裂缝1](cubit_CR1_1.png)、[裂缝2](cubit_CR1_10.png)、[剥落1](cubit_SP0_1.png)、[剥落2](cubit_SP0_10.png)。
- [DeepCrack1](deepcrack_11111.png)、[2](deepcrack_11112.png)、[3](deepcrack_11113.png)。
- [CiF1](cif_IMG_8896.png)、[CiF2](cif_IMG_0345.png)。
