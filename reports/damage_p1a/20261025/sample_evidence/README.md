# 原始标注与重复证据（没有模型预测）

数据：Benyun Zhao, Jihan Zhang, Yijun Huang, Xi Chen, Ben M. Chen, CUBIT-InSeg, Automation in Construction 188 (2026), 106980，[DOI](https://doi.org/10.1016/j.autcon.2026.106980)。来源为[作者仓库](https://github.com/BenyunZhao/CUBIT-InSeg)直接下载入口，CC BY 4.0。叠加、缩略、裁剪及文字由本任务生成，原始图片与标注未修改。

- [confirmed_train_val_duplicate.png](confirmed_train_val_duplicate.png)：实际 train SP1_813 / val SP0_818，同文件及解码 SHA。
- [CR1_1_raw_gt.png](CR1_1_raw_gt.png)、[CR1_10_raw_gt.png](CR1_10_raw_gt.png)：明确只读使用既有 P0 原图，本轮标签来自官方 ZIP；红色仅代表原始 ID 0。
- [SP1_813_raw_gt.png](SP1_813_raw_gt.png)、[SP1_1091_raw_gt.png](SP1_1091_raw_gt.png)：本轮范围取得的官方训练原图；青色仅代表原始 ID 1。

数字 ID 的语义仍未权威确认。图片不作为损伤模型预测、泛化性能、数据集全库质量或全体图像哈希已完成的证据。见上级 data_admission.md、sample_alignment.json 和 duplicate_sample_audit.json。
