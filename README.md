# HTAST-Align
## 最终数据
- ACC=0.9846(Without Dropout)
- ACC=0.9854(With Dropout)
## 不同分支作用
- LibriSpeechAlignmentsPretrain 使用LibriSpeech-ALignment数据集进行预训练
- SpeechCommandV2-PartialFreezing 使用SpeechCommand V2数据集上，冻结部分参数进行训练
- SpeechCommandV2-GlobalTuning 使用SpeechCommand V2数据集上，完全解冻进行微调，得到对中结果
## Release文件作用
- CheckPoint 演示代码中测试功能使用的检查点
- Datafile 训练使用数据的索引
- Export 训练中的全部日志