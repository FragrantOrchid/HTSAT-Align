# HTAST-Align
## 最终数据 (均在 Google Speech Commands v2 数据集上测试)
- ACC=0.9846(Without Dropout)
- ACC=0.9854(With Dropout)
## 不同分支作用
- LibriSpeechAlignmentsPretrain 使用LibriSpeech-ALignment数据集进行预训练
- SpeechCommandV2-PartialFreezing 使用SpeechCommand V2数据集上，冻结部分参数进行训练
- SpeechCommandV2-GlobalTuning 使用SpeechCommand V2数据集上，完全解冻进行微调，得到对中结果
## Release文件作用
### CheckPoint
演示代码中测试功能使用的检查点，来自pytorch-lightning框架自动保存，其原始位置可通过在Export中查看
### Datafile
训练使用数据的索引，文件名均与原始开源数据集中命名相同
### Export
训练中的全部日志，可从此查看模型训练所使用的时间，预训练阶段由于集群限制分割成多个部分。