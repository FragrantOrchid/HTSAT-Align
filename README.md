# HTAST-Align
## 最终数据 (均在 Google Speech Commands v2 数据集上测试)
- ACC=0.9846(Without Dropout)
- ACC=0.9854(With Dropout)

**由于Paperswithcode关闭，我并不能确定当前是否有比本项目在Google Speech Command V2上准确率更高的模型，可以确定的是本项目准确率比Paperswithcode关闭前的SOTA高**
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
## 如何使用
### 准备环境
```conda env create -f environment.yml -n <your_new_eventment_name>```
### 准备数据
- 训练数据索引：https://github.com/FragrantOrchid/HTSAT-Align/releases/tag/Datafile
- 预训练数据本体：https://huggingface.co/datasets/gilkeyio/librispeech-alignments
- Speech Command V2数据：https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- 检查点：https://github.com/FragrantOrchid/HTSAT-Align/releases/tag/Checkpoint

最终文件结构如下文所示：
```
HTSAT-align
├── ckpt
│   ├── htast-align-after-global-tuning.ckpt
│   ├── htast-align-after-partial-freezing.ckpt
│   ├── htast-align-after-pretraining.ckpt
│   ├── htast-align-after-global-tuning-dropout.ckpt
│   ├── htast-align-after-partial-freezing-dropout.ckpt
│   └── htast-align-after-pretraining-dropout.ckpt
├── data
│   ├── LibriSpeech-Alignment-BIES
│   │   ├── class.csv
│   │   ├── dev-clean.csv
│   │   ├── test-clean.csv
│   │   ├── train-clean-100.csv
│   │   ├── train-clean-360.csv
│   │   └── train-other-500.csv
│   └── speechcommand-v2
│        ├── class.csv
│        ├── eval.json
│        ├── test.json
│        └── train.json
......
```
### 测试
```git checkout SpeechCommandV2-GlobalTuning && bash run.sh```
### 从头开始训练
将bash.sh中形如
```
# mode="train"
# ckpt_path="*.ckpt" 
mode="test"
ckpt_path="./ckpt/*.ckpt"
```
修改为形如
```
mode="train"
ckpt_path="*.ckpt" 
# mode="test"
# ckpt_path="./ckpt/*.ckpt"
```
的形式