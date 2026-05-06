import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.model.HTSAT import HTSAT
from src.dataset.HTSATdataset import HTSATdataset
import argparse
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar
import os
import logging
import numpy as np
np.set_printoptions(
    threshold=np.inf,
    precision=4,
    linewidth=np.inf
)
torch.set_float32_matmul_precision('medium')
parser = argparse.ArgumentParser(description='程序描述')
parser.add_argument("-train_file", required=True, help="训练文件路径")
parser.add_argument("-val_file", required=True, help="验证文件路径")
parser.add_argument("-label_csv", required=True, help="标签CSV文件路径")
parser.add_argument("-class_num", type=int, required=True, help="类别数量")
parser.add_argument("-sound_length", type=int, required=True, help="声音长度")
parser.add_argument("-export_path", required=True, help="导出路径")
parser.add_argument("-report_name", required=True, help="报告名称")
parser.add_argument("--entropy_film", action="store_true", help="是否使用熵电影")
parser.add_argument("--vowel_embed", action="store_true", help="是否使用元音嵌入")
args = parser.parse_args()
print(args)
# 1. 初始化数据模块
data_module = HTSATdataset(
    train_file=args.train_file,
    val_file=args.val_file,
    label_csv=args.label_csv,
    sound_length=args.sound_length,
    batch_size=600//args.sound_length
)

# 2. 初始化模型
model = HTSAT(
    class_num=args.class_num,
    sound_length=args.sound_length
)

# 3. 配置训练器
checkpoint_callback = ModelCheckpoint(
    monitor="val_mAP",
    dirpath=os.path.join(args.export_path,"checkpoints"),
    filename="htsat-{epoch:03d}-{val_mAP:.4f}",
    save_top_k=5,
    mode="max"
)
refresh_rate=data_module.train_dataloader().__len__()//50
print(f"Set refresh rate as {refresh_rate}")
trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=800,
    callbacks=[
        checkpoint_callback,
        TQDMProgressBar(refresh_rate=refresh_rate)
    ],
    log_every_n_steps=10,
    enable_progress_bar=True,
    default_root_dir=args.export_path,
    logger=[
        CSVLogger(save_dir=args.export_path,name="csv",version=""),
        TensorBoardLogger(save_dir=args.export_path,name=args.report_name,version="",log_graph=True,)
    ],
    
)
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
logging.getLogger("lightning.pytorch").addHandler(
    logging.FileHandler(os.path.join(args.export_path,"lightning.pytorch.log"))
)



# 4. 安全加载检查点，只加载模型权重
def load_part_of_state_dict(model, state_dict, strict=False):
    """
    加载部分模型状态字典，对于不匹配的参数保持随机初始化
    """
    model_state_dict = model.state_dict()
    
    # 创建一个新的状态字典，只包含匹配的参数
    filtered_state_dict = {}
    unmatched_keys = []
    
    for key, param in state_dict.items():
        if key in model_state_dict:
            # 检查参数形状是否匹配
            if param.shape == model_state_dict[key].shape:
                filtered_state_dict[key] = param
            else:
                print(f"跳过不匹配的参数: {key}, "
                      f"检查点形状: {param.shape}, 当前模型形状: {model_state_dict[key].shape}")
                unmatched_keys.append(key)
        elif not strict:
            # 如果不是严格模式，记录未找到的键
            unmatched_keys.append(key)
        else:
            raise KeyError(f"Unexpected key(s) in state_dict: {key}")
    
    # 将过滤后的状态字典加载到模型中
    model.load_state_dict(filtered_state_dict, strict=strict)
    
    # 打印加载结果
    matched_keys = list(filtered_state_dict.keys())
    print(f"成功加载 {len(matched_keys)} 个参数")
    if unmatched_keys:
        print(f"跳过 {len(unmatched_keys)} 个参数: {unmatched_keys}")
    
    return model

checkpoint_path = "/home/u220110626/HLHTSAT/export/[2026-05-01-00:11:29]/checkpoints/htsat-epoch=572-val_mAP=0.9689.ckpt"
checkpoint = torch.load(checkpoint_path)

state_dict = checkpoint.get('state_dict', checkpoint)

# model = load_part_of_state_dict(model, state_dict, strict=False)

    
trainer.fit(
    model,
    datamodule=data_module,
    ckpt_path=None
)


# trainer.validate(
#     model,
#     datamodule=data_module,
#     ckpt_path=None
# )

