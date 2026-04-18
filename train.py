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
    batch_size=600//args.sound_length,
    entropy_film=args.entropy_film,
    vowel_embed=args.vowel_embed
)

# 2. 初始化模型
model = HTSAT(
    class_num=args.class_num,
    sound_length=args.sound_length,
    entropy_film=args.entropy_film,
    vowel_embed=args.vowel_embed
)

# 3. 配置训练器
checkpoint_callback = ModelCheckpoint(
    monitor="val_mAP",
    dirpath=os.path.join(args.export_path,"checkpoints"),
    filename="htsat-{epoch:03d}-{val_mAP:.4f}",
    save_top_k=5,
    mode="max"
)
refresh_rate=data_module.train_dataloader().__len__()//20
print(f"Set refresh rate as {refresh_rate}")
trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=400,
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
# 4. 开始训练
trainer.fit(model, datamodule=data_module, ckpt_path="/home/u220110626/HLHTSAT/export/[2026-04-15-23:15:00]/checkpoints/htsat-epoch=108-val_mAP=0.0650.ckpt")
# trainer.validate(model,datamodule=data_module, ckpt_path="export/[2026-04-10-11:18:43]/checkpoints/htsat-epoch=34-val_mAP=0.94.ckpt")