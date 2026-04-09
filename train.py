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
parser = argparse.ArgumentParser(description='程序描述')
parser.add_argument("-train_file")
parser.add_argument("-val_file")
parser.add_argument("-label_csv")
parser.add_argument("-class_num",type=int)
parser.add_argument("-sound_length",type=int)
parser.add_argument("-export_path")
args = parser.parse_args()
print(args)
# 1. 初始化数据模块
data_module = HTSATdataset(
    train_file=args.train_file,
    val_file=args.val_file,
    label_csv=args.label_csv,
    sound_length=int(args.sound_length),
    batch_size=600
)

# 2. 初始化模型
model = HTSAT(class_num=int(args.class_num),entropy_film=True)

# 3. 配置训练器
checkpoint_callback = ModelCheckpoint(
    monitor="val_mAP",
    dirpath=os.path.join(args.export_path,"checkpoints"),
    filename="htsat-{epoch:02d}-{val_mAP:.2f}",
    save_top_k=5,
    mode="max"
)


trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    # precision="16-mixed",  # 关键：启用混合精度
    max_epochs=50,
    callbacks=[
        checkpoint_callback,
        TQDMProgressBar(refresh_rate=200)
    ],
    log_every_n_steps=10,
    enable_progress_bar=True,
    default_root_dir=args.export_path,
    logger=[
        CSVLogger(save_dir=args.export_path,name="csv",version=""),
        TensorBoardLogger(save_dir=args.export_path,name="speedcommandv2 常规 无熵偏置",version="",log_graph=True,)
    ],
    
)
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
logging.getLogger("lightning.pytorch").addHandler(
    logging.FileHandler(os.path.join(args.export_path,"lightning.pytorch.log"))
)
# 4. 开始训练
trainer.fit(model, datamodule=data_module)

# trainer.validate(model,datamodule=data_module, ckpt_path="./export/[2026-03-31-00:19:41]/checkpoints/htsat-epoch=45-val_mAP=0.93.ckpt")