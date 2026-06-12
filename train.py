import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.model.HTSAT import HTSAT
from src.dataset.HTSATdataset import HTSATdataset
import argparse
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar, BatchSizeFinder
import os
import logging
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from util.LoadState import load_part_of_state_dict
np.set_printoptions(
    threshold=np.inf,
    precision=4,
    linewidth=np.inf
)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(
    "ignore",
    message="No positive class found in y_true, recall is set to one for all thresholds.",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore", 
    message="At least one mel filterbank has all zero values. The value for `n_mels` .*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="This DataLoader will create .* worker processes in total",
    category=UserWarning
)
torch.set_float32_matmul_precision('medium')

def dev_null_to_none(value):
    if value == "/dev/null":
        return None
    return value

parser = argparse.ArgumentParser(description='程序描述')
parser.add_argument("-train_file", required=True, help="训练文件路径")
parser.add_argument("-val_file", required=True, help="验证文件路径")
parser.add_argument("-test_file", required=True, help="测试文件路径")
parser.add_argument("-label_csv", required=True, help="标签CSV文件路径")
parser.add_argument("-class_num", type=int, required=True, help="类别数量")
parser.add_argument("-sound_length", type=int, required=True, help="声音长度")
parser.add_argument("-export_path", required=True, help="导出路径")
parser.add_argument("-report_name", required=True, help="报告名称")
parser.add_argument("-mode", choices=["train", "val", "test"], required=True, help="运行模式: train(训练), val(验证), test(测试)")
parser.add_argument("-ckpt_path", type=dev_null_to_none, default=None, help="检查点路径(用于验证或测试)")  # 可选参数
args = parser.parse_args()
print(args)

data_module = HTSATdataset(
    train_file=args.train_file,
    val_file=args.val_file,
    test_file=args.test_file,
    label_csv=args.label_csv,
    sound_length=args.sound_length,
    batch_size=128
)

model = HTSAT(
    class_num=args.class_num,
    sound_length=args.sound_length
)

trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=100,
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(args.export_path,"checkpoints"),
            filename="htsat-{epoch:03d}-{val_loss:.4f}",
            save_top_k=5,
            mode="min"
        ),
        ModelCheckpoint(
            monitor="val_acc",
            dirpath=os.path.join(args.export_path,"checkpoints"),
            filename="htsat-{epoch:03d}-{val_acc:.4f}",
            save_top_k=5,
            mode="max"
        ),
        # BatchSizeFinder(
        #     mode="binsearch",      # "binsearch"二分法，"power"指数法
        #     init_val=32,           # 起始测试值
        #     batch_arg_name="batch_size" # 模型或DataModule中对应的参数名
        # ),
        # TQDMProgressBar(refresh_rate=data_module.train_dataloader().__len__()//10+1)
        TQDMProgressBar(refresh_rate=32)
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



# 冻结部分参数
# for param in model.patch_embed.parameters():
#     param.requires_grad = False
# for param in model.swins_transformer.parameters():
#     param.requires_grad = False
# for param in model.linear.parameters():
#     param.requires_grad = False
if args.mode == "train":
    print(f"load from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model = load_part_of_state_dict(model, state_dict, strict=False)
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=None
    )
elif args.mode == "val":
    trainer.validate(
        model,
        datamodule=data_module,
        ckpt_path=args.ckpt_path
    )
elif args.mode == "test":
    trainer.test(
        model,
        datamodule=data_module,
        ckpt_path=args.ckpt_path
    )

