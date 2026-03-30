import pytorch_lightning as pl
from src.dataset.HTSATdataset import HTSATdataset
from torch import nn
import torch
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformerBlockV2
from util.Stat import Stat
import logging
import numpy as np
import gc
import psutil
import os
from sklearn import metrics
class HTSAT(pl.LightningModule):
    def __init__(self,class_num: int):
        super().__init__()
        self.class_num = class_num
        self.outputs = {
            "y": None,
            "target": None
        }
        # B,C,H,W()
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=96,
            kernel_size=(4,1),
            stride=(4,1),
            padding=0
        )
        # all use B,H,W,C 
        self.swins = nn.ModuleList([
            SwinTransformerBlockV2( # B,800,16,96
                dim=96,
                num_heads=4,
                window_size=[8,8],
                shift_size=[4,4]
            ),
            PatchMergingV2( # B,800,16,96 -> B,400,8,96*2
                dim=96
            ),
            SwinTransformerBlockV2( # B,400,8,96*2
                dim=96*2,
                num_heads=8,
                window_size=[8,8],
                shift_size=[4,4]
            ),
            PatchMergingV2( # B,400,8,96*2 -> B,200,4,96*4
                dim=96*2
            ),
            SwinTransformerBlockV2( # B,200,4,96*4
                dim=96*4,
                num_heads=16,
                window_size=[8,4],
                shift_size=[4,2]
            ),
            PatchMergingV2( # B,200,4,96*4 -> B,100,2,96*8
                dim=96*4
            ),
            SwinTransformerBlockV2( # B,100,2,96*8
                dim=96*8,
                num_heads=32,
                window_size=[8,2],
                shift_size=[4,1]
            ),
            PatchMergingV2( # B,100,2,96*8 -> B,50,1,96*16
                dim=96*8
            ),
        ])
        # BCHW
        self.conv = nn.Conv2d(
            in_channels=96*16,
            out_channels=self.class_num,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.avg = nn.AdaptiveAvgPool1d(1) # B,C,sequence_length -> B,C,1

    def forward(self,x):
        # x: B,C,H,W
        # self.print(f"Before patch {x.shape}")
        patch_tokens = self.patch_embed(x) # B,C,H,W
        # self.print(f"After patch {patch_tokens.shape}")
        patch_tokens = patch_tokens.permute(0, 2, 3, 1) # BHWC
        for model in self.swins:
            patch_tokens = model(patch_tokens) # B,H/(P*8),W(P*8),C*8
            # self.print(f"Swin step {patch_tokens.shape}")
        latent_tokens = patch_tokens.permute(0,3,1,2) # B,C*8,H/(P*8),W(P*8)
        event = self.conv(latent_tokens) # B,C(num_class),H,W (H=1,W=50)
        event= event.squeeze(dim=2) # B,C(num_class),W(50)
        target = self.avg(event) # B,C,1
        target = target.squeeze(-1) # B,C(C = num_class)
        return target



    def training_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        target = batch["target"]

        y = self(log_mel)
        # 损失函数自带sigmoid归(0,1)，输出可视化是需要自行sigmoid
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y,target.float())
        # 累积日志
        self.outputs["y"] = y if self.outputs["y"] is None else torch.cat((self.outputs["y"],y),dim=0)
        self.outputs["target"] = target if self.outputs["target"] is None else torch.cat((self.outputs["target"],target),dim=0)
        # logging.getLogger("lightning.pytorch").info(
        #         f'training_step outputs add to {self.outputs["y"].shape}'
        #     )
        if self.outputs["y"].shape == 1600:
            self.on_train_epoch_end()
        return loss

    def validation_step(self, batch, batch_idx):
        # 计算准确率
        # preds = torch.argmax(y, dim=1)
        # true_labels = torch.argmax(target, dim=1)
        # acc = (preds == true_labels).float().mean()
        log_mel = batch["log_mel"]
        target = batch["target"]

        y = self(log_mel)
        # 损失函数自带sigmoid归(0,1)，输出可视化是需要自行sigmoid
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y,target.float())
        # 累积日志
        self.outputs["y"] = y if self.outputs["y"] is None else torch.cat((self.outputs["y"],y),dim=0)
        self.outputs["target"] = target if self.outputs["target"] is None else torch.cat((self.outputs["target"],target),dim=0)
        
        return loss

    def configure_optimizers(self):
        # 1. 优化器配置 (AdamW)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4,  # 初始学习率（建议从1e-4开始，根据任务调整）
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.05,  # 适用于Transformer类模型的权重衰减
        )
    
        # 2. 学习率调度策略（Warmup + 余弦退火）
        def lr_scheduler_fn(epoch):
            # Warmup阶段：前5个epoch线性增加学习率
            if epoch < 5:
                return float(epoch + 1) / 5.0
            # 余弦退火阶段（周期为20个epoch）
            else:
                import math
                progress = (epoch - 5) / 20.0
                return 0.5 * (1.0 + math.cos(math.pi * progress))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_scheduler_fn
        )
    
        # 3. 返回优化器和调度器
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 按epoch调整学习率
                "frequency": 1,
            }
        }
        
    # 初始化样例输入  
    def setup(self, stage):
        if stage == "fit" and self.example_input_array is None:
            # 获取验证集的第一个 batch
            val_dataloader = self.trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))
            log_mel = batch["log_mel"]
            target = batch["target"]
            # 假设输入是 batch 的第一个元素（根据你的数据格式调整）
            self.example_input_array = log_mel  # 取第一个样本，并增加 batch 维度
            
    def on_validation_epoch_start(self):
        self.outputs = {
            "y": None,
            "target": None
        }
    def on_validation_epoch_end(self):
        y = self.outputs["y"]
        target = self.outputs["target"]
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(y, target)
        # numpy
        y = torch.sigmoid(y).detach().cpu().numpy()
        target = target.detach().cpu().numpy().astype(int)
        # calculate
        acc = metrics.accuracy_score(np.argmax(y,1),np.argmax(target,1))
        average_precision_scores = [
            metrics.average_precision_score(target[:, k], y[:, k], average=None)
            for k in range(self.class_num)
        ]
        roc_auc_scores = [
            metrics.roc_auc_score(target[:, k], y[:, k], average=None)
            for k in range(self.class_num)
        ]
        self.log("val_simples",y.shape[0])
        # self.log("val_loss",loss)
        self.log("val_acc",acc)
        self.log("val_mAP",np.mean(average_precision_scores))
        logging.getLogger("lightning.pytorch").info(
            f'Epoch{self.current_epoch:03d}\tvalidation_step\t'
        )
        for k in range(self.class_num):
            logging.getLogger("lightning.pytorch").info(
                f'class_index:{k:03d}\tAP:{average_precision_scores[k]:.4f}\tauc:{roc_auc_scores[k]:.4f}'
            )
        self.outputs = {
            "y": None,
            "target": None
        }
    