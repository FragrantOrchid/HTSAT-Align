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
from src.layer.ScalarFilayer import ScalarFiLMLayer
from torchlibrosa.augmentation import SpecAugmentation
class HTSAT(pl.LightningModule):
    def __init__(self, class_num: int, entropy_film: bool, vowel_embed: bool, sound_length: int):
        super().__init__()
        self.class_num = class_num
        self.entropy_film = entropy_film
        self.vowel_embed = vowel_embed
        self.sound_length = sound_length
        self.spec_aug = SpecAugmentation(time_drop_width=64,time_stripes_num=sound_length*2//5,freq_drop_width=8,freq_stripes_num=2)
        if self.vowel_embed:
            # B,C,Width,Height
            # Input: B,1,sound_length*160,16
            # Output: B,1,sound_length*160,64
            self.vowel_padding = nn.Linear(in_features=10,out_features=64)
        # B,C,H,W()
        # Input: B,2 if self.vowel_embed else 1,64,sound_length*160
        # Output: B,96,16,sound_length*160
        self.patch_embed = nn.Conv2d(in_channels=2 if self.vowel_embed else 1,out_channels=96,kernel_size=(4,1),stride=(4,1),padding=0)
        # all use B,H,W,C 
        # Input: B,16,sound_length*160,96
        # Output: B,1,sound_length*10,96*16
        self.swins = nn.ModuleList([
            SwinTransformerBlockV2(dim=96,num_heads=4,window_size=[8,8],shift_size=[4,4]),
            PatchMergingV2(dim=96),
            SwinTransformerBlockV2(dim=96*2,num_heads=8,window_size=[8,8],shift_size=[4,4]),
            PatchMergingV2(dim=96*2),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[8,4],shift_size=[4,2]),
            PatchMergingV2(dim=96*4),
            SwinTransformerBlockV2(dim=96*8,num_heads=32,window_size=[8,2],shift_size=[4,1]),
            PatchMergingV2(dim=96*8)
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
        if self.entropy_film:
            self.film = ScalarFiLMLayer(num_channel=class_num,hidden_dim=96*4)

    def forward(self,x,entropy,vowel):
        # 扩展元音表
        if self.vowel_embed:
            vowel = vowel.unsqueeze(1)
            vowel = self.vowel_padding(vowel)
            vowel = vowel.permute(0,1,3,2) # B,1,64,sound_length*160
            x = (x-x.mean()) / x.std()
            vowel = (vowel - vowel.mean()) / vowel.std()
            x = torch.cat([vowel,x],dim=1)
        # x: B,C,H,W
        patch_tokens = self.patch_embed(x) # B,C,H,W
        patch_tokens = patch_tokens.permute(0, 2, 3, 1) # BHWC
        for model in self.swins:
            patch_tokens = model(patch_tokens) # B,H/(P*8),W(P*8),C*8
            # self.print(f"Swin step {patch_tokens.shape}")
        latent_tokens = patch_tokens.permute(0,3,1,2) # B,C*8,H/(P*8),W(P*8)
        event = self.conv(latent_tokens) # B,C(num_class),H,W (H=1,W=50)
        event= event.squeeze(dim=2) # B,C(num_class),W(50)
        # 熵偏置
        if self.entropy_film:
            x = event.permute(0,2,1)
            x = self.film(x,entropy)
            event = x.permute(0,2,1)
        
        
        target = self.avg(event) # B,C,1
        target = target.squeeze(-1) # B,C(C = num_class)
        return torch.sigmoid(target)



    def training_step(self, batch, batch_idx):
        log_mel = self.spec_aug(batch["log_mel"])
        target = batch["target"]
        entropy = batch["entropy"] if self.entropy_film else None
        vowel = batch["vowel"] if self.vowel_embed else None

        y = self(log_mel,entropy,vowel)
        loss = torch.nn.functional.binary_cross_entropy(y,target.float())
        # 累积日志
        # self.outputs["y"] = y if self.outputs["y"] is None else torch.cat((self.outputs["y"],y),dim=0)
        # self.outputs["target"] = target if self.outputs["target"] is None else torch.cat((self.outputs["target"],target),dim=0)
        return loss

    def validation_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        target = batch["target"]
        entropy = batch["entropy"] if self.entropy_film else None
        vowel = batch["vowel"] if self.vowel_embed else None

        y = self(log_mel,entropy,vowel)
        # 损失函数
        loss = torch.nn.functional.binary_cross_entropy(y,target.float())
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
        """
    def setup(self, stage):
        if stage == "fit" and self.example_input_array is None:
            # 获取验证集的第一个 batch
            val_dataloader = self.trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))
            log_mel = batch["log_mel"]
            target = batch["target"]
            entropy = batch["entropy"]
            vowel = batch["vowel"]
            # 假设输入是 batch 的第一个元素（根据你的数据格式调整）
            self.example_input_array = (log_mel,entropy,vowel)
        """
    def on_validation_epoch_start(self):
        self.outputs = {
            "y": None,
            "target": None
        }
    def on_validation_epoch_end(self):
        y = self.outputs["y"]
        target = self.outputs["target"]  
        loss = torch.nn.functional.binary_cross_entropy(y, target)
        # numpy
        y = y.detach().cpu().numpy()
        target = target.detach().cpu().numpy().astype(int)
        # calculate
        acc = metrics.accuracy_score(y_true=np.argmax(target,1),y_pred=np.argmax(y,1))
        confusion_matrix = metrics.confusion_matrix(y_true=np.argmax(target,1),y_pred=np.argmax(y,1))
        
        average_precision_scores = [
            metrics.average_precision_score(target[:, k], y[:, k], average=None)
            for k in range(self.class_num)
        ]
        roc_auc_scores = [
            metrics.roc_auc_score(target[:, k], y[:, k], average=None)
            for k in range(self.class_num)
        ]
        self.log("val_simples",y.shape[0])
        self.log("val_loss",loss)
        self.log("val_acc",acc)
        self.log("val_mAP",np.mean(average_precision_scores))
        logging.getLogger("lightning.pytorch").info(
            f'Epoch{self.current_epoch:03d}\tvalidation_step\n{confusion_matrix}'
        )
        for k in range(self.class_num):
            logging.getLogger("lightning.pytorch").info(
                f'class_index:{k:03d}\tAP:{average_precision_scores[k]:.4f}\tauc:{roc_auc_scores[k]:.4f}'
            )
    