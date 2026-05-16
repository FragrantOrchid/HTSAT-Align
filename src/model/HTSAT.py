import pytorch_lightning as pl
from torch import nn
import torch
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformerBlockV2
import logging
import numpy as np
from sklearn import metrics
from torchlibrosa.augmentation import SpecAugmentation
from util.GaussianSpecAugment import GaussianSpecAugment
import torch.nn.functional as F
import math
import torchaudio.transforms as T
from src.layer.Permute import Permute
class HTSAT(pl.LightningModule):
    def __init__(self, class_num: int, sound_length: int):
        super().__init__()
        self.strict_loading = False
        self.class_num = class_num
        self.sound_length = sound_length
        # self.spec_aug = SpecAugmentation(time_drop_width=16,time_stripes_num=1,freq_drop_width=8,freq_stripes_num=2) # 时间掩码向上取正值
        # self.spec_aug = T.SpecAugment(
        #     n_time_masks=0,
        #     time_mask_param=self.sound_length*16, # 10%
        #     n_freq_masks=2,
        #     freq_mask_param=8,
        #     p=0.5
        # )
        self.spec_aug = GaussianSpecAugment(
            patch_size=(8,16),
            mask_ratio=0.25,
            cluster_strength=(0.35,0.50)
        )
        # B,C,H,W() -> B,H,W,C
        # Input: B,2 if self.vowel_embed else 1,64,sound_length*160
        # Output: B,96,16,sound_length*160
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=(8,1),stride=(8,1),padding=0),
            Permute(0,2,3,1)
        )
        
        
        # B,H,W,C
        self.swins_transformer = nn.Sequential(
            SwinTransformerBlockV2(dim=96,num_heads=4,window_size=[7,7],shift_size=[0,0]),
            SwinTransformerBlockV2(dim=96,num_heads=4,window_size=[7,7],shift_size=[3,3]),
            PatchMergingV2(dim=96),
            SwinTransformerBlockV2(dim=96*2,num_heads=8,window_size=[7,7],shift_size=[0,0]),
            SwinTransformerBlockV2(dim=96*2,num_heads=8,window_size=[7,7],shift_size=[3,3]),
            PatchMergingV2(dim=96*2),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[0,0]),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[3,3]),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[0,0]),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[3,3]),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[0,0]),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[3,3]),
            PatchMergingV2(dim=96*4),
            SwinTransformerBlockV2(dim=96*8,num_heads=32,window_size=[7,7],shift_size=[0,0]),
            SwinTransformerBlockV2(dim=96*8,num_heads=32,window_size=[7,7],shift_size=[3,3]),
            PatchMergingV2(dim=96*8)
        )
        self.linear = nn.Linear(
            in_features=96*16,
            out_features=self.class_num
        )
        
        self.phoneme_pool = nn.AdaptiveMaxPool1d(1)
        

    def forward(self,x):
        # x: B,C,H,W
        patch_tokens = self.patch_embed(x) # B,C,H,W
        # print(f"patch_tokens shape {patch_tokens.shape}")
        swin_output = self.swins_transformer(patch_tokens)

        patch_tokens = swin_output.squeeze(1) # B, W , 96*16
        logit = self.linear(patch_tokens) # B, W, C
        return logit
 


    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            print(f"\nlog_mel shape {batch['log_mel'].shape}")
            print(f"\ntarget shape {batch['target'].shape}")
        log_mel = self.spec_aug(batch["log_mel"])
        target = batch["target"]
        logit = self(log_mel)
        loss = F.binary_cross_entropy_with_logits(logit,target)
        return loss

    def validation_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        target = batch["target"]
        logit = self(log_mel)
        loss = F.binary_cross_entropy_with_logits(logit,target)
        # 累积日志
        self.outputs["logit"] = logit if self.outputs["logit"] is None else torch.cat((self.outputs["logit"],logit),dim=0)
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

    def on_validation_epoch_start(self):
        self.outputs = {
            "logit" : None,
            "target" : None
        }
    def on_validation_epoch_end(self):
        logit = self.outputs["logit"]
        target = self.outputs["target"]

        loss = F.binary_cross_entropy_with_logits(logit, target)

        # numpy
        y = torch.sigmoid(logit).flatten(0,1).float().detach().cpu().numpy()
        target = target.flatten(0,1).float().detach().cpu().numpy().astype(int)

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
        # self.log("val_simples",y.shape[0])
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)
        self.log("val_mAP", np.mean(average_precision_scores), sync_dist=True)
        logging.getLogger("lightning.pytorch").info(
            f'Epoch{self.current_epoch:03d}\tvalidation_step\n{confusion_matrix}'
        )
        for k in range(self.class_num):
            logging.getLogger("lightning.pytorch").info(
                f'class_index:{k:03d}\tAP:{average_precision_scores[k]:.4f}\tauc:{roc_auc_scores[k]:.4f}'
            )
        """
        validate_map = {
            "9->9" : [],
            "9->10" : [],
            "10->10" : []
        }
        y_true=np.argmax(target,1)
        y_pred=np.argmax(y,1)
        for index in range(y.shape[0]):
            if y_true[index] == 9 and y_pred[index] == 9:
                validate_map["9->9"].append(index)
            if y_true[index] == 9 and y_pred[index] == 10:
                validate_map["9->10"].append(index)
            if y_true[index] == 10 and y_pred[index] == 10:
                validate_map["10->10"].append(index)
                
        logging.getLogger("lightning.pytorch").info(
            f'{validate_map}'
        )"""
    