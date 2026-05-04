import pytorch_lightning as pl
from torch import nn
import torch
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformerBlockV2
import logging
import numpy as np
from sklearn import metrics
from torchlibrosa.augmentation import SpecAugmentation
import torch.nn.functional as F
class HTSAT(pl.LightningModule):
    def __init__(self, class_num: int, sound_length: int):
        super().__init__()
        self.strict_loading = False
        self.class_num = class_num
        self.sound_length = sound_length
        self.spec_aug = SpecAugmentation(time_drop_width=64,time_stripes_num=sound_length*2//5,freq_drop_width=8,freq_stripes_num=2)

        # B,C,H,W()
        # Input: B,2 if self.vowel_embed else 1,64,sound_length*160
        # Output: B,96,16,sound_length*160
        self.patch_embed = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=(4,1),stride=(4,1),padding=0)
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
        
        self.linear = nn.Linear(
            in_features=96*16,
            out_features=39
        )
        
        self.phoneme_pool = nn.AdaptiveMaxPool1d(1)
        

    def forward(self,x):
        # x: B,C,H,W
        patch_tokens = self.patch_embed(x) # B,C,H,W
        patch_tokens = patch_tokens.permute(0, 2, 3, 1) # BHWC
        for model in self.swins:
            patch_tokens = model(patch_tokens) # B,H/(P*8),W(P*8),C*8

        patch_tokens = patch_tokens.squeeze(1) # B, W , 96*16
        # print(f"Patch Tokens {patch_tokens.shape}")
        phoneme_event = self.linear(patch_tokens) # B, W, 39
        # print(f"Phoneme Event {phoneme_event.shape}")
        phoneme_logit = self.phoneme_pool(phoneme_event.permute(0,2,1)).squeeze(-1)
        # print(f"Phoneme Login {phoneme_logit.shape}")
        return phoneme_logit
 


    
    def training_step(self, batch, batch_idx):
        log_mel = self.spec_aug(batch["log_mel"])
        phoneme_target = batch["phoneme_target"]
        
        phoneme_logit = self(log_mel)

        loss = F.binary_cross_entropy_with_logits(phoneme_logit,phoneme_target)

        return loss

    def validation_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        phoneme_target = batch["phoneme_target"]
        
        phoneme_logit = self(log_mel)

        loss = F.binary_cross_entropy_with_logits(phoneme_logit,phoneme_target)

        # 累积日志
        self.outputs["phoneme_logit"] = phoneme_logit.float() if self.outputs["phoneme_logit"] is None else torch.cat((self.outputs["phoneme_logit"],phoneme_logit.float()),dim=0)
        self.outputs["phoneme_target"] = phoneme_target.float() if self.outputs["phoneme_target"] is None else torch.cat((self.outputs["phoneme_target"],phoneme_target.float()),dim=0)
        
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
            "phoneme_logit" : None,
            "phoneme_target" : None
        }
    def on_validation_epoch_end(self):
        phoneme_logit = self.outputs["phoneme_logit"]
        phoneme_target = self.outputs["phoneme_target"]

        loss = F.binary_cross_entropy_with_logits(phoneme_logit, phoneme_target)

        # numpy
        y = torch.sigmoid(phoneme_logit).float().detach().cpu().numpy()
        target = phoneme_target.float().detach().cpu().numpy().astype(int)

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
    