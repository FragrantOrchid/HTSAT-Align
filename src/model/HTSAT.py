import pytorch_lightning as pl
from torch import nn
import torch
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformerBlockV2
import logging
import numpy as np
from sklearn import metrics
# from src.layer.ScalarFilayer import ScalarFiLMLayer
from torchlibrosa.augmentation import SpecAugmentation
import torch.nn.functional as F
# from src.layer.TransformerClassifier import TransformerClassifier
# from src.layer.GlobalGateAttention import GlobalGateAttention
from src.layer.TemporalAwareAttention import TemporalAwareAttention
from src.layer.SGMWithGEModule import SGMWithGEModule
import torchaudio.transforms as T
from util.GaussianSpecAugment import GaussianSpecAugment
from src.layer.Permute import Permute
import math
class HTSAT(pl.LightningModule):
    def __init__(self, class_num: int, sound_length: int):
        super().__init__()
        self.strict_loading = False
        self.class_num = class_num
        self.sound_length = sound_length
        # self.spec_aug = SpecAugmentation(time_drop_width=64,time_stripes_num=sound_length*2//5,freq_drop_width=8,freq_stripes_num=2)
        # self.spec_aug = T.SpecAugment(
        #     n_time_masks=2,
        #     time_mask_param=self.sound_length*32, # 10%
        #     n_freq_masks=0,
        #     freq_mask_param=8,
        #     p=0.5
        # )
        self.spec_aug = GaussianSpecAugment(
            patch_size=(8,16),
            mask_ratio=0.50,
            cluster_strength=(0.35,0.50)
        )
        # B,C,H,W()
        # Input: B,2 if self.vowel_embed else 1,64,sound_length*160
        # Output: B,96,16,sound_length*160
        self.patch_embed = nn.Sequential(
            nn.LayerNorm([128,320]),
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=(8,1),stride=(8,1),padding=0),
            Permute(0,2,3,1),
            nn.LayerNorm([96])
        )
        # all use B,H,W,C 
        # Input: B,16,sound_length*160,96
        # Output: B,1,sound_length*10,96*16
        self.swins_transformer = nn.Sequential(
            SwinTransformerBlockV2(dim=96,num_heads=4,window_size=[7,7],shift_size=[0,0],stochastic_depth_prob=0.00),
            SwinTransformerBlockV2(dim=96,num_heads=4,window_size=[7,7],shift_size=[3,3],stochastic_depth_prob=0.02),
            PatchMergingV2(dim=96),
            SwinTransformerBlockV2(dim=96*2,num_heads=8,window_size=[7,7],shift_size=[0,0],stochastic_depth_prob=0.04),
            SwinTransformerBlockV2(dim=96*2,num_heads=8,window_size=[7,7],shift_size=[3,3],stochastic_depth_prob=0.06),
            PatchMergingV2(dim=96*2),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[0,0],stochastic_depth_prob=0.08),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[3,3],stochastic_depth_prob=0.10),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[0,0],stochastic_depth_prob=0.12),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[3,3],stochastic_depth_prob=0.14),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[0,0],stochastic_depth_prob=0.16),
            SwinTransformerBlockV2(dim=96*4,num_heads=16,window_size=[7,7],shift_size=[3,3],stochastic_depth_prob=0.18),
            PatchMergingV2(dim=96*4),
            SwinTransformerBlockV2(dim=96*8,num_heads=32,window_size=[7,7],shift_size=[0,0],stochastic_depth_prob=0.22),
            SwinTransformerBlockV2(dim=96*8,num_heads=32,window_size=[7,7],shift_size=[3,3],stochastic_depth_prob=0.26),
            PatchMergingV2(dim=96*8)
        )
        
        self.linear = nn.Linear(
            in_features=96*16,
            out_features=47
        )
        # BCHW
        # 使用简单卷积
        # self.trans = nn.Sequential(
        #     PositionalEncoding(96*16, 0.1),
        #     nn.TransformerEncoder(
        #         encoder_layer=nn.TransformerEncoderLayer(
        #             d_model=96*16,
        #             nhead=4,
        #             dropout=0.1,
        #             batch_first=True
        #         ),
        #         num_layers=6
        #     )
        # )
        # self.linear = nn.Linear(
        #     in_features=96*16,
        #     out_features=43
        # )
        # 
        # # 使用LSTM模块
        self.lstm = nn.LSTM(
            input_size=96*16,
            hidden_size=96*16,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        # self.proj = nn.Sequential(
        #     nn.Linear(2 * 96, 35, bias=True),
        #     nn.GELU(),               # 或 ReLU/Tanh
        #     nn.Dropout(0.0)
        # )
        self.proj_drop = nn.Dropout(0.3)
        self.proj = nn.Linear(
            in_features=96*32,
            out_features=self.class_num
        )
        # 
        # self.phoneme_pool = nn.AdaptiveMaxPool1d(1)
        
    
    def forward(self,x):
        # x: B,C,H,W
        patch_tokens = self.patch_embed(x) # B,H,W,C
        # print(f"patch_tokens shape {patch_tokens.shape}")
        swin_output = self.swins_transformer(patch_tokens) # B,H,W,C

        # lstm_input = self.linear(swin_output.squeeze(1)) # B, W , C'
        # lstm_input = torch.sigmoid(lstm_input)
        lstm_input = swin_output.squeeze(1)
        if self.training:
            # 训练时随机截断（如 40% 概率被截断）
            if torch.rand(1).item() < 0.4:
                lstm_input, mask, _ = self.random_truncate(lstm_input)
            else:
                mask = None
        else:
            mask = None  # 推理时正常输入，不截断
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)


        # logit = self.proj(self.proj_drop(torch.cat([h_n[-2],h_n[-1]],dim=-1)))
        logit = self.proj(torch.cat([h_n[-2],h_n[-1]],dim=-1))
        # phoneme_logit = self.phoneme_pool(phoneme_event.permute(0,2,1)).squeeze(-1)

        return logit
 


    
    def training_step(self, batch, batch_idx):
        log_mel = self.spec_aug(batch["log_mel"])
        # log_mel = batch["log_mel"]
        target = batch["target"]
        
        logit = self(log_mel)

        # loss = F.binary_cross_entropy_with_logits(logit,target)
        loss = F.binary_cross_entropy_with_logits(
            input = logit,
            target = target
        )
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        target = batch["target"]
        
        logit = self(log_mel)

        loss = F.binary_cross_entropy_with_logits(logit,target)


        # 累积日志
        self.outputs["logit"] = logit.float() if self.outputs["logit"] is None else torch.cat((self.outputs["logit"],logit),dim=0)
        self.outputs["target"] = target.float() if self.outputs["target"] is None else torch.cat((self.outputs["target"],target),dim=0)
        return loss
    
    def test_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        target = batch["target"]
        logit = self(log_mel)
        loss = F.binary_cross_entropy_with_logits(logit,target)
        # 累积日志
        self.outputs["logit"] = logit.float() if self.outputs["logit"] is None else torch.cat((self.outputs["logit"],logit),dim=0)
        self.outputs["target"] = target.float() if self.outputs["target"] is None else torch.cat((self.outputs["target"],target),dim=0)
        return loss

    def configure_optimizers(self):
        # 1. 优化器配置 (AdamW)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-5,  # 学习用-4,微调-5
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
        y = torch.sigmoid(logit).float().detach().cpu().numpy()
        target = target.float().detach().cpu().numpy().astype(int)

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
        self.log("val_simples",y.shape[0],sync_dist=True)
        self.log("val_loss",loss,sync_dist=True)
        self.log("val_acc",acc,sync_dist=True)
        self.log("val_mAP",np.mean(average_precision_scores),sync_dist=True)
        logging.getLogger("lightning.pytorch").info(
            f'Epoch{self.current_epoch:03d}\tvalidation_step\n{confusion_matrix}'
        )
        for k in range(self.class_num):
            logging.getLogger("lightning.pytorch").info(
                f'class_index:{k:03d}\tAP:{average_precision_scores[k]:.4f}\tauc:{roc_auc_scores[k]:.4f}'
            )
        y_true=np.argmax(target,1)
        y_pred=np.argmax(y,1)
        for index in range(y.shape[0]):
            if y_true[index] != y_pred[index]:
                    logging.getLogger("lightning.pytorch").info(
                        f'Index={index},{y_true[index]} -> {y_pred[index]} : {self.trainer.val_dataloaders.dataset.get_filename(index)}'
                    )

    def on_test_epoch_start(self):
        self.outputs = {
            "logit" : None,
            "target" : None
        }
    def on_test_epoch_end(self):
        logit = self.outputs["logit"]
        target = self.outputs["target"]

        loss = F.binary_cross_entropy_with_logits(logit, target)

        # numpy
        y = torch.sigmoid(logit).float().detach().cpu().numpy()
        target = target.float().detach().cpu().numpy().astype(int)

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
        self.log("test_simples",y.shape[0],sync_dist=True)
        self.log("test_loss",loss,sync_dist=True)
        self.log("test_acc",acc,sync_dist=True)
        self.log("test_mAP",np.mean(average_precision_scores),sync_dist=True)
        logging.getLogger("lightning.pytorch").info(
            f'Epoch{self.current_epoch:03d}\tvalidation_step\n{confusion_matrix}'
        )
        for k in range(self.class_num):
            logging.getLogger("lightning.pytorch").info(
                f'class_index:{k:03d}\tAP:{average_precision_scores[k]:.4f}\tauc:{roc_auc_scores[k]:.4f}'
            )
        y_true=np.argmax(target,1)
        y_pred=np.argmax(y,1)
        for index in range(y.shape[0]):
            if y_true[index] != y_pred[index]:
                    logging.getLogger("lightning.pytorch").info(
                        f'Index={index},{y_true[index]} -> {y_pred[index]} : {self.trainer.test_dataloaders.dataset.get_filename(index)}'
                    )
    
    def random_truncate(self, x, min_keep_ratio=0.4, max_keep_ratio=1.0):
        """
        x: (B, W, C) - swin_output.squeeze(1) 后的 LSTM 输入
        返回截断后的序列和对应的 mask
        """
        B, W, C = x.shape
        # 为每个样本随机生成保留长度
        keep_ratios = torch.empty(B).uniform_(min_keep_ratio, max_keep_ratio).to(x.device)
        keep_lengths = (keep_ratios * W).long().clamp(min=1)  # 至少保留 1 帧

        # 生成 mask
        mask = torch.arange(W, device=x.device).unsqueeze(0) < keep_lengths.unsqueeze(1)  # (B, W)
        # 将截断部分置零（或填充特定值）
        x_truncated = x * mask.unsqueeze(-1)
        return x_truncated, mask, keep_lengths