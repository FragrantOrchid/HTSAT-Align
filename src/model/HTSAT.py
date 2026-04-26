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
        # BCHW
        # 使用简单卷积
        self.conv = nn.Conv2d(
            in_channels=96*16,
            out_channels=35,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # 使用LSTM模块
        self.lstm = nn.LSTM(
            input_size=96*16,
            hidden_size=96*16,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            proj_size=96
        )
        # self.proj = nn.Sequential(
        #     nn.Linear(2 * 96, 35, bias=True),
        #     nn.GELU(),               # 或 ReLU/Tanh
        #     nn.Dropout(0.0)
        # )
        self.proj = nn.Linear(
            in_features=96*2,
            out_features=35
        )
        

    def forward(self,x):
        # x: B,C,H,W
        patch_tokens = self.patch_embed(x) # B,C,H,W
        patch_tokens = patch_tokens.permute(0, 2, 3, 1) # BHWC
        for model in self.swins:
            patch_tokens = model(patch_tokens) # B,H/(P*8),W(P*8),C*8
            # self.print(f"Swin step {patch_tokens.shape}")
        latent_tokens = patch_tokens.permute(0,3,1,2) # B,C*8,H/(P*8),W(P*8)
        
        # 直接卷积
        # label_map = self.conv(latent_tokens) # B , c', 1, W
        # pooled = nn.AdaptiveAvgPool2d((1, 1))(label_map)  # 输出形状 (B, 35, 1, 1)
        # logit = pooled.view(pooled.size(0), -1)  # 展平为 (B, 35)
        
        # LSTM模块
        # 将特征图重塑为序列格式 (B, T, C)
        B, C, H, W = latent_tokens.shape
        # 将空间维度合并作为时间步长 - 需要转置以获得正确的维度顺序
        latent_tokens = latent_tokens.permute(0, 2, 3, 1)  # (B, H, W, C)
        latent_tokens = latent_tokens.contiguous().view(B, H * W, C)  # (B, H*W, C)
        
        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(latent_tokens)
        
        # 使用双向LSTM的最终状态进行预测
        # 对于双向LSTM，h_n的形状是 (num_layers * num_directions, batch, hidden_size)
        # 我们取最后两层的前向和后向隐藏状态
        h_forward = h_n[-2]  # 前向最后一层
        h_backward = h_n[-1]  # 后向最后一层
        h_cat = torch.cat([h_forward, h_backward], dim=-1)  # 拼接
        
        # 添加层归一化以稳定训练
        h_cat = torch.nn.functional.normalize(h_cat, p=2, dim=1)
        
        logit = self.proj(h_cat)
        return torch.sigmoid(logit)
 


    
    def training_step(self, batch, batch_idx):
        log_mel = self.spec_aug(batch["log_mel"])

        target = batch["target"]

        y_prob = self(log_mel)

        loss = F.binary_cross_entropy(y_prob,target)

        return loss

    def validation_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        
        target = batch["target"]

        y_prob = self(log_mel)

        loss = F.binary_cross_entropy(y_prob,target)

        # 累积日志
        self.outputs["y_prob"] = y_prob.float() if self.outputs["y_prob"] is None else torch.cat((self.outputs["y_prob"],y_prob.float()),dim=0)
        self.outputs["target"] = target.float() if self.outputs["target"] is None else torch.cat((self.outputs["target"],target.float()),dim=0)

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
            "y_prob" : None,
            "target" : None
        }
    def on_validation_epoch_end(self):
        y_prob = self.outputs["y_prob"]
        target = self.outputs["target"]

        loss = F.binary_cross_entropy(y_prob,target)

        # numpy
        y = y_prob.float().detach().cpu().numpy()
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
    