import pytorch_lightning as pl
from src.dataset.HTSATdataset import HTSATdataset
from torch import nn
import torch
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformerBlockV2
import logging
import numpy as np
from sklearn import metrics
from src.layer.ScalarFilayer import ScalarFiLMLayer
from torchlibrosa.augmentation import SpecAugmentation
import torch.nn.functional as F
from g2p_en import G2p
CLASS_NAMES =     [
        "backward",
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "follow",
        "forward",
        "four",
        "go",
        "happy",
        "house",
        "learn",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "visual",
        "wow",
        "yes",
        "zero"
    ]
def levenshtein_distance(s1, s2):
    """纯Python实现编辑距离，避免 nltk 版本兼容问题"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]
    
def compute_phonetic_similarity(labels, alpha=1.0, strip_stress=True):
    """
    生成 [N, N] 发音相似度矩阵
    labels: list[str] 所有类别名称
    alpha: 相似度衰减系数，越大对混淆惩罚越敏感 (默认1.0)
    strip_stress: 是否去除重音标记(1,2,0)，建议开启
    """
    g2p = G2p()
    phoneme_seqs = []
    
    for word in labels:
        try:
            phon = g2p(word)  # e.g., ['F', 'AO1', 'R', 'W', 'AH0', 'R', 'D']
            if strip_stress:
                phon = [p.replace('0','').replace('1','').replace('2','') for p in phon]
            phoneme_seqs.append(phon)
        except:
            # OOV 词 fallback：按字母分拆（保守策略，相似度会偏低）
            phoneme_seqs.append(list(word.lower()))

    n = len(labels)
    sim_matrix = np.eye(n, dtype=np.float32)  # 对角线为 1.0

    for i in range(n):
        for j in range(i + 1, n):
            dist = levenshtein_distance(phoneme_seqs[i], phoneme_seqs[j])
            max_len = max(len(phoneme_seqs[i]), len(phoneme_seqs[j]))
            if max_len == 0:
                sim = 0.0
            else:
                # 指数衰减相似度：sim ∈ (0, 1]
                sim = np.exp(-alpha * dist / max_len)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix

sim_mat = compute_phonetic_similarity(CLASS_NAMES, alpha=1.0)
class HTSAT(pl.LightningModule):
    def __init__(self, class_num: int, entropy_film: bool, vowel_embed: bool, sound_length: int):
        super().__init__()
        self.strict_loading = False
        self.class_num = class_num
        self.entropy_film = entropy_film
        self.vowel_embed = vowel_embed
        self.sound_length = sound_length
        self.spec_aug = SpecAugmentation(time_drop_width=64,time_stripes_num=sound_length*2//5,freq_drop_width=8,freq_stripes_num=2)
        if self.vowel_embed:
            # B,C,Width,Height
            # Input: B,1,sound_length*160,16
            # Output: B,1,sound_length*160,64
            self.vowel_padding = nn.Linear(in_features=12,out_features=64)
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
        # self.avg = nn.AdaptiveAvgPool1d(1) # B,C,sequence_length -> B,C,1
        self.att_pool = nn.Sequential(
            nn.Conv1d(self.class_num, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        if self.entropy_film:
            self.film = ScalarFiLMLayer(num_channel=class_num,hidden_dim=96*4)

    def forward(self,x,entropy,vowel):
        # 扩展元音表
        
        if self.vowel_embed:
            x = (x-x.mean()) / x.std()
            vowel = (vowel - vowel.mean()) / vowel.std()
            vowel = vowel.unsqueeze(1)
            vowel = self.vowel_padding(vowel)
            vowel = vowel.permute(0,1,3,2) # B,1,64,sound_length*160
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
        # print(event)
        # target = self.avg(event) # B,C,1
        # target = target.squeeze(-1) # B,C(C = num_class)
        attn_weights = self.att_pool(event)  # B, 1, W
        target = (event * attn_weights).sum(dim=-1)  # B, C (注意力加权池化)
        target = target.squeeze(-1)
        return torch.sigmoid(target)



    def training_step(self, batch, batch_idx):
        log_mel = self.spec_aug(batch["log_mel"])
        target = batch["target"]
        entropy = batch["entropy"] if self.entropy_film else None
        vowel = batch["vowel"] if self.vowel_embed else None

        y = self(log_mel,entropy,vowel)
        # loss = torch.nn.functional.binary_cross_entropy(y,target.float())
        loss = self.phonetic_weighted_bce_loss(y,target.float())
        return loss

    def validation_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        target = batch["target"]
        entropy = batch["entropy"] if self.entropy_film else None
        vowel = batch["vowel"] if self.vowel_embed else None

        y = self(log_mel,entropy,vowel)
        # 损失函数
        # loss = torch.nn.functional.binary_cross_entropy(y,target.float())
        loss = self.phonetic_weighted_bce_loss(y,target.float())
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

    def on_validation_epoch_start(self):
        self.outputs = {
            "y": None,
            "target": None
        }
    def on_validation_epoch_end(self):
        y = self.outputs["y"]
        target = self.outputs["target"]  
        # loss = torch.nn.functional.binary_cross_entropy(y, target)
        loss = self.phonetic_weighted_bce_loss(y,target)
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
    
    def phonetic_weighted_bce_loss(self, logits, targets, lambda_sim=0.5):
        sim_matrix = torch.tensor(sim_mat, device=logits.device, dtype=logits.dtype)
        # probs = torch.sigmoid(logits)
        # 标准 BCE
        bce = F.binary_cross_entropy(logits, targets, reduction='none')  # B, C

        # 相似性正则项：若 target[i]=1，则推远 logit[j]（j为高相似负类）
        sim_reg = torch.zeros_like(bce)
        for i in range(logits.shape[0]):
            pos_mask = targets[i] > 0.5
            if pos_mask.any():
                # 加权推送：对高相似度类别施加更大排斥力
                sim_reg[i] = (sim_matrix[pos_mask] * (1 - targets[i])).sum(dim=0)

        # 梯度方向：相似负类的预测概率越高，惩罚越大
        sim_loss = (sim_reg * logits).mean()
        return bce.mean() + lambda_sim * sim_loss