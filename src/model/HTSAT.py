import pytorch_lightning as pl
from src.dataset.HTSATdataset import HTSATdataset
from torch import nn
import torch
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformerBlockV2
class HTSAT(pl.LightningModule):
    def __init__(self,class_num: int):
        super().__init__()
        self.class_num = class_num
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
        target = self.avg(event) # B,W,1
        target = target.squeeze(-1) # B,W(W = num_frame)
        return target



    def training_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        target = batch["target"]

        y = self(log_mel)
        loss = torch.nn.functional.cross_entropy(y,target.float())

        # 计算准确率
        preds = torch.argmax(y, dim=1)
        true_labels = torch.argmax(target, dim=1)
        acc = (preds == true_labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss



    def validation_step(self, batch, batch_idx):
        log_mel = batch["log_mel"]
        target = batch["target"]

        y = self(log_mel)
        loss = torch.nn.functional.cross_entropy(y, target)

        # 计算准确率
        preds = torch.argmax(y, dim=1)
        true_labels = torch.argmax(target, dim=1)
        acc = (preds == true_labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
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