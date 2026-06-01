from torch import nn
import torch.nn.functional as F
import torch
class LSTMKeywordDetector(nn.Module):
    def __init__(self, input_dim=1536, proj_dim=256, hidden_size=128,
                 num_classes=35, num_layers=2):
        """
        input_dim: 原始特征维度 1536
        proj_dim: 投影降维后的维度
        hidden_size: LSTM 隐藏层维度
        """
        super().__init__()
        # 逐帧降维（独立作用于每个时间步）
        self.proj = nn.Linear(input_dim, proj_dim)
        self.proj_bn = nn.BatchNorm1d(20)  # 对时间步维度做 BN
        
        # 双向 LSTM
        self.lstm = nn.LSTM(proj_dim, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.2)
        
        # 注意力机制
        self.attn = nn.Linear(hidden_size * 2, 1)
        
        # Dropout + 分类头
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        x: (B, 20, 1536)
        返回: (B, 35) logits
        """
        B, W, C = x.shape
        
        # 逐帧降维
        x = self.proj(x)                           # (B, 20, 256)
        x = self.proj_bn(x)                        # 归一化
        x = F.relu(x)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)                 # (B, 20, 256)  [128*2=256]
        
        # 注意力加权
        attn_scores = self.attn(lstm_out)          # (B, 20, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (B, 256)
        
        context = self.dropout(context)
        logits = self.fc(context)                  # (B, 35)
        return logits
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class KWS_CNN(nn.Module):
    def __init__(self, in_channels=47, num_classes=35, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.pool  = nn.AdaptiveAvgPool1d(1)          # 全局池化，压缩时间维度
        self.fc    = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """ x: (B, 20, 47) → 输出: (B, 35) 未经 sigmoid 的 logit """
        x = x.transpose(1, 2)                         # (B, 47, 20)
        x = F.relu(self.bn1(self.conv1(x)))           # (B, 64, 20)
        x = F.relu(self.bn2(self.conv2(x)))           # (B, 64, 20)
        x = self.pool(x).squeeze(-1)                  # (B, 64)
        return self.fc(x)                             # (B, 35)