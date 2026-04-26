import torch
import torch.nn as nn

class GlobalGateAttention(nn.Module):
    """
    将 (B, W, C) 序列通过类 Luong Gate Attention 机制聚合为 (B, C', hidden_dim)
    核心思想：用全局池化结果作为 Query，对原始序列计算注意力权重，
    加权提取关键特征后与 Query 拼接，经双层 MLP 门控投影到目标维度。
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, prob: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Query 变换网络 (C -> C)
        self.linear_in = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.SELU(),
            nn.Dropout(p=prob),
            nn.Linear(in_channels, in_channels),
            nn.SELU(),
            nn.Dropout(p=prob)
        )
        # 输出融合/门控网络 (2C -> C')
        self.linear_out = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.SELU(),
            nn.Dropout(p=prob),
            nn.Linear(out_channels, out_channels),
            nn.SELU(),
            nn.Dropout(p=prob)
        )
        # 扩展到 hidden_dim 维度 - 修正此处定义
        # 创建一个1x1卷积层或线性层来扩展最后一个维度
        self.expand_to_hidden = nn.Linear(1, hidden_dim)
        self.out_channels = out_channels
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, W, C) 输入序列/窗口特征
        Returns:
            output: (B, C', hidden_dim) 聚合后的全局特征
            weights: (B, W) 归一化注意力权重（可用于可视化/可解释性分析）
        """
        # 1. 构造 Query：全局平均池化提取初始上下文状态
        h = x.mean(dim=1)  # (B, C)

        # 2. Query 非线性变换 & 升维适配 bmm
        gamma_h = self.linear_in(h).unsqueeze(2)  # (B, C, 1)

        # 3. 计算注意力分数 (x @ gamma_h)
        scores = torch.bmm(x, gamma_h).squeeze(2)  # (B, W)

        # 4. Dropout 正则化 + Softmax 归一化
        weights = self.softmax(self.dropout(scores))  # (B, W)

        # 5. 加权聚合上下文向量
        c_t = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, C)

        # 6. 门控融合 & 维度投影
        output = self.linear_out(torch.cat([h, c_t], dim=1))  # (B, 2C) -> (B, C')

        # 7. 扩展至 hidden_dim 维度并重塑输出形状
        # 将 (B, C') reshape 为 (B, C', 1)
        output = output.unsqueeze(2)  # (B, C', 1)
        # 应用线性层从 1 扩展到 hidden_dim
        output = self.expand_to_hidden(output)  # (B, C', hidden_dim)

        return output, weights