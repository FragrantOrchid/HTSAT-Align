import torch
import torch.nn as nn
import math

class TemporalAwareAttention(nn.Module):
    """
    带有位置编码的时间感知Attention模块
    将 (B, W, C) 序列通过带有位置信息的Attention机制聚合为 (B, C', hidden_dim)
    核心思想：在传统的Luong Attention基础上加入位置编码，
    使模型能够感知时间序列的先后顺序
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, prob: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.max_len = 1000  # 假设最大序列长度为1000，可根据需要调整
        
        # 位置编码
        self.position_encoding = self._create_position_encoding()
        
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
        
        # 扩展到 hidden_dim 维度
        self.expand_to_hidden = nn.Linear(1, hidden_dim)
        self.out_channels = out_channels
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=prob)

    def _create_position_encoding(self):
        """
        创建正弦位置编码矩阵
        """
        position_encoding = torch.zeros(self.max_len, self.in_channels)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()

        # 计算一半的维度数
        half_dim = self.in_channels // 2
        
        # 生成一半维度的div_term
        div_term = torch.exp(torch.arange(0, half_dim, 1).float() *
                            -(math.log(10000.0) / self.in_channels))

        # 分别填充偶数和奇数位置
        # 对于偶数索引（0,2,4,...），如果总数是奇数，最后一个索引也会是偶数
        position_encoding[:, 0::2][:, :half_dim] = torch.sin(position * div_term)
        
        # 对于奇数索引（1,3,5,...）
        if self.in_channels > 1:
            position_encoding[:, 1::2] = torch.cos(position * div_term[:self.in_channels // 2])

        return position_encoding.unsqueeze(0)  # (1, max_len, in_channels)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, W, C) 输入序列/窗口特征
        Returns:
            output: (B, C', hidden_dim) 聚合后的全局特征
            weights: (B, W) 归一化注意力权重（可用于可视化/可解释性分析）
        """
        B, W, C = x.size()
        
        # 确保位置编码维度匹配
        pos_encoding = self.position_encoding[:, :W, :].to(x.device)
        
        # 添加位置编码
        x_with_pos = x + pos_encoding[:, :W, :]
        
        # 1. 构造 Query：全局平均池化提取初始上下文状态（使用带有位置信息的x）
        h = x_with_pos.mean(dim=1)  # (B, C)

        # 2. Query 非线性变换 & 升维适配 bmm
        gamma_h = self.linear_in(h).unsqueeze(2)  # (B, C, 1)

        # 3. 计算注意力分数 (x_with_pos @ gamma_h)
        scores = torch.bmm(x_with_pos, gamma_h).squeeze(2)  # (B, W)

        # 4. Dropout 正则化 + Softmax 归一化
        weights = self.softmax(self.dropout(scores))  # (B, W)

        # 5. 加权聚合上下文向量
        c_t = torch.bmm(weights.unsqueeze(1), x_with_pos).squeeze(1)  # (B, C)

        # 6. 门控融合 & 维度投影
        output = self.linear_out(torch.cat([h, c_t], dim=1))  # (B, 2C) -> (B, C')

        # 7. 扩展至 hidden_dim 维度并重塑输出形状
        output = output.unsqueeze(2)  # (B, C', 1)
        output = self.expand_to_hidden(output)  # (B, C', hidden_dim)

        return output, weights