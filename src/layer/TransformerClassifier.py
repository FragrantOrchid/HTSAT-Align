import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, mid_dim, end_dim, n_layers=2, n_heads=3, dropout=0.1):
        super().__init__()
        self.mid_dim = mid_dim
        self.cls = nn.Parameter(torch.randn(1, 1, mid_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(mid_dim, n_heads, dim_feedforward=4*mid_dim, dropout=dropout, batch_first=True),
            n_layers
        )
        self.norm = nn.LayerNorm(mid_dim)
        self.head = nn.Linear(mid_dim, end_dim)

    def forward(self, x):
        # 输入 x 的形状应为 (B, L, D)
        B, L, input_D = x.size()

        # 如果输入维度与期望的mid_dim不匹配，则使用线性投影层调整
        if input_D != self.mid_dim:
            # 创建一个动态投影层
            projection = nn.Linear(input_D, self.mid_dim).to(x.device)
            x = projection(x)

        D = self.mid_dim  # 现在D等于mid_dim

        # 动态生成位置编码 (B, L, D)
        pos_encoding = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        
        position = torch.arange(0, L, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float, device=x.device) *
                            (-torch.log(torch.tensor(10000.0, device=x.device)) / D))
        
        # 确保 pos_encoding 的偶数索引位置填充 sin 值
        pe = torch.zeros(L, D, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term[:min(D//2 + D%2, len(div_term))])
        
        # 确保 pos_encoding 的奇数索引位置填充 cos 值
        if D > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:min(D//2, len(div_term))])
        
        pos_encoding = pe.unsqueeze(0).expand(B, -1, -1)

        # 添加位置编码 (B, L, D)
        x = x + pos_encoding

        # 扩展 CLS token 到整个批次 (B, 1, D)
        cls_tokens = self.cls.expand(B, -1, -1)

        # 将 CLS token 拼接到序列开头 (B, L+1, D)
        x = torch.cat([cls_tokens, x], dim=1)

        # 通过 Transformer 编码器
        out = self.encoder(x)[:, 0]                           # 取 CLS token 输出 (B, D)

        # 应用层归一化和分类头
        return self.head(self.norm(out))                      # (B, end_dim)
