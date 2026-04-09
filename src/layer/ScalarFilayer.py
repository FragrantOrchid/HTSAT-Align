import torch
import torch.nn as nn

class ScalarFiLMLayer(nn.Module):
    """
    用单个标量对 (num_frame, num_channel) 特征进行 FiLM 调制
    """
    def __init__(self, num_channel, hidden_dim=None, use_relu=True):
        super().__init__()
        h = hidden_dim or num_channel
        layers = [nn.Linear(1, h)]
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(h, 2 * num_channel))
        self.generator = nn.Sequential(*layers)
        self.num_channel = num_channel
        self._init_identity()

    def _init_identity(self):
        """初始化为恒等映射：训练初期 gamma=1, beta=0，提升稳定性"""
        last_fc = self.generator[-1]
        nn.init.zeros_(last_fc.weight)
        nn.init.ones_(last_fc.bias[:self.num_channel])
        nn.init.zeros_(last_fc.bias[self.num_channel:])

    def forward(self, x, scalar):
        """
        Args:
            x:      (num_frame, num_channel) 或 (batch, num_frame, num_channel)
            scalar: 标量 或 (batch,) 张量
        Returns:
            modulated x, shape 同输入
        """
        has_batch = x.dim() == 3
        if not has_batch:
            x = x.unsqueeze(0)
            
        # 统一 scalar 形状为 (batch, 1)
        if not torch.is_tensor(scalar):
            scalar = torch.tensor([scalar], device=x.device, dtype=x.dtype)
        elif scalar.dim() == 0:
            scalar = scalar.unsqueeze(0)
        scalar = scalar.view(-1, 1)

        # 生成 gamma, beta -> 形状 (batch, 2*C)
        params = self.generator(scalar)
        gamma, beta = params.chunk(2, dim=-1)  # 各为 (batch, C)

        # 扩展 frame 维度以便广播: (batch, 1, C)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        out = gamma * x + beta
        return out.squeeze(0) if not has_batch else out
