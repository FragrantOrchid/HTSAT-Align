import torch
import torch.nn as nn

class FILM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) 层
    
    参数:
        num_features (int): 输入特征的数量 (num_class)
        use_bias (bool): 是否使用偏置项 (默认为 True)
    """
    def __init__(self, num_features, use_bias=True):
        super(FILM, self).__init__()
        self.num_features = num_features
        self.use_bias = use_bias
        
        # 初始化缩放和偏置参数
        self.scale = nn.Linear(1, num_features)
        if use_bias:
            self.bias = nn.Linear(1, num_features)
        else:
            self.bias = None
    
    def forward(self, x, conditioning):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_frame, num_class) 或 (num_frame, num_class)
            conditioning (torch.Tensor): 条件标量，形状为 (batch_size, 1) 或 (1,)
            
        返回:
            torch.Tensor: 经过 FiLM 调制后的张量
        """
        # 确保输入有正确的维度
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 添加 batch 维度
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(-1)  # 确保 conditioning 是 (batch_size, 1)
        
        batch_size = x.size(0)
        
        # 计算缩放因子
        scale = self.scale(conditioning).unsqueeze(1)  # (batch_size, 1, num_class)
        
        # 计算偏置项（如果启用）
        if self.use_bias:
            bias = self.bias(conditioning).unsqueeze(1)  # (batch_size, 1, num_class)
        else:
            bias = 0
        
        # 应用 FiLM 变换
        return x * scale + bias