import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Union
import random
class GaussianSpecAugment(torch.nn.Module):
    def __init__(self, patch_size, mask_ratio, cluster_strength:Union[float,Tuple[float,float]]):
        super(GaussianSpecAugment, self).__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # 判断cluster_strength是float还是tuple，如果是float，变成一个两个数值相同的tuple
        if isinstance(cluster_strength, float):
            self.cluster_strength = (cluster_strength, cluster_strength)
        else:
            self.cluster_strength = cluster_strength

    def forward(self, specgram: Tensor) -> Tensor:
        B, C, H, W = specgram.shape
        
        # Reshape specgram to combine batch and channel dimensions for vectorized processing
        reshaped_spec = specgram.view(B*C, H, W)
        
        # Apply block masking to all reshaped tensors at once
        masked_spec, _ = block_mask_with_clustering(
            reshaped_spec,
            self.patch_size,
            self.mask_ratio,
            random.uniform(*self.cluster_strength)
        )
        
        # Reshape back to original dimensions
        return masked_spec.view(B, C, H, W)
def block_mask_with_clustering(tensor, patch_size, mask_ratio, cluster_strength=0.0):
    """
    对2D或3D张量进行分块掩码，支持调节聚集程度
    Args:
        tensor: (H, W) 或 (N, H, W) 张量
        patch_size: (ph, pw) 块大小
        mask_ratio: float, 掩码比例 [0, 1]
        cluster_strength: float, 聚集强度 [0, 1] (0=均匀随机, 1=高度聚集)
    Returns:
        masked_tensor: 掩码后的张量（被掩码位置置0）
        mask: (H, W) 或 (N, H, W) 布尔掩码张量 (True表示被掩码)
    """
    original_dims = len(tensor.shape)
    
    # 如果是2D张量，增加一个维度使其变为3D
    if original_dims == 2:
        tensor = tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    N, H, W = tensor.shape
    ph, pw = patch_size
    assert H % ph == 0 and W % pw == 0, "张量尺寸必须能被patch_size整除"
    nh, nw = H // ph, W // pw

    # 1. 生成块级随机图 (N, nh, nw)
    rand_map = torch.rand(N, nh, nw, device=tensor.device)

    # 2. 高斯平滑（控制聚集程度）
    if cluster_strength > 0.0:
        max_sigma = max(nh, nw) / 4.0
        sigma = cluster_strength * max_sigma
        # 构造高斯核尺寸（奇数）
        size = int(2 * sigma + 0.5) * 2 + 1
        if size < 3: size = 3
        # 1D高斯核
        x = torch.arange(size, device=tensor.device) - (size - 1) / 2.0
        k1d = torch.exp(-x**2 / (2 * sigma**2))
        k1d /= k1d.sum()
        # 2D高斯核
        k2d = k1d[:, None] * k1d[None, :]
        k2d = k2d.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

        # 批量卷积平滑 - 使用torch.vmap或循环
        # 由于F.conv2d的限制，我们使用列表推导式和torch.stack
        smoothed_list = []
        for i in range(N):
            rand_4d_single = rand_map[i:i+1].unsqueeze(0)  # (1, 1, nh, nw)
            pad = size // 2
            rand_padded = F.pad(rand_4d_single, (pad, pad, pad, pad), mode='reflect')
            convolved = F.conv2d(rand_padded, k2d)
            smoothed_list.append(convolved.squeeze())
        smoothed = torch.stack(smoothed_list, dim=0)  # (N, nh, nw)
    else:
        smoothed = rand_map

    # 3. 精确选取 mask_ratio 比例的块
    num_patches = nh * nw
    num_masked = int(num_patches * mask_ratio)
    
    if num_masked == 0:
        mask_patches = torch.zeros(N, nh, nw, dtype=torch.bool, device=tensor.device)
    else:
        # 向量化topk操作
        flattened_smoothed = smoothed.view(N, -1)  # (N, nh*nw)
        _, indices = torch.topk(flattened_smoothed, num_masked, largest=False, sorted=False, dim=1)  # (N, num_masked)
        
        # 创建布尔掩码
        mask_flat = torch.zeros(N, num_patches, dtype=torch.bool, device=tensor.device)
        batch_indices = torch.arange(N, device=tensor.device).unsqueeze(1).expand(-1, num_masked)  # (N, num_masked)
        mask_flat[batch_indices, indices] = True
        mask_patches = mask_flat.view(N, nh, nw)

    # 4. 上采样至原图分辨率
    # 使用repeat_interleave对批量数据进行上采样
    mask = mask_patches.repeat_interleave(ph, dim=1).repeat_interleave(pw, dim=2)

    # 5. 应用掩码
    # 计算每个样本的均值用于填充
    fill_values = tensor.mean(dim=(1, 2), keepdim=True)  # (N, 1, 1)
    # 将fill_values扩展到与tensor相同的形状 (N, H, W)
    fill_values_expanded = fill_values.expand_as(tensor)
    # 使用掩码将均值填充到被掩码区域
    masked_tensor = torch.where(mask, fill_values_expanded, tensor)

    # 如果原始输入是2D，则输出也应为2D
    if squeeze_output:
        masked_tensor = masked_tensor.squeeze(0)
        mask = mask.squeeze(0)

    return masked_tensor, mask