import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.GaussianSpecAugment import GaussianSpecAugment

def test_performance():
    # 定义参数
    patch_size = (8, 8)
    mask_ratio = 0.3
    cluster_strength = 0.5
    
    # 创建测试张量
    B, C, H, W = 4, 8, 128, 128  # 减小尺寸以便快速测试
    specgram = torch.randn(B, C, H, W)
    
    # 初始化模型
    model = GaussianSpecAugment(patch_size, mask_ratio, cluster_strength)
    
    # 预热
    for _ in range(5):
        _ = model(specgram)
    
    # 测试优化后模型的性能
    start_time = time.time()
    for _ in range(10):
        output = model(specgram)
    optimized_time = time.time() - start_time
    
    print(f"Optimized version time: {optimized_time:.4f}s for 10 iterations")
    print(f"Average time per iteration: {optimized_time/10:.4f}s")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_performance()