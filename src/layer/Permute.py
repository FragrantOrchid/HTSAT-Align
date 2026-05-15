import torch.nn as nn

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims  # 例如 (0, 2, 3, 1)
        
    def forward(self, x):
        return x.permute(self.dims)