import torch          # 补充主库
import torch.nn as nn
import torch.nn.functional as F  # 补充functional模块


class OrthogonalProjectionHeads(nn.Module):
    """正交投影头生成器"""
    def __init__(self, num_heads, in_dim):
        super().__init__()
        self.proj_matrices = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_dim, in_dim)) 
            for _ in range(num_heads)
        ])
        # 正交初始化
        for mat in self.proj_matrices:
            nn.init.orthogonal_(mat)
    
    def forward(self, x):
        # x: [batch, in_dim] 或 [nodes, in_dim]
        projections = [F.linear(x, mat) for mat in self.proj_matrices]
        return torch.stack(projections, dim=1)  # [batch, num_heads, out_dim]
