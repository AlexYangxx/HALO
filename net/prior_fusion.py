"""
Stage B: 离线 DINO 语义特征在 MST 瓶颈处的投影与融合（concat + 1x1 conv 残差）。
"""
import torch
import torch.nn as nn


class SemanticPriorProjector(nn.Module):
    """将缓存的语义特征通道维投影到瓶颈维数 bottleneck_c。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class SemanticBottleneckFusion(nn.Module):
    """fea 与投影后的语义特征 concat 后经 1x1 卷积，作为注入残差（在 MST 内再乘 scalar）。"""

    def __init__(self, dim: int):
        super().__init__()
        # Normalize both branches before fusion to reduce scale mismatch.
        self.fea_norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.sem_norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        hidden = max(dim // 4, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, fea, sem_proj):
        fea_n = self.fea_norm(fea)
        sem_n = self.sem_norm(sem_proj)
        cat = torch.cat([fea_n, sem_n], dim=1)
        mixed = self.fuse(cat)
        conf = self.gate(cat)
        return conf * mixed
