"""
Windowed Dual-Prior Context-Aware Attention (DP-CAA).
Local window attention with additive logits from semantic affinity and depth-based geometry.
Default off; does not change behavior when disabled at MST level.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_multiple(x: torch.Tensor, window_size: int):
    """
    Pad NCHW so H,W are multiples of window_size.
    Prefer reflect; fall back to replicate when reflect is invalid (PyTorch requires pad < dim per side).
    """
    _, _, H, W = x.shape
    ws = int(window_size)
    ph = (ws - H % ws) % ws
    pw = (ws - W % ws) % ws
    if ph == 0 and pw == 0:
        return x, ph, pw
    # reflect: bottom pad ph must satisfy ph < H; right pad pw must satisfy pw < W
    if ph < H and pw < W:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    else:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, ph, pw


def _window_partition_nchw(x: torch.Tensor, window_size: int):
    """[B,C,H,W] -> qkv windows [B*nw, heads, N, d] helper: first flatten spatial windows."""
    B, C, H, W = x.shape
    ws = int(window_size)
    nh, nw = H // ws, W // ws
    # [B, C, nh, ws, nw, ws] -> [B*nh*nw, ws*ws, C]
    t = x.view(B, C, nh, ws, nw, ws)
    t = t.permute(0, 2, 4, 3, 5, 1).reshape(B * nh * nw, ws * ws, C)
    return t, nh, nw


def _merge_windows(t: torch.Tensor, B: int, nh: int, nw: int, window_size: int, C: int) -> torch.Tensor:
    """[B*nw, ws*ws, C] -> [B,C,H,W]"""
    ws = int(window_size)
    t = t.view(B, nh, nw, ws, ws, C)
    t = t.permute(0, 5, 1, 3, 2, 4).reshape(B, C, nh * ws, nw * ws)
    return t


class WindowedDPCAA(nn.Module):
    """
    Bottleneck-only window attention: softmax(QK^T/sqrt(d) + ls*B_sem + lg*B_geo) V.
    B_sem: cosine similarity of projected semantic tokens (optional).
    B_geo: -alpha * |d_i - d_j| from a 1-channel map from depth features.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        window_size: int = 8,
        sem_in_channels: int = 1024,
        sem_embed_dim: int = 32,
        tau: float = 0.07,
    ):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.window_size = int(window_size)
        self.tau = max(float(tau), 1e-6)
        assert self.dim == self.heads * self.dim_head

        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.sem_stem = nn.Conv2d(int(sem_in_channels), int(sem_embed_dim), 1, bias=False)
        self.geo_reduce = nn.Conv2d(dim, 1, 1, bias=False)

        self.logit_lambda_s = nn.Parameter(torch.tensor(-1.5))
        self.logit_lambda_g = nn.Parameter(torch.tensor(-1.0))
        self.geo_alpha = nn.Parameter(torch.tensor(1.0))

        self.scale = self.dim_head ** -0.5
        init_gamma = torch.logit(torch.tensor(0.25), eps=1e-6)
        self.out_gamma = nn.Parameter(init_gamma.clone())

    def forward(
        self,
        x: torch.Tensor,
        depth_feat: torch.Tensor,
        sem_feat: Optional[torch.Tensor],
        sem_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        x, depth_feat: [B, dim, H, W] (same spatial size)
        sem_feat: [B, C_sem, H, W] or None
        sem_scale: same schedule as train sem_warmup (scales semantic affinity logits; geo unchanged)
        """
        B, _, H0, W0 = x.shape
        ws = self.window_size
        residual = x
        x = self.norm(x)

        x_pad, ph, pw = _pad_to_multiple(x, ws)
        d_pad, _, _ = _pad_to_multiple(depth_feat, ws)
        sem_pad = None
        if sem_feat is not None and sem_feat.numel() > 0:
            if sem_feat.shape[2:] != x_pad.shape[2:]:
                sem_pad = F.interpolate(sem_feat, size=x_pad.shape[2:], mode="bilinear", align_corners=False)
            else:
                sem_pad = sem_feat
            sem_pad, _, _ = _pad_to_multiple(sem_pad, ws)

        H, W = x_pad.shape[2], x_pad.shape[3]
        nh, nw = H // ws, W // ws

        q = self.q(x_pad)
        k = self.k(x_pad)
        v = self.v(x_pad)

        def to_heads(feat: torch.Tensor):
            t, _, _ = _window_partition_nchw(feat, ws)
            Bnw, N, C = t.shape
            t = t.view(Bnw, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
            return t

        qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)
        attn_logits = (qh @ kh.transpose(-2, -1)) * self.scale

        Bnw = B * nh * nw
        N = ws * ws
        ls = torch.sigmoid(self.logit_lambda_s)
        lg = torch.sigmoid(self.logit_lambda_g)

        s_scale = float(max(sem_scale, 0.0))
        if sem_pad is not None:
            sem_e = self.sem_stem(sem_pad)
            sem_t, _, _ = _window_partition_nchw(sem_e, ws)
            sem_t = F.normalize(sem_t, dim=-1, eps=1e-6)
            sem_bias = (sem_t @ sem_t.transpose(-2, -1)) / self.tau
            sem_bias = sem_bias.unsqueeze(1).expand(-1, self.heads, -1, -1)
            attn_logits = attn_logits + ls * s_scale * sem_bias

        geo = self.geo_reduce(d_pad)
        geo_t, _, _ = _window_partition_nchw(geo, ws)
        geo_diff = geo_t - geo_t.transpose(-2, -1)
        geo_bias = -self.geo_alpha.abs() * geo_diff.abs()
        geo_bias = geo_bias.unsqueeze(1).expand(-1, self.heads, -1, -1)
        attn_logits = attn_logits + lg * geo_bias

        attn = F.softmax(attn_logits, dim=-1)
        out = attn @ vh
        out = out.permute(0, 2, 1, 3).reshape(Bnw, N, self.dim)
        out = _merge_windows(out, B, nh, nw, ws, self.dim)
        out = self.proj(out)

        if out.shape[2] != H0 or out.shape[3] != W0:
            out = out[:, :, :H0, :W0]

        gamma = torch.sigmoid(self.out_gamma)
        return residual + gamma * out
