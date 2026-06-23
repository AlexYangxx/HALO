import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from net.LCA import *
from net.DGMixer import DepthGuidedCAMixerSR
from net.freq_blocks import FreqResidualStack
from net.prior_fusion import SemanticBottleneckFusion, SemanticPriorProjector
from net.dp_caa import WindowedDPCAA


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        start = int(step * i)
        inputs[:, i, :, :out_col] = inputs[:, i, :, start:start + out_col]
    return inputs[:, :, :, :out_col]

class MaskGuidedMechanism(nn.Module):
    def __init__(
            self, n_feat):
        super(MaskGuidedMechanism, self).__init__()

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        # x: b,c,h,w
        [bs, nC, row, col] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_emb = res + mask_shift
        return mask_emb


class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

# -------------------------- 方案2+3: Skip/Bottleneck 门控融合模块 --------------------------
class AdaptiveFusion(nn.Module):
    """
    自适应融合模块（用于Encoder skip连接）
    替换硬相加 fea_img + fea_depth，改为可学习权重融合
    参数量: dim * 2 (每个stage)
    """
    def __init__(self, dim):
        super().__init__()
        reduced_dim = max(dim // 4, 1)
        # 轻量级门控：提取融合权重
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, reduced_dim, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(reduced_dim, 2, 1),
            nn.Softmax(dim=1)  # [B, 2, H, W] - 空间自适应权重
        )
    
    def forward(self, img_feat, depth_feat):
        """
        Args:
            img_feat: [B, dim, H, W]
            depth_feat: [B, dim, H, W]
        Returns:
            fused: [B, dim, H, W]
        """
        concat = torch.cat([img_feat, depth_feat], dim=1)
        gate_weights = self.gate(concat)  # [B, 2, H, W]
        img_gate, depth_gate = gate_weights.chunk(2, dim=1)  # 各 [B, 1, H, W]
        fused = img_feat * img_gate + depth_feat * depth_gate
        return fused

class BottleneckGateFusion(nn.Module):
    """
    Bottleneck门控融合模块
    替换硬拼接 Concat([fea_img, fea_depth])，改为门控注入
    参数量: dim * dim * 2 (vs 2*dim * dim * 9，减少约89%)
    """
    def __init__(self, dim):
        super().__init__()
        reduced_dim = max(dim // 4, 1)
        # 分离处理图像/深度特征
        self.img_proj = nn.Conv2d(dim, dim, 1)
        self.depth_proj = nn.Conv2d(dim, dim, 1)
        # 门控权重生成（空间感知）
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, reduced_dim, 3, 1, 1),  # 3x3保留空间信息
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(reduced_dim, 2, 1),
            nn.Softmax(dim=1)
        )
        # 融合投影
        self.fusion_proj = nn.Conv2d(dim, dim, 1)
    
    def forward(self, img_feat, depth_feat):
        """
        Args:
            img_feat: [B, dim, H, W]
            depth_feat: [B, dim, H, W]
        Returns:
            fused: [B, dim, H, W]
        """
        img_proj = self.img_proj(img_feat)
        depth_proj = self.depth_proj(depth_feat)
        # 门控权重（空间自适应）
        concat = torch.cat([img_proj, depth_proj], dim=1)
        gate_weights = self.gate(concat)  # [B, 2, H, W]
        img_gate, depth_gate = gate_weights.chunk(2, dim=1)
        # 门控融合
        fused = img_proj * img_gate + depth_proj * depth_gate
        return self.fusion_proj(fused)

class MST(nn.Module):
    def __init__(
        self,
        in_dim=30,
        out_dim=30,
        dim=30,
        stage=3,
        num_blocks=[2, 4, 4, 4],
        use_freq_branch=False,
        freq_inject_pos="bottleneck",
        fft_mode="both",
        freq_weight=0.2,
        freq_blocks=1,
        use_semantic_prior=False,
        dino_sem_channels=1024,
        semantic_fusion_weight=0.1,
        use_dp_caa=False,
        dp_caa_window=8,
        dp_caa_sem_embed=32,
        dp_caa_tau=0.07,
    ):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage
        bottleneck_c = dim * (2 ** stage)
        self.use_freq_branch = bool(use_freq_branch) and freq_inject_pos == "bottleneck"
        self.freq_weight = float(freq_weight)

        # Input projection
        self.embedding_img = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.embedding_depth = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        self.adaptive_fusions = nn.ModuleList([])  # 方案2: Skip连接自适应融合
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                Dual_LCA(dim=dim_stage, num_heads=dim_stage // dim),
                DepthGuidedCAMixerSR(n_block=[1], n_group=1, in_channel=dim_stage, n_feats=dim_stage, ratio=0.5, window_sizes=16, device='cuda'),
                # MSAB(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            self.adaptive_fusions.append(AdaptiveFusion(dim_stage))  # 每个stage一个自适应融合
            dim_stage *= 2

        # Exp8R: keep high-capacity main fusion path and add gated residual branch (并联残差).
        # Main path: 3x3 conv over concatenated img/depth features.
        self.bottleneck_main = nn.Conv2d(dim_stage * 2, dim_stage, 3, 1, 1, bias=False)
        # Gated residual path.
        self.bottleneck_fusion = BottleneckGateFusion(dim_stage)
        # Learnable residual strengths, constrained to [0, 1] by sigmoid in forward.
        init_alpha = torch.logit(torch.tensor(0.2), eps=1e-6)
        self.skip_gate_alphas = nn.ParameterList(
            [nn.Parameter(init_alpha.clone()) for _ in range(stage)]
        )
        self.bottleneck_gate_alpha = nn.Parameter(init_alpha.clone())
        self.fusion = MSAB(dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        self.freq_stack = None
        if self.use_freq_branch:
            self.freq_stack = FreqResidualStack(
                bottleneck_c, fft_mode=fft_mode, num_blocks=int(freq_blocks)
            )

        self.sem_proj = None
        self.sem_mix = None
        self.sem_weight = float(semantic_fusion_weight)
        if use_semantic_prior:
            self.sem_proj = SemanticPriorProjector(int(dino_sem_channels), bottleneck_c)
            self.sem_mix = SemanticBottleneckFusion(bottleneck_c)

        self.dp_caa = None
        if use_dp_caa:
            heads = max(1, bottleneck_c // dim)
            dim_head = dim
            self.dp_caa = WindowedDPCAA(
                dim=bottleneck_c,
                dim_head=dim_head,
                heads=heads,
                window_size=int(dp_caa_window),
                sem_in_channels=int(dino_sem_channels),
                sem_embed_dim=int(dp_caa_sem_embed),
                tau=float(dp_caa_tau),
            )

        # self.bottleneck = MSAB(
        #     dim=dim_stage*2, dim_head=dim*2, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, depth, sem_feat=None, sem_scale: float = 1.0):
        """
        x: [b,c,h,w]
        sem_feat: optional [b,C_sem,H,W] aligned spatially with x (RGB embedding space); ignored if module not built.
        return out:[b,c,h,w]
        """

        # Embedding
        fea_img = self.embedding_img(x)
        fea_depth = self.embedding_depth(depth)

        # Encoder
        fea_encoder = []
        for i, (Dual_LCA, Fusion, FeaDownSample_img, FeaDownSample_depth) in enumerate(self.encoder_layers):
            fea_img, fea_depth = Dual_LCA(fea_img, fea_depth)
            fea_img = Fusion(fea_img, fea_depth)[0]
            # Exp8R: parallel residual injection for skip fusion (主路径+门控残差).
            skip_main = fea_img + fea_depth
            skip_gate = self.adaptive_fusions[i](fea_img, fea_depth)
            skip_alpha = torch.sigmoid(self.skip_gate_alphas[i])
            fea_encoder.append(skip_main + skip_alpha * skip_gate)
            fea_img = FeaDownSample_img(fea_img)
            fea_depth = FeaDownSample_depth(fea_depth)

        # Exp8R: main bottleneck path + gated residual branch (并联残差).
        fea_main = self.bottleneck_main(torch.cat([fea_img, fea_depth], dim=1))
        fea_gate = self.bottleneck_fusion(fea_img, fea_depth)
        bottleneck_alpha = torch.sigmoid(self.bottleneck_gate_alpha)
        fea = fea_main + bottleneck_alpha * fea_gate
        fea = self.fusion(fea)
        if self.use_freq_branch and self.freq_stack is not None:
            fea = fea + self.freq_weight * self.freq_stack(fea)
        sem_gain = float(sem_scale) * self.sem_weight
        if sem_gain > 0.0 and self.sem_proj is not None and sem_feat is not None and sem_feat.numel() > 0:
            # sem_feat may already be pre-resized to bottleneck grid for speed.
            sem = sem_feat
            if sem.shape[2:] != fea.shape[2:]:
                sem = F.interpolate(sem, size=fea.shape[2:], mode="bilinear", align_corners=False)
            sp = self.sem_proj(sem)
            fea = fea + sem_gain * self.sem_mix(fea, sp)

        if self.dp_caa is not None:
            sem_for_dpcaa = sem_feat if (sem_feat is not None and sem_feat.numel() > 0) else None
            fea = self.dp_caa(fea, fea_depth, sem_for_dpcaa, sem_scale=float(sem_scale))

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class MST_Plus_Plus(nn.Module):
    """Stack of MST bodies. `stage` only controls how many bodies are chained; internal MST depth stays 3."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=16,
        stage=3,
        use_freq_branch=False,
        freq_inject_pos="bottleneck",
        fft_mode="both",
        freq_weight=0.2,
        freq_blocks=1,
        use_semantic_prior=False,
        dino_sem_channels=1024,
        semantic_fusion_weight=0.1,
        use_dp_caa=False,
        dp_caa_window=8,
        dp_caa_sem_embed=32,
        dp_caa_tau=0.07,
    ):
        super(MST_Plus_Plus, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.depth_embedding = nn.Conv2d(1, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        # modules_body = [MST(dim=32, stage=2, num_blocks=[1,1,1]) for _ in range(stage)]
        # self.body = nn.Sequential(*modules_body)
        # DP-CAA only on the last MST body (same bottleneck grid; avoids ~3× redundant compute).
        self.body = nn.ModuleList(
            [
                MST(
                    in_dim=16,
                    out_dim=16,
                    dim=16,
                    stage=3,
                    num_blocks=[1, 1, 1, 1],
                    use_freq_branch=use_freq_branch,
                    freq_inject_pos=freq_inject_pos,
                    fft_mode=fft_mode,
                    freq_weight=freq_weight,
                    freq_blocks=freq_blocks,
                    use_semantic_prior=use_semantic_prior,
                    dino_sem_channels=dino_sem_channels,
                    semantic_fusion_weight=semantic_fusion_weight,
                    use_dp_caa=bool(use_dp_caa) and (i == stage - 1),
                    dp_caa_window=dp_caa_window,
                    dp_caa_sem_embed=dp_caa_sem_embed,
                    dp_caa_tau=dp_caa_tau,
                )
                for i in range(stage)
            ]
        )
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, input, depth, sem_feat=None, sem_scale: float = 1.0):
        """
        x: [b,c,h,w]
        sem_feat: optional [b,C,H,W] same spatial as input (before body); padded with reflect like RGB.
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = input.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        input = F.pad(input, [0, pad_w, 0, pad_h], mode='reflect')
        depth = F.pad(depth, [0, pad_w, 0, pad_h], mode='reflect')
        if sem_feat is not None and sem_feat.numel() > 0:
            sem_feat = F.pad(sem_feat, [0, pad_w, 0, pad_h], mode='reflect')
            # Pre-resize semantic grid to bottleneck resolution once and reuse across MST submodules.
            # stage=3 => spatial downsample factor = 2**3 = 8
            h_pad, w_pad = int(h_inp + pad_h), int(w_inp + pad_w)
            sem_feat = F.interpolate(
                sem_feat, size=(h_pad // 8, w_pad // 8), mode="bilinear", align_corners=False
            )
        x = self.conv_in(input)
        depth_1 = self.depth_embedding(depth)
        for mst_module in self.body:
            x = mst_module(x, depth_1, sem_feat, sem_scale=sem_scale)
        # h = x + input
        h = self.conv_out(x)
        h += input
        # h += input
        
        return h[:, :, :h_inp, :w_inp]
        
        






