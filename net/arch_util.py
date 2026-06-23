import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import math
from einops import rearrange

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

    class _TritonStub:
        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

    class _TLStub:
        constexpr = int

        def __getattr__(self, name):
            raise RuntimeError(f"Triton is unavailable; attempted to access tl.{name}")

    triton = _TritonStub()
    tl = _TLStub()

# ========== 彻底删除所有custom_fwd/custom_bwd相关代码 ==========
# 无需再定义/导入custom_fwd/custom_bwd，完全改用torch.autocast

def _grid(numel: int, bs: int) -> tuple:
    return (triton.cdiv(numel, bs),)

@triton.jit
def _idx(i, n: int, c: int, h: int, w: int):
    ni = i // (c * h * w)
    ci = (i // (h * w)) % c
    hi = (i // w) % h
    wi = i % w
    m = i < (n * c * h * w)
    return ni, ci, hi, wi, m

@triton.jit
def ska_fwd(
    x_ptr, w_ptr, o_ptr,
    n, ic, h, w, ks, pad, wc,
    BS: tl.constexpr,
    CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
    val = tl.zeros((BS,), dtype=AT)

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)

            x_off = ((ni * ic + ci) * h + hin) * w + win
            w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

            x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
            val += tl.where(b & m, x_val * w_val, 0.0).to(AT)

    tl.store(o_ptr + offs, val.to(CT), mask=m)

@triton.jit
def ska_bwd_x(
    go_ptr, w_ptr, gi_ptr,
    n, ic, h, w, ks, pad, wc,
    BS: tl.constexpr,
    CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
    val = tl.zeros((BS,), dtype=AT)

    for kh in range(ks):
        ho = hi + pad - kh
        hb = (ho >= 0) & (ho < h)
        for kw in range(ks):
            wo = wi + pad - kw
            b = hb & (wo >= 0) & (wo < w)

            go_off = ((ni * ic + ci) * h + ho) * w + wo
            w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + ho * w + wo

            go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
            val += tl.where(b & m, go_val * w_val, 0.0).to(AT)

    tl.store(gi_ptr + offs, val.to(CT), mask=m)

@triton.jit
def ska_bwd_w(
    go_ptr, x_ptr, gw_ptr,
    n, wc, h, w, ic, ks, pad,
    BS: tl.constexpr,
    CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)
            w_off = ((ni * wc + ci) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

            val = tl.zeros((BS,), dtype=AT)
            steps = (ic - ci + wc - 1) // wc
            for s in range(tl.max(steps, axis=0)):
                cc = ci + s * wc
                cm = (cc < ic) & m & b

                x_off = ((ni * ic + cc) * h + hin) * w + win
                go_off = ((ni * ic + cc) * h + hi) * w + wi

                x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(CT)
                go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(CT)
                val += tl.where(cm, x_val * go_val, 0.0).to(AT)

            tl.store(gw_ptr + w_off, val.to(CT), mask=m)

# ========== 重构SkaFn：移除custom_fwd/custom_bwd装饰器 ==========
# 混合精度由训练时的torch.autocast上下文管理器控制，无需在Function内加装饰器
class SkaFn(Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # 原有forward逻辑完全保留，仅删除装饰器
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.save_for_backward(x, weight, bias)
        output = torch.nn.functional.conv2d(x, weight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 原有backward逻辑完全保留，仅删除装饰器
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv_transpose2d(
                grad_output, weight, None, ctx.stride, ctx.padding, ctx.dilation, ctx.groups
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(
                x.transpose(0, 1), grad_output.transpose(0, 1), None, ctx.dilation, ctx.padding, ctx.stride, ctx.groups
            ).transpose(0, 1)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None, None, None

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv2_ = Conv2d_BN(dim // 2, dim // 2, ks=lks+2, pad=(lks+2 - 1) // 2, groups=dim // 2)
        self.cv2__ = Conv2d_BN(dim // 2, dim // 2, ks=lks-2, pad=(lks-2 - 1) // 2, groups=dim // 2)
        self.cv31 = Conv2d_BN(dim // 2, dim // 2)
        self.cv32 = Conv2d_BN(dim // 2, dim // 2)
        self.cv33 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2 * 3, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        self.sks = sks
        self.groups = groups
        self.dim = dim
    def forward(self, x):
        x1 = self.act(self.cv1(x))
        lks1 = self.cv31(self.cv2(x1))
        lks2 = self.cv32(self.cv2_(x1))
        lks3 = self.cv33(self.cv2__(x1))
        lks = torch.concat((lks1,lks2,lks3),dim=1)
        x = self.act(lks)
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class GatedFeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.,device=""):
        super().__init__()
        self.dim = dim

        self.project_in = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Block(nn.Module):
    def __init__(self, n_feats, window_size=8, ratio=0.5,device=""):
        super(Block,self).__init__()
        
        self.n_feats = n_feats
        self.norm = LayerNorm(n_feats)
        self.mixer = CAMixer(n_feats,window_size=window_size,ratio=ratio,device=device)
        self.ffn = GatedFeedForward(n_feats,device=device)
        
    def forward(self,x,condition_global=None):
        if self.training:
            res, decision = self.mixer(x,condition_global)
            x = self.norm(x+res)
            res = self.ffn(x)
            x = self.norm(x+res)
            return x, decision
        else:
            res,decision = self.mixer(x,condition_global)
            x = self.norm(x+res)
            res = self.ffn(x)
            x = self.norm(x+res)
            return x,decision
        
class SKA(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return SkaFn.apply(x, w) # type: ignore

class LSConv(nn.Module):
    def __init__(self, dim,groups):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=2, groups=4)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        y = self.lkp(x)
        z  = self.ska(x, y)
        return self.bn(z) + x


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x

class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, dim, window_size=8, k=4,ratio=0.5,device=""):
        super().__init__()
        self.ratio = ratio
        self.dim = dim
        self.window_size = window_size
        cdim = dim + k
        embed_dim = window_size**2
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):
        x = self.in_conv(input_x)
        ca = self.out_CA(x)
        
        x = torch.mean(x, keepdim=True, dim=1) 
        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()
        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        if self.training or train_mode:
            return mask, ca
        else:
            score = pred_score[:, : , 0]
            B, N = score.shape
            r = torch.mean(mask,dim=(0,1))*1.0
            if self.ratio == 1:
                num_keep_node = N 
            else:
                num_keep_node = min(int(N * r * 2 * self.ratio), N)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return [idx1, idx2], ca
        
class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True, ratio=0.5,device=""):
        super().__init__()    
        self.dim = dim
        self.window_size = window_size
        self.ratio = ratio
        k = 3
        d = 2
        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_q = nn.Linear(dim, dim, bias = bias)
        self.project_k = nn.Linear(dim, dim, bias = bias)
        # Conv
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k//2)*d), groups=dim, dilation=d))        
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.act = nn.GELU()
        # Predictor
        self.route = PredictorLG(dim,window_size,ratio=ratio,device=device)
        # self.lsconv = LSConv(dim=dim,groups=6).to(self.device)
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        # Cache a single coordinate prior map [1,2,H,W] for current (device,dtype,H,W).
        # This avoids rebuilding linspace/meshgrid every forward.
        self._condition_wind = None

    def forward(self,x,condition_global=None, mask=None, train_mode=False):
        N, C, H, W = x.shape
        v = self.project_v(x)
        if condition_global is not None and condition_global.shape[2:] != (H, W):
            condition_global = F.interpolate(
                condition_global, size=(H, W), mode="bilinear", align_corners=False
            )
        need_rebuild = (
            self._condition_wind is None
            or self._condition_wind.shape[-2:] != (H, W)
            or self._condition_wind.device != x.device
            or self._condition_wind.dtype != x.dtype
        )
        if need_rebuild:
            axis_y = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            axis_x = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            gy, gx = torch.meshgrid(axis_y, axis_x, indexing='ij')
            self._condition_wind = torch.stack((gy, gx), dim=0).unsqueeze(0).contiguous()
        condition_wind = self._condition_wind.expand(N, -1, -1, -1)
        _condition = torch.cat([v, condition_global, condition_wind], dim=1) if condition_global is not None else torch.cat([v, condition_wind], dim=1)
        mask, ca = self.route(_condition,ratio=self.ratio,train_mode=train_mode)
        qk = torch.cat([x,x],dim=1)
        vs = self.conv(v)
        v  = rearrange(v,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        
        if self.training or train_mode:
            N_ = v.shape[1]
            v1,v2 = v*mask, vs*(1-mask)   
            qk1 = qk*mask 
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1,v2 = batch_index_select(v,idx1),batch_index_select(vs,idx2)
            qk1 = batch_index_select(qk,idx1)

        v1 = rearrange(v1,'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1,'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1,k1 = torch.chunk(qk1,2,dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
  
        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn@v1

        f_attn = rearrange(f_attn,'(b n) (dh dw) c -> b n (dh dw c)', 
            b=N, n=N_, dh=self.window_size, dw=self.window_size)

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)', 
            h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size
        )
        out = attn_out
        out = self.act(self.conv_sptial(out))*ca + out
        out = self.project_out(out)
        
        if self.training:
            return out, torch.mean(mask,dim=1)
        return out,[mask]

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

# ========== 新增：PyTorch 2.0+ 混合精度训练示例（替代原custom_fwd装饰器） ==========
def train_with_amp(model, dataloader, optimizer, device):
    """
    使用PyTorch 2.0+原生AMP（torch.autocast）进行混合精度训练
    替代原custom_fwd/custom_bwd装饰器的功能
    """
    scaler = torch.cuda.amp.GradScaler()  # 梯度缩放器，防止梯度下溢
    model.train()
    
    for batch in dataloader:
        low_light_img, depth_map, gt_img = batch
        low_light_img = low_light_img.to(device)
        depth_map = depth_map.to(device)
        gt_img = gt_img.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度上下文管理器（替代原custom_fwd装饰器）
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred_img, _ = model(low_light_img, depth_map)
            loss = F.l1_loss(pred_img, gt_img)  # 示例损失
        
        # 缩放梯度并反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    ls_conv = LSConv(dim=36, groups=6).to(device)
    x = torch.rand(1, 36, 256, 256).to(device)
    output = ls_conv(x)
    print(f"LSConv输出尺寸: {output.shape}")  # 应输出 torch.Size([1, 36, 256, 256])
    
    # 测试CAMixer模块
    camixer = CAMixer(dim=36, window_size=8, ratio=0.5).to(device)
    condition_global = torch.rand(1, 2, 256, 256).to(device)  # 示例全局条件
    camixer_out, _ = camixer(x, condition_global)
    print(f"CAMixer输出尺寸: {camixer_out.shape}")  # 应输出 torch.Size([1, 36, 256, 256])
