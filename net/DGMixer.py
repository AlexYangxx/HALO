import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.arch_util import LayerNorm, CAMixer, GatedFeedForward  # 复用原有模块

# -------------------------- 1. 深度特征提取与精炼模块（轻量级净化depth） --------------------------
class DepthFeatureRefine(nn.Module):
    """
    轻量级深度特征提取+噪声精炼（净化depth再融合）
    目标：先净化depth噪声，再用于融合，提升融合上限
    """
    def __init__(self, in_dim=1, embed_dim=60):
        super().__init__()
        hidden_dim = max(embed_dim // 2, 1)
        suppress_dim = max(embed_dim // 4, 1)
        # 轻量级深度特征提取（小卷积+归一化）
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1),  # 提取基础特征
            LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, 3, 1, 1),  # 特征扩展
            LayerNorm(embed_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # 噪声抑制模块（空间注意力去噪）
        self.noise_suppress = nn.Sequential(
            nn.Conv2d(embed_dim, suppress_dim, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(suppress_dim, embed_dim, 1),
            nn.Sigmoid()  # 生成抑制权重
        )

    def forward(self, depth):
        """
        Args:
            depth: [B, 1, H, W] 原始深度图（可能有噪声）
        Returns:
            depth_feat: [B, embed_dim, H, W] 净化后的深度特征
        """
        # 提取深度特征
        depth_feat = self.depth_encoder(depth)  # [B, embed_dim, H, W]
        # 噪声抑制（空间自适应）
        suppress_weight = self.noise_suppress(depth_feat)  # [B, embed_dim, H, W]
        depth_feat = depth_feat * suppress_weight  # 抑制噪声区域
        return depth_feat

# -------------------------- 2. 动态跨模态门控融合模块（DCGF）- 优化版：空间+通道混合门控 --------------------------
class DynamicCrossModalGate(nn.Module):
    """
    动态门控融合图像特征与深度先验特征（空间+通道混合）
    目标：注入更准，保留空间结构信息
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        reduced_dim = max(dim // reduction, 1)
        
        # 通道门控分支（全局上下文）
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*dim, reduced_dim, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(reduced_dim, 2*dim, 1),
            nn.Sigmoid()
        )
        
        # 空间门控分支（保留结构信息）- 新增
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2*dim, reduced_dim, 3, 1, 1, groups=reduced_dim),  # 深度可分离卷积
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(reduced_dim, 2, 1),  # 输出2个空间权重图
            nn.Sigmoid()
        )
        
        # 融合权重（平衡通道和空间门控）
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 可学习平衡系数

    def forward(self, img_feat, depth_feat):
        # 强制尺寸对齐（兜底措施）
        if depth_feat.shape[2:] != img_feat.shape[2:]:
            depth_feat = F.interpolate(depth_feat, size=img_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # img_feat/depth_feat: [B, dim, H, W]
        concat_feat = torch.cat([img_feat, depth_feat], dim=1)  # [B, 2dim, H, W]
        
        # 通道门控（全局上下文）
        channel_weight = self.channel_gate(concat_feat)  # [B, 2dim, 1, 1]
        img_channel_gate, depth_channel_gate = torch.chunk(channel_weight, 2, dim=1)
        
        # 空间门控（空间结构）- 新增
        spatial_weight = self.spatial_gate(concat_feat)  # [B, 2, H, W]
        img_spatial_gate, depth_spatial_gate = spatial_weight.chunk(2, dim=1)  # 各 [B, 1, H, W]
        
        # 混合门控：通道 + 空间（可学习平衡）
        alpha = torch.sigmoid(self.fusion_weight)
        img_gate = alpha * img_channel_gate + (1 - alpha) * img_spatial_gate
        depth_gate = alpha * depth_channel_gate + (1 - alpha) * depth_spatial_gate
        
        # 动态融合：自适应调整图像/深度特征的贡献
        fused_feat = img_feat * img_gate + depth_feat * depth_gate
        return fused_feat

# -------------------------- 3. 深度引导的Block（修改原有Block，加入深度先验） --------------------------
class DepthGuidedBlock(nn.Module):
    def __init__(self, n_feats, window_size=8, ratio=0.5, device=""):
        super().__init__()
        self.n_feats = n_feats
        self.norm = LayerNorm(n_feats)
        # 原有CAMixer（保留核心逻辑）
        self.mixer = CAMixer(n_feats, window_size=window_size, ratio=ratio, device=device)
        self.ffn = GatedFeedForward(n_feats, device=device)
        # 深度引导的跨模态融合
        self.dcgf = DynamicCrossModalGate(n_feats)
        # 分层深度注意力（LDA）：调制Mixer输出特征
        self.depth_attention = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, x, depth_feat, condition_global=None):
        # 强制深度特征与x尺寸对齐
        if depth_feat.shape[2:] != x.shape[2:]:
            depth_feat = F.interpolate(depth_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        # Step1: 原有CAMixer前向
        if self.training:
            res, decision = self.mixer(x, condition_global)
        else:
            res, decision = self.mixer(x, condition_global)
        
        # Step2: 深度引导的特征融合（DCGF）
        res_fused = self.dcgf(res, depth_feat)
        
        # Step3: 分层深度注意力调制
        depth_attn = self.depth_attention(depth_feat)
        res_fused = res_fused * depth_attn + res  # 残差保留原有信息
        
        # Step4: 归一化+FFN（复用原有逻辑）
        x = self.norm(x + res_fused)
        res = self.ffn(x)
        x = self.norm(x + res)
        
        if self.training:
            return x, decision
        else:
            return x, decision

# -------------------------- 4. 深度先验指导的CAMixerSR（DG-CAMixerSR） --------------------------
class DepthGuidedCAMixerSR(nn.Module):
    def __init__(self, n_block=[1], n_group=4, in_channel=3, n_feats=60, ratio=0.5, window_sizes=16, device=""):
        super().__init__()
        self.ratio = ratio
        self.n_feats = n_feats
        self.window_sizes = window_sizes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_in_channels = in_channel
        
        # 1. 深度特征提取与精炼（恢复启用）- 方案3
        self.depth_extractor = DepthFeatureRefine(in_dim=in_channel, embed_dim=n_feats)
        
        # 2. 图像特征头（适配暗光图像）
        self.head = nn.Sequential(
            nn.Conv2d(in_channel, n_feats, 3, 1, 1),  # same padding
            LayerNorm(n_feats),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 3. 全局预测器（加入深度先验）
        self.global_predictor = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, 8, 1),  # 拼接图像+深度特征
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 2, 3, 1, 1),  # same padding
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 4. 深度引导的Body（替换原有Block为DepthGuidedBlock）
        self.body = nn.ModuleList([
            DepthGuidedBlock(n_feats, window_size=self.window_sizes, ratio=ratio, device=device) 
            for i in range(n_group)
        ])
        
        # 5. 尾部（保留原有结构，保证输出维度）
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)  # same padding
        self.tail = nn.Conv2d(n_feats, in_channel, 3, 1, 1)  # 输出RGB通道

    def check_image_size(self, x):
        """确保图像尺寸是window_sizes的整数倍，修复padding逻辑"""
        _, _, h, w = x.size()
        wsize = self.window_sizes
        # 计算需要padding的尺寸（向上取整到window_sizes的倍数）
        new_h = ((h + wsize - 1) // wsize) * wsize
        new_w = ((w + wsize - 1) // wsize) * wsize
        # 计算padding值（左右/上下对称padding）
        pad_h = new_h - h
        pad_w = new_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # 对称padding，避免尺寸偏移
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), 'reflect')
        return x, (pad_top, pad_bottom, pad_left, pad_right)  # 返回padding信息，用于后续裁剪

    def _match_depth_channels(self, depth_map):
        """
        Match depth channels to depth extractor expected channels.
        This avoids input-channel mismatch in DepthFeatureRefine.
        """
        c = depth_map.shape[1]
        target_c = self.depth_in_channels
        if c == target_c:
            return depth_map
        if c > target_c:
            return depth_map[:, :target_c, :, :]
        repeat_times = (target_c + c - 1) // c
        return depth_map.repeat(1, repeat_times, 1, 1)[:, :target_c, :, :]

    def forward(self, x, depth_map):
        """
        前向传播：输入暗光图像+深度图，输出增强图像
        Args:
            x: [B, 3, H, W] 暗光RGB图像
            depth_map: [B, 1, H, W] 深度图（归一化到[0,1]）
        Returns:
            output: [B, 3, H, W] 增强后图像
            decision: 注意力决策（训练/测试用）
        """
        B, C, H, W = x.shape
        
        # Step1: 尺寸对齐（返回padding信息，用于后续裁剪）
        x_padded, pad_info = self.check_image_size(x)
        depth_map_padded, _ = self.check_image_size(depth_map)  # 深度图用相同的padding
        depth_map_padded = self._match_depth_channels(depth_map_padded)
        
        # Step2: 提取图像特征和深度特征（恢复depth refine）
        img_feat = self.head(x_padded)  # [B, n_feats, H', W']
        depth_feat = self.depth_extractor(depth_map_padded)  # [B, n_feats, H', W'] → 净化后的深度特征
        
        # Step3: 全局条件预测（融合图像+深度特征）
        global_feat = torch.cat([img_feat, depth_feat], dim=1)  # 现在尺寸匹配
        condition_global = self.global_predictor(global_feat)
        
        # Step4: 深度引导的Body前向
        shortcut = img_feat
        if self.training:
            for blk in self.body:
                img_feat, _ = blk(img_feat, depth_feat, condition_global)
        else:
            decision = []
            for blk in self.body:
                img_feat, mask = blk(img_feat, depth_feat, condition_global)
                decision.extend(mask)
        
        # Step5: 残差连接+尾部输出
        img_feat = self.body_tail(img_feat) + shortcut
        output_padded = self.tail(img_feat)
        
        # Step6: 裁剪回原始尺寸（反向padding）
        pad_top, pad_bottom, pad_left, pad_right = pad_info
        output = output_padded[:, :, pad_top:H+pad_top, pad_left:W+pad_left]
        
        if self.training:
            return output, 2*self.ratio
        else:
            return output, decision

# -------------------------- 测试代码 --------------------------
if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = DepthGuidedCAMixerSR(
        n_block=[1], 
        n_group=4, 
        in_channel=3, 
        n_feats=60, 
        ratio=0.5, 
        window_sizes=16, 
        device=device
    ).to(device)
    
    # 构造测试输入（暗光图像+深度图）
    low_light_img = torch.randn(1, 3, 256, 256).to(device)  # 256×256（16的倍数）
    depth_map = torch.randn(1, 1, 256, 256).to(device)      # 256×256
    
    # 前向传播（无梯度计算）
    model.eval()
    with torch.no_grad():
        enhanced_img, _ = model(low_light_img, depth_map)
    
    # 打印尺寸信息
    print(f"输入暗光图像尺寸: {low_light_img.shape}")
    print(f"深度图尺寸: {depth_map.shape}")
    print(f"增强后图像尺寸: {enhanced_img.shape}")  # 应输出 torch.Size([1, 3, 256, 256])
