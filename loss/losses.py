from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *


_reduction_modes = ['none', 'mean', 'sum']

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
        
        
        
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.tensor([[0.05, 0.25, 0.4, 0.25, 0.05]], dtype=torch.float32)
        kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(0)
        # Keep gaussian kernel device-agnostic; module.to(device) will move it.
        self.register_buffer('kernel', kernel, persistent=False)

        self.weight = loss_weight

    def conv_gauss(self, img):
        c = img.shape[1]
        kernel = self.kernel.to(device=img.device, dtype=img.dtype).repeat(c, 1, 1, 1)
        _, _, kw, kh = kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, kernel, groups=c)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{self.criterion_type} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss




class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.to(device=img1.device, dtype=img1.dtype)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight








import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthEdgeConsistencyLoss(nn.Module):
    """
    Depth Edge Consistency Loss.

    Core idea: enforce the edge gradients of the enhanced image to align with
    the edge gradients of the depth map.

    Formula:
        L_edge = (1/|Omega|) * sum |grad(I_enh)(x) dot grad(d)(x) - gamma * ||grad(d)(x)||_2^2|

    Args:
        edge_consistency_coeff (float): Edge consistency coefficient gamma.
            Controls the strength of alignment between enhanced-image edges and
            depth edges. Default: 1.0.
        grad_method (str): Gradient operator, either 'sobel' (more robust) or
            'central_diff'. Default: 'sobel'.
        eps (float): Numerical stability term. Default: 1e-8.
    """
    def __init__(self, edge_consistency_coeff: float = 1.0, grad_method: str = 'sobel', eps: float = 1e-8):
        super().__init__()
        self.gamma = edge_consistency_coeff
        self.eps = eps
        self.grad_method = grad_method.lower()
        assert self.grad_method in ['sobel', 'central_diff'], "grad_method must be 'sobel' or 'central_diff'"
        
        # Initialize Sobel kernels (works for batch + channel dimensions).
        if self.grad_method == 'sobel':
            # Sobel kernels (x and y directions).
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
            self.register_buffer('sobel_x', sobel_x)  # Registered as buffer; not trainable.
            self.register_buffer('sobel_y', sobel_y)

    def _compute_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image/depth gradients (dx, dy).

        Input x shape: (batch, channels, H, W)
        Output dx, dy shape: (batch, channels, H, W)
        """
        if self.grad_method == 'sobel':
            # Expand Sobel kernels for multi-channel input (e.g., RGB).
            _, channels, _, _ = x.shape
            # Depthwise-conv weight shape should be [channels, 1, k, k].
            sobel_x = self.sobel_x.repeat(channels, 1, 1, 1)
            sobel_y = self.sobel_y.repeat(channels, 1, 1, 1)
            
            # Convolution-based gradient computation (padding=1 keeps size).
            dx = F.conv2d(x, sobel_x, padding=1, groups=channels)
            dy = F.conv2d(x, sobel_y, padding=1, groups=channels)
        
        else:  # central_diff
            dx = (x[..., :, 2:] - x[..., :, :-2]) / 2.0  # x-direction gradient
            dy = (x[..., 2:, :] - x[..., :-2, :]) / 2.0  # y-direction gradient
            # Pad borders to keep output size identical to input.
            dx = F.pad(dx, (1, 1, 0, 0), mode='replicate')
            dy = F.pad(dy, (0, 0, 1, 1), mode='replicate')
        
        return dx, dy

    def forward(self, enhanced_img: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Compute loss in forward pass.

        Args:
            enhanced_img (torch.Tensor): Enhanced image, shape (batch, 3/1, H, W),
                value range [0, 1].
            depth_map (torch.Tensor): Depth map from Depth Anything 3, shape
                (batch, 1, H, W), preferably normalized to [0, 1].

        Returns:
            loss (torch.Tensor): Scalar loss value.
        """
        # 1. Validate input dimensions.
        assert enhanced_img.dim() == 4 and depth_map.dim() == 4, "Input must be a 4D tensor (batch, channels, H, W)"
        assert depth_map.shape[1] == 1, "Depth map must have exactly 1 channel"
        assert enhanced_img.shape[2:] == depth_map.shape[2:], "Enhanced image and depth map must have the same spatial size"
        
        # 2. Compute gradients of enhanced image (average RGB gradients if color image).
        img_dx, img_dy = self._compute_gradient(enhanced_img)
        if enhanced_img.shape[1] == 3:
            img_dx = torch.mean(img_dx, dim=1, keepdim=True)  # (batch, 1, H, W)
            img_dy = torch.mean(img_dy, dim=1, keepdim=True)
        
        # 3. Compute depth-map gradients.
        depth_dx, depth_dy = self._compute_gradient(depth_map)  # (batch, 1, H, W)
        
        # 4. Gradient dot product: grad(I_enh) dot grad(d) = img_dx*depth_dx + img_dy*depth_dy.
        grad_dot = img_dx * depth_dx + img_dy * depth_dy
        
        # 5. Squared L2 norm of depth gradients: ||grad(d)||_2^2 = depth_dx^2 + depth_dy^2.
        depth_grad_norm_sq = depth_dx.pow(2) + depth_dy.pow(2)
        
        # 6. Loss term: |grad_dot - gamma * depth_grad_norm_sq|.
        loss_term = torch.abs(grad_dot - self.gamma * depth_grad_norm_sq)
        
        # 7. Global mean over all pixels.
        loss = torch.mean(loss_term)
        
        return loss


class DepthRegionSmoothLoss(nn.Module):
    """
    Depth Region Smoothness Loss.

    Core idea: enforce pixel smoothness for enhanced-image regions that share
    similar depth values.

    Formula:
        L_smooth = (1/|Omega|) * sum sum |I_enh(x)-I_enh(y)| * exp(-|d(x)-d(y)|/epsilon)

    Args:
        smooth_eps (float): Smoothness coefficient epsilon controlling tolerance to
            depth differences. Default: 0.1.
        kernel_size (int): Neighborhood window size. Default: 3 (8-neighbors).
            Must be odd.
        eps (float): Numerical stability term. Default: 1e-8.
    """
    def __init__(self, smooth_eps: float = 0.1, kernel_size: int = 3, eps: float = 1e-8):
        super().__init__()
        self.smooth_eps = smooth_eps
        self.kernel_size = kernel_size
        self.eps = eps
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.padding = (kernel_size - 1) // 2  # Padding that keeps spatial size unchanged.
        
        # Initialize neighborhood mask: center=0, neighbors=1.
        kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32)
        kernel[:, :, kernel_size//2, kernel_size//2] = 0.0  # Exclude center pixel from neighborhood.
        self.register_buffer('neighbor_mask', kernel)

    def forward(self, enhanced_img: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Compute loss in forward pass.

        Args:
            enhanced_img (torch.Tensor): Enhanced image, shape (batch, 3/1, H, W),
                value range [0, 1].
            depth_map (torch.Tensor): Depth map from Depth Anything 3, shape
                (batch, 1, H, W), preferably normalized to [0, 1].

        Returns:
            loss (torch.Tensor): Scalar loss value.
        """
        # 1. Validate input dimensions.
        assert enhanced_img.dim() == 4 and depth_map.dim() == 4, "Input must be a 4D tensor (batch, channels, H, W)"
        assert depth_map.shape[1] == 1, "Depth map must have exactly 1 channel"
        assert enhanced_img.shape[2:] == depth_map.shape[2:], "Enhanced image and depth map must have the same spatial size"
        
        batch, channels, H, W = enhanced_img.shape
        
        # 2. For color images, compute smoothness per channel then average.
        if channels > 1:
            loss_per_channel = []
            for c in range(channels):
                img_single = enhanced_img[:, c:c+1, :, :]  # (batch, 1, H, W)
                loss_c = self._compute_single_channel_loss(img_single, depth_map)
                loss_per_channel.append(loss_c)
            loss = torch.mean(torch.stack(loss_per_channel))
        else:
            loss = self._compute_single_channel_loss(enhanced_img, depth_map)
        
        return loss

    def _compute_single_channel_loss(self, img_single: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Compute depth-region smoothness loss for a single-channel image.
        """
        # 1. Absolute intensity difference with neighbors: |I_enh(x) - I_enh(y)|.
        # Convolution gives neighbor-value sum at each location.
        img_neighbor_sum = F.conv2d(img_single, self.neighbor_mask, padding=self.padding)
        # Number of neighbor pixels (8 for 3x3, 24 for 5x5).
        neighbor_count = self.neighbor_mask.sum()
        # Expand center pixel to neighborhood count and compute absolute difference.
        img_center_expand = img_single * neighbor_count
        img_diff_abs = torch.abs(img_neighbor_sum - img_center_expand)  # (batch, 1, H, W)
        
        # 2. Absolute depth difference with neighbors: |d(x) - d(y)|.
        depth_neighbor_sum = F.conv2d(depth_map, self.neighbor_mask, padding=self.padding)
        depth_center_expand = depth_map * neighbor_count
        depth_diff_abs = torch.abs(depth_neighbor_sum - depth_center_expand)  # (batch, 1, H, W)
        
        # 3. Exponential weighting term: exp(-|d(x)-d(y)|/epsilon).
        exp_term = torch.exp(-depth_diff_abs / (self.smooth_eps + self.eps))
        
        # 4. Loss term: |I_enh(x)-I_enh(y)| * exp_term.
        loss_term = img_diff_abs * exp_term
        
        # 5. Global mean over all pixels.
        loss = torch.mean(loss_term)
        
        return loss


# ------------------------------ Test Cases ------------------------------
if __name__ == "__main__":
    # 1. Create mock inputs.
    batch_size = 2
    H, W = 256, 256
    # Enhanced image (RGB, value range [0, 1]).
    enhanced_img = torch.rand(batch_size, 3, H, W)
    # Depth map (single channel, normalized to [0, 1]).
    depth_map = torch.rand(batch_size, 1, H, W)
    
    # 2. Initialize loss functions.
    edge_loss_fn = DepthEdgeConsistencyLoss(edge_consistency_coeff=1.0, grad_method='sobel')
    smooth_loss_fn = DepthRegionSmoothLoss(smooth_eps=0.1, kernel_size=3)
    
    # 3. Compute losses.
    edge_loss = edge_loss_fn(enhanced_img, depth_map)
    smooth_loss = smooth_loss_fn(enhanced_img, depth_map)
    
    # 4. Print results.
    print(f"Depth edge consistency loss: {edge_loss.item():.6f}")
    print(f"Depth region smoothness loss: {smooth_loss.item():.6f}")
    
    # 5. Verify backpropagation (loss should be differentiable).
    edge_loss.backward()
    smooth_loss.backward()
    print("Backpropagation passed: gradients are valid.")
