
import os
from PIL import Image
import numpy as np
import torch

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def hw_and_dtype_for_depth_placeholder(im1):
    """从 CHW Tensor 或 PIL 得到 (h, w, dtype)，用于无深度数据集的零深度占位。"""
    if isinstance(im1, torch.Tensor):
        h, w = int(im1.shape[-2]), int(im1.shape[-1])
        return h, w, im1.dtype
    w, h = im1.size
    return int(h), int(w), torch.float32



def load_depth_npz(filepath):
    """Load depth from npz and normalize to [0,1] with a fast-path for pre-normalized maps."""
    with np.load(filepath) as depth_data:
        if 'depth' in depth_data.keys():
            depth = depth_data['depth']
        elif 'arr_0' in depth_data.keys():
            depth = depth_data['arr_0']
        else:
            depth = list(depth_data.values())[0]

    depth = np.asarray(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    d_min = float(depth.min())
    d_max = float(depth.max())

    # Fast path: many cached depth maps are already normalized.
    if d_min >= 0.0 and d_max <= 1.0:
        return Image.fromarray(depth.astype(np.float32), mode='F')

    use_percentile = os.getenv('DEPTH_PERCENTILE_NORM', '1').lower() not in ('0', 'false', 'no')

    if use_percentile:
        valid_mask = depth > 0
        valid_depth = depth[valid_mask] if valid_mask.any() else depth.reshape(-1)
        lo, hi = np.percentile(valid_depth, [2, 98])
    else:
        lo, hi = d_min, d_max

    if hi - lo < 1e-8:
        lo, hi = d_min, d_max

    if hi - lo < 1e-8:
        depth = np.zeros_like(depth, dtype=np.float32)
    else:
        depth = np.clip(depth, lo, hi)
        depth = (depth - lo) / (hi - lo + 1e-8)

    return Image.fromarray(depth.astype(np.float32), mode='F')
