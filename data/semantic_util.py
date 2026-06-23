"""
Stage B: 加载离线 DINO 特征 *.npy，并与 RGB/深度做空间对齐的随机增强（与 RandomCrop/翻转一致）。
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import ToTensor

_CACHE_DIR_RESOLVE_CACHE = {}
_CACHE_DIR_WARNED = set()


def _resolve_cache_dir(cache_dir: str) -> str:
    """
    Resolve semantic cache directory robustly.

    Typical failure case on Linux:
      user passes "/cache_dinov3/xxx" but real dir is "./cache_dinov3/xxx"
      under current project root.
    """
    raw = str(cache_dir or "").strip()
    if not raw:
        return raw

    cached = _CACHE_DIR_RESOLVE_CACHE.get(raw)
    if cached is not None:
        return cached

    expanded = os.path.expanduser(raw)
    candidates = [expanded]

    # Auto-fallback for mistaken leading "/" absolute-like path.
    if expanded.startswith(os.sep):
        candidates.append(os.path.join(os.getcwd(), expanded.lstrip("/\\")))

    # Also try normalized absolute path for relative inputs.
    candidates.append(os.path.abspath(expanded))

    for cand in candidates:
        if os.path.isdir(cand):
            resolved = cand
            if resolved != raw and raw not in _CACHE_DIR_WARNED:
                print(f"[SemanticCache] auto-resolved cache dir: {raw} -> {resolved}")
                _CACHE_DIR_WARNED.add(raw)
            _CACHE_DIR_RESOLVE_CACHE[raw] = resolved
            return resolved

    _CACHE_DIR_RESOLVE_CACHE[raw] = expanded
    return expanded



def load_semantic_npy(cache_dir: str, image_filename: str) -> torch.Tensor:
    """Load semantic cache by image stem, return float32 CHW tensor."""
    if not cache_dir:
        raise ValueError("semantic cache_dir is empty")
    cache_dir = _resolve_cache_dir(cache_dir)
    stem, _ = os.path.splitext(image_filename)
    path = os.path.join(cache_dir, f"{stem}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Semantic cache not found: {path}")

    arr = np.load(path, allow_pickle=False)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    t = torch.from_numpy(np.ascontiguousarray(arr))

    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3:
        raise ValueError(f"Expected CHW semantic map, got shape {tuple(t.shape)} in {path}")
    return t


def resize_sem_to_image_hw(sem: torch.Tensor, pil_img) -> torch.Tensor:
    """Resize semantic CHW map to target PIL image HxW using bilinear interpolation."""
    h = pil_img.size[1]
    w = pil_img.size[0]
    if sem.shape[1] == h and sem.shape[2] == w:
        return sem
    return F.interpolate(sem.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)


def _crop_sem_by_image_window(
    sem: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """Crop semantic map by mapping image-space crop window to semantic-space coordinates.

    For border crops, round+clamp can create off-by-one semantic windows
    (e.g., 7x6 vs 7x7) and break batch stacking. Keep output size stable.
    """
    _, hs, ws = sem.shape
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"Invalid image size: H={img_h}, W={img_w}")

    target_h = max(1, int(round(height * hs / float(img_h))))
    target_w = max(1, int(round(width * ws / float(img_w))))

    top_s = int(round(top * hs / float(img_h)))
    left_s = int(round(left * ws / float(img_w)))
    bottom_s = int(round((top + height) * hs / float(img_h)))
    right_s = int(round((left + width) * ws / float(img_w)))

    top_s = max(0, min(top_s, hs - 1))
    left_s = max(0, min(left_s, ws - 1))
    bottom_s = max(top_s + 1, min(bottom_s, hs))
    right_s = max(left_s + 1, min(right_s, ws))
    cropped = sem[:, top_s:bottom_s, left_s:right_s]
    cur_h, cur_w = int(cropped.shape[1]), int(cropped.shape[2])

    if cur_h < target_h or cur_w < target_w:
        pad_h = max(0, target_h - cur_h)
        pad_w = max(0, target_w - cur_w)
        cropped = F.pad(cropped, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    if cropped.shape[1] > target_h or cropped.shape[2] > target_w:
        cropped = cropped[:, :target_h, :target_w]

    return cropped

def augment_training_pair_with_sem(
    im1,
    im2,
    depth_low,
    depth_high,
    sem: torch.Tensor,
    crop_size: int,
):
    """
    Synchronized RandomCrop + h/v flip + ToTensor for RGB/depth and semantic map.
    Optimization: crop semantic map in its native low-res grid first.
    """
    from torchvision.transforms import functional as TF

    img_h, img_w = im1.size[1], im1.size[0]
    top, left, height, width = T.RandomCrop.get_params(im1, (crop_size, crop_size))

    im1 = TF.crop(im1, top, left, height, width)
    im2 = TF.crop(im2, top, left, height, width)
    depth_low = TF.crop(depth_low, top, left, height, width)
    if depth_high is not None:
        depth_high = TF.crop(depth_high, top, left, height, width)

    sem = _crop_sem_by_image_window(sem, top, left, height, width, img_h, img_w)

    do_hflip = random.random() < 0.5
    do_vflip = random.random() < 0.5

    if do_hflip:
        im1 = TF.hflip(im1)
        im2 = TF.hflip(im2)
        depth_low = TF.hflip(depth_low)
        if depth_high is not None:
            depth_high = TF.hflip(depth_high)
        sem = torch.flip(sem, dims=[2])

    if do_vflip:
        im1 = TF.vflip(im1)
        im2 = TF.vflip(im2)
        depth_low = TF.vflip(depth_low)
        if depth_high is not None:
            depth_high = TF.vflip(depth_high)
        sem = torch.flip(sem, dims=[1])

    sem = sem.contiguous()

    tt = ToTensor()
    im1 = tt(im1)
    im2 = tt(im2)
    depth_low = tt(depth_low)
    if depth_high is not None:
        depth_high = tt(depth_high)
    return im1, im2, depth_low, depth_high, sem


