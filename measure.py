import os
import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse



def _resolve_gt_path(label_dir, name):
    """
    Resolve GT path robustly:
    1) exact filename match
    2) same stem with any extension
    """
    candidate = os.path.join(label_dir, name)
    if os.path.exists(candidate):
        return candidate

    stem, _ = os.path.splitext(name)
    matches = glob.glob(os.path.join(label_dir, stem + ".*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"GT image not found for '{name}' under '{label_dir}'")


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))
    return psnr


def build_lpips_alex(device):
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()
    return loss_fn


def _resize_pred_to_gt_if_needed(pred_rgb, gt_rgb):
    pred = np.array(pred_rgb)
    gt = np.array(gt_rgb)
    if pred.shape[:2] != gt.shape[:2]:
        pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0])))
    return pred, gt


def calculate_pair_metrics(pred_rgb, gt_rgb, loss_fn, device, use_GT_mean=False):
    """
    Compute PSNR/SSIM/LPIPS on a single pair.
    pred_rgb, gt_rgb: RGB uint8-like arrays (H, W, 3).
    """
    pred, gt = _resize_pred_to_gt_if_needed(pred_rgb, gt_rgb)

    if use_GT_mean:
        mean_restored = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY).mean()
        mean_target = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).mean()
        pred_eval = np.clip(pred * (mean_target / max(mean_restored, 1e-6)), 0, 255)
    else:
        pred_eval = pred

    score_psnr = calculate_psnr(pred_eval, gt)
    score_ssim = calculate_ssim(pred_eval, gt)
    ex_pred = lpips.im2tensor(pred_eval).to(device)
    ex_gt = lpips.im2tensor(gt).to(device)
    with torch.no_grad():
        score_lpips = loss_fn.forward(ex_gt, ex_pred).item()
    return score_psnr, score_ssim, score_lpips

def metrics(im_dir, label_dir, use_GT_mean, device=None, empty_cache_interval=0):
    """
    Backward-compatible single-mode API.
    Internally reuses one-pass dual-mode computation for efficiency.
    """
    (
        avg_psnr_gt,
        avg_ssim_gt,
        avg_lpips_gt,
        avg_psnr_raw,
        avg_ssim_raw,
        avg_lpips_raw,
    ) = metrics_both(im_dir, label_dir, device=device, empty_cache_interval=empty_cache_interval)
    if use_GT_mean:
        return avg_psnr_gt, avg_ssim_gt, avg_lpips_gt
    return avg_psnr_raw, avg_ssim_raw, avg_lpips_raw


def metrics_both(im_dir, label_dir, device=None, empty_cache_interval=0):
    """
    Compute both metric sets in one traversal:
      1) w/ GT mean rectification
      2) w/o GT mean rectification (raw)

    Returns:
      (psnr_gt, ssim_gt, lpips_gt, psnr_raw, ssim_raw, lpips_raw)
    """
    avg_psnr_gt = 0.0
    avg_ssim_gt = 0.0
    avg_lpips_gt = 0.0
    avg_psnr_raw = 0.0
    avg_ssim_raw = 0.0
    avg_lpips_raw = 0.0
    n = 0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1

        im1 = Image.open(item).convert("RGB")

        name = os.path.basename(item)

        im2 = Image.open(_resolve_gt_path(label_dir, name)).convert("RGB")
        (h, w) = im2.size
        im1 = im1.resize((h, w))
        im1_raw = np.array(im1)
        im2 = np.array(im2)

        # Raw metrics (w/o GT mean rectification).
        score_psnr_raw = calculate_psnr(im1_raw, im2)
        score_ssim_raw = calculate_ssim(im1_raw, im2)
        ex_raw = lpips.im2tensor(im1_raw).to(device)
        ex_ref = lpips.im2tensor(im2).to(device)
        with torch.no_grad():
            score_lpips_raw = loss_fn.forward(ex_ref, ex_raw)

        # GT-mean-rectified metrics.
        mean_restored = cv2.cvtColor(im1_raw, cv2.COLOR_RGB2GRAY).mean()
        mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
        im1_gt = np.clip(im1_raw * (mean_target / mean_restored), 0, 255)

        score_psnr_gt = calculate_psnr(im1_gt, im2)
        score_ssim_gt = calculate_ssim(im1_gt, im2)
        ex_gt = lpips.im2tensor(im1_gt).to(device)
        with torch.no_grad():
            score_lpips_gt = loss_fn.forward(ex_ref, ex_gt)

        avg_psnr_gt += score_psnr_gt
        avg_ssim_gt += score_ssim_gt
        avg_lpips_gt += score_lpips_gt.item()
        avg_psnr_raw += score_psnr_raw
        avg_ssim_raw += score_ssim_raw
        avg_lpips_raw += score_lpips_raw.item()
        if (
            empty_cache_interval
            and device.type == "cuda"
            and (n % int(empty_cache_interval) == 0)
        ):
            torch.cuda.empty_cache()

    avg_psnr_gt /= n
    avg_ssim_gt /= n
    avg_lpips_gt /= n
    avg_psnr_raw /= n
    avg_ssim_raw /= n
    avg_lpips_raw /= n
    return avg_psnr_gt, avg_ssim_gt, avg_lpips_gt, avg_psnr_raw, avg_ssim_raw, avg_lpips_raw


if __name__ == '__main__':
    
    mea_parser = argparse.ArgumentParser(description='Measure')
    mea_parser.add_argument('--use_GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
    mea_parser.add_argument('--lol', action='store_true', help='measure lolv1 dataset')
    mea_parser.add_argument('--lol_v2_real', action='store_true', help='measure lol_v2_real dataset')
    mea_parser.add_argument('--lol_v2_syn', action='store_true', help='measure lol_v2_syn dataset')
    mea_parser.add_argument('--SICE_grad', action='store_true', help='measure SICE_grad dataset')
    mea_parser.add_argument('--SICE_mix', action='store_true', help='measure SICE_mix dataset')
    mea_parser.add_argument('--fivek', action='store_true', help='measure fivek dataset')
    mea = mea_parser.parse_args()

    out_root = "data_dir/output"
    if mea.lol:
        im_dir = f'{out_root}/LOLv1/*.png'
        label_dir = 'data_dir/eval15/high/'
    if mea.lol_v2_real:
        im_dir = f'{out_root}/LOLv2_real/*.png'
        label_dir = 'data_dir/LOLv2/Real_captured/Test/Normal/'
    if mea.lol_v2_syn:
        im_dir = f'{out_root}/LOLv2_syn/*.png'
        label_dir = 'data_dir/LOLv2/Synthetic/Test/Normal/'
    if mea.SICE_grad:
        im_dir = f'{out_root}/SICE_grad/*.png'
        label_dir = 'data_dir/SICE/SICE_Reshape/'
    if mea.SICE_mix:
        im_dir = f'{out_root}/SICE_mix/*.png'
        label_dir = 'data_dir/SICE/SICE_Reshape/'
    if mea.fivek:
        im_dir = f'{out_root}/fivek/*.jpg'
        label_dir = 'data_dir/FiveK/test/target/'

    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, mea.use_GT_mean)
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))






