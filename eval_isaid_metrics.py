import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import warnings

from data.data import get_eval_set
from measure import _resolve_gt_path, build_lpips_alex, calculate_psnr, calculate_ssim
from net.depth_mst_3 import MST_Plus_Plus
try:
    from skimage.color import rgb2lab as _sk_rgb2lab
    from skimage.color import deltaE_ciede2000 as _sk_deltaE_ciede2000
    _HAS_SKIMAGE_DELTAE = True
except Exception:
    _HAS_SKIMAGE_DELTAE = False


def _prepare_lpips_runtime(allow_online_download: bool) -> None:
    """
    Prepare env for LPIPS weights loading.
    When online download is allowed, clear common offline flags so missing
    pretrained weights can be fetched during normal runtime.
    """
    if not allow_online_download:
        return
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if key in os.environ:
            os.environ.pop(key, None)


def _str2bool(v: str) -> bool:
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _infer_sem_channels_from_cache(cache_dir: str):
    cdir = str(cache_dir or "").strip()
    if not cdir or not os.path.isdir(cdir):
        return None
    for name in sorted(os.listdir(cdir)):
        if not str(name).lower().endswith(".npy"):
            continue
        path = os.path.join(cdir, name)
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception:
            continue
        try:
            if arr.ndim == 2:
                return 1
            if arr.ndim >= 3:
                return int(arr.shape[0])
        finally:
            try:
                del arr
            except Exception:
                pass
    return None


def _model_sd_from_checkpoint(obj):
    if not isinstance(obj, dict):
        return obj
    if "model_state_dict" in obj:
        return obj["model_state_dict"]
    if "state_dict" in obj and "optimizer_state_dict" not in obj:
        return obj["state_dict"]
    return obj


def _resize_pred_to_gt_if_needed(pred_rgb: np.ndarray, gt_rgb: np.ndarray):
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError(
            f"Shape mismatch during metric computation: pred={pred_rgb.shape}, gt={gt_rgb.shape}. "
            "Please align output size to GT before metrics."
        )
    return pred_rgb, gt_rgb


def calc_mae_01(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    pred, gt = _resize_pred_to_gt_if_needed(pred_rgb, gt_rgb)
    return float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)) / 255.0))


def calc_delta_e76(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    pred, gt = _resize_pred_to_gt_if_needed(pred_rgb, gt_rgb)
    pred_lab = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    gt_lab = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    delta = pred_lab - gt_lab
    delta_e = np.sqrt(np.sum(delta * delta, axis=2))
    return float(np.mean(delta_e))


def _delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    # lab1/lab2: (..., 3), using OpenCV LAB ranges. Convert to standard CIELAB first.
    l1 = lab1[..., 0] * (100.0 / 255.0)
    a1 = lab1[..., 1] - 128.0
    b1 = lab1[..., 2] - 128.0
    l2 = lab2[..., 0] * (100.0 / 255.0)
    a2 = lab2[..., 1] - 128.0
    b2 = lab2[..., 2] - 128.0

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = (c1 + c2) / 2.0
    c_bar7 = np.power(c_bar, 7)
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + np.power(25.0, 7) + 1e-12)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.degrees(np.arctan2(b1, a1p))
    h2p = np.degrees(np.arctan2(b2, a2p))
    h1p = np.where(h1p < 0, h1p + 360.0, h1p)
    h2p = np.where(h2p < 0, h2p + 360.0, h2p)

    dlp = l2 - l1
    dcp = c2p - c1p

    dhp = h2p - h1p
    dhp = np.where(c1p * c2p == 0, 0.0, dhp)
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp / 2.0))

    l_bar_p = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0

    h_sum = h1p + h2p
    h_bar_p = np.where(c1p * c2p == 0, h_sum, (h1p + h2p) / 2.0)
    h_bar_p = np.where((c1p * c2p != 0) & (np.abs(h1p - h2p) > 180.0) & (h_sum < 360.0), (h_sum + 360.0) / 2.0, h_bar_p)
    h_bar_p = np.where((c1p * c2p != 0) & (np.abs(h1p - h2p) > 180.0) & (h_sum >= 360.0), (h_sum - 360.0) / 2.0, h_bar_p)

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )

    delta_theta = 30.0 * np.exp(-np.power((h_bar_p - 275.0) / 25.0, 2))
    r_c = 2.0 * np.sqrt(np.power(c_bar_p, 7) / (np.power(c_bar_p, 7) + np.power(25.0, 7) + 1e-12))
    s_l = 1.0 + (0.015 * np.power(l_bar_p - 50.0, 2)) / np.sqrt(20.0 + np.power(l_bar_p - 50.0, 2))
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c

    k_l = 1.0
    k_c = 1.0
    k_h = 1.0
    term_l = dlp / (k_l * s_l + 1e-12)
    term_c = dcp / (k_c * s_c + 1e-12)
    term_h = dhp / (k_h * s_h + 1e-12)
    return np.sqrt(term_l * term_l + term_c * term_c + term_h * term_h + r_t * term_c * term_h)


def calc_delta_e00(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    pred, gt = _resize_pred_to_gt_if_needed(pred_rgb, gt_rgb)
    if _HAS_SKIMAGE_DELTAE:
        pred_lab = _sk_rgb2lab(pred.astype(np.float32) / 255.0)
        gt_lab = _sk_rgb2lab(gt.astype(np.float32) / 255.0)
        delta_e = _sk_deltaE_ciede2000(pred_lab, gt_lab)
        return float(np.mean(delta_e))

    warnings.warn(
        "skimage not available; falling back to internal CIEDE2000 implementation. "
        "Install scikit-image for reference implementation.",
        RuntimeWarning,
    )
    pred_lab = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    gt_lab = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    delta_e = _delta_e_ciede2000(pred_lab, gt_lab)
    return float(np.mean(delta_e))


def main():
    parser = argparse.ArgumentParser(description="Evaluate iSAID model and dump metrics.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (epoch_xxx.pth).")
    parser.add_argument("--low_dir", type=str, required=True, help="Input low-light directory.")
    parser.add_argument("--gt_dir", type=str, required=True, help="GT directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save restored images.")
    parser.add_argument("--report_txt", type=str, required=True, help="Where to save summary txt.")
    parser.add_argument("--depth_dir", type=str, default="", help="Optional depth_maps dir; empty uses sibling low_depth/depth_maps.")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Keep architecture flags consistent with training checkpoint.
    parser.add_argument("--use_freq_branch", type=_str2bool, default=False)
    parser.add_argument("--freq_inject_pos", type=str, default="bottleneck", choices=["bottleneck"])
    parser.add_argument("--fft_mode", type=str, default="both", choices=["amp", "phase", "both"])
    parser.add_argument("--freq_weight", type=float, default=0.2)
    parser.add_argument("--freq_blocks", type=int, default=1)

    parser.add_argument("--use_dinov3", type=_str2bool, default=False)
    parser.add_argument("--dinov3_cache_dir", type=str, default="", help="Required when --use_dinov3 true.")
    parser.add_argument("--dinov3_sem_channels", type=int, default=1024)
    parser.add_argument("--semantic_fusion_weight", type=float, default=0.1)
    parser.add_argument("--sem_scale", type=float, default=1.0)

    parser.add_argument("--use_dp_caa", type=_str2bool, default=False)
    parser.add_argument("--dp_caa_window", type=int, default=8)
    parser.add_argument("--dp_caa_sem_embed", type=int, default=32)
    parser.add_argument("--dp_caa_tau", type=float, default=0.07)
    parser.add_argument(
        "--lpips_allow_online_download",
        type=_str2bool,
        default=True,
        help="Allow LPIPS pretrained weights to download at runtime when missing.",
    )
    args = parser.parse_args()

    if args.use_dinov3 and not str(args.dinov3_cache_dir).strip():
        raise ValueError("--use_dinov3 true requires non-empty --dinov3_cache_dir")
    if args.use_dinov3:
        inferred_c = _infer_sem_channels_from_cache(args.dinov3_cache_dir)
        if inferred_c is not None and int(args.dinov3_sem_channels) != int(inferred_c):
            print(
                f"[Eval iSAID] Override --dinov3_sem_channels: {args.dinov3_sem_channels} -> {inferred_c} "
                "(inferred from semantic cache)"
            )
            args.dinov3_sem_channels = int(inferred_c)
    if not os.path.isdir(args.low_dir):
        raise FileNotFoundError(f"low_dir not found: {args.low_dir}")
    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(f"gt_dir not found: {args.gt_dir}")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_txt)), exist_ok=True)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    eval_set = get_eval_set(
        args.low_dir,
        semantic_cache_dir=(args.dinov3_cache_dir if args.use_dinov3 else None),
        depth_dir=(args.depth_dir if str(args.depth_dir).strip() else None),
    )
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=1)

    model = MST_Plus_Plus(
        use_freq_branch=args.use_freq_branch,
        freq_inject_pos=args.freq_inject_pos,
        fft_mode=args.fft_mode,
        freq_weight=args.freq_weight,
        freq_blocks=args.freq_blocks,
        use_semantic_prior=args.use_dinov3,
        dino_sem_channels=args.dinov3_sem_channels,
        semantic_fusion_weight=args.semantic_fusion_weight,
        use_dp_caa=args.use_dp_caa,
        dp_caa_window=args.dp_caa_window,
        dp_caa_sem_embed=args.dp_caa_sem_embed,
        dp_caa_tau=args.dp_caa_tau,
    ).to(device)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = _model_sd_from_checkpoint(sd)
    model.load_state_dict(sd, strict=True)
    model.eval()

    _prepare_lpips_runtime(bool(args.lpips_allow_online_download))
    lpips_fn = build_lpips_alex(device)

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    mae_sum = 0.0
    delta_e_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            if args.use_dinov3:
                input_img, name, depth_low, _, sem_feat = batch
                sem_feat = sem_feat.to(device, non_blocking=True)
                if sem_feat.shape[1] != int(args.dinov3_sem_channels):
                    raise ValueError(
                        f"Semantic channel mismatch: got {sem_feat.shape[1]} vs {args.dinov3_sem_channels}"
                    )
            else:
                input_img, name, depth_low, _ = batch
                sem_feat = None

            input_img = input_img.to(device, non_blocking=True)
            depth_low = depth_low.to(device, non_blocking=True)
            pred = model(input_img, depth_low, sem_feat, sem_scale=float(max(args.sem_scale, 0.0)))
            pred = torch.clamp(pred, 0, 1)

            pred_pil = transforms.ToPILImage()(pred.squeeze(0).detach().cpu())
            out_name = name[0]
            pred_path = os.path.join(args.output_dir, out_name)
            pred_pil.save(pred_path)

            gt_path = _resolve_gt_path(args.gt_dir, out_name)
            gt_img = Image.open(gt_path).convert("RGB")

            pred_rgb = np.array(pred_pil)
            gt_rgb = np.array(gt_img)
            pred_rgb, gt_rgb = _resize_pred_to_gt_if_needed(pred_rgb, gt_rgb)

            psnr = calculate_psnr(pred_rgb, gt_rgb)
            ssim = calculate_ssim(pred_rgb, gt_rgb)

            ex_pred = torch.from_numpy(np.ascontiguousarray(pred_rgb)).permute(2, 0, 1).unsqueeze(0).float()
            ex_gt = torch.from_numpy(np.ascontiguousarray(gt_rgb)).permute(2, 0, 1).unsqueeze(0).float()
            ex_pred = ex_pred / 127.5 - 1.0
            ex_gt = ex_gt / 127.5 - 1.0
            ex_pred = ex_pred.to(device)
            ex_gt = ex_gt.to(device)
            lpips_score = float(lpips_fn.forward(ex_gt, ex_pred).item())

            mae = calc_mae_01(pred_rgb, gt_rgb)
            delta_e = calc_delta_e00(pred_rgb, gt_rgb)

            psnr_sum += psnr
            ssim_sum += ssim
            lpips_sum += lpips_score
            mae_sum += mae
            delta_e_sum += delta_e
            n += 1

    if n == 0:
        raise RuntimeError("No samples were evaluated.")

    results = {
        "PSNR": psnr_sum / n,
        "SSIM": ssim_sum / n,
        "LPIPS_alex": lpips_sum / n,
        "MAE_01": mae_sum / n,
        "DeltaE00": delta_e_sum / n,
        "NUM_SAMPLES": n,
    }

    lines = [
        f"Time: {datetime.now().isoformat()}",
        f"Checkpoint: {args.ckpt}",
        f"Input low_dir: {args.low_dir}",
        f"GT dir: {args.gt_dir}",
        f"Output dir: {args.output_dir}",
        f"Depth dir: {args.depth_dir if args.depth_dir else '(auto sibling low_depth/depth_maps)'}",
        f"use_dinov3: {args.use_dinov3}",
        f"dinov3_cache_dir: {args.dinov3_cache_dir if args.use_dinov3 else '(disabled)'}",
        "",
        "Averaged Metrics:",
        f"PSNR: {results['PSNR']:.6f}",
        f"SSIM: {results['SSIM']:.6f}",
        f"LPIPS(alex): {results['LPIPS_alex']:.6f}",
        f"MAE[0,1]: {results['MAE_01']:.6f}",
        f"DeltaE00: {results['DeltaE00']:.6f}",
        f"Samples: {results['NUM_SAMPLES']}",
    ]
    report = "\n".join(lines) + "\n"
    with open(args.report_txt, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"Saved report: {args.report_txt}")


if __name__ == "__main__":
    main()
