import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
# from net.CIDNet import CIDNet
# from net.depth_prior_net import FSNet
# from net.depth_cidnet_fusion_2 import RefineNet
from net.depth_mst_3 import MST_Plus_Plus
from PIL import Image
from measure import _resolve_gt_path, calculate_pair_metrics, build_lpips_alex


def _model_sd_from_checkpoint(obj):
    """Support full training checkpoint or plain state dict."""
    if not isinstance(obj, dict):
        return obj
    if "model_state_dict" in obj:
        return obj["model_state_dict"]
    if "state_dict" in obj and "optimizer_state_dict" not in obj:
        return obj["state_dict"]
    return obj


def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


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


def eval(
    model,
    testing_data_loader,
    model_path,
    output_folder,
    norm_size=True,
    LOL=False,
    v2=False,
    unpaired=False,
    alpha=1.0,
    gamma=1.0,
    use_freq_branch=False,
    use_dinov3=False,
    sem_channels=None,
    empty_cache_interval=0,
    reload_weights=True,
    sem_scale: float = 1.0,
    metric_label_dir: str = "",
    metric_mode: str = "none",
    save_outputs: bool = True,
):
    torch.set_grad_enabled(False)
    if reload_weights:
        sd = torch.load(
            model_path,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        sd = _model_sd_from_checkpoint(sd)
        model.load_state_dict(sd, strict=True)
        print("Pre-trained model is loaded (strict=True).")
    model.eval()
    model_device = next(model.parameters()).device
    metric_mode = str(metric_mode).lower().strip()
    metric_enabled = bool(metric_label_dir) and metric_mode in ("raw", "gt")
    metric_use_gt_mean = metric_mode == "gt"
    metric_loss_fn = None
    metric_psnr = 0.0
    metric_ssim = 0.0
    metric_lpips = 0.0
    metric_n = 0
    if metric_enabled:
        if not os.path.isdir(metric_label_dir):
            raise FileNotFoundError(f"[Eval] metric_label_dir is not a directory: {metric_label_dir}")
        metric_loss_fn = build_lpips_alex(model_device)
    print('Evaluation:')
    # if LOL:
    #     model.trans.gated = True
    # elif v2:
    #     model.trans.gated2 = True
    #     model.trans.alpha = alpha
    # elif unpaired:
    #     model.trans.gated2 = True
    #     model.trans.alpha = alpha
    sem_channels_checked = False
    os.makedirs(output_folder, exist_ok=True)
    for idx, batch in enumerate(tqdm(testing_data_loader), start=1):
        with torch.no_grad():
            if norm_size:
                if len(batch) >= 5 and use_dinov3:
                    input, name, depth_low, _, sem_feat = batch[0], batch[1], batch[2], batch[3], batch[4]
                else:
                    input, name, depth_low = batch[0], batch[1], batch[2]
                    sem_feat = None
            else:
                if len(batch) >= 7 and use_dinov3:
                    input, name, h, w, depth_low, _, sem_feat = (
                        batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
                    )
                else:
                    input, name, h, w, depth_low, _ = (
                        batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
                    )
                    sem_feat = None

            input = input.to(model_device, non_blocking=True)
            depth_low = depth_low.to(model_device, non_blocking=True)
            if sem_feat is not None:
                sem_feat = sem_feat.to(model_device, non_blocking=True)
                if sem_channels is not None and not sem_channels_checked:
                    sem_c = int(sem_channels)
                    if sem_feat.shape[1] != sem_c:
                        raise ValueError(
                            f"[Eval] Semantic channel mismatch: batch sem C={sem_feat.shape[1]} vs expected {sem_c}"
                        )
                    sem_channels_checked = True
            # output_2, output_1, output = model(input**gamma, depth_low)
            output = model(
                input**gamma,
                depth_low,
                sem_feat,
                sem_scale=float(max(sem_scale, 0.0)),
            )
            # output = model(input**gamma) 
            
            
        output = torch.clamp(output, 0, 1)
        if not norm_size:
            output = output[:, :, :h, :w]
        
        output_img = transforms.ToPILImage()(output.squeeze(0).detach().cpu())
        if save_outputs:
            output_img.save(os.path.join(output_folder, name[0]))
        if metric_enabled:
            gt_path = _resolve_gt_path(metric_label_dir, name[0])
            gt_img = Image.open(gt_path).convert("RGB")
            score_psnr, score_ssim, score_lpips = calculate_pair_metrics(
                np.array(output_img),
                np.array(gt_img),
                metric_loss_fn,
                model_device,
                use_GT_mean=metric_use_gt_mean,
            )
            metric_psnr += score_psnr
            metric_ssim += score_ssim
            metric_lpips += score_lpips
            metric_n += 1

        if empty_cache_interval > 0 and (idx % int(empty_cache_interval) == 0) and model_device.type == "cuda":
            torch.cuda.empty_cache()
    print('===> End evaluation')
    # if LOL:
    #     model.trans.gated = False
    # elif v2:
    #     model.trans.gated2 = False
    torch.set_grad_enabled(True)
    if metric_enabled:
        if metric_n == 0:
            raise RuntimeError("[Eval] metric_enabled but no samples were processed.")
        return metric_psnr / metric_n, metric_ssim / metric_n, metric_lpips / metric_n
    
if __name__ == '__main__':
    
    eval_parser = argparse.ArgumentParser(description='Eval')
    eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
    eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
    eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
    eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
    eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
    eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')
    eval_parser.add_argument('--fivek', action='store_true', help='output FiveK dataset')

    eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
    eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
    eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

    eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
    eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
    eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
    eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
    eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
    eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
    eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
    eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
    eval_parser.add_argument('--alpha', type=float, default=1.0)
    eval_parser.add_argument('--gamma', type=float, default=1.0)
    eval_parser.add_argument('--unpaired_weights', type=str, default='data_dir/weights/LOLv2_syn/w_perc.pth')
    eval_parser.add_argument('--use_freq_branch', type=_str2bool, default=True,
                             help='Keep consistent with training; enable when checkpoint contains freq_stack keys')
    eval_parser.add_argument('--freq_inject_pos', type=str, default='bottleneck', choices=['bottleneck'])
    eval_parser.add_argument('--fft_mode', type=str, default='both', choices=['amp', 'phase', 'both'])
    eval_parser.add_argument('--freq_weight', type=float, default=0.2)
    eval_parser.add_argument('--freq_blocks', type=int, default=1,
                             help='Keep consistent with training; controls frequency branch depth')
    eval_parser.add_argument('--use_dinov3', type=_str2bool, default=False)
    eval_parser.add_argument('--dinov3_cache_dir', type=str, default='',
                             help='Eval semantic *.npy cache dir (output of cache_dinov3_features.py)')
    eval_parser.add_argument('--dinov3_sem_channels', type=int, default=1024,
                             help='Must match cached *.npy C (ViT-S=384, ViT-B=768, ViT-L=1024)')
    eval_parser.add_argument('--use_dino', dest='use_dinov3', type=_str2bool, default=False,
                             help='Deprecated alias of --use_dinov3')
    eval_parser.add_argument('--dino_cache_dir', dest='dinov3_cache_dir', type=str, default='',
                             help='Deprecated alias of --dinov3_cache_dir')
    eval_parser.add_argument('--dino_sem_channels', dest='dinov3_sem_channels', type=int, default=1024,
                             help='Deprecated alias of --dinov3_sem_channels (same default as --dinov3_sem_channels)')
    eval_parser.add_argument('--semantic_fusion_weight', type=float, default=0.1)
    eval_parser.add_argument('--use_dp_caa', type=_str2bool, default=False,
                             help='Must match training if checkpoint contains dp_caa weights')
    eval_parser.add_argument('--dp_caa_window', type=int, default=8)
    eval_parser.add_argument('--dp_caa_sem_embed', type=int, default=32)
    eval_parser.add_argument('--dp_caa_tau', type=float, default=0.07)
    eval_parser.add_argument('--eval_empty_cache_interval', type=int, default=0,
                             help='Call torch.cuda.empty_cache() every N eval images; 0 disables')
    eval_parser.add_argument('--cpu', action='store_true', help='Run eval on CPU (slow; MST+DINOv3 may be heavy)')
    eval_parser.add_argument(
        '--sem_scale',
        type=float,
        default=1.0,
        help='Forward sem_scale (align with train sem warmup); default 1.0 for full-strength eval',
    )

    ep = eval_parser.parse_args()

    from data.options import _normalize_cache_dir_for_project

    if ep.use_dinov3:
        ep.dinov3_cache_dir = _normalize_cache_dir_for_project(ep.dinov3_cache_dir)

    _sem_eval = ep.dinov3_cache_dir.strip() if ep.use_dinov3 else None
    if ep.use_dinov3 and not _sem_eval:
        raise SystemExit("eval.py: --use_dinov3 true requires non-empty --dinov3_cache_dir")
    if ep.use_dinov3 and _sem_eval and not os.path.isdir(_sem_eval):
        raise SystemExit(f"eval.py: semantic cache is not a directory: {_sem_eval}")
    if ep.use_dinov3 and _sem_eval:
        inferred_c = _infer_sem_channels_from_cache(_sem_eval)
        if inferred_c is not None and int(ep.dinov3_sem_channels) != int(inferred_c):
            print(
                f"[Eval] Override --dinov3_sem_channels: {ep.dinov3_sem_channels} -> {inferred_c} "
                "(inferred from semantic cache)"
            )
            ep.dinov3_sem_channels = int(inferred_c)

    if ep.cpu:
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('No GPU found; eval.py currently requires CUDA, or pass --cpu (slower).')
        device = torch.device('cuda')

    os.makedirs('data_dir/output', exist_ok=True)
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(
            dataset=get_eval_set("data_dir/eval15/low", semantic_cache_dir=_sem_eval),
            num_workers=num_workers, batch_size=1, shuffle=False,
        )
        output_folder = 'data_dir/output/LOLv1/'
        if ep.perc:
            weight_path = 'data_dir/weights/LOLv1/w_perc.pth'
        else:
            weight_path = 'data_dir/weights/LOLv1/wo_perc.pth'
        
            
    elif ep.lol_v2_real:
        eval_data = DataLoader(
            dataset=get_eval_set("data_dir/LOLv2/Real_captured/Test/Low", semantic_cache_dir=_sem_eval),
            num_workers=num_workers, batch_size=1, shuffle=False,
        )
        output_folder = 'data_dir/output/LOLv2_real/'
        if ep.best_GT_mean:
            weight_path = 'data_dir/weights/LOLv2_real/w_perc.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = 'data_dir/weights/LOLv2_real/best_PSNR.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = 'data_dir/weights/LOLv2_real/best_SSIM.pth'
            alpha = 0.82
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(
            dataset=get_eval_set("data_dir/LOLv2/Synthetic/Test/Low", semantic_cache_dir=_sem_eval),
            num_workers=num_workers, batch_size=1, shuffle=False,
        )
        output_folder = 'data_dir/output/LOLv2_syn/'
        if ep.perc:
            weight_path = 'data_dir/weights/LOLv2_syn/w_perc.pth'
        else:
            weight_path = 'data_dir/weights/LOLv2_syn/wo_perc.pth'
            
    elif ep.SICE_grad:
        eval_data = DataLoader(
            dataset=get_SICE_eval_set("data_dir/SICE/SICE_Grad", semantic_cache_dir=_sem_eval),
            num_workers=num_workers, batch_size=1, shuffle=False,
        )
        output_folder = 'data_dir/output/SICE_grad/'
        weight_path = 'data_dir/weights/SICE.pth'
        norm_size = False
        
    elif ep.SICE_mix:
        eval_data = DataLoader(
            dataset=get_SICE_eval_set("data_dir/SICE/SICE_Mix", semantic_cache_dir=_sem_eval),
            num_workers=num_workers, batch_size=1, shuffle=False,
        )
        output_folder = 'data_dir/output/SICE_mix/'
        weight_path = 'data_dir/weights/SICE.pth'
        norm_size = False
        
    elif ep.fivek:
        eval_data = DataLoader(
            dataset=get_fivek_eval_set("data_dir/FiveK/test/input", semantic_cache_dir=_sem_eval),
            num_workers=num_workers, batch_size=1, shuffle=False,
        )
        output_folder = 'data_dir/output/fivek/'
        weight_path = 'data_dir/weights/fivek.pth'
        norm_size = False
    
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("data_dir/DICM", semantic_cache_dir=_sem_eval),
                num_workers=num_workers, batch_size=1, shuffle=False,
            )
            output_folder = 'data_dir/output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("data_dir/LIME", semantic_cache_dir=_sem_eval),
                num_workers=num_workers, batch_size=1, shuffle=False,
            )
            output_folder = 'data_dir/output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("data_dir/MEF", semantic_cache_dir=_sem_eval),
                num_workers=num_workers, batch_size=1, shuffle=False,
            )
            output_folder = 'data_dir/output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("data_dir/NPE", semantic_cache_dir=_sem_eval),
                num_workers=num_workers, batch_size=1, shuffle=False,
            )
            output_folder = 'data_dir/output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("data_dir/VV", semantic_cache_dir=_sem_eval),
                num_workers=num_workers, batch_size=1, shuffle=False,
            )
            output_folder = 'data_dir/output/VV/'
        elif ep.custome:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set(ep.custome_path, semantic_cache_dir=_sem_eval),
                num_workers=num_workers, batch_size=1, shuffle=False,
            )
            output_folder = 'data_dir/output/custome/'
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights
        
    eval_net = MST_Plus_Plus(
        use_freq_branch=ep.use_freq_branch,
        freq_inject_pos=ep.freq_inject_pos,
        fft_mode=ep.fft_mode,
        freq_weight=ep.freq_weight,
        freq_blocks=ep.freq_blocks,
        use_semantic_prior=ep.use_dinov3,
        dino_sem_channels=ep.dinov3_sem_channels,
        semantic_fusion_weight=ep.semantic_fusion_weight,
        use_dp_caa=ep.use_dp_caa,
        dp_caa_window=ep.dp_caa_window,
        dp_caa_sem_embed=ep.dp_caa_sem_embed,
        dp_caa_tau=ep.dp_caa_tau,
    ).to(device)
    eval(
        eval_net,
        eval_data,
        weight_path,
        output_folder,
        norm_size=norm_size,
        LOL=ep.lol,
        v2=ep.lol_v2_real,
        unpaired=ep.unpaired,
        alpha=alpha,
        gamma=ep.gamma,
        use_freq_branch=ep.use_freq_branch,
        use_dinov3=ep.use_dinov3,
        sem_channels=ep.dinov3_sem_channels,
        empty_cache_interval=ep.eval_empty_cache_interval,
        sem_scale=float(ep.sem_scale),
    )








