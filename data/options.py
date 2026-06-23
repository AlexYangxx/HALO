import argparse
import os

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _normalize_cache_dir_for_project(path: str) -> str:
    """
    Normalize semantic cache path with project-root fallback.

    Common mistake on Linux:
      pass '/cache_dinov3/xxx' while actual dir is './cache_dinov3/xxx'
      under current project root.
    """
    raw = str(path or "").strip()
    if not raw:
        return raw

    expanded = os.path.expanduser(raw)
    if os.path.isdir(expanded):
        return expanded

    if expanded.startswith(os.sep):
        fallback = os.path.join(os.getcwd(), expanded.lstrip("/\\"))
        if os.path.isdir(fallback):
            print(f"[Options] auto-resolved cache dir: {raw} -> {fallback}")
            return fallback

    abs_path = os.path.abspath(expanded)
    if os.path.isdir(abs_path):
        return abs_path

    return expanded


def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    # parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (auto-set by torchrun)')
    # parser.add_argument('--distributed', type=_str2bool, default=False, help='Whether to use distributed training (DDP)')
    # parser.add_argument('--gpus', type=str, default='0,1', help='GPU ids to use (e.g. 0,1,2)')
    
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--resume_path', type=str, default='', help='resume checkpoint path; if set, overrides start_epoch loading')
    parser.add_argument('--snapshots', type=int, default=5, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=_str2bool, default=True)
    parser.add_argument('--gpus', type=str, default='0',
                        help='Visible GPU ids, e.g. "0" or "0,1"')
    parser.add_argument('--shuffle', type=_str2bool, default=True)
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')
    parser.add_argument('--pin_memory', type=_str2bool, default=True,
                        help='Enable DataLoader pinned memory for faster H2D copy')
    parser.add_argument('--persistent_workers', type=_str2bool, default=True,
                        help='Keep DataLoader workers alive between epochs (threads>0 only)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Batches prefetched per worker when threads>0')

    # reproducibility
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for reproducible training (set different values for multi-seed runs)')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=_str2bool, default=False)
    parser.add_argument('--cos_restart', type=_str2bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=_str2bool, default=True, help='turn False to train without warmup') 
    parser.add_argument('--warmup_start_factor', type=float, default=0.1,
                        help='Warmup start LR factor in [0,1], e.g. 0.1 means start from 0.1*base_lr')

    # train datasets (统一置于项目根下 data_dir/)
    parser.add_argument('--data_train_lol_v1'       , type=str, default='data_dir/our485')
    parser.add_argument('--data_train_isaid'        , type=str, default='data_dir/iSAID-dark/train')

    # validation input
    parser.add_argument('--data_val_lol_v1'         , type=str, default='data_dir/eval15/low')
    parser.add_argument('--data_val_isaid'          , type=str, default='data_dir/iSAID-dark/val/low')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='data_dir/eval15/high/')
    parser.add_argument('--data_valgt_isaid'        , type=str, default='data_dir/iSAID-dark/val/gt/')

    # Default experiment output folder.
    # Use project-root-relative path so that outputs land under "TGRS/exp/" when running from repo root.
    parser.add_argument('--val_folder', default='exp/SGDF-Net/', help='Root output directory for experiments (weights, validation images, logs, etc.)')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=6.0,
                        help='L1 loss weight (legacy baseline often used 10)')
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)
    parser.add_argument('--legacy_perceptual_scaling', type=_str2bool, default=False,
                        help='If True: perceptual_term = 2 * P_loss(...) (P_weight applied only inside PerceptualLoss)')
    parser.add_argument('--perceptual_repeat', type=int, default=2,
                        help='Repeat factor for perceptual loss when legacy_perceptual_scaling=False')
    parser.add_argument('--depth_edge_weight', type=float, default=0.01,
                        help='Weight for DepthEdgeConsistencyLoss (very small by default)')
    parser.add_argument('--depth_smooth_weight', type=float, default=0.0,
                        help='Weight for DepthRegionSmoothLoss (disabled when 0)')
    
    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=_str2bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=_str2bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=_str2bool, default=True, help='if gradient fluctuates too much, turn-on it')
    parser.add_argument('--grad_clip_max_norm', type=float, default=0.1, help='max norm used by grad clipping')
    parser.add_argument('--use_GT_mean', type=_str2bool, default=True, help='use GT mean rectification during validation metrics')
    parser.add_argument('--best_metric', type=str, default='raw', choices=['raw', 'gt'],
                        help='metric used for selecting best model: raw(w/o GT mean) or gt(w/ GT mean)')

    # Stage A: optional frequency residual branch (DFFN-style amp/phase), default on
    parser.add_argument('--use_freq_branch', type=_str2bool, default=True,
                        help='Enable FFT amp/phase residual at MST bottleneck')
    parser.add_argument('--freq_inject_pos', type=str, default='bottleneck',
                        choices=['bottleneck'],
                        help='Where to inject freq branch (only bottleneck supported in stage A)')
    parser.add_argument('--fft_mode', type=str, default='both', choices=['amp', 'phase', 'both'],
                        help='Which frequency component to modulate')
    parser.add_argument('--freq_weight', type=float, default=0.2,
                        help='Scalar multiplier for frequency-branch residual')
    parser.add_argument('--freq_blocks', type=int, default=1,
                        help='Number of stacked FreqResidualBlock inside the branch')
    parser.add_argument('--freq_loss_weight', type=float, default=0.0,
                        help='Must be 0 in stage A (aux frequency loss not implemented)')

    # Stage B: offline DINOv3 semantic prior (bottleneck fusion); default off
    parser.add_argument('--use_dinov3', type=_str2bool, default=False,
                        help='Use cached DINOv3 patch features (--dinov3_cache_dir); extends batch with sem_feat')
    parser.add_argument('--dinov3_cache_dir', type=str, default='',
                        help='Train split: directory of *.npy per low-light image stem')
    parser.add_argument('--dinov3_cache_dir_val', type=str, default='',
                        help='Val/test split semantic cache; empty = reuse --dinov3_cache_dir')
    parser.add_argument('--dinov3_sem_channels', type=int, default=1024,
                        help='Channel dim C of cached maps（须与 *.npy 一致；如 ViT-S=384, ViT-B=768, ViT-L=1024）')
    parser.add_argument('--use_dino', type=_str2bool, default=False,
                        help='Deprecated alias of --use_dinov3')
    parser.add_argument('--dino_cache_dir', type=str, default='',
                        help='Deprecated alias of --dinov3_cache_dir')
    parser.add_argument('--dino_cache_dir_val', type=str, default='',
                        help='Deprecated alias of --dinov3_cache_dir_val')
    parser.add_argument('--dino_sem_channels', type=int, default=1024,
                        help='Deprecated alias of --dinov3_sem_channels')
    parser.add_argument('--semantic_fusion_weight', type=float, default=0.1,
                        help='Residual base strength for semantic bottleneck injection')
    parser.add_argument('--sem_warmup_epochs', type=int, default=8,
                        help='Epochs over which semantic fusion weight linearly warms up from 0 to semantic_fusion_weight')

    # Stage C: windowed DP-CAA at MST bottleneck (after MSAB + freq + sem_mix); default off
    parser.add_argument('--use_dp_caa', type=_str2bool, default=False,
                        help='Enable windowed dual-prior context-aware attention at bottleneck')
    parser.add_argument('--dp_caa_window', type=int, default=8,
                        help='Window size for DP-CAA (must divide typical feature map after padding)')
    parser.add_argument('--dp_caa_sem_embed', type=int, default=32,
                        help='Embedding dim for semantic affinity inside DP-CAA')
    parser.add_argument('--dp_caa_tau', type=float, default=0.07,
                        help='Temperature for semantic affinity logits')

    # choose which dataset you want to train
    parser.add_argument('--empty_cache_each_epoch', type=_str2bool, default=False,
                        help='Call torch.cuda.empty_cache() at epoch end (usually slower; keep False)')
    parser.add_argument('--eval_empty_cache_interval', type=int, default=0,
                        help='Call torch.cuda.empty_cache() every N eval images; 0 disables')
    parser.add_argument('--save_val_images', type=_str2bool, default=True,
                        help='Whether to save validation predictions to disk during training snapshots')

    parser.add_argument('--dataset', type=str, default='lol_v1',
    choices=['lol_v1', 'isaid_dark'],
    help='Select the dataset to train on (default: %(default)s)')

    args = parser.parse_args()
    if args.freq_loss_weight > 0:
        parser.error(
            'freq_loss_weight must be 0 in stage A (auxiliary frequency loss is not implemented yet).'
        )
    # Backward-compatible alias resolution (prefer explicit dinov3 args).
    if args.use_dino:
        args.use_dinov3 = True
    if str(args.dino_cache_dir).strip() and not str(args.dinov3_cache_dir).strip():
        args.dinov3_cache_dir = args.dino_cache_dir
    if str(args.dino_cache_dir_val).strip() and not str(args.dinov3_cache_dir_val).strip():
        args.dinov3_cache_dir_val = args.dino_cache_dir_val
    _sem_default = 1024
    if args.dino_sem_channels != _sem_default and args.dinov3_sem_channels == _sem_default:
        args.dinov3_sem_channels = args.dino_sem_channels

    args.dinov3_cache_dir = _normalize_cache_dir_for_project(args.dinov3_cache_dir)
    args.dinov3_cache_dir_val = _normalize_cache_dir_for_project(args.dinov3_cache_dir_val)

    if int(args.dp_caa_window) < 2:
        parser.error('dp_caa_window must be >= 2')
    if int(args.dp_caa_sem_embed) < 1:
        parser.error('dp_caa_sem_embed must be >= 1')
    if float(args.dp_caa_tau) <= 0:
        parser.error('dp_caa_tau must be > 0')

    if args.use_dinov3 and not str(args.dinov3_cache_dir).strip():
        parser.error('use_dinov3=True requires a non-empty --dinov3_cache_dir')
    if args.use_dinov3:
        tr = str(args.dinov3_cache_dir).strip()
        if tr and not os.path.isdir(tr):
            parser.error(f'use_dinov3=True but train cache dir is not a directory: {tr}')
        va = str(args.dinov3_cache_dir_val).strip()
        if va and not os.path.isdir(va):
            parser.error(f'use_dinov3=True but val cache dir is not a directory: {va}')

    # Keep legacy attribute names for old call-sites.
    if int(args.warmup_epochs) < 1 and bool(args.start_warmup):
        parser.error('start_warmup=True requires warmup_epochs >= 1')
    if float(args.warmup_start_factor) < 0.0 or float(args.warmup_start_factor) > 1.0:
        parser.error('warmup_start_factor must be in [0, 1]')

    args.use_dino = args.use_dinov3
    args.dino_cache_dir = args.dinov3_cache_dir
    args.dino_cache_dir_val = args.dinov3_cache_dir_val
    args.dino_sem_channels = args.dinov3_sem_channels
    return args





