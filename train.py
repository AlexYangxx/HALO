import os
import torch
import random
import logging
import sys
from collections.abc import Mapping
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.depth_mst_3 import MST_Plus_Plus
from data.options import option
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime

opt = option()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_VAL_SNAPSHOT = {
    "lol_v1": ("mst_2/LOLv1/", "data_valgt_lol_v1", True),
    "isaid_dark": ("iSAID_dark/", "data_valgt_isaid", True),
}


def _weights_train_dir():
    d = os.path.join(opt.val_folder, f"{opt.dataset}_weights", "training")
    os.makedirs(d, exist_ok=True)
    return d


def _legacy_weights_train_dir():
    """Legacy checkpoint dir kept for backward-compatible resume."""
    return os.path.join(opt.val_folder, "weights", "training_mst_2")


def _setup_device():
    global DEVICE
    if opt.gpu_mode and torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")


def safe_collate(batch):
    """Collate for tuple/tensor/str mixes under multi-worker loading."""
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if torch.is_tensor(elem):
        tensors = [b if b.is_contiguous() else b.contiguous() for b in batch]
        shapes = [tuple(t.shape) for t in tensors]
        if len(set(shapes)) == 1:
            return torch.stack(tensors, dim=0)

        # Allow slight spatial-size mismatches (e.g. DINO semantic grid after crop mapping)
        # by padding to the max H/W inside the batch.
        nd = tensors[0].dim()
        if nd < 3:
            raise RuntimeError(f"safe_collate: cannot pad tensor rank={nd}, shapes={shapes}")
        base_prefix = tuple(tensors[0].shape[:-2])
        for t in tensors[1:]:
            if t.dim() != nd:
                raise RuntimeError(f"safe_collate: mixed ranks, shapes={shapes}")
            if tuple(t.shape[:-2]) != base_prefix:
                raise RuntimeError(f"safe_collate: non-spatial shape mismatch, shapes={shapes}")

        max_h = max(int(t.shape[-2]) for t in tensors)
        max_w = max(int(t.shape[-1]) for t in tensors)
        if max_h <= 0 or max_w <= 0:
            raise RuntimeError(f"safe_collate: invalid spatial sizes, shapes={shapes}")

        padded = []
        for t in tensors:
            h, w = int(t.shape[-2]), int(t.shape[-1])
            pad_h = max_h - h
            pad_w = max_w - w
            if pad_h < 0 or pad_w < 0:
                raise RuntimeError(f"safe_collate: negative pad computed, shapes={shapes}")
            if pad_h == 0 and pad_w == 0:
                padded.append(t)
            else:
                # pad order: (left, right, top, bottom)
                padded.append(torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0.0))
        return torch.stack(padded, dim=0)
    if isinstance(elem, np.ndarray):
        if elem.dtype.kind in ("U", "S", "O"):
            return [str(x) for x in batch]
        tensors = [torch.as_tensor(np.ascontiguousarray(b)) for b in batch]
        return torch.stack(tensors, dim=0)
    if isinstance(elem, np.generic):
        return torch.tensor(batch)
    if isinstance(elem, (float, int)):
        return torch.tensor(batch)
    if isinstance(elem, str):
        return list(batch)
    if isinstance(elem, Mapping):
        return {k: safe_collate([d[k] for d in batch]) for k in elem}
    if isinstance(elem, tuple):
        transposed = list(zip(*batch))
        return tuple(safe_collate(list(samples)) for samples in transposed)
    if isinstance(elem, list):
        transposed = list(zip(*batch))
        return [safe_collate(list(samples)) for samples in transposed]
    return list(batch)


def seed_torch(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def get_sem_lambda(epoch: int) -> float:
    use_semantic = bool(getattr(opt, "use_dinov3", False) or getattr(opt, "use_dino", False))
    if not use_semantic:
        return 0.0
    warm_epochs = int(getattr(opt, "sem_warmup_epochs", 0) or 0)
    if warm_epochs <= 0:
        return 1.0
    if epoch <= 0:
        return 0.0
    if epoch >= warm_epochs:
        return 1.0
    return float(epoch) / float(warm_epochs)


def train_init():
    cuda = opt.gpu_mode
    if cuda:
        gpus = str(getattr(opt, "gpus", "") or "").strip()
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    _setup_device()
    if cuda and not torch.cuda.is_available():
        logging.error("[Init] No GPU found! Use --gpu_mode false for CPU, or install CUDA drivers.")
        raise RuntimeError("No GPU found; pass --gpu_mode false to run on CPU.")
    if cuda:
        logging.info(f"[Init] Using GPU devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    seed = seed_torch(int(opt.seed))
    logging.info(f"[Init] Random seed initialized: {seed}")
    # Follow upstream CIDNet behavior: enable cuDNN benchmark for speed.
    # (Deterministic mode intentionally not configured here.)
    cudnn.benchmark = True


def train(epoch):
    model.train()
    loss_print = torch.zeros((), device=DEVICE)
    pic_cnt = 0
    loss_last_10 = torch.zeros((), device=DEVICE)
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter_idx = 0
    sem_channels_checked = False
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    lam_sem = get_sem_lambda(epoch)
    pbar = tqdm(training_data_loader, desc=f"Epoch {epoch} Training")
    for batch in pbar:
        if len(batch) == 9:
            im1, im2, depth_low, _, _, _, _, _, sem_feat = batch
            sem_feat = sem_feat.to(DEVICE, non_blocking=True)
        elif len(batch) == 8:
            im1, im2, depth_low, _, _, _, _, _ = batch
            sem_feat = None
        elif len(batch) == 7:
            im1, im2, depth_low, _, _, _, sem_feat = batch
            sem_feat = sem_feat.to(DEVICE, non_blocking=True)
        elif len(batch) == 6:
            im1, im2, depth_low, _, _, _ = batch
            sem_feat = None
        else:
            raise ValueError(f"[Train] Unexpected batch format with len={len(batch)}")
        im1 = im1.to(DEVICE, non_blocking=True)
        im2 = im2.to(DEVICE, non_blocking=True)
        depth_low = depth_low.to(DEVICE, non_blocking=True)
        if sem_feat is not None and not sem_channels_checked:
            sem_c = int(opt.dinov3_sem_channels)
            if sem_feat.shape[1] != sem_c:
                raise ValueError(
                    f"[Train] Semantic channel mismatch: batch sem C={sem_feat.shape[1]} vs --dinov3_sem_channels {sem_c}"
                )
            sem_channels_checked = True
        sem_scale = float(lam_sem)
        if sem_feat is not None and sem_scale <= 0.0:
            sem_feat = None
        optimizer.zero_grad(set_to_none=True)
        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma, depth_low, sem_feat, sem_scale=sem_scale)
        else:
            output_rgb = model(im1, depth_low, sem_feat, sem_scale=sem_scale)
        gt_rgb = im2
        # VGG/DINO perceptual expects [0,1]; conv_out + residual is not hard-bounded (depth_mst_3).
        perceptual_in_pred = output_rgb.clamp(0.0, 1.0)
        perceptual_in_gt = gt_rgb.clamp(0.0, 1.0)
        perceptual_val = P_loss(perceptual_in_pred, perceptual_in_gt)[0]
        if opt.legacy_perceptual_scaling:
            # P_loss already scales by perceptual_weight (P_weight); do not multiply P_weight again here.
            perceptual_term = 2.0 * perceptual_val
        else:
            perceptual_term = float(opt.perceptual_repeat) * perceptual_val
        loss_rgb = (
            L1_loss(output_rgb, gt_rgb)
            + D_loss(output_rgb, gt_rgb)
            + E_loss(output_rgb, gt_rgb)
            + perceptual_term
        )
        loss = loss_rgb
        if DepthEdge_loss is not None:
            loss = loss + opt.depth_edge_weight * DepthEdge_loss(output_rgb, depth_low)
        if DepthSmooth_loss is not None:
            loss = loss + opt.depth_smooth_weight * DepthSmooth_loss(output_rgb, depth_low)
        iter_idx += 1
        loss.backward()
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip_max_norm, norm_type=2)
        optimizer.step()
        loss_detached = loss.detach()
        loss_print += loss_detached
        loss_last_10 += loss_detached
        pic_cnt += 1
        pic_last_10 += 1
        if iter_idx == train_len:
            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss_batch = float((loss_last_10 / max(1, pic_last_10)).item())
            logging.info(f"[Train] Epoch[{epoch}] | Batch Loss: {avg_loss_batch:.4f} | LR: {current_lr:.6f}")
            # Avoid per-epoch disk I/O; only write quick visuals on snapshot epochs.
            if int(epoch) % int(opt.snapshots) == 0:
                output_img = transforms.ToPILImage()((output_rgb)[0].detach().float().cpu().squeeze(0))
                gt_img = transforms.ToPILImage()((gt_rgb)[0].detach().float().cpu().squeeze(0))
                train_vis_dir = os.path.join(_weights_train_dir(), "preview")
                os.makedirs(train_vis_dir, exist_ok=True)
                output_img.save(os.path.join(train_vis_dir, "test.png"))
                gt_img.save(os.path.join(train_vis_dir, "gt.png"))
    return float(loss_print.item()), int(pic_cnt)


def save_best_model(epoch, current_psnr, prev_best_psnr):
    weight_dir = _weights_train_dir()
    best_model_path = os.path.join(weight_dir, f"best_psnr_model_epoch_{epoch}_psnr_{current_psnr:.4f}.pth")
    for file in os.listdir(weight_dir):
        if file.startswith("best_psnr_model") and file != os.path.basename(best_model_path):
            os.remove(os.path.join(weight_dir, file))
    torch.save(_get_raw_model(model).state_dict(), best_model_path)
    logging.info(f"[Best Model] Updated! Epoch {epoch} | PSNR: {current_psnr:.4f} (prev best: {prev_best_psnr:.4f})")
    logging.info(f"[Best Model] Saved to: {best_model_path}")
    return best_model_path


def checkpoint(epoch):
    d = _weights_train_dir()
    model_out_path = os.path.join(d, f"epoch_{epoch}.pth")
    payload = {
        "epoch": int(epoch),
        "model_state_dict": _get_raw_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(payload, model_out_path)
    logging.info(f"[Checkpoint] Saved full training state (model+optimizer+scheduler) -> {model_out_path}")
    return model_out_path


def load_datasets():
    logging.info(f"[Dataset] Loading dataset: {opt.dataset}")
    sem_tr = opt.dinov3_cache_dir.strip() if opt.use_dinov3 else None
    sem_va = (opt.dinov3_cache_dir_val.strip() or opt.dinov3_cache_dir.strip()) if opt.use_dinov3 else None
    dino_supported = ("lol_v1", "isaid_dark")
    if opt.use_dinov3 and opt.dataset not in dino_supported:
        raise ValueError(
            f"use_dinov3=True only implemented for {dino_supported}; current dataset={opt.dataset}"
        )
    sem_eval = sem_va if (opt.use_dinov3 and opt.dataset in dino_supported) else None
    c = opt.cropSize
    ds = opt.dataset
    if ds == "lol_v1":
        train_set = get_lol_training_set(opt.data_train_lol_v1, size=c, semantic_cache_dir=sem_tr)
        test_set = get_eval_set(opt.data_val_lol_v1, semantic_cache_dir=sem_eval)
    elif ds == "isaid_dark":
        train_set = get_isaid_dark_training_set(
            opt.data_train_isaid, size=c, semantic_cache_dir=sem_tr
        )
        test_set = get_eval_set(opt.data_val_isaid, semantic_cache_dir=sem_eval)
    else:
        logging.error(f"[Dataset] Invalid dataset: {opt.dataset}")
        raise ValueError("Invalid --dataset. Supported: ('lol_v1', 'isaid_dark')")
    train_loader_kwargs = {
        "dataset": train_set,
        "num_workers": opt.threads,
        "batch_size": opt.batchSize,
        "shuffle": bool(opt.shuffle),
        "sampler": None,
        "pin_memory": bool(opt.gpu_mode and opt.pin_memory),
        "persistent_workers": bool(opt.persistent_workers and opt.threads > 0),
        "collate_fn": safe_collate,
    }
    if opt.threads > 0:
        train_loader_kwargs["prefetch_factor"] = max(1, int(opt.prefetch_factor))
    training_data_loader = DataLoader(**train_loader_kwargs)
    eval_workers = max(1, min(4, int(opt.threads))) if int(opt.threads) > 0 else 0
    test_loader_kwargs = {
        "dataset": test_set,
        "num_workers": eval_workers,
        "batch_size": 1,
        "shuffle": False,
        "pin_memory": bool(opt.gpu_mode and opt.pin_memory),
        "persistent_workers": False,
        "collate_fn": safe_collate,
    }
    if eval_workers > 0:
        test_loader_kwargs["prefetch_factor"] = max(1, int(opt.prefetch_factor))
    testing_data_loader = DataLoader(**test_loader_kwargs)
    logging.info(f"[Dataset] Train set size: {len(train_set)} | Test set size: {len(test_set)}")
    return training_data_loader, testing_data_loader


def _get_raw_model(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _normalize_state_dict_for_model(model: torch.nn.Module, state_dict: dict) -> dict:
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    model_has_module = any(k.startswith("module.") for k in model.state_dict().keys())
    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())
    if ckpt_has_module and not model_has_module:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if model_has_module and not ckpt_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def _load_state_dict_compatible(model: torch.nn.Module, ckpt_obj, strict: bool):
    state_dict = _extract_state_dict(ckpt_obj)
    state_dict = _normalize_state_dict_for_model(model, state_dict)
    return model.load_state_dict(state_dict, strict=strict)


def _is_full_training_ckpt(obj) -> bool:
    return (
        isinstance(obj, dict)
        and "model_state_dict" in obj
        and "optimizer_state_dict" in obj
        and "scheduler_state_dict" in obj
    )


def apply_training_checkpoint(model, optimizer, scheduler):
    load_strict = True
    target = _get_raw_model(model)
    if opt.resume_path:
        if not os.path.exists(opt.resume_path):
            logging.error(f"[Model] resume_path not found: {opt.resume_path}")
            raise FileNotFoundError(f"resume_path {opt.resume_path} not exists")
        data = torch.load(
            opt.resume_path, map_location=lambda storage, loc: storage, weights_only=False
        )
        if _is_full_training_ckpt(data):
            _load_state_dict_compatible(target, data["model_state_dict"], load_strict)
            optimizer.load_state_dict(data["optimizer_state_dict"])
            scheduler.load_state_dict(data["scheduler_state_dict"])
            ep = int(data.get("epoch", 0))
            logging.info(
                f"[Resume] Full checkpoint loaded: model+optimizer+scheduler | resume_path={opt.resume_path} | last_epoch={ep}"
            )
            return ep
        _load_state_dict_compatible(target, data, load_strict)
        logging.warning(
            "[Resume] resume_path contains only model weights (no optimizer/scheduler); training will continue with current optimizer/scheduler state. If this is not intended, use a full checkpoint or set --start_epoch explicitly."
        )
        logging.info(f"[Model] Loaded weights from resume_path: {opt.resume_path}")
        return None
    if opt.start_epoch > 0:
        pth_path = os.path.join(_weights_train_dir(), f"epoch_{opt.start_epoch}.pth")
        if not os.path.exists(pth_path):
            legacy_path = os.path.join(_legacy_weights_train_dir(), f"epoch_{opt.start_epoch}.pth")
            if os.path.exists(legacy_path):
                logging.warning(f"[Resume] Falling back to legacy checkpoint dir: {legacy_path}")
                pth_path = legacy_path
            else:
                logging.error(f"[Model] Pretrained weight not found: {pth_path}")
                raise FileNotFoundError(f"Pretrained file {pth_path} not exists")
        data = torch.load(pth_path, map_location=lambda storage, loc: storage, weights_only=False)
        if _is_full_training_ckpt(data):
            _load_state_dict_compatible(target, data["model_state_dict"], load_strict)
            optimizer.load_state_dict(data["optimizer_state_dict"])
            scheduler.load_state_dict(data["scheduler_state_dict"])
            ep = int(data.get("epoch", opt.start_epoch))
            logging.info(f"[Resume] Full checkpoint loaded from epoch file | path={pth_path} | last_epoch={ep}")
            return ep
        _load_state_dict_compatible(target, data, load_strict)
        logging.info(f"[Model] Loaded legacy weights-only: {pth_path}")
        return opt.start_epoch
    return None


def build_model():
    logging.info("[Model] Building FSNet model...")
    base_model = MST_Plus_Plus(
        use_freq_branch=opt.use_freq_branch,
        freq_inject_pos=opt.freq_inject_pos,
        fft_mode=opt.fft_mode,
        freq_weight=opt.freq_weight,
        freq_blocks=opt.freq_blocks,
        use_semantic_prior=opt.use_dinov3,
        dino_sem_channels=opt.dinov3_sem_channels,
        semantic_fusion_weight=opt.semantic_fusion_weight,
        use_dp_caa=opt.use_dp_caa,
        dp_caa_window=opt.dp_caa_window,
        dp_caa_sem_embed=opt.dp_caa_sem_embed,
        dp_caa_tau=opt.dp_caa_tau,
    ).to(DEVICE)
    logging.info("[Model] Initialized (weights are loaded by apply_training_checkpoint based on resume_path/start_epoch)")
    return base_model


def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    logging.info(f"[Optimizer] Adam optimizer initialized | Initial LR: {opt.lr:.6f}")
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[(opt.nEpochs // 4) - opt.warmup_epochs, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[1e-5, 1e-6],
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=opt.warmup_epochs,
                after_scheduler=scheduler_step,
                start_factor=float(opt.warmup_start_factor),
            )
            logging.info(
                f"[Scheduler] CosineRestartCyclicLR + Warmup "
                f"(epochs: {opt.warmup_epochs}, start_factor: {float(opt.warmup_start_factor):.3f})"
            )
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[opt.nEpochs // 4, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[1e-5, 1e-6],
            )
            logging.info("[Scheduler] CosineRestartCyclicLR")
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch],
                restart_weights=[1],
                eta_min=1e-7,
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=opt.warmup_epochs,
                after_scheduler=scheduler_step,
                start_factor=float(opt.warmup_start_factor),
            )
            logging.info(
                f"[Scheduler] CosineRestartLR + Warmup "
                f"(epochs: {opt.warmup_epochs}, start_factor: {float(opt.warmup_start_factor):.3f})"
            )
        else:
            scheduler = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.start_epoch],
                restart_weights=[1],
                eta_min=1e-7,
            )
            logging.info("[Scheduler] CosineRestartLR")
    else:
        logging.error("[Scheduler] No valid scheduler selected")
        raise ValueError("should choose a scheduler")
    return optimizer, scheduler


def init_loss():
    L1_loss = L1Loss(loss_weight=opt.L1_weight, reduction="mean").to(DEVICE)
    D_loss = SSIM(weight=opt.D_weight).to(DEVICE)
    E_loss = EdgeLoss(loss_weight=opt.E_weight).to(DEVICE)
    # Data pipeline uses ToTensor() -> [0, 1]. range_norm=True would apply (x+1)/2 and wrongly
    # squeeze inputs into [0.5, 1] before ImageNet mean/std; keep False to match VGG pretrain convention.
    P_loss = PerceptualLoss(
        {"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},
        perceptual_weight=opt.P_weight,
        criterion="mse",
        range_norm=False,
    ).to(DEVICE)
    if opt.depth_edge_weight > 0:
        depth_edge_loss = DepthEdgeConsistencyLoss(edge_consistency_coeff=1.0, grad_method="sobel").to(DEVICE)
    else:
        depth_edge_loss = None
    if opt.depth_smooth_weight > 0:
        depth_smooth_loss = DepthRegionSmoothLoss(smooth_eps=0.1, kernel_size=3).to(DEVICE)
    else:
        depth_smooth_loss = None
    logging.info(
        f"[Loss] Initialized loss functions | L1: {opt.L1_weight} | SSIM: {opt.D_weight} | "
        f"Edge: {opt.E_weight} | Perceptual: {opt.P_weight} (VGG range_norm=False, input [0,1]) | "
        f"LegacyPerceptual: {opt.legacy_perceptual_scaling} | PerceptualRepeat: {opt.perceptual_repeat} | "
        f"DepthEdge: {opt.depth_edge_weight} | DepthSmooth: {opt.depth_smooth_weight}"
    )
    return L1_loss, P_loss, E_loss, D_loss, depth_edge_loss, depth_smooth_loss


if __name__ == "__main__":
    def init_logging():
        val_dir = opt.val_folder
        os.makedirs(val_dir, exist_ok=True)
        log_dir = _weights_train_dir()
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path, mode="a", encoding="utf-8", delay=False),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
        sys.stdout.flush()
        logging.info("=" * 60)
        logging.info(f"[Log] Log system initialized | Log file: {log_path}")
        logging.info(f"[Log] Log directory writeable: {os.access(log_dir, os.W_OK)}")
        return log_path

    def init_tensorboard():
        tb_log_dir = os.path.join("data_dir", "tb_logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        writer = SummaryWriter(log_dir=tb_log_dir)
        logging.info(f"[TensorBoard] Initialized | Log dir: {tb_log_dir}")
        logging.info(
            f"[TensorBoard] Remote access: Run `tensorboard --logdir={tb_log_dir} --host 0.0.0.0 --port 6006`"
        )
        return writer

    log_path = init_logging()
    tb_writer = None
    train_init()
    tb_writer = init_tensorboard()

    logging.info("=" * 60)
    logging.info("[Experiment Config]")
    logging.info(
        f"Dataset: {opt.dataset} | LR: {opt.lr:.6f} | Batch Size: {opt.batchSize} | Crop Size: {opt.cropSize}"
    )
    logging.info(
        f"Total Epochs: {opt.nEpochs} | Start Epoch: {opt.start_epoch + 1} | Snapshot Interval: {opt.snapshots}"
    )
    logging.info(f"Gamma: {'Enabled' if opt.gamma else 'Disabled'} | Grad Clip: {'Enabled' if opt.grad_clip else 'Disabled'}")
    logging.info(
        f"Resume Path: {opt.resume_path if opt.resume_path else 'None'} | "
        f"Best Metric: {opt.best_metric} | "
        f"Legacy Perceptual: {opt.legacy_perceptual_scaling} | "
        f"Perceptual Repeat: {opt.perceptual_repeat} | "
        f"Grad Clip Max Norm: {opt.grad_clip_max_norm}"
    )
    logging.info(
        f"Freq Branch: {'ON' if opt.use_freq_branch else 'OFF'} | inject={opt.freq_inject_pos} | "
        f"fft_mode={opt.fft_mode} | freq_weight={opt.freq_weight} | freq_blocks={opt.freq_blocks} | "
        f"freq_loss_weight={opt.freq_loss_weight}"
    )
    logging.info(
        f"DINOv3 prior: {'ON' if opt.use_dinov3 else 'OFF'} | cache_train={opt.dinov3_cache_dir or '-'} | "
        f"cache_val={opt.dinov3_cache_dir_val or '(reuse train)'} | sem_C={opt.dinov3_sem_channels} | "
        f"sem_weight={opt.semantic_fusion_weight}"
    )
    logging.info(
        f"DP-CAA: {'ON' if opt.use_dp_caa else 'OFF'} | window={opt.dp_caa_window} | "
        f"sem_embed={opt.dp_caa_sem_embed} | tau={opt.dp_caa_tau}"
    )
    logging.info(f"Val image dump: {'ON' if opt.save_val_images else 'OFF'}")
    logging.info("=" * 60)

    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer, scheduler = make_scheduler()
    resumed_epoch = apply_training_checkpoint(model, optimizer, scheduler)
    L1_loss, P_loss, E_loss, D_loss, DepthEdge_loss, DepthSmooth_loss = init_loss()

    psnr_raw, ssim_raw, lpips_raw = [], [], []
    start_epoch = opt.start_epoch if opt.start_epoch > 0 else 0
    if resumed_epoch is not None:
        start_epoch = resumed_epoch
    dataset_tag = opt.dataset
    if opt.resume_path and resumed_epoch is None and start_epoch == 0:
        logging.warning(
            "[Train] resume_path was provided but no epoch/optimizer state was restored. Training starts from epoch 1. If you expect resume behavior, use a full checkpoint epoch_*.pth or pass --start_epoch."
        )
    val_dir = opt.val_folder
    os.makedirs(val_dir, exist_ok=True)
    logging.info(f"[Train] Start training from epoch {start_epoch + 1} to {start_epoch + opt.nEpochs}")
    val_gamma = ((opt.start_gamma + opt.end_gamma) / 200.0) if opt.gamma else 1.0
    logging.info(f"[Val] Eval gamma set to {val_gamma:.3f}")
    if opt.best_metric != "raw":
        logging.warning("[Val] best_metric=%s is ignored; validation now computes raw metrics only.", opt.best_metric)

    best_psnr = -float("inf")
    best_epoch = -1
    best_model_path = ""

    try:
        for epoch in range(start_epoch + 1, start_epoch + opt.nEpochs + 1):
            epoch_total_loss, epoch_pic_num = train(epoch)
            avg_train_loss = epoch_total_loss / max(1, epoch_pic_num)
            current_lr = optimizer.param_groups[0]["lr"]
            if tb_writer is not None:
                tb_writer.add_scalar("Train/Average_Loss", avg_train_loss, epoch)
                tb_writer.add_scalar("Train/Learning_Rate", current_lr, epoch)
            logging.info(f"[Train Summary] Epoch[{epoch}] | Avg Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")
            scheduler.step()
            if epoch % opt.snapshots == 0:
                model_out_path = checkpoint(epoch)
                sub, gt_key, norm_size = _VAL_SNAPSHOT.get(opt.dataset, ("", "", True))
                label_dir = getattr(opt, gt_key, "") if gt_key else ""
                if not label_dir or not os.path.isdir(label_dir):
                    raise FileNotFoundError(
                        f"[Val] GT label_dir is invalid for dataset={opt.dataset}: '{label_dir}'"
                    )
                val_save_dir = os.path.join(val_dir, sub)
                os.makedirs(val_save_dir, exist_ok=True)
                val_sem_scale = float(get_sem_lambda(epoch)) if opt.use_dinov3 else 1.0
                avg_psnr_raw, avg_ssim_raw, avg_lpips_raw = eval(
                    _get_raw_model(model),
                    testing_data_loader,
                    model_out_path,
                    val_save_dir,
                    norm_size=norm_size,
                    LOL=(opt.dataset == "lol_v1"),
                    v2=False,
                    alpha=0.8,
                    gamma=val_gamma,
                    use_freq_branch=opt.use_freq_branch,
                    use_dinov3=opt.use_dinov3,
                    sem_channels=opt.dinov3_sem_channels,
                    empty_cache_interval=int(getattr(opt, "eval_empty_cache_interval", 0)),
                    reload_weights=False,
                    sem_scale=val_sem_scale,
                    metric_label_dir=label_dir,
                    metric_mode="raw",
                    save_outputs=bool(opt.save_val_images),
                )
                psnr_raw.append(avg_psnr_raw)
                ssim_raw.append(avg_ssim_raw)
                lpips_raw.append(avg_lpips_raw)
                select_psnr = avg_psnr_raw
                if select_psnr > best_psnr:
                    prev_best = best_psnr
                    best_psnr = select_psnr
                    best_epoch = epoch
                    best_model_path = save_best_model(epoch, select_psnr, prev_best)
                if tb_writer is not None:
                    tb_writer.add_scalar("Val/PSNR_raw", avg_psnr_raw, epoch)
                    tb_writer.add_scalar("Val/SSIM_raw", avg_ssim_raw, epoch)
                    tb_writer.add_scalar("Val/LPIPS_raw", avg_lpips_raw, epoch)
                    tb_writer.add_scalar("Val/Best_PSNR", best_psnr, epoch)
                val_msg_raw = (
                    f"[Val] Epoch {epoch} | w/o GT mean | PSNR: {avg_psnr_raw:.4f} dB | "
                    f"SSIM: {avg_ssim_raw:.4f} | LPIPS: {avg_lpips_raw:.4f}"
                )
                print(val_msg_raw)
                logging.info(val_msg_raw)
                best_psnr_label = "w/o GT mean"
                best_msg = f"[Val] Current Best PSNR ({best_psnr_label}): {best_psnr:.4f} (Epoch {best_epoch})"
                print(best_msg)
                logging.info(best_msg)
                sys.stdout.flush()
            if opt.empty_cache_each_epoch and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if tb_writer is not None:
            tb_writer.close()
        best_psnr_label = "w/o GT mean"
        final_msg = (
            f"[Train End] Training completed | Log file: {log_path} | "
            f"Best PSNR ({best_psnr_label}): {best_psnr:.4f} (Epoch {best_epoch})"
        )
        print(final_msg)
        logging.info(final_msg)
        if tb_writer is not None:
            logging.info(f"[Train End] TensorBoard logs saved to: {tb_writer.log_dir}")
        logging.info("=" * 60)
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        metric_md_dir = os.path.join(val_dir, "weights", "training")
        os.makedirs(metric_md_dir, exist_ok=True)
        metric_md_path = os.path.join(metric_md_dir, f"metrics_{now}.md")
        with open(metric_md_path, "w") as f:
            f.write("# Training Metrics\n")
            f.write(f"Dataset: {dataset_tag}\n")
            f.write(f"LR: {opt.lr}\n")
            f.write(f"Batch Size: {opt.batchSize}\n")
            f.write(f"Crop Size: {opt.cropSize}\n")
            f.write(
                f"L1 Weight: {opt.L1_weight} | SSIM Weight: {opt.D_weight} | "
                f"Edge Weight: {opt.E_weight} | Perceptual Weight: {opt.P_weight}\n\n"
            )
            best_psnr_label = "w/o GT mean"
            f.write("## Best Model Info\n")
            f.write(f"- Best Epoch: {best_epoch}\n")
            f.write(f"- Best PSNR ({best_psnr_label}): {best_psnr:.4f} dB\n")
            f.write(f"- Model Path: {best_model_path}\n\n")
            f.write("| Epoch | PSNR (raw) | SSIM (raw) | LPIPS (raw) |\n")
            f.write("|-------|------------|------------|-------------|\n")
            for i in range(len(psnr_raw)):
                epoch_num = start_epoch + (i + 1) * opt.snapshots
                f.write(
                    f"| {epoch_num} | {psnr_raw[i]:.4f} | {ssim_raw[i]:.4f} | {lpips_raw[i]:.4f} |\n"
                )
        logging.info(f"[Metric] Final metrics saved to: {metric_md_path}")
        print(f"Metrics file saved to: {metric_md_path}")

