#!/usr/bin/env python3
"""
Cache offline DINOv3 patch features as `.npy` files.

Online:
  python scripts/cache_dinov3_features.py --input_dir ... --output_dir ...

Offline:
  python scripts/cache_dinov3_features.py --hub_local /path/to/dinov3-main \
    --weights /path/to/dinov3_vitl16_pretrain_sat493m.pth \
    --input_dir ... --output_dir ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from tqdm import tqdm

BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
KNOWN_MODELS = (
    "dinov3_vit7b16",
    "dinov3_vith16plus",
    "dinov3_vitl16plus",
    "dinov3_vitl16",
    "dinov3_vitb16",
    "dinov3_vits16plus",
    "dinov3_vits16",
)
META_KEYS = ("input_dir", "repo", "model", "weights", "resize_mode", "img_size", "patch_size")


def list_images(root: str):
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if n.lower().endswith(IMAGE_EXTS)]


def ceil_to_multiple(v: int, m: int) -> int:
    if m <= 0:
        raise ValueError(f"multiple must be > 0, got {m}")
    return max(m, int(math.ceil(float(v) / float(m))) * m)


def compute_resize_hw(src_h: int, src_w: int, img_size: int, patch_size: int, resize_mode: str):
    if resize_mode == "fixed":
        return ceil_to_multiple(img_size, patch_size), ceil_to_multiple(img_size, patch_size)
    if resize_mode == "short_side":
        short = min(src_h, src_w)
        if short <= 0:
            raise ValueError(f"Invalid source size HxW={src_h}x{src_w}")
        scale = float(img_size) / float(short)
        h = ceil_to_multiple(int(round(src_h * scale)), patch_size)
        w = ceil_to_multiple(int(round(src_w * scale)), patch_size)
        return h, w
    raise ValueError(f"Unsupported --resize_mode: {resize_mode}")


def read_meta(meta_path: str):
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def meta_diff(old_meta: dict, new_meta: dict):
    return [k for k in META_KEYS if str(old_meta.get(k, "")) != str(new_meta.get(k, ""))]


def parse_args():
    p = argparse.ArgumentParser(description="Cache DINOv3 spatial features as .npy")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="dinov3_vitl16")
    p.add_argument("--repo", type=str, default="facebookresearch/dinov3")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--resize_mode", type=str, default="short_side", choices=["short_side", "fixed"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_shards", type=int, default=1, help="Split workload into N shards.")
    p.add_argument("--shard_id", type=int, default=0, help="Shard index in [0, num_shards).")
    p.add_argument("--hub_local", type=str, default="", help="Local DINO repo root (contains hubconf.py)")
    p.add_argument("--weights", type=str, default="", help="Local pretrained weights path")
    p.add_argument("--skip_existing", action="store_true", help="Skip existing .npy files")
    p.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Allow rebuild when existing meta mismatches; disables skip_existing in that case.",
    )
    return p.parse_args()


def maybe_infer_model(requested_model: str, weights_path: str):
    if not weights_path:
        return requested_model, False
    inferred = next((m for m in KNOWN_MODELS if m in os.path.basename(weights_path).lower()), None)
    if inferred and str(requested_model).strip().lower() != inferred:
        return inferred, True
    return requested_model, False


def safe_hub_load(torch_mod, repo_or_dir: str, model_name: str, load_kw: dict, source: str = None):
    try:
        if source is not None:
            return torch_mod.hub.load(repo_or_dir, model_name, source=source, **load_kw)
        return torch_mod.hub.load(repo_or_dir, model_name, **load_kw)
    except TypeError:
        retry_kw = dict(load_kw)
        retry_kw.pop("pretrained", None)
        if source is not None:
            return torch_mod.hub.load(repo_or_dir, model_name, source=source, **retry_kw)
        return torch_mod.hub.load(repo_or_dir, model_name, **retry_kw)


def load_model(torch_mod, args, device, weights_abs: str):
    hub_local = str(args.hub_local).strip()
    load_kw = {"pretrained": True, "trust_repo": True}
    if weights_abs:
        load_kw["weights"] = weights_abs
    if hub_local and not weights_abs:
        print("[WARN] --hub_local is set without --weights; model weights may still be downloaded.", file=sys.stderr)

    print(f"Loading {args.model} from {args.repo} on {device} ...")
    if hub_local:
        local_repo = os.path.abspath(hub_local)
        if not os.path.isfile(os.path.join(local_repo, "hubconf.py")):
            raise SystemExit(f"--hub_local must contain hubconf.py: {local_repo}")
        model = safe_hub_load(torch_mod, local_repo, args.model, load_kw, source="local")
    else:
        model = safe_hub_load(torch_mod, args.repo, args.model, load_kw)
    model.eval()
    model.to(device)
    return model


def build_run_meta(args, weights_abs: str, patch_size: int, num_shards: int):
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": os.path.abspath(args.input_dir),
        "repo": args.repo,
        "model": args.model,
        "weights": os.path.abspath(weights_abs) if weights_abs else "",
        "resize_mode": args.resize_mode,
        "img_size": int(args.img_size),
        "patch_size": int(patch_size),
        "num_shards": int(num_shards),
    }


def resolve_skip_existing(args, output_dir: str, run_meta: dict):
    meta_path = os.path.join(output_dir, "_cache_meta.json")
    existing_meta = read_meta(meta_path)
    has_npy = any(x.lower().endswith(".npy") for x in os.listdir(output_dir))
    diff = meta_diff(existing_meta, run_meta) if existing_meta else []
    if args.skip_existing:
        if has_npy and existing_meta is None and not args.force_overwrite:
            raise SystemExit(
                "Found existing .npy cache but missing/invalid _cache_meta.json under --skip_existing. "
                "Use --force_overwrite to rebuild safely, or use a fresh --output_dir."
            )
        if diff:
            if not args.force_overwrite:
                raise SystemExit(
                    "Cache meta mismatch under --skip_existing. "
                    f"Changed keys: {diff}. "
                    "Use --force_overwrite to rebuild safely, or use a new --output_dir."
                )
            print(
                "[WARN] cache meta mismatch detected; disable --skip_existing to avoid mixed cache. "
                f"Changed keys: {diff}"
            )
            return False
    return bool(args.skip_existing)


def extract_feat_map(torch_mod, model, x, patch_size: int):
    with torch_mod.no_grad():
        try:
            feat = model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=False)[0]
        except TypeError:
            feat = model.get_intermediate_layers(x, n=1)[0]
            if feat.dim() == 3:
                b, n_tokens, c = feat.shape
                gh = int(x.shape[-2]) // patch_size
                gw = int(x.shape[-1]) // patch_size
                expected = gh * gw
                if n_tokens == expected + 1:
                    feat = feat[:, 1:, :]
                    n_tokens = expected
                if n_tokens != expected:
                    raise RuntimeError(
                        f"Token-grid mismatch: n_tokens={n_tokens}, expected={expected} "
                        f"(grid={gh}x{gw}, input={x.shape[-2]}x{x.shape[-1]})"
                    )
                feat = feat.reshape(b, gh, gw, c).permute(0, 3, 1, 2)
    if feat.dim() != 4:
        raise RuntimeError(f"Unexpected feature rank={feat.dim()} shape={tuple(feat.shape)}")
    return feat.squeeze(0).float().cpu().numpy().astype(np.float32)


def cache_one(torch_mod, model, image_path: str, out_path: str, args, patch_size: int, device, to_tensor, normalize):
    with Image.open(image_path) as img:
        image = img.convert("RGB")
        src_w, src_h = image.size
        h, w = compute_resize_hw(src_h, src_w, int(args.img_size), int(patch_size), str(args.resize_mode))
        image = image.resize((w, h), resample=BICUBIC)
    x = normalize(to_tensor(image)).unsqueeze(0).to(device)
    np.save(out_path, extract_feat_map(torch_mod, model, x, patch_size))


def main() -> None:
    args = parse_args()
    import torch
    from torchvision import transforms

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = list_images(args.input_dir)
    if not image_paths:
        raise SystemExit("No images found under --input_dir")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weights_path = str(args.weights).strip()
    if weights_path:
        inferred_model, changed = maybe_infer_model(args.model, weights_path)
        if changed:
            print(
                f"[Model] Override --model: {args.model} -> {inferred_model} "
                f"(inferred from weights filename: {os.path.basename(weights_path)})"
            )
            args.model = inferred_model

    weights_abs = ""
    if weights_path:
        weights_abs = os.path.abspath(weights_path)
        if not os.path.isfile(weights_abs):
            raise SystemExit(f"Weights file not found: {weights_abs}")

    model = load_model(torch, args, device, weights_abs)
    patch_size = int(getattr(model, "patch_size", 16))
    if args.img_size <= 0:
        raise SystemExit("--img_size must be > 0")

    num_shards = int(getattr(args, "num_shards", 1) or 1)
    shard_id = int(getattr(args, "shard_id", 0) or 0)
    if num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if not (0 <= shard_id < num_shards):
        raise SystemExit("--shard_id must be in [0, num_shards)")

    run_meta = build_run_meta(args, weights_abs, patch_size, num_shards)
    skip_existing = resolve_skip_existing(args, args.output_dir, run_meta)
    meta_path = os.path.join(args.output_dir, "_cache_meta.json")
    if shard_id == 0:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=True, indent=2)

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    written = 0
    skipped = 0
    for i, path in enumerate(tqdm(image_paths, desc=f"DINOv3 cache (shard {shard_id}/{num_shards})")):
        if i % num_shards != shard_id:
            continue
        stem, _ = os.path.splitext(os.path.basename(path))
        out_path = os.path.join(args.output_dir, f"{stem}.npy")
        if skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue
        cache_one(torch, model, path, out_path, args, patch_size, device, to_tensor, normalize)
        written += 1

    print(
        f"Done. Saved under {args.output_dir} "
        f"(written={written}, skipped_existing={skipped}, shard={shard_id}/{num_shards})"
    )


if __name__ == "__main__":
    main()
