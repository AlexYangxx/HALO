import gc
import glob
import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# Core config (can be overridden by caller args)
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DA3_ROOT = os.path.join(THIS_DIR, "Depth-Anything-3")
_DA3_SRC = os.path.join(DA3_ROOT, "src")
_DA3_API = os.path.join(_DA3_SRC, "depth_anything_3", "api.py")

INPUT_DIR = ""
OUTPUT_DIR = ""
MODEL_NAME = "depth-anything/DA3MONO-LARGE"
TARGET_GPU_ID = 0
TARGET_SIZE = (600, 400)  # (W, H)
BATCH_SIZE = 8
PROCESS_RES = 600
DEFAULT_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Reduce CUDA fragmentation unless user set this explicitly
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.6")

if not os.path.isfile(_DA3_API):
    raise ImportError(
        "Depth-Anything-3 source not found. Expected file:\n"
        f"  {_DA3_API}\n"
        "Please place official Depth-Anything-3 under repo root, or run:\n"
        "  pip install -e ./Depth-Anything-3"
    )

if _DA3_SRC not in sys.path:
    sys.path.insert(0, _DA3_SRC)
if DA3_ROOT not in sys.path:
    sys.path.insert(0, DA3_ROOT)

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def mask_invalid_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.copy()
    invalid = np.isnan(depth) | np.isinf(depth) | (depth <= 0)
    depth[invalid] = 0.0
    return depth


def resize_depth_map(depth_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    depth_img = Image.fromarray(depth_map.astype(np.float32))
    resized = depth_img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(resized).astype(np.float32)


def resize_vis_image(vis_img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    vis_pil = Image.fromarray(vis_img)
    resized = vis_pil.resize(target_size, Image.Resampling.BILINEAR)
    return np.array(resized).astype(np.uint8)


def clear_gpu_memory(gpu_id: int = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if gpu_id is not None and 0 <= gpu_id < torch.cuda.device_count():
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
            print(f"[Mem] gpu={gpu_id} allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB")


def _resolve_model_source(model_name: str) -> str:
    """
    Accept either:
    - HF repo id (e.g. depth-anything/DA3MONO-LARGE)
    - local snapshot directory containing config/model files
    - direct model file path (fallback to parent directory)
    """
    candidate = os.path.expanduser(model_name)
    if os.path.isdir(candidate):
        return candidate
    if os.path.isfile(candidate):
        return os.path.dirname(candidate)
    return model_name


def _collect_images(input_dir: str, image_ext: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for ext in image_ext:
        ext = ext.lower()
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    # unique + sorted
    return sorted(set(paths))


def _save_single_result(
    img_path: str,
    depth_map: np.ndarray,
    depth_save_dir: str,
    vis_save_dir: str,
    target_size: Optional[Tuple[int, int]],
    keep_input_size: bool = False,
) -> None:
    depth_map = mask_invalid_depth(depth_map)
    if keep_input_size:
        # Align depth output to the original RGB image size (W, H) for strict spatial correspondence.
        with Image.open(img_path) as _im:
            target_size = (_im.size[0], _im.size[1])
    if target_size is None:
        target_size = (int(depth_map.shape[1]), int(depth_map.shape[0]))
    depth_map_resized = resize_depth_map(depth_map, target_size)
    depth_vis = visualize_depth(depth_map).astype(np.uint8)
    depth_vis_resized = resize_vis_image(depth_vis, target_size)

    stem = os.path.splitext(os.path.basename(img_path))[0]
    np.savez_compressed(os.path.join(depth_save_dir, f"{stem}_depth.npz"), depth=depth_map_resized)
    imageio.imwrite(os.path.join(vis_save_dir, f"{stem}_depth_vis.png"), depth_vis_resized, quality=95)


def generate_depth_map(
    input_dir: str,
    output_dir: str,
    model_name: str = MODEL_NAME,
    image_ext: Sequence[str] = DEFAULT_IMAGE_EXT,
    gpu_id: int = TARGET_GPU_ID,
    target_size: Optional[Tuple[int, int]] = TARGET_SIZE,
    batch_size: int = BATCH_SIZE,
    process_res: int = PROCESS_RES,
    keep_input_size: bool = False,
):
    """If keep_input_size is True, each saved depth matches that image's RGB (W, H); target_size is ignored."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU {gpu_id} does not exist; total GPUs={torch.cuda.device_count()}")
        device = f"cuda:{gpu_id}"
        print(f"[GPU] Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = "cpu"
        print("[Warn] CUDA not available. Running on CPU.")

    target_desc = "input_image_size" if keep_input_size else str(target_size)
    print(f"[Cfg] batch_size={batch_size}, process_res={process_res}, target_size={target_desc}")

    depth_save_dir = os.path.join(output_dir, "depth_maps")
    vis_save_dir = os.path.join(output_dir, "depth_vis")
    os.makedirs(depth_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    resolved_model = _resolve_model_source(model_name)
    print(f"[Model] Loading from: {resolved_model}")
    model = DepthAnything3.from_pretrained(resolved_model)
    model = model.to(device=device)
    model.eval()

    image_paths = _collect_images(input_dir, image_ext)
    total = len(image_paths)
    if total == 0:
        raise ValueError(f"No images found: {input_dir}")
    total_batches = int(np.ceil(total / float(batch_size)))
    print(f"[Data] Found {total} images, {total_batches} batches")

    with torch.no_grad():
        for bidx in range(0, total, batch_size):
            batch_paths = image_paths[bidx : bidx + batch_size]
            curr_batch = bidx // batch_size + 1
            print(f"[Run] Batch {curr_batch}/{total_batches}, images={len(batch_paths)}")

            try:
                prediction = model.inference(
                    image=batch_paths,
                    process_res=process_res,
                    process_res_method="upper_bound_resize",
                    ref_view_strategy="saddle_balanced",
                )
                for i, img_path in enumerate(batch_paths):
                    _save_single_result(
                        img_path,
                        prediction.depth[i],
                        depth_save_dir,
                        vis_save_dir,
                        target_size,
                        keep_input_size=keep_input_size,
                    )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"[OOM] Batch {curr_batch} failed, fallback to single-image inference")
                clear_gpu_memory(gpu_id)
                for img_path in batch_paths:
                    single_pred = model.inference(
                        image=[img_path],
                        process_res=process_res,
                        process_res_method="upper_bound_resize",
                        ref_view_strategy="saddle_balanced",
                    )
                    _save_single_result(
                        img_path,
                        single_pred.depth[0],
                        depth_save_dir,
                        vis_save_dir,
                        target_size,
                        keep_input_size=keep_input_size,
                    )
                    clear_gpu_memory(gpu_id)

            clear_gpu_memory(gpu_id)

    print("[Done] Depth generation completed")
    print(f"[Out] depth_maps: {depth_save_dir}")
    print(f"[Out] depth_vis : {vis_save_dir}")


if __name__ == "__main__":
    if INPUT_DIR and OUTPUT_DIR:
        generate_depth_map(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            model_name=MODEL_NAME,
            gpu_id=TARGET_GPU_ID,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            process_res=PROCESS_RES,
        )
    else:
        print("Use scripts/prepare_depth.py --dataset isaid|lol [--root ...] or call generate_depth_map(...) directly.")
