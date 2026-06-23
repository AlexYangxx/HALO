#!/usr/bin/env python3
"""
Offline depth (DA3) for two layouts under one script:

  * isaid — iSAID-dark / iSAID2: train/low, val/low, optional train/gt -> high_depth
  * lol   — LOL / LOL-v1: root contains our485/ and eval15/; our485/low, eval15/low,
            optional our485/high -> high_depth

Depth npz is saved at each input image's native (W, H).

Examples (from repo root):

  python scripts/prepare_depth.py --dataset isaid
  python scripts/prepare_depth.py --dataset isaid --root path/to/iSAID-dark --skip-high-depth

  python scripts/prepare_depth.py --dataset lol
  python scripts/prepare_depth.py --dataset lol --root data_dir/LOL-v1 --skip-high-depth
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Tuple

# Ensure repo-root modules are importable when this script is run from scripts/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


DEFAULT_MODEL = "depth-anything/DA3MONO-LARGE"

Job = Tuple[str, str]  # (src_rel, dst_rel)
Pair = Tuple[str, str]  # (img_dir_rel, npz_dir_rel)


def _resolve_root(dataset: str, root_arg: str) -> str:
    raw = (root_arg or "").strip()
    if raw:
        if not os.path.isdir(raw):
            raise FileNotFoundError(f"--root is not a directory: {raw!r}")
        return os.path.abspath(raw)

    if dataset == "isaid":
        candidates = ["data_dir/iSAID-dark", "data_dir/iSAID2"]
        for c in candidates:
            if os.path.isdir(os.path.join(c, "train", "low")):
                print(f"[INFO] using default root: {c}")
                return os.path.abspath(c)
        raise FileNotFoundError(
            f"No default iSAID root found (need train/low). Tried: {candidates}"
        )

    # lol
    candidates = ["data_dir/LOL-v1", "data_dir"]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "our485", "low")):
            print(f"[INFO] using default root: {c}")
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"No default LOL root found (need our485/low). Tried: {candidates}"
    )


def _check_dirs(root: str, paths: List[str]) -> None:
    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError("Missing required directories:\n  " + "\n  ".join(missing))


def _isaid_layout(root: str, skip_high_depth: bool) -> Tuple[List[Job], List[Pair]]:
    req = [
        os.path.join(root, "train", "low"),
        os.path.join(root, "val", "low"),
    ]
    if not skip_high_depth:
        req.append(os.path.join(root, "train", "gt"))
    _check_dirs(root, req)

    jobs: List[Job] = [
        ("train/low", "train/low_depth"),
        ("val/low", "val/low_depth"),
    ]
    if not skip_high_depth:
        jobs.insert(1, ("train/gt", "train/high_depth"))

    pairs: List[Pair] = [
        ("train/low", "train/low_depth/depth_maps"),
        ("val/low", "val/low_depth/depth_maps"),
    ]
    if not skip_high_depth:
        pairs.insert(1, ("train/gt", "train/high_depth/depth_maps"))
    return jobs, pairs


def _lol_layout(root: str, skip_high_depth: bool) -> Tuple[List[Job], List[Pair]]:
    req = [
        os.path.join(root, "our485", "low"),
        os.path.join(root, "eval15", "low"),
    ]
    if not skip_high_depth:
        req.append(os.path.join(root, "our485", "high"))
    _check_dirs(root, req)

    jobs: List[Job] = [
        ("our485/low", "our485/low_depth"),
        ("eval15/low", "eval15/low_depth"),
    ]
    if not skip_high_depth:
        jobs.insert(1, ("our485/high", "our485/high_depth"))

    pairs: List[Pair] = [
        ("our485/low", "our485/low_depth/depth_maps"),
        ("eval15/low", "eval15/low_depth/depth_maps"),
    ]
    if not skip_high_depth:
        pairs.insert(1, ("our485/high", "our485/high_depth/depth_maps"))
    return jobs, pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare depth maps: iSAID-dark/iSAID2 or LOL/LOL-v1 (depth_estimation.generate_depth_map)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("isaid", "lol"),
        default="isaid",
        help="isaid: train+val under root; lol: our485+eval15 under root (e.g. data_dir or data_dir/LOL-v1).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Dataset root. Empty: isaid -> data_dir/iSAID-dark|iSAID2; lol -> data_dir/LOL-v1|data_dir.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id for depth generation")
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=8,
        help="Sets OMP_NUM_THREADS before importing depth backend",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only run the glob count check, skip generate_depth_map",
    )
    parser.add_argument(
        "--skip-high-depth",
        action="store_true",
        help="Skip GT/high branch (isaid: train/gt; lol: our485/high).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="DepthAnything3 model source: HF repo id or local snapshot directory.",
    )
    args = parser.parse_args()

    if args.omp_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    root = _resolve_root(args.dataset, args.root)
    if args.dataset == "isaid":
        jobs, pairs = _isaid_layout(root, bool(args.skip_high_depth))
    else:
        jobs, pairs = _lol_layout(root, bool(args.skip_high_depth))

    print(f"[INFO] dataset={args.dataset} root={root}")
    print("[INFO] Depth npz saved at each image's native resolution (W, H).")

    if not args.skip_generate:
        import depth_estimation as d

        for src, dst in jobs:
            print(f">>> {src} -> {dst}")
            d.generate_depth_map(
                input_dir=os.path.join(root, src),
                output_dir=os.path.join(root, dst),
                model_name=args.model_name,
                target_size=None,
                gpu_id=args.gpu_id,
                keep_input_size=True,
            )

    for a, b in pairs:
        img_cnt = len(glob.glob(os.path.join(root, a, "*")))
        npz_cnt = len(glob.glob(os.path.join(root, b, "*_depth.npz")))
        print(a, img_cnt, "|", b, npz_cnt)

    return 0


if __name__ == "__main__":
    sys.exit(main())
