import os
import torch
import torch.utils.data as data
import torch.nn.functional as F
from os import listdir
from os.path import join

from data.util import is_image_file, load_img, load_depth_npz
from data.semantic_util import load_semantic_npy


def get_depth_filename(img_filename, depth_ext=".npz"):
    """Map image filename to depth filename, e.g. xxx.png -> xxx_depth.npz."""
    name, _ = os.path.splitext(img_filename)
    return f"{name}_depth{depth_ext}"


class SICEDatasetFromFolderEval(data.Dataset):
    """
    Eval dataset with depth and original size return.
    Return tuple:
      (input_img, img_file, h_ori, w_ori, depth_low, depth_file)
      or with semantic_cache_dir append sem_feat (padded like input).
    """

    def __init__(self, data_dir, transform=None, depth_dir=None, semantic_cache_dir=None):
        super(SICEDatasetFromFolderEval, self).__init__()
        self.img_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        self.img_filenames.sort()
        self.transform = transform
        self.semantic_cache_dir = semantic_cache_dir

        if depth_dir is None:
            self.depth_dir = join(os.path.dirname(data_dir), "low_depth", "depth_maps")
        else:
            self.depth_dir = depth_dir

        self.depth_filenames = []
        for img_path in self.img_filenames:
            _, img_filename = os.path.split(img_path)
            depth_filename = get_depth_filename(img_filename)
            depth_path = join(self.depth_dir, depth_filename)
            if os.path.exists(depth_path):
                self.depth_filenames.append(depth_path)
            else:
                raise FileNotFoundError(
                    f"Depth file not found: {depth_path} (check image-depth filename mapping)"
                )

        assert len(self.img_filenames) == len(self.depth_filenames), (
            f"Image count ({len(self.img_filenames)}) != depth count ({len(self.depth_filenames)})"
        )

    def __getitem__(self, index):
        img_path = self.img_filenames[index]
        input_img = load_img(img_path)
        _, img_file = os.path.split(img_path)
        h_ori, w_ori = input_img.size[1], input_img.size[0]

        depth_path = self.depth_filenames[index]
        depth_low = load_depth_npz(depth_path)
        _, depth_file = os.path.split(depth_path)

        sem = None
        if self.semantic_cache_dir:
            sem = load_semantic_npy(self.semantic_cache_dir, img_file)

        if self.transform:
            input_img = self.transform(input_img)
            depth_low = self.transform(depth_low)

            factor = 8
            h, w = input_img.shape[1], input_img.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = F.pad(input_img.unsqueeze(0), (0, padw, 0, padh), "reflect").squeeze(0)
            depth_low = F.pad(depth_low.unsqueeze(0), (0, padw, 0, padh), "reflect").squeeze(0)
            if sem is not None:
                sem = F.pad(sem.unsqueeze(0), (0, padw, 0, padh), "reflect").squeeze(0)

        if self.semantic_cache_dir:
            return input_img, img_file, h_ori, w_ori, depth_low, depth_file, sem
        return input_img, img_file, h_ori, w_ori, depth_low, depth_file

    def __len__(self):
        return len(self.img_filenames)


class DatasetFromFolderEval(data.Dataset):
    """
    Eval dataset with depth return.
    Return tuple:
      (input_img, img_file, depth_low, depth_file) or with semantic_cache_dir:
      (input_img, img_file, depth_low, depth_file, sem_feat)
    """

    def __init__(self, data_dir, transform=None, depth_dir=None, semantic_cache_dir=None):
        super(DatasetFromFolderEval, self).__init__()
        self.img_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        self.img_filenames.sort()
        self.transform = transform
        self.semantic_cache_dir = semantic_cache_dir

        if depth_dir is None:
            self.depth_dir = join(os.path.dirname(data_dir), "low_depth", "depth_maps")
        else:
            self.depth_dir = depth_dir

        self.depth_filenames = []
        for img_path in self.img_filenames:
            _, img_filename = os.path.split(img_path)
            depth_filename = get_depth_filename(img_filename)
            depth_path = join(self.depth_dir, depth_filename)
            if os.path.exists(depth_path):
                self.depth_filenames.append(depth_path)
            else:
                raise FileNotFoundError(
                    f"Depth file not found: {depth_path} (check image-depth filename mapping)"
                )

        assert len(self.img_filenames) == len(self.depth_filenames), (
            f"Image count ({len(self.img_filenames)}) != depth count ({len(self.depth_filenames)})"
        )

    def __getitem__(self, index):
        img_path = self.img_filenames[index]
        input_img = load_img(img_path)
        _, img_file = os.path.split(img_path)

        depth_path = self.depth_filenames[index]
        depth_low = load_depth_npz(depth_path)
        _, depth_file = os.path.split(depth_path)

        sem = None
        if self.semantic_cache_dir:
            sem = load_semantic_npy(self.semantic_cache_dir, img_file)

        if self.transform:
            input_img = self.transform(input_img)
            depth_low = self.transform(depth_low)
        if self.semantic_cache_dir:
            return input_img, img_file, depth_low, depth_file, sem
        return input_img, img_file, depth_low, depth_file

    def __len__(self):
        return len(self.img_filenames)


