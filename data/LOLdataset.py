import os
import random
import torch.utils.data as data
from os import listdir
from os.path import join

from torchvision import transforms as t
from torchvision.transforms import functional as TF

from data.util import is_image_file, load_depth_npz, load_img
from data.semantic_util import augment_training_pair_with_sem, load_semantic_npy


def _depth_filename(img_filename: str) -> str:
    stem, _ = os.path.splitext(img_filename)
    return f"{stem}_depth.npz"


def augment_training_pair_no_sem(im1, im2, depth_low, crop_size):
    """Synchronized RandomCrop + h/v flip + ToTensor for RGB/depth triplet."""
    top, left, height, width = t.RandomCrop.get_params(im1, (crop_size, crop_size))

    im1 = TF.crop(im1, top, left, height, width)
    im2 = TF.crop(im2, top, left, height, width)
    depth_low = TF.crop(depth_low, top, left, height, width)

    if random.random() < 0.5:
        im1 = TF.hflip(im1)
        im2 = TF.hflip(im2)
        depth_low = TF.hflip(depth_low)
    if random.random() < 0.5:
        im1 = TF.vflip(im1)
        im2 = TF.vflip(im2)
        depth_low = TF.vflip(depth_low)

    im1 = TF.to_tensor(im1)
    im2 = TF.to_tensor(im2)
    depth_low = TF.to_tensor(depth_low)
    depth_high = depth_low
    return im1, im2, depth_low, depth_high


class LOLDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None, crop_size=256, semantic_cache_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size
        self.semantic_cache_dir = semantic_cache_dir
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.img_low_dir = join(data_dir, "low")
        self.img_high_dir = join(data_dir, "high")
        self.depth_low_dir = join(data_dir, "low_depth", "depth_maps")
        self.depth_high_dir = join(data_dir, "high_depth", "depth_maps")

        self.img_low_filenames = sorted([f for f in listdir(self.img_low_dir) if is_image_file(f)])
        self.img_high_filenames = sorted([f for f in listdir(self.img_high_dir) if is_image_file(f)])
        assert len(self.img_low_filenames) == len(self.img_high_filenames), "Low/High image count mismatch!"

    def __getitem__(self, index):
        img_low_name = self.img_low_filenames[index]
        img_high_name = self.img_high_filenames[index]

        im1 = load_img(join(self.img_low_dir, img_low_name))
        im2 = load_img(join(self.img_high_dir, img_high_name))

        depth_low_name = _depth_filename(img_low_name)
        depth_high_name = "none_depth_high.npz"
        depth_low = load_depth_npz(join(self.depth_low_dir, depth_low_name))
        depth_high = None

        file1 = img_low_name
        file2 = img_high_name
        file_depth_low = depth_low_name
        file_depth_high = depth_high_name

        if self.semantic_cache_dir:
            sem = load_semantic_npy(self.semantic_cache_dir, img_low_name)
            im1, im2, depth_low, depth_high, sem = augment_training_pair_with_sem(
                im1, im2, depth_low, None, sem, self.crop_size
            )
            depth_high = depth_low if depth_high is None else depth_high
            return im1, im2, depth_low, depth_high, file1, file2, file_depth_low, file_depth_high, sem

        if self.transform:
            im1, im2, depth_low, depth_high = augment_training_pair_no_sem(
                im1, im2, depth_low, self.crop_size
            )
        return im1, im2, depth_low, depth_high, file1, file2, file_depth_low, file_depth_high

    def __len__(self):
        return len(self.img_low_filenames)


class ISAIDDarkDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None, crop_size=256, semantic_cache_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size
        self.semantic_cache_dir = semantic_cache_dir

        self.img_low_dir = join(data_dir, "low")
        self.img_high_dir = join(data_dir, "gt")
        self.depth_low_dir = join(data_dir, "low_depth", "depth_maps")

        self.img_low_filenames = sorted([f for f in listdir(self.img_low_dir) if is_image_file(f)])
        self.img_high_filenames = []
        for f in self.img_low_filenames:
            gt_path = join(self.img_high_dir, f)
            if not os.path.isfile(gt_path):
                raise FileNotFoundError(f"iSAID-dark GT not found: {gt_path}")
            self.img_high_filenames.append(f)

    def __getitem__(self, index):
        img_low_name = self.img_low_filenames[index]
        img_high_name = self.img_high_filenames[index]

        im1 = load_img(join(self.img_low_dir, img_low_name))
        im2 = load_img(join(self.img_high_dir, img_high_name))

        depth_low_name = _depth_filename(img_low_name)
        depth_low = load_depth_npz(join(self.depth_low_dir, depth_low_name))

        if self.semantic_cache_dir:
            sem = load_semantic_npy(self.semantic_cache_dir, img_low_name)
            im1, im2, depth_low, _, sem = augment_training_pair_with_sem(
                im1, im2, depth_low, None, sem, self.crop_size
            )
            return im1, im2, depth_low, img_low_name, img_high_name, depth_low_name, sem

        if self.transform:
            im1, im2, depth_low, _ = augment_training_pair_no_sem(
                im1, im2, depth_low, self.crop_size
            )
        return im1, im2, depth_low, img_low_name, img_high_name, depth_low_name

    def __len__(self):
        return len(self.img_low_filenames)


class LOLv2DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None, crop_size=256, semantic_cache_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size
        self.semantic_cache_dir = semantic_cache_dir

        self.img_low_dir = join(data_dir, "Low")
        self.img_high_dir = join(data_dir, "Normal")
        self.depth_low_dir = join(data_dir, "low_depth", "depth_maps")
        self.depth_high_dir = join(data_dir, "high_depth", "depth_maps")

        self.img_low_filenames = sorted([f for f in listdir(self.img_low_dir) if is_image_file(f)])
        self.img_high_filenames = sorted([f for f in listdir(self.img_high_dir) if is_image_file(f)])
        assert len(self.img_low_filenames) == len(self.img_high_filenames), "Low/Normal image count mismatch!"

    def __getitem__(self, index):
        img_low_name = self.img_low_filenames[index]
        img_high_name = self.img_high_filenames[index]

        im1 = load_img(join(self.img_low_dir, img_low_name))
        im2 = load_img(join(self.img_high_dir, img_high_name))

        depth_low_name = _depth_filename(img_low_name)
        depth_high_name = "none_depth_high.npz"
        depth_low = load_depth_npz(join(self.depth_low_dir, depth_low_name))
        depth_high = None

        file1 = img_low_name
        file2 = img_high_name
        file_depth_low = depth_low_name
        file_depth_high = depth_high_name

        if self.semantic_cache_dir:
            sem = load_semantic_npy(self.semantic_cache_dir, img_low_name)
            im1, im2, depth_low, depth_high, sem = augment_training_pair_with_sem(
                im1, im2, depth_low, None, sem, self.crop_size
            )
            depth_high = depth_low if depth_high is None else depth_high
            return im1, im2, depth_low, depth_high, file1, file2, file_depth_low, file_depth_high, sem

        if self.transform:
            im1, im2, depth_low, depth_high = augment_training_pair_no_sem(
                im1, im2, depth_low, self.crop_size
            )
        return im1, im2, depth_low, depth_high, file1, file2, file_depth_low, file_depth_high

    def __len__(self):
        return len(self.img_low_filenames)


class LOLv2SynDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None, crop_size=256, semantic_cache_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size
        self.semantic_cache_dir = semantic_cache_dir
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.img_low_dir = join(data_dir, "Low")
        self.img_high_dir = join(data_dir, "Normal")
        self.depth_low_dir = join(data_dir, "low_depth", "depth_maps")
        self.depth_high_dir = join(data_dir, "high_depth", "depth_maps")

        self.img_low_filenames = sorted([f for f in listdir(self.img_low_dir) if is_image_file(f)])
        self.img_high_filenames = sorted([f for f in listdir(self.img_high_dir) if is_image_file(f)])
        assert len(self.img_low_filenames) == len(self.img_high_filenames), "Synthetic dataset image count mismatch!"

    def __getitem__(self, index):
        img_low_name = self.img_low_filenames[index]
        img_high_name = self.img_high_filenames[index]

        im1 = load_img(join(self.img_low_dir, img_low_name))
        im2 = load_img(join(self.img_high_dir, img_high_name))

        depth_low_name = _depth_filename(img_low_name)
        depth_high_name = "none_depth_high.npz"
        depth_low = load_depth_npz(join(self.depth_low_dir, depth_low_name))
        depth_high = None

        file1 = img_low_name
        file2 = img_high_name
        file_depth_low = depth_low_name
        file_depth_high = depth_high_name

        if self.semantic_cache_dir:
            sem = load_semantic_npy(self.semantic_cache_dir, img_low_name)
            im1, im2, depth_low, depth_high, sem = augment_training_pair_with_sem(
                im1, im2, depth_low, None, sem, self.crop_size
            )
            depth_high = depth_low if depth_high is None else depth_high
            return im1, im2, depth_low, depth_high, file1, file2, file_depth_low, file_depth_high, sem

        if self.transform:
            im1, im2, depth_low, depth_high = augment_training_pair_no_sem(
                im1, im2, depth_low, self.crop_size
            )
        return im1, im2, depth_low, depth_high, file1, file2, file_depth_low, file_depth_high

    def __len__(self):
        return len(self.img_low_filenames)

