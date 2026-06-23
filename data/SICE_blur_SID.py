
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from data.util import *
from torchvision import transforms as t
import torch.nn.functional as F

class LOLBlurDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLBlurDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples_by_folder = []
        low_root = join(self.data_dir, "low_blur")
        high_root = join(self.data_dir, "high_sharp_scaled")
        for idx in range(260):
            fill_index = str(idx + 1).zfill(4)
            folder = join(low_root, fill_index)
            folder2 = join(high_root, fill_index)
            if not os.path.exists(folder):
                continue
            data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
            data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
            if len(data_filenames) == 0:
                continue
            self.samples_by_folder.append((data_filenames, data_filenames2))
        if len(self.samples_by_folder) == 0:
            raise RuntimeError(f"No valid LOLBlur samples found under: {self.data_dir}")

    def __getitem__(self, index):
        data_filenames, data_filenames2 = random.choice(self.samples_by_folder)
        num = len(data_filenames)
        index1 = random.randint(1,num)

        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_filenames2[index1-1])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        h, w, dt = hw_and_dtype_for_depth_placeholder(im1)
        depth_low = torch.zeros(1, h, w, dtype=dt)
        depth_high = torch.zeros(1, h, w, dtype=dt)
        bn1 = os.path.basename(data_filenames[index1-1])
        bn2 = os.path.basename(data_filenames2[index1-1])
        return im1, im2, depth_low, depth_high, bn1, bn2, "none_depth_low.npz", "none_depth_high.npz"

    def __len__(self):
        return 10200
    

class SIDDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SIDDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples_by_folder = []
        short_root = join(self.data_dir, "short")
        long_root = join(self.data_dir, "long")
        for idx in range(234):
            fill_index = str(idx + 1).zfill(5)
            folder = join(short_root, fill_index)
            folder2 = join(long_root, fill_index)
            if not os.path.exists(folder):
                continue
            data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
            data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
            if len(data_filenames) == 0:
                continue
            self.samples_by_folder.append((data_filenames, data_filenames2))
        if len(self.samples_by_folder) == 0:
            raise RuntimeError(f"No valid SID samples found under: {self.data_dir}")

    def __getitem__(self, index):
        data_filenames, data_filenames2 = random.choice(self.samples_by_folder)
        num = len(data_filenames)
        index1 = random.randint(1,num)


        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_filenames2[0])
        _, file1 = os.path.split(data_filenames[index1-1])
        _, file2 = os.path.split(data_filenames2[0])
        seed = np.random.randint(random.randint(1, 1000000)) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        h, w, dt = hw_and_dtype_for_depth_placeholder(im1)
        depth_low = torch.zeros(1, h, w, dtype=dt)
        depth_high = torch.zeros(1, h, w, dtype=dt)
        return im1, im2, depth_low, depth_high, file1, file2, "none_depth_low.npz", "none_depth_high.npz"

    def __len__(self):
        return 2099
    
    
    
class SICEDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SICEDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples_by_folder = []
        label_dir = join(os.path.dirname(self.data_dir), "label")
        for idx in range(591):
            fill_index = str(idx + 1)
            folder = join(self.data_dir, fill_index)
            data_gt = join(label_dir, fill_index + ".JPG")
            if not os.path.exists(folder):
                continue
            data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
            if len(data_filenames) == 0:
                continue
            self.samples_by_folder.append((data_filenames, data_gt))
        if len(self.samples_by_folder) == 0:
            raise RuntimeError(f"No valid SICE samples found under: {self.data_dir}")

    def __getitem__(self, index):
        data_filenames, data_gt = random.choice(self.samples_by_folder)
        num = len(data_filenames)
        index1 = random.randint(1,num)

        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_gt)
        _, file1 = os.path.split(data_filenames[index1-1])
        _, file2 = os.path.split(data_gt)
        seed = np.random.randint(random.randint(1, 1000000)) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        h, w, dt = hw_and_dtype_for_depth_placeholder(im1)
        depth_low = torch.zeros(1, h, w, dtype=dt)
        depth_high = torch.zeros(1, h, w, dtype=dt)
        return im1, im2, depth_low, depth_high, file1, file2, "none_depth_low.npz", "none_depth_high.npz"

    def __len__(self):
        return 4803
