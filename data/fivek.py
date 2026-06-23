# Add new fivek dataset follow Retinexformer(https://github.com/caiyuanhao1998/Retinexformer)

import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *

class FiveKDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(FiveKDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.input_dir = join(self.data_dir, "input")
        self.target_dir = join(self.data_dir, "target")
        # Build filename lists once to avoid per-sample directory scans.
        self.data_filenames = [join(self.input_dir, x) for x in listdir(self.input_dir) if is_image_file(x)]
        self.data_filenames2 = [join(self.target_dir, x) for x in listdir(self.target_dir) if is_image_file(x)]

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        _, file1 = os.path.split(self.data_filenames[index])
        _, file2 = os.path.split(self.data_filenames2[index])
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
        return im1, im2, depth_low, depth_high, file1, file2, "none_depth_low.npz", "none_depth_high.npz"

    def __len__(self):
        return 4500
