from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *


def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])


def transform2():
    return Compose([ToTensor()])


def get_lol_training_set(data_dir, size, semantic_cache_dir=None):
    if semantic_cache_dir:
        return LOLDatasetFromFolder(
            data_dir,
            transform=None,
            crop_size=size,
            semantic_cache_dir=semantic_cache_dir,
        )
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_training_set(data_dir, size, semantic_cache_dir=None):
    if semantic_cache_dir:
        return LOLv2DatasetFromFolder(
            data_dir,
            transform=None,
            crop_size=size,
            semantic_cache_dir=semantic_cache_dir,
        )
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir, size):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir, size, semantic_cache_dir=None):
    if semantic_cache_dir:
        return LOLv2SynDatasetFromFolder(
            data_dir,
            transform=None,
            crop_size=size,
            semantic_cache_dir=semantic_cache_dir,
        )
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


def get_isaid_dark_training_set(data_dir, size, semantic_cache_dir=None):
    if semantic_cache_dir:
        return ISAIDDarkDatasetFromFolder(
            data_dir,
            transform=None,
            crop_size=size,
            semantic_cache_dir=semantic_cache_dir,
        )
    return ISAIDDarkDatasetFromFolder(data_dir, transform=transform1(size))


def get_SID_training_set(data_dir, size):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir, size):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_eval_set(data_dir, semantic_cache_dir=None, depth_dir=None):
    return SICEDatasetFromFolderEval(
        data_dir, transform=transform2(), depth_dir=depth_dir, semantic_cache_dir=semantic_cache_dir
    )


def get_eval_set(data_dir, semantic_cache_dir=None, depth_dir=None):
    return DatasetFromFolderEval(
        data_dir, transform=transform2(), depth_dir=depth_dir, semantic_cache_dir=semantic_cache_dir
    )


def get_fivek_training_set(data_dir, size):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size))


def get_fivek_eval_set(data_dir, semantic_cache_dir=None):
    return SICEDatasetFromFolderEval(
        data_dir, transform=transform2(), semantic_cache_dir=semantic_cache_dir
    )
