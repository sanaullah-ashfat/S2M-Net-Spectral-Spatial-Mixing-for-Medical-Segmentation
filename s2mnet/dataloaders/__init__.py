"""
Data Loaders for S2M-Net
==========================

    from s2mnet.dataloaders import PatchDataset, FullImageDataset
"""

from s2mnet.dataloaders.patch_dataset  import PatchDataset
from s2mnet.dataloaders.full_image     import FullImageDataset
from s2mnet.dataloaders.augmentations  import (
    get_patch_augmentation,
    get_full_image_augmentation,
    get_surgical_augmentation,
    get_validation_transform,
)

__all__ = [
    "PatchDataset",
    "FullImageDataset",
    "get_patch_augmentation",
    "get_full_image_augmentation",
    "get_surgical_augmentation",
    "get_validation_transform",
]
