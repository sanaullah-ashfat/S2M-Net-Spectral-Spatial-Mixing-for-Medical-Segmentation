"""
Full-Image Dataset
===================

Used when training on full resized images (polyps, skin lesions,
surgical instruments).  Supports virtual epoch expansion (cycling through
the dataset multiple times per epoch with different random augmentations).

    from s2mnet.dataloaders import FullImageDataset

    train_ds = FullImageDataset(
        data_dir         = "data/Kvasir-SEG/train",
        input_size       = 352,
        augment          = True,
        expansion_factor = 30,   # virtual epoch multiplier
        num_classes      = 1,    # 1 → binary mask, N → one-hot
    )
    model.fit(train_ds, ...)
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from typing import List, Optional, Tuple

from s2mnet.dataloaders.base import load_split
from s2mnet.dataloaders.augmentations import (
    get_full_image_augmentation,
    get_validation_transform,
)


class FullImageDataset(tf.keras.utils.Sequence):
    """
    Keras Sequence for full-image segmentation datasets.

    Args:
        data_dir         : Root split directory (images/ + masks/).
        input_size       : Resize target (square).
        num_classes      : 1 → binary float32 mask;
                           N → one-hot float32 of shape (H, W, N).
        batch_size       : Images per batch (default 8).
        augment          : Apply augmentation (default True for train).
        expansion_factor : How many times to cycle the dataset per epoch.
                           Virtual images/epoch = len(pairs) × factor.
        shuffle          : Shuffle indices each epoch (default True).
        pixel_mapping    : Optional dict {pixel_value: class_index} for
                           multi-class masks with arbitrary label values.
        seed             : Random seed (default 42).
    """

    def __init__(
        self,
        data_dir: str,
        input_size: int = 352,
        num_classes: int = 1,
        batch_size: int = 8,
        augment: bool = True,
        expansion_factor: int = 1,
        shuffle: bool = True,
        pixel_mapping: Optional[dict] = None,
        seed: int = 42,
    ):
        self.data_dir         = data_dir
        self.input_size       = input_size
        self.num_classes      = num_classes
        self.batch_size       = batch_size
        self.augment          = augment
        self.expansion_factor = expansion_factor
        self.shuffle          = shuffle
        self.pixel_mapping    = pixel_mapping

        np.random.seed(seed)

        self._pairs   = load_split(data_dir)
        assert self._pairs, f"No image-mask pairs found in '{data_dir}'"

        self._aug = (
            get_full_image_augmentation(input_size)
            if augment
            else get_validation_transform(input_size)
        )

        self._real_batches    = len(self._pairs) // batch_size
        self._virtual_batches = self._real_batches * expansion_factor
        self._indices         = np.arange(len(self._pairs))

        if shuffle:
            np.random.shuffle(self._indices)

        print(
            f"[FullImageDataset] {len(self._pairs)} images  |  "
            f"real_batches={self._real_batches}  "
            f"virtual_batches={self._virtual_batches}  (×{expansion_factor})"
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._virtual_batches

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Map virtual index → real index (cycle)
        real_ptr  = index % self._real_batches
        start     = real_ptr * self.batch_size
        end       = start + self.batch_size
        batch_idx = self._indices[start:end]

        imgs, masks = [], []
        for i in batch_idx:
            img_path, mask_path = self._pairs[i]

            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if self.num_classes == 1:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask = (mask > 127).astype(np.float32)
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask = self._remap_mask(mask)

            aug   = self._aug(image=image, mask=mask)
            image = aug["image"].astype(np.float32) / 255.0
            mask  = aug["mask"]

            if self.num_classes == 1:
                mask = mask[..., None].astype(np.float32)
            else:
                mask = np.eye(self.num_classes, dtype=np.float32)[mask.astype(np.int32)]

            imgs.append(image)
            masks.append(mask)

        return np.stack(imgs), np.stack(masks)

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert raw pixel values to class indices via pixel_mapping."""
        if self.pixel_mapping is None:
            return mask.astype(np.int32)
        out = np.zeros_like(mask, dtype=np.int32)
        for pixel_val, class_idx in self.pixel_mapping.items():
            out[mask == pixel_val] = class_idx
        return out

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._indices)
