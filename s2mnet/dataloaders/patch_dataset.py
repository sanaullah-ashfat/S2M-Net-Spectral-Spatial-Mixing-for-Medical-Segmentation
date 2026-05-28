"""
Patch-Based Dataset
====================

Used for datasets where images are large and training is done on
randomly sampled sub-patches (e.g. retinal fundus images).

Pre-extraction workflow:
  1. Load full-resolution images.
  2. Apply CLAHE preprocessing (optional).
  3. Slide a window of `patch_size` × `patch_size` with stride `stride`.
  4. Keep only patches where the vessel/foreground ratio ≥ `min_fg_ratio`.
  5. During training, randomly sample from the pre-extracted pool with
     augmentation applied on-the-fly.

    from s2mnet.dataloaders import PatchDataset

    train_ds = PatchDataset(
        data_dir    = "data/CHASE-DB/train",
        patch_size  = 256,
        stride      = 32,
        min_fg_ratio= 0.005,
        augment     = True,
    )
    # Use as a tf.keras.utils.Sequence in model.fit(train_ds, ...)
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from typing import List, Tuple, Optional

from s2mnet.dataloaders.base import load_split
from s2mnet.dataloaders.augmentations import get_patch_augmentation
from s2mnet.utils.preprocessing import apply_clahe, apply_fov_mask


class PatchDataset(tf.keras.utils.Sequence):
    """
    Keras Sequence that serves random patches sampled from a pre-built pool.

    Args:
        data_dir         : Root split directory (must contain images/ and masks/).
        patch_size       : Spatial size of extracted patches (default 256).
        stride           : Sliding window stride for patch extraction (default 32).
        min_fg_ratio     : Minimum foreground pixel ratio to keep a patch.
        batch_size       : Number of patches per batch (default 16).
        patches_per_epoch: Total patches consumed per epoch (default 4000).
        augment          : Apply random augmentations (default True for train).
        use_clahe        : Apply CLAHE preprocessing (default True).
        use_fov_mask     : Apply circular FOV mask for retinal images (default True).
        fov_margin       : Pixel margin to shrink the FOV circle (default 20).
        shuffle          : Shuffle patch indices each epoch (default True).
        seed             : Random seed (default 42).
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: int = 256,
        stride: int = 32,
        min_fg_ratio: float = 0.005,
        batch_size: int = 16,
        patches_per_epoch: int = 4000,
        augment: bool = True,
        use_clahe: bool = True,
        use_fov_mask: bool = True,
        fov_margin: int = 20,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_dir          = data_dir
        self.patch_size        = patch_size
        self.stride            = stride
        self.min_fg_ratio      = min_fg_ratio
        self.batch_size        = batch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment           = augment
        self.use_clahe         = use_clahe
        self.use_fov_mask      = use_fov_mask
        self.fov_margin        = fov_margin
        self.shuffle           = shuffle

        np.random.seed(seed)

        self._augment_fn = get_patch_augmentation() if augment else None
        self._pairs      = load_split(data_dir)

        assert self._pairs, f"No image-mask pairs found in '{data_dir}'"

        self._pool: List[Tuple[np.ndarray, np.ndarray]] = []
        self._extract_patches()

        self.steps = patches_per_epoch // batch_size
        print(
            f"[PatchDataset] {len(self._pairs)} images → "
            f"{len(self._pool):,} patches  |  "
            f"steps/epoch={self.steps}  batch={batch_size}"
        )

    # ------------------------------------------------------------------
    def _preprocess_image(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self.use_clahe:
            return apply_clahe(rgb)
        return rgb.astype(np.float32) / 255.0

    def _extract_patches(self):
        ps = self.patch_size
        st = self.stride

        for img_path, mask_path in self._pairs:
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            image = self._preprocess_image(bgr)

            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                continue
            mask = (mask_raw > 127).astype(np.float32)

            if self.use_fov_mask:
                image, mask = apply_fov_mask(image, mask, margin=self.fov_margin)

            h, w = image.shape[:2]
            for y in range(0, h - ps + 1, st):
                for x in range(0, w - ps + 1, st):
                    m_patch = mask[y: y+ps, x: x+ps]
                    if m_patch.mean() >= self.min_fg_ratio:
                        self._pool.append((
                            image[y: y+ps, x: x+ps].copy(),
                            m_patch.copy(),
                        ))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, _index: int) -> Tuple[np.ndarray, np.ndarray]:
        idxs = np.random.choice(len(self._pool), self.batch_size, replace=False)
        imgs, masks = [], []

        for i in idxs:
            img, msk = self._pool[i]

            if self._augment_fn is not None:
                aug  = self._augment_fn(image=img, mask=msk)
                img  = aug["image"]
                msk  = aug["mask"]

            imgs.append(img.astype(np.float32))
            masks.append(msk[..., None].astype(np.float32))

        return np.stack(imgs), np.stack(masks)

    def on_epoch_end(self):
        pass  # pool is static; shuffling is implicit via random sampling
