"""
Base Dataset Utilities
========================

Shared helpers for discovering image-mask pairs on disk.
"""

import os
from glob import glob
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


def find_image_mask_pairs(
    images_dir: str,
    masks_dir: str,
) -> List[Tuple[str, str]]:
    """
    Scan `images_dir` for all image files, then find the corresponding mask
    in `masks_dir` by matching the stem (base name without extension).

    Tried suffixes: `<stem>.png`, `<stem>.jpg`, `<stem>.tif`, `<stem>_mask.png`,
    `<stem>_mask.jpg`.

    Returns:
        List of (image_path, mask_path) tuples (sorted).
    """
    image_files: List[str] = []
    for ext in _IMAGE_EXTENSIONS:
        image_files.extend(glob(os.path.join(images_dir, ext)))
    image_files = sorted(image_files)

    if not image_files:
        print(f"[DataLoader] Warning: no images found in '{images_dir}'")
        return []

    pairs: List[Tuple[str, str]] = []
    for img_path in image_files:
        stem = Path(img_path).stem
        candidates = [
            f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg",
            f"{stem}.tif", f"{stem}.tiff",
            f"{stem}_mask.png", f"{stem}_mask.jpg",
        ]
        for name in candidates:
            cand = os.path.join(masks_dir, name)
            if os.path.exists(cand):
                pairs.append((img_path, cand))
                break

    return pairs


def load_split(split_dir: str) -> List[Tuple[str, str]]:
    """
    Load pairs from a directory with the layout::

        split_dir/
            images/   <-- RGB images
            masks/    <-- Binary / label masks

    Args:
        split_dir : Path to the split root (train / val / test).

    Returns:
        List of (image_path, mask_path) tuples.
    """
    images_dir = os.path.join(split_dir, "images")
    masks_dir  = os.path.join(split_dir, "masks")
    return find_image_mask_pairs(images_dir, masks_dir)
