"""
Augmentation Pipelines
=======================

All pipelines are built with albumentations and return a
``A.Compose`` transform callable.

    from s2mnet.dataloaders.augmentations import get_patch_augmentation

    transform = get_patch_augmentation()
    result    = transform(image=img, mask=mask)
    img_aug   = result["image"]
    mask_aug  = result["mask"]
"""

import cv2
import albumentations as A


# ---------------------------------------------------------------------------
# Patch-based  (retinal vessels, DRIVE / CHASE-DB)
# ---------------------------------------------------------------------------

def get_patch_augmentation() -> A.Compose:
    """
    Augmentation for patch-based training (256×256 patches).

    Includes heavy geometric transforms (rotations, elastic, grid distortion)
    suitable for circular retinal fundus images.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.9),
        A.ElasticTransform(
            alpha=25, sigma=4, alpha_affine=4,
            border_mode=cv2.BORDER_REFLECT_101, p=0.4,
        ),
        A.GridDistortion(
            num_steps=5, distort_limit=0.1,
            border_mode=cv2.BORDER_REFLECT_101, p=0.3,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ])


# ---------------------------------------------------------------------------
# Full-image  (polyp — Kvasir-SEG / PH2)
# ---------------------------------------------------------------------------

def get_full_image_augmentation(input_size: int = 352) -> A.Compose:
    """
    DuckNet-style aggressive augmentation for full-image training.

    Includes random resized crop, coarse dropout, and colour jitter.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(input_size, input_size),
            scale=(0.5, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_CUBIC,
            p=0.6,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=180,
            border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0,
        ),
        A.CoarseDropout(
            max_holes=12, max_height=48, max_width=48,
            min_holes=5,  min_height=20, min_width=20,
            fill_value=0, mask_fill_value=0, p=0.5,
        ),
        A.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.1, hue=0.01, p=1.0),
        A.Resize(height=input_size, width=input_size),
    ])


# ---------------------------------------------------------------------------
# Surgical instruments  (EndoVis-17)
# ---------------------------------------------------------------------------

def get_surgical_augmentation(input_size: int = 512) -> A.Compose:
    """
    Conservative augmentation for surgical instrument datasets
    where shape preservation matters.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.3),
        A.Resize(height=input_size, width=input_size),
    ])


# ---------------------------------------------------------------------------
# Validation / test  (resize only)
# ---------------------------------------------------------------------------

def get_validation_transform(input_size: int) -> A.Compose:
    """Deterministic resize-only transform for validation and test sets."""
    return A.Compose([A.Resize(input_size, input_size)])
