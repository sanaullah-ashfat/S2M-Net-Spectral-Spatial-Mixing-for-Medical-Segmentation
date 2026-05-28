"""
Image Preprocessing Utilities
================================

    from s2mnet.utils.preprocessing import apply_clahe, apply_fov_mask
"""

import numpy as np
import cv2


def apply_clahe(
    image: np.ndarray,
    use_green_channel: bool = True,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
) -> np.ndarray:
    """
    CLAHE-enhanced preprocessing for retinal fundus images.

    Extracts the green channel (highest contrast for vessels), applies
    Contrast Limited Adaptive Histogram Equalisation, and returns a
    3-channel float32 image in [0, 1].

    Args:
        image             : RGB uint8 image (H, W, 3).
        use_green_channel : If True, use G channel only; else convert to gray.
        clip_limit        : CLAHE clip limit.
        tile_grid         : CLAHE tile grid size.

    Returns:
        Float32 RGB image in [0, 1], shape (H, W, 3).
    """
    if use_green_channel and len(image.shape) == 3:
        channel = image[:, :, 1]
    elif len(image.shape) == 3:
        channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        channel = image

    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(channel.astype(np.uint8)).astype(np.float32) / 255.0
    return np.stack([enhanced, enhanced, enhanced], axis=-1)


def apply_fov_mask(
    image: np.ndarray,
    mask: np.ndarray,
    margin: int = 20,
) -> tuple:
    """
    Zero out pixels outside the circular Field-of-View (FOV) for retinal images.

    Args:
        image  : Float image (H, W, 3) or (H, W).
        mask   : Float mask (H, W).
        margin : Pixels to shrink the circle radius from the image edge.

    Returns:
        (image_masked, mask_masked) with identical shapes to inputs.
    """
    h, w      = image.shape[:2]
    cy, cx    = h // 2, w // 2
    radius    = min(h, w) // 2 - margin

    yy, xx    = np.ogrid[:h, :w]
    fov       = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2

    if len(image.shape) == 3:
        image = image * fov[:, :, None]
    else:
        image = image * fov

    mask = mask * fov
    return image, mask


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Divide a uint8 image by 255 and return float32."""
    return image.astype(np.float32) / 255.0
