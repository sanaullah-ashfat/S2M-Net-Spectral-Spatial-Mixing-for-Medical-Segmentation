"""
Spectral Analysis Utilities
=============================

Helper functions for analysing spectral energy retention and
reconstruction quality at various truncation sizes K.

Used by scripts/analyze_spectral.py and figure.py.
"""

import numpy as np
import cv2


def spectral_energy_retention(image: np.ndarray, K: int) -> float:
    """
    Fraction of total spectral energy retained by a K×K centre crop
    of the 2-D DFT magnitude spectrum.

    Args:
        image : Float image (H, W) or (H, W, C).
        K     : Truncation size.

    Returns:
        Energy retention in [0, 1].
    """
    if image.ndim == 3:
        energies = [_channel_energy(image[:, :, c], K) for c in range(image.shape[2])]
        return float(np.mean(energies))
    return _channel_energy(image, K)


def _channel_energy(channel: np.ndarray, K: int) -> float:
    fft     = np.fft.fftshift(np.fft.fft2(channel))
    total   = np.sum(np.abs(fft) ** 2)
    h, w    = fft.shape
    cy, cx  = h // 2, w // 2
    k2      = K // 2
    crop    = fft[cy - k2: cy + k2, cx - k2: cx + k2]
    retained = np.sum(np.abs(crop) ** 2)
    return float(retained / (total + 1e-10))


def reconstruction_rmse(image: np.ndarray, K: int):
    """
    Reconstruct a grayscale image from a K×K spectral crop and return
    the RMSE between original and reconstruction, plus the reconstruction.

    Args:
        image : Float image (H, W) or (H, W, C).
        K     : Truncation size.

    Returns:
        (rmse: float, reconstructed: np.ndarray of shape (H, W))
    """
    if image.ndim == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = image.astype(np.float32)

    fft     = np.fft.fftshift(np.fft.fft2(gray))
    h, w    = fft.shape
    cy, cx  = h // 2, w // 2
    k2      = K // 2

    mask          = np.zeros_like(fft)
    mask[cy - k2: cy + k2, cx - k2: cx + k2] = fft[cy - k2: cy + k2, cx - k2: cx + k2]

    reconstructed = np.real(np.fft.ifft2(np.fft.ifftshift(mask)))
    rmse          = float(np.sqrt(np.mean((gray - reconstructed) ** 2)))
    return rmse, reconstructed


def analyze_dataset_spectrum(
    image_paths: list,
    truncation_sizes: tuple = (16, 24, 32, 48, 64, 128),
    input_size: int = 352,
    max_samples: int = 50,
) -> dict:
    """
    Compute energy retention and RMSE statistics over a set of images
    for each truncation size K.

    Returns:
        dict mapping K → {'energy': [...], 'rmse': [...]}
    """
    results = {K: {"energy": [], "rmse": []} for K in truncation_sizes}
    paths   = image_paths[:max_samples]

    for path in paths:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size)).astype(np.float32) / 255.0

        for K in truncation_sizes:
            if K < min(input_size, input_size):
                results[K]["energy"].append(spectral_energy_retention(img, K) * 100)
                rmse, _ = reconstruction_rmse(img, K)
                results[K]["rmse"].append(rmse)

    return results
