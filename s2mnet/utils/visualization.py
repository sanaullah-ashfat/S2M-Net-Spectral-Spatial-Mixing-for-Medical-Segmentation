"""
Visualization Utilities
========================

    from s2mnet.utils.visualization import save_prediction_grid, save_comparison_image
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


def save_comparison_image(
    original: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    save_path: str,
    title: str = "",
    threshold: float = 0.5,
    dpi: int = 150,
) -> None:
    """
    Save a four-panel comparison: original | GT | prediction | overlay.

    Args:
        original     : RGB float32 (H, W, 3) in [0, 1].
        ground_truth : Binary float32 mask (H, W) in [0, 1].
        prediction   : Predicted probability map (H, W) in [0, 1].
        save_path    : Output file path (.png / .pdf).
        title        : Figure suptitle.
        threshold    : Binarisation threshold for the prediction.
        dpi          : Figure DPI.
    """
    pred_bin = (prediction > threshold).astype(np.uint8)
    overlay  = (np.clip(original, 0, 1) * 255).astype(np.uint8).copy()
    overlay[pred_bin == 1] = [0, 220, 0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(np.clip(original, 0, 1));     axes[0].set_title("Original",       fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(ground_truth, cmap="gray");   axes[1].set_title("Ground Truth",    fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(prediction,   cmap="gray");   axes[2].set_title("Prediction",      fontweight="bold"); axes[2].axis("off")
    axes[3].imshow(overlay);                     axes[3].set_title("Overlay (green)", fontweight="bold"); axes[3].axis("off")

    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_prediction_grid(
    samples: list,
    save_path: str,
    title: str = "Predictions",
    cols: int = 4,
    dpi: int = 150,
) -> None:
    """
    Save a grid of overlays for a list of samples.

    Args:
        samples   : List of dicts, each with keys:
                    'image'     – float32 RGB (H, W, 3)
                    'true_mask' – float32 (H, W)
                    'pred_mask' – float32 (H, W) probabilities
                    'dice'      – float scalar
        save_path : Output file path.
        title     : Grid title.
        cols      : Number of columns.
        dpi       : Figure DPI.
    """
    rows = (len(samples) + cols - 1) // cols
    fig  = plt.figure(figsize=(5 * cols, 5 * rows))
    gs   = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.2)

    for idx, s in enumerate(samples):
        row, col = divmod(idx, cols)
        ax = fig.add_subplot(gs[row, col])

        img     = np.clip(s["image"], 0, 1)
        tm      = s["true_mask"]
        pm      = s["pred_mask"]
        pb      = (pm > 0.5).astype(np.float32)

        overlay = img.copy()
        # Green channel boost for correct TP
        tp = pb * tm
        fp = pb * (1.0 - tm)
        overlay[:, :, 1] = np.where(tp > 0, np.clip(overlay[:, :, 1] + 0.4, 0, 1), overlay[:, :, 1])
        overlay[:, :, 0] = np.where(fp > 0, np.clip(overlay[:, :, 0] + 0.4, 0, 1), overlay[:, :, 0])

        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(f"Dice: {s['dice']:.4f}", fontsize=10, fontweight="bold")
        ax.axis("off")

    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="TP"),
        Patch(facecolor="red",   alpha=0.7, label="FP"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=11, frameon=False)
    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
