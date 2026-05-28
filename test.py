#!/usr/bin/env python3
"""
S2M-Net Evaluation Script
===========================

Evaluates a trained S2M-Net checkpoint on a test set with optional
Test-Time Augmentation (TTA) and saves per-image metrics + visualisations.

Usage::

    # Basic evaluation
    python test.py --config configs/retinal.yaml \
                   --checkpoint runs/retinal/best_model.h5

    # With TTA (8-way)
    python test.py --config configs/retinal.yaml \
                   --checkpoint runs/retinal/best_model.h5 \
                   --tta

    # Save prediction images
    python test.py --config configs/polyp.yaml \
                   --checkpoint runs/polyp/best_model.h5 \
                   --tta --save-preds
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2mnet.losses               import MorphologyAwareLoss
from s2mnet.utils.metrics        import compute_metrics_numpy
from s2mnet.utils.preprocessing  import apply_clahe, apply_fov_mask, normalize_image
from s2mnet.utils.visualization  import save_comparison_image, save_prediction_grid
from s2mnet.dataloaders.base     import load_split

import yaml


# =============================================================================
# Config
# =============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess(
    bgr_image: np.ndarray,
    cfg: dict,
    target_size: int,
) -> np.ndarray:
    """Resize, optionally CLAHE, and normalise a single image."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    if cfg["dataloader"].get("use_clahe", False):
        img = apply_clahe(rgb)
    else:
        img = normalize_image(rgb)

    img = cv2.resize(img, (target_size, target_size))
    return img.astype(np.float32)


# =============================================================================
# Patch-based inference
# =============================================================================

def predict_full_image(
    model: tf.keras.Model,
    image: np.ndarray,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """Sliding-window patch inference on a full-resolution image."""
    h, w        = image.shape[:2]
    prediction  = np.zeros((h, w), dtype=np.float32)
    counts      = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch      = image[y: y+patch_size, x: x+patch_size]
            pred_patch = model.predict(patch[None], verbose=0)[0, :, :, 0]
            prediction[y: y+patch_size, x: x+patch_size] += pred_patch
            counts[y: y+patch_size, x: x+patch_size]     += 1

    return prediction / (counts + 1e-6)


# =============================================================================
# Test-Time Augmentation
# =============================================================================

def tta_predict(
    model: tf.keras.Model,
    image: np.ndarray,
    mode: str,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """
    8-way TTA: original + H/V/HV flips + 3 rotations + diagonal transpose.
    """
    def _predict(img):
        if mode == "patch":
            return predict_full_image(model, img, patch_size, stride)
        return model.predict(img[None], verbose=0)[0, :, :, 0]

    preds = []

    # Original
    preds.append(_predict(image))

    # Horizontal flip
    img_h = np.flip(image, axis=1)
    preds.append(np.flip(_predict(img_h), axis=1))

    # Vertical flip
    img_v = np.flip(image, axis=0)
    preds.append(np.flip(_predict(img_v), axis=0))

    # H+V flip
    img_hv = np.flip(np.flip(image, 0), 1)
    preds.append(np.flip(np.flip(_predict(img_hv), 0), 1))

    # 90°, 180°, 270° rotations
    for k in [1, 2, 3]:
        img_rot  = np.rot90(image, k, axes=(0, 1))
        pred_rot = _predict(img_rot)
        preds.append(np.rot90(pred_rot, -k, axes=(0, 1)))

    # Diagonal transpose (only for square images)
    if image.shape[0] == image.shape[1]:
        img_diag  = np.transpose(image, (1, 0, 2))
        pred_diag = _predict(img_diag)
        preds.append(np.transpose(pred_diag, (1, 0)))

    return np.mean(preds, axis=0)


# =============================================================================
# Main evaluation loop
# =============================================================================

def evaluate(
    cfg: dict,
    checkpoint: str,
    use_tta: bool = False,
    save_preds: bool = False,
) -> dict:
    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[Test] Loading checkpoint: {checkpoint}")
    model = tf.keras.models.load_model(
        checkpoint,
        custom_objects={"MorphologyAwareLoss": MorphologyAwareLoss},
        compile=False,
    )
    print(f"[Test] Parameters: {model.count_params():,}")

    input_size = cfg["model"]["input_size"]
    mode       = cfg["dataloader"].get("mode", "full_image")
    patch_size = cfg["dataloader"].get("patch_size", 256)
    stride     = cfg["dataloader"].get("patch_stride", 32)
    threshold  = cfg.get("evaluation", {}).get("threshold", 0.5)
    use_fov    = cfg["dataloader"].get("use_fov_mask", False)
    fov_margin = cfg["dataloader"].get("fov_margin", 20)
    num_cls    = cfg["data"].get("num_classes", 1)

    # ── Load test pairs ────────────────────────────────────────────────────────
    test_pairs = load_split(cfg["data"]["test_dir"])
    print(f"[Test] {len(test_pairs)} test images")

    save_dir = cfg["output"]["save_dir"]
    pred_dir = os.path.join(save_dir, "test_predictions")
    viz_dir  = os.path.join(save_dir, "test_visualizations")
    if save_preds:
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(viz_dir,  exist_ok=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_metrics: list[dict] = []
    vis_samples:  list[dict] = []

    for idx, (img_path, mask_path) in enumerate(test_pairs):
        name = Path(img_path).stem

        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"  [SKIP] Cannot read {img_path}")
            continue

        image = preprocess(bgr, cfg, input_size)

        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            continue
        mask = (cv2.resize(mask_raw, (input_size, input_size)) > 127).astype(np.float32)

        if use_fov:
            image, mask = apply_fov_mask(image, mask, margin=fov_margin)

        # Predict
        if use_tta:
            pred = tta_predict(model, image, mode, patch_size, stride)
        elif mode == "patch":
            pred = predict_full_image(model, image, patch_size, stride)
        else:
            pred = model.predict(image[None], verbose=0)[0, :, :, 0]

        metrics = compute_metrics_numpy(mask, pred, threshold=threshold)
        metrics["name"] = name
        all_metrics.append(metrics)

        print(
            f"  [{idx+1:3d}/{len(test_pairs)}] {name:30s} "
            f"Dice={metrics['dice']:.4f}  IoU={metrics['iou']:.4f}"
        )

        if save_preds:
            # Save probability map
            cv2.imwrite(
                os.path.join(pred_dir, f"{name}_prob.png"),
                (pred * 255).astype(np.uint8),
            )
            # Save binary mask
            cv2.imwrite(
                os.path.join(pred_dir, f"{name}_binary.png"),
                ((pred > threshold) * 255).astype(np.uint8),
            )
            # Save comparison figure
            save_comparison_image(
                original    = image,
                ground_truth= mask,
                prediction  = pred,
                save_path   = os.path.join(viz_dir, f"{name}_comparison.png"),
                title       = name,
                threshold   = threshold,
            )

        vis_samples.append({
            "image":     image,
            "true_mask": mask,
            "pred_mask": pred,
            "dice":      metrics["dice"],
        })

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_metrics)
    summary = {}
    for col in ["dice", "iou", "precision", "recall", "f1"]:
        summary[f"mean_{col}"] = float(df[col].mean())
        summary[f"std_{col}"]  = float(df[col].std())

    print("\n" + "="*70)
    print("TEST RESULTS" + (" (with TTA)" if use_tta else ""))
    print("="*70)
    for k, v in summary.items():
        print(f"  {k:<22s}: {v:.4f}")
    print("="*70)

    # ── Save summary ───────────────────────────────────────────────────────────
    summary["checkpoint"]  = checkpoint
    summary["use_tta"]     = use_tta
    summary["n_test"]      = len(all_metrics)

    summary_path = os.path.join(save_dir, "test_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(save_dir, "test_per_image.csv")
    df.to_csv(csv_path, index=False)

    # Grid visualisation
    if save_preds and vis_samples:
        vis_samples.sort(key=lambda s: s["dice"])
        worst  = vis_samples[:3]
        median = vis_samples[len(vis_samples)//2 - 3: len(vis_samples)//2 + 3]
        best   = vis_samples[-3:]
        grid   = worst + median + best

        save_prediction_grid(
            samples   = grid,
            save_path = os.path.join(viz_dir, "prediction_grid.png"),
            title     = f"S2M-Net — Test Predictions (Dice: {summary['mean_dice']:.4f})",
        )

    print(f"\n[Output]")
    print(f"  ├─ {summary_path}")
    print(f"  └─ {csv_path}")
    if save_preds:
        print(f"  ├─ {pred_dir}/")
        print(f"  └─ {viz_dir}/")

    return summary


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate S2M-Net checkpoint")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tta",        action="store_true", help="Enable 8-way TTA")
    parser.add_argument("--save-preds", action="store_true", help="Save prediction images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    evaluate(cfg, args.checkpoint, use_tta=args.tta, save_preds=args.save_preds)
