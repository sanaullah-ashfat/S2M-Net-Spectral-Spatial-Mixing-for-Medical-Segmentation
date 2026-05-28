"""
Segmentation Metrics
=====================

All metrics are TensorFlow functions compatible with model.compile(metrics=[...]).

    from s2mnet.utils.metrics import dice_coefficient, iou_score
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    """Sørensen–Dice coefficient (higher is better)."""
    y_true_f    = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f    = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth: float = 1e-6):
    """Intersection-over-Union / Jaccard index (higher is better)."""
    y_true_f    = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f    = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    union        = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_metric(y_true, y_pred):
    """Precision = TP / (TP + FP)."""
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f   = tf.cast(y_true, tf.float32)
    tp         = K.sum(y_true_f * y_pred_bin)
    pp         = K.sum(y_pred_bin)
    return tp / (pp + K.epsilon())


def recall_metric(y_true, y_pred):
    """Recall / Sensitivity = TP / (TP + FN)."""
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f   = tf.cast(y_true, tf.float32)
    tp         = K.sum(y_true_f * y_pred_bin)
    ap         = K.sum(y_true_f)
    return tp / (ap + K.epsilon())


def f1_score(y_true, y_pred):
    """F1 = 2 × precision × recall / (precision + recall)."""
    p   = precision_metric(y_true, y_pred)
    r   = recall_metric(y_true, y_pred)
    return 2.0 * p * r / (p + r + K.epsilon())


# ---------------------------------------------------------------------------
# NumPy-based evaluation (for test-time patch inference results)
# ---------------------------------------------------------------------------

import numpy as np


def compute_metrics_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict:
    """
    Compute all metrics from NumPy arrays.

    Args:
        y_true     : Ground-truth binary mask (H, W) or (H, W, 1).
        y_pred     : Predicted probability map (H, W) or (H, W, 1).
        threshold  : Binarisation threshold.
        eps        : Numerical stability.

    Returns:
        dict with keys: dice, iou, precision, recall, f1
    """
    y_true = y_true.squeeze().astype(np.float32)
    y_pred = y_pred.squeeze().astype(np.float32)
    y_bin  = (y_pred > threshold).astype(np.float32)

    inter  = np.sum(y_true * y_bin)
    union  = np.sum(y_true) + np.sum(y_bin) - inter
    tp     = inter
    pp     = np.sum(y_bin)
    ap     = np.sum(y_true)

    dice      = (2.0 * inter + eps) / (np.sum(y_true) + np.sum(y_bin) + eps)
    iou       = (inter + eps) / (union + eps)
    precision = tp / (pp + eps)
    recall    = tp / (ap + eps)
    f1        = 2.0 * precision * recall / (precision + recall + eps)

    return dict(dice=dice, iou=iou, precision=precision, recall=recall, f1=f1)
