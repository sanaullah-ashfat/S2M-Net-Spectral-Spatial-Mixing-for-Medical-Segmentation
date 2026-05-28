"""
Individual MAL Loss Components
================================

Each loss component is a callable Keras Layer and can be used independently.

    from s2mnet.losses.components import CoreLoss, BoundaryLoss, TextureLoss

    core = CoreLoss()
    loss = core(y_true, y_pred)
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


# ---------------------------------------------------------------------------
# Morphological helpers (shared across components)
# ---------------------------------------------------------------------------

def _morphological_dilation(x: tf.Tensor, kernel_size: int = 5) -> tf.Tensor:
    """Max-pool approximation of binary dilation."""
    return tf.nn.max_pool2d(x, kernel_size, strides=1, padding="SAME")


def _morphological_erosion(x: tf.Tensor, kernel_size: int = 5) -> tf.Tensor:
    """Min-pool approximation of binary erosion (via negation)."""
    return -tf.nn.max_pool2d(-x, kernel_size, strides=1, padding="SAME")


def _detect_boundary(mask: tf.Tensor, kernel_size: int = 5) -> tf.Tensor:
    """Boundary = dilation − erosion, clipped to [0, 1]."""
    dilated = _morphological_dilation(mask, kernel_size)
    eroded  = _morphological_erosion(mask, kernel_size)
    return tf.clip_by_value(dilated - eroded, 0.0, 1.0)


def _analyze_morphology(y_true: tf.Tensor, eps: float = 1e-6) -> dict:
    """
    Extract per-batch morphological characteristics from the ground-truth mask.

    Returns a dict with scalar tensors:
        tubularity   : skeleton_area / area  (high for thin structures)
        compactness  : 4π·area / perimeter²  (1.0 for a circle)
        irregularity : Laplacian energy of boundary
        object_size  : mean(area / total_pixels)
    """
    area         = tf.reduce_sum(y_true, axis=[1, 2, 3]) + eps
    total_pixels = tf.cast(tf.shape(y_true)[1] * tf.shape(y_true)[2], tf.float32)

    dy = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    dx = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    dy = tf.pad(dy, [[0,0],[0,1],[0,0],[0,0]])
    dx = tf.pad(dx, [[0,0],[0,0],[0,1],[0,0]])
    perimeter = tf.reduce_sum(tf.sqrt(dy**2 + dx**2 + eps), axis=[1,2,3]) + eps

    skeleton    = _morphological_erosion(y_true, kernel_size=3)
    skel_area   = tf.reduce_sum(skeleton, axis=[1,2,3]) + eps

    tubularity   = tf.clip_by_value(tf.reduce_mean(skel_area / area), 0.0, 1.0)
    compactness  = tf.clip_by_value(
        tf.reduce_mean((4.0 * 3.14159 * area) / (perimeter**2 + eps)), 0.0, 1.0
    )

    bnd = _detect_boundary(y_true, kernel_size=5)
    ddy = bnd[:, 2:] - 2*bnd[:, 1:-1] + bnd[:, :-2]
    ddx = bnd[:, :, 2:] - 2*bnd[:, :, 1:-1] + bnd[:, :, :-2]
    irregularity = tf.clip_by_value(
        tf.reduce_mean(tf.abs(ddy)) + tf.reduce_mean(tf.abs(ddx)), 0.0, 1.0
    )
    object_size  = tf.clip_by_value(tf.reduce_mean(area / total_pixels), 0.0, 1.0)

    return dict(
        tubularity=tubularity,
        compactness=compactness,
        irregularity=irregularity,
        object_size=object_size,
    )


# ---------------------------------------------------------------------------
# Core Loss  (L_core = Dice + IoU + boundary-weighted BCE)
# ---------------------------------------------------------------------------

class CoreLoss(Layer):
    """
    Core segmentation loss combining Dice, IoU, and boundary-weighted BCE.

    L_core = 0.4 * L_dice  +  0.3 * L_iou  +  0.3 * L_bce_weighted

    The boundary-weighted BCE upweights pixels near the ground-truth
    boundary by a factor of (1 + 5 * boundary_map).

    Example::

        loss_fn = CoreLoss()
        loss = loss_fn(y_true, y_pred)
    """

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps    = self.eps

        # Dice
        inter      = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
        dice       = (2.0 * inter + eps) / (
            tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3]) + eps
        )
        dice_loss  = 1.0 - tf.reduce_mean(dice)

        # IoU
        union      = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3]) - inter
        iou        = (inter + eps) / (union + eps)
        iou_loss   = 1.0 - tf.reduce_mean(iou)

        # Boundary-weighted BCE
        bnd     = _detect_boundary(y_true, kernel_size=5)
        weights = 1.0 + 5.0 * bnd
        bce     = -(
            y_true * tf.math.log(y_pred + eps) +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred + eps)
        )
        bce_loss = tf.reduce_mean(weights * bce)

        return 0.4 * dice_loss + 0.3 * iou_loss + 0.3 * bce_loss

    def get_config(self):
        cfg = super().get_config()
        cfg["eps"] = self.eps
        return cfg


# ---------------------------------------------------------------------------
# Boundary Loss  (L_bnd)
# ---------------------------------------------------------------------------

class BoundaryLoss(Layer):
    """
    Multi-scale gradient difference boundary loss.

    Computes L1 differences of spatial gradients at scales δ ∈ `scales`
    and combines them with the provided `scale_weights`.

    Example::

        loss_fn = BoundaryLoss(scales=[1, 2, 4])
        loss = loss_fn(y_true, y_pred)
    """

    def __init__(
        self,
        scales: tuple = (1, 2, 4),
        scale_weights: tuple = (0.5, 0.3, 0.2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(scales) == len(scale_weights)
        self.scales        = list(scales)
        self.scale_weights = list(scale_weights)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        total  = 0.0

        for s, w in zip(self.scales, self.scale_weights):
            dy_t = y_true[:, s:] - y_true[:, :-s]
            dy_p = y_pred[:, s:] - y_pred[:, :-s]
            dx_t = y_true[:, :, s:] - y_true[:, :, :-s]
            dx_p = y_pred[:, :, s:] - y_pred[:, :, :-s]
            total += w * (
                tf.reduce_mean(tf.abs(dy_t - dy_p)) +
                tf.reduce_mean(tf.abs(dx_t - dx_p))
            )
        return total

    def get_config(self):
        cfg = super().get_config()
        cfg.update(scales=self.scales, scale_weights=self.scale_weights)
        return cfg


# ---------------------------------------------------------------------------
# Structure Loss  (L_str)
# ---------------------------------------------------------------------------

class StructureLoss(Layer):
    """
    Shape-structure-preserving loss based on compactness difference.

    Penalises the difference in (area / perimeter²) between prediction
    and ground truth, modulated by the batch compactness characteristic.

    Example::

        loss_fn = StructureLoss()
        loss = loss_fn(y_true, y_pred)
    """

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def _compactness(self, mask: tf.Tensor) -> tf.Tensor:
        eps  = self.eps
        area = tf.reduce_sum(mask, axis=[1,2,3]) + eps
        dy   = mask[:, 1:] - mask[:, :-1]
        dx   = mask[:, :, 1:] - mask[:, :, :-1]
        dy   = tf.pad(dy, [[0,0],[0,1],[0,0],[0,0]])
        dx   = tf.pad(dx, [[0,0],[0,0],[0,1],[0,0]])
        peri = tf.reduce_sum(tf.sqrt(dy**2 + dx**2 + eps), axis=[1,2,3]) + eps
        return area / (peri**2 + eps)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        morphology  = _analyze_morphology(y_true, eps=self.eps)
        compactness = morphology["compactness"]

        c_true = self._compactness(y_true)
        c_pred = self._compactness(y_pred)
        return compactness * tf.reduce_mean(tf.abs(c_true - c_pred))

    def get_config(self):
        cfg = super().get_config()
        cfg["eps"] = self.eps
        return cfg


# ---------------------------------------------------------------------------
# Scale-Aware Focal Loss  (L_sca)
# ---------------------------------------------------------------------------

class ScaleAwareFocalLoss(Layer):
    """
    Focal loss with adaptive γ driven by object size.

    γ = 3.0  if object occupies < 5% of pixels   (tiny structures)
    γ = 2.0  if object occupies < 20%             (medium)
    γ = 1.5  otherwise                            (large)

    Example::

        loss_fn = ScaleAwareFocalLoss()
        loss = loss_fn(y_true, y_pred)
    """

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps    = self.eps

        morphology  = _analyze_morphology(y_true, eps=eps)
        size        = morphology["object_size"]

        gamma = tf.cond(
            size < 0.05,
            lambda: 3.0,
            lambda: tf.cond(size < 0.20, lambda: 2.0, lambda: 1.5),
        )

        p       = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        weight  = tf.pow(1.0 - p, gamma)
        bce     = -(
            y_true * tf.math.log(y_pred + eps) +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred + eps)
        )
        return tf.reduce_mean(weight * bce)

    def get_config(self):
        cfg = super().get_config()
        cfg["eps"] = self.eps
        return cfg


# ---------------------------------------------------------------------------
# Texture Loss  (L_tex)
# ---------------------------------------------------------------------------

class TextureLoss(Layer):
    """
    Second-order derivative (Laplacian) difference loss.

    Encourages the prediction to match the fine texture / curvature
    of the ground truth boundary.

    Example::

        loss_fn = TextureLoss()
        loss = loss_fn(y_true, y_pred)
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        ddy_t = y_true[:, 2:] - 2*y_true[:, 1:-1] + y_true[:, :-2]
        ddy_p = y_pred[:, 2:] - 2*y_pred[:, 1:-1] + y_pred[:, :-2]
        ddx_t = y_true[:, :, 2:] - 2*y_true[:, :, 1:-1] + y_true[:, :, :-2]
        ddx_p = y_pred[:, :, 2:] - 2*y_pred[:, :, 1:-1] + y_pred[:, :, :-2]

        return (
            tf.reduce_mean(tf.abs(ddy_t - ddy_p)) +
            tf.reduce_mean(tf.abs(ddx_t - ddx_p))
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

__all__ = [
    "CoreLoss",
    "BoundaryLoss",
    "StructureLoss",
    "ScaleAwareFocalLoss",
    "TextureLoss",
    "_analyze_morphology",
    "_detect_boundary",
]
