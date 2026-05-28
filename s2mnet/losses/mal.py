"""
Morphology-Aware Adaptive Loss (MAL)
======================================

Combines five loss components with:
  (a) Learnable per-component weights w_i (clipped to [0.1, 10])
  (b) Per-batch morphology modulation factors α_i

Total loss:

    L_MAL = Σ_i  (w_i · α_i · L_i)  /  Σ_i (w_i · α_i)

Modulation factors (from ground-truth morphology each batch):

    α_core      = 1 + c_core      × compactness
    α_boundary  = 1 + c_boundary  × tubularity + compactness
    α_structure = 1 + c_structure × tubularity
    α_scale     = 1 + c_scale     × irregularity
    α_texture   = 1 + c_texture   × irregularity

Usage::

    # As a Keras Layer (trainable weights)
    from s2mnet.losses import MorphologyAwareLoss

    loss_layer = MorphologyAwareLoss()
    model.compile(optimizer='adam', loss=loss_layer)

    # As a plain function (wraps an internal Layer instance)
    from s2mnet.losses import morphology_aware_loss

    model.compile(optimizer='adam', loss=morphology_aware_loss)

Ablation / customisation::

    # Disable individual components
    loss_layer = MorphologyAwareLoss(components=['core', 'boundary'])

    # Fixed weights (no learning)
    loss_layer = MorphologyAwareLoss(learned_weights=False)

    # No morphology modulation (α_i = 1 always)
    loss_layer = MorphologyAwareLoss(morphology_modulation=False)

    # Custom morphology coefficients
    loss_layer = MorphologyAwareLoss(
        coefficients=dict(core=0.3, boundary=1.8, structure=1.0, scale=1.5, texture=1.0)
    )
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer

from s2mnet.losses.components import (
    CoreLoss,
    BoundaryLoss,
    StructureLoss,
    ScaleAwareFocalLoss,
    TextureLoss,
    _analyze_morphology,
)


_DEFAULT_COEFFICIENTS = dict(
    core=0.5,
    boundary=1.5,
    structure=1.0,
    scale=1.5,
    texture=1.0,
)

_ALL_COMPONENTS = ("core", "boundary", "structure", "scale", "texture")


class _ClipConstraint(tf.keras.constraints.Constraint):
    """Clamps weight values to [min_value, max_value]."""

    def __init__(self, min_value: float = 0.1, max_value: float = 10.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return dict(min_value=self.min_value, max_value=self.max_value)


class MorphologyAwareLoss(Layer):
    """
    Morphology-Aware Adaptive Loss (MAL).

    Args:
        components           : Subset of ('core','boundary','structure','scale','texture')
                               to include.  Default: all five.
        learned_weights      : If True, each component has a trainable scalar weight
                               w_i (clipped to [0.1, 10]).  Default: True.
        morphology_modulation: If True, α_i factors modulate each weight per batch.
                               Default: True.
        coefficients         : Dict of modulation coefficient values.  Keys are
                               component names.  Defaults to paper values.
        eps                  : Numerical stability epsilon.  Default: 1e-6.

    Example::

        loss_fn = MorphologyAwareLoss()
        model.compile(optimizer='adam', loss=loss_fn)
    """

    def __init__(
        self,
        components: tuple = _ALL_COMPONENTS,
        learned_weights: bool = True,
        morphology_modulation: bool = True,
        coefficients: dict = None,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Validate
        for c in components:
            assert c in _ALL_COMPONENTS, (
                f"Unknown component '{c}'. Valid: {_ALL_COMPONENTS}"
            )

        self.components            = list(components)
        self.learned_weights       = learned_weights
        self.morphology_modulation = morphology_modulation
        self.coefficients          = {**_DEFAULT_COEFFICIENTS, **(coefficients or {})}
        self.eps                   = eps

        # Instantiate component loss layers
        self._losses = {}
        if "core"      in self.components: self._losses["core"]      = CoreLoss(eps=eps)
        if "boundary"  in self.components: self._losses["boundary"]  = BoundaryLoss()
        if "structure" in self.components: self._losses["structure"] = StructureLoss(eps=eps)
        if "scale"     in self.components: self._losses["scale"]     = ScaleAwareFocalLoss(eps=eps)
        if "texture"   in self.components: self._losses["texture"]   = TextureLoss()

    # ------------------------------------------------------------------
    def build(self, input_shape):
        if self.learned_weights:
            clip = _ClipConstraint(0.1, 10.0)
            init_vals = dict(core=1.0, boundary=1.0, structure=1.0, scale=0.5, texture=0.5)

            self._w = {}
            for name in self.components:
                self._w[name] = self.add_weight(
                    name=f"w_{name}",
                    shape=(),
                    initializer=tf.constant_initializer(init_vals[name]),
                    trainable=True,
                    constraint=clip,
                )
        super().build(input_shape)

    # ------------------------------------------------------------------
    def _modulation_factors(self, morphology: dict) -> dict:
        if not self.morphology_modulation:
            return {c: 1.0 for c in self.components}

        c = self.coefficients
        return dict(
            core      = 1.0 + c["core"]      * morphology["compactness"],
            boundary  = 1.0 + c["boundary"]  * morphology["tubularity"]
                                             + morphology["compactness"],
            structure = 1.0 + c["structure"] * morphology["tubularity"],
            scale     = 1.0 + c["scale"]     * morphology["irregularity"],
            texture   = 1.0 + c["texture"]   * morphology["irregularity"],
        )

    # ------------------------------------------------------------------
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Simple dice fallback when only 'core' is active and weights are fixed
        # (reproduces the SIMPLE_DICE_LOSS ablation baseline)
        if (
            self.components == ["core"]
            and not self.learned_weights
            and not self.morphology_modulation
        ):
            inter = tf.reduce_sum(y_true * y_pred)
            dice  = (2.0 * inter + self.eps) / (
                tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.eps
            )
            return 1.0 - dice

        morphology = _analyze_morphology(y_true, eps=self.eps)
        alphas     = self._modulation_factors(morphology)

        total_loss   = 0.0
        total_weight = 0.0

        for name in self.components:
            l_i   = self._losses[name](y_true, y_pred)
            w_i   = self._w[name] if self.learned_weights else 1.0
            w_i   = tf.cast(w_i, tf.float32)
            alpha = tf.cast(alphas[name], tf.float32)

            total_loss   += w_i * alpha * l_i
            total_weight += w_i * alpha

        return total_loss / (total_weight + self.eps)

    # ------------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            components=self.components,
            learned_weights=self.learned_weights,
            morphology_modulation=self.morphology_modulation,
            coefficients=self.coefficients,
            eps=self.eps,
        )
        return cfg

    def get_learned_weights(self) -> dict:
        """Return the current learned weight values (after training)."""
        if not self.learned_weights:
            return {c: 1.0 for c in self.components}
        return {name: float(w.numpy()) for name, w in self._w.items()}


# ---------------------------------------------------------------------------
# Convenience wrapper — a plain Python function backed by a module-level Layer
# ---------------------------------------------------------------------------

_global_mal: MorphologyAwareLoss | None = None


def morphology_aware_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Module-level loss function for use with model.compile(loss=...).

    Uses default MAL settings.  For customised settings, use the
    ``MorphologyAwareLoss`` class directly.
    """
    global _global_mal
    if _global_mal is None:
        _global_mal = MorphologyAwareLoss(name="global_mal")
    return _global_mal(y_true, y_pred)
