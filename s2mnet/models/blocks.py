"""
S2M-Net Novel Architectural Building Blocks
============================================

All three blocks are independently usable layers that can be dropped into
any Keras / TensorFlow encoder-decoder architecture.

    from s2mnet.models.blocks import (
        SpectralSelectiveTokenMixer,   # SSTM
        MRF_SE_Block,                  # Multi-Receptive Field + SE
        BFP_DecoderStage,              # Boundary-Focused Progressive decoder
    )
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Conv2D, DepthwiseConv2D, Dense, Dropout,
    BatchNormalization, Activation, LayerNormalization,
    GlobalAveragePooling2D, Reshape, Multiply, Add,
    Concatenate, UpSampling2D,
)
from tensorflow.keras.regularizers import l2


# =============================================================================
# SPECTRAL-SELECTIVE TOKEN MIXER  (SSTM)
# =============================================================================

class SpectralSelectiveTokenMixer(Layer):
    """
    Spectral-Selective Token Mixer (SSTM).

    Combines two complementary paths:
      - Spectral path  : 2-D FFT → learnable frequency filter (K×K) → IFFT
      - Selective path : gated dense projection (lightweight SSM approximation)

    Both outputs are fused via a dense layer and added residually.

    Args:
        channels        : Number of input/output feature channels.
        num_frequencies : Truncation size K for the spectral filter.
                          Lower K → fewer high-frequency components retained.
                          Ablation showed K=32 is optimal (see paper).
        ssm_state_dim   : Hidden dimension for the SSM projection (unused in
                          forward math but kept for future extension).
        use_spectral    : Enable the spectral path (default True).
        use_ssm         : Enable the SSM/selective path (default True).
        dropout         : Dropout rate applied after fusion (default 0.0).

    Example::

        x = tf.random.normal([2, 32, 32, 64])
        sstm = SpectralSelectiveTokenMixer(channels=64, num_frequencies=32)
        out = sstm(x)          # shape: (2, 32, 32, 64)
    """

    def __init__(
        self,
        channels: int,
        num_frequencies: int = 32,
        ssm_state_dim: int = 16,
        use_spectral: bool = True,
        use_ssm: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels        = channels
        self.num_frequencies = num_frequencies
        self.ssm_state_dim   = ssm_state_dim
        self.use_spectral    = use_spectral
        self.use_ssm         = use_ssm
        self.dropout_rate    = dropout

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self, input_shape):
        H, W = input_shape[1], input_shape[2]
        self.actual_k = (
            min(self.num_frequencies, H, W)
            if (H is not None and W is not None)
            else self.num_frequencies
        )

        if self.use_spectral:
            self.freq_weights = self.add_weight(
                name="freq_weights",
                shape=(self.actual_k, self.actual_k, self.channels),
                initializer=self._gaussian_freq_init(),
                trainable=True,
            )
            self.spectral_norm = LayerNormalization(epsilon=1e-6, name="spectral_norm")

        if self.use_ssm:
            self.ssm_proj   = Dense(self.channels, name="ssm_proj")
            self.ssm_gate   = Dense(self.channels, activation="sigmoid", name="ssm_gate")
            self.ssm_norm   = LayerNormalization(epsilon=1e-6, name="ssm_norm")

        if self.use_spectral and self.use_ssm:
            self.fusion      = Dense(self.channels, name="fusion")
            self.fusion_norm = LayerNormalization(epsilon=1e-6, name="fusion_norm")

        self.out_norm = LayerNormalization(epsilon=1e-6, name="out_norm")
        super().build(input_shape)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _gaussian_freq_init(self):
        """
        Initialize frequency weights with a Gaussian centred at mid-band
        (0.25 normalised frequency).  This biases the model to start by
        preserving clinically relevant mid-frequency structure.
        """
        def init_fn(shape, dtype=None):
            H, W, C = shape
            fh = np.fft.fftfreq(H)[:, None]
            fw = np.fft.fftfreq(W)[None, :]
            mag = np.sqrt(fh ** 2 + fw ** 2)
            gauss = np.exp(-((mag - 0.25) ** 2) / (2 * 0.15 ** 2))
            gauss = np.repeat(gauss[:, :, None], C, axis=2)
            return (gauss * 0.5).astype(np.float32)
        return init_fn

    def _spectral_path(self, x):
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        k = tf.minimum(tf.minimum(H, W), self.actual_k)

        x_c  = tf.cast(x, tf.complex64)
        X    = tf.signal.fft2d(x_c)               # (B, H, W, C)

        # Resize real and imaginary parts to K×K
        Xr = tf.image.resize(tf.math.real(X), [k, k], method="bilinear")
        Xi = tf.image.resize(tf.math.imag(X), [k, k], method="bilinear")
        Xk = tf.complex(Xr, Xi)                   # (B, k, k, C)

        # Apply learnable filter
        filt = tf.cast(self.freq_weights[:k, :k, :], tf.complex64)
        Xk   = Xk * filt

        # Resize back
        Yr = tf.image.resize(tf.math.real(Xk), [H, W], method="bilinear")
        Yi = tf.image.resize(tf.math.imag(Xk), [H, W], method="bilinear")
        Yk = tf.complex(Yr, Yi)

        y = tf.math.real(tf.signal.ifft2d(Yk))    # (B, H, W, C)
        return self.spectral_norm(y)

    def _ssm_path(self, x):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]

        xf   = tf.reshape(x, [B, H * W, self.channels])
        gate = self.ssm_gate(xf)
        y    = self.ssm_proj(xf * gate)
        return self.ssm_norm(tf.reshape(y, [B, H, W, self.channels]))

    # ------------------------------------------------------------------
    # call
    # ------------------------------------------------------------------
    def call(self, x, training=None):
        parts = []
        if self.use_spectral:
            parts.append(self._spectral_path(x))
        if self.use_ssm:
            parts.append(self._ssm_path(x))

        if len(parts) == 2:
            fused = self.fusion_norm(self.fusion(tf.concat(parts, axis=-1)))
        elif len(parts) == 1:
            fused = parts[0]
        else:
            fused = x

        fused = self.out_norm(fused)
        if training and self.dropout_rate > 0.0:
            fused = tf.nn.dropout(fused, rate=self.dropout_rate)
        return x + fused

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            channels=self.channels,
            num_frequencies=self.num_frequencies,
            ssm_state_dim=self.ssm_state_dim,
            use_spectral=self.use_spectral,
            use_ssm=self.use_ssm,
            dropout=self.dropout_rate,
        )
        return cfg


# =============================================================================
# MULTI-RECEPTIVE FIELD SQUEEZE-EXCITATION BLOCK  (MRF-SE)
# =============================================================================

class MRF_SE_Block(Layer):
    """
    Multi-Receptive Field Squeeze-Excitation Block (MRF-SE).

    Pipeline:
      1. Pointwise expansion (expand_ratio × channels)
      2. Parallel depthwise convolutions with kernels in `kernels`
      3. Concatenate + fuse back to expanded dim
      4. Squeeze-Excitation channel attention
      5. Project down to `filters`
      6. Residual addition

    Args:
        filters      : Output (and residual input) channel count.
        kernels      : List of depthwise kernel sizes, default [3, 5, 7].
        se_reduction : Squeeze reduction factor (default 16).
        expand_ratio : Pointwise expansion factor (default 6).
        activation   : Activation string (default 'elu').
        dropout      : Dropout after projection (default 0.0).
        regularizer  : L2 weight decay (default 0.0, disabled).

    Example::

        x = tf.random.normal([2, 64, 64, 32])
        block = MRF_SE_Block(filters=32, kernels=[3, 5, 7])
        out = block(x)          # shape: (2, 64, 64, 32)
    """

    def __init__(
        self,
        filters: int,
        kernels=(3, 5, 7),
        se_reduction: int = 16,
        expand_ratio: int = 6,
        activation: str = "elu",
        dropout: float = 0.0,
        regularizer: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters      = filters
        self.kernels      = list(kernels)
        self.se_reduction = se_reduction
        self.expand_ratio = expand_ratio
        self.activation   = activation
        self.dropout_rate = dropout
        self.regularizer  = regularizer

    def build(self, input_shape):
        F = self.filters * self.expand_ratio
        reg = l2(self.regularizer) if self.regularizer > 0 else None

        # Expand
        self.expand_conv = Conv2D(F, 1, padding="same", kernel_regularizer=reg, name="expand")
        self.expand_bn   = BatchNormalization(name="expand_bn")
        self.expand_act  = Activation(self.activation, name="expand_act")

        # Parallel depthwise branches
        self.dw_convs = [
            DepthwiseConv2D(k, padding="same", depthwise_regularizer=reg, name=f"dw{k}x{k}")
            for k in self.kernels
        ]
        self.dw_bns  = [BatchNormalization(name=f"dw{k}x{k}_bn") for k in self.kernels]
        self.dw_acts = [Activation(self.activation, name=f"dw{k}x{k}_act") for k in self.kernels]

        # Fuse branches
        if len(self.kernels) > 1:
            self.fuse_conv = Conv2D(F, 1, padding="same", kernel_regularizer=reg, name="fuse")
            self.fuse_bn   = BatchNormalization(name="fuse_bn")
            self.fuse_act  = Activation(self.activation, name="fuse_act")

        # Squeeze-Excitation
        se_mid = max(F // self.se_reduction, 8)
        self.se_reduce  = Conv2D(se_mid, 1, activation=self.activation, name="se_reduce")
        self.se_expand  = Conv2D(F, 1, activation="sigmoid",            name="se_expand")

        # Project
        self.proj_conv = Conv2D(self.filters, 1, padding="same", kernel_regularizer=reg, name="project")
        self.proj_bn   = BatchNormalization(name="project_bn")

        if self.dropout_rate > 0:
            self.drop = Dropout(self.dropout_rate, name="dropout")

        super().build(input_shape)

    def call(self, x, training=None):
        # Expand
        h = self.expand_act(self.expand_bn(self.expand_conv(x), training=training))

        # Parallel depthwise
        branches = [
            act(bn(conv(h), training=training))
            for conv, bn, act in zip(self.dw_convs, self.dw_bns, self.dw_acts)
        ]

        if len(branches) > 1:
            combined = tf.concat(branches, axis=-1)
            combined = self.fuse_act(self.fuse_bn(self.fuse_conv(combined), training=training))
        else:
            combined = branches[0]

        # SE attention
        gap  = tf.reduce_mean(combined, axis=[1, 2], keepdims=True)
        se   = self.se_expand(self.se_reduce(gap))
        combined = combined * se

        # Project
        out = self.proj_bn(self.proj_conv(combined), training=training)
        if self.dropout_rate > 0:
            out = self.drop(out, training=training)

        return out + x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            filters=self.filters,
            kernels=self.kernels,
            se_reduction=self.se_reduction,
            expand_ratio=self.expand_ratio,
            activation=self.activation,
            dropout=self.dropout_rate,
            regularizer=self.regularizer,
        )
        return cfg


# =============================================================================
# BOUNDARY-FOCUSED PROGRESSIVE DECODER STAGE  (BFP)
# =============================================================================

class BFP_DecoderStage(Layer):
    """
    Boundary-Focused Progressive Decoder Stage (BFP).

    Upsamples the decoder feature map, concatenates with the skip connection,
    detects a per-pixel boundary probability map, then routes features:

      output = Conv1x1( region * (1 - β)  +  boundary_refined * β )

    where β is the boundary map and routing mode controls how β is applied.

    Args:
        filters : Output channel count.
        routing : One of 'soft' (default), 'hard', 'learned', 'none'.
                  - 'soft'    : β is used directly (continuous blending).
                  - 'hard'    : β is thresholded at 0.5.
                  - 'learned' : β replaced by learned softmax weights.
                  - 'none'    : Region and boundary features concatenated.

    Example::

        decoder = tf.random.normal([2, 8, 8, 128])
        skip    = tf.random.normal([2, 16, 16, 64])
        bfp = BFP_DecoderStage(filters=64, routing='soft')
        out, bmap = bfp(decoder, skip)
        # out:  (2, 16, 16, 64)
        # bmap: (2, 16, 16, 1)   boundary probability map
    """

    ROUTING_MODES = ("soft", "hard", "learned", "none")

    def __init__(self, filters: int, routing: str = "soft", **kwargs):
        super().__init__(**kwargs)
        assert routing in self.ROUTING_MODES, (
            f"routing must be one of {self.ROUTING_MODES}, got '{routing}'"
        )
        self.filters = filters
        self.routing = routing

    def build(self, input_shape):
        # Region processing after upsampling + concat
        self.region_conv1 = Conv2D(self.filters, 3, padding="same", name="region_conv1")
        self.region_bn1   = BatchNormalization(name="region_bn1")
        self.region_conv2 = Conv2D(self.filters, 3, padding="same", name="region_conv2")
        self.region_bn2   = BatchNormalization(name="region_bn2")

        # Boundary detection
        self.bnd_conv = Conv2D(self.filters // 2, 3, padding="same", activation="relu", name="bnd_conv")
        self.bnd_map  = Conv2D(1, 1, padding="same", activation="sigmoid",              name="bnd_map")

        # Boundary refinement
        self.refine_conv = Conv2D(self.filters, 3, padding="same", name="refine_conv")
        self.refine_bn   = BatchNormalization(name="refine_bn")

        # Fusion
        if self.routing == "none":
            # concatenation doubles channels → fuse back
            self.fusion_conv = Conv2D(self.filters, 1, padding="same", name="fusion_conv")
        elif self.routing == "learned":
            self.route_conv  = Conv2D(2, 1, activation="softmax", name="route_conv")
            self.fusion_conv = Conv2D(self.filters, 1, padding="same", name="fusion_conv")
        else:
            self.fusion_conv = Conv2D(self.filters, 1, padding="same", name="fusion_conv")

        self.fusion_bn  = BatchNormalization(name="fusion_bn")
        super().build(input_shape)

    def call(self, decoder_input, skip_features, training=None):
        # Upsample + concat
        x = tf.concat([
            tf.image.resize(decoder_input, tf.shape(skip_features)[1:3], method="bilinear"),
            skip_features,
        ], axis=-1)

        # Region features
        region = Activation("relu")(self.region_bn1(self.region_conv1(x), training=training))
        region = Activation("relu")(self.region_bn2(self.region_conv2(region), training=training))

        # Boundary map
        bnd_feat = self.bnd_conv(region)
        bnd_map  = self.bnd_map(bnd_feat)            # (B, H, W, 1)

        # Boundary-guided feature refinement (applied to boundary-attended region)
        bnd_attended = region * bnd_map
        refined = Activation("relu")(self.refine_bn(self.refine_conv(bnd_attended), training=training))

        # Routing
        if self.routing == "soft":
            blended = region * (1.0 - bnd_map) + refined * bnd_map
        elif self.routing == "hard":
            beta = tf.cast(bnd_map > 0.5, tf.float32)
            blended = region * (1.0 - beta) + refined * beta
        elif self.routing == "learned":
            combined = tf.concat([region, refined], axis=-1)
            weights  = self.route_conv(combined)          # (B, H, W, 2)
            blended  = region * weights[..., 0:1] + refined * weights[..., 1:2]
        else:  # none
            blended = tf.concat([region, refined], axis=-1)

        out = Activation("relu")(self.fusion_bn(self.fusion_conv(blended), training=training))
        return out, bnd_map

    def get_config(self):
        cfg = super().get_config()
        cfg.update(filters=self.filters, routing=self.routing)
        return cfg
