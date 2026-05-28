"""
Baseline Models for Comparison
================================

All baselines share the same interface as S2M-Net and are built with
TensorFlow / Keras functional API.

    from s2mnet.models.baselines import UNet, UNetPlusPlus, TransUNet, UMamba
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, DepthwiseConv2D,
    MaxPooling2D, Concatenate, Add, BatchNormalization, Activation,
    GlobalAveragePooling2D, Reshape, Multiply, Dropout,
    LayerNormalization, Dense, MultiHeadAttention, UpSampling2D, Layer,
)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

def UNet(input_size: int = 256, num_classes: int = 1) -> Model:
    """Classic U-Net (Ronneberger et al., 2015)."""

    def enc_block(x, f, name):
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"{name}_c1")(x)
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"{name}_c2")(x)
        return x

    inp = Input((input_size, input_size, 3))

    c1 = enc_block(inp, 32, "e1");  p1 = MaxPooling2D()(c1)
    c2 = enc_block(p1,  64, "e2");  p2 = MaxPooling2D()(c2)
    c3 = enc_block(p2, 128, "e3");  p3 = MaxPooling2D()(c3)
    c4 = enc_block(p3, 256, "e4");  p4 = MaxPooling2D()(c4)

    bn = enc_block(p4, 512, "bn")

    def dec_block(x, skip, f, name):
        x = Conv2DTranspose(f, 2, strides=2, padding="same", name=f"{name}_up")(x)
        x = Concatenate()([x, skip])
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"{name}_c1")(x)
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"{name}_c2")(x)
        return x

    d4 = dec_block(bn, c4, 256, "d4")
    d3 = dec_block(d4, c3, 128, "d3")
    d2 = dec_block(d3, c2,  64, "d2")
    d1 = dec_block(d2, c1,  32, "d1")

    act = "sigmoid" if num_classes == 1 else "softmax"
    out = Conv2D(num_classes, 1, activation=act, dtype="float32", name="output")(d1)
    return Model(inputs=inp, outputs=out, name="UNet")


# ---------------------------------------------------------------------------
# U-Net++
# ---------------------------------------------------------------------------

def UNetPlusPlus(input_size: int = 256, num_classes: int = 1) -> Model:
    """U-Net++ with dense nested skip connections (Zhou et al., 2018)."""

    def conv_block(x, f, name):
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"{name}_c1")(x)
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"{name}_c2")(x)
        return x

    def up_merge(decoder, *skips, f, name):
        u = Conv2DTranspose(f, 2, strides=2, padding="same", name=f"{name}_up")(decoder)
        x = Concatenate()([u, *skips])
        return conv_block(x, f, name)

    inp = Input((input_size, input_size, 3))

    x00 = conv_block(inp,             32, "x00"); p0 = MaxPooling2D()(x00)
    x10 = conv_block(p0,              64, "x10"); p1 = MaxPooling2D()(x10)
    x20 = conv_block(p1,             128, "x20"); p2 = MaxPooling2D()(x20)
    x30 = conv_block(p2,             256, "x30")

    x01 = up_merge(x10, x00,                   f=32,  name="x01")
    x11 = up_merge(x20, x10,                   f=64,  name="x11")
    x02 = up_merge(x11, x00, x01,              f=32,  name="x02")
    x21 = up_merge(x30, x20,                   f=128, name="x21")
    x12 = up_merge(x21, x10, x11,              f=64,  name="x12")
    x03 = up_merge(x12, x00, x01, x02,         f=32,  name="x03")

    act = "sigmoid" if num_classes == 1 else "softmax"
    out = Conv2D(num_classes, 1, activation=act, dtype="float32", name="output")(x03)
    return Model(inputs=inp, outputs=out, name="UNetPlusPlus")


# ---------------------------------------------------------------------------
# TransUNet
# ---------------------------------------------------------------------------

class _TransformerBlock(Layer):
    """Single Transformer encoder block (MHSA + FFN)."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, **kw):
        super().__init__(**kw)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn  = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.norm2 = LayerNormalization(epsilon=1e-6)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = tf.keras.Sequential([
            Dense(mlp_dim, activation="gelu"),
            Dropout(dropout),
            Dense(embed_dim),
            Dropout(dropout),
        ])

    def call(self, x, training=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


def TransUNet(
    input_size: int = 256,
    num_classes: int = 1,
    num_heads: int = 8,
    transformer_layers: int = 2,
    dropout: float = 0.1,
) -> Model:
    """TransUNet (Chen et al., 2021)."""

    def enc(x, f, stage):
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"te{stage}_c1")(x)
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"te{stage}_c2")(x)
        return x

    inp = Input((input_size, input_size, 3))
    channels = [64, 128, 256]

    x = Conv2D(channels[0], 3, padding="same", activation="relu", name="stem")(inp)
    encoder_outputs = [x]

    for i, f in enumerate(channels[1:], 1):
        x = MaxPooling2D()(x)
        x = enc(x, f, i)
        encoder_outputs.append(x)

    # Transformer bottleneck
    x = MaxPooling2D()(x)
    B_  = tf.shape(x)[0]
    H_  = input_size // (2 ** len(channels))
    C_  = channels[-1]
    x_flat = tf.reshape(x, [B_, H_ * H_, C_])

    pos = tf.Variable(
        tf.random.normal([1, H_ * H_, C_], stddev=0.02),
        trainable=True,
        name="pos_encoding",
    )
    x_flat = x_flat + pos

    for i in range(transformer_layers):
        x_flat = _TransformerBlock(C_, num_heads, dropout=dropout, name=f"transformer_{i}")(x_flat)

    x = tf.reshape(x_flat, [B_, H_, H_, C_])

    # Decoder
    skip_list     = encoder_outputs[::-1]
    filter_list   = channels[::-1]

    for i, (skip, f) in enumerate(zip(skip_list, filter_list), 1):
        x = Conv2DTranspose(f, 2, strides=2, padding="same", name=f"td{i}_up")(x)
        x = Concatenate()([x, skip])
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"td{i}_c1")(x)
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"td{i}_c2")(x)

    act = "sigmoid" if num_classes == 1 else "softmax"
    out = Conv2D(num_classes, 1, activation=act, dtype="float32", name="output")(x)
    return Model(inputs=inp, outputs=out, name="TransUNet")


# ---------------------------------------------------------------------------
# U-Mamba
# ---------------------------------------------------------------------------

class _SSMBlock(Layer):
    """Lightweight SSM-like token mixer (gated dense projection)."""

    def __init__(self, channels, dropout=0.1, **kw):
        super().__init__(**kw)
        self.channels     = channels
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.proj  = Dense(self.channels, name="proj")
        self.gate  = Dense(self.channels, activation="sigmoid", name="gate")
        self.out   = Dense(self.channels, name="out")
        self.norm  = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, x, training=None):
        B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]
        xf = tf.reshape(x, [B, H * W, self.channels])
        y  = self.out(self.proj(xf) * self.gate(xf))
        if training and self.dropout_rate > 0:
            y = tf.nn.dropout(y, rate=self.dropout_rate)
        y = self.norm(y)
        return x + tf.reshape(y, [B, H, W, self.channels])


def UMamba(
    input_size: int = 256,
    num_classes: int = 1,
    channels: tuple = (32, 64, 128, 256, 512),
    dropout: float = 0.1,
) -> Model:
    """U-Mamba (Ma et al., 2024) — depthwise separable encoder with SSM blocks."""

    def enc_block(x, f, stage):
        x = DepthwiseConv2D(3, strides=2, padding="same", name=f"um_e{stage}_dw")(x)
        x = Conv2D(f, 1, padding="same", name=f"um_e{stage}_pw")(x)
        x = BatchNormalization(name=f"um_e{stage}_bn")(x)
        x = Activation("relu", name=f"um_e{stage}_act")(x)
        x = _SSMBlock(f, dropout=dropout, name=f"um_e{stage}_ssm")(x)
        return x

    def dec_block(x, skip, f, stage):
        x = Conv2DTranspose(f, 2, strides=2, padding="same", name=f"um_d{stage}_up")(x)
        x = Concatenate()([x, skip])
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"um_d{stage}_c1")(x)
        x = Conv2D(f, 3, padding="same", activation="relu", name=f"um_d{stage}_c2")(x)
        return x

    inp = Input((input_size, input_size, 3))

    x = Conv2D(channels[0], 3, padding="same", name="um_stem")(inp)
    x = BatchNormalization(name="um_stem_bn")(x)
    x = Activation("relu", name="um_stem_act")(x)

    encoder_outputs = [x]
    for i, f in enumerate(channels[1:], 1):
        x = enc_block(x, f, i)
        encoder_outputs.append(x)

    x = encoder_outputs[-1]
    for i in range(len(encoder_outputs) - 1):
        skip = encoder_outputs[-(i + 2)]
        x    = dec_block(x, skip, skip.shape[-1], i + 1)

    act = "sigmoid" if num_classes == 1 else "softmax"
    out = Conv2D(num_classes, 1, activation=act, dtype="float32", name="output")(x)
    return Model(inputs=inp, outputs=out, name="UMamba")
