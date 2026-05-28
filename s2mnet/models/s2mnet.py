"""
S2M-Net: Spectral-Spatial Mixing Network
==========================================

Full encoder-decoder architecture combining:
  - MRF-SE blocks in every encoder stage
  - SSTM applied per-stage (configurable via sstm_stages mask)
  - BFP decoder stages
  - Flexible multi-class or binary output

    from s2mnet.models import S2MNet

    model = S2MNet(input_size=352, num_classes=1)
    model.summary()
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    UpSampling2D, Dropout,
)

from s2mnet.models.blocks import SpectralSelectiveTokenMixer, MRF_SE_Block, BFP_DecoderStage


def S2MNet(
    input_size: int = 352,
    num_classes: int = 1,
    filters: tuple = (24, 32, 64, 80, 128),
    # MRF-SE settings
    use_mrfse: bool = True,
    mrfse_kernels: tuple = (3, 5, 7),
    se_reduction: int = 16,
    expand_ratio: int = 6,
    # SSTM settings
    use_sstm: bool = True,
    sstm_k: int = 32,
    sstm_ssm_dim: int = 16,
    sstm_stages: tuple = (True, True, True, True, True),
    sstm_use_spectral: tuple = (True, True, True, True, True),
    sstm_use_ssm: tuple = (False, False, True, True, True),
    sstm_dropout: float = 0.1,
    # BFP settings
    use_bfp: bool = True,
    bfp_routing: str = "soft",
    # General
    dropout: float = 0.1,
    l2_reg: float = 1e-4,
    activation: str = "elu",
    name: str = "S2M-Net",
) -> Model:
    """
    Build the S2M-Net model.

    Args:
        input_size          : Spatial input resolution (square).
        num_classes         : 1 → binary sigmoid output; >1 → softmax output.
        filters             : Channel counts for encoder stages 1-5.
        use_mrfse           : Enable MRF-SE blocks.
        mrfse_kernels       : Depthwise kernel sizes for MRF-SE.
        se_reduction        : SE channel reduction factor.
        expand_ratio        : Pointwise expansion ratio inside MRF-SE.
        use_sstm            : Enable SSTM globally.
        sstm_k              : Spectral truncation size K.
        sstm_ssm_dim        : SSM hidden dimension.
        sstm_stages         : Per-stage SSTM enable mask (5 bools).
        sstm_use_spectral   : Per-stage spectral path enable (5 bools).
        sstm_use_ssm        : Per-stage SSM path enable (5 bools).
        sstm_dropout        : SSTM internal dropout.
        use_bfp             : Enable BFP decoder; falls back to plain upsample+cat.
        bfp_routing         : BFP routing mode: 'soft'|'hard'|'learned'|'none'.
        dropout             : Global encoder dropout (applied in MRF-SE).
        l2_reg              : L2 weight decay.
        activation          : Encoder activation.
        name                : Model name.

    Returns:
        tf.keras.Model
    """
    assert len(filters) == 5, "filters must have exactly 5 elements"
    assert len(sstm_stages) == 5 == len(sstm_use_spectral) == len(sstm_use_ssm)

    inp = Input((input_size, input_size, 3), name="input")

    # ------------------------------------------------------------------ stem
    x = Conv2D(16, 3, padding="same", kernel_initializer="he_uniform", name="stem_conv")(inp)
    x = BatchNormalization(name="stem_bn")(x)
    x = Activation(activation, name="stem_act")(x)

    # ------------------------------------------------------------------ encoder
    encoder_outputs = []

    for i, f in enumerate(filters):
        stage = i + 1

        # Strided convolution (downsample)
        x = Conv2D(f, 3, strides=2, padding="same",
                   kernel_initializer="he_uniform",
                   name=f"enc{stage}_down")(x)
        x = BatchNormalization(name=f"enc{stage}_bn")(x)
        x = Activation(activation, name=f"enc{stage}_act")(x)

        # MRF-SE
        if use_mrfse:
            x = MRF_SE_Block(
                filters=f,
                kernels=mrfse_kernels,
                se_reduction=se_reduction,
                expand_ratio=expand_ratio,
                activation=activation,
                dropout=dropout,
                regularizer=l2_reg,
                name=f"mrfse_stage{stage}",
            )(x)

        # SSTM
        if use_sstm and sstm_stages[i]:
            x = SpectralSelectiveTokenMixer(
                channels=f,
                num_frequencies=sstm_k,
                ssm_state_dim=sstm_ssm_dim,
                use_spectral=sstm_use_spectral[i],
                use_ssm=sstm_use_ssm[i],
                dropout=sstm_dropout,
                name=f"sstm_stage{stage}",
            )(x)

        encoder_outputs.append(x)

    # ------------------------------------------------------------------ decoder
    # Reversed: deepest skip first
    skips           = encoder_outputs[::-1]
    decoder_filters = list(filters[::-1][1:]) + [16]

    decoder = skips[0]  # bottleneck features

    for i, (skip, f) in enumerate(zip(skips[1:], decoder_filters)):
        stage = i + 1

        if use_bfp:
            bfp = BFP_DecoderStage(filters=f, routing=bfp_routing, name=f"bfp_stage{stage}")
            decoder, _ = bfp(decoder, skip)
        else:
            # Plain upsampling + concatenation fallback
            decoder = UpSampling2D(2, name=f"up_{stage}")(decoder)
            decoder = tf.concat([decoder, skip], axis=-1)
            decoder = Conv2D(f, 3, padding="same", activation="relu",
                             name=f"dec{stage}_conv1")(decoder)
            decoder = Conv2D(f, 3, padding="same", activation="relu",
                             name=f"dec{stage}_conv2")(decoder)

    # ------------------------------------------------------------------ head
    # Final upsample to restore full resolution (stem halved by stride, so ×2)
    decoder = UpSampling2D(2, name="head_up")(decoder)
    decoder = Conv2D(32, 3, padding="same", activation="relu", name="head_conv1")(decoder)
    decoder = Conv2D(16, 3, padding="same", activation="relu", name="head_conv2")(decoder)

    if num_classes == 1:
        out = Conv2D(1, 1, padding="same", activation="sigmoid",
                     dtype="float32", name="output")(decoder)
    else:
        out = Conv2D(num_classes, 1, padding="same", activation="softmax",
                     dtype="float32", name="output")(decoder)

    return Model(inputs=inp, outputs=out, name=name)
