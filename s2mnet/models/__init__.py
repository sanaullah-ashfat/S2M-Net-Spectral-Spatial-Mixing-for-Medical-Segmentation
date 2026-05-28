"""
S2M-Net model components.

All classes are independently importable:

    # Full model
    from s2mnet.models import S2MNet

    # Individual blocks (use in your own architecture)
    from s2mnet.models.blocks import (
        SpectralSelectiveTokenMixer,
        MRF_SE_Block,
        BFP_DecoderStage,
    )

    # Baseline models for comparison
    from s2mnet.models.baselines import UNet, UNetPlusPlus, TransUNet, UMamba
"""

from s2mnet.models.s2mnet import S2MNet
from s2mnet.models.blocks import SpectralSelectiveTokenMixer, MRF_SE_Block, BFP_DecoderStage
from s2mnet.models.baselines import UNet, UNetPlusPlus, TransUNet, UMamba

__all__ = [
    "S2MNet",
    "SpectralSelectiveTokenMixer",
    "MRF_SE_Block",
    "BFP_DecoderStage",
    "UNet",
    "UNetPlusPlus",
    "TransUNet",
    "UMamba",
]
