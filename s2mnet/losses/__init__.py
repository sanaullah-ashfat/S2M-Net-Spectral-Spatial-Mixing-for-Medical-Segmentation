"""
Loss Functions for S2M-Net
===========================

Independently usable loss functions and components.

    # Full MAL (recommended)
    from s2mnet.losses import MorphologyAwareLoss

    # Individual components
    from s2mnet.losses.components import (
        CoreLoss,
        BoundaryLoss,
        StructureLoss,
        ScaleAwareFocalLoss,
        TextureLoss,
    )
"""

from s2mnet.losses.mal import MorphologyAwareLoss, morphology_aware_loss
from s2mnet.losses.components import (
    CoreLoss,
    BoundaryLoss,
    StructureLoss,
    ScaleAwareFocalLoss,
    TextureLoss,
)

__all__ = [
    "MorphologyAwareLoss",
    "morphology_aware_loss",
    "CoreLoss",
    "BoundaryLoss",
    "StructureLoss",
    "ScaleAwareFocalLoss",
    "TextureLoss",
]
