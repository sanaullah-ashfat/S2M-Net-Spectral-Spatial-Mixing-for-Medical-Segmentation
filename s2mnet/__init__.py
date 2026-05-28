"""
S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive Loss
for Medical Image Segmentation.

Usage:
    from s2mnet.models import S2MNet
    from s2mnet.losses import MorphologyAwareLoss
    from s2mnet.dataloaders import PatchDataset, FullImageDataset
    from s2mnet.utils.metrics import dice_coefficient, iou_score
"""

from s2mnet.models import S2MNet
from s2mnet.losses import MorphologyAwareLoss
from s2mnet.dataloaders import PatchDataset, FullImageDataset

__version__ = "1.0.0"
__author__  = "Sanaullah"
__all__ = ["S2MNet", "MorphologyAwareLoss", "PatchDataset", "FullImageDataset"]
