"""
Hierarchical 4-Stage Backbone Encoder for GDF-Net (Section 3.1).

Extracts hierarchical feature maps {E_1, E_2, E_3, E_4} at strides {4, 8, 16, 32}
with channels {64, 128, 320, 512} for ConvFormer-S18 (or PVTv2-B2).
"""

from typing import Tuple
import torch
import torch.nn as nn
import timm

DEFAULT_BACKBONE = "convformer_s18.sail_in22k_ft_in1k_384"


class HierarchicalBackbone(nn.Module):
    """
    Four-Stage Hierarchical Feature Encoder (Section 3.1, Figure 1a).

    Given an input scan I in R^{3 x H x W}, a 7x7 patch stem (stride s=4) and four
    hierarchical stages ([3, 3, 9, 3] blocks with 7x7 depthwise separable convolutions
    in ConvFormer-S18) extract feature maps {E_1, E_2, E_3, E_4} at spatial resolutions
    {H/4 x W/4, H/8 x W/8, H/16 x W/16, H/32 x W/32} and channel dimensions
    {64, 128, 320, 512}, maintaining linear O(H * W) complexity and translation equivariance.

    Supported backbones:
      - 'convformer_s18.sail_in22k_ft_in1k_384' (default, Pure-CNN MetaFormer, 23.67M params)
      - 'pvt_v2_b2' (Pyramid Vision Transformer v2 baseline for same-module ablation)
      - 'convnextv2_tiny', 'convnextv2_nano', 'inception_next_tiny'
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        in_chans: int = 3,
        drop_path_rate: float = 0.15,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate,
        )
        self.channels = self.backbone.feature_info.channels()

    def get_channels(self) -> Tuple[int, int, int, int]:
        return tuple(self.channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return feats[0], feats[1], feats[2], feats[3]


# Alias for backward compatibility with trained checkpoint scripts
HierarchicalBackboneV4 = HierarchicalBackbone
