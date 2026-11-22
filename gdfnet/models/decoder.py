"""
Cascaded MSCAM Decoder and Stage Segmentation Heads (SH) — Section 3.2, Figure 1b.

Adapts Multi-Scale Convolutional Attention Modules (MSCAM) from EMCAD (Rahman et al., CVPR 2024)
to fuse gated skip features S_i with upsampled decoder states Up_{x2}(D_{i+1}), and predicts
stage logits Z_k = Conv_{1x1}(D_k) for deep supervision and top-down mask feedback.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConvolutionalAttentionModule(nn.Module):
    """
    Multi-Scale Convolutional Attention Module (MSCAM, adapted from EMCAD, CVPR 2024).

    Aggregates multi-scale spatial context via parallel depthwise convolutions
    (k in {1, 3, 5}), followed by 1x1 pointwise projection, Channel Attention (CA),
    Spatial Attention (SA, 7x7 kernel), and 3x3 Conv-BN-ReLU refinement.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Parallel multi-scale depthwise convolutions: k in {1, 3, 5}
        self.dw1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False
        )
        self.dw3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.dw5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels * 3)

        # Pointwise projection
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Channel Attention (CA)
        mid_ch = max(out_channels // 4, 16)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial Attention (SA)
        self.sa = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # Final 3x3 convolutional refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.dw1(x)
        d3 = self.dw3(x)
        d5 = self.dw5(x)
        dw_cat = F.relu(self.bn_dw(torch.cat([d1, d3, d5], dim=1)), inplace=True)
        feat = self.pw(dw_cat)
        feat = feat * self.ca(feat)
        feat = feat * self.sa(feat)
        return self.refine(feat)


class CascadedMSCAMDecoderBlock(nn.Module):
    """
    Cascaded MSCAM Decoder Stage with Segmentation Head (SH: Z_i = Conv_{1x1}(D_i)).

    1. Upsamples deeper decoder feature D_{i+1} by 2x.
    2. Concatenates with mask-gated skip feature S_i.
    3. Refines via MSCAM to produce D_i.
    4. Projects D_i through 1x1 segmentation head SH to obtain stage logits Z_i.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.mscam = MultiScaleConvolutionalAttentionModule(
            in_channels + skip_channels, out_channels
        )
        self.head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(
        self, x: torch.Tensor, gated_skip: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_up = self.upsample(x)
        if x_up.shape[2:] != gated_skip.shape[2:]:
            x_up = F.interpolate(
                x_up, size=gated_skip.shape[2:], mode="bilinear", align_corners=False
            )
        fused = torch.cat([x_up, gated_skip], dim=1)
        d_i = self.mscam(fused)
        z_i = self.head(d_i)
        return d_i, z_i
