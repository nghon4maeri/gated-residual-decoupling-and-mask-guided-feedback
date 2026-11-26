"""
Leave-one-out ablation on ISIC-2018 with a fixed ConvFormer-S18 backbone (Reviewer comments #1-#3).

Derived from GDF-Netv4/training/colab_run_gdfnet_v4_convformer_s18_300ep.py, the script that
trained M1 / GDF-Net (Base) (0.9017 test Dice). Everything except the model flags is unchanged:
  - data: official 2,594 train / 100 val / 1,000 test, 512px train cache, 384px crops, same heavy aug
  - loss: StructureLoss (wBCE + wIoU) + 0.5 SoftMacroDice, deep supervision 1 / 1 / 0.5 / 0.3 / 0.2
  - AdamW (backbone 1e-4, head 3e-4, wd 2e-4), CosineAnnealingWarmRestarts(T_0=50, T_mult=2), 300 epochs
  - B=16, AMP, grad clip 1.0, DropPath 0.2, ModelEMA 0.999, best of RAW/EMA by val Macro Dice
  - seed 42, ImageNet-22K->1K pretrained ConvFormer-S18
Evaluation: single view, fixed threshold 0.50, 1,000 official test images (Dice, mIoU, HD95, ASD, BF2).

Usage (Colab / Kaggle, one configuration per call):
  python scripts/train_ablation_isic2018.py --config cfg2_msfdf_only
Kaggle credentials are read from KAGGLE_USERNAME / KAGGLE_KEY (never hard-coded).
"""

import os
import sys
import glob
import time
import copy
import json
import random
import zipfile
import shutil
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from scipy.ndimage import distance_transform_edt, binary_erosion

import timm
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import argparse

SCRIPT_START_TIME = time.time()

ABLATION_CONFIGS = {
    "cfg1_baseline":             dict(use_msfdf=False, use_lgag=False, mask_guidance=False, detach_mask=False),
    "cfg2_msfdf_only":           dict(use_msfdf=True,  use_lgag=False, mask_guidance=False, detach_mask=False),
    "cfg3_lgag_only":            dict(use_msfdf=False, use_lgag=True,  mask_guidance=False, detach_mask=False),
    "cfg4_msfdf_lgag_nomask":    dict(use_msfdf=True,  use_lgag=True,  mask_guidance=False, detach_mask=False),
    "cfg5_mask_guided_detach":   dict(use_msfdf=True,  use_lgag=True,  mask_guidance=True,  detach_mask=True),
    "cfg6_full_gdfnet_nodetach": dict(use_msfdf=True,  use_lgag=True,  mask_guidance=True,  detach_mask=False),
    "cfg7_plain_ms_control":     dict(use_msfdf=True,  use_lgag=True,  mask_guidance=True,  detach_mask=False,
                                      msfdf_variant="plain_multiscale"),
}

_parser = argparse.ArgumentParser()
_parser.add_argument("--config", required=True, choices=sorted(ABLATION_CONFIGS))
_parser.add_argument("--epochs", type=int, default=300)
_parser.add_argument("--max-hours", type=float, default=11.0,
                     help="Wall-clock safety guard in hours before stopping cleanly and saving latest state.")
_parser.add_argument("--resume-path", type=str, default=None,
                     help="Optional explicit path to *_latest.pth for cross-platform/dataset resume.")
ARGS, _ = _parser.parse_known_args()
CFG_ID = ARGS.config



# ====================================================================

# GDF-Net model: self-contained copy of the release modules (ablation flags)

# ====================================================================

from typing import Dict, List, Optional, Tuple, Union


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


HierarchicalBackboneV4 = HierarchicalBackbone


class MSFDF(nn.Module):
    """
    Multi-Scale Feature Decoupling Filter (MS-FDF, Section 3.1, Eqs. 1-2).

    Decouples homogeneous regional structure (x_low) from local high-gradient
    boundary residuals and high-contrast variations (x_high = E_i - x_low), then
    selectively modulates x_high via a learned channel-spatial soft gate beta.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 16)

        # Parallel multi-scale depthwise convolutions: k in {1, 3, 5}
        self.dw_conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False
        )
        self.dw_conv3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.dw_conv5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False
        )

        # Equation (1): Structural representation x_low = ReLU(BN(Conv_{1x1}([L_1 || L_3 || L_5])))
        self.pw_low = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Soft channel-spatial gate: beta = sigma(W_gate(x_low))
        self.hf_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Equation (2): Pointwise fusion projection applied to F_fuse = [x_low || x_filt]
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # 1. Parallel depthwise branches L_1, L_3, L_5
        low1 = self.dw_conv1(x)
        low3 = self.dw_conv3(x)
        low5 = self.dw_conv5(x)
        l_cat = torch.cat([low1, low3, low5], dim=1)

        # 2. Eq. (1): Structural core x_low and complementary spatial residual x_high
        x_low = self.pw_low(l_cat)
        x_high = x - x_low

        # 3. Soft gating of spatial residuals: x_filt = x_high * beta
        beta = self.hf_gate(x_low)
        x_filt = x_high * beta

        # 4. Eq. (2): Concatenation F_fuse = [x_low || x_filt] and residual addition + E_i
        f_fuse = torch.cat([x_low, x_filt], dim=1)
        out = self.fusion(f_fuse) + x

        if return_intermediates:
            return out, {
                "input": x,
                "x_low": x_low,
                "x_high": x_high,
                "beta": beta,
                "x_filt": x_filt,
                "output": out,
            }
        return out


MultiScaleFrequencyFilter = MSFDF


class PlainMultiScaleConv(nn.Module):
    """
    Complexity-Matched Multi-Scale Convolutional Control Block (without residual decoupling).

    Matches the exact parameter count and FLOPs of MSFDF by retaining identical layer
    dimensions (parallel DWConv_{1,3,5}, pw_low, bottleneck conv block, and 2C->C fusion),
    but replaces spatial residual subtraction (x_high = E_i - x_low) and soft gating
    (x_filt = x_high * beta) with direct feed-forward concatenation.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 16)

        self.dw_conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False
        )
        self.dw_conv3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.dw_conv5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False
        )
        self.pw_low = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.hf_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l_cat = torch.cat([self.dw_conv1(x), self.dw_conv3(x), self.dw_conv5(x)], dim=1)
        feat1 = self.pw_low(l_cat)
        feat2 = self.hf_gate(feat1)
        return self.fusion(torch.cat([feat1, feat2], dim=1)) + x


class MaskGuidedLGAG(nn.Module):
    """
    Mask-Guided Local-Global Attention Gate (Section 3.2, Eq. 3, Figure 3).

    Combines local (d=1) and expanded-context (d=2) depthwise receptive fields guided
    by coarse global semantic priors (D_{i+1}, M_{i+1}) from deeper decoder stages.

    Supports ablation flags:
      - mask_guidance (bool): If False, zeros out the auxiliary mask channel M_up (feature-only LGAG).
      - detach_mask (bool): If True, applies .detach() to M_{i+1} before concatenation.
    """

    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        out_channels: Optional[int] = None,
        mask_guidance: bool = True,
        detach_mask: bool = False,
    ):
        super().__init__()
        out_channels = out_channels or skip_channels
        self.skip_channels = skip_channels
        self.gating_channels = gating_channels
        self.mask_guidance = mask_guidance
        self.detach_mask = detach_mask

        # Project upsampled decoder state D_{i+1} to match skip channels C_s
        self.g_proj = nn.Sequential(
            nn.Conv2d(gating_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
        )

        # Pointwise channel compression on [F_sg || M_up] (C_s + 1 -> mid_ch)
        mid_ch = max(skip_channels // 2, 16)
        self.conv1x1 = nn.Conv2d(skip_channels + 1, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU(inplace=True)

        # Parallel dilated 3x3 depthwise convolutions: local (d=1) and expanded context (d=2)
        self.dw_dil1 = nn.Conv2d(
            mid_ch, mid_ch, kernel_size=3, padding=1, dilation=1, groups=mid_ch, bias=False
        )
        self.dw_dil2 = nn.Conv2d(
            mid_ch, mid_ch, kernel_size=3, padding=2, dilation=2, groups=mid_ch, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(mid_ch)

        # Equation (3): alpha = sigma(BN(Conv_{1x1}(K_sum)))
        self.attn_proj = nn.Sequential(
            nn.Conv2d(mid_ch, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.Sigmoid(),
        )

        # Equation (3): S_i = ReLU(BN(Conv_{3x3}(~E_i * alpha)))
        self.out_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        skip: torch.Tensor,
        gating: torch.Tensor,
        mask_prob: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, C, H, W = skip.shape

        # 1. Upsample deeper decoder state D_{i+1} and project: G = BN(Conv_{1x1}(Up_{x2}(D_{i+1})))
        g_up = F.interpolate(gating, size=(H, W), mode="bilinear", align_corners=False)
        g_proj = self.g_proj(g_up)

        # 2. Auxiliary mask prior M_{i+1} handling (differentiable vs. detached vs. disabled)
        if mask_prob is not None and self.mask_guidance:
            m_src = mask_prob.detach() if self.detach_mask else mask_prob
            m_up = F.interpolate(m_src, size=(H, W), mode="bilinear", align_corners=False)
        else:
            m_up = torch.zeros((B, 1, H, W), device=skip.device, dtype=skip.dtype)

        # 3. Additive fusion F_sg = ~E_i + G followed by mask concatenation [F_sg || M_up]
        f_sg = skip + g_proj
        feat_with_mask = torch.cat([f_sg, m_up], dim=1)

        # 4. Eq. (3): Local-global dilated depthwise context aggregation & attention gating
        x_mid = self.relu(self.bn1(self.conv1x1(feat_with_mask)))
        k_sum = self.relu(self.bn_dw(self.dw_dil1(x_mid) + self.dw_dil2(x_mid)))
        alpha = self.attn_proj(k_sum)

        gated_skip = skip * alpha
        s_i = self.out_conv(gated_skip)
        if return_attention:
            return s_i, alpha
        return s_i


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


class GDFNet(nn.Module):
    """
    GDF-Net: Gated Residual Decoupling and Mask-Guided Feedback for Medical Image Segmentation.

    Parameter breakdown with ConvFormer-S18 backbone (Total: 29,503,181 parameters, 43.09 GFLOPs @ 384^2):
      - ConvFormer-S18 Encoder:   23.67M params | 23.15 GFLOPs
      - MS-FDF Decoupling Blocks:  2.16M params |  1.98 GFLOPs
      - Mask-Guided LGAG Gates:    1.35M params |  3.07 GFLOPs
      - Cascaded MSCAM Decoder:    2.32M params | 14.89 GFLOPs

    Configurable Leave-One-Out Ablation Flags (Reviewer Comments #1, #2, #3):
      - use_msfdf (bool): Enable/disable MS-FDF spatial residual decoupling blocks.
      - use_lgag (bool): Enable/disable LGAG skip attention gates.
      - mask_guidance (bool): Inject top-down auxiliary mask prior M_{i+1} into LGAG.
      - detach_mask (bool): Apply .detach() to M_{i+1} prior to LGAG concatenation.
      - msfdf_variant (str): "residual" (default MS-FDF) or "plain_multiscale" (matched-FLOPs control).

    Fully compatible (strict=True, 534 state_dict keys) with all trained GDF-Net checkpoints.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        in_chans: int = 3,
        decoder_channels: int = 64,
        drop_path_rate: float = 0.15,
        use_msfdf: bool = True,
        use_lgag: bool = True,
        mask_guidance: bool = True,
        detach_mask: bool = False,
        msfdf_variant: str = "residual",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.decoder_channels = decoder_channels
        self.use_msfdf = use_msfdf
        self.use_lgag = use_lgag
        self.mask_guidance = mask_guidance
        self.detach_mask = detach_mask
        self.msfdf_variant = msfdf_variant

        # 1. 4-Stage Hierarchical Encoder: extracts {E_1, E_2, E_3, E_4}
        self.backbone = HierarchicalBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
        )
        c1, c2, c3, c4 = self.backbone.get_channels()

        # 2. Multi-Scale Feature Decoupling Filters (MS-FDF, Eqs. 1-2)
        if not use_msfdf:
            self.fdf1 = nn.Identity()
            self.fdf2 = nn.Identity()
            self.fdf3 = nn.Identity()
            self.fdf4 = nn.Identity()
        elif msfdf_variant == "plain_multiscale":
            self.fdf1 = PlainMultiScaleConv(c1)
            self.fdf2 = PlainMultiScaleConv(c2)
            self.fdf3 = PlainMultiScaleConv(c3)
            self.fdf4 = PlainMultiScaleConv(c4)
        else:
            self.fdf1 = MSFDF(c1)  # Stride 4  (H/4 x W/4)
            self.fdf2 = MSFDF(c2)  # Stride 8  (H/8 x W/8)
            self.fdf3 = MSFDF(c3)  # Stride 16 (H/16 x W/16)
            self.fdf4 = MSFDF(c4)  # Stride 32 (H/32 x W/32)

        # 3. Bottleneck Decoder Stage D_4 & Segmentation Head SH_4 (Z_4)
        dec4_ch = decoder_channels * 4  # 256
        self.dec4_proj = nn.Sequential(
            nn.Conv2d(c4, dec4_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec4_ch),
            nn.ReLU(inplace=True),
        )
        self.head4 = nn.Conv2d(dec4_ch, 1, kernel_size=1)

        # 4. Mask-Guided Local-Global Attention Gates (Mask-Guided LGAG, Eq. 3)
        dec3_ch = decoder_channels * 3  # 192
        dec2_ch = decoder_channels * 2  # 128
        dec1_ch = decoder_channels      # 64

        if use_lgag:
            self.gate3 = MaskGuidedLGAG(
                skip_channels=c3, gating_channels=dec4_ch, out_channels=c3,
                mask_guidance=mask_guidance, detach_mask=detach_mask,
            )
            self.gate2 = MaskGuidedLGAG(
                skip_channels=c2, gating_channels=dec3_ch, out_channels=c2,
                mask_guidance=mask_guidance, detach_mask=detach_mask,
            )
            self.gate1 = MaskGuidedLGAG(
                skip_channels=c1, gating_channels=dec2_ch, out_channels=c1,
                mask_guidance=mask_guidance, detach_mask=detach_mask,
            )
        else:
            self.gate3 = None
            self.gate2 = None
            self.gate1 = None

        # 5. Cascaded MSCAM Decoder Stages with Stage Heads SH_{3..1} (Z_3, Z_2, Z_1)
        self.dec3 = CascadedMSCAMDecoderBlock(
            in_channels=dec4_ch, skip_channels=c3, out_channels=dec3_ch
        )
        self.dec2 = CascadedMSCAMDecoderBlock(
            in_channels=dec3_ch, skip_channels=c2, out_channels=dec2_ch
        )
        self.dec1 = CascadedMSCAMDecoderBlock(
            in_channels=dec2_ch, skip_channels=c1, out_channels=dec1_ch
        )

        # 6. Full-Resolution Logit Refinement Head
        self.final_refine = nn.Sequential(
            nn.Conv2d(dec1_ch, dec1_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec1_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec1_ch, 1, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, return_deep_supervision: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, H, W = x.shape

        # 1. Hierarchical Encoder Features {E_1, E_2, E_3, E_4}
        e1, e2, e3, e4 = self.backbone(x)

        # 2. Spatial Residual Decoupling (MS-FDF, Eqs. 1-2): {~E_1, ~E_2, ~E_3, ~E_4}
        s1 = self.fdf1(e1)
        s2 = self.fdf2(e2)
        s3 = self.fdf3(e3)
        s4 = self.fdf4(e4)

        # 3. Bottleneck Stage 4 (H/32): D_4, Z_4, and auxiliary mask prior M_4 = sigma(Z_4)
        d4 = self.dec4_proj(s4)
        z4 = self.head4(d4)
        m4_prob = torch.sigmoid(z4)

        # 4. Stage 3 (H/16): Mask-Guided LGAG_3 (Eq. 3) + MSCAM Decoder D_3 -> Z_3, M_3 = sigma(Z_3)
        gated_s3 = self.gate3(s3, d4, m4_prob) if self.gate3 is not None else s3
        d3, z3 = self.dec3(d4, gated_s3)
        m3_prob = torch.sigmoid(z3)

        # 5. Stage 2 (H/8): Mask-Guided LGAG_2 (Eq. 3) + MSCAM Decoder D_2 -> Z_2, M_2 = sigma(Z_2)
        gated_s2 = self.gate2(s2, d3, m3_prob) if self.gate2 is not None else s2
        d2, z2 = self.dec2(d3, gated_s2)
        m2_prob = torch.sigmoid(z2)

        # 6. Stage 1 (H/4): Mask-Guided LGAG_1 (Eq. 3) + MSCAM Decoder D_1 -> Z_1
        gated_s1 = self.gate1(s1, d2, m2_prob) if self.gate1 is not None else s1
        d1, z1 = self.dec1(d2, gated_s1)

        # 7. Multi-Scale Stage Logit Upsampling & Summation to Native Resolution (H x W)
        z1_up = F.interpolate(z1, size=(H, W), mode="bilinear", align_corners=False)
        z2_up = F.interpolate(z2, size=(H, W), mode="bilinear", align_corners=False)
        z3_up = F.interpolate(z3, size=(H, W), mode="bilinear", align_corners=False)
        z4_up = F.interpolate(z4, size=(H, W), mode="bilinear", align_corners=False)

        d1_up = F.interpolate(d1, size=(H, W), mode="bilinear", align_corners=False)
        refined_logits = self.final_refine(d1_up)

        z_final = refined_logits + z1_up + z2_up + z3_up + z4_up

        if self.training or return_deep_supervision:
            return {
                "logits": z_final,
                "p1": z1_up,
                "p2": z2_up,
                "p3": z3_up,
                "p4": z4_up,
            }
        return z_final

MAX_RUNTIME_SECONDS = int(ARGS.max_hours * 3600)  # 11.0 hours guard (leaves 1 hour buffer before Kaggle 12h limit)

# ====================================================================
# 1. Reproducibility & Hardware Setup
# ====================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====================================================================
# 2. Expanded 15-Case Clinical Dermoscopy Augmentation Suite
# ====================================================================
class RandomVignettingTransform:
    """Simulates circular dermoscopic optical lens vignetting (dark ring/corners)."""
    def __init__(self, p=0.25, min_radius=0.50, max_radius=0.80):
        self.p = p
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        center_x = random.uniform(W * 0.42, W * 0.58)
        center_y = random.uniform(H * 0.42, H * 0.58)
        radius = random.uniform(self.min_radius, self.max_radius) * min(H, W)
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        sigma = radius * 0.18
        mask = 1.0 - 1.0 / (1.0 + np.exp(-(dist - radius) / max(sigma, 1.0)))
        mask = np.clip(mask, 0.08, 1.0)
        if image.ndim == 3:
            mask = mask[..., None]
        return (image * mask).astype(np.uint8)


class SyntheticHairTransform:
    """Simulates curved dark and light/gray hair strands occluding skin and lesion borders."""
    def __init__(self, p=0.30, max_hairs=5):
        self.p = p
        self.max_hairs = max_hairs

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        num_hairs = random.randint(1, self.max_hairs)
        for _ in range(num_hairs):
            x0, y0 = random.randint(0, W), random.randint(0, H)
            x1, y1 = random.randint(0, W), random.randint(0, H)
            ctrl_x, ctrl_y = random.randint(0, W), random.randint(0, H)
            points = []
            for t in np.linspace(0, 1, 30):
                px = int((1 - t) ** 2 * x0 + 2 * (1 - t) * t * ctrl_x + t ** 2 * x1)
                py = int((1 - t) ** 2 * y0 + 2 * (1 - t) * t * ctrl_y + t ** 2 * y1)
                points.append((px, py))
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            thickness = random.randint(1, 2)
            if random.random() < 0.8:
                c = random.randint(10, 45)
                color = (c, max(0, c - 5), max(0, c - 10))
            else:
                c = random.randint(180, 230)
                color = (c, c, c)
            cv2.polylines(img_aug, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return img_aug


class SyntheticRulerMarkTransform:
    """Simulates dermatoscope millimeter ruler tick marks near image edges."""
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        horizontal = random.random() < 0.5
        step = random.randint(12, 24)
        color = (random.randint(15, 50),) * 3
        if horizontal:
            y_base = random.choice([random.randint(5, int(H * 0.15)), random.randint(int(H * 0.85), H - 5)])
            for i, x in enumerate(range(10, W - 10, step)):
                tick_h = 14 if i % 5 == 0 else 7
                cv2.line(img_aug, (x, y_base), (x, min(H - 1, y_base + tick_h)), color, thickness=random.randint(1, 2))
        else:
            x_base = random.choice([random.randint(5, int(W * 0.15)), random.randint(int(W * 0.85), W - 5)])
            for i, y in enumerate(range(10, H - 10, step)):
                tick_w = 14 if i % 5 == 0 else 7
                cv2.line(img_aug, (x_base, y), (min(W - 1, x_base + tick_w), y), color, thickness=random.randint(1, 2))
        return img_aug


class SpecularGlareTransform:
    """Simulates small bright specular reflection spots from dermoscopic immersion fluid."""
    def __init__(self, p=0.15, max_spots=4):
        self.p = p
        self.max_spots = max_spots

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        for _ in range(random.randint(1, self.max_spots)):
            cx = random.randint(int(W * 0.15), int(W * 0.85))
            cy = random.randint(int(H * 0.15), int(H * 0.85))
            rx = random.randint(3, 10)
            ry = random.randint(2, 7)
            angle = random.randint(0, 180)
            overlay = img_aug.copy()
            cv2.ellipse(overlay, (cx, cy), (rx, ry), angle, 0, 360, (245, 245, 245), -1)
            alpha = random.uniform(0.4, 0.75)
            img_aug = cv2.addWeighted(overlay, alpha, img_aug, 1.0 - alpha, 0)
        return img_aug


class ImageOnlyCutoutTransform:
    """Applies small rectangular occlusions to the RGB image ONLY (keeping GT mask intact)."""
    def __init__(self, p=0.25, max_holes=4, max_size=28):
        self.p = p
        self.max_holes = max_holes
        self.max_size = max_size

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        mean_color = img_aug.mean(axis=(0, 1)).astype(np.uint8).tolist()
        for _ in range(random.randint(1, self.max_holes)):
            h = random.randint(8, self.max_size)
            w = random.randint(8, self.max_size)
            y0 = random.randint(0, max(1, H - h))
            x0 = random.randint(0, max(1, W - w))
            img_aug[y0:y0 + h, x0:x0 + w] = mean_color
        return img_aug


# 3. Model: gdfnet.models.gdfnet.GDFNet (strict state_dict match with the M1 checkpoint)


class ModelEMA:
    """Maintains an Exponential Moving Average of model weights for superior generalization."""
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ====================================================================
# 4. Direct Macro-Dice Optimizing Composite Loss
# ====================================================================
class StructureAndMacroDiceLoss(nn.Module):
    def __init__(self, kernel_size=31, weight_factor=5.0, dice_weight=0.5):
        super().__init__()
        self.weight_factor = weight_factor
        self.dice_weight = dice_weight
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, pred_logits, targets):
        with torch.no_grad():
            local_mean = self.avg_pool(targets)
            boundary_diff = torch.abs(targets - local_mean)
            weight = 1.0 + self.weight_factor * boundary_diff

        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        wbce = (weight * bce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-8)

        probs = torch.sigmoid(pred_logits)
        intersection = (weight * probs * targets).sum(dim=(2, 3))
        union = (weight * (probs + targets - probs * targets)).sum(dim=(2, 3))
        wiou = 1.0 - (intersection + 1.0) / (union + 1.0)

        # Unweighted Per-Image Soft Macro Dice Loss (directly aligns with Macro Dice evaluation)
        raw_inter = (probs * targets).sum(dim=(2, 3))
        raw_card = (probs + targets).sum(dim=(2, 3))
        soft_dice_loss = 1.0 - (2.0 * raw_inter + 1.0) / (raw_card + 1.0)

        return (wbce + wiou + self.dice_weight * soft_dice_loss).mean()


class GDFNetV4DiceOptCompositeLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.base_loss = StructureAndMacroDiceLoss(dice_weight=dice_weight)

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            l_final = self.base_loss(outputs['logits'], targets)
            l_p1 = self.base_loss(outputs['p1'], targets)
            l_p2 = self.base_loss(outputs['p2'], targets)
            l_p3 = self.base_loss(outputs['p3'], targets)
            l_p4 = self.base_loss(outputs['p4'], targets)
            return l_final + l_p1 + 0.5 * l_p2 + 0.3 * l_p3 + 0.2 * l_p4
        return self.base_loss(outputs, targets)


# ====================================================================
# 5. Dataset & Evaluation Engine (With Fast Local SSD Caching & Post-Crop Clinical Aug)
# ====================================================================
IMG_SIZE = 384
TRAIN_CACHE_SIZE = 512

def prepare_fast_ssd_cache(imgs, msks, cache_dir, target_size, desc="Dataset"):
    """
    Pre-resizes huge raw ISIC-2018 JPEGs/PNGs (up to 6000x4000) once to fast local SSD.
    Reduces per-epoch CPU disk/decode overhead from ~500s/epoch down to ~20s/epoch.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cached_imgs, cached_msks = [], []
    need_build = False
    for im_p, mk_p in zip(imgs, msks):
        base_id = os.path.splitext(os.path.basename(im_p))[0]
        c_im = os.path.join(cache_dir, f"{base_id}.jpg")
        c_mk = os.path.join(cache_dir, f"{base_id}_mask.png")
        cached_imgs.append(c_im)
        cached_msks.append(c_mk)
        if not (os.path.exists(c_im) and os.path.exists(c_mk)):
            need_build = True

    if need_build:
        print(f"[*] Building fast local SSD cache for {desc} ({len(imgs)} pairs -> {target_size}x{target_size}) at {cache_dir}...", flush=True)
        t_cache = time.time()
        for idx, (im_p, mk_p, c_im, c_mk) in enumerate(zip(imgs, msks, cached_imgs, cached_msks)):
            if not (os.path.exists(c_im) and os.path.exists(c_mk)):
                img = cv2.imread(im_p, cv2.IMREAD_COLOR)
                if img is None:
                    img = cv2.cvtColor(np.array(Image.open(im_p).convert('RGB')), cv2.COLOR_RGB2BGR)
                msk = cv2.imread(mk_p, cv2.IMREAD_GRAYSCALE)
                if msk is None:
                    msk = np.array(Image.open(mk_p).convert('L'))
                img_r = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
                msk_r = cv2.resize(msk, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(c_im, img_r, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
                cv2.imwrite(c_mk, msk_r)
        print(f"[+] Fast SSD cache ready for {desc} in {time.time() - t_cache:.1f}s!", flush=True)
    else:
        print(f"[+] Using existing fast SSD cache for {desc} at {cache_dir}", flush=True)
    return cached_imgs, cached_msks


class ISICDatasetV4HeavyAug(Dataset):
    def __init__(self, imgs, msks, is_train=False):
        self.imgs = imgs
        self.msks = msks
        self.is_train = is_train

        self.vignetting = RandomVignettingTransform(p=0.25)
        self.hair = SyntheticHairTransform(p=0.30, max_hairs=5)
        self.ruler = SyntheticRulerMarkTransform(p=0.15)
        self.glare = SpecularGlareTransform(p=0.15)
        self.cutout = ImageOnlyCutoutTransform(p=0.25, max_holes=4, max_size=28)

        if is_train:
            # Stage 1: Spatial & Geometric transforms (outputs 384x384 uint8 image & float32 mask)
            self.spatial_transform = A.Compose([
                A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), p=0.75),
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.82, 1.18),
                    translate_percent=(-0.10, 0.10),
                    rotate=(-180, 180),
                    shear=(-10, 10),
                    p=0.65
                ),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.25, p=1.0),
                    A.OpticalDistortion(distort_limit=0.25, p=1.0),
                    A.ElasticTransform(alpha=30, sigma=5, p=1.0),
                ], p=0.35),
            ])
            # Stage 2: Photometric & Normalization transforms (after 384x384 clinical synthesis)
            self.photo_transform = A.Compose([
                A.OneOf([
                    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ], p=0.60),
                A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.30),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(p=1.0),
                ], p=0.20),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.eval_transform = A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(self.imgs[idx]).convert('RGB'))

        msk = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = np.array(Image.open(self.msks[idx]).convert('L'))
        msk = (msk > 127).astype(np.float32)

        if self.is_train:
            # 1. Crop & Geometric transforms first -> image is now 384x384 (162x fewer pixels than raw 24MP)
            sp = self.spatial_transform(image=img, mask=msk)
            img, msk = sp['image'], sp['mask']
            # 2. Apply 5 Clinical Dermoscopy Artifact Synthesis transforms at 384x384 resolution
            img = self.vignetting(img)
            img = self.hair(img)
            img = self.ruler(img)
            img = self.glare(img)
            img = self.cutout(img)
            # 3. Apply Photometric & ImageNet Normalization
            augmented = self.photo_transform(image=img, mask=msk)
        else:
            augmented = self.eval_transform(image=img, mask=msk)

        img = np.transpose(augmented['image'].astype(np.float32), (2, 0, 1))
        msk = np.expand_dims(augmented['mask'].astype(np.float32), axis=0)
        return torch.from_numpy(img).float(), torch.from_numpy(msk).float()


def compute_metrics_v4(p, t, compute_surface=True):
    p = p.astype(bool)
    t = t.astype(bool)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    miou = tp / (tp + fp + fn + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    fpr  = fp / (fp + tn + 1e-8)

    hd95, asd, bf2 = 0.0, 0.0, 0.0
    if compute_surface and t.sum() > 0 and p.sum() > 0:
        bg = (t ^ binary_erosion(t)).astype(bool)
        bp = (p ^ binary_erosion(p)).astype(bool)
        if bg.sum() > 0 and bp.sum() > 0:
            dt_g = distance_transform_edt(~bg)
            dt_p = distance_transform_edt(~bp)
            dists = np.concatenate([dt_g[bp], dt_p[bg]])
            hd95 = float(np.percentile(dists, 95))
            asd  = float((dt_g[bp].mean() + dt_p[bg].mean()) / 2.0)
            prec_b = (dt_g[bp] <= 2).sum() / bp.sum()
            rec_b  = (dt_p[bg] <= 2).sum() / bg.sum()
            bf2 = float(2 * prec_b * rec_b / (prec_b + rec_b + 1e-8))

    return {
        'dice': float(dice), 'miou': float(miou), 'sensitivity': float(sens),
        'specificity': float(spec), 'precision': float(prec), 'fpr': float(fpr),
        'hd95': hd95, 'asd': asd, 'bf2': bf2,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
    }


@torch.no_grad()
def evaluate_model_v4(model, loader, device, use_tta=False, prob_thresh=0.5, compute_surface=True):
    model.eval()
    records = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0

    for imgs, msks in loader:
        imgs = imgs.to(device)
        with autocast('cuda'):
            if use_tta:
                p1 = torch.sigmoid(model(imgs))
                p2 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[3]))), dims=[3])
                p3 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[2]))), dims=[2])
                p4 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[2, 3]))), dims=[2, 3])
                probs = (p1 + p2 + p3 + p4) / 4.0
            else:
                probs = torch.sigmoid(model(imgs))

        preds = (probs > prob_thresh).cpu().numpy()[:, 0]
        msks_np = (msks > 0.5).numpy()[:, 0]

        for i in range(len(preds)):
            m = compute_metrics_v4(preds[i], msks_np[i], compute_surface=compute_surface)
            tot_tp += m['tp']
            tot_fp += m['fp']
            tot_fn += m['fn']
            records.append(m)

    df = pd.DataFrame(records)
    global_dice = float((2.0 * tot_tp) / (2.0 * tot_tp + tot_fp + tot_fn + 1e-8))
    global_iou  = float(tot_tp / (tot_tp + tot_fp + tot_fn + 1e-8))

    summary = []
    for col in ['dice', 'miou', 'sensitivity', 'specificity', 'precision', 'fpr', 'hd95', 'asd', 'bf2']:
        summary.append({
            'Metric': col,
            'Mean': float(df[col].mean()),
            'Std': float(df[col].std()),
            'Median': float(df[col].median()),
            'Min': float(df[col].min()),
            'Max': float(df[col].max())
        })
    j_vals = df['miou'].values
    j_thresh = float(np.where(j_vals >= 0.65, j_vals, 0.0).mean())
    failed = float((j_vals < 0.65).sum())
    summary.append({'Metric': 'j_thresh', 'Mean': j_thresh, 'Std': 0.0, 'Median': j_thresh, 'Min': 0.0, 'Max': 1.0})
    summary.append({'Metric': 'failed_cases_j_lt_0.65', 'Mean': failed, 'Std': 0.0, 'Median': failed, 'Min': 0.0, 'Max': float(len(df))})

    summary_df = pd.DataFrame(summary)
    return df, summary_df, global_dice, global_iou


@torch.no_grad()
def calibrate_threshold_on_val(model, val_loader, device):
    """Finds the probability threshold T in [0.35, 0.75] that maximizes Macro Dice on Official Val (N=100)."""
    model.eval()
    all_probs = []
    all_msks = []
    for imgs, msks in val_loader:
        imgs = imgs.to(device)
        with autocast('cuda'):
            p1 = torch.sigmoid(model(imgs))
            p2 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[3]))), dims=[3])
            p3 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[2]))), dims=[2])
            p4 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[2, 3]))), dims=[2, 3])
            probs = (p1 + p2 + p3 + p4) / 4.0
        all_probs.append(probs.cpu().numpy()[:, 0])
        all_msks.append((msks > 0.5).numpy()[:, 0])

    probs_np = np.concatenate(all_probs, axis=0)
    msks_np = np.concatenate(all_msks, axis=0)

    best_t = 0.50
    best_dice = -1.0
    for t in np.linspace(0.35, 0.75, 41):
        preds = probs_np > t
        tp = np.logical_and(preds, msks_np).sum(axis=(1, 2))
        fp = np.logical_and(preds, ~msks_np).sum(axis=(1, 2))
        fn = np.logical_and(~preds, msks_np).sum(axis=(1, 2))
        d = ((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)).mean()
        if d > best_dice:
            best_dice = float(d)
            best_t = float(t)
    return best_t, best_dice


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Active Compute Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[+] GPU Hardware: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)", flush=True)

    OUTPUT_DIR = os.path.join("/kaggle/working" if os.path.exists("/kaggle/working") else (
        "/content/results_ablation" if os.path.exists("/content") else "./results_ablation"), CFG_ID)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DRIVE_DIR = f"/content/drive/MyDrive/GDFNet_Ablation_ISIC2018/{CFG_ID}"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        print(f"[+] Google Drive Connected: Automatic Resume & Syncing to {DRIVE_DIR}", flush=True)

    DATA_ROOT = "/kaggle/temp/isic-2018" if os.path.exists("/kaggle") else (
        "/content/isic-2018" if os.path.exists("/content") else "./isic-2018"
    )
    os.makedirs(DATA_ROOT, exist_ok=True)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    k_user = os.environ.get("KAGGLE_USERNAME")
    k_key = os.environ.get("KAGGLE_KEY")
    if k_user and k_key and not os.path.exists(os.path.join(kaggle_dir, "kaggle.json")):
        with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
            json.dump({"username": k_user, "key": k_key}, f)
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

    def find_dir_by_name(target_name, search_roots):
        for base in search_roots:
            if not os.path.exists(base):
                continue
            for root, dirs, files in os.walk(base):
                if os.path.basename(root).lower() == target_name.lower():
                    return root
        return None

    def _unzip_all(root):
        for z in glob.glob(os.path.join(root, "*.zip")):
            print(f"[*] Extracting: {os.path.basename(z)}...", flush=True)
            try:
                with zipfile.ZipFile(z) as f:
                    f.extractall(root)
                os.remove(z)
            except Exception as e:
                print(f"[!] Warning while unzipping {z}: {e}", flush=True)

    search_roots = ["/kaggle/input", DATA_ROOT]
    train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input", search_roots)
    test_gt_dir   = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth", search_roots)

    if not train_img_dir:
        print("[*] Downloading ISIC-2018 Task 1 Dataset via Kaggle API...", flush=True)
        subprocess.run(["kaggle", "datasets", "download", "-d",
                        "tschandl/isic2018-challenge-task1-data-segmentation", "-p", DATA_ROOT], check=True)
        _unzip_all(DATA_ROOT)
    if not test_gt_dir:
        print("[*] Downloading ISIC-2018 Test GroundTruth via Kaggle API...", flush=True)
        subprocess.run(["kaggle", "datasets", "download", "-d",
                        "mfahimhossen/isic2018-test-groundtruth", "-p", DATA_ROOT], check=True)
        _unzip_all(DATA_ROOT)

    val_img_dir = find_dir_by_name("ISIC2018_Task1-2_Validation_Input", search_roots)
    val_gt_dir  = find_dir_by_name("ISIC2018_Task1_Validation_GroundTruth", search_roots)
    if not val_img_dir:
        print("[*] Downloading Official Val Input from S3...", flush=True)
        subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_input.zip",
                        "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"], check=True)
    if not val_gt_dir:
        print("[*] Downloading Official Val GT from S3...", flush=True)
        subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_gt.zip",
                        "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"], check=True)
    _unzip_all(DATA_ROOT)

    train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input", search_roots)
    train_gt_dir  = find_dir_by_name("ISIC2018_Task1_Training_GroundTruth", search_roots)
    val_img_dir   = find_dir_by_name("ISIC2018_Task1-2_Validation_Input", search_roots)
    val_gt_dir    = find_dir_by_name("ISIC2018_Task1_Validation_GroundTruth", search_roots)
    test_img_dir  = find_dir_by_name("ISIC2018_Task1-2_Test_Input", search_roots)
    test_gt_dir   = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth", search_roots)

    def isic_id(path):
        base = os.path.basename(path).replace("_segmentation", "")
        return os.path.splitext(base)[0]

    def pair_data(img_dir, gt_dir):
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        msks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        id_map = {isic_id(m): m for m in msks}
        paired_x, paired_y = [], []
        for im in imgs:
            imid = isic_id(im)
            if imid in id_map:
                paired_x.append(im)
                paired_y.append(id_map[imid])
        return paired_x, paired_y

    train_x_raw, train_y_raw = pair_data(train_img_dir, train_gt_dir)
    val_x_raw, val_y_raw     = pair_data(val_img_dir, val_gt_dir)
    test_x_raw, test_y_raw   = pair_data(test_img_dir, test_gt_dir)
    print(f"[+] Paired Datasets -> Train: {len(train_x_raw)} | Official Val: {len(val_x_raw)} | Blind Test: {len(test_x_raw)}", flush=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    CACHE_ROOT = os.path.join(DATA_ROOT, "fast_ssd_cache")
    train_x, train_y = prepare_fast_ssd_cache(train_x_raw, train_y_raw, os.path.join(CACHE_ROOT, "train_512"), TRAIN_CACHE_SIZE, desc="Train Set")
    val_x, val_y     = prepare_fast_ssd_cache(val_x_raw, val_y_raw, os.path.join(CACHE_ROOT, "val_384"), IMG_SIZE, desc="Official Val Set")
    test_x, test_y   = prepare_fast_ssd_cache(test_x_raw, test_y_raw, os.path.join(CACHE_ROOT, "test_384"), IMG_SIZE, desc="Official Test Set")

    BATCH_SIZE = 16
    EPOCHS = ARGS.epochs
    NUM_WORKERS = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(ISICDatasetV4HeavyAug(train_x, train_y, is_train=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=(NUM_WORKERS > 0), drop_last=True)
    val_loader   = DataLoader(ISICDatasetV4HeavyAug(val_x, val_y, is_train=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=(NUM_WORKERS > 0))
    test_loader  = DataLoader(ISICDatasetV4HeavyAug(test_x, test_y, is_train=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=(NUM_WORKERS > 0))

    model = GDFNet(backbone_name="convformer_s18.sail_in22k_ft_in1k_384", pretrained=True, drop_path_rate=0.2,
                   **ABLATION_CONFIGS[CFG_ID]).to(device)
    print(f"[+] Ablation configuration {CFG_ID}: {ABLATION_CONFIGS[CFG_ID]}", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] GDFNetV4 Initialized | Total Parameters: {total_params:,} | State Dict Keys: {len(model.state_dict())}", flush=True)

    checkpoint_best = os.path.join(OUTPUT_DIR, f"{CFG_ID}_best.pth")
    checkpoint_latest = os.path.join(OUTPUT_DIR, f"{CFG_ID}_latest.pth")
    history_csv = os.path.join(OUTPUT_DIR, f"training_history_{CFG_ID}.csv")

    best_val_dice = 0.0
    best_epoch = 0
    best_source = "scratch"
    start_epoch = 1

    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 3e-4}
    ], weight_decay=2e-4)
    ema = ModelEMA(model, decay=0.999)
    criterion = GDFNetV4DiceOptCompositeLoss(dice_weight=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    scaler = GradScaler('cuda')

    # 1. Check for existing resume checkpoint & best/history artifacts (explicit -> Drive -> Output Dir -> Kaggle Input)
    resume_candidates = []
    if ARGS.resume_path:
        resume_candidates.append(ARGS.resume_path)
    resume_candidates.extend([
        os.path.join(DRIVE_DIR, f"{CFG_ID}_latest.pth"),
        checkpoint_latest,
    ])
    kaggle_input_best = None
    kaggle_input_hist = None
    if os.path.exists("/kaggle/input"):
        for r, d, files in os.walk("/kaggle/input"):
            if f"{CFG_ID}_latest.pth" in files:
                resume_candidates.append(os.path.join(r, f"{CFG_ID}_latest.pth"))
            if f"{CFG_ID}_best.pth" in files and kaggle_input_best is None:
                kaggle_input_best = os.path.join(r, f"{CFG_ID}_best.pth")
            if f"training_history_{CFG_ID}.csv" in files and kaggle_input_hist is None:
                kaggle_input_hist = os.path.join(r, f"training_history_{CFG_ID}.csv")

    resumed = False
    last_completed_epoch = 0
    for rpath in resume_candidates:
        if rpath and os.path.exists(rpath):
            print(f"[*] Found valid resume checkpoint at: {rpath}. Restoring full state...", flush=True)
            ckpt = torch.load(rpath, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            if 'ema_state' in ckpt:
                ema.ema.load_state_dict(ckpt['ema_state'])
            else:
                ema = ModelEMA(model, decay=0.999)
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            scaler.load_state_dict(ckpt['scaler_state'])
            last_completed_epoch = int(ckpt['epoch'])
            start_epoch = last_completed_epoch + 1
            best_val_dice = ckpt.get('best_val_dice', 0.0)
            best_epoch = ckpt.get('best_epoch', ckpt['epoch'])
            best_source = ckpt.get('best_source', 'resume')
            resumed = True
            print(f"[+] Successfully Resumed from Epoch {start_epoch}/{EPOCHS} (Best Val Dice: {best_val_dice:.4f} @ Epoch {best_epoch})", flush=True)
            break

    # Restore best checkpoint from Drive or /kaggle/input so Run 2 evaluates the true global best model
    for best_src in (os.path.join(DRIVE_DIR, f"{CFG_ID}_best.pth"), kaggle_input_best):
        if best_src and os.path.exists(best_src) and not os.path.exists(checkpoint_best):
            shutil.copy2(best_src, checkpoint_best)
            print(f"[+] Restored prior best checkpoint from {best_src}", flush=True)
            break

    if not resumed:
        print("[+] Training from Scratch: ConvFormer-S18 (ImageNet-22K Pretrained, TPAMI 2024)", flush=True)

    history = []
    for hist_src in (history_csv, os.path.join(DRIVE_DIR, f"training_history_{CFG_ID}.csv"), kaggle_input_hist):
        if hist_src and os.path.exists(hist_src):
            try:
                history = pd.read_csv(hist_src).to_dict('records')
                if os.path.abspath(hist_src) != os.path.abspath(history_csv):
                    shutil.copy2(hist_src, history_csv)
                break
            except Exception:
                pass
    print(f"\n=======================================================", flush=True)
    print(f"[*] STARTING GDF-NETV4 300-EPOCH HEAVY-AUG DICE OPTIMIZATION", flush=True)
    print(f"=======================================================", flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        last_completed_epoch = epoch
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outs = model(imgs)
                loss = criterion(outs, msks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            running_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - t0
        epoch_loss = running_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[-1]

        # Fast Validation on 100 Official Val Images (both Raw and EMA models)
        _, val_sum_raw, val_glob_raw, _ = evaluate_model_v4(model, val_loader, device, use_tta=False, compute_surface=False)
        val_dice_raw = float(val_sum_raw.loc[val_sum_raw['Metric'] == 'dice', 'Mean'].values[0])

        _, val_sum_ema, val_glob_ema, _ = evaluate_model_v4(ema.ema, val_loader, device, use_tta=False, compute_surface=False)
        val_dice_ema = float(val_sum_ema.loc[val_sum_ema['Metric'] == 'dice', 'Mean'].values[0])

        epoch_best_dice = max(val_dice_raw, val_dice_ema)
        epoch_best_src = "EMA" if val_dice_ema >= val_dice_raw else "RAW"

        history.append({
            'epoch': epoch,
            'loss': epoch_loss,
            'lr': current_lr,
            'val_macro_dice_raw': val_dice_raw,
            'val_global_dice_raw': val_glob_raw,
            'val_macro_dice_ema': val_dice_ema,
            'val_global_dice_ema': val_glob_ema,
            'elapsed_sec': elapsed
        })
        pd.DataFrame(history).to_csv(history_csv, index=False)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f} "
            f"| Val Raw Dice: {val_dice_raw:.4f} | Val EMA Dice: {val_dice_ema:.4f} | Best: {best_val_dice:.4f}",
            flush=True
        )

        if epoch_best_dice > best_val_dice:
            best_val_dice = epoch_best_dice
            best_epoch = epoch
            best_source = epoch_best_src
            best_state = ema.ema.state_dict() if epoch_best_src == "EMA" else model.state_dict()
            torch.save(best_state, checkpoint_best)
            print(f"[*] New Best Model ({best_source}) Saved at Epoch {epoch} (Official Val Macro Dice: {best_val_dice:.4f})", flush=True)

        save_latest_now = os.path.exists(DRIVE_DIR) or (epoch % 5 == 0) or (epoch == EPOCHS)
        if save_latest_now:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'ema_state': ema.ema.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch,
                'best_source': best_source
            }, checkpoint_latest)

        if os.path.exists(DRIVE_DIR):
            try:
                shutil.copy2(history_csv, DRIVE_DIR)
                shutil.copy2(checkpoint_latest, DRIVE_DIR)
                if epoch == best_epoch:
                    shutil.copy2(checkpoint_best, DRIVE_DIR)
            except Exception as e:
                print(f"[!] Warning syncing to Drive: {e}", flush=True)

        total_elapsed = time.time() - SCRIPT_START_TIME
        if total_elapsed > MAX_RUNTIME_SECONDS:
            print(f"[!] Wall-clock safety guard reached ({total_elapsed/3600:.2f} hours). Saving state and stopping training cleanly to run Blind Test evaluation...", flush=True)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'ema_state': ema.ema.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch,
                'best_source': best_source
            }, checkpoint_latest)
            break

    print("\n=======================================================", flush=True)
    print(f"[*] STARTING OFFICIAL 1,000 BLIND TEST EVALUATION (Best Val Dice: {best_val_dice:.4f} @ Epoch {best_epoch} [{best_source}])", flush=True)
    print("=======================================================", flush=True)

    if os.path.exists(checkpoint_best):
        model.load_state_dict(torch.load(checkpoint_best, map_location=device, weights_only=False))

    # 1. Single-View Evaluation (T = 0.50)
    print("\n[*] Evaluating Best Model on 1,000 Official Blind Test Images (Single-View, T=0.50)...", flush=True)
    sv_df, sv_summary, sv_glob_dice, sv_glob_iou = evaluate_model_v4(
        model, test_loader, device, use_tta=False, prob_thresh=0.50, compute_surface=True
    )
    sv_summary.to_csv(os.path.join(OUTPUT_DIR, f"benchmark_summary_{CFG_ID}_SingleView.csv"), index=False)
    sv_df.to_csv(os.path.join(OUTPUT_DIR, f"benchmark_raw_{CFG_ID}_SingleView.csv"), index=False)
    print("\n================ GDF-NETV4 300EP BLIND TEST (SINGLE VIEW) ================", flush=True)
    print(sv_summary.to_string(index=False), flush=True)
    print(f"[+] Dataset Global Micro Dice: {sv_glob_dice:.4f} | IoU: {sv_glob_iou:.4f}", flush=True)

    summary = {m: float(sv_summary.loc[sv_summary['Metric'] == m, 'Mean'].values[0])
               for m in ('dice', 'miou', 'hd95', 'asd', 'bf2')}
    summary.update(config=CFG_ID, flags=ABLATION_CONFIGS[CFG_ID], params=total_params,
                   best_val_dice=best_val_dice, best_epoch=best_epoch, best_source=best_source,
                   epochs=EPOCHS, epochs_completed=int(last_completed_epoch),
                   completed=bool(last_completed_epoch >= EPOCHS), threshold=0.50,
                   low_iou_failures=int((sv_df['miou'] < 0.65).sum()))
    with open(os.path.join(OUTPUT_DIR, f"ablation_summary_{CFG_ID}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if os.path.exists(DRIVE_DIR):
        for name in (f"ablation_summary_{CFG_ID}.json", f"benchmark_raw_{CFG_ID}_SingleView.csv"):
            shutil.copy2(os.path.join(OUTPUT_DIR, name), DRIVE_DIR)
    print(json.dumps(summary, indent=2), flush=True)

    print(f"\n[+] Ablation {CFG_ID} complete!", flush=True)


if __name__ == "__main__":
    main()
