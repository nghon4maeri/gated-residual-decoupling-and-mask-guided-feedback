"""
Full GDF-Net Architecture (29,503,181 Parameters with ConvFormer-S18) — Section 3, Figure 1.

Includes:
  1. GDFNet (CanonicalGDFNetV4): Main architecture used across all deployment regimes
     (Base, +Bi-ALT, Soup, Real-Time Distilled Student, and 8-View Ensemble).
  2. FeedbackGDFNet: Auxiliary feedback-modulated teacher architecture (M_3) used exclusively
     inside the frozen 4-model teacher ensemble during knowledge distillation on ISIC-2018.
  3. build_llrd_param_groups: Stage-wise Layer-Wise Learning Rate Decay (LLRD = 0.65).
"""

from typing import Dict, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HierarchicalBackbone, DEFAULT_BACKBONE
from .ms_fdf import MultiScaleFrequencyFilter
from .lgag import MaskGuidedLGAG
from .decoder import CascadedMSCAMDecoderBlock


class GDFNet(nn.Module):
    """
    GDF-Net: Gated Residual Decoupling and Mask-Guided Feedback for Medical Image Segmentation.

    Parameter breakdown with ConvFormer-S18 backbone (Total: 29,503,181 parameters, 43.09 GFLOPs @ 384^2):
      - ConvFormer-S18 Encoder:   23.67M params | 23.15 GFLOPs
      - MS-FDF Decoupling Blocks:  2.16M params |  1.98 GFLOPs
      - Mask-Guided LGAG Gates:    1.35M params |  3.07 GFLOPs
      - Cascaded MSCAM Decoder:    2.32M params | 14.89 GFLOPs

    Fully compatible (strict=True, 534 state_dict keys) with all trained GDF-Net checkpoints.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        in_chans: int = 3,
        decoder_channels: int = 64,
        drop_path_rate: float = 0.15,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.decoder_channels = decoder_channels

        # 1. 4-Stage Hierarchical Encoder: extracts {E_1, E_2, E_3, E_4}
        self.backbone = HierarchicalBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
        )
        c1, c2, c3, c4 = self.backbone.get_channels()

        # 2. Multi-Scale Feature Decoupling Filters (MS-FDF, Eqs. 1-2)
        self.fdf1 = MultiScaleFrequencyFilter(c1)  # Stride 4  (H/4 x W/4)
        self.fdf2 = MultiScaleFrequencyFilter(c2)  # Stride 8  (H/8 x W/8)
        self.fdf3 = MultiScaleFrequencyFilter(c3)  # Stride 16 (H/16 x W/16)
        self.fdf4 = MultiScaleFrequencyFilter(c4)  # Stride 32 (H/32 x W/32)

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

        self.gate3 = MaskGuidedLGAG(skip_channels=c3, gating_channels=dec4_ch, out_channels=c3)
        self.gate2 = MaskGuidedLGAG(skip_channels=c2, gating_channels=dec3_ch, out_channels=c2)
        self.gate1 = MaskGuidedLGAG(skip_channels=c1, gating_channels=dec2_ch, out_channels=c1)

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

        # 3. Bottleneck Stage 4 (H/32): D_4, Z_4, and differentiable mask prior M_4 = sigma(Z_4)
        d4 = self.dec4_proj(s4)
        z4 = self.head4(d4)
        m4_prob = torch.sigmoid(z4)

        # 4. Stage 3 (H/16): Mask-Guided LGAG_3 (Eq. 3) + MSCAM Decoder D_3 -> Z_3, M_3 = sigma(Z_3)
        gated_s3 = self.gate3(s3, d4, m4_prob)
        d3, z3 = self.dec3(d4, gated_s3)
        m3_prob = torch.sigmoid(z3)

        # 5. Stage 2 (H/8): Mask-Guided LGAG_2 (Eq. 3) + MSCAM Decoder D_2 -> Z_2, M_2 = sigma(Z_2)
        gated_s2 = self.gate2(s2, d3, m3_prob)
        d2, z2 = self.dec2(d3, gated_s2)
        m2_prob = torch.sigmoid(z2)

        # 6. Stage 1 (H/4): Mask-Guided LGAG_1 (Eq. 3) + MSCAM Decoder D_1 -> Z_1
        gated_s1 = self.gate1(s1, d2, m2_prob)
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


# Alias for backward compatibility with training scripts and checkpoints
CanonicalGDFNetV4 = GDFNet


# ==============================================================================
# Auxiliary M_3 Teacher Variant (FeedbackGDFNet) for 4-Teacher Online KD
# ==============================================================================
class _MultiScaleFrequencyFilterFeedback(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.dw_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.dw_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.dw_conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        self.pw_low = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.hf_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_low = self.pw_low(torch.cat([self.dw_conv1(x), self.dw_conv3(x), self.dw_conv5(x)], dim=1))
        f_high_raw = x - f_low
        f_high = f_high_raw * self.hf_gate(f_high_raw)
        out = self.fusion(torch.cat([f_low, f_high], dim=1))
        return x + self.gamma * out


class _AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1, bias=False), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1, bias=False), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(f_int, 1, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        return x * self.psi(self.W_g(g) + self.W_x(x))


class _FeedbackModulator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.3)

    def forward(self, feat: torch.Tensor, coarse_prob: torch.Tensor) -> torch.Tensor:
        if coarse_prob.shape[2:] != feat.shape[2:]:
            coarse_prob = F.interpolate(coarse_prob, size=feat.shape[2:], mode="bilinear", align_corners=False)
        gate = self.conv_gate(torch.cat([feat, coarse_prob], dim=1))
        return feat * (1.0 + self.alpha * gate)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DeepSupervisionDecoder(nn.Module):
    def __init__(self, c1: int, c2: int, c3: int, c4: int, num_classes: int = 1):
        super().__init__()
        self.ag3 = _AttentionGate(f_g=c4, f_l=c3, f_int=c3 // 2)
        self.ag2 = _AttentionGate(f_g=c3, f_l=c2, f_int=c2 // 2)
        self.ag1 = _AttentionGate(f_g=c2, f_l=c1, f_int=c1 // 2)
        self.dec4 = _ConvBlock(c4, c4)
        self.dec3 = _ConvBlock(c4 + c3, c3)
        self.dec2 = _ConvBlock(c3 + c2, c2)
        self.dec1 = _ConvBlock(c2 + c1, c1)
        self.head4 = nn.Conv2d(c4, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(c3, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(c2, num_classes, kernel_size=1)
        self.head1 = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.final_refine = nn.Sequential(
            nn.Conv2d(c1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        d4 = self.dec4(f4)
        d4_up = F.interpolate(d4, size=f3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d4_up, self.ag3(g=d4_up, x=f3)], dim=1))
        d3_up = F.interpolate(d3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d3_up, self.ag2(g=d3_up, x=f2)], dim=1))
        d2_up = F.interpolate(d2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2_up, self.ag1(g=d2_up, x=f1)], dim=1))
        d1_full = F.interpolate(d1, size=target_size, mode="bilinear", align_corners=False)
        return self.final_refine(d1_full)


class FeedbackGDFNet(nn.Module):
    """
    Feedback-Modulated Teacher Variant (M_3 in Section 3.3).
    Used exclusively as one of the four frozen ConvFormer-S18 teachers during online
    knowledge distillation on ISIC-2018.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = False,
        num_classes: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone = HierarchicalBackbone(
            backbone_name=backbone_name, pretrained=pretrained, in_chans=3, drop_path_rate=drop_path_rate
        )
        c1, c2, c3, c4 = self.backbone.get_channels()
        self.freq1 = _MultiScaleFrequencyFilterFeedback(c1)
        self.freq2 = _MultiScaleFrequencyFilterFeedback(c2)
        self.freq3 = _MultiScaleFrequencyFilterFeedback(c3)
        self.freq4 = _MultiScaleFrequencyFilterFeedback(c4)
        self.fb1 = _FeedbackModulator(c1)
        self.fb2 = _FeedbackModulator(c2)
        self.fb3 = _FeedbackModulator(c3)
        self.decoder = _DeepSupervisionDecoder(c1, c2, c3, c4, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[2:]
        s1, s2, s3, s4 = self.backbone(x)
        f1, f2, f3, f4 = self.freq1(s1), self.freq2(s2), self.freq3(s3), self.freq4(s4)
        coarse_prob = torch.sigmoid(self.decoder.head4(self.decoder.dec4(f4)))
        return self.decoder(
            self.fb1(f1, coarse_prob),
            self.fb2(f2, coarse_prob),
            self.fb3(f3, coarse_prob),
            f4,
            target_size=target_size,
        )


def build_llrd_param_groups(
    model: nn.Module,
    base_lr: float = 4.5e-5,
    weight_decay: float = 1e-4,
    llrd_decay: float = 0.65,
) -> List[Dict]:
    """
    Constructs AdamW parameter groups with Stage-Wise Layer-Wise Learning Rate Decay
    (LLRD = 0.65, Section 4 & Supplementary Table 5):
      - Decoder / MS-FDF / LGAG heads:eta * 0.65^0
      - Encoder Stage 4:              eta * 0.65^1
      - Encoder Stage 3:              eta * 0.65^2
      - Encoder Stage 2:              eta * 0.65^3
      - Encoder Stem & Stage 1:       eta * 0.65^4
    """
    group_map = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.startswith("backbone."):
            g_name = "head_decoder"
            lr = base_lr
        elif "stages.3" in name or "Stages.3" in name or ".3." in name:
            g_name = "stage4"
            lr = base_lr * (llrd_decay ** 1)
        elif "stages.2" in name or "Stages.2" in name or ".2." in name:
            g_name = "stage3"
            lr = base_lr * (llrd_decay ** 2)
        elif "stages.1" in name or "Stages.1" in name or ".1." in name:
            g_name = "stage2"
            lr = base_lr * (llrd_decay ** 3)
        else:
            g_name = "stem_stage1"
            lr = base_lr * (llrd_decay ** 4)

        no_decay = (
            (param.ndim == 1)
            or name.endswith(".bias")
            or ("norm" in name.lower())
            or ("bn" in name.lower())
        )
        wd = 0.0 if no_decay else weight_decay
        key = (g_name, lr, wd)
        group_map.setdefault(key, []).append(param)

    return [
        {"params": params, "lr": lr, "weight_decay": wd, "group_name": g_name}
        for (g_name, lr, wd), params in group_map.items()
    ]
