"""
Full GDF-Net Architecture (29,503,181 Parameters with ConvFormer-S18) — Section 3, Figure 1.

Includes:
  1. GDFNet (CanonicalGDFNetV4): Main architecture used across all deployment regimes
     (Base, +Bi-ALT, Soup, Real-Time Distilled Student, and 8-View Ensemble), with built-in
     leave-one-out ablation flags (use_msfdf, use_lgag, mask_guidance, detach_mask, msfdf_variant).
  2. FeedbackGDFNet: Auxiliary feedback-modulated teacher architecture (M_3) used exclusively
     inside the frozen 4-model teacher ensemble during knowledge distillation on ISIC-2018.
  3. build_llrd_param_groups: Stage-wise Layer-Wise Learning Rate Decay (LLRD = 0.65).
"""

from typing import Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HierarchicalBackbone, DEFAULT_BACKBONE
from .ms_fdf import MSFDF, MultiScaleFrequencyFilter, PlainMultiScaleConv
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
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.relu(self.w_g(g) + self.w_x(x)))


class _FeedbackModulator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.3)

    def forward(self, feat: torch.Tensor, coarse_prob: torch.Tensor) -> torch.Tensor:
        m = F.interpolate(coarse_prob, size=feat.shape[2:], mode="bilinear", align_corners=False)
        m_feat = torch.sigmoid(self.mask_conv(m))
        s_attn = self.spatial_attn(torch.cat([feat, m], dim=1))
        return feat * (1.0 + self.alpha * m_feat * s_attn)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DeepSupervisionDecoder(nn.Module):
    def __init__(self, c1: int, c2: int, c3: int, c4: int, num_classes: int = 1):
        super().__init__()
        d3, d2, d1 = 256, 128, 64
        self.up3 = nn.ConvTranspose2d(c4, d3, kernel_size=2, stride=2)
        self.ag3 = _AttentionGate(f_g=d3, f_l=c3, f_int=d3 // 2)
        self.dec3 = _ConvBlock(d3 + c3, d3)

        self.up2 = nn.ConvTranspose2d(d3, d2, kernel_size=2, stride=2)
        self.ag2 = _AttentionGate(f_g=d2, f_l=c2, f_int=d2 // 2)
        self.dec2 = _ConvBlock(d2 + c2, d2)

        self.up1 = nn.ConvTranspose2d(d2, d1, kernel_size=2, stride=2)
        self.ag1 = _AttentionGate(f_g=d1, f_l=c1, f_int=d1 // 2)
        self.dec1 = _ConvBlock(d1 + c1, d1)

        self.head1 = nn.Conv2d(d1, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(d2, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(d3, num_classes, kernel_size=1)
        self.head4 = nn.Conv2d(c4, num_classes, kernel_size=1)

    def forward(
        self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor, input_shape
    ) -> Dict[str, torch.Tensor]:
        u3 = self.up3(f4)
        if u3.shape[2:] != f3.shape[2:]:
            u3 = F.interpolate(u3, size=f3.shape[2:], mode="bilinear", align_corners=False)
        x3 = self.dec3(torch.cat([u3, self.ag3(u3, f3)], dim=1))

        u2 = self.up2(x3)
        if u2.shape[2:] != f2.shape[2:]:
            u2 = F.interpolate(u2, size=f2.shape[2:], mode="bilinear", align_corners=False)
        x2 = self.dec2(torch.cat([u2, self.ag2(u2, f2)], dim=1))

        u1 = self.up1(x2)
        if u1.shape[2:] != f1.shape[2:]:
            u1 = F.interpolate(u1, size=f1.shape[2:], mode="bilinear", align_corners=False)
        x1 = self.dec1(torch.cat([u1, self.ag1(u1, f1)], dim=1))

        p1 = F.interpolate(self.head1(x1), size=input_shape, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.head2(x2), size=input_shape, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.head3(x3), size=input_shape, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.head4(f4), size=input_shape, mode="bilinear", align_corners=False)
        return {"logits": p1, "p1": p1, "p2": p2, "p3": p3, "p4": p4}


class FeedbackGDFNet(nn.Module):
    """Auxiliary Feedback-Modulated Teacher (M_3) used in 4-teacher online distillation."""

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = False,
        num_classes: int = 1,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        import timm
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate,
        )
        c1, c2, c3, c4 = self.encoder.feature_info.channels()
        self.freq1 = _MultiScaleFrequencyFilterFeedback(c1)
        self.freq2 = _MultiScaleFrequencyFilterFeedback(c2)
        self.freq3 = _MultiScaleFrequencyFilterFeedback(c3)

        self.fb1 = _FeedbackModulator(c1)
        self.fb2 = _FeedbackModulator(c2)
        self.fb3 = _FeedbackModulator(c3)
        self.decoder = _DeepSupervisionDecoder(c1, c2, c3, c4, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[2:]
        e1, e2, e3, e4 = self.encoder(x)
        f1 = self.freq1(e1)
        f2 = self.freq2(e2)
        f3 = self.freq3(e3)

        pass1 = self.decoder(f1, f2, f3, e4, input_shape)
        coarse_prob = torch.sigmoid(pass1["logits"])

        f1_ref = self.fb1(f1, coarse_prob)
        f2_ref = self.fb2(f2, coarse_prob)
        f3_ref = self.fb3(f3, coarse_prob)
        pass2 = self.decoder(f1_ref, f2_ref, f3_ref, e4, input_shape)
        return 0.70 * pass2["logits"] + 0.30 * pass1["logits"]


def build_llrd_param_groups(
    model: nn.Module, base_lr: float = 5e-4, weight_decay: float = 1e-4, llrd_factor: float = 0.65
) -> List[Dict]:
    """
    Build stage-wise Layer-Wise Learning Rate Decay (LLRD = 0.65) parameter groups.
    """
    stage_scales = {
        "backbone.backbone.stem": llrd_factor ** 4,
        "backbone.backbone.stages_0": llrd_factor ** 4,
        "backbone.backbone.stages_1": llrd_factor ** 3,
        "backbone.backbone.stages_2": llrd_factor ** 2,
        "backbone.backbone.stages_3": llrd_factor ** 1,
    }
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        scale = 1.0
        for prefix, s in stage_scales.items():
            if name.startswith(prefix):
                scale = s
                break
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        wd = 0.0 if no_decay else weight_decay
        key = (round(scale, 6), wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr * scale, "weight_decay": wd}
        groups[key]["params"].append(param)
    return list(groups.values())
