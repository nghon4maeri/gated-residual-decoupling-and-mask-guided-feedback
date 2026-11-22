"""
Mask-Guided Local-Global Attention Gate (Mask-Guided LGAG) — Section 3.2, Equation (3), Figure 3.

Mathematical Formulation:
  Given decoupled skip feature ~E_i in R^{C_s x H_i x W_i}, deeper decoder state
  D_{i+1} in R^{C_g x H_i/2 x W_i/2}, and deeper auxiliary mask probability
  M_{i+1} = sigma(Z_{i+1}) in [0, 1]^{1 x H_i/2 x W_i/2}:
    1. Semantic Gating Projection:
         G = BN(Conv_{1x1}(Up_{x2}(D_{i+1})))
         F_sg = ~E_i + G
    2. Differentiable Mask Prior Concatenation (NO .detach()):
         M_up = Up_{x2}(M_{i+1})
         X_mid = ReLU(BN(Conv_{1x1}([F_sg || M_up])))
    3. Local (d=1) and Expanded-Context (d=2) Dilated Depthwise Branches:
         K_sum = ReLU(BN(DW_{3x3, d=1}(X_mid) + DW_{3x3, d=2}(X_mid)))
    4. Attention Gate & Gated Skip Refinement (Equation 3):
         alpha = sigma(BN(Conv_{1x1}(K_sum)))
         S_i   = ReLU(BN(Conv_{3x3}(~E_i (x) alpha)))
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGuidedLGAG(nn.Module):
    """
    Mask-Guided Local-Global Attention Gate (Section 3.2, Eq. 3, Figure 3).

    Combines local (d=1) and expanded-context (d=2) depthwise receptive fields guided
    by coarse global semantic priors (D_{i+1}, M_{i+1}) from deeper decoder stages.
    Preserves end-to-end top-down boundary gradient flow by concatenating M_up without .detach().
    """

    def __init__(self, skip_channels: int, gating_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or skip_channels
        self.skip_channels = skip_channels
        self.gating_channels = gating_channels

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
    ) -> torch.Tensor:
        B, C, H, W = skip.shape

        # 1. Upsample deeper decoder state D_{i+1} and project: G = BN(Conv_{1x1}(Up_{x2}(D_{i+1})))
        g_up = F.interpolate(gating, size=(H, W), mode="bilinear", align_corners=False)
        g_proj = self.g_proj(g_up)

        # 2. Upsample auxiliary mask prior M_{i+1} without .detach()
        if mask_prob is not None:
            m_up = F.interpolate(mask_prob, size=(H, W), mode="bilinear", align_corners=False)
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
        return self.out_conv(gated_skip)
