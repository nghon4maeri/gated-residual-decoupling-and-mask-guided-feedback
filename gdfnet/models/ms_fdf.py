"""
Multi-Scale Feature Decoupling Filter (MS-FDF) — Section 3.1, Equations (1) & (2), Figure 2.

Mathematical Formulation:
  1. Multi-Scale Depthwise Extraction:
       L_k = DWConv_{k x k}(E_i),   k in {1, 3, 5}
       L_cat = [L_1 || L_3 || L_5] in R^{3C x H_i x W_i}
  2. Spatial Residual Decoupling (Equation 1):
       x_low  = ReLU(BN(Conv_{1x1}(L_cat)))
       x_high = E_i - x_low
  3. Channel-Spatial Soft Gating & Residual Fusion (Equation 2):
       beta   = sigma(W_gate(x_low))
       x_filt = x_high (x) beta
       F_fuse = [x_low || x_filt]
       ~E_i   = Proj(F_fuse) + E_i
"""

import torch
import torch.nn as nn


class MultiScaleFrequencyFilter(nn.Module):
    """
    Multi-Scale Feature Decoupling Filter (MS-FDF, Section 3.1, Eqs. 1-2).

    Decouples homogeneous regional structure (x_low) from local high-gradient
    boundary residuals and high-contrast artifacts (x_high = E_i - x_low), then
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        out = self.fusion(f_fuse)
        return out + x
