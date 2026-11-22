"""
Weight-Space Model Soups, Generalized Mean (GeM) Multi-Scale TTA, and ModelEMA — Section 3.3,
Equation (4), and Supplementary Appendix B Equation (11).

Mathematical Formulation:
  1. Weight-Space Model Soup:
       theta_Soup = 0.70 * theta_Base + 0.30 * theta_Asym
  2. Generalized Mean (GeM) Multi-Scale Test-Time Augmentation (Equation 4):
       P_fused = ( sum_{s in S} w_s * (1 / |V_s|) * sum_{v in V_s} P_{s,v}^p )^(1 / p)
     where p = 1.75 for 8-view test-time ensembling and p = 1.5 for online teacher fusion.
"""

import copy
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights (decay lambda = 0.996, Section 4).
    """

    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach().to(v.device), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


def build_weight_soup(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Tuple[float, ...] = (0.70, 0.30),
) -> Dict[str, torch.Tensor]:
    """
    Interpolates checkpoints in parameter space (Wortsman et al., ICML 2022; Section 3.3):
      theta_Soup = sum_k alpha_k * theta_k,   with sum_k alpha_k = 1.
    Default weights (0.70, 0.30) combine canonical baseline M_1 and recall-oriented M_4.
    """
    w_sum = float(sum(weights))
    norm_w = [float(w) / w_sum for w in weights]
    soup_sd = {}
    ref_sd = state_dicts[0]

    for k, v0 in ref_sd.items():
        if v0.dtype.is_floating_point:
            acc = torch.zeros_like(v0, dtype=torch.float32, device="cpu")
            for sd, w in zip(state_dicts, norm_w):
                if w != 0.0:
                    acc.add_(sd[k].detach().cpu().float(), alpha=w)
            soup_sd[k] = acc.to(v0.dtype)
        else:
            soup_sd[k] = v0.detach().cpu().clone()
    return soup_sd


def build_decoupled_weight_soup(
    state_dicts: List[Dict[str, torch.Tensor]],
    enc_weights: Tuple[float, ...],
    dec_weights: Tuple[float, ...],
) -> Dict[str, torch.Tensor]:
    """
    Stage-decoupled weight soup allowing separate interpolation ratios for encoder vs. decoder.
    """
    out_sd = {}
    ref_sd = state_dicts[0]
    for k, v0 in ref_sd.items():
        if v0.dtype.is_floating_point:
            w_vec = enc_weights if k.startswith("backbone.") else dec_weights
            acc = torch.zeros_like(v0, dtype=torch.float32, device="cpu")
            for idx, sd in enumerate(state_dicts):
                if w_vec[idx] != 0.0:
                    acc.add_(sd[k].detach().cpu().float(), alpha=float(w_vec[idx]))
            out_sd[k] = acc.to(v0.dtype)
        else:
            out_sd[k] = v0.detach().cpu().clone()
    return out_sd


@torch.no_grad()
def extract_multiscale_views(
    model: nn.Module,
    loader,
    device: torch.device,
    scales: Tuple[int, int, int] = (384, 448, 352),
):
    """
    Extracts 8 multi-scale and dihedral flip views for a single 29.50M GDF-Net model:
      - Scale s_1 (384x384): 4 views (Original, Horizontal Flip, Vertical Flip, HV Flip)
      - Scale s_2 (448x448): 2 views (Original, Horizontal Flip)
      - Scale s_3 (352x352): 2 views (Original, Horizontal Flip)
    """
    model.eval()
    s1, s2, s3 = scales
    v1_list, v2_list, v3_list = [], [], []
    gt_list, names_list = [], []
    use_amp = device.type == "cuda"

    for imgs, msks, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        B, C, H, W = imgs.shape

        with autocast("cuda", enabled=use_amp):
            p0 = torch.sigmoid(model(imgs))
            p1 = torch.flip(torch.sigmoid(model(torch.flip(imgs, [3]))), [3])
            p2 = torch.flip(torch.sigmoid(model(torch.flip(imgs, [2]))), [2])
            p3 = torch.flip(torch.sigmoid(model(torch.flip(imgs, [2, 3]))), [2, 3])

            imgs_s2 = F.interpolate(imgs, size=(s2, s2), mode="bilinear", align_corners=False)
            p4 = F.interpolate(torch.sigmoid(model(imgs_s2)), size=(H, W), mode="bilinear", align_corners=False)
            p5 = F.interpolate(
                torch.flip(torch.sigmoid(model(torch.flip(imgs_s2, [3]))), [3]),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

            imgs_s3 = F.interpolate(imgs, size=(s3, s3), mode="bilinear", align_corners=False)
            p6 = F.interpolate(torch.sigmoid(model(imgs_s3)), size=(H, W), mode="bilinear", align_corners=False)
            p7 = F.interpolate(
                torch.flip(torch.sigmoid(model(torch.flip(imgs_s3, [3]))), [3]),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        v1_list.append(torch.stack([p0, p1, p2, p3], dim=1).squeeze(2).cpu().numpy().astype(np.float16))
        v2_list.append(torch.stack([p4, p5], dim=1).squeeze(2).cpu().numpy().astype(np.float16))
        v3_list.append(torch.stack([p6, p7], dim=1).squeeze(2).cpu().numpy().astype(np.float16))
        gt_list.append(msks.squeeze(1).numpy().astype(np.uint8))
        names_list.extend(names)

    return (
        np.concatenate(v1_list, axis=0),
        np.concatenate(v2_list, axis=0),
        np.concatenate(v3_list, axis=0),
        np.concatenate(gt_list, axis=0),
        names_list,
    )


def fuse_gem_views(
    v_base: np.ndarray,
    v_up: np.ndarray,
    v_down: np.ndarray,
    mode: str = "ms8_gem",
    gem_p: float = 1.75,
    scale_weights: Tuple[float, float, float] = (4.0 / 6.0, 2.0 / 6.0, 0.0),
) -> np.ndarray:
    """
    Generalized Mean (GeM) Multi-Scale Test-Time Augmentation Fusion — Section 3.3, Eq. (4)
    and Supplementary Appendix B, Eq. (11):
      P_fused = ( sum_{s in S} w_s * (1 / |V_s|) * sum_{v in V_s} P_{s,v}^p )^(1 / p)
    """
    if mode == "sv":
        return v_base[:, 0].astype(np.float32)

    eps = 1e-6
    if mode == "tta4":
        v = np.clip(v_base.astype(np.float32), eps, 1.0)
        if abs(gem_p - 1.0) < 1e-4:
            return np.mean(v, axis=1)
        return np.power(np.mean(np.power(v, gem_p), axis=1), 1.0 / gem_p)

    w1, w2, w3 = scale_weights
    v1 = np.clip(v_base.astype(np.float32), eps, 1.0)
    v2 = np.clip(v_up.astype(np.float32), eps, 1.0)
    v3 = np.clip(v_down.astype(np.float32), eps, 1.0)

    if abs(gem_p - 1.0) < 1e-4:
        return w1 * np.mean(v1, axis=1) + w2 * np.mean(v2, axis=1) + w3 * np.mean(v3, axis=1)

    p1 = np.mean(np.power(v1, gem_p), axis=1)
    p2 = np.mean(np.power(v2, gem_p), axis=1)
    p3 = np.mean(np.power(v3, gem_p), axis=1)
    return np.power(w1 * p1 + w2 * p2 + w3 * p3, 1.0 / gem_p)
