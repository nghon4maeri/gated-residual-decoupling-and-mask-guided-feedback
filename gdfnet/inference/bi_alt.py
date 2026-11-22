"""
Scale-Anchored Bi-Parametric Adaptive Thresholding (Bi-ALT) — Section 3.4, Equations (6)-(7),
Supplementary Appendix A Equations (8)-(10), Figure 4.

Mathematical Formulation:
  Given probability map P = sigma(Z) in [0, 1]^{H x W} and validation-calibrated base
  cutoff T_base:
    1. Scale-Anchored Probe Thresholds (Appendix A, Eq. 8):
         t_low  = clip(T_base - 0.39, 0.25, 0.65)   [= 0.35 at T_base = 0.74]
         t_mid  = clip(T_base - 0.24, 0.35, 0.75)   [= 0.50 at T_base = 0.74]
         t_high = clip(T_base + 0.01, 0.55, 0.93)   [= 0.75 at T_base = 0.74]
    2. Morphological Ambiguity & Scale Indicators (Section 3.4, Eq. 6):
         A_c   = sum_{u,v} I(P(u, v) > t_c)
         rho   = A_low / max(A_high, 1.0)
         a_mid = A_mid / (H * W)
    3. Three-Branch Top-Down Decision Policy (Section 3.4, Eq. 7):
         T_i = max(0.45, T_base - dt_high),   if rho > 1.65 or a_mid > 0.45   (Branch 1: Diffuse)
         T_i = max(0.50, T_base - dt_mid),    elif rho > 1.32 or a_mid > 0.28 (Branch 2: Transitional)
         T_i = min(0.94, T_base + dt_crisp),  otherwise                       (Branch 3: Crisp)
       where (dt_high, dt_mid, dt_crisp) = (0.18, 0.06, 0.00), yielding T_i in {0.56, 0.68, 0.74}
       at T_base = 0.74.
    4. Binary Segmentation Mask:
         ^Y(u, v) = I(P(u, v) >= T_i)
"""

from typing import Dict, Tuple, Union, Any, Optional
import cv2
import numpy as np
import torch


def compute_bi_alt_threshold(
    prob_map: Union[np.ndarray, torch.Tensor],
    t_base: float = 0.74,
    dt_high: float = 0.18,
    dt_mid: float = 0.06,
    dt_crisp: float = 0.00,
    rho_diffuse: float = 1.65,
    rho_trans: float = 1.32,
    area_diffuse: float = 0.45,
    area_trans: float = 0.28,
    custom_probes: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, int, float, float]:
    """
    Computes the sample-specific threshold T_i, branch index (1, 2, or 3),
    boundary ambiguity ratio rho, and intermediate area fraction a_mid
    according to Equations (6) & (7) of the manuscript.
    """
    if isinstance(prob_map, torch.Tensor):
        p = prob_map.detach().cpu().float().numpy()
    else:
        p = np.asarray(prob_map, dtype=np.float32)

    if p.ndim == 3:
        p = p.squeeze(0)

    if custom_probes is not None:
        t_low, t_mid, t_high = custom_probes
    else:
        # Appendix A Equation (8): Scale-anchored probes
        t_low = float(np.clip(t_base - 0.39, 0.25, 0.65))
        t_mid = float(np.clip(t_base - 0.24, 0.35, 0.75))
        t_high = float(np.clip(t_base + 0.01, 0.55, 0.93))

    total_pixels = float(p.shape[0] * p.shape[1])
    area_low = float(np.sum(p > t_low))
    area_high = float(np.sum(p > t_high))
    a_mid = float(np.sum(p > t_mid)) / max(total_pixels, 1.0)

    # Equation (6): Boundary ambiguity ratio rho
    rho = area_low / max(area_high, 1.0)

    # Equation (7): Top-down three-branch decision rule
    if rho > rho_diffuse or a_mid > area_diffuse:
        t_i = max(0.45, t_base - dt_high)
        branch = 1  # Diffuse / Large Lesion
    elif rho > rho_trans or a_mid > area_trans:
        t_i = max(0.50, t_base - dt_mid)
        branch = 2  # Transitional Margin
    else:
        t_i = min(0.94, t_base + dt_crisp)
        branch = 3  # Crisp Margin

    return float(t_i), int(branch), float(rho), float(a_mid)


class ScaleAnchoredBiALT:
    """
    Training-Free Scale-Anchored Bi-Parametric Adaptive Thresholding (Bi-ALT) Module.

    Supports:
      - Per-image adaptive threshold assignment T_i and binary mask generation ^Y = I(P >= T_i)
      - Batch threshold computation across N test images
      - Validation grid calibration of T_base and step offsets on held-out validation splits
    """

    def __init__(
        self,
        t_base: float = 0.74,
        dt_high: float = 0.18,
        dt_mid: float = 0.06,
        dt_crisp: float = 0.00,
        rho_diffuse: float = 1.65,
        rho_trans: float = 1.32,
        area_diffuse: float = 0.45,
        area_trans: float = 0.28,
    ):
        self.t_base = t_base
        self.dt_high = dt_high
        self.dt_mid = dt_mid
        self.dt_crisp = dt_crisp
        self.rho_diffuse = rho_diffuse
        self.rho_trans = rho_trans
        self.area_diffuse = area_diffuse
        self.area_trans = area_trans

    def get_threshold(
        self, prob_map: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[float, int, float, float]:
        return compute_bi_alt_threshold(
            prob_map,
            t_base=self.t_base,
            dt_high=self.dt_high,
            dt_mid=self.dt_mid,
            dt_crisp=self.dt_crisp,
            rho_diffuse=self.rho_diffuse,
            rho_trans=self.rho_trans,
            area_diffuse=self.area_diffuse,
            area_trans=self.area_trans,
        )

    def get_batch_thresholds(self, probs: np.ndarray) -> np.ndarray:
        """Computes adaptive threshold vector T = [T_1, ..., T_N] for a batch [N, H, W]."""
        n_samples = len(probs)
        t_vec = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            t_vec[i], _, _, _ = self.get_threshold(probs[i])
        return t_vec

    def binarize(self, prob_map: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Returns binary segmentation mask ^Y(u, v) = I(P(u, v) >= T_i)."""
        if isinstance(prob_map, torch.Tensor):
            p = prob_map.detach().cpu().float().numpy()
        else:
            p = np.asarray(prob_map, dtype=np.float32)
        if p.ndim == 3 and p.shape[0] > 1:
            t_vec = self.get_batch_thresholds(p)
            return (p >= t_vec[:, None, None]).astype(np.uint8)
        t_i, _, _, _ = self.get_threshold(p)
        return (p >= t_i).astype(np.uint8)


def compute_otsu_threshold(prob_map: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Classical Otsu intensity-histogram adaptive thresholding baseline
    (reported in Table 3 Panel B: 0.9096 Validation Macro Dice on ISIC-2018).
    """
    p_u8 = np.clip(np.round(prob_map * 255.0), 0, 255).astype(np.uint8)
    otsu_val, bin_mask = cv2.threshold(p_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(otsu_val) / 255.0, bin_mask.astype(np.uint8)


def _fast_batch_dice(probs: np.ndarray, gts: np.ndarray, thresh_vec: Union[float, np.ndarray]) -> float:
    if np.ndim(thresh_vec) == 0:
        preds = (probs >= float(thresh_vec)).astype(np.uint8)
    else:
        preds = (probs >= thresh_vec[:, None, None]).astype(np.uint8)
    p_f = preds.reshape(len(preds), -1)
    g_f = gts.reshape(len(gts), -1)
    inter = (p_f * g_f).sum(axis=1)
    union = p_f.sum(axis=1) + g_f.sum(axis=1)
    dices = np.where(union == 0, 1.0, (2.0 * inter + 1e-7) / (union + 1e-7))
    return float(np.mean(dices))


def calibrate_bi_alt_on_validation(
    val_probs: np.ndarray, val_gts: np.ndarray
) -> Dict[str, Any]:
    """
    Calibrates T_base in [0.35, 0.93] and step offsets (dt_high, dt_mid, dt_crisp)
    solely on the held-out validation cohort (N=100) prior to test evaluation.
    """
    best_t_base, best_base_dice = 0.74, -1.0
    for t in np.linspace(0.35, 0.93, 59):
        d = _fast_batch_dice(val_probs, val_gts, float(t))
        if d > best_base_dice:
            best_base_dice = d
            best_t_base = float(t)

    best_alt_dice = best_base_dice
    best_params = {
        "t_base": best_t_base,
        "dt_high": 0.18,
        "dt_mid": 0.06,
        "dt_crisp": 0.00,
        "val_base_dice": best_base_dice,
        "val_alt_dice": best_base_dice,
    }

    for dt_h in [0.00, 0.06, 0.10, 0.14, 0.18, 0.22, 0.25]:
        for dt_m in [0.00, 0.03, 0.06, 0.09, 0.12]:
            if dt_m > dt_h:
                continue
            for dt_c in [0.00, 0.01, 0.02]:
                bi_alt = ScaleAnchoredBiALT(
                    t_base=best_t_base, dt_high=dt_h, dt_mid=dt_m, dt_crisp=dt_c
                )
                t_vec = bi_alt.get_batch_thresholds(val_probs)
                d = _fast_batch_dice(val_probs, val_gts, t_vec)
                if d > best_alt_dice:
                    best_alt_dice = d
                    best_params.update(
                        {
                            "dt_high": float(dt_h),
                            "dt_mid": float(dt_m),
                            "dt_crisp": float(dt_c),
                            "val_alt_dice": float(d),
                        }
                    )
    return best_params


def run_125_probe_sensitivity_grid(
    val_probs: np.ndarray, val_gts: np.ndarray, t_base: float = 0.74
) -> Dict[str, float]:
    """
    Executes the exhaustive 5 x 5 x 5 = 125 validation probe cutoff sensitivity analysis
    reported in Section 3.4 and Supplementary Appendix A.2:
      t_low  in {0.25, 0.30, 0.35, 0.40, 0.45}
      t_mid  in {0.40, 0.45, 0.50, 0.55, 0.60}
      t_high in {0.65, 0.70, 0.75, 0.80, 0.85}
    """
    t_low_grid = [0.25, 0.30, 0.35, 0.40, 0.45]
    t_mid_grid = [0.40, 0.45, 0.50, 0.55, 0.60]
    t_high_grid = [0.65, 0.70, 0.75, 0.80, 0.85]

    grid_dices = []
    for tl in t_low_grid:
        for tm in t_mid_grid:
            for th in t_high_grid:
                t_vec = np.array(
                    [
                        compute_bi_alt_threshold(
                            p, t_base=t_base, custom_probes=(tl, tm, th)
                        )[0]
                        for p in val_probs
                    ],
                    dtype=np.float32,
                )
                grid_dices.append(_fast_batch_dice(val_probs, val_gts, t_vec))

    arr = np.array(grid_dices, dtype=np.float64)
    return {
        "num_configs": len(arr),
        "mean_val_dice": float(np.mean(arr)),
        "std_val_dice": float(np.std(arr)),
        "min_val_dice": float(np.min(arr)),
        "max_val_dice": float(np.max(arr)),
        "spread_pp": float((np.max(arr) - np.min(arr)) * 100.0),
    }
