"""
Loss Functions for GDF-Net — Section 3.3, Equation (5), and Supplementary Table 5.

Includes:
  1. StructureLoss & GDFNetCompositeLoss: Edge-weighted BCE + Edge-weighted IoU across
     deep-supervision heads (used for canonical baseline M_1).
  2. AsymmetricFocalTverskyLoss: Recall-oriented loss penalizing false negatives
     (alpha=0.30, beta=0.70, gamma=1.25-1.33, used for recall constituent M_4 in Weight Soup).
  3. EnsembleDistillationLoss: Online Ensemble-to-Single Knowledge Distillation objective
     (Section 3.3, Equation 5 & Supplementary Table 5):
       L_total = 0.45 * L_GT(Z_s, Y_GT) + 0.55 * L_KD(Z_s, Q_t, Z_t)
       L_GT    = 0.30 * L_wBCE(w_pos=1.18) + 0.35 * L_AsymFT + 0.35 * L_Dice
                 weighted by (0.50, 0.20, 0.15, 0.10, 0.05) across main + 4 aux heads
       L_KD    = 0.45 * L_SoftDice(sigma(Z_s), Q_t)
                 + 0.35 * L_wSoftBCE(Z_s, Q_t)
                 + 0.20 * tau^2 * MSE(sigma(Z_s / tau), sigma(Z_t / tau)),  tau = 2.0
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    """
    Edge-Weighted Structure Loss (Weighted BCE + Weighted IoU).

    Increases pixel supervision weight w = 1 + 5 * |Y - AvgPool_{31x31}(Y)| near
    ambiguous anatomical boundaries.
    """

    def __init__(self, kernel_size: int = 31, weight_factor: float = 5.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_factor = weight_factor
        self.avg_pool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            local_mean = self.avg_pool(targets)
            boundary_diff = torch.abs(targets - local_mean)
            weight = 1.0 + self.weight_factor * boundary_diff

        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
        wbce = (weight * bce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-8)

        probs = torch.sigmoid(pred_logits)
        intersection = (weight * probs * targets).sum(dim=(2, 3))
        union = (weight * (probs + targets - probs * targets)).sum(dim=(2, 3))
        wiou = 1.0 - (intersection + 1.0) / (union + 1.0)

        return (wbce + wiou).mean()


class GDFNetCompositeLoss(nn.Module):
    """
    Deep-Supervision Structure Loss + Direct Macro-Dice Loss across main and 4 auxiliary heads.
    """

    def __init__(self, macro_dice_weight: float = 0.5):
        super().__init__()
        self.structure_loss = StructureLoss()
        self.macro_dice_weight = macro_dice_weight

    @staticmethod
    def _soft_macro_dice_loss(pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(pred_logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

    def _single_head_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_struct = self.structure_loss(logits, targets)
        if self.macro_dice_weight > 0.0:
            return l_struct + self.macro_dice_weight * self._soft_macro_dice_loss(logits, targets)
        return l_struct

    def forward(
        self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], targets: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(outputs, dict):
            l_final = self._single_head_loss(outputs["logits"], targets)
            l_p1 = self._single_head_loss(outputs["p1"], targets)
            l_p2 = self._single_head_loss(outputs["p2"], targets)
            l_p3 = self._single_head_loss(outputs["p3"], targets)
            l_p4 = self._single_head_loss(outputs["p4"], targets)
            return l_final + l_p1 + 0.5 * l_p2 + 0.3 * l_p3 + 0.2 * l_p4
        return self._single_head_loss(outputs, targets)


class AsymmetricFocalTverskyLoss(nn.Module):
    """
    Recall-Oriented Asymmetric Focal-Tversky Loss (Section 3.3, M_4 checkpoint in Weight Soup).

    Penalizes false negatives (beta = 0.70) more strongly than false positives (alpha = 0.30)
    with focal exponent gamma = 1.25 to recover faint peripheral lesion margins.
    """

    def __init__(
        self,
        alpha: float = 0.30,
        beta: float = 0.70,
        gamma: float = 1.25,
        kernel_size: int = 31,
        weight_factor: float = 4.0,
        tversky_weight: float = 0.60,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight_factor = weight_factor
        self.tversky_weight = tversky_weight
        self.avg_pool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward_single(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            edge = torch.abs(targets - self.avg_pool(targets))
            weight = 1.0 + self.weight_factor * edge

        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
        wbce = (weight * bce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-8)

        probs = torch.sigmoid(pred_logits)
        intersection = (weight * probs * targets).sum(dim=(2, 3))
        union = (weight * (probs + targets - probs * targets)).sum(dim=(2, 3))
        wiou = 1.0 - (intersection + 1.0) / (union + 1.0)

        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(2, 3))
        tversky_idx = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        focal_tversky = torch.pow(torch.clamp(1.0 - tversky_idx, min=1e-6), self.gamma)

        return (0.5 * (wbce + wiou) + self.tversky_weight * focal_tversky).mean()

    def forward(
        self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], targets: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(outputs, dict):
            l_final = self.forward_single(outputs["logits"], targets)
            l_p1 = self.forward_single(outputs["p1"], targets)
            l_p2 = self.forward_single(outputs["p2"], targets)
            l_p3 = self.forward_single(outputs["p3"], targets)
            l_p4 = self.forward_single(outputs["p4"], targets)
            return l_final + 0.6 * l_p1 + 0.4 * l_p2 + 0.2 * l_p3 + 0.1 * l_p4
        return self.forward_single(outputs, targets)


class EnsembleDistillationLoss(nn.Module):
    """
    Ensemble-to-Single Knowledge Distillation Loss — Section 3.3, Equation (5) & Appendix C Table 5.

    Mathematical Formulation:
      L_total = (1 - lambda_KD) * L_GT(Z_s, Y_GT) + lambda_KD * L_KD(Z_s, Q_t, Z_t),  lambda_KD = 0.55

      1. Ground-Truth Deep-Supervision Loss L_GT across main + 4 aux heads
         with normalized weights (0.50, 0.20, 0.15, 0.10, 0.05) = (1.0, 0.4, 0.3, 0.2, 0.1) / 2.0:
           L_head(Z, Y) = 0.30 * L_wBCE(w_pos=1.18) + 0.35 * L_AsymFT(alpha=0.42, beta=0.58, gamma=1.25) + 0.35 * L_Dice

      2. Multi-Component Knowledge Distillation Loss L_KD (Equation 5):
           L_KD = 0.45 * L_SoftDice(sigma(Z_s), Q_t)           [global region overlap alignment]
                + 0.35 * L_wSoftBCE(Z_s, Q_t)                  [edge-weighted boundary recall alignment]
                + 0.20 * tau^2 * MSE(~P_s, ~P_t)               [temperature-scaled dark-knowledge calibration, tau=2.0]
         where ~P = sigma(Z / tau).
    """

    def __init__(
        self,
        alpha_fp: float = 0.42,
        beta_fn: float = 0.58,
        gamma: float = 1.25,
        pos_weight: float = 1.18,
        kd_weight: float = 0.55,
        temp: float = 2.0,
        ds_weights: Tuple[float, float, float, float, float] = (0.50, 0.20, 0.15, 0.10, 0.05),
    ):
        super().__init__()
        self.alpha_fp = alpha_fp
        self.beta_fn = beta_fn
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.kd_weight = kd_weight
        self.temp = temp
        self.ds_weights = ds_weights

    def _single_scale_gt_loss(
        self, pred_logits: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        edge = torch.abs(
            target_mask - F.avg_pool2d(target_mask, kernel_size=31, stride=1, padding=15)
        )
        weight = 1.0 + 4.0 * edge

        pw = torch.tensor(
            [self.pos_weight], device=pred_logits.device, dtype=pred_logits.dtype
        )
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target_mask, pos_weight=pw, reduction="none"
        )
        wbce = (weight * bce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-8)

        pred = torch.sigmoid(pred_logits)
        smooth = 1e-5
        tp = (pred * target_mask).sum(dim=(1, 2, 3))
        fp = (pred * (1.0 - target_mask)).sum(dim=(1, 2, 3))
        fn = ((1.0 - pred) * target_mask).sum(dim=(1, 2, 3))

        tversky = (tp + smooth) / (tp + self.alpha_fp * fp + self.beta_fn * fn + smooth)
        asym_focal_tversky = torch.pow(torch.clamp(1.0 - tversky, min=1e-6), self.gamma)
        macro_dice_loss = 1.0 - (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)

        return (
            0.30 * wbce.squeeze(1) + 0.35 * asym_focal_tversky + 0.35 * macro_dice_loss
        ).mean()

    def _kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_prob: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Soft Dice loss on probability maps: 0.45 * L_SoftDice(sigma(Z_s), Q_t)
        pred_s = torch.sigmoid(student_logits)
        smooth = 1e-5
        inter = (pred_s * teacher_prob).sum(dim=(1, 2, 3))
        denom = (pred_s + teacher_prob).sum(dim=(1, 2, 3))
        soft_dice_kd = (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()

        # 2. Edge-weighted Soft BCE loss: 0.35 * L_wSoftBCE(Z_s, Q_t)
        edge = torch.abs(
            teacher_prob - F.avg_pool2d(teacher_prob, kernel_size=15, stride=1, padding=7)
        )
        w = 1.0 + 3.0 * edge
        soft_bce = F.binary_cross_entropy_with_logits(
            student_logits, teacher_prob, reduction="none"
        )
        wbce_kd = ((w * soft_bce).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-8)).mean()

        # 3. Temperature-scaled probability MSE: 0.20 * tau^2 * MSE(~P_s, ~P_t) at tau=2.0
        p_s_temp = torch.sigmoid(student_logits / self.temp)
        p_t_temp = torch.sigmoid(teacher_logits / self.temp)
        temp_mse = F.mse_loss(p_s_temp, p_t_temp) * (self.temp ** 2)

        return 0.45 * soft_dice_kd + 0.35 * wbce_kd + 0.20 * temp_mse

    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]],
        target: torch.Tensor,
        teacher_prob: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, dict):
            out_final = outputs["logits"]
            p1, p2, p3, p4 = outputs["p1"], outputs["p2"], outputs["p3"], outputs["p4"]
        elif isinstance(outputs, (tuple, list)):
            out_final, p1, p2, p3, p4 = outputs
        else:
            out_final = outputs
            p1 = p2 = p3 = p4 = None

        w0, w1, w2, w3, w4 = self.ds_weights
        if p1 is not None:
            loss_gt = (
                w0 * self._single_scale_gt_loss(out_final, target)
                + w1 * self._single_scale_gt_loss(p1, target)
                + w2 * self._single_scale_gt_loss(p2, target)
                + w3 * self._single_scale_gt_loss(p3, target)
                + w4 * self._single_scale_gt_loss(p4, target)
            )
            loss_kd = 0.75 * self._kd_loss(
                out_final, teacher_prob, teacher_logits
            ) + 0.25 * self._kd_loss(p1, teacher_prob, teacher_logits)
        else:
            loss_gt = self._single_scale_gt_loss(out_final, target)
            loss_kd = self._kd_loss(out_final, teacher_prob, teacher_logits)

        return (1.0 - self.kd_weight) * loss_gt + self.kd_weight * loss_kd


# Backward-compatible aliases
GDFNetV4CompositeLoss = GDFNetCompositeLoss
PA2PlusDistillationLoss = EnsembleDistillationLoss
