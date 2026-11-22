"""
Evaluation Metrics and Statistical Testing Utilities (Section 4 & Section 5).

Computes:
  - Overlap metrics: Macro Dice, Median Dice, Global Micro Dice, mIoU (Jaccard),
    Sensitivity (Recall), Specificity, Precision, and Low-IoU failure count (J < 0.65).
  - Boundary metrics: 95th-percentile Hausdorff Distance (HD95, px), Average Surface
    Distance (ASD, px), and Boundary F2 score (BF2, 2-px tolerance).
  - Statistical hypothesis tests (with GDF-Net (Real-Time) as reference [Ref.]):
    Two-sided paired t-test, Wilcoxon signed-rank test, Holm-Bonferroni correction,
    and 2,000-sample non-parametric bootstrap 95% confidence intervals.
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.stats import ttest_rel, wilcoxon


def compute_surface_distances(
    pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: float = 2.0
) -> Tuple[float, float, float]:
    """
    Computes HD95 (px), ASD (px), and Boundary F2 (BF2 at 2-px tolerance).
    """
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)
    if not np.any(pred_b) and not np.any(gt_b):
        return 0.0, 0.0, 1.0
    if not np.any(pred_b) or not np.any(gt_b):
        diag = float(np.sqrt(pred_b.shape[0] ** 2 + pred_b.shape[1] ** 2))
        return diag, diag / 2.0, 0.0

    pred_border = np.logical_xor(pred_b, binary_erosion(pred_b))
    gt_border = np.logical_xor(gt_b, binary_erosion(gt_b))
    if not np.any(pred_border):
        pred_border = pred_b
    if not np.any(gt_border):
        gt_border = gt_b

    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    hd95 = float(np.percentile(all_d, 95))
    asd = float((d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0)

    prec_b = float(np.mean(d_pred_to_gt <= tolerance_px))
    rec_b = float(np.mean(d_gt_to_pred <= tolerance_px))
    bf2 = (
        float(2.0 * prec_b * rec_b / (prec_b + rec_b + 1e-8))
        if (prec_b + rec_b) > 0
        else 0.0
    )
    return hd95, asd, bf2


def compute_segmentation_metrics(
    pred_mask: np.ndarray, gt_mask: np.ndarray
) -> Dict[str, float]:
    """Computes all overlap and boundary metrics for a single (pred_mask, gt_mask) pair."""
    p_f = (pred_mask > 0).flatten().astype(np.uint8)
    g_f = (gt_mask > 0).flatten().astype(np.uint8)

    tp = float(np.sum((p_f == 1) & (g_f == 1)))
    fp = float(np.sum((p_f == 1) & (g_f == 0)))
    fn = float(np.sum((p_f == 0) & (g_f == 1)))
    tn = float(np.sum((p_f == 0) & (g_f == 0)))

    dice = (2.0 * tp + 1e-7) / (2.0 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    j_thresh = iou if iou >= 0.65 else 0.0
    sens = (tp + 1e-7) / (tp + fn + 1e-7)
    spec = (tn + 1e-7) / (tn + fp + 1e-7)
    prec = (tp + 1e-7) / (tp + fp + 1e-7)

    hd95, asd, bf2 = compute_surface_distances(pred_mask, gt_mask)
    return {
        "Dice": float(dice),
        "mIoU": float(iou),
        "J_thresh": float(j_thresh),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "Precision": float(prec),
        "HD95": float(hd95),
        "ASD": float(asd),
        "BF2": float(bf2),
    }


def evaluate_cohort_predictions(
    probs: np.ndarray,
    gts: np.ndarray,
    names: List[str],
    thresholds: Union[float, np.ndarray] = 0.50,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluates a batch of probability maps [N, H, W] against ground-truth masks [N, H, W]
    and returns both summary statistics and per-image DataFrame.
    """
    n_samples = len(probs)
    if np.ndim(thresholds) == 0:
        t_vec = np.full(n_samples, float(thresholds), dtype=np.float32)
    else:
        t_vec = np.asarray(thresholds, dtype=np.float32)

    records = []
    for i in range(n_samples):
        pred_bin = (probs[i] >= float(t_vec[i])).astype(np.uint8)
        gt_bin = (gts[i] > 0).astype(np.uint8)
        m = compute_segmentation_metrics(pred_bin, gt_bin)
        m["image"] = names[i]
        m["threshold"] = float(t_vec[i])
        records.append(m)

    df = pd.DataFrame(records)
    low_iou_count = int((df["mIoU"] < 0.65).sum())

    summary = {
        "N": n_samples,
        "Mean_Threshold": float(np.mean(t_vec)),
        "Macro_Dice": float(df["Dice"].mean()),
        "Median_Dice": float(df["Dice"].median()),
        "mIoU": float(df["mIoU"].mean()),
        "BF2": float(df["BF2"].mean()),
        "HD95": float(df["HD95"].mean()),
        "ASD": float(df["ASD"].mean()),
        "Sensitivity": float(df["Sensitivity"].mean()),
        "Specificity": float(df["Specificity"].mean()),
        "Precision": float(df["Precision"].mean()),
        "Low_IoU_Count": low_iou_count,
    }
    return summary, df


def paired_statistical_tests(
    ref_scores: np.ndarray,
    comp_scores: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes two-sided paired t-test, Wilcoxon signed-rank test, and 2,000-sample
    bootstrap 95% confidence interval for (ref_scores - comp_scores), taking
    GDF-Net (Real-Time) as reference [Ref.] (Section 4 & Section 5).
    """
    ref_arr = np.asarray(ref_scores, dtype=np.float64)
    comp_arr = np.asarray(comp_scores, dtype=np.float64)
    diff = ref_arr - comp_arr

    t_stat, t_pval = ttest_rel(ref_arr, comp_arr)
    try:
        w_stat, w_pval = wilcoxon(ref_arr, comp_arr, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_stat, w_pval = 0.0, 1.0

    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_means = np.array(
        [np.mean(diff[rng.integers(0, n, size=n)]) for _ in range(n_bootstrap)]
    )
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "ref_mean": float(np.mean(ref_arr)),
        "comp_mean": float(np.mean(comp_arr)),
        "mean_diff": float(np.mean(diff)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "ttest_t": float(t_stat),
        "ttest_p": float(t_pval),
        "wilcoxon_w": float(w_stat),
        "wilcoxon_p": float(w_pval),
    }
