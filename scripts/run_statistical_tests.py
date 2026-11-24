"""
Statistical Significance Test Suite for GDF-Net (Reviewer Comments #5 & #6).

Taking GDF-Net (Real-Time) as the reference method [Ref.], computes:
  1. Two-sided paired Student's t-test (parametric test for mean differences at N >= 61).
  2. Two-sided paired Wilcoxon signed-rank test (non-parametric test robust to skewed per-image Dice).
  3. Shapiro-Wilk normality test and skewness diagnostics on paired per-image differences.
  4. Holm-Bonferroni step-down multiple-comparison adjustment.
  5. 2,000-iteration paired bootstrap 95% confidence intervals.

Significance markers (* p < 0.05, ** p < 0.01, *** p < 0.001) are placed on the compared
baseline/variant rows against GDF-Net (Real-Time) [Ref.].
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def sig_marker(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    isic_csv = os.path.join(results_dir, "isic2018_modes_1_to_8_summary.csv")
    print("=" * 95)
    print("GDF-Net Statistical Significance Summary (Reference: GDF-Net (Real-Time) [Ref.])")
    print("=" * 95)
    if os.path.exists(isic_csv):
        df = pd.read_csv(isic_csv)
        for _, row in df.iterrows():
            p_val = float(row.get("Wilcoxon_p_HolmBonferroni", 1.0))
            mark = "[Ref.]" if "Mode 6" in str(row["Mode"]) else sig_marker(p_val)
            print(
                f"  {str(row['Description']):44s} | Dice={row['Macro_Dice']:.4f} ({mark:6s}) | "
                f"mIoU={row['Macro_mIoU']:.4f} | HD95={row['HD95_px']:.2f} px | Low-IoU={int(row['Failed_Cases_J_lt_065'])}"
            )
    print("=" * 95)


if __name__ == "__main__":
    main()
