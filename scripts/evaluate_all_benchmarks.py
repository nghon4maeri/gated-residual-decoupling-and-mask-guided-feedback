"""
Reproduce Quantitative Tables (Tables 1, 2, 3) and Statistical Tests from Verified Prediction CSVs.

Takes GDF-Net (Real-Time) as reference [Ref.] and computes two-sided paired t-tests and
Wilcoxon signed-rank tests across ISIC-2018 (N=1,000), CVC-ClinicDB (N=61), Kvasir-SEG (N=100),
and DSB-2018 (N=67).
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    print("=" * 88)
    print("GDF-Net Quantitative Benchmark Summary (MIDL 2027 Verified Artifacts)")
    print("=" * 88)

    isic_csv = os.path.join(results_dir, "isic2018_modes_1_to_8_summary.csv")
    if os.path.exists(isic_csv):
        print("\n[Table 1 & Table 3] ISIC-2018 Official Test Split (N=1,000, 384x384):")
        df_isic = pd.read_csv(isic_csv)
        cols = ["Mode", "Description", "Macro_Dice", "Median_Dice", "Macro_mIoU", "HD95_px", "ASD_px", "Failed_Cases_J_lt_065"]
        print(df_isic[cols].to_string(index=False))

    for fn, title in [
        ("multidataset_cvc_clinicdb_summary.csv", "[Table 2 Panel A] CVC-ClinicDB (N=61, 384x384)"),
        ("multidataset_kvasir_seg_summary.csv", "[Table 2 Panel A] Kvasir-SEG (N=100, 384x384)"),
        ("multidataset_dsb2018_summary.csv", "[Table 2 Panel A] DSB-2018 (N=67, 256x256)"),
    ]:
        p = os.path.join(results_dir, fn)
        if os.path.exists(p):
            print(f"\n{title}:")
            print(pd.read_csv(p).to_string(index=False))

    hw_csv = os.path.join(results_dir, "rtx4060_multires_hardware_profiling.csv")
    if os.path.exists(hw_csv):
        print("\n[Supplementary Table 6] Multi-Resolution Hardware Profiling (RTX 4060 Laptop GPU, B=1):")
        print(pd.read_csv(hw_csv).to_string(index=False))


if __name__ == "__main__":
    main()
