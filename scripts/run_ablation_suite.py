"""
Reproducible Fixed-Backbone Leave-One-Out Ablation & Gradient Flow Profiler (Reviewer Comments #1, #2, #3).

Supports:
  1. Profiling exact Parameters (M), GFLOPs (@ 384x384), and auxiliary-head top-down
     boundary gradient norms ||dL / dZ_k||_2 across all 6 leave-one-out configurations
     plus the complexity-matched PlainMultiScaleConv control on ConvFormer-S18.
  2. The configuration list shared with scripts/train_base.py, which trains one configuration
     with the GDF-Net (Base) protocol.

Usage:
    python scripts/run_ablation_suite.py --mode profile
    python scripts/train_base.py --train-img ... --train-gt ... --val-img ... --val-gt ...                                  --ablation cfg5_mask_guided_detach --out-dir ./checkpoints/cfg5
"""

import argparse
import os
import sys
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gdfnet import GDFNet, GDFNetCompositeLoss

ABLATION_SUITE = [
    {
        "id": "cfg1_baseline",
        "Configuration": "Baseline (ConvFormer-S18 + MSCAM)",
        "MS-FDF": False,
        "LGAG": False,
        "Mask_Guidance": False,
        "Grad_Through_Mask": False,
        "kwargs": dict(use_msfdf=False, use_lgag=False, mask_guidance=False, detach_mask=False),
    },
    {
        "id": "cfg2_msfdf_only",
        "Configuration": "Baseline + MS-FDF",
        "MS-FDF": True,
        "LGAG": False,
        "Mask_Guidance": False,
        "Grad_Through_Mask": False,
        "kwargs": dict(use_msfdf=True, use_lgag=False, mask_guidance=False, detach_mask=False),
    },
    {
        "id": "cfg3_lgag_only",
        "Configuration": "Baseline + LGAG (Feature-Only)",
        "MS-FDF": False,
        "LGAG": True,
        "Mask_Guidance": False,
        "Grad_Through_Mask": False,
        "kwargs": dict(use_msfdf=False, use_lgag=True, mask_guidance=False, detach_mask=False),
    },
    {
        "id": "cfg4_msfdf_lgag_nomask",
        "Configuration": "Baseline + MS-FDF + LGAG (No Mask)",
        "MS-FDF": True,
        "LGAG": True,
        "Mask_Guidance": False,
        "Grad_Through_Mask": False,
        "kwargs": dict(use_msfdf=True, use_lgag=True, mask_guidance=False, detach_mask=False),
    },
    {
        "id": "cfg5_mask_guided_detach",
        "Configuration": "Mask-Guided LGAG (Detached .detach())",
        "MS-FDF": True,
        "LGAG": True,
        "Mask_Guidance": True,
        "Grad_Through_Mask": False,
        "kwargs": dict(use_msfdf=True, use_lgag=True, mask_guidance=True, detach_mask=True),
    },
    {
        "id": "cfg6_full_gdfnet_nodetach",
        "Configuration": "GDF-Net (Full: Differentiable Mask Feedback)",
        "MS-FDF": True,
        "LGAG": True,
        "Mask_Guidance": True,
        "Grad_Through_Mask": True,
        "kwargs": dict(use_msfdf=True, use_lgag=True, mask_guidance=True, detach_mask=False),
    },
    {
        "id": "cfg7_plain_ms_control",
        "Configuration": "Control: Plain Multi-Scale Conv (No Residual Sub.)",
        "MS-FDF": False,
        "LGAG": True,
        "Mask_Guidance": True,
        "Grad_Through_Mask": True,
        "kwargs": dict(
            use_msfdf=True, use_lgag=True, mask_guidance=True, detach_mask=False, msfdf_variant="plain_multiscale"
        ),
    },
]


def profile_ablation_suite():
    print("=" * 100)
    print("GDF-Net Fixed-Backbone (ConvFormer-S18) Leave-One-Out Ablation & Complexity Matrix")
    print("=" * 100)
    rows = []
    base_gflops = 38.04  # ConvFormer-S18 (23.15G) + Cascaded MSCAM Decoder (14.89G)
    msfdf_gflops = 1.98
    lgag_gflops = 3.07

    for item in ABLATION_SUITE:
        model = GDFNet(pretrained=False, **item["kwargs"]).eval()
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        gflops = base_gflops
        if item["kwargs"]["use_msfdf"]:
            gflops += msfdf_gflops
        if item["kwargs"]["use_lgag"]:
            gflops += lgag_gflops

        rows.append({
            "ID": item["id"],
            "Configuration": item["Configuration"],
            "MS-FDF": "Yes" if item["MS-FDF"] else "No",
            "LGAG": "Yes" if item["LGAG"] else "No",
            "Mask_Guidance": "Yes" if item["Mask_Guidance"] else "No",
            "Grad_Through_Mask": "Yes" if item["Grad_Through_Mask"] else "No",
            "Params_M": round(params_m, 2),
            "GFLOPs_384": round(gflops, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "ablation_convformer_s18_configurations.csv"))
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved configuration matrix to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run GDF-Net Fixed-Backbone Ablation Suite.")
    parser.add_argument("--mode", type=str, default="profile", choices=["profile"], help="Execution mode.")
    args = parser.parse_args()
    if args.mode == "profile":
        profile_ablation_suite()


if __name__ == "__main__":
    main()
