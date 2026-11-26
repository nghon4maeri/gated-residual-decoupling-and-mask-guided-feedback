"""
Pretrained Checkpoint Registry, Downloader, and Verifier for GDF-Net.

  - Lists the released GDF-Net checkpoints (.pth, 29,503,181 parameters, 534 state_dict keys)
    for ISIC-2018, CVC-ClinicDB, Kvasir-SEG, and DSB-2018.
  - Downloads checkpoints from the corresponding Kaggle kernel outputs (requires the Kaggle CLI).
  - Verifies local checkpoints by strict PyTorch state_dict loading into GDFNet and prints
    their SHA-256 digests.

Usage:
    python scripts/download_weights.py --out-dir ./checkpoints --download-kaggle --verify
"""

import argparse
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CHECKPOINT_REGISTRY = [
    {
        "filename": "GDFNet_ISIC2018_Base_M1_ConvFormerS18_300ep_best.pth",
        "cohort": "ISIC-2018 (384x384)",
        "variant": "GDF-Net (Base, M1 Teacher)",
        "test_dice": 0.9017,
        "test_dice_bialt": 0.9105,
        "params": 29503181,
        "size_mb": 112.83,
        "kaggle_kernel": "namnguynnnn/gdfnetv4-convformer-s18-300ep",
    },
    {
        "filename": "GDFNet_ISIC2018_Teacher_M2_LLRD_MultiScale_best.pth",
        "cohort": "ISIC-2018 (384x384)",
        "variant": "GDF-Net (M2 Multi-Scale LLRD Teacher)",
        "test_dice": 0.9032,
        "test_dice_bialt": 0.9108,
        "params": 29503181,
        "size_mb": 112.83,
        "kaggle_kernel": "namnguynnnn/gdfnetv4-s18-llrd-multiscale-isic2018",
    },
    {
        "filename": "GDFNet_ISIC2018_Soup_M1_M4_WiSE_best.pth",
        "cohort": "ISIC-2018 (384x384)",
        "variant": "GDF-Net (Soup 0.70*M1 + 0.30*M4 / 8V Ensemble)",
        "test_dice": 0.9126,
        "test_dice_bialt": 0.9146,
        "params": 29503181,
        "size_mb": 112.79,
        "kaggle_kernel": "namnguynnnn/gdfnetv4-s18-asym-tversky-isic2018",
    },
    {
        "filename": "GDFNet_ISIC2018_RealTime_Distilled_Student_best.pth",
        "cohort": "ISIC-2018 (384x384)",
        "variant": "GDF-Net (Real-Time Distilled Student, 31.3 FPS)",
        "test_dice": 0.9127,
        "test_dice_bialt": 0.9127,
        "params": 29503181,
        "size_mb": 112.83,
        "kaggle_kernel": "namnguynnnn/gdfnetv4-s18-pa2-plus-distill-isic2018",
    },
    {
        "filename": "GDFNet_CVC_ClinicDB_ConvFormerS18_best.pth",
        "cohort": "CVC-ClinicDB (384x384)",
        "variant": "GDF-Net (Base / Soup / Ensemble)",
        "test_dice": 0.9249,
        "test_dice_bialt": 0.9492,
        "params": 29503181,
        "size_mb": 112.84,
        "kaggle_kernel": "namnguynnnn/gdfnetv4-canonical-s18-polyp-nuclei",
    },
    {
        "filename": "GDFNet_Kvasir_SEG_ConvFormerS18_best.pth",
        "cohort": "Kvasir-SEG (384x384)",
        "variant": "GDF-Net (Base / Real-Time Distilled)",
        "test_dice": 0.9291,
        "test_dice_bialt": 0.9409,
        "params": 29503181,
        "size_mb": 112.84,
        "kaggle_kernel": "namnguynnnn/gdfnetv4-canonical-s18-polyp-nuclei",
    },
    {
        "filename": "GDFNet_DSB2018_ConvFormerS18_best.pth",
        "cohort": "DSB-2018 (256x256)",
        "variant": "GDF-Net (Base / Real-Time Distilled)",
        "test_dice": 0.8955,
        "test_dice_bialt": 0.8992,
        "params": 29503181,
        "size_mb": 112.84,
        "kaggle_kernel": "namnguynnnn/gdfnetv4-canonical-s18-polyp-nuclei",
    },
]


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def print_registry():
    print("=" * 105)
    print("GDF-Net Pretrained Model Zoo Registry (MIDL 2027 Reproducibility Checklist)")
    print("=" * 105)
    print(f"{'Checkpoint Filename':52s} | {'Cohort':22s} | {'Dice':6s} | {'+Bi-ALT':7s} | {'Size':7s}")
    print("-" * 105)
    for item in CHECKPOINT_REGISTRY:
        print(
            f"{item['filename']:52s} | {item['cohort']:22s} | "
            f"{item['test_dice']:.4f} | {item['test_dice_bialt']:.4f}  | {item['size_mb']:5.2f}MB"
        )
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="Download and verify GDF-Net pretrained weights.")
    parser.add_argument("--out-dir", type=str, default="./checkpoints", help="Target directory for checkpoints.")
    parser.add_argument("--download-kaggle", action="store_true", help="Pull checkpoints via Kaggle CLI.")
    parser.add_argument(
        "--download-ablation",
        action="store_true",
        help="Pull the 300-epoch ISIC-2018 ablation checkpoints from trnminh0108/gdfnet-isic2018-ablation-checkpoints.",
    )
    parser.add_argument("--verify", action="store_true", help="Verify local checkpoints against GDFNet(strict=True).")
    args = parser.parse_args()

    print_registry()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.download_kaggle:
        kernels = sorted({item["kaggle_kernel"] for item in CHECKPOINT_REGISTRY})
        for k in kernels:
            print(f"[*] Downloading kernel output: {k} -> {args.out_dir}")
            subprocess.run(["kaggle", "kernels", "output", k, "-p", args.out_dir], check=False)

    if args.download_ablation:
        abl_dir = os.path.join(args.out_dir, "ablation")
        os.makedirs(abl_dir, exist_ok=True)
        ds = "trnminh0108/gdfnet-isic2018-ablation-checkpoints"
        print(f"[*] Downloading ablation dataset: {ds} -> {abl_dir}")
        subprocess.run(["kaggle", "datasets", "download", "-d", ds, "-p", abl_dir, "--unzip"], check=False)

    if args.verify:
        import torch
        from gdfnet import GDFNet

        model = GDFNet(pretrained=False).eval()
        for f in sorted(os.listdir(args.out_dir)):
            if f.endswith(".pth"):
                fp = os.path.join(args.out_dir, f)
                sha = compute_sha256(fp)[:12]
                sd = torch.load(fp, map_location="cpu", weights_only=False)
                if "model_state_dict" in sd:
                    sd = sd["model_state_dict"]
                elif "state_dict" in sd:
                    sd = sd["state_dict"]
                model.load_state_dict(sd, strict=True)
                print(f"[+] Verified {f}: keys={len(sd)}, sha256={sha}...")


if __name__ == "__main__":
    main()
