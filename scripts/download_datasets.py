"""
Automated Dataset Downloader & Preprocessor for GDF-Net Benchmarks (Section 4 & Appendix E).

Supports downloading and preparing:
  1. ISIC-2018 (Task 1 Lesion Boundary Segmentation): 2,594 train / 100 val / 1,000 test (384x384)
  2. CVC-ClinicDB (Colonoscopy Polyp Frames):         612 scans (490 train / 61 val / 61 test, 384x384)
  3. Kvasir-SEG (Gastrointestinal Endoscopy Polyps):  1,000 scans (800 train / 100 val / 100 test, 384x384)
  4. DSB-2018 (2018 Data Science Bowl Nuclei):        670 scans (536 train / 67 val / 67 test, 256x256, binary union Y = U_k I_k)
  5. Extended cohorts (Appendix E): EM-Dataset (512x512), CHASE-DB1 (768x768), DRIVE (576x576)

Usage:
  python scripts/download_datasets.py --data-root ./data --dataset all
  python scripts/download_datasets.py --data-root ./data --dataset isic2018
"""

import os
import sys
import glob
import zipfile
import argparse
import urllib.request
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gdfnet.data import merge_dsb2018_instance_masks


DATASET_SPECIFICATIONS = {
    "isic2018": {
        "name": "ISIC-2018 (Task 1: Skin Lesion Boundary Segmentation)",
        "modality": "Dermoscopy (RGB)",
        "resolution": "384 x 384",
        "splits": "2,594 Train / 100 Official Validation / 1,000 Official Test (Total: 3,694)",
        "kaggle_train": "tschandl/isic2018-challenge-task1-data-segmentation",
        "kaggle_test_gt": "mfahimhossen/isic2018-test-groundtruth",
        "s3_val_input": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip",
        "s3_val_gt": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
    },
    "cvc_clinicdb": {
        "name": "CVC-ClinicDB (Colonoscopy Polyp Segmentation)",
        "modality": "Colonoscopy Video Frames (RGB)",
        "resolution": "384 x 384",
        "splits": "490 Train / 61 Validation / 61 Test (80%/10%/10%, Total: 612 frames across 29 sequences)",
        "kaggle_slug": "balraj98/cvcclinicdb",
    },
    "kvasir_seg": {
        "name": "Kvasir-SEG (Gastrointestinal Polyp Segmentation)",
        "modality": "Gastrointestinal Endoscopy (RGB)",
        "resolution": "384 x 384",
        "splits": "800 Train / 100 Validation / 100 Test (80%/10%/10%, Total: 1,000 scans)",
        "kaggle_slug": "debeshjha1/kvasirseg",
        "official_url": "https://datasets.simula.no/downloads/kvasir-seg.zip",
    },
    "dsb2018": {
        "name": "2018 Data Science Bowl (DSB-2018 Nuclei Foreground Segmentation)",
        "modality": "Brightfield, Histopathology & Fluorescence Microscopy",
        "resolution": "256 x 256",
        "splits": "536 Train / 67 Validation / 67 Test (80%/10%/10%, Total: 670 scans; instance union Y = U_k I_k)",
        "kaggle_competition": "data-science-bowl-2018",
        "kaggle_mirror": "gangadhar/data-science-bowl-2018",
    },
    "em_dataset": {
        "name": "ISBI-2012 EM-Dataset (Electron Microscopy Neuronal Membranes, Appendix E)",
        "modality": "Serial-Section Transmission Electron Microscopy (ssTEM)",
        "resolution": "512 x 512",
        "splits": "153 Train / 6 Validation / 6 Test (165 slices from 30x512x512 volume)",
        "kaggle_slug": "namnguynnnn/isbi2012-em-dataset",
    },
    "chasedb1": {
        "name": "CHASE-DB1 (Retinal Vessel Segmentation, Appendix E)",
        "modality": "Retinal Fundus Photography",
        "resolution": "768 x 768",
        "splits": "20 Train / 4 Validation / 4 Test (Total: 28 scans)",
        "kaggle_slug": "khoongweihao/chasedb1",
    },
    "drive": {
        "name": "DRIVE (Digital Retinal Images for Vessel Extraction, Appendix E)",
        "modality": "Retinal Fundus Photography",
        "resolution": "576 x 576",
        "splits": "12 Train / 2 Validation / 6 Test (Total: 20 scans)",
        "kaggle_slug": "andrewmvd/drive-digital-retinal-images-for-vessel-extraction",
    },
}


def _unzip_all(target_dir: str) -> None:
    for z_path in glob.glob(os.path.join(target_dir, "*.zip")):
        print(f"  [*] Extracting {os.path.basename(z_path)} -> {target_dir}...")
        with zipfile.ZipFile(z_path, "r") as zf:
            zf.extractall(target_dir)
        os.remove(z_path)


def download_isic2018(data_root: str) -> None:
    out_dir = os.path.join(data_root, "ISIC-2018")
    os.makedirs(out_dir, exist_ok=True)
    spec = DATASET_SPECIFICATIONS["isic2018"]
    print(f"\n[+] Downloading {spec['name']} -> {out_dir}")

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", spec["kaggle_train"], "-p", out_dir],
        check=True,
    )
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", spec["kaggle_test_gt"], "-p", out_dir],
        check=True,
    )
    val_in_zip = os.path.join(out_dir, "ISIC2018_Task1-2_Validation_Input.zip")
    val_gt_zip = os.path.join(out_dir, "ISIC2018_Task1_Validation_GroundTruth.zip")
    urllib.request.urlretrieve(spec["s3_val_input"], val_in_zip)
    urllib.request.urlretrieve(spec["s3_val_gt"], val_gt_zip)
    _unzip_all(out_dir)
    print("  [+] ISIC-2018 ready!")


def download_kaggle_dataset(slug: str, out_dir: str, title: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[+] Downloading {title} ({slug}) -> {out_dir}")
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", out_dir], check=True)
    _unzip_all(out_dir)
    print(f"  [+] {title} ready!")


def download_dsb2018(data_root: str) -> None:
    out_dir = os.path.join(data_root, "DSB-2018")
    os.makedirs(out_dir, exist_ok=True)
    spec = DATASET_SPECIFICATIONS["dsb2018"]
    print(f"\n[+] Downloading {spec['name']} -> {out_dir}")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", spec["kaggle_competition"], "-f", "stage1_train.zip", "-p", out_dir],
            check=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", spec["kaggle_mirror"], "-p", out_dir],
            check=True,
        )
    stage1_dir = os.path.join(out_dir, "stage1_train")
    os.makedirs(stage1_dir, exist_ok=True)
    for z_path in glob.glob(os.path.join(out_dir, "*.zip")):
        with zipfile.ZipFile(z_path, "r") as zf:
            zf.extractall(stage1_dir)
        os.remove(z_path)

    fused_dir = os.path.join(out_dir, "dsb2018_masks")
    imgs, msks = merge_dsb2018_instance_masks(stage1_dir, fused_dir)
    print(f"  [+] Merged {len(imgs)} DSB-2018 instance masks via logical union Y = U_k I_k!")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare GDF-Net datasets.")
    parser.add_argument("--data-root", type=str, default="./data", help="Root folder for datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "isic2018", "cvc_clinicdb", "kvasir_seg", "dsb2018", "em_dataset", "chasedb1", "drive", "info"],
        help="Dataset to download (or 'info' to print dataset specifications).",
    )
    args = parser.parse_args()

    if args.dataset == "info":
        for k, spec in DATASET_SPECIFICATIONS.items():
            print(f"[{k}] {spec['name']}")
            print(f"    Modality   : {spec['modality']}")
            print(f"    Resolution : {spec['resolution']}")
            print(f"    Splits     : {spec['splits']}\n")
        return

    targets = (
        ["isic2018", "cvc_clinicdb", "kvasir_seg", "dsb2018", "em_dataset", "chasedb1", "drive"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for ds in targets:
        if ds == "isic2018":
            download_isic2018(args.data_root)
        elif ds == "dsb2018":
            download_dsb2018(args.data_root)
        else:
            spec = DATASET_SPECIFICATIONS[ds]
            folder_map = {
                "cvc_clinicdb": "CVC-ClinicDB",
                "kvasir_seg": "Kvasir-SEG",
                "em_dataset": "EM-Dataset",
                "chasedb1": "CHASE_DB1",
                "drive": "DRIVE",
            }
            download_kaggle_dataset(
                spec["kaggle_slug"],
                os.path.join(args.data_root, folder_map[ds]),
                spec["name"],
            )


if __name__ == "__main__":
    main()
