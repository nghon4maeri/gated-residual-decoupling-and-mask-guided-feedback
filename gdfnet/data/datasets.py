"""
Dataset Loaders and Partition Builders for All Benchmarks (Section 4 & Appendix E).

Supported Benchmarks:
  1. ISIC-2018 (Dermoscopy, Task 1): 2,594 train / 100 official val / 1,000 official test (384x384)
  2. CVC-ClinicDB (Colonoscopy):     612 frames (490 train / 61 val / 61 test, 80/10/10 split, 384x384)
  3. Kvasir-SEG (Endoscopy):         1,000 polyp scans (800 train / 100 val / 100 test, 80/10/10 split, 384x384)
  4. DSB-2018 (Cellular Microscopy): 670 scans (536 train / 67 val / 67 test, 256x256, binary union Y = U_k I_k)
  5. EM-Dataset (Electron Microscopy, Appendix E): 165 slices (153 train / 6 val / 6 test, 512x512)
  6. CHASE-DB1 (Retinal Vessels, Appendix E):      28 scans (20 train / 4 val / 4 test, 768x768)
  7. DRIVE (Retinal Vessels, Appendix E):          20 scans (12 train / 2 val / 6 test, 576x576)
"""

import os
import glob
import random
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .augmentations import (
    RandomVignettingTransform,
    SyntheticHairTransform,
    SyntheticRulerMarkTransform,
    SpecularGlareTransform,
    ImageOnlyCutoutTransform,
    build_train_transforms,
    build_eval_transforms,
)


class MedicalSegmentationDataset(Dataset):
    """
    Unified PyTorch Dataset for 2D Medical Image Segmentation.
    """

    def __init__(
        self,
        img_paths: List[str],
        mask_paths: List[str],
        img_size: int = 384,
        is_train: bool = False,
        use_derm_artifacts: bool = False,
    ):
        self.img_paths = list(img_paths)
        self.mask_paths = list(mask_paths)
        self.img_size = img_size
        self.is_train = is_train
        self.use_derm_artifacts = use_derm_artifacts and is_train

        if self.use_derm_artifacts:
            self.artifact_transforms = [
                RandomVignettingTransform(p=0.25),
                SyntheticHairTransform(p=0.30),
                SyntheticRulerMarkTransform(p=0.15),
                SpecularGlareTransform(p=0.15),
                ImageOnlyCutoutTransform(p=0.25),
            ]
        else:
            self.artifact_transforms = []

        self.transform = (
            build_train_transforms(img_size) if is_train else build_eval_transforms(img_size)
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        im_p = self.img_paths[idx]
        mk_p = self.mask_paths[idx]

        img = cv2.imread(im_p, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(im_p).convert("RGB"))

        msk = cv2.imread(mk_p, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = np.array(Image.open(mk_p).convert("L"))

        if self.use_derm_artifacts:
            for tf in self.artifact_transforms:
                img = tf(img)

        msk_bin = (msk > 127).astype(np.float32)
        aug = self.transform(image=img, mask=msk_bin)
        img_t = torch.from_numpy(aug["image"].transpose(2, 0, 1)).float()
        msk_t = torch.from_numpy((aug["mask"] > 0.5).astype(np.float32)).unsqueeze(0)
        return img_t, msk_t, os.path.basename(im_p)


def pair_isic2018_split(img_dir: str, gt_dir: str) -> Tuple[List[str], List[str]]:
    """Pairs ISIC-2018 RGB scans (*.jpg) with expert ground-truth masks (*_segmentation.png)."""
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    msks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

    def _isic_id(path: str) -> str:
        base = os.path.basename(path).replace("_segmentation", "")
        return os.path.splitext(base)[0]

    id_map = {_isic_id(m): m for m in msks}
    paired_x, paired_y = [], []
    for im in imgs:
        imid = _isic_id(im)
        if imid in id_map:
            paired_x.append(im)
            paired_y.append(id_map[imid])
    return paired_x, paired_y


def split_80_10_10(
    img_paths: List[str], mask_paths: List[str], seed: int = 42
) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    """
    Deterministic 80% / 10% / 10% train/val/test partition used for CVC-ClinicDB (490/61/61),
    Kvasir-SEG (800/100/100), and DSB-2018 (536/67/67).
    """
    combined = list(zip(sorted(img_paths), sorted(mask_paths)))
    rng = random.Random(seed)
    rng.shuffle(combined)
    xs, ys = zip(*combined)
    xs, ys = list(xs), list(ys)

    n_total = len(xs)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)

    train_split = (xs[:n_train], ys[:n_train])
    val_split = (xs[n_train : n_train + n_val], ys[n_train : n_train + n_val])
    test_split = (xs[n_train + n_val :], ys[n_train + n_val :])
    return train_split, val_split, test_split


def merge_dsb2018_instance_masks(stage1_train_dir: str, output_mask_dir: str) -> Tuple[List[str], List[str]]:
    """
    Merges per-nucleus instance masks {I_k} via logical union (Y = U_k I_k, Section 4)
    for DSB-2018 binary foreground separation.
    """
    os.makedirs(output_mask_dir, exist_ok=True)
    sample_dirs = sorted(
        [d for d in glob.glob(os.path.join(stage1_train_dir, "*")) if os.path.isdir(d)]
    )
    img_paths, mask_paths = [], []

    for s_dir in sample_dirs:
        im_files = glob.glob(os.path.join(s_dir, "images", "*.png"))
        mk_files = glob.glob(os.path.join(s_dir, "masks", "*.png"))
        if not im_files or not mk_files:
            continue
        sid = os.path.basename(s_dir)
        fused_path = os.path.join(output_mask_dir, f"{sid}.png")
        if not os.path.exists(fused_path):
            first = cv2.imread(mk_files[0], cv2.IMREAD_GRAYSCALE)
            union_mask = np.zeros_like(first, dtype=np.uint8)
            for mf in mk_files:
                m_arr = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
                if m_arr is not None:
                    union_mask = np.maximum(union_mask, m_arr)
            cv2.imwrite(fused_path, union_mask)
        img_paths.append(im_files[0])
        mask_paths.append(fused_path)

    return img_paths, mask_paths
