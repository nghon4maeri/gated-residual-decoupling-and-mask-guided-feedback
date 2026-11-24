"""
300-Epoch Baseline Training Script for GDF-Net (Base: ConvFormer-S18, 29.50M Params) — Section 4.

Trains GDF-Net with:
  - ImageNet-22K/1K pretrained ConvFormer-S18 backbone ('convformer_s18.sail_in22k_ft_in1k_384')
  - Multi-Scale Feature Decoupling Filters (MS-FDF, Eqs. 1-2)
  - Differentiable Mask-Guided Local-Global Attention Gates (Mask-Guided LGAG, Eq. 3)
  - Cascaded MSCAM Decoder with 4 Deep-Supervision Stage Heads (SH)
  - AdamW (encoder lr=1e-4, decoder/skip modules lr=3e-4, weight_decay=2e-4)
  - CosineAnnealingWarmRestarts (T_0=50, T_mult=2, eta_min=1e-6) -> 50 + 100 + 150 = 300 epochs
  - DropPath 0.2, ModelEMA (decay=0.999), AMP FP16, grad clip 1.0, 15-case clinical augmentation suite
  - checkpoint = best of RAW / EMA weights by official validation Macro Dice
These are the settings of the script that produced GDF-Net (Base) / M_1 (0.9017 test Dice).

Fixed-backbone leave-one-out ablation (Table 3, Panel A) uses the same protocol:
  python scripts/train_base.py ... --ablation cfg2_msfdf_only --out-dir ./checkpoints/cfg2

Usage:
  python scripts/train_base.py --train-img ./data/ISIC-2018/ISIC2018_Task1-2_Training_Input \
                               --train-gt  ./data/ISIC-2018/ISIC2018_Task1_Training_GroundTruth \
                               --val-img   ./data/ISIC-2018/ISIC2018_Task1-2_Validation_Input \
                               --val-gt    ./data/ISIC-2018/ISIC2018_Task1_Validation_GroundTruth \
                               --epochs 300 --batch-size 16 --img-size 384 --out-dir ./checkpoints
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdfnet import (
    GDFNet,
    GDFNetCompositeLoss,
    ModelEMA,
)
from gdfnet.data import MedicalSegmentationDataset, pair_isic2018_split
from run_ablation_suite import ABLATION_SUITE

ABLATION_KWARGS = {item["id"]: item["kwargs"] for item in ABLATION_SUITE}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_validation_dice(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dices = []
    use_amp = device.type == "cuda"
    for imgs, msks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
        preds = (probs >= 0.50).astype(np.uint8)
        gts = (msks.numpy() > 0.5).astype(np.uint8)
        for p, g in zip(preds, gts):
            inter = float(np.sum((p == 1) & (g == 1)))
            union = float(np.sum(p == 1) + np.sum(g == 1))
            dices.append(1.0 if union == 0 else (2.0 * inter + 1e-7) / (union + 1e-7))
    return float(np.mean(dices))


def main():
    parser = argparse.ArgumentParser(description="Train GDF-Net (Base: ConvFormer-S18).")
    parser.add_argument("--train-img", type=str, required=True)
    parser.add_argument("--train-gt", type=str, required=True)
    parser.add_argument("--val-img", type=str, required=True)
    parser.add_argument("--val-gt", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="convformer_s18.sail_in22k_ft_in1k_384")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--drop-path", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ablation", type=str, default="cfg6_full_gdfnet_nodetach", choices=sorted(ABLATION_KWARGS),
                        help="leave-one-out configuration; the default is the full GDF-Net")
    parser.add_argument("--out-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_x, tr_y = pair_isic2018_split(args.train_img, args.train_gt)
    va_x, va_y = pair_isic2018_split(args.val_img, args.val_gt)
    print(f"[+] Training Set: {len(tr_x)} pairs | Validation Set: {len(va_x)} pairs | Device: {device}")

    train_loader = DataLoader(
        MedicalSegmentationDataset(tr_x, tr_y, img_size=args.img_size, is_train=True, use_derm_artifacts=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        MedicalSegmentationDataset(va_x, va_y, img_size=args.img_size, is_train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = GDFNet(backbone_name=args.backbone, pretrained=True, drop_path_rate=args.drop_path,
                   **ABLATION_KWARGS[args.ablation]).to(device)
    print(f"[+] Configuration: {args.ablation} {ABLATION_KWARGS[args.ablation]}")
    ema = ModelEMA(model, decay=args.ema_decay)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] GDF-Net initialized ({total_params:,} parameters, {len(model.state_dict())} state_dict keys)")

    criterion = GDFNetCompositeLoss(macro_dice_weight=0.5)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": args.lr_backbone},
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for imgs, msks, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs, return_deep_supervision=True)
                loss = criterion(outputs, msks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)
        val_raw = evaluate_validation_dice(model, val_loader, device)
        val_ema = evaluate_validation_dice(ema.ema, val_loader, device)
        cur_best = max(val_raw, val_ema)

        tag = ""
        if cur_best > best_val_dice:
            best_val_dice = cur_best
            best_sd = ema.ema.state_dict() if val_ema >= val_raw else model.state_dict()
            torch.save(best_sd, os.path.join(args.out_dir, "GDFNet_ConvFormerS18_best.pth"))
            tag = " [NEW BEST]"

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} [{time.time() - t0:.1f}s] | "
            f"Loss: {avg_loss:.4f} | Val Raw: {val_raw:.4f} | Val EMA: {val_ema:.4f} | Best: {best_val_dice:.4f}{tag}"
        )
        history.append({"epoch": epoch, "loss": avg_loss, "val_raw_dice": val_raw, "val_ema_dice": val_ema, "best_val_dice": best_val_dice})
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_history.csv"), index=False)


if __name__ == "__main__":
    main()
