"""
Weight-Space Model Soup, 8-View GeM MS-TTA, Online Ensemble Knowledge Distillation,
and Scale-Anchored Bi-ALT Calibration Pipeline (Section 3.3 & Section 3.4).

Executes:
  1. Weight-Space Model Soup: theta_Soup = 0.70 * theta_Base (M_1) + 0.30 * theta_Asym (M_4)
  2. GDF-Net (Soup, SV) & GDF-Net (Ensemble, 8V GeM MS-TTA at p=1.75) with Scale-Anchored Bi-ALT
  3. Online Ensemble-to-Single Knowledge Distillation (Equation 5, tau=2.0, GeM teacher fusion p=1.5)
     to produce GDF-Net (Real-Time, 31.3 FPS @ 384x384)
"""

import os
import sys
import copy
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gdfnet import (
    GDFNet,
    FeedbackGDFNet,
    EnsembleDistillationLoss,
    ModelEMA,
    ScaleAnchoredBiALT,
    calibrate_bi_alt_on_validation,
    build_weight_soup,
    build_llrd_param_groups,
    extract_multiscale_views,
    fuse_gem_views,
    evaluate_cohort_predictions,
)
from gdfnet.data import MedicalSegmentationDataset, pair_isic2018_split


def load_clean_state_dict(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state", ckpt.get("model", ckpt.get("state_dict", ckpt)))
    else:
        sd = ckpt
    return {k.replace("module.", ""): v for k, v in sd.items()}


def main():
    parser = argparse.ArgumentParser(description="GDF-Net Weight Soup, GeM MS-TTA & Online Distillation.")
    parser.add_argument("--train-img", type=str, required=True)
    parser.add_argument("--train-gt", type=str, required=True)
    parser.add_argument("--val-img", type=str, required=True)
    parser.add_argument("--val-gt", type=str, required=True)
    parser.add_argument("--test-img", type=str, required=True)
    parser.add_argument("--test-gt", type=str, required=True)
    parser.add_argument("--ckpt-m1", type=str, required=True, help="Canonical baseline M_1 checkpoint.")
    parser.add_argument("--ckpt-m4", type=str, required=True, help="Recall-oriented AsymTversky M_4 checkpoint.")
    parser.add_argument("--ckpt-m2", type=str, default=None, help="Optional LLRD-MS M_2 teacher checkpoint.")
    parser.add_argument("--ckpt-m3", type=str, default=None, help="Optional FeedbackGDFNet M_3 teacher checkpoint.")
    parser.add_argument("--distill-epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=4.5e-5)
    parser.add_argument("--out-dir", type=str, default="./results_distill")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_x, tr_y = pair_isic2018_split(args.train_img, args.train_gt)
    va_x, va_y = pair_isic2018_split(args.val_img, args.val_gt)
    te_x, te_y = pair_isic2018_split(args.test_img, args.test_gt)

    train_loader = DataLoader(
        MedicalSegmentationDataset(tr_x, tr_y, img_size=384, is_train=True),
        batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        MedicalSegmentationDataset(va_x, va_y, img_size=384, is_train=False),
        batch_size=16, shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        MedicalSegmentationDataset(te_x, te_y, img_size=384, is_train=False),
        batch_size=16, shuffle=False, num_workers=2, pin_memory=True,
    )

    sd_m1 = load_clean_state_dict(args.ckpt_m1)
    sd_m4 = load_clean_state_dict(args.ckpt_m4)

    # 1. Construct Weight-Space Model Soup: theta_Soup = 0.70 * theta_Base + 0.30 * theta_Asym
    soup_sd = build_weight_soup([sd_m1, sd_m4], weights=(0.70, 0.30))
    model = GDFNet(pretrained=False, drop_path_rate=0.15).to(device)
    model.load_state_dict(soup_sd, strict=True)

    # 2. Evaluate GDF-Net (Soup, SV) & GDF-Net (Ensemble, 8V GeM MS-TTA)
    v384_va, v448_va, v352_va, val_gts, _ = extract_multiscale_views(model, val_loader, device)
    v384_te, v448_te, v352_te, test_gts, test_names = extract_multiscale_views(model, test_loader, device)

    val_ms8 = fuse_gem_views(v384_va, v448_va, v352_va, mode="ms8_gem", gem_p=1.75)
    alt_cfg = calibrate_bi_alt_on_validation(val_ms8, val_gts)
    bi_alt_ens = ScaleAnchoredBiALT(
        t_base=alt_cfg["t_base"], dt_high=alt_cfg["dt_high"], dt_mid=alt_cfg["dt_mid"], dt_crisp=alt_cfg["dt_crisp"]
    )
    te_ms8 = fuse_gem_views(v384_te, v448_te, v352_te, mode="ms8_gem", gem_p=1.75)
    ens_summary, _ = evaluate_cohort_predictions(
        te_ms8, test_gts, test_names, thresholds=bi_alt_ens.get_batch_thresholds(te_ms8)
    )
    print(f"[+] GDF-Net (Ensemble, 8V GeM MS-TTA) Test Summary: {ens_summary}")

    # 3. Online Ensemble-to-Single Knowledge Distillation (Equation 5)
    t_m1 = GDFNet(pretrained=False, drop_path_rate=0.0).to(device).eval()
    t_m1.load_state_dict(sd_m1, strict=True)
    t_m4 = GDFNet(pretrained=False, drop_path_rate=0.0).to(device).eval()
    t_m4.load_state_dict(sd_m4, strict=True)

    t_m2 = None
    if args.ckpt_m2 and os.path.exists(args.ckpt_m2):
        t_m2 = GDFNet(pretrained=False, drop_path_rate=0.0).to(device).eval()
        t_m2.load_state_dict(load_clean_state_dict(args.ckpt_m2), strict=True)

    t_m3 = None
    if args.ckpt_m3 and os.path.exists(args.ckpt_m3):
        t_m3 = FeedbackGDFNet(pretrained=False, drop_path_rate=0.0).to(device).eval()
        t_m3.load_state_dict(load_clean_state_dict(args.ckpt_m3), strict=True)

    ema = ModelEMA(model, decay=0.996)
    criterion = EnsembleDistillationLoss(kd_weight=0.55, temp=2.0)
    optimizer = optim.AdamW(build_llrd_param_groups(model, base_lr=args.lr, weight_decay=1e-4, llrd_decay=0.65))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.distill_epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_kd_val = -1.0
    best_kd_sd = copy.deepcopy(soup_sd)

    for ep in range(1, args.distill_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for imgs, msks, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)

            with torch.no_grad(), autocast("cuda", enabled=(device.type == "cuda")):
                z1 = t_m1(imgs)
                z4 = t_m4(imgs)
                p1 = torch.clamp(torch.sigmoid(z1), 1e-6, 1.0 - 1e-6)
                p4 = torch.clamp(torch.sigmoid(z4), 1e-6, 1.0 - 1e-6)
                gem_p = 1.5
                if t_m2 is not None and t_m3 is not None:
                    z2, z3 = t_m2(imgs), t_m3(imgs)
                    p2 = torch.clamp(torch.sigmoid(z2), 1e-6, 1.0 - 1e-6)
                    p3 = torch.clamp(torch.sigmoid(z3), 1e-6, 1.0 - 1e-6)
                    q_t = torch.pow(
                        0.38 * p1.pow(gem_p) + 0.14 * p2.pow(gem_p) + 0.36 * p3.pow(gem_p) + 0.12 * p4.pow(gem_p),
                        1.0 / gem_p,
                    )
                    z_t = 0.38 * z1 + 0.14 * z2 + 0.36 * z3 + 0.12 * z4
                else:
                    q_t = torch.pow(0.70 * p1.pow(gem_p) + 0.30 * p4.pow(gem_p), 1.0 / gem_p)
                    z_t = 0.70 * z1 + 0.30 * z4

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs, return_deep_supervision=True)
                loss = criterion(outputs, msks, q_t, z_t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            total_loss += loss.item()

        scheduler.step()
        v384_e, v448_e, v352_e, _, _ = extract_multiscale_views(ema.ema, val_loader, device)
        val_sv_e = fuse_gem_views(v384_e, v448_e, v352_e, mode="sv")
        alt_e = calibrate_bi_alt_on_validation(val_sv_e, val_gts)
        if alt_e["val_alt_dice"] > best_kd_val:
            best_kd_val = alt_e["val_alt_dice"]
            best_kd_sd = copy.deepcopy(ema.ema.state_dict())
            torch.save(best_kd_sd, os.path.join(args.out_dir, "GDFNet_RealTime_Distilled_best.pth"))
        print(
            f"KD Epoch {ep:02d}/{args.distill_epochs:02d} [{time.time() - t0:.1f}s] | "
            f"Loss: {total_loss / max(len(train_loader), 1):.4f} | Val Bi-ALT Dice: {alt_e['val_alt_dice']:.4f}"
        )


if __name__ == "__main__":
    main()
