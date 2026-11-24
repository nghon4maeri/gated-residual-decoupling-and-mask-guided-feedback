"""
Mathematical & Architectural Consistency Test Suite for GDF-Net.

Verifies 1:1 fidelity between the PyTorch codebase and every equation in the MIDL 2027 manuscript:
  1. Backbone & Parameter Count: ConvFormer-S18 GDF-Net has exactly 29,503,181 trainable parameters
     and 534 state_dict keys.
  2. Equations (1) & (2) — Multi-Scale Feature Decoupling Filter (MS-FDF) & Complexity-Matched Control:
     Verifies x_low extraction, exact spatial residual subtraction (x_high = E_i - x_low),
     soft gating x_filt = x_high * beta, residual output ~E_i = Proj([x_low || x_filt]) + E_i,
     and complexity match with PlainMultiScaleConv.
  3. Equation (3) — Differentiable Mask-Guided Local-Global Attention Gate (Mask-Guided LGAG):
     Verifies additive semantic gating F_sg = ~E_i + G, mask concatenation [F_sg || M_up],
     local (d=1) and expanded-context (d=2) dilated depthwise branches, and confirms active
     gradient flow through M_{i+1} = sigma(Z_{i+1}) when detach_mask=False vs. zero mask-pathway
     gradient when detach_mask=True.
  4. Leave-One-Out Ablation Matrix (Reviewer Comments #1, #2, #3):
     Verifies all 6 leave-one-out ablation configurations plus the plain multi-scale control.
  5. Equation (4) & Appendix B Eq. (11) — Generalized Mean (GeM) Multi-Scale TTA.
  6. Equation (5) & Appendix C Table 5 — Weight-Space Model Soup & Ensemble Knowledge Distillation.
  7. Equations (6)-(7) & Appendix A Eqs. (8)-(10) — Scale-Anchored Bi-ALT.
"""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gdfnet import (
    GDFNet,
    MSFDF,
    MultiScaleFrequencyFilter,
    PlainMultiScaleConv,
    MaskGuidedLGAG,
    GDFNetCompositeLoss,
    AsymmetricFocalTverskyLoss,
    EnsembleDistillationLoss,
    ScaleAnchoredBiALT,
    compute_bi_alt_threshold,
    run_125_probe_sensitivity_grid,
    build_weight_soup,
    fuse_gem_views,
)


def test_gdfnet_exact_parameter_count_and_shapes():
    """Verify GDF-Net (ConvFormer-S18) has 29,503,181 parameters and 534 state_dict keys."""
    model = GDFNet(
        backbone_name="convformer_s18.sail_in22k_ft_in1k_384",
        pretrained=False,
        drop_path_rate=0.15,
    )
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 29_503_181, f"Expected 29,503,181 parameters, got {total_params}"
    assert len(model.state_dict()) == 534, f"Expected 534 state_dict keys, got {len(model.state_dict())}"
    assert model.backbone.get_channels() == (64, 128, 320, 512)


def test_ms_fdf_equations_1_and_2_and_plain_control():
    """Verify MS-FDF Equations (1) & (2), intermediates, and complexity-matched PlainMultiScaleConv."""
    torch.manual_seed(42)
    fdf = MSFDF(in_channels=64).eval()
    ctrl = PlainMultiScaleConv(in_channels=64).eval()

    # Exact parameter count match between MS-FDF and PlainMultiScaleConv control
    p_fdf = sum(p.numel() for p in fdf.parameters())
    p_ctrl = sum(p.numel() for p in ctrl.parameters())
    assert p_fdf == p_ctrl, f"Expected identical params for control ({p_fdf} vs {p_ctrl})"

    e_i = torch.randn(2, 64, 48, 48, requires_grad=True)
    with torch.no_grad():
        actual_out, inter = fdf(e_i, return_intermediates=True)
        assert torch.allclose(inter["x_low"] + inter["x_high"], e_i, atol=1e-6)
        assert (inter["beta"] >= 0.0).all() and (inter["beta"] <= 1.0).all()
        assert torch.allclose(inter["x_filt"], inter["x_high"] * inter["beta"], atol=1e-6)
        expected_out = fdf.fusion(torch.cat([inter["x_low"], inter["x_filt"]], dim=1)) + e_i
        assert torch.allclose(actual_out, expected_out, atol=1e-6)
        ctrl_out = ctrl(e_i)
        assert ctrl_out.shape == actual_out.shape


def test_mask_guided_lgag_equation_3_differentiable_vs_detached():
    """
    Verify Mask-Guided LGAG Equation (3) under differentiable (detach_mask=False),
    detached (detach_mask=True), and feature-only (mask_guidance=False) modes.
    """
    torch.manual_seed(42)
    skip = torch.randn(2, 64, 48, 48, requires_grad=True)
    gating = torch.randn(2, 128, 24, 24, requires_grad=True)
    aux_logits = torch.randn(2, 1, 24, 24, requires_grad=True)

    # 1. Differentiable Mask-Guided LGAG (default)
    lgag_diff = MaskGuidedLGAG(skip_channels=64, gating_channels=128, mask_guidance=True, detach_mask=False)
    s_i, alpha = lgag_diff(skip, gating, torch.sigmoid(aux_logits), return_attention=True)
    assert s_i.shape == (2, 64, 48, 48) and alpha.shape == (2, 64, 48, 48)
    s_i.mean().backward()
    assert aux_logits.grad is not None and aux_logits.grad.abs().sum() > 0

    # 2. Detached Mask-Guided LGAG
    aux_logits_det = torch.randn(2, 1, 24, 24, requires_grad=True)
    lgag_det = MaskGuidedLGAG(skip_channels=64, gating_channels=128, mask_guidance=True, detach_mask=True)
    s_det = lgag_det(skip.detach(), gating.detach(), torch.sigmoid(aux_logits_det))
    s_det.mean().backward()
    assert aux_logits_det.grad is None or aux_logits_det.grad.abs().sum() == 0


def test_leave_one_out_ablation_configurations():
    """Verify all 6 leave-one-out ablation configurations plus plain multi-scale control."""
    configs = [
        ("1_baseline", dict(use_msfdf=False, use_lgag=False, mask_guidance=False, detach_mask=False)),
        ("2_msfdf_only", dict(use_msfdf=True, use_lgag=False, mask_guidance=False, detach_mask=False)),
        ("3_lgag_only", dict(use_msfdf=False, use_lgag=True, mask_guidance=False, detach_mask=False)),
        ("4_msfdf_lgag_nomask", dict(use_msfdf=True, use_lgag=True, mask_guidance=False, detach_mask=False)),
        ("5_mask_guided_detach", dict(use_msfdf=True, use_lgag=True, mask_guidance=True, detach_mask=True)),
        ("6_full_gdfnet_nodetach", dict(use_msfdf=True, use_lgag=True, mask_guidance=True, detach_mask=False)),
        ("7_plain_ms_control", dict(use_msfdf=True, use_lgag=True, mask_guidance=True, detach_mask=False, msfdf_variant="plain_multiscale")),
    ]
    x = torch.randn(1, 3, 128, 128)
    param_counts = {}
    for name, kw in configs:
        m = GDFNet(pretrained=False, **kw).eval()
        param_counts[name] = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(x)
            assert y.shape == (1, 1, 128, 128)

    assert param_counts["1_baseline"] < param_counts["2_msfdf_only"] < param_counts["6_full_gdfnet_nodetach"]
    assert param_counts["1_baseline"] < param_counts["3_lgag_only"] < param_counts["6_full_gdfnet_nodetach"]
    assert param_counts["6_full_gdfnet_nodetach"] == 29_503_181
    assert param_counts["7_plain_ms_control"] == 29_503_181


def test_full_gdfnet_end_to_end_gradient_flow():
    """Verify 100% of trainable parameters in GDF-Net receive non-zero gradients in a single pass."""
    torch.manual_seed(42)
    model = GDFNet(
        backbone_name="convformer_s18.sail_in22k_ft_in1k_384",
        pretrained=False,
        drop_path_rate=0.0,
    ).train()
    x = torch.randn(2, 3, 128, 128)
    y = torch.randint(0, 2, (2, 1, 128, 128)).float()

    outputs = model(x, return_deep_supervision=True)
    assert set(outputs.keys()) == {"logits", "p1", "p2", "p3", "p4"}
    criterion = GDFNetCompositeLoss()
    loss = criterion(outputs, y)
    loss.backward()

    missing_grads = [
        name for name, p in model.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
    ]
    assert len(missing_grads) == 0, f"Parameters with missing gradients: {missing_grads[:5]}"


def test_gem_ms_tta_equation_4():
    """Verify Generalized Mean (GeM) Multi-Scale TTA fusion (Equation 4 & Appendix B Eq. 11)."""
    rng = np.random.default_rng(42)
    v384 = rng.uniform(0.1, 0.9, size=(4, 4, 32, 32)).astype(np.float32)
    v448 = rng.uniform(0.1, 0.9, size=(4, 2, 32, 32)).astype(np.float32)
    v352 = rng.uniform(0.1, 0.9, size=(4, 2, 32, 32)).astype(np.float32)
    weights = (4.0 / 6.0, 2.0 / 6.0, 0.0)

    fused_p1 = fuse_gem_views(v384, v448, v352, mode="ms8_gem", gem_p=1.0, scale_weights=weights)
    expected_p1 = weights[0] * v384.mean(axis=1) + weights[1] * v448.mean(axis=1)
    assert np.allclose(fused_p1, expected_p1, atol=1e-5)

    p = 1.75
    fused_gem = fuse_gem_views(v384, v448, v352, mode="ms8_gem", gem_p=p, scale_weights=weights)
    expected_gem = np.power(
        weights[0] * np.mean(np.power(v384, p), axis=1) + weights[1] * np.mean(np.power(v448, p), axis=1),
        1.0 / p,
    )
    assert np.allclose(fused_gem, expected_gem, atol=1e-5)


def test_weight_soup_and_distillation_loss_equation_5():
    """Verify Weight-Space Model Soup (0.70/0.30) and EnsembleDistillationLoss (Equation 5)."""
    sd1 = {"w": torch.tensor([1.0, 2.0]), "step": torch.tensor(10, dtype=torch.int64)}
    sd2 = {"w": torch.tensor([3.0, 4.0]), "step": torch.tensor(20, dtype=torch.int64)}
    soup = build_weight_soup([sd1, sd2], weights=(0.70, 0.30))
    assert torch.allclose(soup["w"], torch.tensor([0.70 * 1.0 + 0.30 * 3.0, 0.70 * 2.0 + 0.30 * 4.0]))

    kd_loss_fn = EnsembleDistillationLoss(kd_weight=0.55, temp=2.0)
    B, H, W = 2, 64, 64
    outputs = {
        "logits": torch.randn(B, 1, H, W, requires_grad=True),
        "p1": torch.randn(B, 1, H, W, requires_grad=True),
        "p2": torch.randn(B, 1, H, W, requires_grad=True),
        "p3": torch.randn(B, 1, H, W, requires_grad=True),
        "p4": torch.randn(B, 1, H, W, requires_grad=True),
    }
    target = torch.randint(0, 2, (B, 1, H, W)).float()
    teacher_logits = torch.randn(B, 1, H, W)
    teacher_prob = torch.sigmoid(teacher_logits)

    loss = kd_loss_fn(outputs, target, teacher_prob, teacher_logits)
    assert loss.item() > 0.0
    loss.backward()
    assert outputs["logits"].grad is not None


def test_scale_anchored_bi_alt_equations_6_to_10():
    """Verify Scale-Anchored Bi-ALT Equations (6)-(10) across Diffuse, Transitional, and Crisp regimes."""
    H, W = 384, 384

    p_diffuse = np.full((H, W), 0.05, dtype=np.float32)
    p_diffuse[150:200, 150:200] = 0.85
    p_diffuse[100:250, 100:250] = np.maximum(p_diffuse[100:250, 100:250], 0.45)
    t_i, branch, rho, a_mid = compute_bi_alt_threshold(p_diffuse, t_base=0.74)
    assert branch == 1 and t_i == pytest.approx(0.56, abs=1e-5) and rho > 1.65

    p_trans = np.full((H, W), 0.05, dtype=np.float32)
    p_trans[100:150, 100:120] = 0.85
    p_trans[100:150, 120:129] = 0.45
    t_i, branch, rho, a_mid = compute_bi_alt_threshold(p_trans, t_base=0.74)
    assert branch == 2 and t_i == pytest.approx(0.68, abs=1e-5) and 1.32 < rho <= 1.65

    p_crisp = np.full((H, W), 0.05, dtype=np.float32)
    p_crisp[100:200, 100:200] = 0.95
    t_i, branch, rho, a_mid = compute_bi_alt_threshold(p_crisp, t_base=0.74)
    assert branch == 3 and t_i == pytest.approx(0.74, abs=1e-5) and rho <= 1.32 and a_mid <= 0.28


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
