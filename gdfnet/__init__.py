"""
GDF-Net: Gated Residual Decoupling and Mask-Guided Feedback for Medical Image Segmentation.

Official PyTorch implementation aligned 1:1 with the mathematical formulations in the paper:
  - Section 3.1 (Eqs. 1-2): Pure-CNN MetaFormer Backbone & Multi-Scale Feature Decoupling Filter (MS-FDF)
  - Section 3.2 (Eq. 3):    Mask-Guided Local-Global Attention Gating (Mask-Guided LGAG) & Cascaded MSCAM Decoder
  - Section 3.3 (Eqs. 4-5): Weight-Space Model Soups, Generalized Mean (GeM) MS-TTA, & Ensemble Knowledge Distillation
  - Section 3.4 (Eqs. 6-7): Scale-Anchored Bi-Parametric Adaptive Thresholding (Bi-ALT)
"""

from .models import (
    HierarchicalBackbone,
    MultiScaleFrequencyFilter,
    MaskGuidedLGAG,
    MultiScaleConvolutionalAttentionModule,
    CascadedMSCAMDecoderBlock,
    GDFNet,
    CanonicalGDFNetV4,
    FeedbackGDFNet,
    build_llrd_param_groups,
)
from .losses import (
    StructureLoss,
    GDFNetCompositeLoss,
    AsymmetricFocalTverskyLoss,
    EnsembleDistillationLoss,
)
from .inference import (
    ScaleAnchoredBiALT,
    compute_bi_alt_threshold,
    compute_otsu_threshold,
    calibrate_bi_alt_on_validation,
    run_125_probe_sensitivity_grid,
    ModelEMA,
    build_weight_soup,
    build_decoupled_weight_soup,
    extract_multiscale_views,
    fuse_gem_views,
)
from .utils import (
    compute_surface_distances,
    compute_segmentation_metrics,
    evaluate_cohort_predictions,
    paired_statistical_tests,
)

__version__ = "1.0.0"
__all__ = [
    "HierarchicalBackbone",
    "MultiScaleFrequencyFilter",
    "MaskGuidedLGAG",
    "MultiScaleConvolutionalAttentionModule",
    "CascadedMSCAMDecoderBlock",
    "GDFNet",
    "CanonicalGDFNetV4",
    "FeedbackGDFNet",
    "build_llrd_param_groups",
    "StructureLoss",
    "GDFNetCompositeLoss",
    "AsymmetricFocalTverskyLoss",
    "EnsembleDistillationLoss",
    "ScaleAnchoredBiALT",
    "compute_bi_alt_threshold",
    "compute_otsu_threshold",
    "calibrate_bi_alt_on_validation",
    "run_125_probe_sensitivity_grid",
    "ModelEMA",
    "build_weight_soup",
    "build_decoupled_weight_soup",
    "extract_multiscale_views",
    "fuse_gem_views",
    "compute_surface_distances",
    "compute_segmentation_metrics",
    "evaluate_cohort_predictions",
    "paired_statistical_tests",
]
