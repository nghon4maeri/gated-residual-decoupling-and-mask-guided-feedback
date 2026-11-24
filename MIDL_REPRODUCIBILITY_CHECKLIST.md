# MIDL 2027 Official Reproducibility Checklist (`https://www.midl.io/reproducibility`)

This document maps every requirement of the **[MIDL Reproducibility Guidelines](https://www.midl.io/reproducibility)** to the exact files, scripts, pretrained weights, dataset partitions, and manuscript sections in this release repository.

---

## 1. General Approach & Documentation

| MIDL Reproducibility Question | Answer | Verification & Location in Paper / Repository |
| :--- | :---: | :--- |
| **Is the reproducibility of the work addressed?** | **Yes** | Addressed in Section 4 (*Experimental Setup*) and Supplementary Appendices A–E of the paper, and verified via automated mathematical consistency tests (`tests/test_mathematical_consistency.py`). |
| **Are all publicly available materials detailed in the main body of the paper?** | **Yes** | Section 4 (*Experimental Setup*) explicitly states the public availability of source code, declarative YAML configurations, deterministic split manifests, per-image metric logs, and pretrained PyTorch checkpoints. |

---

## 2. Code Availability

| MIDL Reproducibility Question | Answer | Verification & Location in Paper / Repository |
| :--- | :---: | :--- |
| **Is the code publicly available?** | **Yes** | Released under the permissive **MIT License** (`LICENSE`). |
| **Do property rights permit making the code public?** | **Yes** | All code is original research software released under the MIT License with no institutional or proprietary restrictions. |
| **Is a link to a long-term public Git repository included in the main body of the paper?** | **Yes** | Explicitly linked in Section 4 (*Experimental Setup*) of the main manuscript (`https://github.com/nghon4maeri/feedback-attention-network`, anonymized as `https://anonymous.4open.science/r/GDF-Net-MIDL2027` in blind review builds). |
| **Are dependencies, environment specifications, and containerization provided?** | **Yes** | Provided in `requirements.txt` (pinned versions), `environment.yml` (Conda environment), `Dockerfile` (CUDA 12.4 + PyTorch 2.6.0 container), and `.github/workflows/ci.yml` (automated unit test CI). |

---

## 3. Trained Model Weights & Artifacts

| MIDL Reproducibility Question | Answer | Verification & Location in Paper / Repository |
| :--- | :---: | :--- |
| **Is the presented trained model publicly available?** | **Yes** | All trained checkpoints (`.pth`, PyTorch `state_dict` format, 534 keys, 29,503,181 parameters) for ISIC-2018, CVC-ClinicDB, Kvasir-SEG, and DSB-2018 are released as Kaggle kernel outputs. |
| **Are weights hosted on long-term storage platforms?** | **Yes** | `scripts/download_weights.py` downloads the checkpoints via the Kaggle CLI and verifies each one by strict `state_dict` loading into `GDFNet`. |
| **Are standard open model formats used?** | **Yes** | Standard PyTorch `.pth` state dictionaries (`torch.load(..., weights_only=True)`) and optional ONNX export (`scripts/profile_hardware.py --export-onnx`). |

---

## 4. Datasets & Deterministic Partitions

| MIDL Reproducibility Question | Answer | Verification & Location in Paper / Repository |
| :--- | :---: | :--- |
| **Are publicly available datasets used to train and evaluate the model?** | **Yes** | Evaluated exclusively on seven public biomedical benchmarks: **ISIC-2018** (Task 1), **CVC-ClinicDB**, **Kvasir-SEG**, **DSB-2018**, **ISBI-2012 EM-Dataset**, **CHASE-DB1**, and **DRIVE**. |
| **Are automated download, preprocessing, and deterministic split scripts provided?** | **Yes** | `scripts/download_datasets.py` downloads, extracts, verifies image-mask pairs, and applies the exact deterministic partitions (`splits/split_manifest.json` & `gdfnet/data/datasets.py`). |
| **Are per-image evaluation logs and statistical test scripts provided?** | **Yes** | Summary and per-image logs are provided in `results/`, with paired $t$-tests, Wilcoxon signed-rank tests, Shapiro-Wilk normality diagnostics, Holm-Bonferroni corrections, and 2,000-sample bootstrap 95% CIs reproducible via `scripts/run_statistical_tests.py`. |

---

## 5. Paper Table / Figure to Command Mapping

| Manuscript Artifact | Reproducibility Command | Verified Output File |
| :--- | :--- | :--- |
| **Mathematical & Ablation Unit Tests** (Eqs. 1–11, 6 Ablation Configs) | `pytest tests/test_mathematical_consistency.py -v` | 100% passing assertions (29,503,181 params, 534 keys, gradient flow) |
| **Table 1 & Table 2** (ISIC-2018 & Multi-Dataset Benchmarks + Statistical Tests) | `python scripts/run_statistical_tests.py` | Paired $t$-test & Wilcoxon $p$-values vs. `GDF-Net (Real-Time) [Ref.]` |
| **Table 3** (Fixed-Backbone Leave-One-Out Ablation & Lesion Stratification) | `python scripts/run_ablation_suite.py --config configs/ablation_leave_one_out.yaml` | `results/ablation_convformer_s18_summary.csv` |
| **Appendix Table 4 & Figure 6** (Bi-ALT Sensitivity & Adaptive Threshold Comparison) | `python scripts/evaluate_all_benchmarks.py --include-bialt-sensitivity` | `results/bialt_125grid_validation_sensitivity.csv` |
| **Appendix Table 6** (Multi-Resolution RTX 4060 Latency, FPS & VRAM Profiling) | `python scripts/profile_hardware.py --resolutions 256 384 512` | `results/rtx4060_multires_hardware_profiling.csv` |
