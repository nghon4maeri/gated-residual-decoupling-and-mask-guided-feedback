"""
CUDA-Synchronized Multi-Resolution Hardware Latency, Throughput (FPS), and Peak VRAM Profiler
(Section 4 & Supplementary Appendix D Table 6).

Profiles GDF-Net (ConvFormer-S18, 29,503,181 parameters) at batch size B=1 over 100 warmup
iterations and 500 CUDA-synchronized timed runs across resolutions {256^2, 384^2, 512^2, 768^2}.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gdfnet import GDFNet, ScaleAnchoredBiALT


def profile_resolution(
    model: torch.nn.Module,
    device: torch.device,
    res: int = 384,
    warmup: int = 50,
    runs: int = 200,
    fp16: bool = False,
):
    model.eval()
    x = torch.randn(1, 3, res, res, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            if fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(x)
            else:
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(runs):
            if fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = model(x)
            else:
                out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_s = time.perf_counter() - t0

    latency_ms = (total_s / runs) * 1000.0
    fps = 1000.0 / latency_ms
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        if device.type == "cuda"
        else 0.0
    )

    # Profile Bi-ALT post-processing overhead
    prob_np = torch.sigmoid(out).squeeze().detach().cpu().float().numpy()
    bi_alt = ScaleAnchoredBiALT(t_base=0.74)
    t_b0 = time.perf_counter()
    for _ in range(100):
        _ = bi_alt.binarize(prob_np)
    bialt_ms = ((time.perf_counter() - t_b0) / 100.0) * 1000.0

    return {
        "resolution": f"{res}x{res}",
        "precision": "FP16" if fp16 else "FP32",
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 2),
        "bialt_overhead_ms": round(bialt_ms, 3),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile GDF-Net latency, FPS, and VRAM.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[256, 384, 512, 768])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GDFNet(pretrained=False, drop_path_rate=0.0).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"[+] Profiling GDF-Net ({params:,} parameters) on {device}")
    if device.type == "cuda":
        print(f"[+] GPU: {torch.cuda.get_device_name(0)}")

    for res in args.resolutions:
        stats = profile_resolution(model, device, res=res, warmup=args.warmup, runs=args.runs, fp16=False)
        print(
            f"  Res {stats['resolution']:7s} ({stats['precision']}) | "
            f"Latency: {stats['latency_ms']:6.2f} ms ({stats['fps']:5.2f} FPS) | "
            f"Bi-ALT: {stats['bialt_overhead_ms']:.2f} ms | Peak VRAM: {stats['peak_vram_mb']:.2f} MB"
        )


if __name__ == "__main__":
    main()
