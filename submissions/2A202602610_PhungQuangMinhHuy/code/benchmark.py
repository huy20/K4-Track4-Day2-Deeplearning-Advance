"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], None], warmup: int = 10, iters: int = 100, sync: Callable[[], None] | None = None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây."""
    # Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times, ddof=1)),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   fused_bn: bool = False,
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    if dtype == "fp16":
        model = model.half()
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float16, device=dev)
    else:
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float32, device=dev)

    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    if dtype == "amp" and dev.type == "cuda":
        def forward_fn():
            with torch.inference_mode(), torch.cuda.amp.autocast():
                _ = model(dummy_input)
    else:
        def forward_fn():
            with torch.inference_mode():
                _ = model(dummy_input)

    res = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    images_per_s = (batch_size / (res["p50"] / 1000.0)) if res["p50"] > 0 else 0.0

    report = {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "fused_bn": fused_bn,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"],
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }
    return report


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Đo độ trễ thực tế của TTA K views."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    def tta_fn():
        with torch.inference_mode():
            # Chạy K views (ví dụ nguyên bản + lật)
            for i in range(k_views):
                inp = torch.flip(dummy_input, dims=[-1]) if (i % 2 == 1) else dummy_input
                _ = model(inp)

    res = bench(tta_fn, warmup=warmup, iters=iters, sync=sync_fn)
    return {
        "k_views": k_views,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"],
    }
