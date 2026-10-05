"""selfcheck.py - kiểm tra nhanh các phần dễ sai (RUBRIC.md mục H, mục 1.3 GUIDE).

Chạy:  python code/selfcheck.py
Không cần GPU. In "ALL SELFTEST OK" nếu mọi kiểm tra đạt.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import losses
import inference as inf


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)

    # 1. focal gamma=0 phải bằng CE
    logits = torch.randn(8, 9)
    y = torch.randint(0, 9, (8,))
    ce = F.cross_entropy(logits, y)
    fl0 = losses.FocalLoss(gamma=0.0)(logits, y)
    assert torch.allclose(ce, fl0, atol=1e-5), (ce.item(), fl0.item())
    print(f"[ok] focal(gamma=0) == CE ({float(ce):.4f})")

    # 2. LabelSmoothing(0) phải bằng CE
    ls0 = losses.LabelSmoothingCE(smoothing=0.0)(logits, y)
    assert torch.allclose(ce, ls0, atol=1e-6)
    print("[ok] LabelSmoothing(eps=0) == CE")

    # 3. CutMix/Mixup: shape đúng, lam hợp lệ, loss tính được
    x = torch.randn(4, 3, 32, 32)
    yb = torch.arange(4)
    xm, (ya2, yb2, lam) = losses.mix_batch(x, yb, alpha=1.0, mode="cutmix")
    assert xm.shape == x.shape and 0.0 <= lam <= 1.0
    _ = losses.mixed_loss(nn.CrossEntropyLoss(), torch.randn(4, 4), (ya2, yb2, lam))
    xm2, _ = losses.mix_batch(x, yb, alpha=0.8, mode="mixup")
    assert xm2.shape == x.shape
    print(f"[ok] CutMix/Mixup (lam={lam:.3f})")

    # 4. class_weights: tổng/normalize hợp lệ
    w = losses.class_weights([100, 100, 1000], beta=0.0)
    assert w.shape == (3,) and (w > 0).all()
    print(f"[ok] class_weights = {w.tolist()}")

    # 5. fuse_conv_bn: tương đương đầu ra trên ResNet-50 (torchvision, BN nặng)
    torchvision = __import__("torchvision")
    model = torchvision.models.resnet50(weights=None).eval()
    xin = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        out_a = model(xin)
    fused = inf.fuse_conv_bn(model).eval()
    with torch.inference_mode():
        out_b = fused(xin)
    maxdiff = (out_a - out_b).abs().max().item()
    assert maxdiff < 1e-4, maxdiff
    print(f"[ok] fuse_conv_bn: {getattr(fused, '_fused_conv_bn_pairs', '?')} cặp, "
          f"max|Δ|={maxdiff:.2e}")

    # 6. Temperature scaling: softmax hợp lệ, NLL không tăng trên chính val
    vlog = np.random.randn(400, 9).astype(np.float64) * 0.5
    vlab = np.random.randint(0, 9, 400)
    T = inf.fit_temperature(vlog, vlab)
    p_cal = inf.apply_temperature(vlog, T)
    assert p_cal.shape == (400, 9) and np.allclose(p_cal.sum(1), 1.0, atol=1e-6)
    nll = lambda p: -np.log(np.clip(p[np.arange(400), vlab], 1e-12, None)).mean()
    assert nll(p_cal) <= nll(inf.apply_temperature(vlog, 1.0)) + 1e-6
    print(f"[ok] temperature scaling T={T:.3f}, NLL {nll(inf.apply_temperature(vlog,1.0)):.4f} "
          f"-> {nll(p_cal):.4f}")

    # 7. TTA views
    xb = torch.randn(2, 3, 256, 256)
    crops = inf.views_multicrop(xb, crop=224)
    assert len(crops) == 5 and all(c.shape == (2, 3, 224, 224) for c in crops)
    agg = inf.aggregate_views([np.random.randn(10, 9), np.random.randn(10, 9)], space="prob")
    assert agg.shape == (10, 9) and np.allclose(agg.sum(1), 1.0, atol=1e-6)
    print("[ok] TTA 5-crop + aggregate_views")

    print("\nALL SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
