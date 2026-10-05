"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).
"""
from __future__ import annotations

import copy
import math
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model: nn.Module, loader, device: torch.device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []

    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)

            outputs = model(images)

            all_filenames.extend(filenames)
            all_y_true.extend(targets.numpy() if isinstance(targets, torch.Tensor) else targets)
            all_logits.append(outputs.cpu().numpy())

    return all_filenames, np.array(all_y_true), np.concatenate(all_logits, axis=0)


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> list[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop`."""
    _, _, h, w = x.shape
    assert h >= crop and w >= crop, f"Ảnh ({h}x{w}) nhỏ hơn crop size ({crop})"

    crops = [
        x[:, :, :crop, :crop],                    # Top-left
        x[:, :, :crop, w - crop:],                # Top-right
        x[:, :, h - crop:, :crop],                # Bottom-left
        x[:, :, h - crop:, w - crop:],            # Bottom-right
        x[:, :, (h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]  # Center
    ]
    return crops


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    scaled = []
    for s in sizes:
        scaled.append(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False))
    return scaled


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    if space == "prob":
        probs_list = []
        for l in logits_per_view:
            z = l - np.max(l, axis=1, keepdims=True)
            e = np.exp(z)
            p = e / np.sum(e, axis=1, keepdims=True)
            probs_list.append(p)
        return np.mean(probs_list, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        z = avg_logits - np.max(avg_logits, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)
    else:
        raise ValueError(f"Không hỗ trợ aggregate space: {space}")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69)."""
    logits_t = torch.from_numpy(val_logits).float()
    labels_t = torch.from_numpy(val_labels).long()

    def nll_func(log_t: float) -> float:
        t = math.exp(log_t)
        scaled_logits = logits_t / t
        loss = F.cross_entropy(scaled_logits, labels_t)
        return float(loss.item())

    # Tối ưu một biến log(T)
    res = scipy.optimize.minimize_scalar(nll_func, bounds=(-2.0, 2.0), method="bounded")
    optimal_T = float(np.exp(res.x))
    return optimal_T


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    scaled = logits / max(1e-6, T)
    z = scaled - np.max(scaled, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Gộp một cặp Conv-BN thành Conv có bias (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)
    """
    w_conv = conv.weight.clone().reshape(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_weight = torch.mm(w_bn, w_conv).reshape(conv.weight.size())

    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device)

    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )
    fused_conv.weight.data.copy_(fused_weight)
    fused_conv.bias.data.copy_(fused_bias)
    return fused_conv


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp mọi cặp Conv2d-BatchNorm2d liền kề (trong `nn.Sequential` và trong các block
    như BasicBlock/Bottleneck của ResNet) thành một Conv duy nhất, chính xác lúc suy luận.

    Không thay đổi đầu ra về mặt toán học; chỉ giảm số phép toán và độ trễ.
    """
    model = copy.deepcopy(model)
    model.eval()

    fused_pairs = 0
    for module in list(model.modules()):
        children = list(module.named_children())
        i = 0
        while i < len(children) - 1:
            name_a, child_a = children[i]
            name_b, child_b = children[i + 1]
            if (isinstance(child_a, nn.Conv2d) and isinstance(child_b, nn.BatchNorm2d)
                    and child_b.running_var is not None
                    and child_a.out_channels == child_b.num_features):
                fused = _fuse_conv_bn_pair(child_a, child_b)
                setattr(module, name_a, fused)
                setattr(module, name_b, nn.Identity())
                fused_pairs += 1
                i += 2
            else:
                i += 1

    model._fused_conv_bn_pairs = fused_pairs
    return model
