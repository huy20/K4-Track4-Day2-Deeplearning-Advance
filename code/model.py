"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện chuẩn:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init`:
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    if timm is None:
        raise ImportError("Cần cài đặt timm (`pip install timm`) để sử dụng build_model.")

    is_pretrained = pretrained if init != "scratch" else False
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )

    if init == "frozen":
        freeze_backbone(model)

    # Lưu tag cấu hình pretrained để ghi log
    pretrained_tag = getattr(model, "pretrained_cfg", {}).get("tag", None) if hasattr(model, "pretrained_cfg") else None
    if not pretrained_tag and hasattr(model, "default_cfg"):
        pretrained_tag = model.default_cfg.get("tag", None) or model.default_cfg.get("architecture", "")
    model.pretrained_tag = pretrained_tag or ("scratch" if init == "scratch" else "default")

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head."""
    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for param in classifier.parameters():
            param.requires_grad = True
    elif isinstance(classifier, nn.Parameter):
        classifier.requires_grad = True


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    head_params = set()
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for p in classifier.parameters():
            head_params.add(p)
    elif isinstance(classifier, nn.Parameter):
        head_params.add(classifier)

    bb_weights_decay = []
    bb_weights_no_decay = []
    head_weights = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in head_params:
            head_weights.append(param)
        elif param.ndim > 1:
            bb_weights_decay.append(param)
        else:
            bb_weights_no_decay.append(param)

    groups = []
    if bb_weights_decay:
        groups.append({"params": bb_weights_decay, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"})
    if bb_weights_no_decay:
        groups.append({"params": bb_weights_no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"})
    if head_weights:
        groups.append({"params": head_weights, "lr": lr_head, "weight_decay": weight_decay, "name": "head"})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return float(total) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size."""
    try:
        from thop import profile
        dummy = torch.randn(1, 3, img_size, img_size)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy = torch.randn(1, 3, img_size, img_size)
            flops = FlopCountAnalysis(model, dummy).total()
            return float(flops) / 1e9
        except Exception:
            # Ước lượng chuẩn theo các backbone phổ biến nếu không có thop/fvcore
            name = getattr(model, "pretrained_cfg", {}).get("architecture", "") if hasattr(model, "pretrained_cfg") else ""
            estimates = {
                "resnet50": 4.12,
                "resnext50_32x4d": 4.24,
                "convnext_tiny": 4.46,
                "swin_tiny_patch4_window7_224": 4.50,
                "deit_small_patch16_224": 4.60,
                "mobilenetv3_large_100": 0.23,
                "efficientnet_b0": 0.39
            }
            for k, v in estimates.items():
                if k in str(name).lower():
                    return v
            # Fallback tính xấp xỉ từ số lượng tham số
            total_params = sum(p.numel() for p in model.parameters())
            return round((total_params * (img_size / 224)**2 * 1.6) / 1e7, 2)
