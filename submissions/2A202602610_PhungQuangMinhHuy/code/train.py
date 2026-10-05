"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình: đổi thí nghiệm chỉ bằng cách đổi `Config`.
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR

# Import các module nội bộ
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset
import model as model_utils
import losses
from eval import save_predictions, compute_metrics


def _autocast():
    """Context AMP tương thích cả torch cũ (`torch.cuda.amp`) và mới (`torch.amp`)."""
    try:
        return torch.amp.autocast(device_type="cuda")
    except Exception:
        return torch.cuda.amp.autocast()


def _grad_scaler(enabled: bool):
    """GradScaler tương thích nhiều phiên bản torch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    optimizer_name: str = "adamw"      # adamw | sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config):
    """Xây dựng optimizer với 3 nhóm tham số."""
    groups = model_utils.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay
    )
    if cfg.optimizer_name == "sgd":
        return SGD(groups, momentum=0.9)
    return AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Linear warmup + Cosine Annealing scheduler."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential Moving Average của trọng số mô hình."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model.train()
    # Nếu backbone bị đóng băng, giữ các tầng backbone và BN ở eval mode
    if cfg.init == "frozen":
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()

    total_loss = 0.0
    num_batches = len(loader)
    use_amp = cfg.amp and torch.cuda.is_available()

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if cfg.mix:
            mixed_images, mixed_targets = losses.mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)
            if use_amp:
                with _autocast():
                    outputs = model(mixed_images)
                    loss = losses.mixed_loss(criterion, outputs, mixed_targets)
            else:
                outputs = model(mixed_images)
                loss = losses.mixed_loss(criterion, outputs, mixed_targets)
        else:
            if use_amp:
                with _autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()

    avg_loss = total_loss / max(1, num_batches)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": avg_loss, "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    """Chạy model trên loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    total_loss = 0.0
    all_filenames = []
    all_y_true = []
    all_logits = []

    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * len(images)
            all_filenames.extend(filenames)
            all_y_true.extend(targets.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    total_samples = len(all_filenames)
    avg_loss = total_loss / max(1, total_samples)
    all_logits = np.concatenate(all_logits, axis=0)
    all_y_true = np.array(all_y_true)

    # Softmax probabilities
    probs = np.exp(all_logits - np.max(all_logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    metrics = compute_metrics(all_y_true, probs.argmax(axis=1), probs)
    metrics["loss"] = avg_loss

    return all_filenames, all_y_true, all_logits, probs, metrics


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]
    val_top1 = [h["val_top1"] * 100 for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss Curve
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, val_loss, "s--", label="Val Loss", color="#ff7f0e")
    axes[0].set_title("Loss theo Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Macro-F1 & Top-1 Accuracy Curve
    axes[1].plot(epochs, val_macro_f1, "o-", label="Val Macro-F1", color="#2ca02c")
    axes[1].set_title("Val Macro-F1 theo Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Top-1 Accuracy (%)
    axes[2].plot(epochs, val_top1, "^-", label="Val Top-1 Acc (%)", color="#d62728")
    axes[2].set_title("Val Top-1 Accuracy (%)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.suptitle(f"Biểu đồ Huấn luyện: {title}", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu kết quả."""
    # 1. Khởi tạo
    set_seed(cfg.seed)
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Dữ liệu & kiểm tra split
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    # 3. Transforms & Loaders
    train_transform = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_transform = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_transform,
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler,
        num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_transform,
        batch_size=cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    # 4. Model, Loss, Optimizer, Scheduler
    model = model_utils.build_model(
        name=cfg.backbone,
        pretrained=(cfg.init != "scratch"),
        num_classes=dataset.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init
    ).to(device)

    # GMAC & Params
    params_m = model_utils.count_params(model)
    gmacs = model_utils.count_gmacs(model, img_size=cfg.img_size)

    # Loss
    loss_kwargs = {}
    if cfg.loss == "ls":
        loss_kwargs["smoothing"] = cfg.label_smoothing if cfg.label_smoothing > 0 else 0.1
    elif cfg.loss == "focal":
        loss_kwargs["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        c_weights = losses.class_weights(train_df["Label"].value_counts().to_dict(), beta=cfg.class_weight_beta or 0.0)
        loss_kwargs["weight"] = c_weights.to(device)

    criterion = losses.build_criterion(cfg.loss, **loss_kwargs).to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = _grad_scaler(enabled=(cfg.amp and torch.cuda.is_available()))

    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 5. Loop training
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    best_val_results = None

    start_time = time.time()
    epoch_times = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        # Val evaluation (using EMA weights if enabled)
        if ema is not None:
            ema.apply_shadow(model)

        val_names, val_true, val_logits, val_probs, val_metrics = evaluate(model, val_loader, val_criterion, device)

        if ema is not None:
            ema.restore(model)

        val_macro_f1 = val_metrics["macro_f1"]
        val_top1 = val_metrics["top1"]
        val_loss = val_metrics["loss"]

        history.append({
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "lr": train_res["lr"],
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_top1": val_top1,
            "epoch_time": t_epoch
        })

        # Checkpoint selection based on Val Macro-F1 (README S2 & GUIDE N3)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            # Nếu dùng EMA, lưu đúng trọng số EMA đã dùng để đo val (không phải trọng số thô)
            if ema is not None:
                ema.apply_shadow(model)
                best_state_dict = copy.deepcopy(model.state_dict())
                ema.restore(model)
            else:
                best_state_dict = copy.deepcopy(model.state_dict())
            best_val_results = (val_names, val_true, val_logits, val_probs, val_metrics)
            torch.save(best_state_dict, r_dir / "best_model.pt")

    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0

    # 6. Load best model and save Val predictions
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if best_val_results is not None:
        val_names, val_true, val_logits, val_probs, val_metrics = best_val_results
        save_predictions(pred_path(cfg, "val"), val_names, val_true, val_probs)
        np.save(r_dir / "val_logits.npy", val_logits)

    # 7. Test predictions (ONLY in Step 4 when save_test_predictions is True)
    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, val_transform,
            batch_size=cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        test_names, test_true, test_logits, test_probs, test_metrics = evaluate(model, test_loader, val_criterion, device)
        save_predictions(pred_path(cfg, "test"), test_names, test_true, test_probs)
        np.save(r_dir / "test_logits.npy", test_logits)

    # 8. Save history & plot curves
    history_df = pd.DataFrame(history)
    history_df.to_csv(r_dir / "history.csv", index=False)

    curve_title = f"{cfg.exp_id} ({cfg.backbone}) - Best Val Macro-F1: {best_macro_f1:.4f} (ep {best_epoch})"
    curve_img_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_img_path, curve_title)

    summary = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "best_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "val_top1": best_val_results[4]["top1"] if best_val_results else 0.0,
        "params_m": params_m,
        "gmacs": gmacs,
        "train_time_per_epoch": avg_epoch_time,
        "test_macro_f1": test_metrics["macro_f1"] if test_metrics else None,
        "test_top1": test_metrics["top1"] if test_metrics else None,
    }

    with open(r_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Parse list key=value pairs into dictionary with proper Config types."""
    overrides = {}
    field_types = {f.name: f.type for f in dataclasses.fields(Config)}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ: {pair}. Định dạng yêu cầu: KEY=VALUE")
        key, val_str = pair.split("=", 1)
        key = key.strip()
        val_str = val_str.strip()

        if key not in field_types:
            raise ValueError(f"Config không có trường: '{key}'")

        if val_str.lower() in ("none", "null"):
            overrides[key] = None
        elif val_str.lower() in ("true", "1", "yes"):
            overrides[key] = True
        elif val_str.lower() in ("false", "0", "no"):
            overrides[key] = False
        else:
            # Type casting
            target_type = field_types[key]
            if "int" in str(target_type):
                overrides[key] = int(val_str)
            elif "float" in str(target_type):
                overrides[key] = float(val_str)
            else:
                overrides[key] = val_str

    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Lab Day 2 (DeepWeeds)")
    parser.add_argument("--set", nargs="+", default=[], help="Cặp KEY=VALUE để ghi đè Config (ví dụ exp_id=B01 seed=0)")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"=== Bắt đầu thí nghiệm: {cfg.exp_id} (backbone: {cfg.backbone}, seed: {cfg.seed}) ===")
    res = run(cfg)
    print(f"=== Hoàn thành: Best Val Macro-F1 = {res['val_macro_f1']:.4f}, Top-1 = {res['val_top1']*100:.2f}% ===")


if __name__ == "__main__":
    main()
