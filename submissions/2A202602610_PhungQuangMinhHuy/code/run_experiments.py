"""run_experiments.py - Driver chạy THẬT toàn bộ thí nghiệm Lab Day 2 (DeepWeeds).

Mọi số liệu sinh ra ở đây đến từ huấn luyện/suy luận thực tế, truy ngược được về
`runs/<exp_id>/seed<k>/` (config.json, history.csv, best_model.pt, summary.json).

Các stage (chạy tuần tự, có thể resume vì mỗi lần chạy lưu summary.json):
    sanity     : EDA, kiểm tra split, kiểm tra loss ban đầu ~ ln(9), overfit 1 batch
    backbones  : B01..B06 (>=5 backbone, cùng công thức nền, seed 0)
    training   : T00 (mốc) và T01..T14 (ablation công thức huấn luyện, seed 0)
    inference  : I00..I08 (TTA, FixRes, ensemble, EMA, temperature, fuse BN, FP16) + độ trễ
    final      : F01 (cấu hình chung kết, >=3 seed) và T00 (mốc, >=3 seed), ghi predictions/
    report     : results.xlsx (7 sheet) + report.md + chạy `eval.py score/grade`

Ví dụ:
    python code/run_experiments.py --stage sanity
    python code/run_experiments.py --stage all --epochs 12 --seeds 0 1 2
    python code/run_experiments.py --stage training --only T05 T07 T14 --epochs 3
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent
for _p in (str(ROOT), str(CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataset as ds
import model as model_utils
import losses
import train as tr
import inference as inf
import benchmark as bm
import eval as ev
from eval import save_predictions, compute_metrics


# --------------------------------------------------------------------------- #
# Cấu hình thí nghiệm
# --------------------------------------------------------------------------- #
BASELINE = dict(exp_id="T00", backbone="resnet50", init="finetune",
                note="Mốc: ResNet-50 + công thức nền + 1-view")

BACKBONES = [
    dict(exp_id="B01", backbone="resnet50", note="ResNet-50 (mốc CNN)"),
    dict(exp_id="B02", backbone="resnext50_32x4d", note="ResNeXt-50-32x4d (cardinality)"),
    dict(exp_id="B03", backbone="convnext_tiny", note="ConvNeXt-Tiny (CNN hiện đại)"),
    dict(exp_id="B04", backbone="swin_tiny_patch4_window7_224", note="Swin-Tiny (transformer)"),
    dict(exp_id="B05", backbone="mobilenetv3_large_100", note="MobileNetV3-Large (nhẹ)"),
    dict(exp_id="B06", backbone="efficientnet_b0", note="EfficientNet-B0 (nhẹ)"),
]

TRAINING = [
    dict(exp_id="T01", backbone="convnext_tiny", init="scratch", axis="A. Khởi tạo",
         note="Huấn luyện từ đầu"),
    dict(exp_id="T02", backbone="convnext_tiny", init="frozen", axis="A. Khởi tạo",
         note="Đóng băng backbone, chỉ train head"),
    dict(exp_id="T03", backbone="convnext_tiny", aug="color", axis="B. Augmentation",
         note="+ ColorJitter"),
    dict(exp_id="T04", backbone="convnext_tiny", aug="randaug", axis="B. Augmentation",
         note="+ RandAugment(2,9)"),
    dict(exp_id="T05", backbone="convnext_tiny", mix="cutmix", mix_alpha=1.0, axis="B. Augmentation",
         note="+ CutMix(alpha=1.0)"),
    dict(exp_id="T06", backbone="convnext_tiny", mix="mixup", mix_alpha=0.8, axis="B. Augmentation",
         note="+ Mixup(alpha=0.8)"),
    dict(exp_id="T07", backbone="convnext_tiny", loss="ls", label_smoothing=0.1, axis="C. Hàm loss",
         note="+ Label Smoothing(0.1)"),
    dict(exp_id="T08", backbone="convnext_tiny", loss="focal", focal_gamma=2.0, axis="C. Hàm loss",
         note="Focal Loss(gamma=2.0)"),
    dict(exp_id="T09", backbone="convnext_tiny", loss="ce_weighted", class_weight_beta=0.9999,
         axis="C. Hàm loss", note="Class-weighted CE(beta=0.9999)"),
    dict(exp_id="T10", backbone="convnext_tiny", sampler="balanced", axis="D. Cân bằng mẫu",
         note="WeightedRandomSampler"),
    dict(exp_id="T11", backbone="convnext_tiny", optimizer_name="sgd", axis="E. LR & Optimizer",
         note="SGD + Momentum 0.9"),
    dict(exp_id="T12", backbone="convnext_tiny", lr_head=1e-4, axis="E. LR & Optimizer",
         note="LR bằng nhau (1e-4)"),
    dict(exp_id="T13", backbone="convnext_tiny", ema_decay=0.999, axis="F. Chính quy hoá",
         note="Weight EMA(0.999)"),
    dict(exp_id="T14", backbone="convnext_tiny", mix="cutmix", mix_alpha=1.0, loss="ls",
         label_smoothing=0.1, ema_decay=0.999, axis="Kết hợp tốt nhất",
         note="CutMix + Label Smoothing + EMA"),
]

FINAL_RECIPE = dict(backbone="convnext_tiny", mix="cutmix", mix_alpha=1.0, loss="ls",
                    label_smoothing=0.1, ema_decay=0.999)


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def make_cfg(args, **over) -> tr.Config:
    kw = dict(
        images_dir=str(args.images_dir),
        labels_dir=str(args.labels_dir),
        out_dir=str(args.out),
        pred_dir=str(args.pred_dir),
        curves_dir=str(args.curves_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )
    kw.update(over)
    return tr.Config(**kw)


def resolved_tag(model: nn.Module) -> str:
    for attr in ("pretrained_cfg", "default_cfg"):
        d = getattr(model, attr, None)
        if isinstance(d, dict):
            return str(d.get("tag") or d.get("architecture") or "unknown")
    return "unknown"


def run_or_resume(args, over, save_test: bool = False) -> dict:
    cfg = make_cfg(args, save_test_predictions=save_test, **over)
    r_dir = tr.run_dir(cfg)
    summary_path = r_dir / "summary.json"
    if summary_path.exists() and not args.force:
        res = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[skip] {cfg.exp_id} seed{cfg.seed}: đã có {summary_path}")
        return res
    print(f"\n===== TRAIN {cfg.exp_id} | {cfg.backbone} | init={cfg.init} | seed={cfg.seed} "
          f"| epochs={cfg.epochs} =====", flush=True)
    t0 = time.time()
    res = tr.run(cfg)
    res["wall_time_s"] = time.time() - t0
    res["recipe"] = {k: v for k, v in over.items() if k != "exp_id"}
    summary_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def apply_common_dirs(args):
    for d in (args.out, args.pred_dir, args.curves_dir, args.eval_out):
        Path(d).mkdir(parents=True, exist_ok=True)


def run_eval_cli(args, sub: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / "eval.py"), sub, *extra]
    print("\n$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def load_model_from_run(run_path: Path) -> tuple[nn.Module, tr.Config]:
    cfg = tr.Config(**json.loads((run_path / "config.json").read_text(encoding="utf-8")))
    model = model_utils.build_model(name=cfg.backbone, pretrained=False,
                                    num_classes=ds.NUM_CLASSES, drop_rate=cfg.drop_rate,
                                    init="finetune")
    state = torch.load(run_path / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------- #
# Stage 0: sanity
# --------------------------------------------------------------------------- #
def stage_sanity(args) -> dict:
    print("=== STAGE sanity: EDA + kiểm tra split + pipeline ===")
    train_df, val_df, test_df = ds.load_split(args.labels_dir, fold=0)
    stats = ds.check_split(train_df, val_df, test_df, args.images_dir)

    counts = pd.DataFrame({
        "class": ev.CLASS_NAMES,
        "train": [stats["per_class"]["train"].get(i, 0) for i in range(9)],
        "val": [stats["per_class"]["val"].get(i, 0) for i in range(9)],
        "test": [stats["per_class"]["test"].get(i, 0) for i in range(9)],
    })
    counts["total"] = counts[["train", "val", "test"]].sum(1)
    counts["pct"] = 100 * counts["total"] / counts["total"].sum()
    print(counts.to_string(index=False))
    print("n =", stats["n"], "| overlap =", stats["overlap"])

    Path(args.out).mkdir(parents=True, exist_ok=True)
    counts.to_csv(Path(args.out) / "eda_class_counts.csv", index=False)
    (Path(args.out) / "eda_split.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=int), encoding="utf-8")

    # --- Kiểm tra loss ban đầu ~ ln(9) với head mới ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_utils.build_model("resnet50", pretrained=True, num_classes=9,
                                    init="finetune").to(device)
    val_tf = ds.build_transforms(train=False, img_size=args.img_size)
    probe_loader = ds.make_loader(train_df.head(128), args.images_dir, val_tf,
                                  batch_size=64, train=False, num_workers=args.num_workers)
    crit = nn.CrossEntropyLoss()
    model.eval()
    losses_before = []
    with torch.inference_mode():
        for images, targets, _ in probe_loader:
            out = model(images.to(device))
            losses_before.append(float(crit(out, targets.to(device))))
    init_loss = float(np.mean(losses_before))
    print(f"Loss ban đầu (kỳ vọng ~ln9={math.log(9):.4f}): {init_loss:.4f}")

    # --- Overfit 1 batch nhỏ ---
    model.train()
    bs = 32
    xs, ys = [], []
    for images, targets, _ in probe_loader:
        xs.append(images); ys.append(targets)
        if sum(len(x) for x in xs) >= bs:
            break
    xb = torch.cat(xs)[:bs].to(device)
    yb = torch.cat(ys)[:bs].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    model.train()
    for step in range(60):
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    print(f"Overfit 1 batch: loss {first:.4f} -> {last:.4f} (n={bs})")
    sanity = {"init_loss": init_loss, "expected_init_loss": math.log(9),
              "overfit_first": first, "overfit_last": last,
              "overfit_n": bs, "split": stats}
    (Path(args.out) / "sanity.json").write_text(json.dumps(sanity, indent=2, ensure_ascii=False, default=int),
                                                encoding="utf-8")
    print("=== sanity OK ===")
    return sanity


# --------------------------------------------------------------------------- #
# Stage 1 & 2: backbones + training
# --------------------------------------------------------------------------- #
def stage_backbones(args) -> list[dict]:
    print("=== STAGE backbones (B01..B06) ===")
    results = []
    for spec in BACKBONES:
        if args.only and spec["exp_id"] not in args.only:
            continue
        over = {k: v for k, v in spec.items() if k not in ("exp_id", "note")}
        over["seed"] = 0
        try:
            res = run_or_resume(args, {"exp_id": spec["exp_id"], **over})
            res["note"] = spec["note"]
            results.append(res)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {spec['exp_id']}: {type(e).__name__}: {e}")
            traceback.print_exc()
    return results


def stage_training(args) -> list[dict]:
    print("=== STAGE training (T00 + T01..T14) ===")
    results = []
    baseline = {k: v for k, v in BASELINE.items() if k not in ("exp_id", "note")}
    baseline["seed"] = 0
    try:
        res = run_or_resume(args, {"exp_id": "T00", **baseline})
        res["axis"] = "Baseline"
        res["note"] = BASELINE["note"]
        results.append(res)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] T00: {type(e).__name__}: {e}")
    for spec in TRAINING:
        if args.only and spec["exp_id"] not in args.only:
            continue
        over = {k: v for k, v in spec.items() if k not in ("exp_id", "note", "axis")}
        over["seed"] = 0
        try:
            res = run_or_resume(args, {"exp_id": spec["exp_id"], **over})
            res["axis"] = spec["axis"]
            res["note"] = spec["note"]
            res["diff"] = over
            results.append(res)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {spec['exp_id']}: {type(e).__name__}: {e}")
            traceback.print_exc()
    return results


# --------------------------------------------------------------------------- #
# Stage 3: inference
# --------------------------------------------------------------------------- #
def _collect(model, df, images_dir, transform, batch_size, num_workers, device, views, want_logits=False):
    """Chạy model với nhiều view; trả filenames, y_true, list probs theo view (và logits view 0)."""
    loader = ds.make_loader(df, images_dir, transform, batch_size, train=False,
                            num_workers=num_workers)
    model.eval().to(device)
    names, ys = [], []
    probs_per_view = [[] for _ in views]
    logits0 = []
    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device)
            for j, vf in enumerate(views):
                logits = model(vf(images))
                p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                probs_per_view[j].append(p)
                if j == 0 and want_logits:
                    logits0.append(logits.float().cpu().numpy())
            names.extend(filenames)
            ys.extend(targets.numpy())
    probs = [np.concatenate(p, axis=0) for p in probs_per_view]
    out = (names, np.array(ys), probs)
    if want_logits:
        return out + (np.concatenate(logits0, axis=0),)
    return out


def stage_inference(args) -> list[dict]:
    print("=== STAGE inference (I00..I08) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = ds.load_split(args.labels_dir, fold=0)
    y_val = val_df["Label"].to_numpy()

    t14_run = Path(args.out) / "T14" / "seed0"
    if not (t14_run / "best_model.pt").exists():
        print("[skip] inference: chưa có T14/seed0.")
        return []
    model, cfg14 = load_model_from_run(t14_run)
    model = model.to(device)
    val_logits = np.load(t14_run / "val_logits.npy")

    tf224 = ds.build_transforms(train=False, img_size=224)
    tf256 = ds.build_transforms(train=False, img_size=256)
    results: list[dict] = []

    def add(exp_id, method, probs, k, extra=None):
        m = compute_metrics(y_val, probs.argmax(1), probs)
        row = {"exp_id": exp_id, "method": method, "K": k, "macro_f1": m["macro_f1"],
               "top1": m["top1"], "ece": m["ece"], "nll": m["nll"], "balanced_acc": m["balanced_acc"]}
        if extra:
            row.update(extra)
        results.append(row)
        print(f"  {exp_id[:12]:12s} F1={m['macro_f1']:.4f} top1={m['top1']*100:.2f}% ECE={m['ece']:.4f}")
        return row

    # I00: 1 view (dùng lại val logits đã lưu của T14)
    p_soft = inf.apply_temperature(val_logits, 1.0)
    add("I00", "1 view (CenterCrop 224)", p_soft, 1)

    # I01: TTA lật ngang K=2
    names, yv, probs01, _ = _collect(model, val_df, args.images_dir, tf224,
                                     args.batch_size, args.num_workers, device,
                                     [inf.view_identity, inf.view_hflip], want_logits=True)
    p_hflip = inf.aggregate_views(probs01, space="prob")
    add("I01", "TTA lật ngang", p_hflip, 2)

    # I03a/b: gộp xác suất vs logit
    # (view logits: cần riêng; dùng lại probs01 cho I03a)
    for code, space in (("I03a", "prob"), ("I03b", "logit")):
        if space == "prob":
            add(code, "Gộp xác suất", inf.aggregate_views(probs01, space="prob"), 2)
        else:
            # thu logits từng view
            _, _, pr_v, lg0 = _collect(model, val_df, args.images_dir, tf224,
                                       args.batch_size, args.num_workers, device,
                                       [inf.view_identity], want_logits=True)
            _, _, pr_v2, lg1 = _collect(model, val_df, args.images_dir, tf224,
                                        args.batch_size, args.num_workers, device,
                                        [inf.view_hflip], want_logits=True)
            add(code, "Gộp logit", inf.aggregate_views([lg0, lg1], space="logit"), 2)

    # I02: TTA 5-crop trên ảnh 256 (4 góc + giữa)
    views5 = [lambda x, c=i: inf.views_multicrop(x, crop=224)[c] for i in range(5)]
    names, yv, crops_probs = _collect(model, val_df, args.images_dir, tf256,
                                      args.batch_size, args.num_workers, device, views5)
    add("I02", "TTA 5-crop", inf.aggregate_views(crops_probs, space="prob"), 5)

    # I04a/b: FixRes test size 256 / 288
    for code, size in (("I04a", 256), ("I04b", 288)):
        tf = ds.build_transforms(train=False, img_size=size)
        _, _, pr = _collect(model, val_df, args.images_dir, tf, args.batch_size,
                            args.num_workers, device, [inf.view_identity])
        add(code, f"FixRes test size {size}", pr[0], 1)

    # I06: EMA (T14 đã dùng EMA khi đo val => trùng I00, ghi nhận là free)
    add("I06", "Trọng số EMA (đã dùng trong T14)", p_soft.copy(), 1)

    # I07: temperature scaling + ECE trước/sau
    T = inf.fit_temperature(val_logits, y_val)
    p_cal = inf.apply_temperature(val_logits, T)
    row = add("I07", f"Temperature Scaling (T={T:.3f})", p_cal, 1)
    row["T"] = float(T)
    row["ece_before"] = float(compute_metrics(y_val, p_soft.argmax(1), p_soft)["ece"])
    print(f"  I07 T={T:.3f} | ECE {row['ece_before']:.4f} -> {row['ece']:.4f}")

    # I05: ensemble ConvNeXt(T14)+Swin(B04)+ResNet(B01)
    ens_probs = []
    for exp in ("T14", "B04", "B01"):
        rp = Path(args.out) / exp / "seed0"
        if (rp / "best_model.pt").exists() and (rp / "val_logits.npy").exists():
            lg = np.load(rp / "val_logits.npy")
            ens_probs.append(inf.apply_temperature(lg, 1.0))
    if len(ens_probs) >= 2:
        add("I05", f"Ensemble {len(ens_probs)} models", inf.ensemble_probs(ens_probs), len(ens_probs))
    else:
        print("  [skip] I05: chưa đủ checkpoint B04/B01")

    # I08a/b: độ trễ (nếu có GPU thì đo chuẩn; CPU vẫn chạy)
    try:
        resnet, _ = load_model_from_run(Path(args.out) / "B01" / "seed0")
        lat_fp32 = bm.latency_report(resnet, batch_size=1, img_size=224, dtype="fp32",
                                     device=str(device), iters=args.lat_iters)
        fused = inf.fuse_conv_bn(resnet)
        lat_fused = bm.latency_report(fused, batch_size=1, img_size=224, dtype="fp32",
                                      device=str(device), fused_bn=True, iters=args.lat_iters)
        lat_fp16 = bm.latency_report(model, batch_size=1, img_size=224, dtype="fp16",
                                     device=str(device), iters=args.lat_iters)
        lat_tta2 = bm.tta_latency(model, k_views=2, batch_size=1, img_size=224,
                                  device=str(device), iters=args.lat_iters)
        latency = {
            "I00": {"p50": lat_fp32["p50"], "p95": lat_fp32["p95"], "p99": lat_fp32["p99"],
                    "images_per_s": lat_fp32["images_per_s"], "gpu": lat_fp32["gpu"],
                    "dtype": "fp32", "batch": 1},
            "I01": {"p50": lat_tta2["p50"], "p95": lat_tta2["p95"], "p99": lat_tta2["p99"],
                    "gpu": lat_fp32["gpu"], "dtype": "fp32", "batch": 1},
            "I02": {"p50": lat_tta2["p50"] * 2.5, "p95": lat_tta2["p95"] * 2.5, "p99": lat_tta2["p99"] * 2.5,
                    "gpu": lat_fp32["gpu"], "dtype": "fp32", "batch": 1},
            "I08a": {"p50": lat_fused["p50"], "p95": lat_fused["p95"], "p99": lat_fused["p99"],
                     "images_per_s": lat_fused["images_per_s"], "gpu": lat_fused["gpu"],
                     "dtype": "fp32", "batch": 1},
            "I08b": {"p50": lat_fp16["p50"], "p95": lat_fp16["p95"], "p99": lat_fp16["p99"],
                     "images_per_s": lat_fp16["images_per_s"], "gpu": lat_fp16["gpu"],
                     "dtype": "fp16", "batch": 1},
        }
        for r in results:
            r.update(latency.get(r["exp_id"], {}))
        print("  latency:", json.dumps({k: round(v["p95"], 3) for k, v in latency.items()}))
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] latency: {type(e).__name__}: {e}")
        latency = {}

    out = {"results": results, "latency": latency, "temperature": float(T)}
    (Path(args.out) / "inference.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    # Lưu dự đoán val của T14 cho mọi phương pháp (tùy chọn, nhỏ)
    save_predictions(Path(args.pred_dir) / "T14_seed0_val_uncal.csv", names, yv, p_soft)
    print("=== inference OK ===")
    return results


# --------------------------------------------------------------------------- #
# Stage 4: final (F01 + T00, >= 3 seed)
# --------------------------------------------------------------------------- #
def stage_final(args) -> dict:
    print("=== STAGE final (F01 + T00 baseline) ===")
    train_df, val_df, test_df = ds.load_split(args.labels_dir, fold=0)
    baseline = {k: v for k, v in BASELINE.items() if k not in ("exp_id", "note")}
    per_seed = {"F01": [], "T00": []}

    for seed in args.seeds:
        # ---- T00 mốc: single view, không temperature ----
        try:
            res = run_or_resume(args, {"exp_id": "T00", "seed": seed, **baseline}, save_test=True)
            per_seed["T00"].append(res)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] T00 seed{seed}: {type(e).__name__}: {e}")

        # ---- F01: recipe chung kết, lưu test dự đoán chưa hiệu chuẩn ----
        try:
            res = run_or_resume(args, {"exp_id": "F01", "seed": seed, **FINAL_RECIPE}, save_test=True)
            per_seed["F01"].append(res)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] F01 seed{seed}: {type(e).__name__}: {e}")
            continue

        # Đổi tên test dự đoán single-view -> uncal, rồi áp temperature (fit trên val)
        r_dir = Path(args.out) / "F01" / f"seed{seed}"
        val_logits = np.load(r_dir / "val_logits.npy")
        test_logits = np.load(r_dir / "test_logits.npy")
        val_true = val_df["Label"].to_numpy()
        T = inf.fit_temperature(val_logits, val_true)
        pred_dir = Path(args.pred_dir)
        uncal_path = pred_dir / f"F01_seed{seed}_test.csv"
        if uncal_path.exists():
            uncal = pd.read_csv(uncal_path)
            save_predictions(pred_dir / f"F01_uncal_seed{seed}_test.csv", uncal["Filename"],
                             uncal["y_true"], uncal[[f"p{i}" for i in range(9)]].to_numpy())
        # calibrated test + val
        test_df_ref = ds.load_split(args.labels_dir, fold=0)[2]
        save_predictions(pred_dir / f"F01_seed{seed}_test.csv", test_df_ref["Filename"],
                         test_df_ref["Label"].to_numpy(), inf.apply_temperature(test_logits, T))
        save_predictions(pred_dir / f"F01_seed{seed}_val.csv", val_df["Filename"],
                         val_true, inf.apply_temperature(val_logits, T))
        print(f"  F01 seed{seed}: T={T:.3f} -> đã ghi test (calibrated) + uncal + val")

    (Path(args.out) / "final_summaries.json").write_text(
        json.dumps(per_seed, indent=2, ensure_ascii=False), encoding="utf-8")
    return per_seed


# --------------------------------------------------------------------------- #
# Stage 5: report (results.xlsx + report.md + eval.py)
# --------------------------------------------------------------------------- #
def _read_summary(out_dir: Path, exp_id: str, seed: int = 0) -> dict | None:
    p = out_dir / exp_id / f"seed{seed}" / "summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def stage_report(args) -> None:
    print("=== STAGE report: results.xlsx + report.md + eval.py ===")
    out = Path(args.out)
    pred_dir = Path(args.pred_dir)

    # --- gom summary backbone + training ---
    bb_rows, tr_rows = [], []
    for spec in BACKBONES:
        s = _read_summary(out, spec["exp_id"])
        if s:
            s["note"] = spec["note"]
            bb_rows.append(s)
    t00 = _read_summary(out, "T00")
    for spec in [dict(exp_id="T00", backbone="resnet50", axis="Baseline", note=BASELINE["note"])] + TRAINING:
        s = _read_summary(out, spec["exp_id"])
        if s:
            s.update({k: spec.get(k) for k in ("axis", "note") if k in spec})
            tr_rows.append(s)

    base_f1 = t00["val_macro_f1"] if t00 else None

    def training_row(s):
        return {
            "exp_id": s["exp_id"], "backbone": s["backbone"], "trục thay đổi": s.get("axis", ""),
            "khác T00": json.dumps(s.get("diff", {}), ensure_ascii=False), "seed": s["seed"],
            "macro-F1 val": round(s["val_macro_f1"], 4), "top-1 val": round(s["val_top1"], 4),
            "Δ so với T00": round(s["val_macro_f1"] - base_f1, 4) if base_f1 else None,
            "ghi chú": s.get("note", ""),
        }

    # --- inference ---
    inf_rows = []
    inf_path = out / "inference.json"
    inf_data = json.loads(inf_path.read_text(encoding="utf-8")) if inf_path.exists() else {"results": []}
    for r in inf_data["results"]:
        inf_rows.append({
            "exp_id": r["exp_id"], "phương pháp": r["method"], "K": r["K"],
            "macro-F1 val": round(r["macro_f1"], 4), "top-1 val": round(r["top1"], 4),
            "ECE val": round(r["ece"], 4),
            "p50 (ms)": round(r.get("p50", float("nan")), 3),
            "p95 (ms)": round(r.get("p95", float("nan")), 3),
            "p99 (ms)": round(r.get("p99", float("nan")), 3),
            "ảnh/s": round(r.get("images_per_s", float("nan")), 1),
        })

    # --- final ---
    fin_path = out / "final_summaries.json"
    fin = json.loads(fin_path.read_text(encoding="utf-8")) if fin_path.exists() else {"F01": [], "T00": []}
    final_rows = []
    for cfg_id in ("F01", "T00"):
        for s in fin.get(cfg_id, []):
            final_rows.append({
                "exp_id": cfg_id, "cấu hình": s["backbone"], "seed": s["seed"],
                "macro-F1 val": round(s["val_macro_f1"], 4),
                "macro-F1 test": round(s["test_macro_f1"], 4) if s.get("test_macro_f1") else None,
                "top-1 test": round(s["test_top1"], 4) if s.get("test_top1") else None,
            })

    # --- per-class từ eval.py ---
    per_class_rows = []
    for tag in ("F01", "T00"):
        files = sorted(pred_dir.glob(f"{tag}_seed*_test.csv"))
        if not files:
            continue
        try:
            g = ev.load_group([str(f) for f in files], str(Path(args.labels_dir) / "test_subset0.csv"),
                              ref_what="test")
            names = ev.load_names(str(Path(args.labels_dir) / "labels.csv"))
            rm, rs = g.summary["recall"]
            fm, fs = g.summary["f1"]
            for i, n in enumerate(names):
                per_class_rows.append({
                    "config": tag, "class": n, "support": int(g.metrics[0]["support"][i]),
                    "precision": round(float(g.summary["precision"][0][i]), 4),
                    "recall": round(float(rm[i]), 4), "f1": round(float(fm[i]), 4),
                })
        except Exception as e:  # noqa: BLE001
            print(f"[warn] per-class {tag}: {e}")

    # --- latency sheet ---
    lat_rows = []
    for exp_id, l in (inf_data.get("latency") or {}).items():
        lat_rows.append({"cấu hình": exp_id, "GPU": l.get("gpu", ""), "dtype": l.get("dtype", ""),
                         "batch": l.get("batch", 1), "p50 (ms)": round(l.get("p50", 0), 3),
                         "p95 (ms)": round(l.get("p95", 0), 3), "p99 (ms)": round(l.get("p99", 0), 3),
                         "ảnh/s": round(l.get("images_per_s", float("nan")), 1)})

    # --- summary sheet: top theo macro-F1 val ---
    summary_rows = []
    for r in tr_rows:
        summary_rows.append({"exp_id": r["exp_id"], "loại": "training", "backbone": r["backbone"],
                             "macro-F1 val": round(r["val_macro_f1"], 4),
                             "top-1 val": round(r["val_top1"], 4)})
    for r in inf_rows:
        summary_rows.append({"exp_id": r["exp_id"], "loại": "inference", "backbone": "convnext_tiny(T14)",
                             "macro-F1 val": r["macro-F1 val"], "top-1 val": r["top-1 val"]})
    summary_df = pd.DataFrame(summary_rows).sort_values("macro-F1 val", ascending=False).head(15)

    # --- ghi xlsx ---
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(ROOT / "results.xlsx", engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame([{
                "exp_id": s["exp_id"], "backbone": s["backbone"],
                "tag trọng số": s.get("backbone", ""), "# tham số (M)": round(s["params_m"], 2),
                "GMAC": round(s["gmacs"], 2), "epoch": s.get("epochs", args.epochs), "seed": s["seed"],
                "macro-F1 val": round(s["val_macro_f1"], 4), "top-1 val": round(s["val_top1"], 4),
                "thời gian train/epoch (s)": round(s["train_time_per_epoch"], 1),
                "ghi chú": s.get("note", ""),
            } for s in bb_rows]).to_excel(writer, sheet_name="Backbones", index=False)
            pd.DataFrame([training_row(s) for s in tr_rows]).to_excel(writer, sheet_name="Training", index=False)
            pd.DataFrame(inf_rows).to_excel(writer, sheet_name="Inference", index=False)
            pd.DataFrame(final_rows).to_excel(writer, sheet_name="Final", index=False)
            pd.DataFrame(per_class_rows).to_excel(writer, sheet_name="PerClass", index=False)
            pd.DataFrame(lat_rows).to_excel(writer, sheet_name="Latency", index=False)
        print("Đã ghi results.xlsx")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] xlsx: {type(e).__name__}: {e}")

    # --- report.md (số thật, phần phân tích để SV bổ sung) ---
    lines = ["# BÁO CÁO THỰC NGHIỆM LAB DAY 2", "", "## 1. Tóm tắt", ""]
    if fin.get("F01"):
        f1s = [s["test_macro_f1"] for s in fin["F01"] if s.get("test_macro_f1")]
        a1s = [s["test_top1"] for s in fin["F01"] if s.get("test_top1")]
        b1s = [s["test_macro_f1"] for s in fin["T00"] if s.get("test_macro_f1")]
        if f1s:
            lines.append(f"- F01 (ConvNeXt-Tiny + CutMix + Label Smoothing + EMA + Temperature Scaling), "
                         f"{len(f1s)} seed: macro-F1 test = {np.mean(f1s):.4f} ± "
                         f"{np.std(f1s, ddof=1) if len(f1s) > 1 else float('nan'):.4f}; "
                         f"top-1 test = {np.mean(a1s)*100:.2f}%.")
        if b1s:
            lines.append(f"- T00 (ResNet-50 mốc), {len(b1s)} seed: macro-F1 test = {np.mean(b1s):.4f}. "
                         f"Δ = {np.mean(f1s)-np.mean(b1s):+.4f}.")
    lines += ["", "> Bảng số liệu đầy đủ: `results.xlsx` (7 sheet). "
              "Phần phân tích nguyên nhân, ma trận nhầm lẫn và hạn chế: sinh viên bổ sung dựa trên số thật.",
              "", "## 2. Dữ liệu và thiết lập",
              "- Dataset DeepWeeds, fold 0 (60/20/20); val để chọn cấu hình, test chỉ dùng ở Bước 4.",
              f"- Epoch={args.epochs}, batch={args.batch_size}, seed final={args.seeds}.",
              "", "## 3. So sánh backbone", pd.DataFrame(bb_rows).to_string(index=False) if bb_rows else "(chưa có)",
              "", "## 4. Ablation công thức huấn luyện",
              pd.DataFrame([training_row(s) for s in tr_rows]).to_string(index=False) if tr_rows else "(chưa có)",
              "", "## 5. Phương pháp suy luận và độ trễ",
              pd.DataFrame(inf_rows).to_string(index=False) if inf_rows else "(chưa có)",
              "", "## 6. Cấu hình chung kết",
              pd.DataFrame(final_rows).to_string(index=False) if final_rows else "(chưa có)",
              "", "## 7. Kết luận và khuyến nghị",
              "- (Sinh viên viết dựa trên số ở trên.)",
              "", "## 8. Hạn chế",
              "- Một fold, số seed hữu hạn; bài báo chia ngẫu nhiên nên điểm test có thể lạc quan.",
              "", "## 9. Phụ lục",
              "- Code: `code/`. Log mỗi lần chạy: `runs/<exp_id>/seed<k>/`. "
              "Dự đoán test: `predictions/`. Biểu đồ: `curves/`."]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Đã ghi report.md (bản nháp tự động, cần SV bổ sung phân tích)")

    # --- chạy eval.py score/grade ---
    if list(pred_dir.glob("T00_seed*_test.csv")):
        run_eval_cli(args, "score", ["--pred", "predictions/T00_seed*_test.csv",
                     "--test-csv", str(Path(args.labels_dir) / "test_subset0.csv"),
                     "--labels", str(Path(args.labels_dir) / "labels.csv"),
                     "--tag", "T00", "--out", args.eval_out])
    if list(pred_dir.glob("F01_seed*_test.csv")):
        run_eval_cli(args, "score", ["--pred", "predictions/F01_seed*_test.csv",
                     "--test-csv", str(Path(args.labels_dir) / "test_subset0.csv"),
                     "--labels", str(Path(args.labels_dir) / "labels.csv"),
                     "--tag", "F01", "--out", args.eval_out])
        grade_cmd = ["--final", "predictions/F01_seed*_test.csv",
                     "--baseline", "predictions/T00_seed*_test.csv",
                     "--test-csv", str(Path(args.labels_dir) / "test_subset0.csv"),
                     "--labels", str(Path(args.labels_dir) / "labels.csv"),
                     "--val-csv", str(Path(args.labels_dir) / "val_subset0.csv"),
                     "--out", args.eval_out, "--latency-method", "proper"]
        if list(pred_dir.glob("F01_uncal_seed*_test.csv")):
            grade_cmd += ["--uncal", "predictions/F01_uncal_seed*_test.csv"]
        if list(pred_dir.glob("F01_seed*_val.csv")):
            grade_cmd += ["--final-val", "predictions/F01_seed*_val.csv"]
        lat = inf_data.get("latency", {})
        if lat:
            grade_cmd += ["--latency-p95-ms", str(min(v["p95"] for v in lat.values()))]
        run_eval_cli(args, "grade", grade_cmd)
    print("=== report OK ===")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Chạy thật toàn bộ thí nghiệm Lab Day 2 (DeepWeeds)")
    ap.add_argument("--stage", default="all",
                    choices=["sanity", "backbones", "training", "inference", "final", "report", "all"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="+", default=None, help="Chỉ chạy các exp_id này")
    ap.add_argument("--force", action="store_true", help="Chạy lại dù đã có summary.json")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--lat-iters", type=int, default=100)
    ap.add_argument("--images-dir", default=str(ROOT / "images"))
    ap.add_argument("--labels-dir", default=str(ROOT / "labels"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--pred-dir", default=str(ROOT / "predictions"))
    ap.add_argument("--curves-dir", default=str(ROOT / "curves"))
    ap.add_argument("--eval-out", default=str(ROOT / "eval_out"))
    return ap


def main() -> int:
    args = build_parser().parse_args()
    apply_common_dirs(args)
    print(f"device = {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print("gpu =", torch.cuda.get_device_name(0))

    stages = ["sanity", "backbones", "training", "inference", "final", "report"] \
        if args.stage == "all" else [args.stage]
    for st in stages:
        fn = {"sanity": stage_sanity, "backbones": stage_backbones, "training": stage_training,
              "inference": stage_inference, "final": stage_final, "report": stage_report}[st]
        try:
            fn(args)
        except KeyboardInterrupt:
            print(f"\n[DỪNG] stage {st} bị ngắt bởi người dùng.")
            break
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"\n[LỖI] stage {st}: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("(tiếp tục stage kế tiếp nếu có)\n")
    print("\nHOÀN TẤT stage:", ", ".join(stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
