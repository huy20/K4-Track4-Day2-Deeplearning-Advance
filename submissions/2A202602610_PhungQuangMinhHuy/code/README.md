# Lab Day 2 — DeepWeeds: Hướng dẫn tái lập

> **Môn**: Deep Learning Nâng cao — Day 2: Backbone, công thức huấn luyện & suy luận
> **Dataset**: DeepWeeds, **fold 0** chia sẵn (60/20/20) — dùng đúng CSV gốc, không chia lại.

Repo này chạy **thật** trên Google Colab/Kaggle. Mọi con số trong `results.xlsx` và
`report.md` sinh ra từ log huấn luyện trong `runs/<exp_id>/seed<k>/` (config, history,
checkpoint, summary) và từ file `predictions/` do `eval.py` tính lại.

> **Quan trọng (trung thực học thuật — RUBRIC P2):** không bịa số, không sinh logit giả.
> Nếu chưa chạy, các file kết quả **chưa tồn tại**; chúng chỉ được tạo khi bạn chạy notebook/driver.

---

## 1. Môi trường & phiên bản thư viện

- Python ≥ 3.9 (Colab/Kaggle dùng sẵn Python 3.10+).
- PyTorch ≥ 2.0 + CUDA (Colab T4/V100/A100; Kaggle GPU).
- Cần cài: `timm`, `thop`, `openpyxl`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`.

```bash
pip install "timm>=0.9.12" thop openpyxl scipy matplotlib pandas scikit-learn
```

Phiên bản đã dùng khi phát triển: `timm 1.0.x`, `torch>=2.1`, `scipy`, `openpyxl`.
Ghi lại phiên bản thật trong báo cáo (notebook in ra `torch.__version__`, `timm.__version__`).

---

## 2. Dữ liệu

- Ảnh: Zenodo `images.zip` (~490 MB):
  `https://zenodo.org/records/7939060/files/images.zip?download=1`
  **MD5** = `b7b30f96d466fba86016aa5a26606e0f`.
- Nhãn/fold: `labels/labels.csv`, `labels/train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`
  (đã có sẵn trong repo; lấy từ github.com/AlexOlsen/DeepWeeds).

Cấu trúc sau khi giải nén:

```
images/            # 17.509 ảnh .jpg (256x256)
labels/
  labels.csv
  train_subset0.csv  val_subset0.csv  test_subset0.csv
```

Notebook tự tải và kiểm tra MD5; hoặc tải tay rồi giải nén sao cho ảnh nằm trong `images/`.

---

## 3. Chạy nhanh trên Colab/Kaggle

1. Mở `code/lab_day2.ipynb` (Colab/Kaggle), chọn Runtime **GPU**.
2. Đặt thư mục project trong `MyDrive` (hoặc upload zip giải nén vào `/content`), rồi chạy lần lượt các cell:
   - Cài thư viện → mount Drive → tìm project.
   - Tải dataset + kiểm tra MD5.
   - `selfcheck` + stage `sanity` (EDA, kiểm tra split, loss ≈ ln 9, overfit 1 batch).
   - (Tùy chọn) chạy thử 1 backbone 1 epoch.
   - Chạy FULL: backbones → training → inference → final (3 seed) → report.
   - Xem `results.xlsx`, `eval_out/grade_I.json`; lưu sản phẩm ra Drive.

> Mỗi stage lưu `summary.json` nên **chạy lại sẽ resume** (bỏ qua cái đã xong). Dùng `--force` để chạy lại.

---

## 4. Chạy bằng CLI (không cần notebook)

Từ thư mục gốc repo:

```bash
# Kiểm tra code dễ sai (focal gamma=0 == CE, fuse BN, temperature, TTA)
python code/selfcheck.py

# EDA + kiểm tra split + sanity pipeline
python code/run_experiments.py --stage sanity

# B01..B06 (>=5 backbone, cùng công thức nền, seed 0)
python code/run_experiments.py --stage backbones --epochs 12

# T00 (mốc) và T01..T14 (ablation công thức huấn luyện, seed 0)
python code/run_experiments.py --stage training --epochs 12

# I00..I08 (TTA, FixRes, ensemble, EMA, temperature scaling, fuse BN, FP16) + đo độ trễ
python code/run_experiments.py --stage inference

# Chung kết F01 + mốc T00, >=3 seed, ghi predictions/
python code/run_experiments.py --stage final --epochs 12 --seeds 0 1 2

# results.xlsx + report.md + chạy eval.py score/grade
python code/run_experiments.py --stage report --epochs 12 --seeds 0 1 2

# Hoặc chạy tất cả
python code/run_experiments.py --stage all --epochs 12 --seeds 0 1 2
```

Chạy nhanh để thử: `--epochs 2 --only B05 T05 F01` (ghi rõ trong báo cáo nếu giảm epoch).

---

## 5. Danh sách thí nghiệm

| Nhóm | exp_id | Nội dung |
|---|---|---|
| Backbone | B01–B06 | ResNet-50, ResNeXt-50-32x4d, ConvNeXt-Tiny, Swin-Tiny, MobileNetV3-Large, EfficientNet-B0 (công thức nền T00, seed 0) |
| Công thức | T00 | Mốc: ResNet-50 + công thức nền + 1-view |
| Công thức | T01–T14 | Ablation một biến: khởi tạo (scratch/frozen), augmentation (color/randaug/cutmix/mixup), loss (LS/focal/class-weighted), sampler, optimizer/LR, EMA; T14 = kết hợp tốt nhất |
| Suy luận | I00–I08 | 1-view, TTA lật, TTA 5-crop, gộp prob/logit, FixRes 256/288, ensemble, EMA, temperature scaling, fuse BN, FP16 + độ trễ p50/p95/p99 |
| Chung kết | F01 | ConvNeXt-Tiny + CutMix + Label Smoothing + EMA + temperature scaling, ≥3 seed |

**Seed**: sàng lọc (B, T, I) dùng seed 0; chung kết (F01, T00 baseline) dùng 3 seed `0, 1, 2`.
Seed không đổi cách chia dữ liệu (quy tắc S5).

---

## 6. Sản phẩm sinh ra

```
results.xlsx        # 7 sheet: Summary, Backbones, Training, Inference, Final, PerClass, Latency
report.md           # báo cáo (bản nháp tự động từ số thật; bổ sung phân tích/lỗi)
curves/*.png        # biểu đồ loss/metric theo epoch cho từng exp_id huấn luyện
predictions/        # dự đoán test/val đúng định dạng eval.py (F01, T00, F01_uncal, F01_val)
runs/<exp>/seed<k>/ # config.json, history.csv, best_model.pt, val/test_logits.npy, summary.json
eval_out/           # JSON/CSV do eval.py sinh (score/grade phần I)
```

Chỉ commit file nhỏ: `results.xlsx`, `report.md`, `curves/`, `predictions/`, `code/`.
**Không commit** dataset và checkpoint (`*.pt`) — đã có trong `.gitignore`.

---

## 7. Kiểm tra trước khi nộp

```bash
# Chỉ số chi tiết theo seed cho F01 và mốc T00
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag F01 --out eval_out
python eval.py score --pred "predictions/T00_seed*_test.csv" \
    --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag T00 --out eval_out

# Tự chấm Rubric phần I (I1-I5)
python eval.py grade \
    --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01_uncal_seed*_test.csv" \
    --final-val "predictions/F01_seed*_val.csv" \
    --latency-p95-ms <p95_that_do_duoc> --latency-method proper \
    --test-csv labels/test_subset0.csv --val-csv labels/val_subset0.csv \
    --labels labels/labels.csv --out eval_out
```

Checklist (RUBRIC mục 5): đủ 5 backbone, ≥3 trục công thức, ≥4 phương pháp suy luận,
F01+T00 ≥3 seed, test chạy 1 lần/seed, mọi số khớp `eval.py`, không sửa `eval.py`,
mỗi thí nghiệm có ảnh `curves/`, `exp_id` khớp giữa biểu đồ – xlsx – predictions.

---

## 8. Ghi chú kỹ thuật

- Công thức nền (GUIDE 1.4): pretrained ImageNet, AdamW 3 nhóm tham số
  (backbone `1e-4`, head `1e-3`, weight decay 0.05, không áp dụng cho norm/bias),
  warmup 1 epoch + cosine, batch 64, AMP, chọn checkpoint theo macro-F1 val.
- Val/test: resize + center-crop (224), chuẩn hoá ImageNet; không augmentation ngẫu nhiên.
- Temperature scaling khớp **một** T trên val, áp dụng sang test.
- Độ trễ đo đúng: warmup ≥10, `torch.cuda.synchronize()`, ≥50 lần, báo p50/p95/p99 (xem `code/benchmark.py`).
