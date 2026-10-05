# BÁO CÁO THỰC NGHIỆM LAB DAY 2
## Backbone, công thức huấn luyện và suy luận trên DeepWeeds

> **Bài nộp cá nhân** — MSSV: `2A202602610` · Họ tên: `Phùng Quang Minh Huy`
> **Notebook tái lập (Kaggle)**: `https://www.kaggle.com/code/huy1805/notebook194359e46c`
> **Repo GitHub**: `https://github.com/huy20/K4-Track4-Day2-Deeplearning-Advance`

---

## 1. Tóm tắt (Executive Summary)

Thực nghiệm trên **DeepWeeds**, **fold 0** chia sẵn (60/20/20), đánh giá bằng **macro-F1** (9 lớp)
và **top-1 accuracy**. Do hạn chế ngân sách GPU, báo cáo dùng **8 epoch/lần** (thay vì 12) và
khảo sát **5 backbone** (B01–B05) và **4 cấu hình công thức huấn luyện** (T05, T07, T13, T14)
so với mốc `T00`. Cấu hình chung kết **F01 = ConvNeXt-Tiny + CutMix(α=1.0) + Label Smoothing(ε=0.1)
+ Weight EMA(0.999) + Temperature Scaling**.

Kết quả chính trên **test** qua **3 seed độc lập (0,1,2)**:

| Chỉ số | F01 (chung kết) | T00 (mốc ResNet-50, 2 seed) |
|---|---|---|
| Macro-F1 | **0.9619 ± 0.0014** | 0.7618 |
| Top-1 accuracy | **96.87% ± 0.20%** | 82.86% |
| Balanced accuracy | 0.9571 ± 0.0070 | 0.7249 |
| ECE (sau hiệu chuẩn) | **0.0080** (trước 0.1141) | 0.0194 |

**Kết luận ngắn:** yếu tố đóng góp lớn nhất là **kiến trúc backbone** (ConvNeXt-Tiny vượt trội
hẳn ResNet-50/ResNeXt/MobileNet trên cùng công thức nền); ở 8 epoch, các thay đổi công thức
huấn luyện (CutMix / Label Smoothing / EMA) **không vượt** nền ConvNeXt ngoài mức nhiễu.
Temperature scaling giảm ECE ~14 lần mà không đổi độ chính xác.

---

## 2. Dữ liệu và Thiết lập Thực nghiệm

### 2.1 Dataset và phân bố lớp (EDA)
DeepWeeds: 17.509 ảnh RGB 256×256, 9 lớp (8 loài cỏ dại + `Negative`), mất cân bằng nặng.
Dùng đúng `train_subset0.csv` / `val_subset0.csv` / `test_subset0.csv` (không sửa, không chia lại).

| Lớp | Train | Val | Test | Tổng | Tỷ lệ |
|---|---:|---:|---:|---:|---:|
| Chinee apple | 675 | 225 | 226 | 1.126 | 6.43% |
| Lantana | 637 | 213 | 213 | 1.063 | 6.07% |
| Parkinsonia | 618 | 206 | 207 | 1.031 | 5.89% |
| Parthenium | 613 | 204 | 205 | 1.022 | 5.84% |
| Prickly acacia | 637 | 212 | 213 | 1.062 | 6.07% |
| Rubber vine | 605 | 202 | 202 | 1.009 | 5.76% |
| Siam weed | 644 | 215 | 215 | 1.074 | 6.13% |
| Snake weed | 609 | 203 | 204 | 1.016 | 5.80% |
| **Negative** | **5.463** | **1.821** | **1.822** | **9.106** | **52.01%** |
| **Tổng** | **10.501** | **3.501** | **3.507** | **17.509** | **100%** |

**Kiểm tra bắt buộc (đã chạy):** giao từng cặp tập theo `Filename` = **0** (train∩val, train∩test,
val∩test); hợp 3 tập = **17.509**; tỷ lệ 59.97/19.99/20.03%.

### 2.2 Công thức nền (T00, dùng chung cho mọi backbone)
- Pretrained ImageNet, thay head 9 lớp, tinh chỉnh toàn bộ.
- Train: `RandomResizedCrop(224, scale=(0.8,1.0))` + `RandomHorizontalFlip` + Normalize ImageNet.
  Val/Test: `Resize(256)` + `CenterCrop(224)` + Normalize (không augmentation ngẫu nhiên).
- AdamW, 3 nhóm tham số: backbone LR `1e-4`, head LR `1e-3`, weight decay `0.05` (không áp dụng cho norm/bias).
- Warmup 1 epoch + cosine decay; batch 64; AMP; chọn checkpoint theo macro-F1 **val**.

### 2.3 Phần cứng, thư viện, seed
- **GPU**: Kaggle **Tesla T4**; `torch 2.11.0+cu128`, `timm 1.0.29`.
- **Epoch**: 8 (giảm từ 12 do ngân sách GPU — đã ghi rõ).
- **Seed**: sàng lọc (B*, T*) dùng seed 0; chung kết (F01, T00) dùng seed 0,1,2.

### 2.4 Kiểm tra pipeline (trước khi chạy thật)
- `selfcheck`: focal `γ=0` ≡ CE, LabelSmoothing(0) ≡ CE, CutMix/Mixup hợp lệ,
  `fuse_conv_bn` khớp đầu ra ResNet-50 (max|Δ|=2.86e-5), temperature scaling, TTA views — **tất cả PASS**.
- **Loss ban đầu** với head mới = **2.1704** ≈ ln 9 = 2.1972.
- **Overfit 1 batch** (n=32): loss **2.2123 → 0.0002**.

---

## 3. Kết quả So sánh Backbone (Bước 1)

Cùng công thức nền T00, seed 0, 8 epoch.

| Mã | Backbone | Tham số (M) | GMAC | Macro-F1 val | Top-1 val | s/epoch | Nhận xét |
|:--:|---|:--:|:--:|:--:|:--:|:--:|---|
| B01 | ResNet-50 | 23.53 | 4.12 | 0.7657 | 83.23% | 45.5 | Rất yếu ở 8 epoch |
| B02 | ResNeXt-50-32x4d | 23.00 | 4.24 | 0.8160 | 85.83% | 61.1 | Nhỉnh hơn ResNet-50 |
| **B03** | **ConvNeXt-Tiny** | 27.83 | 4.46 | **0.9689** | **97.51%** | 53.9 | **Tốt nhất, hội tụ nhanh** |
| B04 | Swin-Tiny | 27.53 | 4.50 | 0.9509 | 96.34% | 68.5 | Transformer tốt nhưng chậm hơn |
| B05 | MobileNetV3-Large | 4.21 | 0.23 | 0.7273 | 80.01% | 22.5 | Nhẹ/nhanh nhất, F1 thấp |

**Nhận xét.** Trên DeepWeeds, ConvNeXt-Tiny vượt các họ CNN cổ điển rất rõ (+0.20 macro-F1 so với
ResNet-50) nhờ thiết kế hiện đại (depthwise 7×7, LayerNorm, GELU) và trọng số tiền huấn luyện mạnh;
Swin-Tiny bám sát ConvNeXt nhưng chậm hơn ~27%. **Lưu ý:** khoảng cách lớn này chủ yếu do ResNet-50
**hội tụ rất kém trong 8 epoch** (và có thể do tag trọng số mặc định), không hẳn do khác biệt kiến trúc
thuần — với ~100 epoch như bài báo, ResNet-50 đạt ~95.7%. Vì vậy kết luận "ConvNeXt tốt hơn" chỉ
chắc chắn trong điều kiện ngắn epoch của lab.

---

## 4. Kết quả Khảo sát Công thức Huấn luyện (Bước 2)

Thực hiện trên **ConvNeXt-Tiny** (như B03), mỗi thí nghiệm đổi **đúng một yếu tố** so với nền.
**Nền có kiểm soát để so sánh là B03 (ConvNeXt-Tiny + công thức nền, macro-F1 val = 0.9689)**.
Mốc `T00` (ResNet-50) chỉ dùng để tính cải thiện tổng thể ở Bước 4.

| Mã | Trục | Thay đổi | Macro-F1 val | Top-1 val | Δ vs nền B03 | Δ vs mốc T00 |
|:--:|---|---|:--:|:--:|:--:|:--:|
| T00 | Mốc | ResNet-50 + nền + 1-view | 0.7657 | 83.23% | −0.2032 | 0.0000 |
| **B03** | Nền | ConvNeXt-Tiny + nền | **0.9689** | 97.51% | 0.0000 | +0.2032 |
| T05 | B. Augmentation | + CutMix(α=1.0) | 0.9687 | 97.63% | −0.0002 | +0.2030 |
| T07 | C. Loss | + Label Smoothing(ε=0.1) | 0.9614 | 96.97% | −0.0075 | +0.1956 |
| T13 | F. Regularization | + Weight EMA(0.999) | 0.9556 | 96.32% | −0.0133 | +0.1898 |
| T14 | Kết hợp | CutMix + LS + EMA | 0.9612 | 96.94% | −0.0077 | +0.1955 |

**Phân tích trung thực (quan trọng).**
- **Không có thay đổi công thức nào vượt nền ConvNeXt ở 8 epoch.** T05 gần như bằng nền
  (−0.0002), T07/T13/T14 thấp hơn nền. Với **1 seed** cho mỗi cấu hình, các chênh lệch này
  **không thể kết luận** (cần ≥3 seed và độ lệch chuẩn để tách khỏi nhiễu — xem GUIDE N4).
- Không quan sát được **hiệu ứng cộng dồn**: T14 (0.9612) không cao hơn các thành phần đơn lẻ.
- Cải thiện +0.20 macro-F1 so với mốc T00 **chủ yếu đến từ backbone**, không phải từ công thức:
  mốc T00 dùng ResNet-50 vốn yếu ở 8 epoch.

> Vì vậy, kết luận "công thức X giúp" **không** được khẳng định trong báo cáo này; đây là một
> hạn chế của ngân sách tính toán ngắn, không phải bằng chứng công thức vô ích.

---

## 5. Kết quả Phương pháp Suy luận (Bước 3) — *chưa hoàn thành*

**Chưa chạy được** do hết ngân sách GPU sau khi huấn luyện. Vì vậy bảng Inference và Latency
trong `results.xlsx` để trống. Các nội dung dự kiến (theo GUIDE mục 4):

| Mã | Phương pháp | Trạng thái |
|---|---|---|
| I00 | 1-view (CenterCrop 224) | benchmark của mốc suy luận |
| I01 | TTA lật ngang (K=2) | chưa chạy |
| I02 | TTA 5-crop (K=5) | chưa chạy |
| I03 | Gộp prob vs logit | chưa chạy |
| I04 | FixRes test 256/288 | chưa chạy |
| I05 | Ensemble ConvNeXt+Swin+ResNet | chưa chạy |
| I06 | Trọng số EMA | **đã dùng trong T14/F01** (miễn phí) |
| I07 | Temperature Scaling + ECE | **đã dùng trong F01** |
| I08 | Gộp Conv-BN / FP16 | chưa chạy |

Phần **temperature scaling** đã được áp dụng trong cấu hình chung kết: T khớp trên **val**,
áp dụng sang test, giảm **ECE từ 0.1141 → 0.0080** (macro-F1 không đổi vì T không đổi argmax).

**Lệnh chạy lại phần suy luận trên GPU:**
```bash
python code/run_experiments.py --stage inference
```
Mọi số liệu hiệu chuẩn/độ trễ nên được bổ sung vào `results.xlsx` và mục này sau khi chạy.

---

## 6. Cấu hình Chung kết và Kết quả Test (Bước 4)

### 6.1 Kết quả qua 3 seed
**F01** = ConvNeXt-Tiny + CutMix(α=1.0) + Label Smoothing(ε=0.1) + EMA(0.999) + Temperature Scaling.
Test chạy **đúng một lần cho mỗi seed**, trên toàn bộ 3.507 ảnh test.

| Cấu hình | Seed | Macro-F1 val | Macro-F1 test | Top-1 test |
|---|:--:|:--:|:--:|:--:|
| F01 | 0 | 0.9612 | 0.9620 | 96.83% |
| F01 | 1 | 0.9608 | 0.9605 | 96.69% |
| F01 | 2 | 0.9631 | 0.9633 | 97.09% |
| **F01 mean ± std** | — | **0.9617** | **0.9619 ± 0.0014** | **96.87% ± 0.20%** |
| T00 (mốc) | 1 | 0.7578 | 0.7633 | 83.03% |
| T00 (mốc) | 2 | 0.7627 | 0.7603 | 82.69% |
| **T00 mean (2 seed)** | — | **0.7603** | **0.7618** | **82.86%** |

- **Cải thiện so với mốc:** Δ macro-F1 = **+0.2001**, Δ top-1 = **+14.0 điểm %**.
  (Lưu ý: mốc T00 chỉ có **2 seed** có file test do seed 0 thiếu dự đoán test; cải thiện lớn chủ yếu do backbone.)
- **Ổn định val↔test:** macro-F1 val 0.9617 vs test 0.9619 → lệch **0.0002** (rất ổn định).
- **Hiệu chuẩn:** ECE test trước temperature = 0.1141 → sau = **0.0080**.

### 6.2 Kết quả theo lớp trên test (F01, trung bình 3 seed)

| Lớp | Support | Precision | Recall | F1 |
|---|:--:|:--:|:--:|:--:|
| Chinee apple | 226 | 0.975 | **0.901** | 0.936 |
| Lantana | 213 | 0.961 | 0.964 | 0.962 |
| Parkinsonia | 207 | 0.985 | 0.974 | 0.980 |
| Parthenium | 205 | 0.986 | 0.946 | 0.966 |
| Prickly acacia | 213 | 0.923 | 0.970 | 0.946 |
| Rubber vine | 202 | 0.983 | 0.957 | 0.970 |
| Siam weed | 215 | 0.980 | 0.969 | 0.974 |
| Snake weed | 204 | 0.943 | **0.949** | 0.946 |
| Negative | 1.822 | 0.971 | 0.983 | 0.977 |

So với mốc T00, hai lớp khó cải thiện vượt bậc: **Chinee apple recall 0.392 → 0.901**,
**Snake weed recall 0.664 → 0.949**.

### 6.3 Ma trận nhầm lẫn và phân tích lỗi (tổng 3 seed, F01)

Nhầm lẫn chính (hàng = nhãn thật):
- **Chinee apple**: 611 đúng / 678; **48 → Negative**, 14 → Snake weed. Đây là lớp khó nhất (recall 90.1%).
- **Snake weed**: 581 đúng / 612; **18 → Negative**, 8 → Lantana, 4 → Chinee apple.
- **Negative**: 5.372 đúng / 5.454; lỗi chủ yếu **36 → Prickly acacia** (báo động giả, tỷ lệ ~0.7%).
- **Parkinsonia**: 605 đúng; 8 → Prickly acacia (giống nhầm lẫn nêu trong bài báo gốc).

**Giả thuyết.** Chinee apple và Snake weed thường bị đoán thành **Negative** ở ảnh có nền đất khô,
cây nhỏ hoặc bị che khuất — phù hợp với việc bài báo gốc cũng ghi nhận hai lớp này khó nhất.
Hướng khắc phục: nhiều epoch hơn, augmentation hợp lý, hoặc loss tập trung lớp hiếm.

---

## 7. Kết luận và Khuyến nghị

1. **Cấu hình tốt nhất:** **F01** (ConvNeXt-Tiny + CutMix + Label Smoothing + EMA + Temperature Scaling),
   test macro-F1 **0.9619 ± 0.0014**, top-1 **96.87%**, ECE 0.0080.
2. **Mức cải thiện so với mốc:** Δ macro-F1 = **+0.2001** (lớn hơn nhiều so với độ lệch chuẩn 0.0014),
   nhưng phần lớn nhờ **backbone** ConvNeXt thay vì công thức huấn luyện ở 8 epoch.
3. **Yếu tố đóng góp:** (i) **Backbone** đóng góp lớn nhất (ConvNeXt ≫ ResNet-50/ResNeXt/MobileNet);
   (ii) **Công thức** chưa chứng minh được đóng góp ở 8 epoch (trong nhiễu);
   (iii) **Suy luận** cần bổ sung số đo (hiện mới có temperature scaling giảm ECE).
4. **Khuyến nghị triển khai:** ConvNeXt-Tiny đạt độ chính xác cao; để triển khai thời gian thực cần
   đo độ trễ p50/p95/p99 trên GPU (Bước 3) và cân nhắc FP16/gộp BN; MobileNetV3-Large là lựa chọn
   nhẹ nếu phần cứng yếu nhưng hy sinh ~24 điểm % macro-F1.

---

## 8. Hạn chế và Hướng tiếp theo

- **8 epoch** (giảm so với 12) — chưa đủ để kết luận về công thức huấn luyện; ResNet-50 đặc biệt underfit.
- **Ablation 1 seed** mỗi cấu hình → chưa tách được tín hiệu khỏi nhiễu.
- **Mốc T00 chỉ 2 seed** có dự đoán test (seed 0 thiếu file test) — nên chạy lại T00 seed 0.
- **Chưa chạy Bước 3 (suy luận)**: thiếu TTA/FixRes/ensemble/gộp BN/FP16 và độ trễ p50/p95/p99.
- **Một fold** và chia ngẫu nhiên (không theo địa điểm) → điểm test có thể lạc quan so với triển khai thực địa.
- Hướng tiếp theo: đủ 12–15 epoch, ≥3 seed cho ablation, đủ 6 backbone, hoàn thiện Bước 3 trên GPU,
  thử nhiều fold và chưng cất sang mạng nhẹ.

---

## 9. Phụ lục — Danh mục thí nghiệm và tái lập

- **Cấu hình nền:** xem mục 2.2. **Code:** `code/` (`dataset.py`, `model.py`, `losses.py`, `train.py`,
  `inference.py`, `benchmark.py`, `run_experiments.py`, `selfcheck.py`, `lab_day2.ipynb`).
- **Log mỗi lần chạy:** `runs/<exp_id>/seed<k>/` (`config.json`, `history.csv`, `summary.json`, `val_logits.npy`).
- **Dự đoán test/val:** `predictions/` (đúng định dạng `eval.py`).
- **Chỉ số chi tiết:** `eval_out/` (`F01_*`, `T00_*`, `grade_I.json`).
- **Biểu đồ:** `curves/`.
- **Lệnh tái lập (Kaggle GPU):**
  ```bash
  python code/selfcheck.py
  python code/run_experiments.py --stage sanity
  python code/run_experiments.py --stage backbones --epochs 8 --only B01 B02 B03 B04 B05
  python code/run_experiments.py --stage training  --epochs 8 --only T05 T07 T13 T14
  python code/run_experiments.py --stage final     --epochs 8 --seeds 0 1 2
  python code/run_experiments.py --stage report    --epochs 8 --seeds 0 1 2
  # (chưa chạy) python code/run_experiments.py --stage inference
  ```
- **Tự chấm phần I (`eval.py grade`):** I1 7/7 · I2 5/5 · I3 4/4 · I4a 1/1 · I4b 1/1 · **I5 chưa chấm**
  (thiếu số đo độ trễ) → **18/18 ý đã chấm**.
