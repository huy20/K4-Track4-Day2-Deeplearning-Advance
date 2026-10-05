# Lab Day 2 — DeepWeeds (Bài nộp)

- **MSSV**: `2A202602610`
- **Họ tên**: `Phùng Quang Minh Huy`
- **Repo GitHub**: https://github.com/huy20/K4-Track4-Day2-Deeplearning-Advance
- **Notebook chạy lại**: https://www.kaggle.com/code/huy1805/notebook194359e46c
- **Báo cáo đầy đủ**: [`report.md`](report.md)

## Nội dung bài nộp

```
2A202602610_PhungQuangMinhHuy/
├── README.md          # file này
├── report.md          # báo cáo 9 phần (số liệu thật)
├── results.xlsx       # 7 sheet: Summary, Backbones, Training, Inference, Final, PerClass, Latency
├── curves/            # biểu đồ train/val theo epoch cho từng exp_id
├── predictions/       # dự đoán test/val đúng định dạng eval.py
├── eval_out/          # chỉ số chi tiết + grade_I.json do eval.py sinh
├── runs/              # log mỗi lần chạy (config, history, summary, val_logits)
└── code/              # toàn bộ mã nguồn
```

## Môi trường

- **GPU**: Kaggle Tesla T4; `torch 2.11.0+cu128`, `timm 1.0.29`.
- Cài đặt: `pip install "timm>=0.9.12" thop openpyxl scipy matplotlib pandas scikit-learn`
- **Dataset**: DeepWeeds, fold 0 chia sẵn. Ảnh tải từ Zenodo
  (`https://zenodo.org/records/7939060/files/images.zip?download=1`, MD5 `b7b30f96d466fba86016aa5a26606e0f`).
  Nhãn: `labels/labels.csv`, `labels/{train,val,test}_subset0.csv`.

## Thứ tự chạy

```bash
python code/selfcheck.py                                              # kiểm tra code dễ sai
python code/run_experiments.py --stage sanity                         # EDA + split + loss ~ ln9 + overfit 1 batch
python code/run_experiments.py --stage backbones --epochs 8 --only B01 B02 B03 B04 B05
python code/run_experiments.py --stage training  --epochs 8 --only T05 T07 T13 T14
python code/run_experiments.py --stage final     --epochs 8 --seeds 0 1 2
python code/run_experiments.py --stage report     --epochs 8 --seeds 0 1 2
# python code/run_experiments.py --stage inference                    # Bước 3 (chưa chạy vì hết ngân sách GPU)
```

Sau đó chấm điểm:
```bash
python eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag F01 --out eval_out
python eval.py score --pred "predictions/T00_seed*_test.csv" --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag T00 --out eval_out
python eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01_uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" \
    --test-csv labels/test_subset0.csv --val-csv labels/val_subset0.csv --labels labels/labels.csv --out eval_out
```

## Seed đã dùng

- Sàng lọc backbone & ablation (B*, T*): **seed 0**.
- Chung kết F01 và mốc T00: **seed 0, 1, 2**.

## Ghi chú về phạm vi thực nghiệm

- Chạy **8 epoch** (giảm từ 12 do ngân sách GPU) — đã ghi rõ trong báo cáo.
- Khảo sát **5 backbone** (B01–B05) và **4 cấu hình công thức** (T05, T07, T13, T14) so với mốc T00.
- **Bước 3 (suy luận) chưa hoàn thành**: thiếu TTA/FixRes/ensemble/gộp BN/FP16 và độ trễ p50/p95/p99.
  Đã áp dụng temperature scaling trong cấu hình chung kết (ECE test 0.1141 → 0.0080).
- Mốc **T00 seed 0** thiếu file dự đoán test (chỉ có seed 1, 2).
