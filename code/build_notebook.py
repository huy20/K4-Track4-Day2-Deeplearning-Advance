"""build_notebook.py - sinh file code/lab_day2.ipynb (notebook Colab/Kaggle chạy THẬT).

Chạy: python code/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

cells: list[dict] = []


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src})


md(r"""# Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

Notebook chạy **thật** trên Google Colab / Kaggle (GPU T4 trở lên). Mọi số liệu sinh ra
từ huấn luyện/suy luận thực tế và truy ngược được về `runs/<exp_id>/seed<k>/`.

- **Dataset**: DeepWeeds, fold 0 chia sẵn (60/20/20). Val để chọn cấu hình; **test chỉ dùng ở Bước 4**.
- **Quy trình**: sanity/EDA → backbones (B01–B06) → ablation công thức (T00–T14) →
  inference (I00–I08) + độ trễ → chung kết F01 (3 seed) + mốc T00 (3 seed) → results.xlsx + report.md.
- **Lưu ý**: mỗi stage lưu `summary.json`, chạy lại sẽ **resume** (bỏ qua cái đã xong). Dùng `--force` để chạy lại.

> Có thể chỉnh `EPOCHS`, `SEEDS` và dùng `--only` để chạy nhanh một phần.""")

md("## 0. Cài đặt thư viện")
code(r"""!pip -q install "timm>=0.9.12" thop openpyxl scipy matplotlib pandas scikit-learn
import torch, timm
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| timm", timm.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))""")

md("""## 1. Kết nối Google Drive và trỏ tới thư mục project

Đặt thư mục project (repo này) trong `MyDrive`, hoặc upload file zip lên `/content` rồi giải nén.
Nếu cần, sửa `PROJECT_OVERRIDE` thành đường dẫn thật.""")
code(r"""import os, sys, shutil, subprocess, hashlib, zipfile, urllib.request
from pathlib import Path

PROJECT_OVERRIDE = None  # ví dụ: '/content/drive/MyDrive/Lab/K4-Track4-Day2-Deeplearning-Advance'
PROJECT_NAME = "K4-Track4-Day2-Deeplearning-Advance"

IN_COLAB = False
try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    IN_COLAB = True
except Exception as e:  # Kaggle / local
    print("Không chạy được google.colab.drive (bình thường trên Kaggle/local):", e)

candidates = []
if PROJECT_OVERRIDE:
    candidates.append(Path(PROJECT_OVERRIDE))
candidates += [
    Path("/content/drive/MyDrive") / PROJECT_NAME,
    Path("/content/drive/MyDrive/Lab") / PROJECT_NAME,
    Path("/content") / PROJECT_NAME,
    Path.cwd(),
]
PROJECT = next((p for p in candidates if (p / "eval.py").exists()), None)
if PROJECT is None:
    print("Các vị trí đã thử:", [str(c) for c in candidates])
    raise SystemExit(
        "Không tìm thấy project. Hãy upload zip repo lên Colab và giải nén vào /content, "
        "hoặc đặt thư mục project trong MyDrive rồi đặt PROJECT_OVERRIDE."
    )
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "code"))
print("PROJECT =", PROJECT)
print(sorted(x.name for x in PROJECT.iterdir())[:30])""")

md("""## 2. Tải dataset DeepWeeds và kiểm tra MD5

Ảnh ~490 MB từ Zenodo. Bỏ qua nếu `images/` đã có đủ. MD5: `b7b30f96d466fba86016aa5a26606e0f`.""")
code(r"""IMAGE_ZIP_URL = "https://zenodo.org/records/7939060/files/images.zip?download=1"
IMAGE_ZIP_MD5 = "b7b30f96d466fba86016aa5a26606e0f"

def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

IMAGES = PROJECT / "images"

def count_imgs():
    return len(list(IMAGES.glob("*.jpg"))) if IMAGES.exists() else 0

if count_imgs() < 17000:
    zip_path = PROJECT / "images.zip"
    if not zip_path.exists():
        print("Đang tải images.zip (~490 MB)...")
        urllib.request.urlretrieve(IMAGE_ZIP_URL, zip_path)
    digest = md5sum(zip_path)
    print("MD5 =", digest)
    assert digest == IMAGE_ZIP_MD5, f"MD5 sai! mong đợi {IMAGE_ZIP_MD5}"
    print("Giải nén...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(PROJECT)
    IMAGES.mkdir(exist_ok=True)
    moved = 0
    for p in list(PROJECT.glob("*.jpg")):
        shutil.move(str(p), IMAGES / p.name)
        moved += 1
    print("Đã chuyển", moved, "ảnh vào images/")
else:
    print("Đã có", count_imgs(), "ảnh, bỏ qua tải.")

assert (PROJECT / "labels" / "labels.csv").exists(), "Thiếu labels/labels.csv"
assert count_imgs() >= 17000, f"Chỉ có {count_imgs()} ảnh"
print("Số ảnh:", count_imgs())""")

md("""## 3. Kiểm tra pipeline (self-test + EDA + split + loss ~ ln(9) + overfit 1 batch)

Đây là bằng chứng cho RUBRIC mục A và H. Stage `sanity` in phân bố lớp, kiểm tra giao rỗng,
in loss ban đầu (kỳ vọng ≈ 2.197) và overfit 1 batch.""")
code(r"""!python code/selfcheck.py
!python code/run_experiments.py --stage sanity""")

md("""## 4. (Tùy chọn) Chạy thử nhanh 1 backbone — xác nhận GPU trước khi chạy full

Chạy MobileNetV3 1 epoch (~1 phút trên T4). Nếu cell này OK thì chạy full ở cell dưới.""")
code(r"""!python code/run_experiments.py --stage backbones --only B05 --epochs 1""")

md("""## 5. Chạy FULL toàn bộ thí nghiệm

`EPOCHS=12` theo GUIDE mục 1.4; `SEEDS=[0,1,2]` cho vòng chung kết. Có thể giảm `EPOCHS`
nếu hết ngân sách GPU (nhớ ghi lại trong báo cáo). Tiến trình được resume theo `summary.json`.""")
code(r"""import subprocess, sys

EPOCHS = 12
SEEDS = [0, 1, 2]

def run(*args):
    print(">>>", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=False)

run("code/run_experiments.py", "--stage", "backbones", "--epochs", EPOCHS)
run("code/run_experiments.py", "--stage", "training", "--epochs", EPOCHS)
run("code/run_experiments.py", "--stage", "inference", "--epochs", EPOCHS)
run("code/run_experiments.py", "--stage", "final", "--epochs", EPOCHS, "--seeds", *SEEDS)
run("code/run_experiments.py", "--stage", "report", "--epochs", EPOCHS, "--seeds", *SEEDS)""")

md("## 6. Xem kết quả (results.xlsx + tự chấm RUBRIC phần I)")
code(r"""import pandas as pd
xl = pd.ExcelFile("results.xlsx")
for s in ["Summary", "Backbones", "Training", "Inference", "Final", "PerClass", "Latency"]:
    if s in xl.sheet_names:
        print("=" * 25, s)
        display(pd.read_excel(xl, sheet_name=s, nrows=30))

import json
p = Path("eval_out/grade_I.json")
if p.exists():
    print("\n== Tự chấm phần I ==")
    print(p.read_text(encoding="utf-8"))""")

md("## 7. Lưu sản phẩm ra Google Drive (không lưu checkpoint lớn)")
code(r"""if IN_COLAB:
    OUT = Path("/content/drive/MyDrive") / (PROJECT_NAME + "_output")
else:
    OUT = PROJECT / "submission"
OUT.mkdir(parents=True, exist_ok=True)
ignore = shutil.ignore_patterns("*.pt", "*.pth", "*.ckpt", "*.zip", "*.jpg")
for item in ["results.xlsx", "report.md", "curves", "predictions", "eval_out", "runs", "code"]:
    src = PROJECT / item
    if not src.exists():
        continue
    dst = OUT / item
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=ignore)
    else:
        shutil.copy2(src, dst)
print("Đã lưu sản phẩm vào:", OUT)
for root, dirs, files in os.walk(OUT):
    print(root, "->", len(files), "files")""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "lab_day2.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Đã ghi", out, "với", len(cells), "cells")
