"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện chuẩn:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame nguyên bản.
    """
    labels_path = Path(labels_dir)
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    if not train_file.exists() or not val_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Không tìm thấy đủ file split fold {fold} trong {labels_dir}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir`
    """
    images_path = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_samples = n_train + n_val + n_test

    # 1. Kiểm tra tổng số mẫu và tỷ lệ
    assert total_samples == 17509, f"Tổng số ảnh không đúng: {total_samples} != 17509"
    r_train, r_val, r_test = n_train / total_samples, n_val / total_samples, n_test / total_samples

    # 2. Giao rỗng
    set_train = set(train_df["Filename"].astype(str))
    set_val = set(val_df["Filename"].astype(str))
    set_test = set(test_df["Filename"].astype(str))

    tv_overlap = set_train.intersection(set_val)
    tt_overlap = set_train.intersection(set_test)
    vt_overlap = set_val.intersection(set_test)

    assert len(tv_overlap) == 0, f"Giao train∩val không rỗng: {len(tv_overlap)} ảnh"
    assert len(tt_overlap) == 0, f"Giao train∩test không rỗng: {len(tt_overlap)} ảnh"
    assert len(vt_overlap) == 0, f"Giao val∩test không rỗng: {len(vt_overlap)} ảnh"

    # 3. Hợp đủ 17509
    union_set = set_train.union(set_val).union(set_test)
    assert len(union_set) == 17509, f"Hợp 3 tập không đủ 17509: {len(union_set)}"

    # 4. Kiểm tra TẤT CẢ file tồn tại (README.md mục 2.1)
    missing_files = []
    if images_path.exists():
        for name in union_set:
            if not (images_path / name).exists():
                missing_files.append(str(images_path / name))
        if missing_files:
            raise FileNotFoundError(
                f"Thiếu {len(missing_files)} file ảnh, ví dụ: {missing_files[:5]}")
    else:
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_path}")

    # Đếm số lượng theo lớp
    train_counts = train_df["Label"].value_counts().sort_index().to_dict()
    val_counts = val_df["Label"].value_counts().sort_index().to_dict()
    test_counts = test_df["Label"].value_counts().sort_index().to_dict()

    stats = {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": total_samples},
        "ratio": {"train": r_train, "val": r_val, "test": r_test},
        "overlap": {"train_val": len(tv_overlap), "train_test": len(tt_overlap), "val_test": len(vt_overlap)},
        "per_class": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        }
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> T.Compose:
    """Tạo torchvision transform cho train hoặc val/test.

    `aug` hỗ trợ: "basic", "color", "randaug", "trivial".
    """
    if train:
        transforms_list = [
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
        ]
        if aug == "color":
            transforms_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        elif aug == "randaug":
            transforms_list.append(T.RandAugment(num_ops=2, magnitude=9))
        elif aug == "trivial":
            transforms_list.append(T.TrivialAugmentWide())
        elif aug == "basic":
            pass
        else:
            raise ValueError(f"Không hỗ trợ augmentation: {aug}")

        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        if img_size == 256:
            transforms_list = [
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]
        elif img_size > 256:
            # FixRes: kiểm tra ở độ phân giải cao hơn lúc train, resize thẳng (không crop/pad)
            transforms_list = [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]
        else:
            transforms_list = [
                T.Resize(256),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]

    return T.Compose(transforms_list)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform: Callable | None = None,
                 preload: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()
        self.preload = preload
        self.cached_images = {}
        if self.preload:
            for fname in self.filenames:
                p = self.images_dir / fname
                if p.exists():
                    with Image.open(p) as img:
                        self.cached_images[fname] = img.convert("RGB")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        fname = self.filenames[idx]
        label = self.labels[idx]

        if fname in self.cached_images:
            image = self.cached_images[fname]
        else:
            img_path = self.images_dir / fname
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, fname


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform: Callable | None,
                batch_size: int, train: bool, sampler: str | None = None,
                num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().to_dict()
            sample_weights = [1.0 / class_counts[y] for y in df["Label"]]
            sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
            data_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=data_sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                worker_init_fn=seed_worker,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                worker_init_fn=seed_worker,
            )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return loader
