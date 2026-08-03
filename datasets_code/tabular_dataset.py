from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TabularCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        feature_columns: List[str],
        label_column: str = "label_id",
        max_samples: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.feature_columns = feature_columns
        self.label_column = label_column

        usecols = feature_columns + [label_column]
        df = pd.read_csv(self.csv_path, usecols=usecols)

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

        self.x = df[feature_columns].to_numpy(dtype=np.float32)
        self.y = df[label_column].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.x[index])
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_columns(dataset_dir: Path) -> List[str]:
    dataset_dir = Path(dataset_dir)
    feature_json = dataset_dir / "feature_columns.json"
    shared_json = dataset_dir / "shared_feature_columns.json"

    if feature_json.exists():
        return load_json(feature_json)
    if shared_json.exists():
        return load_json(shared_json)

    raise FileNotFoundError(f"未找到 feature_columns.json 或 shared_feature_columns.json: {dataset_dir}")


def load_label_mapping(dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir)
    mapping_path = dataset_dir / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"未找到 label_mapping.json: {mapping_path}")
    return load_json(mapping_path)


def create_dataloader(
    csv_path: Path,
    feature_columns: List[str],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[TabularCSVDataset, DataLoader]:
    dataset = TabularCSVDataset(
        csv_path=csv_path,
        feature_columns=feature_columns,
        max_samples=max_samples,
        random_state=random_state,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return dataset, loader


def read_feature_matrix_and_labels(
    csv_path: Path,
    feature_columns: List[str],
    label_column: str = "label_id",
    max_samples: Optional[int] = None,
    random_state: int = 42,
):
    usecols = feature_columns + [label_column]
    df = pd.read_csv(csv_path, usecols=usecols)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    x = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[label_column].to_numpy(dtype=np.int64)
    return x, y


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    dataset_dir = PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_binary"
    features = load_feature_columns(dataset_dir)
    train_csv = dataset_dir / "train.csv"

    ds = TabularCSVDataset(train_csv, features, max_samples=5)
    print("样本数:", len(ds))
    x0, y0 = ds[0]
    print("单条特征 shape:", x0.shape)
    print("标签:", y0.item())