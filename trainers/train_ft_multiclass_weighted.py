from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# =========================================================
# 放到：trainers/train_ft_multiclass_weighted.py
# 用法：在 PyCharm 里直接点绿色三角运行
# 任务：带 class_weight 的 FT-Transformer 多分类训练
# 说明：这版不再依赖原 tabular_dataset.py 的标签列假设
# =========================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.ft_transformer import FTTransformer

RANDOM_STATE = 42
DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_7class"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
TEST_FILE = DATA_DIR / "test.csv"
FEATURE_FILE = DATA_DIR / "feature_columns.json"
LABEL_FILE = DATA_DIR / "label_mapping.json"

# 训练参数
BATCH_SIZE = 1024
NUM_WORKERS = 0
PIN_MEMORY = True
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

# 模型参数
D_TOKEN = 64
N_HEADS = 8
N_LAYERS = 4
DIM_FEEDFORWARD = 128
DROPOUT = 0.1

CKPT_PATH = OUTPUT_CKPT_DIR / "ft_multiclass_weighted_best.pt"
REPORT_JSON = OUTPUT_REPORT_DIR / "ft_multiclass_weighted_test_metrics.json"
REPORT_TXT = OUTPUT_REPORT_DIR / "ft_multiclass_weighted_classification_report.txt"
HISTORY_JSON = OUTPUT_REPORT_DIR / "ft_multiclass_weighted_history.json"
HISTORY_PNG = OUTPUT_FIGURE_DIR / "ft_multiclass_weighted_history.png"
CM_PNG = OUTPUT_FIGURE_DIR / "ft_multiclass_weighted_confusion_matrix.png"


def ensure_dirs() -> None:
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_colname(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def detect_label_column(df: pd.DataFrame, feature_columns: list[str]) -> str:
    normalized_map = {normalize_colname(c): c for c in df.columns}

    candidates = [
        "label",
        "label_id",
        "target",
        "target_id",
        "class",
        "class_id",
        "y",
        "category",
    ]
    for cand in candidates:
        if cand in normalized_map:
            return normalized_map[cand]

    feature_set = set(feature_columns)
    extra_cols = [c for c in df.columns if c not in feature_set]

    useless_cols = {
        "label_name",
        "class_name",
        "split",
        "source",
        "dataset",
        "subset",
        "Unnamed: 0",
    }
    filtered_extra_cols = [c for c in extra_cols if c not in useless_cols]

    if len(filtered_extra_cols) == 1:
        print(f"自动推断标签列为：{filtered_extra_cols[0]}")
        return filtered_extra_cols[0]

    low_cardinality_cols = []
    for col in filtered_extra_cols:
        uniques = pd.Series(df[col]).dropna().unique().tolist()
        if len(uniques) <= 20:
            low_cardinality_cols.append(col)

    if len(low_cardinality_cols) == 1:
        print(f"自动推断标签列为：{low_cardinality_cols[0]}")
        return low_cardinality_cols[0]

    raise ValueError(
        "未能自动识别多分类标签列。\n"
        f"全部列名：{list(df.columns)}\n"
        f"非特征列候选：{filtered_extra_cols}\n"
        f"低类别候选：{low_cardinality_cols}"
    )


def encode_labels(y_raw: pd.Series | np.ndarray, label_mapping: dict) -> np.ndarray:
    y_series = pd.Series(y_raw).copy()

    # 情况1：已经是整数标签
    try:
        y_int = y_series.astype(int)
        unique_vals = set(y_int.unique().tolist())
        mapping_vals = set(int(v) for v in label_mapping.values())
        if unique_vals.issubset(mapping_vals):
            return y_int.values
    except Exception:
        pass

    # 情况2：文本标签 -> label_mapping[key]
    if all(isinstance(k, str) for k in label_mapping.keys()):
        mapped = y_series.map(label_mapping)
        if mapped.isna().any():
            missing = sorted(y_series[mapped.isna()].astype(str).unique().tolist())
            raise ValueError(
                f"以下文本标签无法在 label_mapping.json 中找到映射：{missing}\n"
                f"label_mapping keys: {list(label_mapping.keys())}"
            )
        return mapped.astype(int).values

    # 情况3：兜底反转
    reverse_mapping = {str(v): int(k) for k, v in label_mapping.items() if str(k).isdigit()}
    mapped = y_series.astype(str).map(reverse_mapping)
    if mapped.notna().all():
        return mapped.astype(int).values

    raise ValueError(
        "无法将标签编码为整数，请检查 label_mapping.json 与标签列内容是否一致。\n"
        f"标签列示例：{y_series.head(10).tolist()}\n"
        f"label_mapping: {label_mapping}"
    )


class TabularCSVDataset(Dataset):
    def __init__(self, csv_path: Path, feature_columns: list[str], label_mapping: dict):
        self.df = pd.read_csv(csv_path)
        self.feature_columns = feature_columns
        self.label_mapping = label_mapping
        self.label_col = detect_label_column(self.df, feature_columns)

        self.x = self.df[self.feature_columns].astype(np.float32).values
        self.y = encode_labels(self.df[self.label_col], self.label_mapping).astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def create_dataloader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )


def compute_class_weights(train_dataset: TabularCSVDataset, num_classes: int) -> np.ndarray:
    counts = Counter(train_dataset.y.tolist())
    total = sum(counts.values())
    weights = []
    for class_id in range(num_classes):
        count = counts.get(class_id, 1)
        weights.append(total / (num_classes * count))
    return np.asarray(weights, dtype=np.float32)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            pred = torch.argmax(logits, dim=1)

            total_loss += loss.item() * x.size(0)
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    avg_loss = total_loss / len(loader.dataset)

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    return metrics, y_true, y_pred


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    avg_loss = total_loss / len(loader.dataset)

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    return metrics


def plot_history(history: list[dict], save_path: Path):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_f1_macro = [row["val_f1_macro"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.plot(epochs, val_f1_macro, label="Val F1 Macro")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("FT-Transformer Weighted Multi-class Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion(cm: np.ndarray, class_names: list[str], save_path: Path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("FT Weighted Multi-class Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    ensure_dirs()
    set_seed(RANDOM_STATE)
    device = get_device()

    feature_columns = load_json(FEATURE_FILE)
    label_mapping = load_json(LABEL_FILE)
    inv_label_mapping = {int(v): k for k, v in label_mapping.items()}
    class_names = [inv_label_mapping[i] for i in range(len(inv_label_mapping))]
    num_classes = len(class_names)

    train_dataset = TabularCSVDataset(TRAIN_FILE, feature_columns, label_mapping)
    val_dataset = TabularCSVDataset(VAL_FILE, feature_columns, label_mapping)
    test_dataset = TabularCSVDataset(TEST_FILE, feature_columns, label_mapping)

    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset, shuffle=False)
    test_loader = create_dataloader(test_dataset, shuffle=False)

    class_weights = compute_class_weights(train_dataset, num_classes)

    print("Using device:", device)
    print("Detected label column:", train_dataset.label_col)
    print("Class weights:", class_weights.tolist())

    model = FTTransformer(
        num_features=len(feature_columns),
        num_classes=num_classes,
        d_token=D_TOKEN,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    patience = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1_macro={train_metrics['f1_macro']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_mapping": label_mapping,
                    "feature_columns": feature_columns,
                    "label_column": train_dataset.label_col,
                    "model_config": {
                        "num_features": len(feature_columns),
                        "num_classes": num_classes,
                        "d_token": D_TOKEN,
                        "n_heads": N_HEADS,
                        "n_layers": N_LAYERS,
                        "dim_feedforward": DIM_FEEDFORWARD,
                        "dropout": DROPOUT,
                    },
                    "class_weights": class_weights.tolist(),
                },
                CKPT_PATH,
            )
            print(f"Saved best checkpoint to: {CKPT_PATH}")
        else:
            patience += 1
            print(f"EarlyStopping patience: {patience}/{EARLY_STOPPING_PATIENCE}")
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                **test_metrics,
                "label_mapping": label_mapping,
                "label_column": train_dataset.label_col,
                "class_weights": class_weights.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_history(history, HISTORY_PNG)
    plot_confusion(cm, class_names, CM_PNG)

    print("Weighted multiclass test metrics:", test_metrics)
    print("Weighted multiclass metrics saved to:", REPORT_JSON)
    print("Weighted multiclass report saved to:", REPORT_TXT)
    print("Weighted multiclass history saved to:", HISTORY_JSON)
    print("Weighted multiclass history figure saved to:", HISTORY_PNG)
    print("Weighted multiclass confusion matrix saved to:", CM_PNG)


if __name__ == "__main__":
    main()