from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RANDOM_STATE
from models.ft_transformer import FTTransformer

ALIGNED_DIR = PROCESSED_DATA_DIR / "cross_dataset_aligned_binary"

TRAIN_CSV = ALIGNED_DIR / "main_train_aligned.csv"
VAL_CSV = ALIGNED_DIR / "main_val_aligned.csv"
MAIN_TEST_CSV = ALIGNED_DIR / "main_test_aligned.csv"
EXTERNAL_TEST_CSV = ALIGNED_DIR / "external_test_aligned.csv"
FEATURE_JSON = ALIGNED_DIR / "shared_feature_columns.json"

BATCH_SIZE = 2048
NUM_WORKERS = 0
PIN_MEMORY = True

EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4

D_TOKEN = 64
N_HEADS = 8
N_LAYERS = 4
FFN_DIM = 128
DROPOUT = 0.1

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CHECKPOINT_NAME = "ft_binary_cross_dataset_best.pt"
METRICS_JSON = "ft_binary_cross_dataset_metrics.json"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleAlignedDataset(Dataset):
    def __init__(self, csv_path: Path, feature_columns):
        df = pd.read_csv(csv_path, usecols=feature_columns + ["label_id"])
        self.x = df[feature_columns].to_numpy(dtype=np.float32)
        self.y = df["label_id"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def create_loader(csv_path: Path, feature_columns, batch_size, shuffle):
    ds = SimpleAlignedDataset(csv_path, feature_columns)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    return ds, loader


def compute_metrics(y_true, y_pred, y_prob_attack, attack_label_id):
    y_true_bin = (y_true == attack_label_id).astype(int)
    y_pred_bin = (y_pred == attack_label_id).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall_attack": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1_attack": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "roc_auc_attack": float(roc_auc_score(y_true_bin, y_prob_attack)),
        "pr_auc_attack": float(average_precision_score(y_true_bin, y_prob_attack)),
    }


def run_epoch(model, loader, criterion, optimizer, device, attack_label_id, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * x.size(0)
        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(preds.cpu().numpy())
        y_prob_all.append(probs[:, attack_label_id].detach().cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all)

    metrics = compute_metrics(y_true, y_pred, y_prob, attack_label_id)
    metrics["loss"] = float(total_loss / len(loader.dataset))
    return metrics


def evaluate_loader(model, loader, device, attack_label_id):
    model.eval()

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true_all.append(y.numpy())
            y_pred_all.append(preds.cpu().numpy())
            y_prob_all.append(probs[:, attack_label_id].cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all)
    return compute_metrics(y_true, y_pred, y_prob, attack_label_id)


def main():
    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(REPORT_DIR)

    set_seed(RANDOM_STATE)
    device = get_device()
    print(f"Using device: {device}")

    with open(FEATURE_JSON, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    attack_label_id = 0
    num_classes = 2

    train_ds, train_loader = create_loader(TRAIN_CSV, feature_columns, BATCH_SIZE, True)
    val_ds, val_loader = create_loader(VAL_CSV, feature_columns, BATCH_SIZE, False)
    main_test_ds, main_test_loader = create_loader(MAIN_TEST_CSV, feature_columns, BATCH_SIZE, False)
    ext_test_ds, ext_test_loader = create_loader(EXTERNAL_TEST_CSV, feature_columns, BATCH_SIZE, False)

    print(f"Aligned train samples: {len(train_ds)}")
    print(f"Aligned val samples: {len(val_ds)}")
    print(f"Aligned main test samples: {len(main_test_ds)}")
    print(f"Aligned external test samples: {len(ext_test_ds)}")
    print(f"Shared features: {len(feature_columns)}")

    model = FTTransformer(
        num_features=len(feature_columns),
        num_classes=num_classes,
        d_token=D_TOKEN,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_feedforward=FFN_DIM,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, attack_label_id, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, attack_label_id, train=False)

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['f1_attack']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1_attack']:.4f}"
        )

        if val_metrics["f1_attack"] > best_val_f1:
            best_val_f1 = val_metrics["f1_attack"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": feature_columns,
                    "attack_label_id": attack_label_id,
                    "model_config": {
                        "num_features": len(feature_columns),
                        "num_classes": num_classes,
                        "d_token": D_TOKEN,
                        "n_heads": N_HEADS,
                        "n_layers": N_LAYERS,
                        "dim_feedforward": FFN_DIM,
                        "dropout": DROPOUT,
                    },
                },
                CHECKPOINT_DIR / CHECKPOINT_NAME,
            )
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    checkpoint = torch.load(CHECKPOINT_DIR / CHECKPOINT_NAME, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    main_metrics = evaluate_loader(model, main_test_loader, device, attack_label_id)
    ext_metrics = evaluate_loader(model, ext_test_loader, device, attack_label_id)

    print("Main aligned test metrics:", main_metrics)
    print("External aligned test metrics:", ext_metrics)

    payload = {
        "main_aligned_test": main_metrics,
        "external_aligned_test": ext_metrics,
        "shared_feature_count": len(feature_columns),
    }
    with open(REPORT_DIR / METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Cross-dataset metrics saved to: {REPORT_DIR / METRICS_JSON}")


if __name__ == "__main__":
    main()