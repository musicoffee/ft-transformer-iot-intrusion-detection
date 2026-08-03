from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RANDOM_STATE
from datasets_code.tabular_dataset import create_dataloader, load_feature_columns, load_label_mapping
from models.ft_transformer import FTTransformer


# =========================
# 你主要改这里
# =========================
DATASET_DIR = PROCESSED_DATA_DIR / "ciciot2023_7class"

BATCH_SIZE = 2048
NUM_WORKERS = 0
PIN_MEMORY = True
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

D_TOKEN = 64
N_HEADS = 8
N_LAYERS = 4
FFN_DIM = 128
DROPOUT = 0.1

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

CHECKPOINT_NAME = "ft_multiclass_best.pt"
LAST_CHECKPOINT_NAME = "ft_multiclass_last.pt"
HISTORY_JSON_NAME = "ft_multiclass_train_history.json"
HISTORY_PNG_NAME = "ft_multiclass_train_history.png"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return metrics


def run_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, Dict[str, float]]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

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

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)
        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_multiclass_metrics(y_true, y_pred)
    metrics["loss"] = float(avg_loss)
    return avg_loss, metrics


def save_history_plot(history: dict, save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["val_f1_macro"], label="Val F1 Macro")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("FT-Transformer Multi-class Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_DIR)

    set_seed(RANDOM_STATE)
    device = get_device()
    print(f"Using device: {device}")

    feature_columns = load_feature_columns(DATASET_DIR)
    label_mapping = load_label_mapping(DATASET_DIR)
    num_classes = len(label_mapping)

    train_csv = DATASET_DIR / "train.csv"
    val_csv = DATASET_DIR / "val.csv"

    train_dataset, train_loader = create_dataloader(
        csv_path=train_csv,
        feature_columns=feature_columns,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_samples=MAX_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
    )
    val_dataset, val_loader = create_dataloader(
        csv_path=val_csv,
        feature_columns=feature_columns,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_samples=MAX_VAL_SAMPLES,
        random_state=RANDOM_STATE,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Num features: {len(feature_columns)}")
    print(f"Label mapping: {label_mapping}")

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1_macro": [],
        "val_f1_macro": [],
        "val_accuracy": [],
        "val_f1_weighted": [],
        "lr": [],
    }

    best_val_f1_macro = -1.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss, val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["f1_macro"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1_weighted"].append(val_metrics["f1_weighted"])
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"train_f1_macro={train_metrics['f1_macro']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
            f"val_f1_weighted={val_metrics['f1_weighted']:.4f} | "
            f"lr={current_lr:.6f}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "label_mapping": label_mapping,
            "feature_columns": feature_columns,
            "model_config": {
                "num_features": len(feature_columns),
                "num_classes": num_classes,
                "d_token": D_TOKEN,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "dim_feedforward": FFN_DIM,
                "dropout": DROPOUT,
            },
            "best_val_f1_macro": best_val_f1_macro,
        }

        torch.save(checkpoint_payload, CHECKPOINT_DIR / LAST_CHECKPOINT_NAME)

        if val_metrics["f1_macro"] > best_val_f1_macro:
            best_val_f1_macro = val_metrics["f1_macro"]
            patience_counter = 0
            checkpoint_payload["best_val_f1_macro"] = best_val_f1_macro
            torch.save(checkpoint_payload, CHECKPOINT_DIR / CHECKPOINT_NAME)
            print(f"Saved best checkpoint to: {CHECKPOINT_DIR / CHECKPOINT_NAME}")
        else:
            patience_counter += 1
            print(f"EarlyStopping patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    with open(REPORT_DIR / HISTORY_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    save_history_plot(history, FIGURE_DIR / HISTORY_PNG_NAME)
    print(f"History JSON saved to: {REPORT_DIR / HISTORY_JSON_NAME}")
    print(f"History figure saved to: {FIGURE_DIR / HISTORY_PNG_NAME}")


if __name__ == "__main__":
    main()