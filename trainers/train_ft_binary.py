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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch.optim import AdamW

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RANDOM_STATE
from datasets_code.tabular_dataset import create_dataloader, load_feature_columns, load_label_mapping
from models.ft_transformer import FTTransformer

DATASET_DIR = PROCESSED_DATA_DIR / "ciciot2023_binary"

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

CHECKPOINT_NAME = "ft_binary_best.pt"
LAST_CHECKPOINT_NAME = "ft_binary_last.pt"
HISTORY_JSON_NAME = "ft_binary_train_history.json"
HISTORY_PNG_NAME = "ft_binary_train_history.png"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_attack: np.ndarray,
    attack_label_id: int,
) -> Dict[str, float]:
    y_true_bin = (y_true == attack_label_id).astype(int)
    y_pred_bin = (y_pred == attack_label_id).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall_attack": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1_attack": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "roc_auc_attack": float(roc_auc_score(y_true_bin, y_prob_attack)) if len(np.unique(y_true_bin)) > 1 else float("nan"),
        "pr_auc_attack": float(average_precision_score(y_true_bin, y_prob_attack)) if len(np.unique(y_true_bin)) > 1 else float("nan"),
    }
    return metrics


def run_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    attack_label_id: int,
    train: bool = True,
) -> Tuple[float, Dict[str, float]]:
    model.train() if train else model.eval()

    total_loss = 0.0
    all_y_true = []
    all_y_pred = []
    all_y_prob_attack = []

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
        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(preds.detach().cpu().numpy())
        all_y_prob_attack.append(probs[:, attack_label_id].detach().cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_prob_attack = np.concatenate(all_y_prob_attack)

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob_attack, attack_label_id)
    metrics["loss"] = float(avg_loss)
    return avg_loss, metrics


def save_history_plot(history: dict, save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["val_f1_attack"], label="Val F1(attack)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("FT-Transformer Binary Training History")
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

    if "attack" not in label_mapping:
        raise ValueError(f"label_mapping.json 中未找到 'attack'。当前映射：{label_mapping}")

    attack_label_id = int(label_mapping["attack"])
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
        "train_f1_attack": [],
        "val_f1_attack": [],
        "val_accuracy": [],
        "val_roc_auc_attack": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            attack_label_id=attack_label_id,
            train=True,
        )

        val_loss, val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            attack_label_id=attack_label_id,
            train=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["f1_attack"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_f1_attack"].append(train_metrics["f1_attack"])
        history["val_f1_attack"].append(val_metrics["f1_attack"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_roc_auc_attack"].append(val_metrics["roc_auc_attack"])
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"train_f1_attack={train_metrics['f1_attack']:.4f} | "
            f"val_f1_attack={val_metrics['f1_attack']:.4f} | "
            f"val_auc={val_metrics['roc_auc_attack']:.4f} | "
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
            "best_val_f1_attack": best_val_f1,
        }

        torch.save(checkpoint_payload, CHECKPOINT_DIR / LAST_CHECKPOINT_NAME)

        if val_metrics["f1_attack"] > best_val_f1:
            best_val_f1 = val_metrics["f1_attack"]
            patience_counter = 0
            checkpoint_payload["best_val_f1_attack"] = best_val_f1
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