from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

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

# 放到：trainers/run_ablation_ft_binary.py
# 直接在 PyCharm 里点绿色三角运行

DATASET_DIR = PROCESSED_DATA_DIR / "ciciot2023_binary"

BATCH_SIZE = 2048
NUM_WORKERS = 0
PIN_MEMORY = True

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None

EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4

SEARCH_SPACE = [
    {"name": "dtoken_32", "d_token": 32, "n_heads": 4, "n_layers": 4, "dim_feedforward": 128, "dropout": 0.1},
    {"name": "dtoken_64", "d_token": 64, "n_heads": 8, "n_layers": 4, "dim_feedforward": 128, "dropout": 0.1},
    {"name": "dtoken_128", "d_token": 128, "n_heads": 8, "n_layers": 4, "dim_feedforward": 256, "dropout": 0.1},
]

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

SUMMARY_JSON = "ablation_ft_binary_summary.json"
SUMMARY_CSV = "ablation_ft_binary_summary.csv"
SUMMARY_PNG = "ablation_ft_binary_summary.png"


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


def run_one_epoch(model, loader, criterion, optimizer, device, attack_label_id, train=True):
    if train:
        model.train()
    else:
        model.eval()

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
    return metrics


def evaluate_test(model, loader, device, attack_label_id):
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_y_prob_attack = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_y_true.append(y.numpy())
            all_y_pred.append(preds.cpu().numpy())
            all_y_prob_attack.append(probs[:, attack_label_id].cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_prob_attack = np.concatenate(all_y_prob_attack)
    return compute_binary_metrics(y_true, y_pred, y_prob_attack, attack_label_id)


def save_summary_plot(rows: List[dict], save_path: Path):
    trial_names = [r["trial_name"] for r in rows]
    val_f1 = [r["best_val_f1_attack"] for r in rows]
    test_f1 = [r["test_f1_attack"] for r in rows]

    x = np.arange(len(trial_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, val_f1, width=width, label="Best Val F1")
    plt.bar(x + width / 2, test_f1, width=width, label="Test F1")
    plt.xticks(x, trial_names, rotation=15)
    plt.ylabel("F1 Score")
    plt.title("FT-Transformer Binary Ablation Summary")
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
    test_csv = DATASET_DIR / "test.csv"

    _, train_loader = create_dataloader(
        csv_path=train_csv,
        feature_columns=feature_columns,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_samples=MAX_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
    )
    _, val_loader = create_dataloader(
        csv_path=val_csv,
        feature_columns=feature_columns,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_samples=MAX_VAL_SAMPLES,
        random_state=RANDOM_STATE,
    )
    _, test_loader = create_dataloader(
        csv_path=test_csv,
        feature_columns=feature_columns,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_samples=MAX_TEST_SAMPLES,
        random_state=RANDOM_STATE,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Num features: {len(feature_columns)}")
    print(f"Label mapping: {label_mapping}")

    all_results = []

    for trial in SEARCH_SPACE:
        trial_name = trial["name"]
        print("\n" + "=" * 80)
        print(f"Start trial: {trial_name}")
        print(f"Config: {trial}")

        set_seed(RANDOM_STATE)

        model = FTTransformer(
            num_features=len(feature_columns),
            num_classes=num_classes,
            d_token=trial["d_token"],
            n_heads=trial["n_heads"],
            n_layers=trial["n_layers"],
            dim_feedforward=trial["dim_feedforward"],
            dropout=trial["dropout"],
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

        best_val_f1 = -1.0
        patience_counter = 0
        best_checkpoint_path = CHECKPOINT_DIR / f"{trial_name}_best.pt"

        for epoch in range(1, EPOCHS + 1):
            train_metrics = run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                attack_label_id=attack_label_id,
                train=True,
            )
            val_metrics = run_one_epoch(
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

            print(
                f"[{trial_name}] Epoch [{epoch}/{EPOCHS}] | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"train_f1={train_metrics['f1_attack']:.4f} | "
                f"val_f1={val_metrics['f1_attack']:.4f} | "
                f"val_auc={val_metrics['roc_auc_attack']:.4f} | "
                f"lr={current_lr:.6f}"
            )

            payload = {
                "epoch": epoch,
                "trial_name": trial_name,
                "model_state_dict": model.state_dict(),
                "label_mapping": label_mapping,
                "feature_columns": feature_columns,
                "model_config": {
                    "num_features": len(feature_columns),
                    "num_classes": num_classes,
                    "d_token": trial["d_token"],
                    "n_heads": trial["n_heads"],
                    "n_layers": trial["n_layers"],
                    "dim_feedforward": trial["dim_feedforward"],
                    "dropout": trial["dropout"],
                },
                "best_val_f1_attack": best_val_f1,
            }

            if val_metrics["f1_attack"] > best_val_f1:
                best_val_f1 = val_metrics["f1_attack"]
                patience_counter = 0
                payload["best_val_f1_attack"] = best_val_f1
                torch.save(payload, best_checkpoint_path)
                print(f"Saved best checkpoint: {best_checkpoint_path}")
            else:
                patience_counter += 1
                print(f"EarlyStopping patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        best_payload = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_payload["model_state_dict"])
        test_metrics = evaluate_test(model, test_loader, device, attack_label_id)

        result_row = {
            "trial_name": trial_name,
            "d_token": trial["d_token"],
            "n_heads": trial["n_heads"],
            "n_layers": trial["n_layers"],
            "dim_feedforward": trial["dim_feedforward"],
            "dropout": trial["dropout"],
            "best_val_f1_attack": float(best_val_f1),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_precision_attack": float(test_metrics["precision_attack"]),
            "test_recall_attack": float(test_metrics["recall_attack"]),
            "test_f1_attack": float(test_metrics["f1_attack"]),
            "test_roc_auc_attack": float(test_metrics["roc_auc_attack"]),
            "test_pr_auc_attack": float(test_metrics["pr_auc_attack"]),
        }
        all_results.append(result_row)

        print(f"Finished trial: {trial_name}")
        print("Test metrics:", test_metrics)

    all_results = sorted(all_results, key=lambda x: x["best_val_f1_attack"], reverse=True)

    with open(REPORT_DIR / SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    csv_path = REPORT_DIR / SUMMARY_CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    save_summary_plot(all_results, FIGURE_DIR / SUMMARY_PNG)

    print("\n" + "=" * 80)
    print("Ablation finished.")
    print(f"Summary JSON: {REPORT_DIR / SUMMARY_JSON}")
    print(f"Summary CSV: {REPORT_DIR / SUMMARY_CSV}")
    print(f"Summary figure: {FIGURE_DIR / SUMMARY_PNG}")
    print("Best trial:", all_results[0])


if __name__ == "__main__":
    main()
