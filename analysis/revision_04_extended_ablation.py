from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RANDOM_STATE
from datasets_code.tabular_dataset import create_dataloader
from models.ft_transformer import FTTransformer


# ============================================================
# Extended ablation study for revision
# ------------------------------------------------------------
# Purpose:
#   Reviewer comment:
#   "The current ablation study only varies token dimension.
#    Please evaluate the number of Transformer encoder layers
#    and the number of attention heads."
#
# Output:
#   outputs/revision_tables/extended_ablation_results.csv
#   outputs/revision_tables/extended_ablation_results_for_word.txt
#   outputs/checkpoints/revision_ablation_*.pt
# ============================================================

DATASET_DIR = PROCESSED_DATA_DIR / "ciciot2023_binary"
TRAIN_FILE = DATASET_DIR / "train.csv"
VAL_FILE = DATASET_DIR / "val.csv"
TEST_FILE = DATASET_DIR / "test.csv"

MAIN_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "ft_binary_best.pt"

OUT_DIR = PROJECT_ROOT / "outputs" / "revision_tables"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2048
EPOCHS = 12
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
PIN_MEMORY = True

# 为了控制时间，扩展消融默认使用完整 train/val/test。
# 如果你跑得太慢，可以把 MAX_TRAIN_SAMPLES 改成 200000。
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None

ATTACK_LABEL_NAME = "attack"


ABLATION_CONFIGS = [
    {
        "trial_name": "layers_2_heads_4",
        "d_token": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    {
        "trial_name": "layers_6_heads_4",
        "d_token": 64,
        "n_heads": 4,
        "n_layers": 6,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    {
        "trial_name": "layers_4_heads_2",
        "d_token": 64,
        "n_heads": 2,
        "n_layers": 4,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    {
        "trial_name": "layers_4_heads_8",
        "d_token": 64,
        "n_heads": 8,
        "n_layers": 4,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_main_info(device: torch.device):
    checkpoint = torch.load(MAIN_CHECKPOINT, map_location=device)
    feature_columns = checkpoint["feature_columns"]
    label_mapping = checkpoint["label_mapping"]
    attack_label_id = int(label_mapping[ATTACK_LABEL_NAME])
    main_config = checkpoint["model_config"]
    return checkpoint, feature_columns, label_mapping, attack_label_id, main_config


def build_loaders(feature_columns: list[str]):
    _, train_loader = create_dataloader(
        csv_path=TRAIN_FILE,
        feature_columns=feature_columns,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_samples=MAX_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
    )

    _, val_loader = create_dataloader(
        csv_path=VAL_FILE,
        feature_columns=feature_columns,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_samples=MAX_VAL_SAMPLES,
        random_state=RANDOM_STATE,
    )

    _, test_loader = create_dataloader(
        csv_path=TEST_FILE,
        feature_columns=feature_columns,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_samples=MAX_TEST_SAMPLES,
        random_state=RANDOM_STATE,
    )

    return train_loader, val_loader, test_loader


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_binary_metrics(y_true, y_pred, prob_attack, attack_label_id: int):
    y_true_attack = (y_true == attack_label_id).astype(int)
    y_pred_attack = (y_pred == attack_label_id).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true_attack, y_pred_attack, zero_division=0)),
        "recall_attack": float(recall_score(y_true_attack, y_pred_attack, zero_division=0)),
        "f1_attack": float(f1_score(y_true_attack, y_pred_attack, zero_division=0)),
        "roc_auc_attack": float(roc_auc_score(y_true_attack, prob_attack)),
        "pr_auc_attack": float(average_precision_score(y_true_attack, prob_attack)),
    }


@torch.no_grad()
def evaluate(model, loader, device, attack_label_id: int, criterion=None):
    model.eval()

    all_y_true = []
    all_y_pred = []
    all_prob_attack = []
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * len(y)
            total_samples += len(y)

        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(preds.detach().cpu().numpy())
        all_prob_attack.append(probs[:, attack_label_id].detach().cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    prob_attack = np.concatenate(all_prob_attack)

    metrics = compute_binary_metrics(y_true, y_pred, prob_attack, attack_label_id)
    if criterion is not None:
        metrics["loss"] = total_loss / max(total_samples, 1)

    return metrics


def train_one_config(
    config: dict,
    feature_columns: list[str],
    label_mapping: dict,
    attack_label_id: int,
    train_loader,
    val_loader,
    test_loader,
    device,
):
    trial_name = config["trial_name"]
    print("\n" + "=" * 80)
    print(f"Training trial: {trial_name}")
    print(config)

    set_seed(RANDOM_STATE)

    model = FTTransformer(
        num_features=len(feature_columns),
        num_classes=len(label_mapping),
        d_token=config["d_token"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(y)
            total_samples += len(y)

        train_loss = total_loss / max(total_samples, 1)
        val_metrics = evaluate(model, val_loader, device, attack_label_id, criterion)
        val_f1 = val_metrics["f1_attack"]

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_f1_attack={val_f1:.6f} "
            f"val_auc={val_metrics['roc_auc_attack']:.6f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_columns": feature_columns,
                "label_mapping": label_mapping,
                "model_config": {
                    "num_features": len(feature_columns),
                    "num_classes": len(label_mapping),
                    "d_token": config["d_token"],
                    "n_heads": config["n_heads"],
                    "n_layers": config["n_layers"],
                    "dim_feedforward": config["dim_feedforward"],
                    "dropout": config["dropout"],
                },
                "trial_name": trial_name,
                "best_epoch": best_epoch,
                "best_val_f1_attack": best_val_f1,
            }
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError(f"No best state saved for trial {trial_name}")

    ckpt_path = CKPT_DIR / f"revision_ablation_{trial_name}.pt"
    torch.save(best_state, ckpt_path)

    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, attack_label_id, criterion)

    result = {
        "trial_name": trial_name,
        "d_token": config["d_token"],
        "n_layers": config["n_layers"],
        "n_heads": config["n_heads"],
        "dim_feedforward": config["dim_feedforward"],
        "dropout": config["dropout"],
        "best_epoch": best_epoch,
        "best_val_f1_attack": best_val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_attack": test_metrics["precision_attack"],
        "test_recall_attack": test_metrics["recall_attack"],
        "test_f1_attack": test_metrics["f1_attack"],
        "test_roc_auc_attack": test_metrics["roc_auc_attack"],
        "test_pr_auc_attack": test_metrics["pr_auc_attack"],
        "checkpoint": str(ckpt_path),
    }

    print(f"Trial result: {result}")
    return result


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint, feature_columns, label_mapping, attack_label_id, main_config = load_main_info(device)

    print(f"Feature count: {len(feature_columns)}")
    print(f"Label mapping: {label_mapping}")
    print(f"Main config: {main_config}")

    train_loader, val_loader, test_loader = build_loaders(feature_columns)

    results = []

    # 主配置直接使用 Table 1 结果，避免 Table 1 / Table 6 再次不一致
    main_row = {
        "trial_name": "main_layers_4_heads_4",
        "d_token": 64,
        "n_layers": 4,
        "n_heads": 4,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "best_epoch": "reported_main",
        "best_val_f1_attack": 0.980400,
        "test_accuracy": 0.980675,
        "test_precision_attack": 0.996283,
        "test_recall_attack": 0.964950,
        "test_f1_attack": 0.980366,
        "test_roc_auc_attack": 0.995014,
        "test_pr_auc_attack": 0.996261,
        "checkpoint": str(MAIN_CHECKPOINT),
    }
    results.append(main_row)

    for config in ABLATION_CONFIGS:
        result = train_one_config(
            config=config,
            feature_columns=feature_columns,
            label_mapping=label_mapping,
            attack_label_id=attack_label_id,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )
        results.append(result)

    df = pd.DataFrame(results)

    out_csv = OUT_DIR / "extended_ablation_results.csv"
    out_txt = OUT_DIR / "extended_ablation_results_for_word.txt"
    out_json = OUT_DIR / "extended_ablation_results.json"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_txt.write_text(df.to_string(index=False), encoding="utf-8")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nExtended ablation results:")
    print(df.to_string(index=False))

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved TXT: {out_txt}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()