from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RANDOM_STATE
from datasets_code.tabular_dataset import create_dataloader
from models.ft_transformer import FTTransformer


# =========================
# 你主要改这里
# =========================
DATASET_DIR = PROCESSED_DATA_DIR / "ciciot2023_7class"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "ft_multiclass_best.pt"

BATCH_SIZE = 4096
NUM_WORKERS = 0
PIN_MEMORY = True
MAX_TEST_SAMPLES = None

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

METRICS_JSON_NAME = "ft_multiclass_test_metrics.json"
CLASSIFICATION_REPORT_NAME = "ft_multiclass_test_classification_report.txt"
CONFUSION_MATRIX_PNG = "ft_multiclass_confusion_matrix.png"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_confusion_matrix(cm: np.ndarray, class_names, save_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Multi-class Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_DIR)

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    feature_columns = checkpoint["feature_columns"]
    label_mapping = checkpoint["label_mapping"]

    model_config = checkpoint["model_config"]
    model = FTTransformer(
        num_features=model_config["num_features"],
        num_classes=model_config["num_classes"],
        d_token=model_config["d_token"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_csv = DATASET_DIR / "test.csv"
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

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_y_true.append(y.numpy())
            all_y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    print("Multi-class test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    with open(REPORT_DIR / METRICS_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [inverse_label_mapping[i] for i in range(len(inverse_label_mapping))]

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names, FIGURE_DIR / CONFUSION_MATRIX_PNG)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(REPORT_DIR / CLASSIFICATION_REPORT_NAME, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Metrics JSON saved to: {REPORT_DIR / METRICS_JSON_NAME}")
    print(f"Classification report saved to: {REPORT_DIR / CLASSIFICATION_REPORT_NAME}")
    print(f"Confusion matrix saved to: {FIGURE_DIR / CONFUSION_MATRIX_PNG}")


if __name__ == "__main__":
    main()