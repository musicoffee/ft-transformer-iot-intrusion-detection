from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RANDOM_STATE
from models.ft_transformer import FTTransformer

try:
    import shap
except ImportError:
    shap = None

DATASET_DIR = PROCESSED_DATA_DIR / "ciciot2023_binary"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "ft_binary_best.pt"

BACKGROUND_SAMPLES = 128
EXPLAIN_SAMPLES = 200

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

SHAP_BAR_PNG = "ft_binary_shap_bar.png"
SHAP_SUMMARY_PNG = "ft_binary_shap_summary.png"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_numpy_from_csv(csv_path: Path, feature_columns, max_samples=None, random_state=42):
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=feature_columns + ["label_id"])
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    x = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["label_id"].to_numpy(dtype=np.int64)
    return x, y


def main():
    if shap is None:
        raise ImportError("未安装 shap。请先运行：pip install shap")

    ensure_dir(FIGURE_DIR)

    device = get_device()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    feature_columns = checkpoint["feature_columns"]
    attack_label_id = int(checkpoint["label_mapping"]["attack"])
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

    train_csv = DATASET_DIR / "train.csv"
    test_csv = DATASET_DIR / "test.csv"

    x_background, _ = load_numpy_from_csv(
        train_csv, feature_columns, max_samples=BACKGROUND_SAMPLES, random_state=RANDOM_STATE
    )
    x_explain, _ = load_numpy_from_csv(
        test_csv, feature_columns, max_samples=EXPLAIN_SAMPLES, random_state=RANDOM_STATE
    )

    def predict_attack_proba(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs[:, attack_label_id].detach().cpu().numpy()

    print("SHAP background shape:", x_background.shape)
    print("SHAP explain shape:", x_explain.shape)

    explainer = shap.KernelExplainer(predict_attack_proba, x_background)
    shap_values = explainer.shap_values(x_explain, nsamples=100)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[0] if len(shap_values) == 1 else shap_values
        if isinstance(shap_values_to_plot, list):
            shap_values_to_plot = shap_values_to_plot[0]
    else:
        shap_values_to_plot = shap_values

    plt.figure()
    shap.summary_plot(
        shap_values_to_plot,
        x_explain,
        feature_names=feature_columns,
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / SHAP_BAR_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values_to_plot,
        x_explain,
        feature_names=feature_columns,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / SHAP_SUMMARY_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"SHAP bar plot saved to: {FIGURE_DIR / SHAP_BAR_PNG}")
    print(f"SHAP summary plot saved to: {FIGURE_DIR / SHAP_SUMMARY_PNG}")


if __name__ == "__main__":
    main()