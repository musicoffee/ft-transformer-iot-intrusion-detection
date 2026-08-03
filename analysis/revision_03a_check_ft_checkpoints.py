from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.ft_transformer import FTTransformer  # noqa: E402


DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_binary"
TEST_FILE = DATA_DIR / "test.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_DIR = PROJECT_ROOT / "outputs" / "revision_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4096
ATTACK_LABEL_ID = 0

LABEL_CANDIDATES = [
    "label_id", "Label", "label", "label_text",
    "Class", "class", "Category", "category",
    "target", "Target", "y",
]


def detect_label_column(df: pd.DataFrame) -> str:
    if "label_id" in df.columns:
        return "label_id"
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot detect label column. Columns: {df.columns.tolist()}")


def get_numeric_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    exclude = set(LABEL_CANDIDATES)
    exclude.add(label_col)

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def load_test_data():
    df_test = pd.read_csv(TEST_FILE)
    label_col = detect_label_column(df_test)
    feature_cols = get_numeric_feature_columns(df_test, label_col)

    X = df_test[feature_cols].values.astype(np.float32)
    y = df_test[label_col].values.astype(int)

    print(f"Test file: {TEST_FILE}")
    print(f"Test shape: {X.shape}")
    print(f"Label column: {label_col}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))

    return X, y, feature_cols


def load_state_dict(path: Path):
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]

        tensor_values = [v for v in ckpt.values() if torch.is_tensor(v)]
        if len(tensor_values) > 0:
            return ckpt

    raise ValueError(f"Unknown checkpoint format: {path}")


def build_model(num_features: int, num_classes: int = 2):
    return FTTransformer(
        num_features=num_features,
        num_classes=num_classes,
        d_token=64,
        n_heads=4,
        n_layers=4,
        dim_feedforward=128,
        dropout=0.1,
    )


@torch.no_grad()
def predict_checkpoint(path: Path, X: np.ndarray, num_features: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = load_state_dict(path)
    model = build_model(num_features=num_features, num_classes=2)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    preds = []
    probs_attack = []

    for start in range(0, len(X), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X))
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)

        logits = model(xb)
        if isinstance(logits, tuple):
            logits = logits[0]

        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        preds.append(pred.detach().cpu().numpy())
        probs_attack.append(prob[:, ATTACK_LABEL_ID].detach().cpu().numpy())

    y_pred = np.concatenate(preds).astype(int)
    y_prob_attack = np.concatenate(probs_attack).astype(float)

    return y_pred, y_prob_attack


def compute_metrics(y_true, y_pred, y_prob_attack):
    y_true_attack = (y_true == ATTACK_LABEL_ID).astype(int)
    y_pred_attack = (y_pred == ATTACK_LABEL_ID).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_attack": precision_score(y_true_attack, y_pred_attack, zero_division=0),
        "recall_attack": recall_score(y_true_attack, y_pred_attack, zero_division=0),
        "f1_attack": f1_score(y_true_attack, y_pred_attack, zero_division=0),
        "roc_auc_attack": roc_auc_score(y_true_attack, y_prob_attack),
        "pr_auc_attack": average_precision_score(y_true_attack, y_prob_attack),
    }


def main():
    X, y, feature_cols = load_test_data()

    rows = []

    for ckpt_path in sorted(CHECKPOINT_DIR.glob("*.pt")):
        print("\n" + "=" * 80)
        print(f"Testing checkpoint: {ckpt_path.name}")

        try:
            y_pred, y_prob_attack = predict_checkpoint(
                path=ckpt_path,
                X=X,
                num_features=len(feature_cols),
            )

            metrics = compute_metrics(y, y_pred, y_prob_attack)
            row = {"checkpoint": ckpt_path.name, **metrics}
            rows.append(row)

            print(row)

        except Exception as e:
            print(f"[SKIP] {ckpt_path.name}: {repr(e)}")
            rows.append({
                "checkpoint": ckpt_path.name,
                "error": repr(e),
            })

    df = pd.DataFrame(rows)

    out_csv = OUT_DIR / "ft_checkpoint_diagnosis.csv"
    out_txt = OUT_DIR / "ft_checkpoint_diagnosis_for_word.txt"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_txt.write_text(df.to_string(index=False), encoding="utf-8")

    print("\nDone.")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved TXT: {out_txt}")


if __name__ == "__main__":
    main()