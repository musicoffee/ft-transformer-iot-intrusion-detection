from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
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
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier


# ============================================================
# Revision round 2: feature-union external validation
# ------------------------------------------------------------
# Purpose:
#   Respond to Reviewer 1:
#   The original feature-intersection protocol may remove
#   dataset discrepancy in advance.
#
# This script creates a stricter additional validation protocol:
#   1. Use the union of numerical feature columns from CICIoT2023
#      and CICIoMT2024.
#   2. For missing features in a dataset, fill values with 0.
#   3. Add missing-feature indicator columns.
#   4. Train on CICIoT2023 and evaluate on:
#      - CICIoT2023 test set
#      - CICIoMT2024 external test set
#
# Models:
#   - Random Forest
#   - XGBoost
#   - MLP
#   - FT-Transformer from project model
#
# Outputs:
#   outputs/revision_round2/feature_union_validation_results.csv
#   outputs/revision_round2/feature_union_validation_results_for_word.txt
#   outputs/revision_round2/feature_union_feature_list.json
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.ft_transformer import FTTransformer  # noqa: E402


RANDOM_STATE = 42
ATTACK_LABEL_ID = 0

MAIN_DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_binary"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "ciciomt2024_binary"

TRAIN_FILE = MAIN_DATA_DIR / "train.csv"
VAL_FILE = MAIN_DATA_DIR / "val.csv"
MAIN_TEST_FILE = MAIN_DATA_DIR / "test.csv"
EXTERNAL_TEST_FILE = EXTERNAL_DATA_DIR / "test.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "revision_round2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAIN_SAMPLES = 200_000
MAX_VAL_SAMPLES = None
MAX_MAIN_TEST_SAMPLES = None
MAX_EXTERNAL_TEST_SAMPLES = None

BATCH_SIZE = 4096
EPOCHS = 15
PATIENCE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4

RUN_RF = True
RUN_XGB = True
RUN_MLP = True
RUN_FT_TRANSFORMER = True

LABEL_CANDIDATES = [
    "label_id",
    "label",
    "Label",
    "label_text",
    "Class",
    "class",
    "Category",
    "category",
    "target",
    "Target",
    "y",
]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_label_column(df: pd.DataFrame) -> str:
    if "label_id" in df.columns:
        return "label_id"
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot detect label column. Columns: {df.columns.tolist()[:50]}")


def read_header_numeric_columns(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, nrows=1000)
    label_col = detect_label_column(df)

    numeric_cols = []
    for col in df.columns:
        if col == label_col or col in LABEL_CANDIDATES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols


def build_union_features() -> tuple[list[str], list[str], list[str]]:
    main_cols = set()
    external_cols = set()

    for path in [TRAIN_FILE, VAL_FILE, MAIN_TEST_FILE]:
        main_cols.update(read_header_numeric_columns(path))

    external_cols.update(read_header_numeric_columns(EXTERNAL_TEST_FILE))

    union_cols = sorted(main_cols.union(external_cols))
    shared_cols = sorted(main_cols.intersection(external_cols))

    print(f"Main numeric feature count: {len(main_cols)}")
    print(f"External numeric feature count: {len(external_cols)}")
    print(f"Shared feature count: {len(shared_cols)}")
    print(f"Union feature count: {len(union_cols)}")

    return union_cols, sorted(main_cols), sorted(external_cols)


def stratified_sample_df(
    df: pd.DataFrame,
    label_col: str,
    n_samples: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_samples is None or len(df) <= n_samples:
        return df.reset_index(drop=True)

    class_counts = df[label_col].value_counts()
    total = len(df)

    allocated = {}
    for cls, count in class_counts.items():
        cls_n = int(round(n_samples * count / total))
        cls_n = max(1, min(cls_n, count))
        allocated[cls] = cls_n

    diff = n_samples - sum(allocated.values())
    sorted_classes = class_counts.sort_values(ascending=False).index.tolist()
    idx = 0

    while diff != 0 and idx < 100000:
        cls = sorted_classes[idx % len(sorted_classes)]

        if diff > 0:
            if allocated[cls] < class_counts[cls]:
                allocated[cls] += 1
                diff -= 1
        else:
            if allocated[cls] > 1:
                allocated[cls] -= 1
                diff += 1

        idx += 1

    parts = []
    for cls, cls_n in allocated.items():
        parts.append(
            df[df[label_col] == cls].sample(
                n=cls_n,
                random_state=random_state,
                replace=False,
            )
        )

    sampled = pd.concat(parts, axis=0)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return sampled


def make_union_matrix(
    df: pd.DataFrame,
    union_features: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    For each union feature:
      - if present: use its value
      - if missing: fill with 0

    For each union feature:
      - add missing indicator:
        1 means this feature is missing in the current dataset
        0 means this feature is present in the current dataset
    """
    n = len(df)

    feature_arrays = []
    output_feature_names = []

    for feat in union_features:
        if feat in df.columns:
            arr = pd.to_numeric(df[feat], errors="coerce").fillna(0).values.astype(np.float32)
        else:
            arr = np.zeros(n, dtype=np.float32)

        feature_arrays.append(arr.reshape(-1, 1))
        output_feature_names.append(feat)

    for feat in union_features:
        missing_value = 0.0 if feat in df.columns else 1.0
        arr = np.full(n, missing_value, dtype=np.float32)
        feature_arrays.append(arr.reshape(-1, 1))
        output_feature_names.append(f"{feat}__missing")

    X = np.hstack(feature_arrays).astype(np.float32)

    return X, output_feature_names


def load_union_xy(
    csv_path: Path,
    union_features: list[str],
    max_samples: int | None,
    random_state: int = 42,
):
    print(f"Loading union matrix from: {csv_path}")

    df = pd.read_csv(csv_path)
    label_col = detect_label_column(df)

    df = stratified_sample_df(df, label_col, max_samples, random_state)

    y = df[label_col].values.astype(int)
    X, output_feature_names = make_union_matrix(df, union_features)

    # 防止极端值和 NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    print(f"Union X shape: {X.shape}, y shape: {y.shape}")
    print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))

    return X, y, output_feature_names


def compute_metrics(y_true, y_pred, prob_attack, attack_label_id: int = 0) -> dict:
    y_true_attack = (y_true == attack_label_id).astype(int)
    y_pred_attack = (y_pred == attack_label_id).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(
            precision_score(y_true_attack, y_pred_attack, zero_division=0)
        ),
        "recall_attack": float(
            recall_score(y_true_attack, y_pred_attack, zero_division=0)
        ),
        "f1_attack": float(f1_score(y_true_attack, y_pred_attack, zero_division=0)),
        "roc_auc_attack": float(roc_auc_score(y_true_attack, prob_attack)),
        "pr_auc_attack": float(average_precision_score(y_true_attack, prob_attack)),
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class MLPBaseline(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_dim = num_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate_torch_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
):
    model.eval()

    all_pred = []
    all_prob_attack = []

    start_time = time.perf_counter()

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)

        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_pred.append(preds.detach().cpu().numpy())
        all_prob_attack.append(probs[:, ATTACK_LABEL_ID].detach().cpu().numpy())

    infer_time = time.perf_counter() - start_time

    y_pred = np.concatenate(all_pred).astype(int)
    prob_attack = np.concatenate(all_prob_attack).astype(float)

    metrics = compute_metrics(y, y_pred, prob_attack, ATTACK_LABEL_ID)
    metrics["inference_time_seconds"] = float(infer_time)
    metrics["inference_ms_per_sample"] = float(infer_time * 1000 / len(X))

    return metrics


def train_torch_model(
    model_name: str,
    model: nn.Module,
    X_train,
    y_train,
    X_val,
    y_val,
    test_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
):
    print("\n" + "=" * 90)
    print(f"Training {model_name} under feature-union protocol")
    print("=" * 90)

    set_seed(RANDOM_STATE)
    model = model.to(device)

    param_count = count_parameters(model)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    train_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(yb)
            total_samples += len(yb)

        train_loss = total_loss / max(total_samples, 1)

        val_metrics = evaluate_torch_model(model, X_val, y_val, device, BATCH_SIZE)
        val_f1 = val_metrics["f1_attack"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.5f} "
            f"val_f1_attack={val_f1:.6f} "
            f"val_auc={val_metrics['roc_auc_attack']:.6f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    training_time = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    rows = []

    for eval_name, (X_test, y_test) in test_sets.items():
        metrics = evaluate_torch_model(model, X_test, y_test, device, BATCH_SIZE)

        row = {
            "protocol": "feature_union_with_missing_indicators",
            "model": model_name,
            "eval_set": eval_name,
            "num_features": int(X_train.shape[1]),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "parameters": int(param_count),
            "best_epoch": int(best_epoch),
            "best_val_f1_attack": float(best_val_f1),
            "training_time_seconds": float(training_time),
        }
        row.update(metrics)
        rows.append(row)

        print(f"\n{model_name} | {eval_name}")
        print(metrics)

    return rows


def train_eval_rf(X_train, y_train, test_sets):
    print("\n" + "=" * 90)
    print("Training Random Forest under feature-union protocol")
    print("=" * 90)

    start_time = time.perf_counter()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    rows = []

    for eval_name, (X_test, y_test) in test_sets.items():
        infer_start = time.perf_counter()

        pred = model.predict(X_test).astype(int)
        prob = model.predict_proba(X_test)
        prob_attack = prob[:, list(model.classes_).index(ATTACK_LABEL_ID)]

        infer_time = time.perf_counter() - infer_start

        metrics = compute_metrics(y_test, pred, prob_attack, ATTACK_LABEL_ID)
        metrics["inference_time_seconds"] = float(infer_time)
        metrics["inference_ms_per_sample"] = float(infer_time * 1000 / len(X_test))

        row = {
            "protocol": "feature_union_with_missing_indicators",
            "model": "Random Forest",
            "eval_set": eval_name,
            "num_features": int(X_train.shape[1]),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "parameters": None,
            "best_epoch": None,
            "best_val_f1_attack": None,
            "training_time_seconds": float(training_time),
        }
        row.update(metrics)
        rows.append(row)

        print(f"\nRandom Forest | {eval_name}")
        print(metrics)

    return rows


def train_eval_xgb(X_train, y_train, test_sets):
    print("\n" + "=" * 90)
    print("Training XGBoost under feature-union protocol")
    print("=" * 90)

    start_time = time.perf_counter()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    rows = []

    for eval_name, (X_test, y_test) in test_sets.items():
        infer_start = time.perf_counter()

        pred = model.predict(X_test).astype(int)
        prob = model.predict_proba(X_test)
        prob_attack = prob[:, ATTACK_LABEL_ID]

        infer_time = time.perf_counter() - infer_start

        metrics = compute_metrics(y_test, pred, prob_attack, ATTACK_LABEL_ID)
        metrics["inference_time_seconds"] = float(infer_time)
        metrics["inference_ms_per_sample"] = float(infer_time * 1000 / len(X_test))

        row = {
            "protocol": "feature_union_with_missing_indicators",
            "model": "XGBoost",
            "eval_set": eval_name,
            "num_features": int(X_train.shape[1]),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "parameters": None,
            "best_epoch": None,
            "best_val_f1_attack": None,
            "training_time_seconds": float(training_time),
        }
        row.update(metrics)
        rows.append(row)

        print(f"\nXGBoost | {eval_name}")
        print(metrics)

    return rows


def main():
    warnings.filterwarnings("ignore")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using device: {get_device()}")

    union_features, main_features, external_features = build_union_features()

    feature_info = {
        "main_feature_count": len(main_features),
        "external_feature_count": len(external_features),
        "shared_feature_count": len(set(main_features).intersection(set(external_features))),
        "union_feature_count": len(union_features),
        "final_feature_count_with_missing_indicators": len(union_features) * 2,
        "union_features": union_features,
        "main_features": main_features,
        "external_features": external_features,
    }

    with open(OUT_DIR / "feature_union_feature_list.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    X_train, y_train, union_output_features = load_union_xy(
        TRAIN_FILE,
        union_features,
        max_samples=MAX_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
    )

    X_val, y_val, _ = load_union_xy(
        VAL_FILE,
        union_features,
        max_samples=MAX_VAL_SAMPLES,
        random_state=RANDOM_STATE,
    )

    X_main_test, y_main_test, _ = load_union_xy(
        MAIN_TEST_FILE,
        union_features,
        max_samples=MAX_MAIN_TEST_SAMPLES,
        random_state=RANDOM_STATE,
    )

    X_external_test, y_external_test, _ = load_union_xy(
        EXTERNAL_TEST_FILE,
        union_features,
        max_samples=MAX_EXTERNAL_TEST_SAMPLES,
        random_state=RANDOM_STATE,
    )

    with open(OUT_DIR / "feature_union_final_input_features.json", "w", encoding="utf-8") as f:
        json.dump(union_output_features, f, ensure_ascii=False, indent=2)

    test_sets = {
        "main_test": (X_main_test, y_main_test),
        "external_test": (X_external_test, y_external_test),
    }

    rows = []

    if RUN_RF:
        rows.extend(train_eval_rf(X_train, y_train, test_sets))

    if RUN_XGB:
        rows.extend(train_eval_xgb(X_train, y_train, test_sets))

    device = get_device()

    if RUN_MLP:
        mlp = MLPBaseline(
            num_features=X_train.shape[1],
            num_classes=2,
            hidden_dims=(256, 128, 64),
            dropout=0.2,
        )

        rows.extend(
            train_torch_model(
                model_name="MLP",
                model=mlp,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                test_sets=test_sets,
                device=device,
            )
        )

    if RUN_FT_TRANSFORMER:
        ft_model = FTTransformer(
            num_features=X_train.shape[1],
            num_classes=2,
            d_token=64,
            n_heads=4,
            n_layers=4,
            dim_feedforward=128,
            dropout=0.1,
        )

        rows.extend(
            train_torch_model(
                model_name="FT-Transformer",
                model=ft_model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                test_sets=test_sets,
                device=device,
            )
        )

    df = pd.DataFrame(rows)

    out_csv = OUT_DIR / "feature_union_validation_results.csv"
    out_txt = OUT_DIR / "feature_union_validation_results_for_word.txt"
    out_json = OUT_DIR / "feature_union_validation_results.json"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_txt.write_text(df.to_string(index=False), encoding="utf-8")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\nFeature-union validation results:")
    print(df.to_string(index=False))

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved TXT: {out_txt}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved feature info: {OUT_DIR / 'feature_union_feature_list.json'}")


if __name__ == "__main__":
    main()