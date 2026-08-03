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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier


# ============================================================
# Revision round 2: Additional standardized NetFlow IoT validation
# ------------------------------------------------------------
# Purpose:
#   Respond to Reviewer 1:
#   The previous CICIoT2023 -> CICIoMT2024 setting is pairwise
#   and relies on feature alignment.
#
# This script adds a third external validation experiment using
# standardized NetFlow IoT datasets:
#   Train/Val/Internal Test: NF-ToN-IoT
#   External Test:          NF-BoT-IoT
#
# Models:
#   - Random Forest
#   - XGBoost
#   - MLP
#   - FT-Transformer
#   - TabTransformer-style numerical Transformer
#
# Outputs:
#   outputs/revision_round2/netflow_iot_external_validation_results.csv
#   outputs/revision_round2/netflow_iot_external_validation_results_for_word.txt
#   outputs/revision_round2/netflow_iot_feature_columns.json
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.ft_transformer import FTTransformer  # noqa: E402


RANDOM_STATE = 42

# NF 数据集通常是 Label: 0 = benign, 1 = attack
ATTACK_LABEL_ID = 1

TON_DIR = PROJECT_ROOT / "datasets" / "raw" / "nf_ton_iot"
BOT_DIR = PROJECT_ROOT / "datasets" / "raw" / "nf_bot_iot"

TON_FILE = TON_DIR / "NF-ToN-IoT.csv"
BOT_FILE = BOT_DIR / "NF-BoT-IoT.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "revision_round2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 为了 9 天内稳妥完成，这里训练最多取 200000 条。
# 外部测试默认全量。太慢再改小。
MAX_TRAIN_SAMPLES = 200_000
MAX_VAL_SAMPLES = 80_000
MAX_INTERNAL_TEST_SAMPLES = 120_000
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
RUN_TABTRANSFORMER_STYLE = True

LABEL_CANDIDATES = [
    "Label",
    "label",
    "label_id",
    "Attack",
    "attack",
    "class",
    "Class",
    "target",
    "Target",
    "y",
]

NON_FEATURE_CANDIDATES = [
    "Label",
    "label",
    "label_id",
    "Attack",
    "attack",
    "Class",
    "class",
    "target",
    "Target",
    "y",
    "Dataset",
    "dataset",
    "Attack Type",
    "Attack_Type",
    "attack_type",
    "Category",
    "category",
]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auto_find_csv(expected_path: Path, keyword: str) -> Path:
    if expected_path.exists():
        return expected_path

    candidates = list(expected_path.parent.glob("*.csv"))
    for p in candidates:
        if keyword.lower() in p.name.lower():
            return p

    raise FileNotFoundError(
        f"Cannot find {expected_path}. "
        f"Please put the CSV into: {expected_path.parent}"
    )


def detect_label_column(df: pd.DataFrame) -> str:
    # 优先使用数值型 Label，通常 0/1
    for col in ["Label", "label", "label_id"]:
        if col in df.columns:
            return col

    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(f"Cannot detect label column. Columns: {df.columns.tolist()}")


def make_binary_labels(series: pd.Series) -> np.ndarray:
    """
    Robust binary label mapping:
    - If label is numeric 0/1, use it directly.
    - If label is string, map benign/normal to 0 and others to 1.
    """
    s = series.copy()

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.95:
        y = numeric.fillna(0).astype(int).values
        unique_values = sorted(np.unique(y).tolist())

        # 如果正好是 0/1，直接用
        if set(unique_values).issubset({0, 1}):
            return y

        # 如果不是 0/1，最小值当 benign，其余当 attack
        min_value = min(unique_values)
        return (y != min_value).astype(int)

    s_text = s.astype(str).str.lower().str.strip()

    benign_keywords = ["benign", "normal", "0"]
    y = np.ones(len(s_text), dtype=int)

    for kw in benign_keywords:
        y[s_text == kw] = 0

    return y


def read_csv_safely(path: Path) -> pd.DataFrame:
    print(f"Reading CSV: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    # 去除列名多余空格
    df.columns = [str(c).strip() for c in df.columns]

    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def infer_common_numeric_features(df_train: pd.DataFrame, df_external: pd.DataFrame, label_col_train: str, label_col_external: str):
    exclude_train = set(NON_FEATURE_CANDIDATES + [label_col_train])
    exclude_external = set(NON_FEATURE_CANDIDATES + [label_col_external])

    train_numeric = []
    for col in df_train.columns:
        if col in exclude_train:
            continue
        converted = pd.to_numeric(df_train[col], errors="coerce")
        if converted.notna().mean() > 0.95:
            train_numeric.append(col)

    external_numeric = []
    for col in df_external.columns:
        if col in exclude_external:
            continue
        converted = pd.to_numeric(df_external[col], errors="coerce")
        if converted.notna().mean() > 0.95:
            external_numeric.append(col)

    common_features = sorted(list(set(train_numeric).intersection(set(external_numeric))))

    if not common_features:
        raise ValueError("No common numerical features found between NF-ToN-IoT and NF-BoT-IoT.")

    print("\nDetected common numerical feature columns:")
    for i, col in enumerate(common_features, start=1):
        print(f"{i:02d}. {col}")

    print(f"\nFeature count: {len(common_features)}")

    return common_features


def build_xy(df: pd.DataFrame, feature_cols: list[str], label_col: str):
    X_df = pd.DataFrame()

    for col in feature_cols:
        X_df[col] = pd.to_numeric(df[col], errors="coerce")

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(0.0)

    X = X_df.values.astype(np.float32)
    y = make_binary_labels(df[label_col])

    return X, y


def stratified_sample(X: np.ndarray, y: np.ndarray, max_samples: int | None, random_state: int = 42):
    if max_samples is None or len(y) <= max_samples:
        return X, y

    indices = np.arange(len(y))

    sampled_idx, _ = train_test_split(
        indices,
        train_size=max_samples,
        stratify=y,
        random_state=random_state,
    )

    return X[sampled_idx], y[sampled_idx]


def split_ton_dataset(X: np.ndarray, y: np.ndarray):
    """
    Split NF-ToN-IoT into train/val/internal test.
    Ratio: 70/15/15.
    """
    indices = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=y[temp_idx],
        random_state=RANDOM_STATE,
    )

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def compute_metrics(y_true, y_pred, prob_attack, attack_label_id: int = 1) -> dict:
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


class NumericTabTransformerStyle(nn.Module):
    """
    TabTransformer-style numerical Transformer baseline.
    Each numerical feature is projected into a token, followed by Transformer Encoder.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))
        nn.init.xavier_uniform_(self.weight)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, num_classes),
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0, :])


@torch.no_grad()
def evaluate_torch_model(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device):
    model.eval()

    all_pred = []
    all_prob_attack = []

    start_time = time.perf_counter()

    for start in range(0, len(X), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X))
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
    print(f"Training {model_name}")
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

        val_metrics = evaluate_torch_model(model, X_val, y_val, device)
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
        metrics = evaluate_torch_model(model, X_test, y_test, device)

        row = {
            "protocol": "standardized_netflow_iot_external_validation",
            "train_dataset": "NF-ToN-IoT",
            "external_dataset": "NF-BoT-IoT",
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
    print("Training Random Forest")
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

        classes = list(model.classes_)
        attack_col = classes.index(ATTACK_LABEL_ID)
        prob_attack = prob[:, attack_col]

        infer_time = time.perf_counter() - infer_start

        metrics = compute_metrics(y_test, pred, prob_attack, ATTACK_LABEL_ID)
        metrics["inference_time_seconds"] = float(infer_time)
        metrics["inference_ms_per_sample"] = float(infer_time * 1000 / len(X_test))

        row = {
            "protocol": "standardized_netflow_iot_external_validation",
            "train_dataset": "NF-ToN-IoT",
            "external_dataset": "NF-BoT-IoT",
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
    print("Training XGBoost")
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

        # XGBoost 默认类别顺序是 0,1
        prob_attack = prob[:, 1]

        infer_time = time.perf_counter() - infer_start

        metrics = compute_metrics(y_test, pred, prob_attack, ATTACK_LABEL_ID)
        metrics["inference_time_seconds"] = float(infer_time)
        metrics["inference_ms_per_sample"] = float(infer_time * 1000 / len(X_test))

        row = {
            "protocol": "standardized_netflow_iot_external_validation",
            "train_dataset": "NF-ToN-IoT",
            "external_dataset": "NF-BoT-IoT",
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
    set_seed(RANDOM_STATE)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using device: {get_device()}")

    ton_path = auto_find_csv(TON_FILE, "ton")
    bot_path = auto_find_csv(BOT_FILE, "bot")

    df_ton = read_csv_safely(ton_path)
    df_bot = read_csv_safely(bot_path)

    label_col_ton = detect_label_column(df_ton)
    label_col_bot = detect_label_column(df_bot)

    print(f"NF-ToN-IoT label column: {label_col_ton}")
    print(f"NF-BoT-IoT label column: {label_col_bot}")

    feature_cols = infer_common_numeric_features(
        df_train=df_ton,
        df_external=df_bot,
        label_col_train=label_col_ton,
        label_col_external=label_col_bot,
    )

    X_ton, y_ton = build_xy(df_ton, feature_cols, label_col_ton)
    X_bot, y_bot = build_xy(df_bot, feature_cols, label_col_bot)

    print("\nRaw label distributions:")
    print("NF-ToN-IoT:", dict(zip(*np.unique(y_ton, return_counts=True))))
    print("NF-BoT-IoT:", dict(zip(*np.unique(y_bot, return_counts=True))))

    X_train, y_train, X_val, y_val, X_internal_test, y_internal_test = split_ton_dataset(X_ton, y_ton)

    # 训练/验证/内部测试抽样，外部测试默认全量
    X_train, y_train = stratified_sample(X_train, y_train, MAX_TRAIN_SAMPLES, RANDOM_STATE)
    X_val, y_val = stratified_sample(X_val, y_val, MAX_VAL_SAMPLES, RANDOM_STATE)
    X_internal_test, y_internal_test = stratified_sample(
        X_internal_test,
        y_internal_test,
        MAX_INTERNAL_TEST_SAMPLES,
        RANDOM_STATE,
    )
    X_external_test, y_external_test = stratified_sample(
        X_bot,
        y_bot,
        MAX_EXTERNAL_TEST_SAMPLES,
        RANDOM_STATE,
    )

    print("\nAfter splitting/sampling:")
    print("Train:", X_train.shape, dict(zip(*np.unique(y_train, return_counts=True))))
    print("Val:", X_val.shape, dict(zip(*np.unique(y_val, return_counts=True))))
    print("Internal test:", X_internal_test.shape, dict(zip(*np.unique(y_internal_test, return_counts=True))))
    print("External test:", X_external_test.shape, dict(zip(*np.unique(y_external_test, return_counts=True))))

    # StandardScaler: fit only on NF-ToN-IoT train split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_internal_test = scaler.transform(X_internal_test).astype(np.float32)
    X_external_test = scaler.transform(X_external_test).astype(np.float32)

    test_sets = {
        "NF-ToN-IoT internal test": (X_internal_test, y_internal_test),
        "NF-BoT-IoT external test": (X_external_test, y_external_test),
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

    if RUN_TABTRANSFORMER_STYLE:
        tab_model = NumericTabTransformerStyle(
            num_features=X_train.shape[1],
            num_classes=2,
            d_token=64,
            n_heads=4,
            n_layers=3,
            dim_feedforward=128,
            dropout=0.1,
        )

        rows.extend(
            train_torch_model(
                model_name="TabTransformer-style",
                model=tab_model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                test_sets=test_sets,
                device=device,
            )
        )

    df_results = pd.DataFrame(rows)

    out_csv = OUT_DIR / "netflow_iot_external_validation_results.csv"
    out_txt = OUT_DIR / "netflow_iot_external_validation_results_for_word.txt"
    out_json = OUT_DIR / "netflow_iot_external_validation_results.json"
    out_features = OUT_DIR / "netflow_iot_feature_columns.json"

    df_results.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_txt.write_text(df_results.to_string(index=False), encoding="utf-8")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(out_features, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_dataset": "NF-ToN-IoT",
                "external_dataset": "NF-BoT-IoT",
                "feature_count": len(feature_cols),
                "feature_columns": feature_cols,
                "label_column_ton": label_col_ton,
                "label_column_bot": label_col_bot,
                "attack_label_id": ATTACK_LABEL_ID,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFinal results:")
    print(df_results.to_string(index=False))

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved TXT: {out_txt}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved features: {out_features}")


if __name__ == "__main__":
    main()