from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Revision round 3:
# Domain-Aligned FT-Transformer with CORAL representation loss
# ------------------------------------------------------------
# Purpose:
#   Respond to Reviewer 1:
#   "What training methodology or representation-learning strategy
#    is actually required to achieve meaningful cross-dataset
#    generalization?"
#
# This script evaluates:
#   1. Source-only FT-Transformer
#   2. Domain-aligned FT-Transformer-CORAL
#
# Scenarios:
#   A. CICIoT2023 -> CICIoMT2024
#      using 38-feature aligned dataset
#
#   B. NF-ToN-IoT -> NF-BoT-IoT
#      using standardized NetFlow IoT datasets
#
# Important:
#   Target labels are NOT used during CORAL training.
#   The target dataset is split into:
#       - target adaptation split: features only, no labels used
#       - target test split: labels used only for final evaluation
#
# Outputs:
#   outputs/revision_round3/domain_aligned_ft_transformer_results.csv
#   outputs/revision_round3/domain_aligned_ft_transformer_results_for_word.txt
#   outputs/revision_round3/domain_aligned_ft_transformer_results.json
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

RANDOM_STATE = 42

OUT_DIR = PROJECT_ROOT / "outputs" / "revision_round3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Data paths
# -----------------------------
ALIGNED_DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "cross_dataset_aligned_binary"

CIC_SOURCE_TRAIN = ALIGNED_DATA_DIR / "main_train_aligned.csv"
CIC_SOURCE_VAL = ALIGNED_DATA_DIR / "main_val_aligned.csv"
CIC_SOURCE_TEST = ALIGNED_DATA_DIR / "main_test_aligned.csv"
CIC_TARGET_EXTERNAL = ALIGNED_DATA_DIR / "external_test_aligned.csv"

NF_TON_DIR = PROJECT_ROOT / "datasets" / "raw" / "nf_ton_iot"
NF_BOT_DIR = PROJECT_ROOT / "datasets" / "raw" / "nf_bot_iot"
NF_TON_FILE = NF_TON_DIR / "NF-ToN-IoT.csv"
NF_BOT_FILE = NF_BOT_DIR / "NF-BoT-IoT.csv"

# -----------------------------
# Training controls
# -----------------------------
BATCH_SIZE = 4096
EPOCHS = 15
PATIENCE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Source train 最大采样数。时间紧就改 100_000。
MAX_SOURCE_TRAIN_SAMPLES = 200_000

# source val / source test 最大采样数。None 表示尽量全量。
MAX_SOURCE_VAL_SAMPLES = 80_000
MAX_SOURCE_TEST_SAMPLES = 120_000

# 目标域无标签适配样本数。时间紧就改 100_000。
MAX_TARGET_ADAPT_SAMPLES = 200_000

# 目标域测试样本数。None 表示全量。太慢就改 300_000。
MAX_TARGET_TEST_SAMPLES = None

# CORAL 损失权重。
# 先跑 0.05 比较稳。如果想更全面，可以改成 [0.01, 0.05, 0.1]
CORAL_LAMBDAS = [0.05]

# 是否同时跑 source-only baseline。
RUN_SOURCE_ONLY = True
RUN_CORAL = True

LABEL_CANDIDATES = [
    "label_id",
    "Label",
    "label",
    "Attack",
    "attack",
    "Class",
    "class",
    "target",
    "Target",
    "y",
]

NON_FEATURE_CANDIDATES = [
    "label_id",
    "Label",
    "label",
    "label_text",
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


# ============================================================
# Utility functions
# ============================================================

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
    raise ValueError(f"Cannot detect label column. Columns: {df.columns.tolist()[:80]}")


def read_csv_safely(path: Path) -> pd.DataFrame:
    print(f"Reading CSV: {path}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded shape: {df.shape}")
    return df


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


def make_binary_labels(series: pd.Series) -> np.ndarray:
    """
    Robust binary mapping:
    - Numeric 0/1: use directly.
    - Other numeric labels: minimum value = benign, others = attack.
    - String labels: benign/normal/0 = benign, others = attack.
    """
    s = series.copy()

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.95:
        y = numeric.fillna(0).astype(int).values
        unique_values = sorted(np.unique(y).tolist())

        if set(unique_values).issubset({0, 1}):
            return y

        min_value = min(unique_values)
        return (y != min_value).astype(int)

    s_text = s.astype(str).str.lower().str.strip()
    y = np.ones(len(s_text), dtype=int)

    benign_keywords = {"benign", "normal", "0"}
    y[s_text.isin(benign_keywords)] = 0

    return y


def load_json_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["feature_columns", "shared_feature_columns", "features"]:
            if key in obj:
                return obj[key]

    raise ValueError(f"Unsupported feature json format: {path}")


def infer_numeric_features_from_df(
    df: pd.DataFrame,
    label_col: str,
    required_common: Optional[set[str]] = None,
) -> list[str]:
    exclude = set(NON_FEATURE_CANDIDATES + [label_col])
    feature_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        if required_common is not None and col not in required_common:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.95:
            feature_cols.append(col)

    return feature_cols


def stratified_sample_xy(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
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


def stratified_sample_df(
    df: pd.DataFrame,
    label_col: str,
    max_samples: Optional[int],
    random_state: int = 42,
) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df.reset_index(drop=True)

    indices = np.arange(len(df))
    sampled_idx, _ = train_test_split(
        indices,
        train_size=max_samples,
        stratify=df[label_col].values,
        random_state=random_state,
    )
    return df.iloc[sampled_idx].reset_index(drop=True)


def split_target_for_adaptation(
    X_target: np.ndarray,
    y_target: np.ndarray,
    max_adapt_samples: Optional[int],
    max_test_samples: Optional[int],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split target dataset into:
      - unlabeled target adaptation set: only X used for CORAL
      - held-out target test set: X and y used only for evaluation

    We use labels only for stratified splitting and sampling,
    not for optimization.
    """
    indices = np.arange(len(y_target))

    adapt_idx, test_idx = train_test_split(
        indices,
        test_size=0.50,
        stratify=y_target,
        random_state=random_state,
    )

    X_adapt = X_target[adapt_idx]
    y_adapt = y_target[adapt_idx]

    X_test = X_target[test_idx]
    y_test = y_target[test_idx]

    X_adapt, y_adapt = stratified_sample_xy(
        X_adapt,
        y_adapt,
        max_adapt_samples,
        random_state=random_state,
    )

    X_test, y_test = stratified_sample_xy(
        X_test,
        y_test,
        max_test_samples,
        random_state=random_state,
    )

    return X_adapt, X_test, y_test


def build_xy_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    force_binary_label: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    X_df = pd.DataFrame()

    for col in feature_cols:
        X_df[col] = pd.to_numeric(df[col], errors="coerce")

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(0.0)

    X = X_df.values.astype(np.float32)

    if force_binary_label:
        y = make_binary_labels(df[label_col])
    else:
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).values

    return X, y


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_attack: np.ndarray,
    attack_label_id: int,
) -> dict:
    y_true_attack = (y_true == attack_label_id).astype(int)
    y_pred_attack = (y_pred == attack_label_id).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(
            precision_score(y_true_attack, y_pred_attack, zero_division=0)
        ),
        "recall_attack": float(
            recall_score(y_true_attack, y_pred_attack, zero_division=0)
        ),
        "f1_attack": float(
            f1_score(y_true_attack, y_pred_attack, zero_division=0)
        ),
    }

    try:
        metrics["roc_auc_attack"] = float(roc_auc_score(y_true_attack, prob_attack))
    except Exception:
        metrics["roc_auc_attack"] = float("nan")

    try:
        metrics["pr_auc_attack"] = float(average_precision_score(y_true_attack, prob_attack))
    except Exception:
        metrics["pr_auc_attack"] = float("nan")

    return metrics


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ============================================================
# Model definition
# ============================================================

class FTTransformerWithRepresentation(nn.Module):
    """
    Numerical FT-Transformer with CLS representation output.

    Each numerical feature is mapped to an independent token:
        t_i = x_i * w_i + b_i

    A learnable CLS token is prepended.
    The CLS output is used as the sample-level representation.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.d_token = d_token

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

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, num_classes),
        )

    def forward(self, x: torch.Tensor, return_representation: bool = False):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.encoder(tokens)
        cls_representation = encoded[:, 0, :]

        logits = self.head(cls_representation)

        if return_representation:
            return logits, cls_representation

        return logits


# ============================================================
# CORAL loss
# ============================================================

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CORAL loss aligns second-order statistics between source and target features.

    source: [B, D]
    target: [B, D]
    """
    if source.size(0) <= 1 or target.size(0) <= 1:
        return torch.tensor(0.0, device=source.device)

    d = source.size(1)

    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    source_cov = (source_centered.t() @ source_centered) / (source.size(0) - 1)
    target_cov = (target_centered.t() @ target_centered) / (target.size(0) - 1)

    loss = torch.mean((source_cov - target_cov) ** 2)
    loss = loss / (4.0 * d * d)

    return loss


# ============================================================
# Training and evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    attack_label_id: int,
    batch_size: int = 4096,
) -> dict:
    model.eval()

    all_pred = []
    all_prob_attack = []

    start_time = time.perf_counter()

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)

        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        all_pred.append(pred.detach().cpu().numpy())
        all_prob_attack.append(probs[:, attack_label_id].detach().cpu().numpy())

    inference_time = time.perf_counter() - start_time

    y_pred = np.concatenate(all_pred).astype(int)
    prob_attack = np.concatenate(all_prob_attack).astype(float)

    metrics = compute_metrics(y, y_pred, prob_attack, attack_label_id)
    metrics["inference_time_seconds"] = float(inference_time)
    metrics["inference_ms_per_sample"] = float(inference_time * 1000 / len(X))

    return metrics


def train_source_only(
    scenario_name: str,
    X_source_train: np.ndarray,
    y_source_train: np.ndarray,
    X_source_val: np.ndarray,
    y_source_val: np.ndarray,
    test_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    attack_label_id: int,
    device: torch.device,
) -> list[dict]:
    print("\n" + "=" * 100)
    print(f"Scenario: {scenario_name}")
    print("Training strategy: Source-only FT-Transformer")
    print("=" * 100)

    set_seed(RANDOM_STATE)

    model = FTTransformerWithRepresentation(
        num_features=X_source_train.shape[1],
        num_classes=2,
        d_token=64,
        n_heads=4,
        n_layers=4,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    param_count = count_parameters(model)

    source_dataset = TensorDataset(
        torch.tensor(X_source_train, dtype=torch.float32),
        torch.tensor(y_source_train, dtype=torch.long),
    )

    source_loader = DataLoader(
        source_dataset,
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

        for xs, ys in source_loader:
            xs = xs.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()
            logits = model(xs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(ys)
            total_samples += len(ys)

        train_loss = total_loss / max(total_samples, 1)

        val_metrics = evaluate_model(
            model,
            X_source_val,
            y_source_val,
            device,
            attack_label_id=attack_label_id,
            batch_size=BATCH_SIZE,
        )

        val_f1 = val_metrics["f1_attack"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.6f} "
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
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            device,
            attack_label_id=attack_label_id,
            batch_size=BATCH_SIZE,
        )

        row = {
            "scenario": scenario_name,
            "training_strategy": "Source-only FT-Transformer",
            "lambda_coral": 0.0,
            "eval_set": eval_name,
            "num_features": int(X_source_train.shape[1]),
            "source_train_samples": int(len(X_source_train)),
            "target_adapt_samples": 0,
            "test_samples": int(len(X_test)),
            "parameters": int(param_count),
            "best_epoch": int(best_epoch),
            "best_val_f1_attack": float(best_val_f1),
            "training_time_seconds": float(training_time),
        }
        row.update(metrics)
        rows.append(row)

        print(f"\nSource-only | {scenario_name} | {eval_name}")
        print(metrics)

    return rows


def train_coral(
    scenario_name: str,
    X_source_train: np.ndarray,
    y_source_train: np.ndarray,
    X_source_val: np.ndarray,
    y_source_val: np.ndarray,
    X_target_adapt: np.ndarray,
    test_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    attack_label_id: int,
    device: torch.device,
    lambda_coral: float,
) -> list[dict]:
    print("\n" + "=" * 100)
    print(f"Scenario: {scenario_name}")
    print(f"Training strategy: Domain-aligned FT-Transformer-CORAL, lambda={lambda_coral}")
    print("=" * 100)

    set_seed(RANDOM_STATE)

    model = FTTransformerWithRepresentation(
        num_features=X_source_train.shape[1],
        num_classes=2,
        d_token=64,
        n_heads=4,
        n_layers=4,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    param_count = count_parameters(model)

    source_dataset = TensorDataset(
        torch.tensor(X_source_train, dtype=torch.float32),
        torch.tensor(y_source_train, dtype=torch.long),
    )

    target_dataset = TensorDataset(
        torch.tensor(X_target_adapt, dtype=torch.float32),
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
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
        total_ce_loss = 0.0
        total_coral_loss = 0.0
        total_samples = 0

        target_iter = iter(target_loader)

        for xs, ys in source_loader:
            try:
                (xt,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                (xt,) = next(target_iter)

            xs = xs.to(device)
            ys = ys.to(device)
            xt = xt.to(device)

            optimizer.zero_grad()

            logits_s, rep_s = model(xs, return_representation=True)
            _, rep_t = model(xt, return_representation=True)

            ce = criterion(logits_s, ys)
            coral = coral_loss(rep_s, rep_t)
            loss = ce + lambda_coral * coral

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(ys)
            total_ce_loss += float(ce.item()) * len(ys)
            total_coral_loss += float(coral.item()) * len(ys)
            total_samples += len(ys)

        train_loss = total_loss / max(total_samples, 1)
        ce_loss_avg = total_ce_loss / max(total_samples, 1)
        coral_loss_avg = total_coral_loss / max(total_samples, 1)

        val_metrics = evaluate_model(
            model,
            X_source_val,
            y_source_val,
            device,
            attack_label_id=attack_label_id,
            batch_size=BATCH_SIZE,
        )

        val_f1 = val_metrics["f1_attack"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={train_loss:.6f} "
            f"ce={ce_loss_avg:.6f} "
            f"coral={coral_loss_avg:.8f} "
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
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            device,
            attack_label_id=attack_label_id,
            batch_size=BATCH_SIZE,
        )

        row = {
            "scenario": scenario_name,
            "training_strategy": "Domain-aligned FT-Transformer-CORAL",
            "lambda_coral": float(lambda_coral),
            "eval_set": eval_name,
            "num_features": int(X_source_train.shape[1]),
            "source_train_samples": int(len(X_source_train)),
            "target_adapt_samples": int(len(X_target_adapt)),
            "test_samples": int(len(X_test)),
            "parameters": int(param_count),
            "best_epoch": int(best_epoch),
            "best_val_f1_attack": float(best_val_f1),
            "training_time_seconds": float(training_time),
        }
        row.update(metrics)
        rows.append(row)

        print(f"\nCORAL lambda={lambda_coral} | {scenario_name} | {eval_name}")
        print(metrics)

    return rows


# ============================================================
# Scenario A: CICIoT2023 -> CICIoMT2024
# ============================================================

def load_cic_feature_columns() -> list[str]:
    candidates = [
        ALIGNED_DATA_DIR / "shared_feature_columns.json",
        ALIGNED_DATA_DIR / "feature_columns.json",
    ]

    for path in candidates:
        if path.exists():
            return load_json_list(path)

    df_head = pd.read_csv(CIC_SOURCE_TRAIN, nrows=100)
    label_col = detect_label_column(df_head)
    return infer_numeric_features_from_df(df_head, label_col)


def load_cic_scenario():
    print("\n" + "#" * 100)
    print("Loading scenario A: CICIoT2023 -> CICIoMT2024")
    print("#" * 100)

    feature_cols = load_cic_feature_columns()

    print(f"CIC aligned feature count: {len(feature_cols)}")

    df_train = read_csv_safely(CIC_SOURCE_TRAIN)
    df_val = read_csv_safely(CIC_SOURCE_VAL)
    df_source_test = read_csv_safely(CIC_SOURCE_TEST)
    df_target = read_csv_safely(CIC_TARGET_EXTERNAL)

    label_train = detect_label_column(df_train)
    label_val = detect_label_column(df_val)
    label_source_test = detect_label_column(df_source_test)
    label_target = detect_label_column(df_target)

    df_train = stratified_sample_df(df_train, label_train, MAX_SOURCE_TRAIN_SAMPLES, RANDOM_STATE)
    df_val = stratified_sample_df(df_val, label_val, MAX_SOURCE_VAL_SAMPLES, RANDOM_STATE)
    df_source_test = stratified_sample_df(df_source_test, label_source_test, MAX_SOURCE_TEST_SAMPLES, RANDOM_STATE)

    X_train, y_train = build_xy_from_df(df_train, feature_cols, label_train)
    X_val, y_val = build_xy_from_df(df_val, feature_cols, label_val)
    X_source_test, y_source_test = build_xy_from_df(df_source_test, feature_cols, label_source_test)
    X_target, y_target = build_xy_from_df(df_target, feature_cols, label_target)

    # CICIoT2023 binary setting in this project:
    # label_id = 0 is attack, label_id = 1 is benign
    attack_label_id = 0

    X_target_adapt, X_target_test, y_target_test = split_target_for_adaptation(
        X_target,
        y_target,
        max_adapt_samples=MAX_TARGET_ADAPT_SAMPLES,
        max_test_samples=MAX_TARGET_TEST_SAMPLES,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_source_test = scaler.transform(X_source_test).astype(np.float32)
    X_target_adapt = scaler.transform(X_target_adapt).astype(np.float32)
    X_target_test = scaler.transform(X_target_test).astype(np.float32)

    print("\nCIC scenario distributions:")
    print("Source train:", X_train.shape, dict(zip(*np.unique(y_train, return_counts=True))))
    print("Source val:", X_val.shape, dict(zip(*np.unique(y_val, return_counts=True))))
    print("Source internal test:", X_source_test.shape, dict(zip(*np.unique(y_source_test, return_counts=True))))
    print("Target adapt:", X_target_adapt.shape)
    print("Target test:", X_target_test.shape, dict(zip(*np.unique(y_target_test, return_counts=True))))

    return {
        "scenario_name": "CICIoT2023-to-CICIoMT2024",
        "feature_columns": feature_cols,
        "attack_label_id": attack_label_id,
        "X_source_train": X_train,
        "y_source_train": y_train,
        "X_source_val": X_val,
        "y_source_val": y_val,
        "X_target_adapt": X_target_adapt,
        "test_sets": {
            "CICIoT2023 internal aligned test": (X_source_test, y_source_test),
            "CICIoMT2024 held-out target test": (X_target_test, y_target_test),
        },
    }


# ============================================================
# Scenario B: NF-ToN-IoT -> NF-BoT-IoT
# ============================================================

def load_netflow_scenario():
    print("\n" + "#" * 100)
    print("Loading scenario B: NF-ToN-IoT -> NF-BoT-IoT")
    print("#" * 100)

    ton_path = auto_find_csv(NF_TON_FILE, "ton")
    bot_path = auto_find_csv(NF_BOT_FILE, "bot")

    df_ton = read_csv_safely(ton_path)
    df_bot = read_csv_safely(bot_path)

    label_ton = detect_label_column(df_ton)
    label_bot = detect_label_column(df_bot)

    train_features = infer_numeric_features_from_df(df_ton, label_ton)
    target_features = infer_numeric_features_from_df(df_bot, label_bot)
    common_features = sorted(list(set(train_features).intersection(set(target_features))))

    if not common_features:
        raise ValueError("No common numeric features found between NF-ToN-IoT and NF-BoT-IoT.")

    print(f"NetFlow common feature count: {len(common_features)}")
    print("NetFlow common features:", common_features)

    X_ton, y_ton = build_xy_from_df(
        df_ton,
        common_features,
        label_ton,
        force_binary_label=True,
    )

    X_bot, y_bot = build_xy_from_df(
        df_bot,
        common_features,
        label_bot,
        force_binary_label=True,
    )

    # NF dataset: 1 = attack, 0 = benign
    attack_label_id = 1

    indices = np.arange(len(y_ton))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=y_ton,
        random_state=RANDOM_STATE,
    )

    val_idx, source_test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=y_ton[temp_idx],
        random_state=RANDOM_STATE,
    )

    X_train = X_ton[train_idx]
    y_train = y_ton[train_idx]

    X_val = X_ton[val_idx]
    y_val = y_ton[val_idx]

    X_source_test = X_ton[source_test_idx]
    y_source_test = y_ton[source_test_idx]

    X_train, y_train = stratified_sample_xy(
        X_train,
        y_train,
        MAX_SOURCE_TRAIN_SAMPLES,
        RANDOM_STATE,
    )

    X_val, y_val = stratified_sample_xy(
        X_val,
        y_val,
        MAX_SOURCE_VAL_SAMPLES,
        RANDOM_STATE,
    )

    X_source_test, y_source_test = stratified_sample_xy(
        X_source_test,
        y_source_test,
        MAX_SOURCE_TEST_SAMPLES,
        RANDOM_STATE,
    )

    X_target_adapt, X_target_test, y_target_test = split_target_for_adaptation(
        X_bot,
        y_bot,
        max_adapt_samples=MAX_TARGET_ADAPT_SAMPLES,
        max_test_samples=MAX_TARGET_TEST_SAMPLES,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_source_test = scaler.transform(X_source_test).astype(np.float32)
    X_target_adapt = scaler.transform(X_target_adapt).astype(np.float32)
    X_target_test = scaler.transform(X_target_test).astype(np.float32)

    print("\nNetFlow scenario distributions:")
    print("Source train:", X_train.shape, dict(zip(*np.unique(y_train, return_counts=True))))
    print("Source val:", X_val.shape, dict(zip(*np.unique(y_val, return_counts=True))))
    print("Source internal test:", X_source_test.shape, dict(zip(*np.unique(y_source_test, return_counts=True))))
    print("Target adapt:", X_target_adapt.shape)
    print("Target test:", X_target_test.shape, dict(zip(*np.unique(y_target_test, return_counts=True))))

    return {
        "scenario_name": "NF-ToN-IoT-to-NF-BoT-IoT",
        "feature_columns": common_features,
        "attack_label_id": attack_label_id,
        "X_source_train": X_train,
        "y_source_train": y_train,
        "X_source_val": X_val,
        "y_source_val": y_val,
        "X_target_adapt": X_target_adapt,
        "test_sets": {
            "NF-ToN-IoT internal test": (X_source_test, y_source_test),
            "NF-BoT-IoT held-out target test": (X_target_test, y_target_test),
        },
    }


# ============================================================
# Run scenarios
# ============================================================

def run_scenario(scenario: dict) -> list[dict]:
    device = get_device()

    scenario_name = scenario["scenario_name"]
    attack_label_id = scenario["attack_label_id"]

    X_source_train = scenario["X_source_train"]
    y_source_train = scenario["y_source_train"]
    X_source_val = scenario["X_source_val"]
    y_source_val = scenario["y_source_val"]
    X_target_adapt = scenario["X_target_adapt"]
    test_sets = scenario["test_sets"]

    rows = []

    if RUN_SOURCE_ONLY:
        rows.extend(
            train_source_only(
                scenario_name=scenario_name,
                X_source_train=X_source_train,
                y_source_train=y_source_train,
                X_source_val=X_source_val,
                y_source_val=y_source_val,
                test_sets=test_sets,
                attack_label_id=attack_label_id,
                device=device,
            )
        )

    if RUN_CORAL:
        for lam in CORAL_LAMBDAS:
            rows.extend(
                train_coral(
                    scenario_name=scenario_name,
                    X_source_train=X_source_train,
                    y_source_train=y_source_train,
                    X_source_val=X_source_val,
                    y_source_val=y_source_val,
                    X_target_adapt=X_target_adapt,
                    test_sets=test_sets,
                    attack_label_id=attack_label_id,
                    device=device,
                    lambda_coral=lam,
                )
            )

    return rows


def main():
    warnings.filterwarnings("ignore")
    set_seed(RANDOM_STATE)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using device: {get_device()}")
    print(f"Output dir: {OUT_DIR}")

    all_rows = []
    feature_info = {}

    # Scenario A
    if CIC_SOURCE_TRAIN.exists() and CIC_TARGET_EXTERNAL.exists():
        cic_scenario = load_cic_scenario()
        feature_info[cic_scenario["scenario_name"]] = {
            "feature_count": len(cic_scenario["feature_columns"]),
            "feature_columns": cic_scenario["feature_columns"],
            "attack_label_id": cic_scenario["attack_label_id"],
        }
        all_rows.extend(run_scenario(cic_scenario))
    else:
        print("[SKIP] CIC aligned files not found.")

    # Scenario B
    ton_path_exists = NF_TON_FILE.exists() or any(NF_TON_DIR.glob("*.csv"))
    bot_path_exists = NF_BOT_FILE.exists() or any(NF_BOT_DIR.glob("*.csv"))

    if ton_path_exists and bot_path_exists:
        netflow_scenario = load_netflow_scenario()
        feature_info[netflow_scenario["scenario_name"]] = {
            "feature_count": len(netflow_scenario["feature_columns"]),
            "feature_columns": netflow_scenario["feature_columns"],
            "attack_label_id": netflow_scenario["attack_label_id"],
        }
        all_rows.extend(run_scenario(netflow_scenario))
    else:
        print("[SKIP] NetFlow files not found.")

    if not all_rows:
        raise RuntimeError("No scenario was executed. Please check dataset paths.")

    df_results = pd.DataFrame(all_rows)

    out_csv = OUT_DIR / "domain_aligned_ft_transformer_results.csv"
    out_txt = OUT_DIR / "domain_aligned_ft_transformer_results_for_word.txt"
    out_json = OUT_DIR / "domain_aligned_ft_transformer_results.json"
    out_features = OUT_DIR / "domain_aligned_feature_info.json"

    df_results.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_txt.write_text(df_results.to_string(index=False), encoding="utf-8")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    with open(out_features, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    print("\nFinal domain-aligned FT-Transformer results:")
    print(df_results.to_string(index=False))

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved TXT: {out_txt}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved feature info: {out_features}")


if __name__ == "__main__":
    main()