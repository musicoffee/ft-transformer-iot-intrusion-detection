from __future__ import annotations

import json
import sys
import time
import warnings
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
from torch.utils.data import TensorDataset, DataLoader

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except Exception:
    TABNET_AVAILABLE = False


# ============================================================
# Revision round 2: additional modern tabular baselines
# ------------------------------------------------------------
# Purpose:
#   Respond to Reviewer 1:
#   "Only RF and XGBoost are compared; more relevant modern
#    tabular representation learning baselines should be considered,
#    e.g., TabTransformer, SAINT, TabPFN."
#
# This script adds:
#   1. MLP baseline
#   2. TabNet baseline, when pytorch-tabnet is installed
#   3. TabTransformer-style numerical Transformer baseline
#
# Scenarios:
#   A. CICIoT2023 binary in-dataset test
#   B. Feature-aligned cross-dataset validation:
#      train on aligned CICIoT2023, test on aligned CICIoT2023
#      and external CICIoMT2024.
#
# Outputs:
#   outputs/revision_round2/modern_tabular_baseline_results.csv
#   outputs/revision_round2/modern_tabular_baseline_results_for_word.txt
#   outputs/revision_round2/modern_tabular_baseline_results.json
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

RANDOM_STATE = 42
ATTACK_LABEL_ID = 0

MAIN_DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_binary"
ALIGNED_DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "cross_dataset_aligned_binary"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "ft_binary_best.pt"

OUT_DIR = PROJECT_ROOT / "outputs" / "revision_round2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 控制训练时间。Reviewer 1 要补实验，不一定非要全量训练。
# train 使用 200000 条比较稳；test/external 默认全量评估。
MAX_TRAIN_SAMPLES = 200_000
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None
MAX_EXTERNAL_TEST_SAMPLES = None

BATCH_SIZE = 4096
EPOCHS = 15
PATIENCE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4

RUN_MLP = True
RUN_TABTRANSFORMER_STYLE = True
RUN_TABNET = True

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


def load_main_feature_columns() -> list[str]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    feature_columns = checkpoint["feature_columns"]
    return list(feature_columns)


def load_aligned_feature_columns() -> list[str]:
    candidates = [
        ALIGNED_DATA_DIR / "shared_feature_columns.json",
        ALIGNED_DATA_DIR / "feature_columns.json",
    ]

    for path in candidates:
        if path.exists():
            return load_json_list(path)

    # fallback: detect from main_train_aligned.csv
    df_head = pd.read_csv(ALIGNED_DATA_DIR / "main_train_aligned.csv", nrows=10)
    label_col = detect_label_column(df_head)

    feature_cols = []
    for col in df_head.columns:
        if col == label_col or col in LABEL_CANDIDATES:
            continue
        if pd.api.types.is_numeric_dtype(df_head[col]):
            feature_cols.append(col)

    return feature_cols


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


def load_xy(
    csv_path: Path,
    feature_columns: list[str],
    max_samples: int | None,
    random_state: int = 42,
):
    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path)
    label_col = detect_label_column(df)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {csv_path}: {missing[:20]}")

    df = stratified_sample_df(df, label_col, max_samples, random_state)

    X = df[feature_columns].values.astype(np.float32)
    y = df[label_col].values.astype(int)

    print(f"Loaded shape: X={X.shape}, y={y.shape}")
    print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))

    return X, y


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


class NumericTabTransformerStyle(nn.Module):
    """
    A TabTransformer-style numerical Transformer baseline.

    Since the current datasets mainly contain numerical tabular traffic statistics,
    each numerical feature is mapped into a token using a learnable linear
    projection, and self-attention is applied across feature tokens.
    This serves as a modern Transformer-based tabular baseline distinct from
    tree-based models.
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

        self.num_features = num_features
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

    def forward(self, x):
        # x: [B, F]
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]
        return self.head(cls_out)


@torch.no_grad()
def evaluate_torch_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    attack_label_id: int = 0,
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
        all_prob_attack.append(probs[:, attack_label_id].detach().cpu().numpy())

    infer_time = time.perf_counter() - start_time

    y_pred = np.concatenate(all_pred).astype(int)
    prob_attack = np.concatenate(all_prob_attack).astype(float)

    metrics = compute_metrics(y, y_pred, prob_attack, attack_label_id)
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
    X_test_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    scenario_name: str,
    feature_protocol: str,
):
    print("\n" + "=" * 90)
    print(f"Training {model_name} on scenario: {scenario_name}")
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

        val_metrics = evaluate_torch_model(
            model=model,
            X=X_val,
            y=y_val,
            device=device,
            batch_size=BATCH_SIZE,
            attack_label_id=ATTACK_LABEL_ID,
        )

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

    for eval_name, (X_test, y_test) in X_test_dict.items():
        test_metrics = evaluate_torch_model(
            model=model,
            X=X_test,
            y=y_test,
            device=device,
            batch_size=BATCH_SIZE,
            attack_label_id=ATTACK_LABEL_ID,
        )

        row = {
            "scenario": scenario_name,
            "feature_protocol": feature_protocol,
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
        row.update(test_metrics)
        rows.append(row)

        print(f"\n{model_name} | {scenario_name} | {eval_name}")
        print(test_metrics)

    return rows


def train_tabnet_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    scenario_name: str,
    feature_protocol: str,
):
    if not TABNET_AVAILABLE:
        print("\n[SKIP] pytorch-tabnet is not installed. Run: pip install pytorch-tabnet")
        return []

    print("\n" + "=" * 90)
    print(f"Training TabNet on scenario: {scenario_name}")
    print("=" * 90)

    set_seed(RANDOM_STATE)

    clf = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={
            "step_size": 5,
            "gamma": 0.9,
        },
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        seed=RANDOM_STATE,
        verbose=1,
    )

    train_start = time.perf_counter()

    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=30,
        patience=5,
        batch_size=8192,
        virtual_batch_size=1024,
        num_workers=0,
        drop_last=False,
    )

    training_time = time.perf_counter() - train_start

    rows = []

    classes = list(clf.classes_)
    attack_col = classes.index(ATTACK_LABEL_ID)

    for eval_name, (X_test, y_test) in X_test_dict.items():
        infer_start = time.perf_counter()

        prob = clf.predict_proba(X_test)
        pred = clf.predict(X_test).astype(int)

        infer_time = time.perf_counter() - infer_start
        prob_attack = prob[:, attack_col]

        metrics = compute_metrics(y_test, pred, prob_attack, ATTACK_LABEL_ID)
        metrics["inference_time_seconds"] = float(infer_time)
        metrics["inference_ms_per_sample"] = float(infer_time * 1000 / len(X_test))

        row = {
            "scenario": scenario_name,
            "feature_protocol": feature_protocol,
            "model": "TabNet",
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

        print(f"\nTabNet | {scenario_name} | {eval_name}")
        print(metrics)

    return rows


def run_scenario(
    scenario_name: str,
    feature_protocol: str,
    train_file: Path,
    val_file: Path,
    test_files: dict[str, Path],
    feature_columns: list[str],
):
    print("\n" + "#" * 90)
    print(f"Scenario: {scenario_name}")
    print(f"Feature protocol: {feature_protocol}")
    print(f"Feature count: {len(feature_columns)}")
    print("#" * 90)

    X_train, y_train = load_xy(
        train_file,
        feature_columns,
        max_samples=MAX_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
    )
    X_val, y_val = load_xy(
        val_file,
        feature_columns,
        max_samples=MAX_VAL_SAMPLES,
        random_state=RANDOM_STATE,
    )

    test_dict = {}

    for eval_name, path in test_files.items():
        max_samples = MAX_EXTERNAL_TEST_SAMPLES if "external" in eval_name else MAX_TEST_SAMPLES
        X_test, y_test = load_xy(
            path,
            feature_columns,
            max_samples=max_samples,
            random_state=RANDOM_STATE,
        )
        test_dict[eval_name] = (X_test, y_test)

    device = get_device()
    all_rows = []

    if RUN_MLP:
        mlp = MLPBaseline(
            num_features=len(feature_columns),
            num_classes=2,
            hidden_dims=(256, 128, 64),
            dropout=0.2,
        )

        all_rows.extend(
            train_torch_model(
                model_name="MLP",
                model=mlp,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test_dict=test_dict,
                device=device,
                scenario_name=scenario_name,
                feature_protocol=feature_protocol,
            )
        )

    if RUN_TABTRANSFORMER_STYLE:
        tabtransformer = NumericTabTransformerStyle(
            num_features=len(feature_columns),
            num_classes=2,
            d_token=64,
            n_heads=4,
            n_layers=3,
            dim_feedforward=128,
            dropout=0.1,
        )

        all_rows.extend(
            train_torch_model(
                model_name="TabTransformer-style",
                model=tabtransformer,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test_dict=test_dict,
                device=device,
                scenario_name=scenario_name,
                feature_protocol=feature_protocol,
            )
        )

    if RUN_TABNET:
        all_rows.extend(
            train_tabnet_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test_dict=test_dict,
                scenario_name=scenario_name,
                feature_protocol=feature_protocol,
            )
        )

    return all_rows


def main():
    warnings.filterwarnings("ignore")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using device: {get_device()}")
    print(f"TabNet available: {TABNET_AVAILABLE}")

    all_rows = []

    # Scenario A: main CICIoT2023 binary test
    main_features = load_main_feature_columns()

    all_rows.extend(
        run_scenario(
            scenario_name="CICIoT2023 Binary",
            feature_protocol="original processed feature space",
            train_file=MAIN_DATA_DIR / "train.csv",
            val_file=MAIN_DATA_DIR / "val.csv",
            test_files={
                "main_test": MAIN_DATA_DIR / "test.csv",
            },
            feature_columns=main_features,
        )
    )

    # Scenario B: 38-feature aligned external validation
    aligned_features = load_aligned_feature_columns()

    all_rows.extend(
        run_scenario(
            scenario_name="Feature-Aligned External Validation",
            feature_protocol="38 shared features",
            train_file=ALIGNED_DATA_DIR / "main_train_aligned.csv",
            val_file=ALIGNED_DATA_DIR / "main_val_aligned.csv",
            test_files={
                "main_aligned_test": ALIGNED_DATA_DIR / "main_test_aligned.csv",
                "external_aligned_test": ALIGNED_DATA_DIR / "external_test_aligned.csv",
            },
            feature_columns=aligned_features,
        )
    )

    df = pd.DataFrame(all_rows)

    out_csv = OUT_DIR / "modern_tabular_baseline_results.csv"
    out_txt = OUT_DIR / "modern_tabular_baseline_results_for_word.txt"
    out_json = OUT_DIR / "modern_tabular_baseline_results.json"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_txt.write_text(df.to_string(index=False), encoding="utf-8")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved TXT: {out_txt}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()