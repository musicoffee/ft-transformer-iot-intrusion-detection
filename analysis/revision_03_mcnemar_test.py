from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier


# ============================================================
# McNemar significance test for binary classification
# ------------------------------------------------------------
# This version follows trainers/evaluate_binary.py exactly:
#   1. Load feature_columns from FT checkpoint.
#   2. Use the same feature order for FT, RF, and XGBoost.
#   3. Do NOT apply an extra scaler here.
#
# Output:
#   outputs/revision_tables/mcnemar_model_metrics.csv
#   outputs/revision_tables/mcnemar_model_metrics_for_word.txt
#   outputs/revision_tables/mcnemar_test_results.csv
#   outputs/revision_tables/mcnemar_test_results_for_word.txt
#   outputs/revision_tables/mcnemar_predictions_main_binary.csv
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RANDOM_STATE
from datasets_code.tabular_dataset import create_dataloader, read_feature_matrix_and_labels
from models.ft_transformer import FTTransformer


DATASET_DIR = PROCESSED_DATA_DIR / "ciciot2023_binary"
TRAIN_FILE = DATASET_DIR / "train.csv"
TEST_FILE = DATASET_DIR / "test.csv"

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "ft_binary_best.pt"

OUT_DIR = PROJECT_ROOT / "outputs" / "revision_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4096
NUM_WORKERS = 0
PIN_MEMORY = True

# 和你原 baseline 设置保持一致：RF/XGBoost 训练最多 200000 条
MAX_TRAIN_SAMPLES = 200_000

# 测试集用完整 120000 条，保证和 Table 1 FT 评估一致
MAX_TEST_SAMPLES = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(device: torch.device):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    feature_columns = checkpoint["feature_columns"]
    label_mapping = checkpoint["label_mapping"]
    attack_label_id = int(label_mapping["attack"])
    model_config = checkpoint["model_config"]

    return checkpoint, feature_columns, label_mapping, attack_label_id, model_config


@torch.no_grad()
def predict_ft_transformer(
    checkpoint,
    feature_columns: list[str],
    attack_label_id: int,
    model_config: dict,
):
    device = get_device()
    print(f"Using device for FT-Transformer: {device}")

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

    all_y_true = []
    all_y_pred = []
    all_y_prob_attack = []

    for x, y in test_loader:
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

    return y_true, y_pred, y_prob_attack


def load_train_test_for_baselines(feature_columns: list[str]):
    """
    这里必须使用 checkpoint 中的 feature_columns。
    这样 RF/XGBoost 和 FT 在同一个特征顺序、同一个测试集上比较。
    """
    X_train, y_train = read_feature_matrix_and_labels(
        csv_path=TRAIN_FILE,
        feature_columns=feature_columns,
        label_column="label_id",
        max_samples=MAX_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
    )

    X_test, y_test = read_feature_matrix_and_labels(
        csv_path=TEST_FILE,
        feature_columns=feature_columns,
        label_column="label_id",
        max_samples=MAX_TEST_SAMPLES,
        random_state=RANDOM_STATE,
    )

    return X_train, y_train, X_test, y_test


def train_predict_rf(X_train, y_train, X_test, attack_label_id: int):
    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(int)

    prob = model.predict_proba(X_test)
    prob_attack = prob[:, attack_label_id]

    print("Random Forest prediction done.")
    return pred, prob_attack


def train_predict_xgb(X_train, y_train, X_test, attack_label_id: int):
    print("Training XGBoost...")

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
    pred = model.predict(X_test).astype(int)

    prob = model.predict_proba(X_test)
    prob_attack = prob[:, attack_label_id]

    print("XGBoost prediction done.")
    return pred, prob_attack


def metrics_binary(y_true, y_pred, prob_attack, attack_label_id: int):
    y_true_attack = (y_true == attack_label_id).astype(int)
    y_pred_attack = (y_pred == attack_label_id).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_attack": precision_score(y_true_attack, y_pred_attack, zero_division=0),
        "recall_attack": recall_score(y_true_attack, y_pred_attack, zero_division=0),
        "f1_attack": f1_score(y_true_attack, y_pred_attack, zero_division=0),
        "roc_auc_attack": roc_auc_score(y_true_attack, prob_attack),
        "pr_auc_attack": average_precision_score(y_true_attack, prob_attack),
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray):
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b = int(np.sum(correct_a & (~correct_b)))      # A correct, B wrong
    c = int(np.sum((~correct_a) & correct_b))      # A wrong, B correct

    if b + c == 0:
        chi_square = 0.0
        p_value = 1.0
    else:
        chi_square = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(chi2.sf(chi_square, df=1))

    return b, c, float(chi_square), p_value


def main():
    device = get_device()
    checkpoint, feature_columns, label_mapping, attack_label_id, model_config = load_checkpoint(device)

    print("Feature columns loaded from checkpoint:")
    print(feature_columns)
    print(f"Number of features: {len(feature_columns)}")
    print(f"Label mapping: {label_mapping}")
    print(f"Attack label id: {attack_label_id}")

    # 1. FT prediction, exactly following evaluate_binary.py
    y_true_ft, y_pred_ft, prob_ft_attack = predict_ft_transformer(
        checkpoint=checkpoint,
        feature_columns=feature_columns,
        attack_label_id=attack_label_id,
        model_config=model_config,
    )

    # 2. Baseline train/test with the same feature_columns
    X_train, y_train, X_test, y_test = load_train_test_for_baselines(feature_columns)

    # 确保 FT 的 y_true 和 baseline 的 y_test 完全一致
    if not np.array_equal(y_true_ft, y_test):
        raise ValueError(
            "The FT test labels and baseline test labels are not identical. "
            "Please check test loading order."
        )

    print(f"Train shape for baselines: {X_train.shape}")
    print(f"Test shape for all models: {X_test.shape}")
    print("Test label distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

    y_pred_rf, prob_rf_attack = train_predict_rf(X_train, y_train, X_test, attack_label_id)
    y_pred_xgb, prob_xgb_attack = train_predict_xgb(X_train, y_train, X_test, attack_label_id)

    preds = {
        "FT-Transformer": y_pred_ft,
        "Random Forest": y_pred_rf,
        "XGBoost": y_pred_xgb,
    }

    probs_attack = {
        "FT-Transformer": prob_ft_attack,
        "Random Forest": prob_rf_attack,
        "XGBoost": prob_xgb_attack,
    }

    # 保存预测结果
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "pred_ft_transformer": y_pred_ft,
        "pred_random_forest": y_pred_rf,
        "pred_xgboost": y_pred_xgb,
        "prob_attack_ft_transformer": prob_ft_attack,
        "prob_attack_random_forest": prob_rf_attack,
        "prob_attack_xgboost": prob_xgb_attack,
    })

    pred_path = OUT_DIR / "mcnemar_predictions_main_binary.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 保存同一测试集上的模型指标
    metric_rows = []
    for name in preds.keys():
        row = {"Model": name}
        row.update(metrics_binary(y_test, preds[name], probs_attack[name], attack_label_id))
        metric_rows.append(row)

    metric_df = pd.DataFrame(metric_rows)

    metric_csv = OUT_DIR / "mcnemar_model_metrics.csv"
    metric_txt = OUT_DIR / "mcnemar_model_metrics_for_word.txt"

    metric_df.to_csv(metric_csv, index=False, encoding="utf-8-sig")
    metric_txt.write_text(metric_df.to_string(index=False), encoding="utf-8")

    # McNemar test
    rows = []
    model_names = list(preds.keys())

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model_a = model_names[i]
            model_b = model_names[j]

            b_count, c_count, stat, p = mcnemar_test(
                y_true=y_test,
                pred_a=preds[model_a],
                pred_b=preds[model_b],
            )

            rows.append({
                "Model A": model_a,
                "Model B": model_b,
                "Accuracy A": accuracy_score(y_test, preds[model_a]),
                "Accuracy B": accuracy_score(y_test, preds[model_b]),
                "b (A correct, B wrong)": b_count,
                "c (A wrong, B correct)": c_count,
                "McNemar chi-square": stat,
                "p-value": p,
                "Significant at 0.05": "Yes" if p < 0.05 else "No",
            })

    result_df = pd.DataFrame(rows)

    out_csv = OUT_DIR / "mcnemar_test_results.csv"
    out_txt = OUT_DIR / "mcnemar_test_results_for_word.txt"

    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(
            "Table X. McNemar's test results for pairwise model comparison "
            "on the CICIoT2023 binary test set.\n\n"
        )
        f.write(result_df.to_string(index=False))

    print("\nModel metrics on the same CICIoT2023 binary test set:")
    print(metric_df.to_string(index=False))

    print("\nMcNemar test results:")
    print(result_df.to_string(index=False))

    print(f"\nSaved predictions: {pred_path}")
    print(f"Saved metrics CSV: {metric_csv}")
    print(f"Saved metrics TXT: {metric_txt}")
    print(f"Saved McNemar CSV: {out_csv}")
    print(f"Saved McNemar TXT: {out_txt}")


if __name__ == "__main__":
    main()