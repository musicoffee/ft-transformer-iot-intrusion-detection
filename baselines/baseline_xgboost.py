from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RANDOM_STATE
from datasets_code.tabular_dataset import (
    load_feature_columns,
    load_label_mapping,
    read_feature_matrix_and_labels,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# =========================
# 你主要改这里
# =========================
DATASET_DIR = PROCESSED_DATA_DIR / "ciciot2023_binary"

MAX_TRAIN_SAMPLES = 200000
MAX_TEST_SAMPLES = 100000

N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_JSON_NAME = "baseline_xgboost_metrics.json"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main():
    if XGBClassifier is None:
        raise ImportError("未安装 xgboost。请先运行：pip install xgboost")

    ensure_dir(REPORT_DIR)

    feature_columns = load_feature_columns(DATASET_DIR)
    label_mapping = load_label_mapping(DATASET_DIR)

    if "attack" not in label_mapping:
        raise ValueError(f"label_mapping.json 中未找到 'attack'。当前映射：{label_mapping}")

    attack_label_id = int(label_mapping["attack"])

    train_csv = DATASET_DIR / "train.csv"
    test_csv = DATASET_DIR / "test.csv"

    x_train, y_train = read_feature_matrix_and_labels(
        train_csv,
        feature_columns,
        max_samples=MAX_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
    )
    x_test, y_test = read_feature_matrix_and_labels(
        test_csv,
        feature_columns,
        max_samples=MAX_TEST_SAMPLES,
        random_state=RANDOM_STATE,
    )

    print("XGBoost train shape:", x_train.shape)
    print("XGBoost test shape:", x_test.shape)
    print("Label mapping:", label_mapping)

    # 二分类任务，直接用 binary:logistic
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    )

    model.fit(x_train, y_train)

    # predict_proba 返回 [n_samples, 2]
    proba = model.predict_proba(x_test)

    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"predict_proba 输出形状异常：{proba.shape}")

    # 用概率取 argmax 得到一维标签预测，避免 multilabel-indicator 问题
    y_pred = np.argmax(proba, axis=1)

    # attack 对应哪一列，就取哪一列概率
    y_prob_attack = proba[:, attack_label_id]

    y_true_bin = (y_test == attack_label_id).astype(int)
    y_pred_bin = (y_pred == attack_label_id).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_attack": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall_attack": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1_attack": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "roc_auc_attack": float(roc_auc_score(y_true_bin, y_prob_attack)),
        "pr_auc_attack": float(average_precision_score(y_true_bin, y_prob_attack)),
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_test_samples": MAX_TEST_SAMPLES,
        "attack_label_id": attack_label_id,
        "label_mapping": label_mapping,
    }

    print("XGBoost metrics:", metrics)

    with open(REPORT_DIR / REPORT_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"XGBoost metrics saved to: {REPORT_DIR / REPORT_JSON_NAME}")


if __name__ == "__main__":
    main()