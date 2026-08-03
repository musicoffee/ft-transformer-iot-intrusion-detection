from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import shuffle as sk_shuffle
from xgboost import XGBClassifier

# =========================================================
# 放到：analysis/run_cross_dataset_baselines.py
# 用法：在 PyCharm 里直接点绿色三角运行
# 任务：对齐特征后的跨数据集二分类基线（RF + XGBoost）
# =========================================================

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "cross_dataset_aligned_binary"
MAIN_BINARY_DIR = PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_binary"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# 你当前真实文件名
TRAIN_FILE = DATA_DIR / "main_train_aligned.csv"
VAL_FILE = DATA_DIR / "main_val_aligned.csv"
MAIN_TEST_FILE = DATA_DIR / "main_test_aligned.csv"
EXTERNAL_TEST_FILE = DATA_DIR / "external_test_aligned.csv"

# 你当前真实特征文件
FEATURE_FILE = DATA_DIR / "shared_feature_columns.json"

# 复用主二分类数据集的标签映射
LABEL_FILE = MAIN_BINARY_DIR / "label_mapping.json"

USE_VAL_IN_TRAIN = True

# 为了更稳更快，默认先限制训练集大小；内存够可以改成 None
MAX_TRAIN_SAMPLES = 300000
MAX_MAIN_TEST_SAMPLES = None
MAX_EXTERNAL_TEST_SAMPLES = None

# ---------------------------
# Random Forest 参数
# ---------------------------
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_N_JOBS = -1

# ---------------------------
# XGBoost 参数
# ---------------------------
XGB_N_ESTIMATORS = 350
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.08
XGB_SUBSAMPLE = 0.9
XGB_COLSAMPLE_BYTREE = 0.9
XGB_TREE_METHOD = "hist"

REPORT_JSON = OUTPUT_REPORT_DIR / "cross_dataset_baseline_compare.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir() -> None:
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_if_needed(df: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=RANDOM_STATE).reset_index(drop=True)


def normalize_colname(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def detect_label_column(df: pd.DataFrame, feature_columns: list[str]) -> str:
    """
    先用候选名字找标签列；
    找不到时，用“所有列 - 特征列”推断标签列；
    如果还是找不到，就报错并打印全部列名。
    """
    normalized_map = {normalize_colname(c): c for c in df.columns}

    # 常见标签列候选
    candidates = [
        "label",
        "binary_label",
        "label_id",
        "target",
        "y",
        "class",
        "class_id",
        "attack_label",
        "binary_target",
        "gt",
    ]

    for cand in candidates:
        if cand in normalized_map:
            return normalized_map[cand]

    # 如果没有常见标签列，就用“非特征列”推断
    feature_set = set(feature_columns)
    extra_cols = [c for c in df.columns if c not in feature_set]

    # 优先排除明显不是标签的辅助列
    useless_cols = {
        "split",
        "source",
        "dataset",
        "subset",
        "index",
        "Unnamed: 0",
    }
    extra_cols = [c for c in extra_cols if c not in useless_cols]

    if len(extra_cols) == 1:
        print(f"自动推断标签列为：{extra_cols[0]}")
        return extra_cols[0]

    # 如果有多个非特征列，优先找整数/二值型列
    binary_like = []
    for col in extra_cols:
        uniques = pd.Series(df[col]).dropna().unique().tolist()
        if len(uniques) <= 10:
            binary_like.append(col)

    if len(binary_like) == 1:
        print(f"自动推断标签列为：{binary_like[0]}")
        return binary_like[0]

    raise ValueError(
        "未能自动识别标签列。\n"
        f"全部列名：{list(df.columns)}\n"
        f"共享特征列数：{len(feature_columns)}\n"
        f"非特征列候选：{extra_cols}\n"
        f"二值/低类别候选：{binary_like}"
    )


def detect_attack_label_id(label_mapping: dict | None) -> int:
    # 优先从 label_mapping.json 读取
    if label_mapping is not None and "attack" in label_mapping:
        return int(label_mapping["attack"])
    # 兜底：你整个工程默认一直是 attack=0, benign=1
    return 0


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_attack: np.ndarray,
    attack_label_id: int,
) -> dict:
    y_true_bin = (y_true == attack_label_id).astype(int)
    y_pred_bin = (y_pred == attack_label_id).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall_attack": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1_attack": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "roc_auc_attack": float(roc_auc_score(y_true_bin, y_prob_attack)),
        "pr_auc_attack": float(average_precision_score(y_true_bin, y_prob_attack)),
    }


def evaluate_binary(model, x, y, attack_label_id: int) -> dict:
    proba = model.predict_proba(x)
    pred = np.argmax(proba, axis=1)
    y_prob_attack = proba[:, attack_label_id]
    return binary_metrics(y, pred, y_prob_attack, attack_label_id)


def main() -> None:
    ensure_dir()
    set_seed(RANDOM_STATE)

    # ---------- 文件检查 ----------
    for f in [TRAIN_FILE, VAL_FILE, MAIN_TEST_FILE, EXTERNAL_TEST_FILE, FEATURE_FILE]:
        if not f.exists():
            raise FileNotFoundError(f"未找到文件：{f}")

    # ---------- 读取特征列 ----------
    feature_columns = load_json(FEATURE_FILE)

    # ---------- 读取标签映射 ----------
    label_mapping = None
    if LABEL_FILE.exists():
        label_mapping = load_json(LABEL_FILE)
    else:
        print(f"警告：未找到 label_mapping.json，将默认 attack_label_id=0。路径：{LABEL_FILE}")

    attack_label_id = detect_attack_label_id(label_mapping)

    # ---------- 读取 CSV ----------
    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)
    df_main_test = pd.read_csv(MAIN_TEST_FILE)
    df_external_test = pd.read_csv(EXTERNAL_TEST_FILE)

    # ---------- 自动识别标签列 ----------
    label_col = detect_label_column(df_train, feature_columns)

    # ---------- 训练集拼接 ----------
    if USE_VAL_IN_TRAIN:
        df_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    # ---------- 抽样 ----------
    df_train = sample_if_needed(df_train, MAX_TRAIN_SAMPLES)
    df_main_test = sample_if_needed(df_main_test, MAX_MAIN_TEST_SAMPLES)
    df_external_test = sample_if_needed(df_external_test, MAX_EXTERNAL_TEST_SAMPLES)

    df_train = sk_shuffle(df_train, random_state=RANDOM_STATE).reset_index(drop=True)

    # ---------- 取特征与标签 ----------
    x_train = df_train[feature_columns].values
    y_train = df_train[label_col].values.astype(int)

    x_main = df_main_test[feature_columns].values
    y_main = df_main_test[label_col].values.astype(int)

    x_ext = df_external_test[feature_columns].values
    y_ext = df_external_test[label_col].values.astype(int)

    print(f"Aligned train shape: {x_train.shape}")
    print(f"Aligned main test shape: {x_main.shape}")
    print(f"Aligned external test shape: {x_ext.shape}")
    print(f"Shared feature count: {len(feature_columns)}")
    print(f"Detected label column: {label_col}")
    print(f"Attack label id: {attack_label_id}")
    print(f"Label mapping: {label_mapping}")

    # ---------- Random Forest ----------
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS,
    )
    rf_model.fit(x_train, y_train)

    # ---------- XGBoost ----------
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        tree_method=XGB_TREE_METHOD,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=0,
    )
    xgb_model.fit(x_train, y_train)

    # ---------- 评估 ----------
    results = {
        "settings": {
            "use_val_in_train": USE_VAL_IN_TRAIN,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_main_test_samples": MAX_MAIN_TEST_SAMPLES,
            "max_external_test_samples": MAX_EXTERNAL_TEST_SAMPLES,
            "shared_feature_count": len(feature_columns),
            "label_column": label_col,
            "attack_label_id": attack_label_id,
            "label_mapping": label_mapping,
        },
        "rf": {
            "main_aligned_test": evaluate_binary(rf_model, x_main, y_main, attack_label_id),
            "external_test": evaluate_binary(rf_model, x_ext, y_ext, attack_label_id),
        },
        "xgboost": {
            "main_aligned_test": evaluate_binary(xgb_model, x_main, y_main, attack_label_id),
            "external_test": evaluate_binary(xgb_model, x_ext, y_ext, attack_label_id),
        },
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nCross-dataset baseline compare saved to:")
    print(REPORT_JSON)
    print("\nResults:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()