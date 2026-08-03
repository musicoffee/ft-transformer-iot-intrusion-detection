from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils import shuffle as sk_shuffle

# =========================================================
# 放到：baselines/baseline_rf_multiclass.py
# 用法：在 PyCharm 里直接点绿色三角运行
# 任务：CICIoT2023 多分类 Random Forest 基线
# =========================================================

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_7class"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
TEST_FILE = DATA_DIR / "test.csv"
FEATURE_FILE = DATA_DIR / "feature_columns.json"
LABEL_FILE = DATA_DIR / "label_mapping.json"

USE_VAL_IN_TRAIN = True
MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None

N_ESTIMATORS = 300
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
N_JOBS = -1

REPORT_JSON = OUTPUT_REPORT_DIR / "baseline_rf_multiclass_metrics.json"
REPORT_TXT = OUTPUT_REPORT_DIR / "baseline_rf_multiclass_classification_report.txt"
CM_PNG = OUTPUT_FIGURE_DIR / "baseline_rf_multiclass_confusion_matrix.png"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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
    normalized_map = {normalize_colname(c): c for c in df.columns}

    candidates = [
        "label",
        "label_id",
        "target",
        "target_id",
        "class",
        "class_id",
        "y",
        "category",
    ]
    for cand in candidates:
        if cand in normalized_map:
            return normalized_map[cand]

    feature_set = set(feature_columns)
    extra_cols = [c for c in df.columns if c not in feature_set]

    useless_cols = {
        "label_name",
        "class_name",
        "split",
        "source",
        "dataset",
        "subset",
        "Unnamed: 0",
    }
    filtered_extra_cols = [c for c in extra_cols if c not in useless_cols]

    if len(filtered_extra_cols) == 1:
        print(f"自动推断标签列为：{filtered_extra_cols[0]}")
        return filtered_extra_cols[0]

    low_cardinality_cols = []
    for col in filtered_extra_cols:
        uniques = pd.Series(df[col]).dropna().unique().tolist()
        if len(uniques) <= 20:
            low_cardinality_cols.append(col)

    if len(low_cardinality_cols) == 1:
        print(f"自动推断标签列为：{low_cardinality_cols[0]}")
        return low_cardinality_cols[0]

    raise ValueError(
        "未能自动识别多分类标签列。\n"
        f"全部列名：{list(df.columns)}\n"
        f"非特征列候选：{filtered_extra_cols}\n"
        f"低类别候选：{low_cardinality_cols}"
    )


def encode_labels(y_raw: pd.Series | np.ndarray, label_mapping: dict) -> np.ndarray:
    """
    把标签统一编码成 int：
    1. 如果本来就是整数，直接返回
    2. 如果是文本标签，如 'DoS'，按 label_mapping 映射
    """
    y_series = pd.Series(y_raw).copy()

    # 情况1：已经是整数
    try:
        y_int = y_series.astype(int)
        unique_vals = set(y_int.unique().tolist())
        mapping_vals = set(int(v) for v in label_mapping.values())
        if unique_vals.issubset(mapping_vals):
            return y_int.values
    except Exception:
        pass

    # 情况2：文本标签 -> label_mapping[key]
    if all(isinstance(k, str) for k in label_mapping.keys()):
        mapped = y_series.map(label_mapping)
        if mapped.isna().any():
            missing = sorted(y_series[mapped.isna()].astype(str).unique().tolist())
            raise ValueError(
                f"以下文本标签无法在 label_mapping.json 中找到映射：{missing}\n"
                f"label_mapping keys: {list(label_mapping.keys())}"
            )
        return mapped.astype(int).values

    # 情况3：兜底，尝试把 label_mapping 反过来
    reverse_mapping = {str(v): int(k) for k, v in label_mapping.items() if str(k).isdigit()}
    mapped = y_series.astype(str).map(reverse_mapping)
    if mapped.notna().all():
        return mapped.astype(int).values

    raise ValueError(
        "无法将标签编码为整数，请检查 label_mapping.json 与标签列内容是否一致。\n"
        f"标签列示例：{y_series.head(10).tolist()}\n"
        f"label_mapping: {label_mapping}"
    )


def plot_confusion(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("RF Multi-class Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    set_seed(RANDOM_STATE)

    feature_columns = load_json(FEATURE_FILE)
    label_mapping = load_json(LABEL_FILE)

    inv_label_mapping = {int(v): k for k, v in label_mapping.items()}
    class_names = [inv_label_mapping[i] for i in range(len(inv_label_mapping))]

    df_train = pd.read_csv(TRAIN_FILE)
    if USE_VAL_IN_TRAIN:
        df_val = pd.read_csv(VAL_FILE)
        df_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    df_test = pd.read_csv(TEST_FILE)

    label_col = detect_label_column(df_train, feature_columns)

    df_train = sample_if_needed(df_train, MAX_TRAIN_SAMPLES)
    df_test = sample_if_needed(df_test, MAX_TEST_SAMPLES)
    df_train = sk_shuffle(df_train, random_state=RANDOM_STATE).reset_index(drop=True)

    x_train = df_train[feature_columns].values
    y_train = encode_labels(df_train[label_col], label_mapping)

    x_test = df_test[feature_columns].values
    y_test = encode_labels(df_test[label_col], label_mapping)

    print(f"RF multiclass train shape: {x_train.shape}")
    print(f"RF multiclass test shape: {x_test.shape}")
    print(f"Detected label column: {label_col}")
    print(f"Label mapping: {label_mapping}")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "use_val_in_train": USE_VAL_IN_TRAIN,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_test_samples": MAX_TEST_SAMPLES,
        "label_column": label_col,
        "label_mapping": label_mapping,
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    plot_confusion(cm, class_names, CM_PNG)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print("RF multiclass metrics:", metrics)
    print(f"RF multiclass metrics saved to: {REPORT_JSON}")
    print(f"RF multiclass report saved to: {REPORT_TXT}")
    print(f"RF multiclass confusion matrix saved to: {CM_PNG}")


if __name__ == "__main__":
    main()