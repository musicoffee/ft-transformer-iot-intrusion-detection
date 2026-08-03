from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, RANDOM_STATE
from datasets_code.tabular_dataset import load_feature_columns, load_label_mapping, read_feature_matrix_and_labels

DATASET_DIR = PROCESSED_DATA_DIR / "ciciot2023_binary"

MAX_TRAIN_SAMPLES = 200000
MAX_TEST_SAMPLES = 100000

N_ESTIMATORS = 300
MAX_DEPTH = None
N_JOBS = -1

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_JSON_NAME = "baseline_rf_metrics.json"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dir(REPORT_DIR)

    feature_columns = load_feature_columns(DATASET_DIR)
    label_mapping = load_label_mapping(DATASET_DIR)
    attack_label_id = int(label_mapping["attack"])

    train_csv = DATASET_DIR / "train.csv"
    test_csv = DATASET_DIR / "test.csv"

    x_train, y_train = read_feature_matrix_and_labels(
        train_csv, feature_columns, max_samples=MAX_TRAIN_SAMPLES, random_state=RANDOM_STATE
    )
    x_test, y_test = read_feature_matrix_and_labels(
        test_csv, feature_columns, max_samples=MAX_TEST_SAMPLES, random_state=RANDOM_STATE
    )

    print("RF train shape:", x_train.shape)
    print("RF test shape:", x_test.shape)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob_attack = model.predict_proba(x_test)[:, attack_label_id]

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
    }

    print("RF metrics:", metrics)
    with open(REPORT_DIR / REPORT_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"RF metrics saved to: {REPORT_DIR / REPORT_JSON_NAME}")


if __name__ == "__main__":
    main()