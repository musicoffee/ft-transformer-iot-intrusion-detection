from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs" / "revision_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 200_000


# =========================
# 路径配置
# =========================

DATASETS = {
    "CICIoT2023 Binary": {
        "dir": PROCESSED_DIR / "ciciot2023_binary",
        "files": {
            "Train": "train.csv",
            "Validation": "val.csv",
            "Test": "test.csv",
        },
    },
    "CICIoMT2024 Binary": {
        "dir": PROCESSED_DIR / "ciciomt2024_binary",
        "files": {
            "Train": "train.csv",
            "Validation": "val.csv",
            "Test": "test.csv",
        },
    },
    "CICIoT2023 Multi-class": {
        "dir": PROCESSED_DIR / "ciciot2023_7class",
        "files": {
            "Train": "train.csv",
            "Validation": "val.csv",
            "Test": "test.csv",
        },
    },
    "Cross-Dataset Aligned Binary": {
        "dir": PROCESSED_DIR / "cross_dataset_aligned_binary",
        "files": {
            "Main Train Aligned": "main_train_aligned.csv",
            "Main Validation Aligned": "main_val_aligned.csv",
            "Main Test Aligned": "main_test_aligned.csv",
            "External Test Aligned": "external_test_aligned.csv",
        },
    },
}


LABEL_CANDIDATES = [
    "label_id",
    "Label",
    "label",
    "Class",
    "class",
    "Category",
    "category",
    "Attack",
    "attack",
    "target",
    "Target",
    "y",
]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_label_column(csv_path: Path) -> Optional[str]:
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception as e:
        print(f"[WARN] Cannot read header: {csv_path} | {e}")
        return None

    cols = list(header.columns)
    for c in LABEL_CANDIDATES:
        if c in cols:
            return c

    return None


def count_rows(csv_path: Path) -> int:
    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE):
        total += len(chunk)
    return total


def count_label_distribution(csv_path: Path, label_col: str) -> dict:
    counts = {}

    for chunk in pd.read_csv(csv_path, usecols=[label_col], chunksize=CHUNKSIZE):
        vc = chunk[label_col].value_counts(dropna=False)
        for k, v in vc.items():
            key = str(k)
            counts[key] = counts.get(key, 0) + int(v)

    return dict(sorted(counts.items(), key=lambda x: x[0]))


def save_shared_features() -> None:
    cross_dir = PROCESSED_DIR / "cross_dataset_aligned_binary"

    candidate_files = [
        cross_dir / "shared_feature_columns.json",
        cross_dir / "feature_columns.json",
    ]

    shared_features = None
    used_file = None

    for p in candidate_files:
        data = load_json(p)
        if data is not None:
            shared_features = data
            used_file = p
            break

    if shared_features is None:
        print("[WARN] shared_feature_columns.json not found.")
        return

    df = pd.DataFrame({
        "No.": list(range(1, len(shared_features) + 1)),
        "Shared Feature": shared_features,
    })

    out_csv = OUT_DIR / "Table_S1_shared_38_features.csv"
    out_txt = OUT_DIR / "Table_S1_shared_38_features_for_word.txt"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Table S1. List of the shared features used for cross-dataset alignment.\n\n")
        f.write(df.to_string(index=False))

    print(f"[OK] Shared features loaded from: {used_file}")
    print(f"[OK] Saved: {out_csv}")
    print(f"[OK] Saved: {out_txt}")


def save_dataset_statistics() -> None:
    split_rows = []
    class_rows = []

    for dataset_name, info in DATASETS.items():
        dataset_dir = info["dir"]
        files = info["files"]

        label_mapping_path = dataset_dir / "label_mapping.json"
        label_mapping = load_json(label_mapping_path)

        print(f"\n=== {dataset_name} ===")
        print(f"Directory: {dataset_dir}")
        if label_mapping:
            print(f"Label mapping: {label_mapping}")

        for split_name, filename in files.items():
            csv_path = dataset_dir / filename

            if not csv_path.exists():
                print(f"[SKIP] Missing file: {csv_path}")
                continue

            label_col = detect_label_column(csv_path)
            n_rows = count_rows(csv_path)

            split_rows.append({
                "Task/Dataset": dataset_name,
                "Split": split_name,
                "File": str(csv_path.relative_to(PROJECT_ROOT)),
                "Samples": n_rows,
                "Label Column": label_col if label_col else "Not detected",
            })

            print(f"[OK] {split_name}: {n_rows} samples | label_col={label_col}")

            if label_col:
                counts = count_label_distribution(csv_path, label_col)
                for label_value, count in counts.items():
                    class_rows.append({
                        "Task/Dataset": dataset_name,
                        "Split": split_name,
                        "Label Column": label_col,
                        "Class/Label": label_value,
                        "Samples": count,
                    })

    split_df = pd.DataFrame(split_rows)
    class_df = pd.DataFrame(class_rows)

    split_csv = OUT_DIR / "dataset_split_summary.csv"
    class_csv = OUT_DIR / "class_distribution_summary.csv"

    split_txt = OUT_DIR / "dataset_split_summary_for_word.txt"
    class_txt = OUT_DIR / "class_distribution_summary_for_word.txt"

    split_df.to_csv(split_csv, index=False, encoding="utf-8-sig")
    class_df.to_csv(class_csv, index=False, encoding="utf-8-sig")

    with open(split_txt, "w", encoding="utf-8") as f:
        f.write("Table X. Dataset split statistics used in the experiments.\n\n")
        f.write(split_df.to_string(index=False))

    with open(class_txt, "w", encoding="utf-8") as f:
        f.write("Table X. Class distribution statistics used in the experiments.\n\n")
        f.write(class_df.to_string(index=False))

    print(f"\n[OK] Saved: {split_csv}")
    print(f"[OK] Saved: {class_csv}")
    print(f"[OK] Saved: {split_txt}")
    print(f"[OK] Saved: {class_txt}")


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed dir: {PROCESSED_DIR}")
    print(f"Output dir: {OUT_DIR}")

    save_shared_features()
    save_dataset_statistics()

    print("\nDone.")
    print("Please check outputs/revision_tables/.")


if __name__ == "__main__":
    main()