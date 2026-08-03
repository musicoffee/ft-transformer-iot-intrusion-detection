import json
from pathlib import Path

import pandas as pd

from config import (
    CICIOMT2024_BINARY_NAME,
    CICIOT2023_BINARY_NAME,
    CROSS_DATASET_ALIGNED_NAME,
    LOG_DIR,
    PROCESSED_DATA_DIR,
)
from utils.data_utils import ensure_dir
from utils.logger_utils import get_logger


def load_feature_columns(folder: Path):
    with open(folder / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    logger = get_logger(LOG_DIR, "04_align_cross_dataset_binary")

    src_main = PROCESSED_DATA_DIR / CICIOT2023_BINARY_NAME
    src_ext = PROCESSED_DATA_DIR / CICIOMT2024_BINARY_NAME
    save_dir = PROCESSED_DATA_DIR / CROSS_DATASET_ALIGNED_NAME
    ensure_dir(save_dir)

    main_train = pd.read_csv(src_main / "train.csv")
    main_val = pd.read_csv(src_main / "val.csv")
    main_test = pd.read_csv(src_main / "test.csv")
    ext_test = pd.read_csv(src_ext / "test.csv")

    main_features = set(load_feature_columns(src_main))
    ext_features = set(load_feature_columns(src_ext))
    shared_features = sorted(list(main_features & ext_features))

    if len(shared_features) < 10:
        raise ValueError(f"共享特征过少（{len(shared_features)} 个），不建议直接做跨数据集实验。")

    keep_cols = shared_features + ["label_text", "label_id"]

    main_train = main_train[keep_cols].copy()
    main_val = main_val[keep_cols].copy()
    main_test = main_test[keep_cols].copy()
    ext_test = ext_test[keep_cols].copy()

    main_train.to_csv(save_dir / "main_train_aligned.csv", index=False)
    main_val.to_csv(save_dir / "main_val_aligned.csv", index=False)
    main_test.to_csv(save_dir / "main_test_aligned.csv", index=False)
    ext_test.to_csv(save_dir / "external_test_aligned.csv", index=False)

    with open(save_dir / "shared_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(shared_features, f, ensure_ascii=False, indent=2)

    logger.info("共享特征数：%s", len(shared_features))
    logger.info("已保存对齐后的跨数据集文件到：%s", save_dir)


if __name__ == "__main__":
    main()
