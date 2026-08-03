from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    CICIOMT2024_BINARY_NAME,
    CICIOMT2024_WIFI_MQTT_TEST_DIR,
    CICIOMT2024_WIFI_MQTT_TRAIN_DIR,
    FAST_DEBUG_MODE,
    LABEL_COLUMN_CANDIDATES,
    LOG_DIR,
    MAX_FILES_DEBUG_CICIOMT2024,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TIMESTAMP_COLUMN_CANDIDATES,
)
from utils.data_utils import (
    drop_timestamp_columns,
    encode_labels,
    ensure_dir,
    find_label_column,
    fit_and_save_scaler,
    get_label_column_from_df_or_path,
    keep_numeric_and_factorize_others,
    list_csv_files,
    map_to_binary_label,
    replace_inf_and_handle_nan,
    safe_read_csv,
    save_metadata,
    save_splits,
    summarize_class_distribution,
)
from utils.logger_utils import get_logger


def read_split(folder: Path, split_name: str, logger) -> pd.DataFrame:
    csv_files = list_csv_files(folder)
    if not csv_files:
        raise FileNotFoundError(f"{split_name} 目录中未找到 CSV 文件：{folder}")

    if FAST_DEBUG_MODE:
        csv_files = csv_files[:MAX_FILES_DEBUG_CICIOMT2024]
        logger.info("%s：FAST_DEBUG_MODE=True，仅使用前 %s 个文件。", split_name, len(csv_files))
    else:
        logger.info("%s：使用全部 %s 个文件。", split_name, len(csv_files))

    frames = []
    for file_path in tqdm(csv_files, desc=f"读取 {split_name}"):
        try:
            df = safe_read_csv(file_path)
            df, label_col, inferred = get_label_column_from_df_or_path(
                df, file_path, LABEL_COLUMN_CANDIDATES
            )

            if inferred:
                logger.info(
                    "文件 %s 未显式提供标签列，已从路径推断标签：%s",
                    file_path.name,
                    df[label_col].iloc[0]
                )

            df = drop_timestamp_columns(df, TIMESTAMP_COLUMN_CANDIDATES)
            df[label_col] = df[label_col].apply(map_to_binary_label)

            df, _ = keep_numeric_and_factorize_others(df, exclude_cols=[label_col])
            df = replace_inf_and_handle_nan(df, label_col=label_col)

            frames.append(df)
        except Exception as e:
            logger.warning("%s 中跳过文件 %s，原因：%s", split_name, file_path.name, e)

    if not frames:
        raise RuntimeError(f"{split_name} 未成功读取任何文件。")

    full_df = pd.concat(frames, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
    label_col = find_label_column(full_df, LABEL_COLUMN_CANDIDATES)

    logger.info("%s 形状：%s", split_name, full_df.shape)
    logger.info("%s 标签分布：%s", split_name, summarize_class_distribution(full_df, label_col))
    return full_df


def main() -> None:
    logger = get_logger(LOG_DIR, "03_build_ciciomt2024_binary")
    save_dir = PROCESSED_DATA_DIR / CICIOMT2024_BINARY_NAME
    ensure_dir(save_dir)

    train_df = read_split(CICIOMT2024_WIFI_MQTT_TRAIN_DIR, "train", logger)
    test_df = read_split(CICIOMT2024_WIFI_MQTT_TEST_DIR, "test", logger)

    train_label_col = find_label_column(train_df, LABEL_COLUMN_CANDIDATES)
    test_label_col = find_label_column(test_df, LABEL_COLUMN_CANDIDATES)

    if train_label_col != test_label_col:
        raise ValueError(f"train 标签列为 {train_label_col}，test 标签列为 {test_label_col}，请手动检查。")

    label_col = train_label_col

    common_cols = [c for c in train_df.columns if c in test_df.columns]
    train_df = train_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    train_df, label_mapping = encode_labels(train_df, label_col=label_col)

    inverse_map = {k: v for k, v in label_mapping.items()}
    test_df["label_text"] = test_df[label_col].astype(str)
    test_df["label_id"] = test_df[label_col].astype(str).map(inverse_map)

    unseen_mask = test_df["label_id"].isna()
    if unseen_mask.any():
        logger.warning("测试集中存在训练集中未见过的标签，已剔除 %s 行。", int(unseen_mask.sum()))
        test_df = test_df[~unseen_mask].copy()

    test_df["label_id"] = test_df["label_id"].astype(int)

    from sklearn.model_selection import train_test_split
    train_core, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=train_df["label_id"]
    )

    exclude_cols = [label_col, "label_text", "label_id"]
    feature_cols = [c for c in train_core.columns if c not in exclude_cols]

    train_core, val_df, test_df = fit_and_save_scaler(
        train_df=train_core,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        save_dir=save_dir,
    )

    save_splits(train_core, val_df, test_df, save_dir)
    save_metadata(
        save_dir=save_dir,
        feature_cols=feature_cols,
        label_mapping=label_mapping,
        extra={
            "dataset_name": "CICIoMT2024 Binary",
            "fast_debug_mode": FAST_DEBUG_MODE,
            "train_shape": list(train_core.shape),
            "val_shape": list(val_df.shape),
            "test_shape": list(test_df.shape),
            "train_distribution": summarize_class_distribution(train_core, "label_text"),
            "val_distribution": summarize_class_distribution(val_df, "label_text"),
            "test_distribution": summarize_class_distribution(test_df, "label_text"),
        }
    )

    logger.info("已保存到：%s", save_dir)


if __name__ == "__main__":
    main()