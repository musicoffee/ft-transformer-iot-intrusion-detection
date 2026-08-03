from pathlib import Path
from math import ceil
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    CICIOT2023_7CLASS_NAME,
    CICIOT2023_RAW_DIR,
    FAST_DEBUG_MODE,
    LABEL_COLUMN_CANDIDATES,
    LOG_DIR,
    MAX_FILES_DEBUG_CICIOT2023,
    MAX_SAMPLES_PER_CLASS_7CLASS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TIMESTAMP_COLUMN_CANDIDATES,
    VAL_SIZE_WITHIN_TRAIN,
)
from utils.data_utils import (
    drop_timestamp_columns,
    encode_labels,
    ensure_dir,
    find_label_column,
    fit_and_save_scaler,
    get_label_column_from_df_or_path,
    infer_label_from_file_path,
    keep_numeric_and_factorize_others,
    list_csv_files,
    map_ciciot2023_to_7class,
    replace_inf_and_handle_nan,
    safe_read_csv,
    save_metadata,
    save_splits,
    split_train_val_test,
    summarize_class_distribution,
)
from utils.logger_utils import get_logger


# =========================
# 正式模式下的 7 类总采样规模
# =========================
# 解释：
# - 7 类比二分类更占内存
# - 这里默认每类目标最多 80000
# - 理想上限总量约 7 * 80000 = 560000
# - 对你现在这台机器更稳
FORMAL_TARGET_SAMPLES_PER_CLASS_7CLASS = 80000


def build_file_groups(csv_files: list[Path]) -> dict[str, list[Path]]:
    """
    先根据路径推断原始攻击名，再映射到 7 类。
    Unknown 类先丢掉，不纳入后续处理。
    """
    groups = defaultdict(list)

    for file_path in csv_files:
        raw_label = infer_label_from_file_path(file_path)
        class_7 = map_ciciot2023_to_7class(raw_label)

        if class_7 == "Unknown":
            continue

        groups[class_7].append(file_path)

    return groups


def compute_per_file_quota(groups: dict[str, list[Path]], target_per_class: int) -> dict[str, int]:
    quota_map = {}
    for class_name, files in groups.items():
        if len(files) == 0:
            quota_map[class_name] = target_per_class
        else:
            quota_map[class_name] = max(1, ceil(target_per_class / len(files)))
    return quota_map


def align_columns(df: pd.DataFrame, reference_columns: list[str], label_col: str) -> pd.DataFrame:
    df = df.copy()

    for col in reference_columns:
        if col not in df.columns:
            if col == label_col:
                df[col] = "unknown"
            else:
                df[col] = 0.0

    extra_cols = [c for c in df.columns if c not in reference_columns]
    if extra_cols:
        df = df.drop(columns=extra_cols, errors="ignore")

    df = df[reference_columns].copy()
    return df


def sample_file_df(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def build_dataframe(logger) -> pd.DataFrame:
    csv_files = list_csv_files(CICIOT2023_RAW_DIR)
    if not csv_files:
        raise FileNotFoundError(f"未在 {CICIOT2023_RAW_DIR} 中找到 CSV 文件。")

    if FAST_DEBUG_MODE:
        selected_files = csv_files[:MAX_FILES_DEBUG_CICIOT2023]
        target_per_class = MAX_SAMPLES_PER_CLASS_7CLASS
        logger.info("FAST_DEBUG_MODE=True，仅使用前 %s 个文件做首轮验证。", len(selected_files))
    else:
        selected_files = csv_files
        target_per_class = FORMAL_TARGET_SAMPLES_PER_CLASS_7CLASS
        logger.info("FAST_DEBUG_MODE=False，将使用全部 %s 个文件。", len(selected_files))
        logger.info("正式模式下，每类目标样本数上限：%s", target_per_class)

    groups = build_file_groups(selected_files)
    quota_map = compute_per_file_quota(groups, target_per_class)

    logger.info("7类文件分组统计：%s", {k: len(v) for k, v in groups.items()})
    logger.info("单文件采样上限：%s", quota_map)

    frames = []
    reference_columns = None
    reference_label_col = None

    for file_path in tqdm(selected_files, desc="读取 CICIoT2023_7class"):
        try:
            raw_label_from_path = infer_label_from_file_path(file_path)
            class_7_from_path = map_ciciot2023_to_7class(raw_label_from_path)

            if class_7_from_path == "Unknown":
                logger.info("文件 %s 映射后为 Unknown，已跳过。", file_path.name)
                continue

            per_file_quota = quota_map[class_7_from_path]

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
            df[label_col] = df[label_col].apply(map_ciciot2023_to_7class)
            df = df[df[label_col] != "Unknown"].copy()

            if df.empty:
                logger.info("文件 %s 在 7 类映射后为空，已跳过。", file_path.name)
                continue

            df, _ = keep_numeric_and_factorize_others(df, exclude_cols=[label_col])
            df = replace_inf_and_handle_nan(df, label_col=label_col)

            df = sample_file_df(
                df=df,
                max_rows=per_file_quota,
                random_state=RANDOM_STATE
            )

            feature_cols = [c for c in df.columns if c != label_col]
            df[feature_cols] = df[feature_cols].astype(np.float32)

            if reference_columns is None:
                reference_label_col = label_col
                reference_columns = list(df.columns)
            else:
                if label_col != reference_label_col:
                    df = df.rename(columns={label_col: reference_label_col})
                    label_col = reference_label_col

                df = align_columns(df, reference_columns=reference_columns, label_col=reference_label_col)

            frames.append(df)

        except Exception as e:
            logger.warning("跳过文件 %s，原因：%s", file_path.name, e)

    if not frames:
        raise RuntimeError("没有成功读取任何文件。")

    full_df = pd.concat(frames, axis=0, ignore_index=True)
    label_col = find_label_column(full_df, LABEL_COLUMN_CANDIDATES)

    balanced_frames = []
    for class_name, group in full_df.groupby(label_col):
        if len(group) > target_per_class:
            group = group.sample(n=target_per_class, random_state=RANDOM_STATE)
        balanced_frames.append(group)

    full_df = pd.concat(balanced_frames, axis=0, ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    logger.info("合并后数据形状：%s", full_df.shape)
    logger.info("7类标签分布：%s", summarize_class_distribution(full_df, label_col))
    return full_df


def main() -> None:
    logger = get_logger(LOG_DIR, "02_build_ciciot2023_7class")
    save_dir = PROCESSED_DATA_DIR / CICIOT2023_7CLASS_NAME
    ensure_dir(save_dir)

    df = build_dataframe(logger)
    label_col = find_label_column(df, LABEL_COLUMN_CANDIDATES)

    df, label_mapping = encode_labels(df, label_col=label_col)

    exclude_cols = [label_col, "label_text", "label_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    train_df, val_df, test_df = split_train_val_test(
        df,
        label_col_for_split="label_id",
        test_size=TEST_SIZE,
        val_size_within_train=VAL_SIZE_WITHIN_TRAIN,
        random_state=RANDOM_STATE,
    )

    train_df, val_df, test_df = fit_and_save_scaler(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        save_dir=save_dir,
    )

    save_splits(train_df, val_df, test_df, save_dir)
    save_metadata(
        save_dir=save_dir,
        feature_cols=feature_cols,
        label_mapping=label_mapping,
        extra={
            "dataset_name": "CICIoT2023 7-class",
            "fast_debug_mode": FAST_DEBUG_MODE,
            "formal_target_samples_per_class_7class": FORMAL_TARGET_SAMPLES_PER_CLASS_7CLASS,
            "train_shape": list(train_df.shape),
            "val_shape": list(val_df.shape),
            "test_shape": list(test_df.shape),
            "train_distribution": summarize_class_distribution(train_df, "label_text"),
            "val_distribution": summarize_class_distribution(val_df, "label_text"),
            "test_distribution": summarize_class_distribution(test_df, "label_text"),
        }
    )

    logger.info("已保存到：%s", save_dir)
    logger.info("train.csv / val.csv / test.csv / scaler.joblib / feature_columns.json / label_mapping.json / metadata.json 已生成。")


if __name__ == "__main__":
    main()