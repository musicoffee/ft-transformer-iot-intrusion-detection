from pathlib import Path

from config import (
    CICIOT2023_RAW_DIR,
    CICIOMT2024_WIFI_MQTT_TEST_DIR,
    CICIOMT2024_WIFI_MQTT_TRAIN_DIR,
    LABEL_COLUMN_CANDIDATES,
    LOG_DIR,
)
from utils.data_utils import list_csv_files, safe_read_csv, get_label_column_from_df_or_path
from utils.logger_utils import get_logger


def inspect_one_dataset(dataset_name: str, folder: Path, logger) -> None:
    csv_files = list_csv_files(folder)

    logger.info("=" * 80)
    logger.info("数据集名称：%s", dataset_name)
    logger.info("目录：%s", folder)
    logger.info("CSV 文件数量：%s", len(csv_files))

    if not csv_files:
        logger.warning("没有发现任何 CSV 文件，请检查目录是否放对。")
        return

    sample_file = csv_files[0]
    logger.info("示例文件：%s", sample_file)

    df = safe_read_csv(sample_file)
    logger.info("示例文件形状：%s", df.shape)
    logger.info("前 20 个列名：%s", list(df.columns[:20]))

    try:
        df, label_col, inferred = get_label_column_from_df_or_path(
            df, sample_file, LABEL_COLUMN_CANDIDATES
        )
        if inferred:
            logger.warning("原始文件中未找到标签列，已根据路径自动推断标签列：%s", label_col)
        else:
            logger.info("识别到标签列：%s", label_col)

        logger.info("前 10 个唯一标签示例：%s", df[label_col].astype(str).unique()[:10])
    except Exception as e:
        logger.error("标签列识别失败：%s", e)


def main() -> None:
    logger = get_logger(LOG_DIR, "00_check_raw_data")

    inspect_one_dataset("CICIoT2023", CICIOT2023_RAW_DIR, logger)
    inspect_one_dataset("CICIoMT2024_train", CICIOMT2024_WIFI_MQTT_TRAIN_DIR, logger)
    inspect_one_dataset("CICIoMT2024_test", CICIOMT2024_WIFI_MQTT_TEST_DIR, logger)

    logger.info("检查完成。下一步请先运行 01_build_ciciot2023_binary.py。")


if __name__ == "__main__":
    main()