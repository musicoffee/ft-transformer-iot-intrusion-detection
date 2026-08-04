import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

_raw_data_root = os.environ.get("FT_IOT_RAW_DATA_DIR")
RAW_DATA_DIR = (
    Path(_raw_data_root).expanduser()
    if _raw_data_root
    else PROJECT_ROOT / "datasets" / "raw"
)
PROCESSED_DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
ARTIFACT_DIR = PROJECT_ROOT / "outputs" / "artifacts"

# ========= 运行模式 =========
# True：先用小规模子集验证流程是否跑通（非常适合 PyCharm 首次运行）
# False：尽量读取更多文件，适合正式实验前的完整预处理
FAST_DEBUG_MODE = False

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.125  # 先切 train_temp/test，再从 train_temp 切出 val；约等于 7:1:2

# ========= 数据路径 =========
CICIOT2023_RAW_DIR = RAW_DATA_DIR / "ciciot2023" / "CSV"
CICIOMT2024_WIFI_MQTT_TRAIN_DIR = RAW_DATA_DIR / "ciciomt2024" / "WiFi_and_MQTT" / "attacks" / "csv" / "train"
CICIOMT2024_WIFI_MQTT_TEST_DIR = RAW_DATA_DIR / "ciciomt2024" / "WiFi_and_MQTT" / "attacks" / "csv" / "test"

# ========= 调试模式下的采样规模 =========
MAX_FILES_DEBUG_CICIOT2023 = 12
MAX_FILES_DEBUG_CICIOMT2024 = 6
MAX_SAMPLES_PER_CLASS_BINARY = 50000
MAX_SAMPLES_PER_CLASS_7CLASS = 20000

# ========= 通用候选列名 =========
LABEL_COLUMN_CANDIDATES = ["label", "Label", "attack_type", "Attack", "class", "Class"]
TIMESTAMP_COLUMN_CANDIDATES = ["Timestamp", "timestamp", "ts", "flow_timestamp"]

# ========= 输出文件名 =========
CICIOT2023_BINARY_NAME = "ciciot2023_binary"
CICIOT2023_7CLASS_NAME = "ciciot2023_7class"
CICIOMT2024_BINARY_NAME = "ciciomt2024_binary"
CROSS_DATASET_ALIGNED_NAME = "cross_dataset_aligned_binary"
