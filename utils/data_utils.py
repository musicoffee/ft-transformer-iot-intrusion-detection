from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_csv_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.csv"))


def find_label_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    raise ValueError(f"未找到标签列。当前列名示例：{list(df.columns)[:20]}")


def infer_label_from_file_path(file_path: Path) -> str:
    """
    当原始 CSV 不带标签列时，从文件夹名或文件名推断标签。
    适配：
    - CICIoT2023: CSV/Backdoor_Malware/Backdoor_Malware.pcap.csv
    - CICIoMT2024: train/ARP_Spoofing_train.pcap.csv
    """
    generic_folder_names = {
        "csv", "train", "test", "attacks", "wifi_and_mqtt",
        "raw", "datasets", "ciciot2023", "ciciomt2024"
    }

    parent_name = file_path.parent.name.strip()
    parent_lower = parent_name.lower()

    if parent_lower not in generic_folder_names and parent_name:
        return parent_name

    name = file_path.name
    while True:
        changed = False
        for suffix in [".csv", ".pcap", "_train", "_test"]:
            if name.lower().endswith(suffix):
                name = name[:-len(suffix)]
                changed = True
        if not changed:
            break

    return name.strip()


def get_label_column_from_df_or_path(
    df: pd.DataFrame,
    file_path: Path,
    candidates: List[str],
    created_label_col_name: str = "Label"
) -> Tuple[pd.DataFrame, str, bool]:
    """
    优先从表里找标签列；
    如果没有，就从路径推断，并新建一个标签列。

    返回：
    - df
    - label_col
    - inferred(bool): 是否是推断出来的
    """
    try:
        label_col = find_label_column(df, candidates)
        return df, label_col, False
    except ValueError:
        inferred_label = infer_label_from_file_path(file_path)
        df = df.copy()
        df[created_label_col_name] = inferred_label
        return df, created_label_col_name, True


def drop_timestamp_columns(df: pd.DataFrame, timestamp_candidates: List[str]) -> pd.DataFrame:
    drop_cols = [
        c for c in df.columns
        if c in timestamp_candidates or c.lower() in [x.lower() for x in timestamp_candidates]
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_label_text(value: object) -> str:
    text = str(value).strip()
    text = text.replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    return text.lower()


def is_benign_label(raw_label: object) -> bool:
    """
    判断一个标签是否属于正常流量。
    这里不用“完全相等”，而是做更宽松匹配，
    因为 CICIoT2023 里会出现 Benign_Final 这种标签。
    """
    text = clean_label_text(raw_label)

    benign_keywords = [
        "benign",
        "normal",
        "background",
        "benign final",
        "benign traffic"
    ]

    return any(k in text for k in benign_keywords)


def map_to_binary_label(raw_label: object) -> str:
    if is_benign_label(raw_label):
        return "benign"
    return "attack"


def map_ciciot2023_to_7class(raw_label: object) -> str:
    text = clean_label_text(raw_label)

    if is_benign_label(raw_label):
        return "Benign"

    if "ddos" in text:
        return "DDoS"

    if text.startswith("dos ") or text == "dos" or "dos " in text:
        return "DoS"

    recon_keywords = [
        "recon", "scan", "host discovery", "os scan", "port scan",
        "ping sweep", "vulnerability scan"
    ]
    if any(k in text for k in recon_keywords):
        return "Recon"

    web_keywords = [
        "sql injection", "xss", "command injection",
        "uploading attack", "browser hijacking", "backdoor malware"
    ]
    if any(k in text for k in web_keywords):
        return "Web-Based"

    brute_keywords = [
        "brute force",
        "dictionary brute force",
        "password",
        "dictionary"
    ]
    if any(k in text for k in brute_keywords):
        return "Brute Force"

    spoof_keywords = [
        "spoof",
        "mitm",
        "arp spoof",
        "dns spoof"
    ]
    if any(k in text for k in spoof_keywords):
        return "Spoofing"

    if "mirai" in text:
        return "Mirai"

    return "Unknown"


def safe_read_csv(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, low_memory=False, encoding="latin1")
    return standardize_column_names(df)


def keep_numeric_and_factorize_others(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        non_na_ratio = converted.notna().mean()

        if non_na_ratio > 0.8:
            df[col] = converted
        else:
            df[col] = pd.factorize(df[col].astype(str).fillna("missing"))[0]

    return df, feature_cols


def replace_inf_and_handle_nan(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=[label_col])

    feature_cols = [c for c in df.columns if c != label_col]
    for col in feature_cols:
        if df[col].isna().any():
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            df[col] = df[col].fillna(median_value)

    return df


def cap_samples_per_class(
    df: pd.DataFrame,
    label_col: str,
    max_per_class: Optional[int],
    random_state: int
) -> pd.DataFrame:
    if max_per_class is None:
        return df

    frames = []
    for label, group in df.groupby(label_col):
        if len(group) > max_per_class:
            group = group.sample(n=max_per_class, random_state=random_state)
        frames.append(group)

    return pd.concat(frames, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def encode_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    labels = sorted(df[label_col].astype(str).unique().tolist())
    mapping = {label: idx for idx, label in enumerate(labels)}

    df = df.copy()
    df["label_text"] = df[label_col].astype(str)
    df["label_id"] = df[label_col].astype(str).map(mapping).astype(int)
    return df, mapping


def split_train_val_test(
    df: pd.DataFrame,
    label_col_for_split: str,
    test_size: float,
    val_size_within_train: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_temp, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col_for_split]
    )

    train_df, val_df = train_test_split(
        train_temp,
        test_size=val_size_within_train,
        random_state=random_state,
        stratify=train_temp[label_col_for_split]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_and_save_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    save_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(save_dir)

    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # 统一转 float，避免 pandas 对 int64 列回写浮点数时出现 FutureWarning
    train_df[feature_cols] = train_df[feature_cols].astype(float)
    val_df[feature_cols] = val_df[feature_cols].astype(float)
    test_df[feature_cols] = test_df[feature_cols].astype(float)

    scaler.fit(train_df[feature_cols])

    train_df[feature_cols] = scaler.transform(train_df[feature_cols]).astype(np.float32)
    val_df[feature_cols] = scaler.transform(val_df[feature_cols]).astype(np.float32)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols]).astype(np.float32)

    joblib.dump(scaler, save_dir / "scaler.joblib")
    return train_df, val_df, test_df


def save_metadata(
    save_dir: Path,
    feature_cols: List[str],
    label_mapping: Dict[str, int],
    extra: Optional[Dict] = None
) -> None:
    ensure_dir(save_dir)

    with open(save_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    with open(save_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    if extra is not None:
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(extra, f, ensure_ascii=False, indent=2)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: Path
) -> None:
    ensure_dir(save_dir)
    train_df.to_csv(save_dir / "train.csv", index=False)
    val_df.to_csv(save_dir / "val.csv", index=False)
    test_df.to_csv(save_dir / "test.csv", index=False)


def summarize_class_distribution(df: pd.DataFrame, col: str) -> Dict[str, int]:
    vc = df[col].value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}