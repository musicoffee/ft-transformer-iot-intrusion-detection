from __future__ import annotations

import re
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "revision_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUT_DIR / "revision_02_scaler_and_baseline_report.txt"


SCALER_PATHS = [
    PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_binary" / "scaler.joblib",
    PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_7class" / "scaler.joblib",
    PROJECT_ROOT / "datasets" / "processed" / "ciciomt2024_binary" / "scaler.joblib",
    PROJECT_ROOT / "datasets" / "processed" / "cross_dataset_aligned_binary" / "scaler.joblib",
]

SOURCE_FILES = [
    PROJECT_ROOT / "baselines" / "baseline_rf.py",
    PROJECT_ROOT / "baselines" / "baseline_xgboost.py",
    PROJECT_ROOT / "baselines" / "baseline_rf_multiclass.py",
    PROJECT_ROOT / "baselines" / "baseline_xgboost_multiclass.py",
    PROJECT_ROOT / "analysis" / "run_cross_dataset_baselines.py",
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_call_block(text: str, keywords: list[str]) -> list[str]:
    """
    提取 RandomForestClassifier(...) / XGBClassifier(...) 这类初始化代码块。
    """
    lines = text.splitlines()
    blocks = []

    for i, line in enumerate(lines):
        if any(k in line for k in keywords):
            block = [line]
            paren_balance = line.count("(") - line.count(")")

            j = i + 1
            while paren_balance > 0 and j < len(lines):
                block.append(lines[j])
                paren_balance += lines[j].count("(") - lines[j].count(")")
                j += 1

            blocks.append("\n".join(block))

    return blocks


def find_constant_assignments(text: str) -> list[str]:
    """
    提取类似 MAX_TRAIN_SAMPLES = 200000 这类大写常量。
    """
    results = []
    pattern = re.compile(r"^[A-Z][A-Z0-9_]*\s*=\s*.+$", re.MULTILINE)

    for m in pattern.finditer(text):
        line = m.group(0).strip()
        if any(key in line for key in [
            "MAX_", "N_", "NUM_", "RANDOM", "SEED", "TEST", "TRAIN",
            "N_ESTIMATORS", "LEARNING", "DEPTH", "SAMPLE"
        ]):
            results.append(line)

    return results


def inspect_scalers() -> list[str]:
    lines = []
    lines.append("=" * 80)
    lines.append("1. Scaler information")
    lines.append("=" * 80)

    for path in SCALER_PATHS:
        lines.append(f"\nPath: {path}")

        if not path.exists():
            lines.append("Status: NOT FOUND")
            continue

        try:
            scaler = joblib.load(path)
            lines.append(f"Scaler class: {scaler.__class__.__name__}")
            lines.append(f"Scaler module: {scaler.__class__.__module__}")

            for attr in ["mean_", "scale_", "var_", "data_min_", "data_max_", "data_range_", "n_features_in_"]:
                if hasattr(scaler, attr):
                    value = getattr(scaler, attr)
                    if hasattr(value, "shape"):
                        lines.append(f"{attr}: shape={value.shape}")
                    else:
                        lines.append(f"{attr}: {value}")

        except Exception as e:
            lines.append(f"Error loading scaler: {repr(e)}")

    return lines


def inspect_baseline_code() -> list[str]:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("2. Baseline model initialization code")
    lines.append("=" * 80)

    for path in SOURCE_FILES:
        lines.append(f"\nFile: {path}")

        if not path.exists():
            lines.append("Status: NOT FOUND")
            continue

        text = read_text(path)

        constants = find_constant_assignments(text)
        if constants:
            lines.append("\nDetected constants:")
            for c in constants:
                lines.append(f"  {c}")

        blocks = extract_call_block(
            text,
            keywords=[
                "RandomForestClassifier",
                "XGBClassifier",
                "XGBRFClassifier",
            ],
        )

        if not blocks:
            lines.append("\nNo RandomForestClassifier or XGBClassifier block detected.")
        else:
            lines.append("\nDetected model blocks:")
            for idx, b in enumerate(blocks, start=1):
                lines.append(f"\n--- Block {idx} ---")
                lines.append(b)

    return lines


def main() -> None:
    lines = []
    lines.extend(inspect_scalers())
    lines.extend(inspect_baseline_code())

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\n" + "=" * 80)
    print(f"Saved report to: {REPORT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()