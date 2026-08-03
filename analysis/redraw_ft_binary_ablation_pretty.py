from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 新脚本：重绘二分类 d_token 消融柱状图
# 放到：analysis/redraw_ft_binary_ablation_pretty.py
# 直接运行，不覆盖旧图
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

ABLATION_FILE = REPORT_DIR / "ablation_ft_binary_summary.json"

OUTPUT_PNG = FIGURE_DIR / "ft_binary_ablation_summary_pretty.png"
OUTPUT_PDF = FIGURE_DIR / "ft_binary_ablation_summary_pretty.pdf"

TITLE = "FT-Transformer Binary Ablation Summary"
Y_LABEL = "F1 Score"

FIGSIZE = (7.0, 4.6)
DPI = 600


def load_ablation_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"未找到消融结果文件：{path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("当前脚本要求 ablation JSON 是 list[dict] 格式。")

    if len(data) == 0:
        raise ValueError("ablation JSON 为空。")

    return data


def dtoken_sort_key(row: dict) -> int:
    if "d_token" in row:
        return int(row["d_token"])

    name = str(row.get("trial_name", ""))
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 999999


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_ablation_rows(ABLATION_FILE)
    rows = sorted(rows, key=dtoken_sort_key)

    required_keys = ["trial_name", "best_val_f1_attack", "test_f1_attack"]
    for row in rows:
        for key in required_keys:
            if key not in row:
                raise KeyError(f"消融结果缺少字段：{key}，当前行：{row}")

    labels = [str(row["trial_name"]) for row in rows]
    best_val_f1 = [float(row["best_val_f1_attack"]) for row in rows]
    test_f1 = [float(row["test_f1_attack"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.34

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)

    bars1 = ax.bar(
        x - width / 2,
        best_val_f1,
        width=width,
        label="Best Val F1",
    )
    bars2 = ax.bar(
        x + width / 2,
        test_f1,
        width=width,
        label="Test F1",
    )

    ax.set_title(TITLE)
    ax.set_xlabel("Token Dimension Setting")
    ax.set_ylabel(Y_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    all_vals = best_val_f1 + test_f1
    ymin = min(all_vals) - 0.0025
    ymax = max(all_vals) + 0.0025
    ax.set_ylim(ymin, ymax)

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.00015,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved pretty ablation PNG to: {OUTPUT_PNG}")
    print(f"Saved pretty ablation PDF to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()