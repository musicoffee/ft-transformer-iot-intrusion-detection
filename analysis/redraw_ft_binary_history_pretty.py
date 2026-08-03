from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


# =========================================================
# 新脚本：重绘 FT-Transformer 二分类训练历史曲线
# 放到：analysis/redraw_ft_binary_history_pretty.py
# 直接运行，不覆盖旧图
# 输出：
# outputs/figures/ft_binary_train_history_pretty.png
# outputs/figures/ft_binary_train_history_pretty.pdf
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

HISTORY_FILE = REPORT_DIR / "ft_binary_train_history.json"

OUTPUT_PNG = FIGURE_DIR / "ft_binary_train_history_pretty.png"
OUTPUT_PDF = FIGURE_DIR / "ft_binary_train_history_pretty.pdf"

TITLE = "FT-Transformer Binary Training History"
X_LABEL = "Epoch"
LOSS_LABEL = "Loss"
F1_LABEL = "F1 Score"

FIGSIZE = (7.2, 4.8)
DPI = 600


def load_history(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"未找到历史文件：{path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "当前脚本要求 history JSON 是 dict 格式，例如："
            "{'train_loss': [...], 'val_loss': [...], 'val_f1_attack': [...]}"
        )

    required_keys = ["train_loss", "val_loss", "val_f1_attack"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"history JSON 缺少字段：{key}，当前字段：{list(data.keys())}")

    return data


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    history = load_history(HISTORY_FILE)

    train_loss = [float(x) for x in history["train_loss"]]
    val_loss = [float(x) for x in history["val_loss"]]
    val_f1 = [float(x) for x in history["val_f1_attack"]]

    epochs = list(range(1, len(train_loss) + 1))

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

    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    # 左轴：Loss
    line1, = ax1.plot(
        epochs,
        train_loss,
        color="#1f77b4",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Train Loss",
    )

    line2, = ax1.plot(
        epochs,
        val_loss,
        color="#ff7f0e",
        linewidth=2.0,
        marker="s",
        markersize=4,
        label="Val Loss",
    )

    ax1.set_xlabel(X_LABEL)
    ax1.set_ylabel(LOSS_LABEL)
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # 右轴：F1，绿色
    ax2 = ax1.twinx()
    line3, = ax2.plot(
        epochs,
        val_f1,
        color="green",
        linewidth=2.2,
        marker="^",
        markersize=5,
        label="Val F1 (Attack)",
    )

    ax2.set_ylabel(F1_LABEL)

    f1_min = min(val_f1)
    f1_max = max(val_f1)
    pad = max((f1_max - f1_min) * 0.2, 0.002)
    ax2.set_ylim(f1_min - pad, f1_max + pad)

    ax1.set_title(TITLE)

    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved pretty binary history PNG to: {OUTPUT_PNG}")
    print(f"Saved pretty binary history PDF to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()