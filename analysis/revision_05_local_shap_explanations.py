from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
    import shap
except ImportError as e:
    raise ImportError(
        "SHAP is not installed. Please run: pip install shap"
    ) from e


# ============================================================
# Local SHAP explanations for revision
# ------------------------------------------------------------
# Purpose:
#   Respond to reviewer comment:
#   "SHAP analysis remains superficial. Please include
#    localized explanations for specific misclassified samples."
#
# Output:
#   outputs/revision_tables/local_shap_selected_samples.csv
#   outputs/revision_tables/local_shap_values.csv
#   outputs/figures/local_shap_correct_attack.png
#   outputs/figures/local_shap_misclassified_attack_to_benign.png
#   outputs/figures/local_shap_combined.png
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.ft_transformer import FTTransformer  # noqa: E402


DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "ciciot2023_binary"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "ft_binary_best.pt"

OUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "revision_tables"
OUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BATCH_SIZE = 4096

# SHAP 计算参数
# 如果太慢，把 BACKGROUND_SIZE 改成 20，NSAMPLES 改成 100
BACKGROUND_SIZE = 50
NSAMPLES = 200
TOP_K = 12


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(device: torch.device):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    feature_columns = checkpoint["feature_columns"]
    label_mapping = checkpoint["label_mapping"]
    model_config = checkpoint["model_config"]

    attack_label_id = int(label_mapping["attack"])
    benign_label_id = int(label_mapping["benign"])

    return checkpoint, feature_columns, label_mapping, model_config, attack_label_id, benign_label_id


def build_model(checkpoint, model_config: dict, device: torch.device):
    model = FTTransformer(
        num_features=model_config["num_features"],
        num_classes=model_config["num_classes"],
        d_token=model_config["d_token"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_feature_data(feature_columns: list[str]):
    """
    必须严格按照 checkpoint 里的 feature_columns 读取特征。
    这样和 evaluate_binary.py / Table 1 的评估流程一致。
    """
    usecols = feature_columns + ["label_id"]

    train_df = pd.read_csv(TRAIN_FILE, usecols=usecols)
    test_df = pd.read_csv(TEST_FILE, usecols=usecols)

    X_train = train_df[feature_columns].values.astype(np.float32)
    y_train = train_df["label_id"].values.astype(int)

    X_test = test_df[feature_columns].values.astype(np.float32)
    y_test = test_df["label_id"].values.astype(int)

    return X_train, y_train, X_test, y_test


@torch.no_grad()
def predict_proba_numpy(model, X: np.ndarray, device: torch.device, batch_size: int = 4096):
    """
    输入 numpy，输出 softmax 概率。
    SHAP KernelExplainer 会调用这个函数。
    """
    model.eval()

    all_probs = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)

        logits = model(xb)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.detach().cpu().numpy())

    return np.vstack(all_probs)


def select_representative_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_attack: np.ndarray,
    attack_label_id: int,
    benign_label_id: int,
):
    """
    选择两个代表样本：
    1. 正确分类的 attack 样本：选择 attack probability 最高的样本；
    2. attack 被误分为 benign 的样本：选择 attack probability 最低的 false negative。
    """
    correct_attack_idx = np.where(
        (y_true == attack_label_id) & (y_pred == attack_label_id)
    )[0]

    fn_attack_to_benign_idx = np.where(
        (y_true == attack_label_id) & (y_pred == benign_label_id)
    )[0]

    if len(correct_attack_idx) == 0:
        raise ValueError("No correctly classified attack samples found.")

    if len(fn_attack_to_benign_idx) == 0:
        raise ValueError("No attack-to-benign misclassified samples found.")

    # 正确 attack：选模型最有信心的 attack
    correct_selected = correct_attack_idx[
        np.argmax(prob_attack[correct_attack_idx])
    ]

    # 误分 attack->benign：选模型最不认为它是 attack 的 false negative
    mis_selected = fn_attack_to_benign_idx[
        np.argmin(prob_attack[fn_attack_to_benign_idx])
    ]

    return int(correct_selected), int(mis_selected)


def extract_attack_shap_values(shap_values, attack_label_id: int):
    """
    兼容不同 SHAP 版本的返回格式：
    1. list: shap_values[class_id] -> (n_samples, n_features)
    2. ndarray 3D: (n_samples, n_features, n_outputs)
    3. ndarray 2D: (n_samples, n_features)
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[attack_label_id])

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        return shap_values[:, :, attack_label_id]

    if shap_values.ndim == 2:
        return shap_values

    raise ValueError(f"Unsupported SHAP values shape: {shap_values.shape}")


def extract_expected_value(expected_value, attack_label_id: int):
    if isinstance(expected_value, list):
        return float(expected_value[attack_label_id])

    arr = np.asarray(expected_value)

    if arr.ndim == 0:
        return float(arr)

    return float(arr[attack_label_id])


def make_local_bar_plot(
    shap_row: np.ndarray,
    x_row: np.ndarray,
    feature_names: list[str],
    base_value: float,
    pred_prob_attack: float,
    true_label_name: str,
    pred_label_name: str,
    title: str,
    out_path: Path,
):
    """
    画单个样本的 local SHAP 条形图。
    红色表示推动模型更倾向 attack；
    蓝色表示降低 attack 概率。
    """
    abs_order = np.argsort(np.abs(shap_row))[-TOP_K:]
    # 从小到大，barh 显示更自然
    plot_indices = abs_order[np.argsort(np.abs(shap_row[abs_order]))]

    values = shap_row[plot_indices]
    names = [feature_names[i] for i in plot_indices]
    feat_values = x_row[plot_indices]

    y_pos = np.arange(len(plot_indices))

    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in values]

    labels = [
        f"{name} = {feat_values[idx]:.4g}"
        for idx, name in enumerate(names)
    ]

    plt.figure(figsize=(9, 6))
    plt.barh(y_pos, values, color=colors, alpha=0.85)
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y_pos, labels, fontsize=9)
    plt.xlabel("SHAP value for attack-class probability")
    plt.title(title, fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    subtitle = (
        f"Base value = {base_value:.4f}, "
        f"Predicted P(attack) = {pred_prob_attack:.4f}, "
        f"True = {true_label_name}, Predicted = {pred_label_name}"
    )

    plt.figtext(0.5, 0.01, subtitle, ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_combined_plot(
    shap_correct: np.ndarray,
    shap_mis: np.ndarray,
    x_correct: np.ndarray,
    x_mis: np.ndarray,
    feature_names: list[str],
    prob_correct: float,
    prob_mis: float,
    out_path: Path,
):
    """
    生成合并图，方便论文作为 Figure 12 使用。
    """
    def prepare(shap_row, x_row):
        abs_order = np.argsort(np.abs(shap_row))[-TOP_K:]
        plot_indices = abs_order[np.argsort(np.abs(shap_row[abs_order]))]
        values = shap_row[plot_indices]
        names = [feature_names[i] for i in plot_indices]
        feat_values = x_row[plot_indices]
        labels = [f"{n}={v:.3g}" for n, v in zip(names, feat_values)]
        colors = ["#d62728" if v >= 0 else "#1f77b4" for v in values]
        return values, labels, colors

    values_a, labels_a, colors_a = prepare(shap_correct, x_correct)
    values_b, labels_b, colors_b = prepare(shap_mis, x_mis)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    y_a = np.arange(len(values_a))
    axes[0].barh(y_a, values_a, color=colors_a, alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_yticks(y_a)
    axes[0].set_yticklabels(labels_a, fontsize=8)
    axes[0].set_title(f"(a) Correctly classified attack\nP(attack) = {prob_correct:.4f}")
    axes[0].set_xlabel("SHAP value for attack-class probability")
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    y_b = np.arange(len(values_b))
    axes[1].barh(y_b, values_b, color=colors_b, alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(y_b)
    axes[1].set_yticklabels(labels_b, fontsize=8)
    axes[1].set_title(f"(b) Misclassified attack as benign\nP(attack) = {prob_mis:.4f}")
    axes[1].set_xlabel("SHAP value for attack-class probability")
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Local SHAP explanations for representative samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint, feature_columns, label_mapping, model_config, attack_label_id, benign_label_id = load_checkpoint(device)

    print(f"Feature count: {len(feature_columns)}")
    print(f"Label mapping: {label_mapping}")
    print(f"Attack label id: {attack_label_id}")
    print(f"Benign label id: {benign_label_id}")

    model = build_model(checkpoint, model_config, device)

    X_train, y_train, X_test, y_test = load_feature_data(feature_columns)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # 先得到全测试集预测，找代表样本
    probs_test = predict_proba_numpy(model, X_test, device, batch_size=BATCH_SIZE)
    y_pred = np.argmax(probs_test, axis=1)
    prob_attack = probs_test[:, attack_label_id]

    correct_idx, mis_idx = select_representative_samples(
        y_true=y_test,
        y_pred=y_pred,
        prob_attack=prob_attack,
        attack_label_id=attack_label_id,
        benign_label_id=benign_label_id,
    )

    print(f"Selected correctly classified attack sample index: {correct_idx}")
    print(f"Selected misclassified attack->benign sample index: {mis_idx}")
    print(f"Correct sample P(attack): {prob_attack[correct_idx]:.6f}")
    print(f"Misclassified sample P(attack): {prob_attack[mis_idx]:.6f}")

    selected_samples = pd.DataFrame([
        {
            "sample_type": "correctly_classified_attack",
            "test_index": correct_idx,
            "true_label": int(y_test[correct_idx]),
            "pred_label": int(y_pred[correct_idx]),
            "prob_attack": float(prob_attack[correct_idx]),
        },
        {
            "sample_type": "misclassified_attack_to_benign",
            "test_index": mis_idx,
            "true_label": int(y_test[mis_idx]),
            "pred_label": int(y_pred[mis_idx]),
            "prob_attack": float(prob_attack[mis_idx]),
        },
    ])

    selected_path = OUT_TABLE_DIR / "local_shap_selected_samples.csv"
    selected_samples.to_csv(selected_path, index=False, encoding="utf-8-sig")

    # 背景样本：从训练集随机抽一小部分
    rng = np.random.default_rng(RANDOM_STATE)
    background_idx = rng.choice(
        len(X_train),
        size=min(BACKGROUND_SIZE, len(X_train)),
        replace=False,
    )
    X_background = X_train[background_idx]

    X_explain = np.vstack([
        X_test[correct_idx],
        X_test[mis_idx],
    ]).astype(np.float32)

    # SHAP prediction function
    def shap_predict_fn(x_numpy):
        x_numpy = np.asarray(x_numpy, dtype=np.float32)
        return predict_proba_numpy(model, x_numpy, device, batch_size=1024)

    print("Building SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(shap_predict_fn, X_background)

    print("Computing local SHAP values...")
    shap_values = explainer.shap_values(X_explain, nsamples=NSAMPLES)

    attack_shap = extract_attack_shap_values(shap_values, attack_label_id)
    base_value_attack = extract_expected_value(explainer.expected_value, attack_label_id)

    shap_correct = attack_shap[0]
    shap_mis = attack_shap[1]

    # 保存 shap values
    shap_df_rows = []

    for sample_name, sample_idx, shap_row, x_row in [
        ("correctly_classified_attack", correct_idx, shap_correct, X_test[correct_idx]),
        ("misclassified_attack_to_benign", mis_idx, shap_mis, X_test[mis_idx]),
    ]:
        for feature_name, feature_value, shap_value in zip(feature_columns, x_row, shap_row):
            shap_df_rows.append({
                "sample_type": sample_name,
                "test_index": sample_idx,
                "feature": feature_name,
                "feature_value": float(feature_value),
                "shap_value_attack": float(shap_value),
                "abs_shap_value": float(abs(shap_value)),
            })

    shap_df = pd.DataFrame(shap_df_rows)
    shap_df = shap_df.sort_values(
        by=["sample_type", "abs_shap_value"],
        ascending=[True, False],
    )

    shap_values_path = OUT_TABLE_DIR / "local_shap_values.csv"
    shap_df.to_csv(shap_values_path, index=False, encoding="utf-8-sig")

    # 单图
    correct_fig = OUT_FIG_DIR / "local_shap_correct_attack.png"
    mis_fig = OUT_FIG_DIR / "local_shap_misclassified_attack_to_benign.png"
    combined_fig = OUT_FIG_DIR / "local_shap_combined.png"

    make_local_bar_plot(
        shap_row=shap_correct,
        x_row=X_test[correct_idx],
        feature_names=feature_columns,
        base_value=base_value_attack,
        pred_prob_attack=float(prob_attack[correct_idx]),
        true_label_name="attack",
        pred_label_name="attack",
        title="Local SHAP explanation: correctly classified attack sample",
        out_path=correct_fig,
    )

    make_local_bar_plot(
        shap_row=shap_mis,
        x_row=X_test[mis_idx],
        feature_names=feature_columns,
        base_value=base_value_attack,
        pred_prob_attack=float(prob_attack[mis_idx]),
        true_label_name="attack",
        pred_label_name="benign",
        title="Local SHAP explanation: misclassified attack as benign",
        out_path=mis_fig,
    )

    make_combined_plot(
        shap_correct=shap_correct,
        shap_mis=shap_mis,
        x_correct=X_test[correct_idx],
        x_mis=X_test[mis_idx],
        feature_names=feature_columns,
        prob_correct=float(prob_attack[correct_idx]),
        prob_mis=float(prob_attack[mis_idx]),
        out_path=combined_fig,
    )

    print("\nSaved outputs:")
    print(f"Selected samples: {selected_path}")
    print(f"Local SHAP values: {shap_values_path}")
    print(f"Correct sample figure: {correct_fig}")
    print(f"Misclassified sample figure: {mis_fig}")
    print(f"Combined figure: {combined_fig}")

    print("\nTop local SHAP features:")
    print(shap_df.groupby("sample_type").head(TOP_K).to_string(index=False))


if __name__ == "__main__":
    main()