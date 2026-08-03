# FT-Transformer IoT/NIDS 预处理 Starter（PyCharm 直接运行版）

## 1. 当前阶段目标
这一版先解决两件事：

1. 把 **CICIoT2023** 下载并放到正确目录；
2. 先把 **数据预处理流程跑通**，生成：
   - `train.csv`
   - `val.csv`
   - `test.csv`
   - `scaler.joblib`
   - `feature_columns.json`
   - `label_mapping.json`

## 2. 目录结构
```text
ft_transformer_iot_preprocess_starter/
├─ config.py
├─ requirements.txt
├─ README_first_run.md
├─ utils/
│  ├─ logger_utils.py
│  └─ data_utils.py
├─ preprocessing/
│  ├─ 00_check_raw_data.py
│  ├─ 01_build_ciciot2023_binary.py
│  ├─ 02_build_ciciot2023_7class.py
│  ├─ 03_build_ciciomt2024_binary.py
│  └─ 04_align_cross_dataset_binary.py
├─ datasets/
│  ├─ raw/
│  │  ├─ CICIoT2023/
│  │  │  └─ CSV/
│  │  └─ CICIoMT2024/
│  │     └─ WiFi_and_MQTT/attacks/csv/
│  │        ├─ train/
│  │        └─ test/
│  └─ processed/
└─ outputs/
   ├─ logs/
   └─ artifacts/
```

## 3. 你现在先做什么
### 第一步：先把 CICIoT2023 的 CSV 文件放到这里
```text
datasets/raw/CICIoT2023/CSV/
```

### 第二步：在 PyCharm 里先运行
```text
preprocessing/00_check_raw_data.py
```
如果成功，你会看到：
- 找到多少个 CSV 文件
- 示例文件列名
- 自动识别到的标签列

### 第三步：运行二分类预处理
```text
preprocessing/01_build_ciciot2023_binary.py
```
如果成功，你会在：
```text
datasets/processed/ciciot2023_binary/
```
看到预处理后的结果文件。

## 4. 为什么先跑二分类
因为二分类最稳，最适合先验证你的环境、路径、依赖、编码、标签映射是否都正确。  
等二分类稳了，再跑 7 类分类和跨数据集实验。

## 5. 调试模式说明
`config.py` 里默认：
```python
FAST_DEBUG_MODE = True[raw](../nids_paper_project/data/raw)
```
这表示：
- 只读取部分文件
- 每类只保留一定样本
- 目的：先确认流程跑通，不让你一上来就把电脑跑崩

等你确认没问题后，再改成：
```python
FAST_DEBUG_MODE = False
```

## 6. 当前这版还没做什么
这一版还没开始训练 FT-Transformer。  
我们下一步会接着写：
- FT-Transformer 模型文件
- dataloader
- 训练脚本
- 指标与画图脚本
- SHAP 分析脚本
