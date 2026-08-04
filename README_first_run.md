# First preprocessing run

This is the short preprocessing guide. Start with the formal project overview in [README.md](README.md), and use [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete experiment order.

## 1. Prepare the environment

Open the repository folder in PyCharm, create or select a Python interpreter, and install `requirements.txt`. For full CUDA training, install the PyTorch build appropriate for the training machine.

## 2. Place raw datasets

The default case-sensitive layout is:

```text
datasets/raw/
├── ciciot2023/
│   └── CSV/
└── ciciomt2024/
    └── WiFi_and_MQTT/
        └── attacks/
            └── csv/
                ├── train/
                └── test/
```

Alternatively, set `FT_IOT_RAW_DATA_DIR` to the directory that contains `ciciot2023/` and `ciciomt2024/`. Raw and processed datasets are intentionally ignored by Git.

## 3. Check the files

Open `preprocessing/00_check_raw_data.py` in PyCharm and click the green triangle. The script reports the discovered CSV count, a representative schema, and the detected label column without training a model.

## 4. Build CICIoT2023 binary data

Open `preprocessing/01_build_ciciot2023_binary.py` and click the green triangle. Its generated data files are written to:

```text
datasets/processed/ciciot2023_binary/
```

The generated directory can include train/validation/test CSV files, a scaler, feature columns, label mapping, and metadata. These local products are not committed; only selected schema JSON files are archived under `artifacts/data_schema/`.

## 5. Continue to training or other tasks

- Binary training: `trainers/train_ft_binary.py`
- Binary evaluation: `trainers/evaluate_binary.py`
- Multiclass preprocessing: `preprocessing/02_build_ciciot2023_7class.py`
- CICIoMT2024 preprocessing: `preprocessing/03_build_ciciomt2024_binary.py`
- Cross-dataset alignment: `preprocessing/04_align_cross_dataset_binary.py`

`FAST_DEBUG_MODE` in `config.py` controls whether preprocessing uses a reduced subset. Confirm its value before any formal experiment. Existing archived results are historical outputs and are not regenerated merely by following this first-run guide.

Terminal alternative:

```bash
python preprocessing/00_check_raw_data.py
python preprocessing/01_build_ciciot2023_binary.py
```
