# Reproducibility guide

This guide describes the code paths and expected artifacts. The selected files under `results/` are historical outputs copied from the completed project; organizing this repository did not rerun the paper experiments.

## Hardware and environment

The paper reports an NVIDIA RTX 3070 Ti Laptop GPU, PyCharm 2024.3.2 Community Edition, Python 3.12.2, PyTorch 2.7.0+cu128, scikit-learn 1.6.1, XGBoost 3.0.3, NumPy 2.2.2, and Pandas 2.2.3. The complete transitive environment was not archived as a lockfile.

Large preprocessing and neural-network training should be run on the game laptop with approximately 8 GB of NVIDIA GPU memory. The MacBook Air is recommended for code reading, documentation, result inspection, and the data-free smoke test. If GPU memory is insufficient, reduce batch size before changing the model definition.

In PyCharm, open the repository, select the intended interpreter, open each entry file, and click its green triangle. Terminal commands shown below are alternatives.

## 0. Install and smoke-test

Input: `requirements.txt` and no dataset.

Output: a `(4, 2)` finite-logit assertion from the main model.

```bash
python -m pip install -r requirements.txt
python tests/smoke_test.py
```

For CUDA training, choose the PyTorch wheel that matches the installed CUDA environment. Do not infer a CUDA build from the generic `torch==2.7.0` requirement alone.

## 1. Check raw data

Entry: `preprocessing/00_check_raw_data.py`

Input: raw CICIoT2023 and CICIoMT2024 directories described in [DATASETS.md](DATASETS.md).

Output: console-only discovery and schema checks.

```bash
python preprocessing/00_check_raw_data.py
```

## 2. Build CICIoT2023 binary splits

Entry: `preprocessing/01_build_ciciot2023_binary.py`

Input: `datasets/raw/ciciot2023/CSV/*.csv`

Output: `datasets/processed/ciciot2023_binary/` containing train/validation/test data, scaler, features, labels, and metadata.

```bash
python preprocessing/01_build_ciciot2023_binary.py
```

Check `FAST_DEBUG_MODE` in `config.py` before treating a run as formal. The scaler is fitted on the training split in this main pipeline.

## 3. Train and evaluate the main binary model

Entries:

- `trainers/train_ft_binary.py`: consumes the binary train and validation splits; writes checkpoints, history, and figures under `outputs/`.
- `trainers/evaluate_binary.py`: consumes the test split and saved best checkpoint; writes metrics, classification report, and evaluation figures.

```bash
python trainers/train_ft_binary.py
python trainers/evaluate_binary.py
```

The evidence-backed summary is in `configs/binary.json`. The main model uses 8 attention heads. Checkpoints are local products and are intentionally excluded from Git.

## 4. Run classical binary baselines

Classical baseline code is in `baselines/`, while paired-comparison and audit code is in `analysis/`. Historical files in `results/legacy_100k_test/` used 200,000 training and 100,000 test samples. `analysis/revision_03_mcnemar_test.py` uses up to 200,000 training samples and the full test split (`MAX_TEST_SAMPLES = None`), producing the 120,000-sample paper/McNemar comparison.

```bash
python analysis/revision_03_mcnemar_test.py
```

Do not compare the legacy 100k accuracy values directly with the 120k paper values without labeling the differing test population.

## 5. Build, train, and evaluate multiclass data

Entries:

- `preprocessing/02_build_ciciot2023_7class.py`
- `trainers/train_ft_multiclass.py`
- `trainers/evaluate_multiclass.py`
- optional weighted variant: `trainers/train_ft_multiclass_weighted.py`

```bash
python preprocessing/02_build_ciciot2023_7class.py
python trainers/train_ft_multiclass.py
python trainers/evaluate_multiclass.py
python trainers/train_ft_multiclass_weighted.py
```

Despite the historical processed-directory name `ciciot2023_7class`, the label mapping contains **7 attack families plus Benign, 8 classes in total**.

## 6. Align CICIoT2023 and CICIoMT2024

Entries:

- `preprocessing/03_build_ciciomt2024_binary.py`
- `preprocessing/04_align_cross_dataset_binary.py`
- `analysis/run_cross_dataset.py`
- `analysis/run_cross_dataset_baselines.py`

Input: processed CICIoT2023 and CICIoMT2024 binary splits.

Output: aligned local CSVs, a 38-feature list, model checkpoint, and cross-dataset metrics.

```bash
python preprocessing/03_build_ciciomt2024_binary.py
python preprocessing/04_align_cross_dataset_binary.py
python analysis/run_cross_dataset.py
python analysis/run_cross_dataset_baselines.py
```

The historical alignment step transformed source and target data using their respective scaler statistics. Therefore, the archived external result is not strict source-only zero-shot evaluation.

## 7. Generate SHAP analyses

Entries:

- `analysis/shap_binary.py`: global binary-model SHAP outputs.
- `analysis/revision_05_local_shap_explanations.py`: selected local explanations.

Input: processed test data and a locally available checkpoint.

Output: figures/reports under `outputs/`; selected small tabular outputs are archived in `results/explainability/`.

```bash
python analysis/shap_binary.py
python analysis/revision_05_local_shap_explanations.py
```

## 8. Run revision validations

| Entry | Purpose | Archived output |
|---|---|---|
| `analysis/revision_01_make_reproducibility_tables.py` | Split, class, and shared-feature tables | `results/tables/` |
| `analysis/revision_02_check_scaler_and_baselines.py` | Scaler and baseline audit | no selected formal result |
| `analysis/revision_03_mcnemar_test.py` | Full-test model comparison | `results/mcnemar/` |
| `analysis/revision_03a_check_ft_checkpoints.py` | Checkpoint configuration audit | no checkpoint is distributed |
| `analysis/revision_10_modern_tabular_baselines.py` | Modern tabular baselines | `results/revision_validations/modern_tabular_baseline_results.json` |
| `analysis/revision_11_feature_union_validation.py` | Feature-union validation | `results/revision_validations/feature_union_validation_results.json` |
| `analysis/revision_12_netflow_iot_external_validation.py` | NF-ToN-IoT to NF-BoT-IoT validation | `results/revision_validations/netflow_iot_external_validation_results.json` |
| `analysis/revision_13_domain_aligned_ft_coral.py` | Historical CORAL comparison | `results/domain_alignment/original/` |

Run any selected file through PyCharm's green triangle after verifying its data paths and output directory. The historical extended-ablation output is not distributed because its main-model row incorrectly labels the checkpoint as 4 heads; it must be audited or rerun before release.

## 9. Trace results

Use [results/README.md](../results/README.md) to map each directory to its producing script and interpretation. Use `artifacts/data_schema/` to confirm saved feature/label structure. A complete reproduction requires third-party datasets, enough compute, and newly generated local checkpoints; those prerequisites are not bundled.
