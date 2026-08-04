# FT-Transformer-Based IoT Network Attack Detection and Cross-Dataset Generalization Analysis

Official code and archived experiment artifacts for:

> Fapeng Li, Yatong Tao, and Leilei Qu. “FT-Transformer-Based IoT Network Attack Detection and Cross-Dataset Generalization Analysis.” *Electronics* 15, no. 12 (2026): 2516. [https://doi.org/10.3390/electronics15122516](https://doi.org/10.3390/electronics15122516)

This repository studies whether an FT-Transformer can detect malicious IoT network traffic, distinguish attack families, and retain useful performance when evaluated across datasets with different feature distributions. It contains preprocessing, training, evaluation, baseline, interpretability, and revision-validation code together with selected small machine-readable results. Raw datasets and model weights are not distributed.

## Tasks and datasets

- CICIoT2023 binary classification: Attack versus Benign.
- CICIoT2023 multiclass classification: **7 attack families plus Benign, 8 classes in total**.
- CICIoT2023 to CICIoMT2024 aligned-feature evaluation using 38 shared features.
- NF-ToN-IoT to NF-BoT-IoT external validation in revision experiments.
- Historical CORAL domain-alignment experiments, retained with explicit caveats.

Dataset sources and expected local layout are documented in [docs/DATASETS.md](docs/DATASETS.md).

## Model

The executable main model is in [`models/ft_transformer.py`](models/ft_transformer.py). Numerical features are converted to tokens, a learnable CLS token is prepended, and Transformer encoder layers feed a classification head. The saved main checkpoint configuration and the main training scripts use:

| Setting | Value |
|---|---:|
| Input features | 39 |
| Token dimension | 64 |
| Attention heads | **8** |
| Encoder layers | 4 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |

The historical CORAL script defines a separate FT-Transformer-like class with 4 heads and architectural differences; it does not fully reuse the main model.

![Conceptual FT-Transformer architecture](docs/assets/ft-transformer-architecture-concept.png)

The image above is a conceptual model illustration, not an experimental result or an exact executable layer specification. In particular, the repository implementation is the authoritative source for architecture details.

## Repository structure

```text
.
├── preprocessing/          # Dataset checks, preprocessing, and feature alignment
├── datasets_code/          # Dataset and DataLoader helpers
├── models/                 # Main FT-Transformer implementation
├── trainers/               # Main training, evaluation, and ablations
├── baselines/              # Random Forest and XGBoost baselines
├── analysis/               # Cross-dataset, SHAP, revision, and audit scripts
├── artifacts/data_schema/  # Selected schemas, mappings, and metadata only
├── configs/                # Evidence-backed experiment configuration summaries
├── results/                # Selected historical machine-readable results
├── outputs/figures/        # Existing experiment figures
├── docs/                   # Dataset, reproduction, and limitation notes
└── tests/smoke_test.py      # Data-free model smoke test
```

## Environment

The paper reports Python 3.12.2, PyTorch 2.7.0+cu128, scikit-learn 1.6.1, XGBoost 3.0.3, NumPy 2.2.2, Pandas 2.2.3, and PyCharm 2024.3.2 Community Edition. The complete original environment was not archived as a lockfile, so unreported transitive versions remain unknown.

Create a virtual environment, select it as the PyCharm interpreter, and install:

```bash
python -m pip install -r requirements.txt
```

CUDA-enabled PyTorch wheels depend on the target CUDA platform. If GPU training is required, install the appropriate official PyTorch build before or after the remaining requirements rather than assuming the generic package index provides `+cu128`.

## Data preparation

1. Download datasets from their official providers; do not commit them to this repository.
2. Place them under `datasets/raw/` using the layout in [docs/DATASETS.md](docs/DATASETS.md), or set `FT_IOT_RAW_DATA_DIR` to another raw-data root.
3. In PyCharm, run `preprocessing/00_check_raw_data.py` with the green triangle.
4. Run the required preprocessing scripts in numeric order.
5. Confirm generated metadata and split sizes before training.

For a short preprocessing-only guide, see [README_first_run.md](README_first_run.md). For all experiment entry points, inputs, outputs, and terminal alternatives, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Recommended PyCharm run order

1. `preprocessing/00_check_raw_data.py`
2. `preprocessing/01_build_ciciot2023_binary.py`
3. `trainers/train_ft_binary.py`
4. `trainers/evaluate_binary.py`
5. `preprocessing/02_build_ciciot2023_7class.py`
6. `trainers/train_ft_multiclass.py`
7. `trainers/evaluate_multiclass.py`
8. `preprocessing/03_build_ciciomt2024_binary.py`
9. `preprocessing/04_align_cross_dataset_binary.py`
10. `analysis/run_cross_dataset.py`

Open each file in PyCharm, select the project interpreter, and click the green Run triangle. Large preprocessing and training are intended for a machine comparable to the reported NVIDIA RTX 3070 Ti Laptop GPU with about 8 GB VRAM; a MacBook Air is suitable for reading, result inspection, and the small smoke test.

Terminal alternatives:

```bash
python preprocessing/00_check_raw_data.py
python preprocessing/01_build_ciciot2023_binary.py
python trainers/train_ft_binary.py
python trainers/evaluate_binary.py
python tests/smoke_test.py
```

## Archived key results

These values are copied from existing experiment files; they were **not rerun while organizing this repository**.

| Experiment | Accuracy | F1 | ROC-AUC | PR-AUC | Source |
|---|---:|---:|---:|---:|---|
| Main binary FT-Transformer | 0.980675 | 0.980366 attack F1 | 0.995014 | **0.9962608943722311** | `results/main_binary/ft_binary_test_metrics.json` |
| Main multiclass FT-Transformer | 0.809554 | 0.724006 macro / 0.806680 weighted | — | — | `results/multiclass/ft_multiclass_test_metrics.json` |
| 38-feature CICIoMT2024 external test | 0.985987 | 0.992860 attack F1 | 0.987292 | 0.999671 | `results/cross_dataset/ft_binary_cross_dataset_metrics.json` |

The archived evaluation JSON records a PR-AUC of `0.9962608943722311`, which rounds to `0.996261` and agrees with the abstract and Table 7. Table 2 of the published article instead lists `0.996027` for FT-Transformer. This is an internal reporting inconsistency in the article. The repository preserves the archived JSON value while explicitly recording the discrepancy; the experiments were not rerun during repository organization. The `0.996027` value has not been experimentally disproved by a new run.

The FT-Transformer did **not** comprehensively outperform Random Forest and XGBoost. Original baseline files under `results/legacy_100k_test/` use 100,000 test samples, whereas the paper/McNemar comparison uses the complete 120,000-sample test split.

The 38-feature cross-dataset pipeline used target-domain scaler statistics, so this result must not be described as strict source-only zero-shot evaluation. In the feature-union validation, the FT-Transformer had comparatively low ROC-AUC; the machine-readable result is retained without reinterpretation. Historical CORAL results remain uncorrected and require strict revalidation.

![Binary confusion matrix](outputs/figures/ft_binary_confusion_matrix_pretty.png)

![Multiclass confusion matrix](outputs/figures/ft_multiclass_confusion_matrix_pretty.png)

## Reproduction scope and known limitations

The repository supports code inspection, data preprocessing, training, evaluation, and result tracing once users obtain the datasets. Included JSON/CSV/TXT files are historical experiment artifacts, not proof that the experiments were rerun after publication. Checkpoints and full data are intentionally excluded.

Important discrepancies and validity constraints—including the 8-head/4-head conflict, baseline sample-count mismatch, target-domain preprocessing, independent CSV factorization, and CORAL implementation issues—are recorded in [docs/LIMITATIONS.md](docs/LIMITATIONS.md). Result provenance is described in [results/README.md](results/README.md).

## Citation

```bibtex
@article{li2026fttransformer,
  author  = {Li, Fapeng and Tao, Yatong and Qu, Leilei},
  title   = {FT-Transformer-Based IoT Network Attack Detection and Cross-Dataset Generalization Analysis},
  journal = {Electronics},
  year    = {2026},
  volume  = {15},
  number  = {12},
  pages   = {2516},
  doi     = {10.3390/electronics15122516}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff). A code license has not yet been selected; dataset terms remain governed by their providers.
