# Result artifact index

All files below are selected **historical experiment outputs** copied without changing their recorded values. They were not regenerated during repository organization. Raw data, complete logs, scalers, and checkpoints are excluded.

| Directory | Experiment | Producing or evidence script | Important interpretation |
|---|---|---|---|
| `main_binary/` | Main CICIoT2023 binary FT-Transformer | `trainers/train_ft_binary.py`, `trainers/evaluate_binary.py` | Main checkpoint uses 8 heads; archived JSON and published Table 2 disagree on PR-AUC |
| `multiclass/` | Standard and weighted CICIoT2023 multiclass runs | `trainers/train_ft_multiclass.py`, `trainers/train_ft_multiclass_weighted.py` | 7 attack families plus Benign, 8 classes in total |
| `cross_dataset/` | 38-feature CICIoT2023/CICIoMT2024 evaluation | `analysis/run_cross_dataset.py`, `analysis/run_cross_dataset_baselines.py` | Target-domain scaler statistics were used; not strict zero-shot |
| `tables/` | Split, class, and shared-feature tables | `analysis/revision_01_make_reproducibility_tables.py` | Small descriptive CSVs only |
| `legacy_100k_test/` | Original RF/XGBoost files | baseline scripts | Exactly 100,000 test samples; not the paper's full-test values |
| `mcnemar/` | Full-test model metrics, paired tests, and predictions | `analysis/revision_03_mcnemar_test.py` | 120,000 predictions; no raw traffic features |
| `explainability/` | Selected local SHAP records | `analysis/revision_05_local_shap_explanations.py` | Standardized feature values and SHAP values only |
| `revision_validations/` | Modern baselines, feature union, and NetFlow validation | revision scripts 10–12 | Machine-readable revision results; feature-union FT ROC-AUC is not hidden |
| `domain_alignment/original/` | Historical source-only and CORAL runs | `analysis/revision_13_domain_aligned_ft_coral.py` | Original 4-head reimplementation with unresolved validity issues |

The historical extended-ablation CSV/JSON is deliberately not included because it labels the main checkpoint as 4 heads while the checkpoint configuration indicates 8. It requires script correction or a fresh run.

For cross-cutting caveats, see [docs/LIMITATIONS.md](../docs/LIMITATIONS.md). For experiment order, see [docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
