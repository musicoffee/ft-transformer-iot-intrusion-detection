# Full-test McNemar comparison

Historical outputs from `analysis/revision_03_mcnemar_test.py` using the complete 120,000-sample main binary test split:

- `mcnemar_model_metrics.csv`: FT-Transformer, Random Forest, and XGBoost metrics.
- `mcnemar_test_results.csv`: paired McNemar comparisons.
- `mcnemar_predictions_main_binary.csv`: true labels, three model predictions, and attack probabilities.

The prediction file contains exactly these seven columns: `y_true`, three prediction columns, and three probability columns. It does not contain raw traffic features, sample payloads, IP addresses, device identifiers, or an original record ID. It is retained to make the paired statistical tests auditable.
