# Main binary FT-Transformer

Historical outputs from `trainers/train_ft_binary.py` and `trainers/evaluate_binary.py` on the 120,000-sample CICIoT2023 binary test split.

- `ft_binary_test_metrics.json`: final binary metrics; the archived PR-AUC is `0.9962608943722311`.
- `ft_binary_test_classification_report.txt`: class-level precision, recall, F1, and support.
- `ft_binary_train_history.json`: epoch-wise training and validation history.

The archived PR-AUC rounds to `0.996261`, matching the published abstract and Table 7. Table 2 of the published article instead reports `0.996027` for FT-Transformer. This is an internal reporting inconsistency: the repository retains the machine-readable JSON value and records the Table 2 difference, but no experiment was rerun to disprove the alternate published value.

The inspected main checkpoint configuration uses 8 attention heads. The checkpoint itself is not distributed. These files were not regenerated during repository organization.
