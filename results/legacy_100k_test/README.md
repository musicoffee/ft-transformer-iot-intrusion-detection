# Legacy 100,000-test-sample baselines

`baseline_rf_metrics.json` and `baseline_xgboost_metrics.json` explicitly record `max_test_samples: 100000` and `max_train_samples: 200000`.

These files must not be confused with the complete 120,000-sample paper/McNemar comparison. On the 120,000-sample version, the archived model metrics report:

- Random Forest accuracy: `0.9818333333333333` (paper: `0.981833`)
- XGBoost accuracy: `0.9823166666666666` (paper: `0.982317`)

The 120k values are in `results/mcnemar/mcnemar_model_metrics.csv`. Both sets are historical outputs, not new reruns.
