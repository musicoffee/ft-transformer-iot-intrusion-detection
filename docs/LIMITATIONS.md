# Known limitations and evidence boundaries

This document records differences found by comparing the executable code, saved checkpoint configuration, JSON/CSV results, metadata, and logs. It is intended to prevent historical artifacts from being overinterpreted.

## 1. Main attention-head count

The main binary checkpoint configuration and `trainers/train_ft_binary.py` use **8 attention heads**. The main multiclass and 38-feature cross-dataset training scripts also set 8 heads. A historical extended-ablation table labels the main checkpoint row as 4 heads; that label conflicts with the checkpoint and is not distributed as final evidence. The ablation should be corrected at the script level and rerun before publication as a repository result.

## 2. Binary PR-AUC

The archived evaluation JSON at `results/main_binary/ft_binary_test_metrics.json` records a PR-AUC of `0.9962608943722311`, which rounds to `0.996261` and agrees with the abstract and Table 7. Table 2 of the published article instead lists `0.996027` for FT-Transformer. This is an internal reporting inconsistency in the article. The repository preserves the archived JSON value while explicitly recording the discrepancy; the experiments were not rerun during repository organization, and the Table 2 value has not been experimentally disproved by a new run.

## 3. Baseline test-population mismatch

The archived original RF and XGBoost JSON files specify `max_test_samples: 100000`. They are isolated under `results/legacy_100k_test/`. The paper/McNemar model metrics use the full 120,000-sample main test split; their reported accuracies are RF `0.981833` and XGBoost `0.982317`. These are different evaluations and must not be presented as a single directly interchangeable table without the sample-count distinction.

## 4. Multiclass naming

Some filenames and metadata say “7class,” but the actual label mapping contains Benign plus Brute Force, DDoS, DoS, Mirai, Recon, Spoofing, and Web-Based. The executable model has eight outputs: **7 attack families plus Benign, 8 classes in total**.

## 5. Historical CORAL architecture

`analysis/revision_13_domain_aligned_ft_coral.py` defines `FTTransformerWithRepresentation` inside the file. It does not import and reuse the main `FTTransformer` end to end. It defaults to 4 heads, omits the main tokenizer's separate learnable `feature_embedding`, and places normalization within a differently organized classification head. Thus the historical CORAL result is not a controlled change that only adds CORAL to the main checkpoint.

## 6. Historical CORAL loss scaling

The code first computes `mean((source_cov - target_cov) ** 2)`, which already divides the summed squared difference by the matrix element count, and then divides again by `4 * d * d`. Relative to the common squared-Frobenius formulation divided once by `4d²`, this appears to introduce an additional `d²` factor. Corrected-CORAL has not been rerun, so the stored values remain historical and uncorrected.

## 7. Target labels in splitting and sampling

The CORAL training loss receives target features and does not send target labels into gradient computation. However, target labels are used for stratified adaptation/test splitting and stratified sampling before features are supplied to training. The experiment is therefore not fully label-agnostic at the data-selection level.

## 8. Scaler fitting and target statistics

The main dataset preprocessors fit scalers on their respective training partitions rather than on their test partitions. However, the historical 38-feature cross-dataset alignment uses the already target-domain-scaled CICIoMT2024 data, so target-domain training statistics influence the target transformation. This is not strict source-only zero-shot preprocessing.

The NF-ToN-IoT to NF-BoT-IoT revision script explicitly fits a `StandardScaler` on the NF-ToN-IoT training split only. Interpret each experimental path separately rather than generalizing one scaler statement to every script.

## 9. Independent categorical factorization

The preprocessing scripts call the shared factorization utility while reading individual CSV files, before concatenating them. A categorical value can therefore receive different integers in different CSVs if category sets or orders differ. The effect is limited when selected features are already numeric, but the pipeline does not archive a single global categorical vocabulary.

## 10. Environment capture

The paper reports core versions: Python 3.12.2, PyTorch 2.7.0+cu128, scikit-learn 1.6.1, XGBoost 3.0.3, NumPy 2.2.2, and Pandas 2.2.3. A complete lockfile or exported environment with every direct and transitive version was not archived. `requirements.txt` pins only versions supported by the paper record and leaves other packages unpinned rather than inventing precision.

## 11. Result status

All selected result files are historical experiment products copied without changing their numeric values. They are not evidence of a new post-publication run. Data files, scalers, checkpoints, and large intermediate logs are intentionally excluded, so reproduction still requires the official datasets and new local training runs.

## 12. Conceptual figures

Files under `docs/assets/` copied from `Figure/ChatGPT Image ...png` are conceptual illustrations. They are not measured outputs, and some labels or drawn components do not exactly match the executable repository. The Python implementation and evidence-backed JSON configurations take precedence.
