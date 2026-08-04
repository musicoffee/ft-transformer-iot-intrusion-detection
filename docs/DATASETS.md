# Datasets

Raw and processed datasets are not included in this repository. Download each dataset from its provider, review the provider's current terms, and keep the files outside version control.

## Official sources

| Dataset | Provider page | Use in this repository |
|---|---|---|
| CICIoT2023 | [Canadian Institute for Cybersecurity: CIC IoT dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | Main binary task, 7 attack families plus Benign multiclass task, and source side of aligned evaluation |
| CICIoMT2024 | [Canadian Institute for Cybersecurity: CIC IoMT dataset 2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) | External binary evaluation and target side of aligned evaluation |
| NF-ToN-IoT and NF-BoT-IoT | [University of Queensland Cyber Research Centre: Machine Learning-Based NIDS Datasets](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets) | Revision external-validation and domain-alignment experiments |

The links above are dataset landing pages, not mirrored copies. Dataset licensing and redistribution conditions can change; users must verify the terms shown by the provider at download time. This repository does not claim ownership of third-party data or a right to redistribute it.

## Default local layout

Paths are case-sensitive on Linux. The repository now uses lowercase dataset directory names consistently while preserving provider-specific internal names:

```text
datasets/
├── raw/
│   ├── ciciot2023/
│   │   └── CSV/
│   │       └── *.csv
│   ├── ciciomt2024/
│   │   └── WiFi_and_MQTT/
│   │       └── attacks/
│   │           └── csv/
│   │               ├── train/
│   │               │   └── *.csv
│   │               └── test/
│   │                   └── *.csv
│   ├── nf_ton_iot/
│   │   └── NF-ToN-IoT.csv
│   └── nf_bot_iot/
│       └── NF-BoT-IoT.csv
└── processed/
    ├── ciciot2023_binary/
    ├── ciciot2023_7class/
    ├── ciciomt2024_binary/
    └── cross_dataset_aligned_binary/
```

You may keep raw data elsewhere by setting `FT_IOT_RAW_DATA_DIR` to the directory containing the dataset subdirectories. Processed data remains under `datasets/processed/` unless the code is deliberately changed.

## Archived schemas only

`artifacts/data_schema/` contains small JSON descriptions copied from the historical processed directories:

- feature names;
- label mappings;
- split shapes and class counts;
- the 38-feature intersection used by the aligned experiment.

It does not contain traffic samples, scalers, IP addresses, device identifiers, or model weights. A schema confirms expected structure but cannot replace the original data or preprocessing run.

## Data handling notes

- Do not commit raw CSVs or generated train/validation/test CSVs.
- Do not commit `scaler.joblib` or other serialized preprocessing objects.
- Confirm labels and columns before running a long job; dataset releases or extracted directory structures may differ.
- The historical preprocessing utilities factorized non-numeric columns within individual CSV files before concatenation. This can produce inconsistent integer meanings across files; see [LIMITATIONS.md](LIMITATIONS.md).
- The historical 38-feature aligned pipeline used statistics from each domain's own scaler. It is not strict source-only zero-shot preprocessing.
