# Archived data schemas

These files are selected historical preprocessing metadata, not datasets. They contain feature-name lists, label mappings, split shapes/class counts, and the cross-dataset shared-feature list.

| Directory | Contents |
|---|---|
| `ciciot2023_binary/` | 39 feature names, binary label mapping, and 420k/60k/120k split metadata |
| `ciciot2023_multiclass/` | 39 feature names, eight-label mapping, and split metadata |
| `ciciomt2024_binary/` | 45 feature names, binary mapping, and split metadata |
| `cross_dataset/` | 38 shared feature names |

No sample rows, scaler objects, IP addresses, device identifiers, or checkpoints are included. The historical metadata text `7-class` refers to seven attack families; the actual mapping contains Benign as an eighth output class.
