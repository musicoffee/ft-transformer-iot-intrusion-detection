# Configuration evidence

These JSON files summarize existing experiment settings; they are not a new configuration framework and are not automatically consumed by the historical scripts.

Each file contains an `evidence` array pointing to repository-relative code or artifact paths. Main-model architecture values come from the training code and the inspected saved checkpoint configuration. Dataset sizes and class counts come from archived metadata. Unknown values are represented by `null` instead of being inferred.

- `binary.json`: main CICIoT2023 binary run.
- `multiclass.json`: main unweighted eight-output multiclass run.
- `cross_dataset.json`: 38-feature aligned FT-Transformer run.
- `coral_original.json`: historical, uncorrected CORAL script settings.

The complete original dependency environment was not archived as a lockfile. See `docs/LIMITATIONS.md` before comparing configurations.
