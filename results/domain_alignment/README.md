# Historical domain-alignment results

`original/` contains the CSV and JSON produced by `analysis/revision_13_domain_aligned_ft_coral.py`. They are archived for transparency and were not rerun or numerically edited during repository organization.

Important constraints:

1. These are historical original results, not a new reproduction.
2. The script reimplements an FT-Transformer-like model inside the file.
3. It does not fully reuse the main `models/ft_transformer.py` architecture or checkpoint.
4. The local model defaults to 4 attention heads, while the main checkpoint uses 8.
5. Its tokenizer lacks the main model's separate learnable feature embedding, and its classification head is organized differently.
6. The CORAL loss takes a mean squared covariance difference and then divides by `4 * d * d`, apparently adding an extra `d²` scaling relative to the common Frobenius formulation.
7. Target labels do not enter gradient computation, but they are used for stratified adaptation/test splitting and sampling.
8. Corrected-CORAL has not been rerun.

Do not present these artifacts as a controlled CORAL extension of the main FT-Transformer or as corrected results.
