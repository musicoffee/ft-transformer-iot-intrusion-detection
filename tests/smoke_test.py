"""Minimal data-free forward-pass test for the main FT-Transformer."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed. Select the project interpreter and install "
        "the appropriate PyTorch build before running this smoke test."
    ) from exc

from models.ft_transformer import FTTransformer


def main() -> None:
    torch.manual_seed(42)
    model = FTTransformer(
        num_features=39,
        num_classes=2,
        d_token=64,
        n_heads=8,
        n_layers=4,
        dim_feedforward=128,
        dropout=0.1,
    )
    model.eval()

    inputs = torch.randn(4, 39)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (4, 2), f"Unexpected output shape: {outputs.shape}"
    assert torch.isfinite(outputs).all().item(), "Model output contains NaN or Inf."
    print("Smoke test passed: finite logits with shape (4, 2).")


if __name__ == "__main__":
    main()
