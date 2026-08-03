from __future__ import annotations

import torch
import torch.nn as nn


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))
        self.feature_embedding = nn.Parameter(torch.randn(num_features, d_token) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        tokens = x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0) + self.feature_embedding.unsqueeze(0)
        return tokens


class FTTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_token: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.tokenizer = NumericalFeatureTokenizer(num_features=num_features, d_token=d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, num_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.encoder(x)
        cls_out = x[:, 0, :]
        cls_out = self.norm(cls_out)
        return self.head(cls_out)


if __name__ == "__main__":
    model = FTTransformer(
        num_features=38,
        num_classes=2,
        d_token=64,
        n_heads=8,
        n_layers=4,
        dim_feedforward=128,
        dropout=0.1,
    )
    x = torch.randn(8, 38)
    y = model(x)
    print("输出 shape:", y.shape)