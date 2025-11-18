# src/models.py
import torch
import torch.nn as nn
from typing import Tuple


class GraphWeightEstimator(nn.Module):
    def __init__(
        self,
        original_size: Tuple[int, int] = (400, 400),
        embed_dim: int = 8,
        kernel_size: int = 11,
        dilation: int = 1,
        fc_hidden_dims: list = None,
    ):
        super().__init__()
        self.original_size = original_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.embed_dim = embed_dim

        if fc_hidden_dims is None:
            fc_hidden_dims = [128, 64]

        # Effective kernel size accounting for dilation
        effective_kernel = dilation * (kernel_size - 1) + 1
        self.padding = (effective_kernel - 1) // 2

        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=1,
            dilation=self.dilation,
        )

        self.fold = nn.Fold(
            output_size=self.original_size,
            kernel_size=1,
            padding=0,
            stride=1,
        )

        # Build MLP
        input_dim = 3 * kernel_size * kernel_size
        layers = []

        for hidden_dim in fc_hidden_dims:
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.embed_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.unfold(x)  # (B, C*k*k, H*W)
        y = self.fc(x.permute(0, 2, 1))  # (B, H*W, embed_dim)
        y = y.permute(0, 2, 1)  # (B, embed_dim, H*W)
        y = self.fold(y)  # (B, embed_dim, H, W)
        y = y.squeeze(0).permute(1, 2, 0)  # (H, W, embed_dim)
        y = torch.nn.functional.normalize(y, p=2, dim=-1)
        return y