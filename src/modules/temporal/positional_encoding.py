"""src/modules/temporal/positional_encoding.py

Positional encoding for temporal sequences.

Provides the `PositionalEncoding` class, which implements sinusoidal
positional encoding to provide temporal order information to temporal
models like Transformers.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences.

    Implements sinusoidal positional encoding to provide temporal
    order information to the Transformer model, enabling it to
    distinguish between different time steps.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize PositionalEncoding.

        Args:
            d_model (int):
                Dimension of the model (must match Transformer d_model).
            max_len (int):
                Maximum length of sequences. Default is 5000.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor:
                Input with positional encoding added.
        """
        return x + self.pe[:, : x.size(1), :]
