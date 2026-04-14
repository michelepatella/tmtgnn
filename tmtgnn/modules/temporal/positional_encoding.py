"""tmtgnn/modules/temporal/positional_encoding.py

Positional encoding for temporal sequences.

Provides the `PositionalEncoding` class, which implements sinusoidal
positional encoding to provide temporal order information to temporal
models like Transformers.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences with sinusoidal embeddings.

    Implements sinusoidal positional encoding to inject temporal order
    information into time series. Each position in the sequence gets a unique
    encoding based on sine and cosine functions at different frequencies,
    enabling models like Transformers to distinguish temporal dependencies
    and learn absolute/relative position-aware representations.

    Attributes:
        pe (torch.Tensor):
            Precomputed sinusoidal positional encoding matrix of shape
            (1, max_sequence_length, hidden_dim).
    """

    def __init__(self, hidden_dim: int, max_sequence_length: int) -> None:
        """Initialize PositionalEncoding.

        Args:
            hidden_dim (int):
                Embedding dimension, must match the feature dimensionality
                used in subsequent layers.
            max_sequence_length (int):
                Maximum sequence length to precompute positional encodings for.
                Sequences longer than max_sequence_length will truncate encodings.
        """
        super().__init__()

        # Create position encoding matrix
        pe = torch.zeros(max_sequence_length, hidden_dim)

        # Position indices for broadcasting with frequency terms
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)

        # Exponentially scaled frequency terms for each dimension pair,
        # using base 10000 to spread frequencies across logarithmic range
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * -(math.log(10000.0) / hidden_dim)
        )

        # Apply sine function to even-indexed dimensions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine function to odd-indexed dimensions
        # handling odd hidden_dim case by using one fewer frequency term for cosine
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with model but doesn't
        # participate in gradient computation or optimization
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to temporal sequence features.

        Adds precomputed sinusoidal positional encodings to input features,
        injecting temporal position information while preserving the original
        feature values through addition. The encoding provides models with
        awareness of position within the sequence.

        Args:
            x (torch.Tensor):
                Input tensor of shape (b, l, d), where:
                    - b: batch size
                    - l: sequence length
                    - d: feature dimension (must equal hidden_dim from __init__)

        Returns:
            torch.Tensor:
                Input with positional encoding added, shape (b, l, d).
                Each position gets a unique sinusoidal signature based on its
                index in the sequence, enabling models to learn position-aware patterns.
        """
        # Add precomputed positional encodings to input features,
        # broadcasting across batch dimension and extracting only needed sequence length
        return x + self.pe[:, : x.size(1), :]
