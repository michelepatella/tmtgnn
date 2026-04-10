"""src/modules/temporal/transformer.py

Transformer module.

Provides the `Transformer` class, which replaces classical
temporal convolution in spatio-temporal graph neural networks
with a Transformer-based temporal modeling block. The module
applies self-attention over the temporal dimension for each node
independently, enabling long-range temporal dependency modeling without
explicit convolutional inductive bias.
"""

import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """Transformer module.

    Applies a Transformer encoder over the temporal dimension of each node
    independently. The module replaces classical temporal convolution in
    spatio-temporal graph neural networks. The input is first reshaped so
    that each node is treated as an independent sequence over time. A shared
    Transformer is then applied across all nodes.

    Notes:
        - Temporal order is encoded via sinusoidal positional encoding
        - Temporal dependencies are learned independently per node
        - Positional encoding enables the model to explicitly model temporal relationships
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_head: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_sequence_length: int = 5000,
    ) -> None:
        """Initialize Transformer.

        Args:
            in_channels (int):
                Number of input feature channels.
            out_channels (int):
                Number of output feature channels.
            num_head (int):
                Number of attention heads in the Transformer encoder.
                Default is 4.
            num_layers (int):
                Number of stacked Transformer encoder layers.
                Default is 2.
            dropout (float):
                Dropout rate used inside Transformer layers.
                Default is 0.3.
            max_sequence_length (int):
                Maximum sequence length for positional encoding.
                Default is 5000.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.projection = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.positional_encoding = PositionalEncoding(
            d_model=out_channels,
            max_len=max_sequence_length,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=num_head,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute temporal representation using Transformer encoder.

        Applies self-attention over the temporal dimension for each node
        independently. The input is reshaped so that each node is treated
        as a separate temporal sequence, a projection is applied, positional
        encoding is added, and a shared Transformer encoder is applied across
        all nodes.

        Args:
            x (torch.Tensor):
                Input feature map of shape (b, c, v, l), where:
                    - b: batch size
                    - c: number of input channels
                    - v: number of nodes
                    - l: sequence length (temporal dimension)

        Returns:
            torch.Tensor:
                Output feature map of shape (b, c_out, v, l), where:
                    - b: batch size (same as input)
                    - c_out: number of output channels
                    - v: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        n, c, v, l = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(n * v, l, c)

        x = self.projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)

        x = x.view(n, v, l, self.out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x
