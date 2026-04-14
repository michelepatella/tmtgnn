"""tmtgnn/modules/temporal/transformer.py

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
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """Transformer module for temporal self-attention modeling.

    Applies a Transformer encoder over the temporal dimension of each node
    independently, replacing classical temporal convolution in spatio-temporal
    graph neural networks. Enables long-range temporal dependency modeling through
    multi-head self-attention.

    Attributes:
        out_channels (int):
            Number of output feature channels.
        projection (nn.Module):
            Optional learnable linear projection to transform input channel
            dimension to output dimension. Identity if dimensions match.
        positional_encoding (PositionalEncoding):
            Sinusoidal positional encoding to inject temporal order information.
        transformer (nn.TransformerEncoder):
            Stacked Transformer encoder layers for temporal self-attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_sequence_length: int,
    ) -> None:
        """Initialize Transformer.

        Args:
            in_channels (int):
                Number of input feature channels before temporal modeling.
            out_channels (int):
                Number of output feature channels after projection.
            num_heads (int):
                Number of attention heads in the Transformer encoder,
                controlling parallel attention subspace information.
            num_layers (int):
                Number of stacked Transformer encoder layers, controlling
                depth and receptive field over temporal dimension.
            dropout (float):
                Dropout rate applied inside Transformer layers for regularization.
            max_sequence_length (int):
                Maximum temporal sequence length to precompute positional encodings.
        """
        super().__init__()

        self.out_channels = out_channels

        # Create optional learnable projection to transform input channels
        # to output channels (if dimensions match, use identity to avoid
        # unnecessary computation)
        self.projection = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Create positional encoding for temporal order information,
        # enabling model to distinguish position in temporal sequences
        self.positional_encoding = PositionalEncoding(
            hidden_dim=out_channels,
            max_sequence_length=max_sequence_length,
        )

        # Create single encoder layer with specified configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Stack multiple encoder layers to build hierarchical temporal
        # representations for multivariate temporal modeling across
        # multiple feature channels
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute temporal representation via multi-head self-attention.

        Applies Transformer encoder over the temporal dimension for each node
        independently. Reshapes input so each node is treated as a separate
        temporal sequence, applies projection and positional encoding, then
        feeds through stacked Transformer layers for self-attention computation.

        Args:
            x (torch.Tensor):
                Input feature map of shape (b, c, n, l), where:
                    - b: batch size
                    - c: number of input channels
                    - n: number of nodes
                    - l: sequence length (temporal dimension)

        Returns:
            torch.Tensor:
                Output feature map of shape (b, c_out, n, l), where:
                    - b: batch size (same as input)
                    - c_out: number of output channels
                    - n: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        # Extract dimensions
        batch, channels, num_nodes, seq_length = x.shape

        # Reshape for per-node temporal processing (this treats each node
        # independently as a separate sequence for self-attention)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch * num_nodes, seq_length, channels)

        # Project input channels to output dimensionality
        # for consistent feature space
        x = self.projection(x)

        # Add sinusoidal positional encoding to inject temporal
        # position information, enabling model to distinguish and
        # reason about position in temporal sequences
        x = self.positional_encoding(x)

        # Apply stacked Transformer encoder layers for multi-head self-attention,
        # learning temporal dependencies and patterns across the sequence
        x = self.transformer(x)

        # Reshape back to original node structure and
        # restore original dimension order
        x = x.view(batch, num_nodes, seq_length, self.out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x
