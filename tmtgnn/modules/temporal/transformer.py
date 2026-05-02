"""tmtgnn/modules/temporal/transformer.py

Transformer module.

Provides the `Transformer` class, which enables multi-mode temporal/node-level
self-attention for spatio-temporal graph neural networks. Supports:
- Temporal attention: Along time dimension with causal masking
- Node attention: Along node dimension
"""

import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """Flexible Transformer module for multi-mode self-attention.

    Applies multi-head self-attention with positional encoding, supporting:
    1. Temporal mode: Self-attention over time dimension per node (with causal masking)
    2. Node mode: Self-attention over node dimension
    Enables rich temporal/node-level dependency modeling without convolutional
    inductive bias, applicable to diverse graph structures.

    Attributes:
        out_channels (int):
            Number of output feature channels.
        projection (nn.Module):
            Optional learnable linear projection to transform input channel
            dimension to output dimension. Identity if dimensions match.
        positional_encoding (PositionalEncoding):
            Sinusoidal positional encoding for temporal or node position info.
        transformer (nn.TransformerEncoder):
            Stacked Transformer encoder layers for self-attention.
        mode (str):
            Active attention mode ("temporal" or "node").
        max_sequence_length (int):
            Maximum sequence length for positional encoding and causal mask.
        causal_mask (torch.Tensor):
            Precomputed causal mask for temporal mode to prevent attention to future tokens.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_sequence_length: int,
        mode: str,
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
                depth and receptive field over attention dimension.
            dropout (float):
                Dropout rate applied inside Transformer layers for regularization.
            max_sequence_length (int):
                Maximum sequence length for positional encoding and causal mask.
            mode (str):
                Attention mode:
                - "temporal": Self-attention over time dimension per node (causal)
                - "node": Self-attention over node dimension
        """
        super().__init__()

        self.out_channels = out_channels
        self.mode = mode
        self.max_sequence_length = max_sequence_length

        # Create optional learnable projection to transform input channels
        # to output channels (if dimensions match, use identity to avoid
        # unnecessary computation)
        self.projection = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Create positional encoding for temporal order information,
        # enabling model to distinguish position in sequences or node graphs
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

        # Stack multiple encoder layers to build hierarchical representations
        # for multivariate temporal/spatial modeling across multiple scales
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Precompute causal mask for temporal mode to prevent attention to future positions
        self._register_causal_mask(max_sequence_length)

    def _register_causal_mask(self, seq_length: int) -> None:
        """Precompute and register causal mask for temporal attention.

        Creates a lower triangular mask that prevents attention to future positions.
        The mask is registered as a buffer so it moves with the model and doesn't
        participate in gradient computation.

        Args:
            seq_length (int):
                Maximum sequence length for which to precompute the mask.
        """
        # Create lower triangular matrix: True where attention is allowed
        # Position i can attend to positions 0...i (all positions <= i)
        causal_mask = torch.tril(torch.ones(seq_length, seq_length)) == 1

        # Convert to PyTorch's attention mask format: True where to mask (block attention)
        # Invert so that True means "block this position"
        causal_mask = ~causal_mask

        # Register as buffer (moves with model, no gradients)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute representation via multi-head self-attention.

        Applies Transformer encoder with explicitly configured attention mode:
        - Temporal mode: Self-attention per node over time dimension with causal masking
        - Node mode: Self-attention over node dimension

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
        batch, in_channels, num_nodes, seq_length = x.shape

        if self.mode == "temporal":
            return self._forward_temporal(x, batch, in_channels, num_nodes, seq_length)
        else:
            return self._forward_node(x, batch, in_channels, num_nodes, seq_length)

    def _forward_temporal(
        self,
        x: torch.Tensor,
        batch: int,
        in_channels: int,
        num_nodes: int,
        seq_length: int,
    ) -> torch.Tensor:
        """Forward pass with temporal mode: self-attention over time per node (causal).

        Classical multi-node temporal attention approach with causal masking. Treats each
        node independently and applies Transformer self-attention over the time dimension
        with causal masking to ensure each position can only attend to the past.
        Each node's temporal sequence is processed separately, enabling the model to learn
        node-specific temporal patterns without direct cross-node temporal interaction or
        information leakage from future timesteps.

        Args:
            x (torch.Tensor):
                Input feature tensor of shape (b, c, n, l), where:
                    - b: batch size
                    - c: number of input channels
                    - n: number of nodes
                    - l: sequence length (temporal dimension)
            batch (int):
                Batch size extracted from input shape
            in_channels (int):
                Number of input feature channels.
            num_nodes (int):
                Number of nodes in the graph.
            seq_length (int):
                Length of temporal sequence.

        Returns:
            torch.Tensor:
                Output feature tensor of shape (b, c_out, n, l), where:
                    - b: batch size (same as input)
                    - c_out: number of output channels
                    - n: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        # Reshape for per-node temporal processing
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch * num_nodes, seq_length, in_channels)

        # Project to output dimension
        x = self.projection(x)

        # Add positional encoding for temporal positions
        x = self.positional_encoding(x)

        # Extract causal mask for current sequence length
        # Mask shape: (seq_length, seq_length)
        mask = self.causal_mask[:seq_length, :seq_length]

        # Apply Transformer over temporal dimension with causal masking
        # Causal mask ensures position t can only attend to positions 0...t
        x = self.transformer(x, src_mask=mask)

        # Reshape back to (b, c_out, n, l)
        x = x.view(batch, num_nodes, seq_length, self.out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

    def _forward_node(
        self,
        x: torch.Tensor,
        batch: int,
        in_channels: int,
        num_nodes: int,
        seq_length: int,
    ) -> torch.Tensor:
        """Forward pass with node mode: self-attention over nodes per timestep.

        Spatial attention approach for large spatio-temporal graphs. Treats each
        timestep independently and applies Transformer self-attention over the node
        dimension. Each timestep's node features are processed separately, enabling the
        model to learn global spatial correlations and cross-node dependencies at each
        temporal step.

        Args:
            x (torch.Tensor):
                Input feature tensor of shape (b, c, n, l), where:
                    - b: batch size
                    - c: number of input channels
                    - n: number of nodes
                    - l: sequence length (temporal dimension)
            batch (int):
                Batch size extracted from input shape.
            in_channels (int):
                Number of input feature channels.
            num_nodes (int):
                Number of nodes in the graph.
            seq_length (int):
                Length of temporal sequence.

        Returns:
            torch.Tensor:
                Output feature tensor of shape (b, c_out, n, l), where:
                    - b: batch size (same as input)
                    - c_out: number of output channels
                    - n: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        # Reshape for node-level processing
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch * seq_length, num_nodes, in_channels)

        # Project to output dimension
        x = self.projection(x)

        # Add positional encoding for node positions
        x = self.positional_encoding(x)

        # Apply Transformer over node dimension (no causal mask needed for spatial)
        x = self.transformer(x)

        # Reshape back to (b, c_out, n, l)
        x = x.view(batch, seq_length, num_nodes, self.out_channels)
        x = x.permute(0, 3, 2, 1).contiguous()

        return x
