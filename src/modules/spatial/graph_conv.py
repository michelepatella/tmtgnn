"""src/modules/spatial/graph_conv.py

Graph convolution layer.

Provides the `GraphConv` class, which implements message passing
over a batch-wise adjacency matrix. The layer aggregates node features
using edge weights shared across samples and time steps.
"""

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """Graph convolution layer.

    Performs message passing over a graph using a batch-wise adjacency matrix,
    allowing each sample in the batch to use a different graph structure.

    Notes:
        - This layer is stateless (no learnable parameters)
        - The adjacency matrix can be either static (N, N) or batch-specific (B, N, N).
    """

    def __init__(self) -> None:
        """Initialize GraphConv layer.

        This layer does not contain learnable parameters and
        only defines the forward message passing operation.
        """
        super().__init__()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Compute graph convolution.

        Performs message passing over a graph using a batch-wise
        adjacency matrix.

        Args:
            x (torch.Tensor):
                Input feature map of shape (b, c, v, l), where:
                    - b: batch size
                    - c: number of channels
                    - v: number of source nodes
                    - l: sequence length

            adj (torch.Tensor):
                Adjacency matrix of shape (v, w) or (b, v, w),
                representing graph connectivity.
                    - b: batch size
                    - v: number of source nodes
                    - w: number of target nodes

        Returns:
            torch.Tensor:
                Output feature map of shape (b, c, w, l), where:
                    - b: batch size (same as input)
                    - c: number of channels (same as input)
                    - w: number of target nodes
                    - l: sequence length (same as input)
        """
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)

        x = torch.einsum("bcvl,bvw->bcwl", x, adj)
        return x.contiguous()
