"""src/modules/spatial/graph_conv.py

Graph convolution layer.

Provides the `GraphConv` class, which implements message passing
over a batch-wise adjacency matrix. The layer aggregates node features
using edge weights shared across samples and time steps.
"""

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """Graph convolution layer with flexible node aggregation.

    Performs spatial message passing over a graph using a batch-wise adjacency matrix.
    Aggregates node features through weighted sums based on edge weights, allowing
    each sample in the batch to use a different graph structure. This enables
    adaptive per-batch graph topology for spatial feature propagation.
    """

    def __init__(self) -> None:
        """Initialize GraphConv layer."""
        super().__init__()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Compute graph convolution via weighted neighbor aggregation.

        Performs spatial message passing by aggregating node features using
        adjacency weights. Each target node's features are computed as a weighted
        sum of source node features, where weights come from the adjacency matrix.
        This implements the spatial convolution operation for graph neural networks.

        Args:
            x (torch.Tensor):
                Input feature map of shape (b, c, n, l), where:
                    - b: batch size
                    - c: number of channels
                    - n: number of source nodes
                    - l: sequence length

            adj (torch.Tensor):
                Adjacency matrix of shape (n, m) or (b, n, m),
                representing weighted graph connectivity.
                    - b: batch size (optional, enables per-sample graphs)
                    - n: number of source nodes
                    - m: number of target nodes

        Returns:
            torch.Tensor:
                Output feature map of shape (b, c, m, l), where:
                    - b: batch size (same as input)
                    - c: number of channels (same as input)
                    - m: number of target nodes
                    - l: sequence length (same as input)
        """
        # Normalize adjacency to batch format (b, n, m) by
        # adding batch dimension if input is 2D static adjacency,
        # enabling unified einsum computation
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)

        # Perform weighted neighbor aggregation: sum over source nodes
        # weighted by adjacency scores, computing target node features as
        # weighted sum of source node features across all channels and time steps
        x = torch.einsum("bcnl,bnm->bcml", x, adj)
        return x.contiguous()
