"""src/models/spatial/graph_conv.py

Graph convolution layer.

Provides the `GraphConv` class, which implements message passing
over a static adjacency matrix. The layer aggregates node features
using fixed edge weights shared across samples and time steps.
"""

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """Graph convolution layer.

    Performs message passing over a static graph using a fixed
    adjacency matrix shared across the batch and time dimension.

    Notes:
        - This layer is stateless (no learnable parameters)
        - The adjacency matrix is expected to be predefined and fixed
    """

    def __init__(self):
        """Initialize GraphConv layer.

        This layer does not contain learnable parameters and
        only defines the forward message passing operation.
        """
        super().__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Compute graph convolution.

        Performs message passing over a static graph using a fixed
        adjacency matrix.

        Args:
            x (torch.Tensor):
                Input feature map of shape (n, c, v, l), where:
                    - n: batch size
                    - c: number of channels
                    - v: number of nodes
                    - l: sequence length
            A (torch.Tensor):
                Adjacency matrix of shape (v, w), representing
                fixed graph connectivity, where:
                    - v: number of source nodes
                    - w: number of target nodes

        Returns:
            torch.Tensor:
                Output feature map of shape (n, c, w, l), where:
                    - w: number of output nodes
        """
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()
