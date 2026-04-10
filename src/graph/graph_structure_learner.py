"""src/graph/graph_structure_learner.py

Graph structure learning module.

Provides the `GraphStructureLearner` class, which learns an
adjacency matrix from node embeddings. The learned graph
is sparsified using top-k selection to enforce locality
and reduce noise.
"""

import torch
import torch.nn as nn


class GraphStructureLearner(nn.Module):
    """Graph structure learning module.

    Learns a directed adjacency matrix from node embeddings,
    producing a sparse graph via top-k selection, enabling
    adaptive structure construction.

    Notes:
        - Operates on external node representations provided at runtime
        - Produces asymmetric adjacency (directional structure)
        - Uses top-k sparsification for stability and efficiency
    """

    def __init__(
        self,
        top_k: int,
        hidden_dim: int,
        sigmoid_alpha: float,
        noise_scale: float
    ) -> None:
        """Initialize GraphStructureLearner.

        Args:
            top_k (int):
                Number of outgoing edges per node (top-k sparsification).
            hidden_dim (int):
                Embedding dimension.
            sigmoid_alpha (float):
                Scaling factor for sigmoid non-linearity sharpness.
            noise_scale (float):
                Scale of random noise added to adjacency scores for stability.
        """
        super().__init__()

        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.sigmoid_alpha = sigmoid_alpha
        self.noise_scale = noise_scale

        self.src_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.dst_encoder = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_repr: torch.Tensor) -> torch.Tensor:
        """Compute sparse learned adjacency matrix.

        Learns a directed sparse adjacency matrix from node representations
        using asymmetric similarity scoring, non-linear transformation, and
        stochastic top-k sparsification. The resulting adjacency encodes
        learned graph structure used for downstream message passing.

        Args:
            node_repr (torch.Tensor):
                Node representation tensor of shape (n, hidden_dim), where:
                    - n: number of nodes in the graph
                    - hidden_dim: embedding dimension

                This representation can be:
                    - Learned node embeddings
                    - Projected external node features
                    - Combination of both

        Returns:
            torch.Tensor:
                Sparse learned adjacency matrix of shape (n, n), where:
                    - n: number of nodes in the graph

                The matrix is:
                    - Asymmetric (directed graph structure)
                    - Sparse (enforced by top-k selection)
                    - Non-negative (after sigmoid activation)
                    - Stochastic during training (due to noise injection)
        """
        node_src = torch.tanh(self.src_encoder(node_repr))
        node_dst = torch.tanh(self.dst_encoder(node_repr))

        score = torch.mm(node_src, node_dst.t()) - torch.mm(node_dst, node_src.t())

        adj = torch.sigmoid(self.sigmoid_alpha * score)

        if self.training and self.noise_scale > 0.0:
            adj_for_topk = adj + torch.rand_like(adj) * self.noise_scale
        else:
            adj_for_topk = adj

        k = min(self.top_k, adj_for_topk.size(1))
        _, top_idx = adj_for_topk.topk(k, dim=1)

        mask = torch.zeros_like(adj).scatter(1, top_idx, 1.0)

        return (adj * mask).contiguous()
