"""src/graph/graph_structure_learner.py

Graph structure learning module.

Provides the `GraphStructureLearner` class, which learns a
task-adaptive adjacency matrix from node embeddings or static
node features. The learned graph is sparsified using top-k
selection to enforce locality and reduce noise.
"""

import torch
import torch.nn as nn


class GraphStructureLearner(nn.Module):
    """Graph structure learning module.

    Learns a directed adjacency matrix from node embeddings or
    static node features, producing a sparse graph via top-k
    selection, enabling adaptive structure construction.

    Notes:
        - Supports learnable node embeddings or external features
        - Produces asymmetric adjacency (directional structure)
        - Uses top-k sparsification for stability and efficiency
    """

    def __init__(
        self,
        num_nodes: int,
        top_k: int,
        hidden_dim: int,
        alpha: float = 3.0,
        noise_scale: float = 0.01,
        node_features: torch.Tensor | None = None,
    ) -> None:
        """Initialize GraphStructureLearner.

        Args:
            num_nodes (int):
                Number of nodes in the graph.
            top_k (int):
                Number of outgoing edges per node (top-k sparsification).
            hidden_dim (int):
                Embedding dimension.
            alpha (float):
                Scaling factor for non-linearity sharpness. Default is 3.0.
            noise_scale (float):
                Scale of random noise added to adjacency scores for stability. 
                Default is 0.01.
            node_features (torch.Tensor | None):
                Precomputed node features of shape (num_nodes, feature_dim).
                Default is None.
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.node_features = node_features

        if node_features is not None:
            feature_dim = node_features.shape[1]

            self.src_encoder = nn.Linear(feature_dim, hidden_dim)
            self.dst_encoder = nn.Linear(feature_dim, hidden_dim)
        else:
            self.src_embedding = nn.Embedding(num_nodes, hidden_dim)
            self.dst_embedding = nn.Embedding(num_nodes, hidden_dim)

            self.src_encoder = nn.Linear(hidden_dim, hidden_dim)
            self.dst_encoder = nn.Linear(hidden_dim, hidden_dim)

    def _encode_nodes(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute node representations for graph structure learning.

        Produces two distinct node embeddings used to construct an
        asymmetric adjacency matrix. The two representations are
        parameterized separately to enable directed edge modeling.

        Args:
            idx (torch.Tensor): 
                Node index tensor of shape (n,), where n is the number
                of selected nodes used to build the local graph structure.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Tuple containing:
                    - node_src (torch.Tensor): 
                        Source node representations of shape (n, hidden_dim)
                    - node_dst (torch.Tensor): 
                        Destination node representations of shape (n, hidden_dim)
        """
        if self.node_features is None:
            node_src = self.src_embedding(idx)
            node_dst = self.dst_embedding(idx)
        else:
            node_src = self.node_features[idx]
            node_dst = node_src

        node_src = torch.tanh(self.alpha * self.src_encoder(node_src))
        node_dst = torch.tanh(self.alpha * self.dst_encoder(node_dst))

        return node_src, node_dst

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute sparse learned adjacency matrix.

        Learns a directed sparse adjacency matrix from node representations
        using asymmetric similarity scoring, non-linear transformation, and
        stochastic top-k sparsification. The resulting adjacency encodes
        learned graph structure used for downstream message passing.

        Args:
            idx (torch.Tensor):
                Node index tensor of shape (n,), where:
                    - n: number of nodes in the selected subgraph used
                         for adjacency construction

        Returns:
            torch.Tensor:
                Sparse learned adjacency matrix of shape (n, n), where:
                    - n: number of nodes in the selected subgraph

                The matrix is:
                    - Asymmetric (directed graph structure)
                    - Sparse (enforced by top-k selection)
                    - Non-negative (after ReLU activation)
                    - Stochastic during training (due to noise injection)
        """
        node_src, node_dst = self._encode_nodes(idx)

        score = torch.mm(node_src, node_dst.t()) - torch.mm(node_dst, node_src.t())

        adj = torch.relu(torch.tanh(self.alpha * score))

        noise = torch.rand_like(adj) * self.noise_scale
        adj_noisy = adj + noise

        _, top_idx = adj_noisy.topk(self.top_k, dim=1)

        mask = torch.zeros_like(adj)
        mask.scatter_(1, top_idx, 1.0)

        return (adj * mask).contiguous()
