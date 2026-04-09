import torch
from dataclasses import dataclass


@dataclass
class GraphConfig:
    """Configuration for graph structure learning.

    Attributes:
        subgraph_size (int):
            Number of outgoing edges per node in learned graph.
            Default is 20.
        node_dim (int):
            Node embedding dimension used in graph learning.
            Default is 40.
        alpha (float):
            Scaling factor for graph learning non-linearity.
            Default is 3.0.
        noise_scale (float):
            Noise scale used during adjacency construction.
            Default is 0.01.
        node_features (torch.Tensor | None):
            External node feature matrix used to condition graph
            construction. Default is None.
    """

    subgraph_size: int = 20
    node_dim: int = 40
    alpha: float = 3.0
    noise_scale: float = 0.01
    node_features: torch.Tensor | None = None
