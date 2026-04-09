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

    def __post_init__(self) -> None:
        """Validates the configuration parameters after initialization."""
        assert isinstance(self.subgraph_size, int), "subgraph_size must be int"
        assert self.subgraph_size > 0, "subgraph_size must be > 0"

        assert isinstance(self.node_dim, int), "node_dim must be int"
        assert self.node_dim > 0, "node_dim must be > 0"

        assert isinstance(self.alpha, float), "alpha must be float"
        assert self.alpha > 0.0, "alpha must be > 0"

        assert isinstance(self.noise_scale, float), "noise_scale must be float"
        assert self.noise_scale >= 0.0, "noise_scale must be >= 0"

        if self.node_features is not None:
            assert isinstance(self.node_features, torch.Tensor), (
                "node_features must be torch.Tensor"
            )
            assert self.node_features.dim() == 2, (
                "node_features must be [num_nodes, feat_dim]"
            )
