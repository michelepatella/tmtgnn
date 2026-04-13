"""src/config/graph_config.py

Graph configuration.

Provides the `GraphConfig` class, which implements configuration
for graph structure learning.
"""

from dataclasses import dataclass


@dataclass
class GraphConfig:
    """Configuration for graph structure learning.

    Attributes:
        top_k (int):
            Number of outgoing edges per node in learned graph.
            Default is 20.
        sigmoid_alpha (float):
            Scaling factor for sigmoid non-linearity sharpness.
            Default is 3.0.
        noise_scale (float):
            Noise scale used during adjacency construction.
            Default is 0.01.
        ema_alpha (float):
            Exponential moving average factor.
            Default is 0.8.
    """

    top_k: int = 20
    sigmoid_alpha: float = 3.0
    noise_scale: float = 0.01
    ema_alpha: float = 0.8

    def __post_init__(self) -> None:
        """Validates the configuration parameters after initialization."""
        try:
            assert isinstance(self.top_k, int), "top_k must be int"
            assert isinstance(self.sigmoid_alpha, float), "sigmoid_alpha must be float"
            assert isinstance(self.noise_scale, float), "noise_scale must be float"
            assert isinstance(self.ema_alpha, float), "ema_alpha must be float"
        except AssertionError as e:
            raise TypeError(f"Invalid GraphConfig parameter: {e}")

        try:
            assert self.top_k > 0, "top_k must be > 0"
            assert self.sigmoid_alpha > 0.0, "sigmoid_alpha must be > 0"
            assert self.noise_scale >= 0.0, "noise_scale must be >= 0"
            assert 0.0 <= self.ema_alpha <= 1.0, "ema_alpha must be in [0, 1]"
        except AssertionError as e:
            raise ValueError(f"Invalid GraphConfig parameter: {e}")
