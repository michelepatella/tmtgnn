"""src/config/tmtgnn_config.py

TMTGNN configuration.

Provides the `TMTGNNConfig` class, which implements configuration
for the TMTGNN model.
"""

from dataclasses import dataclass


@dataclass
class TMTGNNConfig:
    """Configuration for TMTGNN model.

    Attributes:
        hidden_dim (int):
            Hidden feature dimension used throughout the model.
            Default is 32.
        skip_dim (int):
            Skip connection feature dimension used for multi-layer aggregation.
            Default is 64.
        head_dim (int):
            Head dimension before output projection in the final layers.
            Default is 128.
        num_layers (int):
            Number of spatio-temporal blocks in the model.
            Default is 3.
        dropout (float):
            Dropout rate applied to the model's layers.
            Default is 0.
    """

    hidden_dim: int = 32
    skip_dim: int = 64
    head_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3

    def __post_init__(self) -> None:
        """Validates the configuration parameters after initialization."""
        assert isinstance(self.hidden_dim, int), "hidden_dim must be int"
        assert self.hidden_dim > 0, "hidden_dim must be > 0"

        assert isinstance(self.skip_dim, int), "skip_dim must be int"
        assert self.skip_dim > 0, "skip_dim must be > 0"
        assert self.skip_dim >= self.hidden_dim, "skip_dim should be >= hidden_dim"

        assert isinstance(self.head_dim, int), "head_dim must be int"
        assert self.head_dim > 0, "head_dim must be > 0"
        assert self.head_dim >= self.hidden_dim, "head_dim should be >= hidden_dim"

        assert isinstance(self.num_layers, int), "num_layers must be int"
        assert self.num_layers > 0, "num_layers must be > 0"

        assert isinstance(self.dropout, float), "dropout must be float"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0.0, 1.0]"
