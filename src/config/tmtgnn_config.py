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
            Default is 0.3.
        num_forecast_steps (int):
            Number of future time steps to predict.
            Default is 1.
    """

    hidden_dim: int = 32
    skip_dim: int = 64
    head_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    num_forecast_steps: int = 1

    def __post_init__(self) -> None:
        """Validates the configuration parameters after initialization.
        
        Raises:
            TypeError:
                If any parameter has incorrect type.
            ValueError:
                If any parameter value violates constraints.
        """
        try:
            assert isinstance(self.hidden_dim, int), "hidden_dim must be int"
            assert isinstance(self.skip_dim, int), "skip_dim must be int"
            assert isinstance(self.head_dim, int), "head_dim must be int"
            assert isinstance(self.num_layers, int), "num_layers must be int"
            assert isinstance(self.dropout, float), "dropout must be float"
            assert isinstance(self.num_forecast_steps, int), (
                "num_forecast_steps must be int"
            )
        except AssertionError as e:
            raise TypeError(f"Invalid TMTGNNConfig parameter: {e}")

        try:
            assert self.hidden_dim > 0, "hidden_dim must be > 0"
            assert self.skip_dim > 0, "skip_dim must be > 0"
            assert self.skip_dim >= self.hidden_dim, "skip_dim must be >= hidden_dim"
            assert self.head_dim > 0, "head_dim must be > 0"
            assert self.head_dim >= self.hidden_dim, "head_dim must be >= hidden_dim"
            assert self.num_layers > 0, "num_layers must be > 0"
            assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0.0, 1.0]"
            assert self.num_forecast_steps > 0, "num_forecast_steps must be > 0"
        except AssertionError as e:
            raise ValueError(f"Invalid TMTGNNConfig parameter: {e}")
