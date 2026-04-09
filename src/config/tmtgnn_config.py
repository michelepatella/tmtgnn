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
    """

    hidden_dim: int = 32
    skip_dim: int = 64
    head_dim: int = 128
    num_layers: int = 3
