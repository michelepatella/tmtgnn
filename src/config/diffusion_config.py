from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for graph diffusion layers.

    Attributes:
        gcn_depth (int):
            Number of diffusion steps in graph diffusion layers.
            Default is 2.
        residual_alpha (float):
            Residual propagation coefficient.
            Default is 0.05.
        projection_bias (bool):
            Whether the projection layers inside graph diffusion use bias terms.
            Default is True.
    """

    gcn_depth: int = 2
    residual_alpha: float = 0.05
    projection_bias: bool = True
