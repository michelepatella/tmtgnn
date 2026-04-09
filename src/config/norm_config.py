from dataclasses import dataclass


@dataclass
class NormConfig:
    """Configuration for normalization layers.

    Attributes:
        eps (float):
            Numerical stability epsilon used in the normalization layer.
            Default is 1e-5.
        affine (bool):
            Whether the normalization layer uses learnable affine parameters.
            Default is True.
    """

    eps: float = 1e-5
    affine: bool = True
