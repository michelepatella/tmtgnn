"""src/config/norm_config.py

Normalization configuration.

Provides the `NormConfig` class, which implements configuration
for normalization layers.
"""

from dataclasses import dataclass


@dataclass
class NormConfig:
    """Configuration for normalization layers.

    Attributes:
        eps (float):
            Numerical stability epsilon used in the normalization layer.
            Default is 1e-5.
        elementwise_affine (bool):
            Whether the normalization layer uses learnable affine parameters.
            Default is True.
    """

    eps: float = 1e-5
    elementwise_affine: bool = True

    def __post_init__(self) -> None:
        """Validates the configuration parameters after initialization."""
        try:
            assert isinstance(self.eps, float), "eps must be float"
            assert isinstance(self.elementwise_affine, bool), "elementwise_affine must be bool"
        except AssertionError as e:
            raise TypeError(f"Invalid NormConfig parameter: {e}")

        try:
            assert 0.0 < self.eps < 1.0, "eps must be in (0, 1)"
        except AssertionError as e:
            raise ValueError(f"Invalid NormConfig parameter: {e}")
