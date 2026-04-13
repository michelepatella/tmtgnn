"""src/config/diffusion_config.py

Diffusion configuration.

Provides the `DiffusionConfig` class, which implements configuration
for graph diffusion layers.
"""

from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for graph diffusion layers.

    Attributes:
        diffusion_steps (int):
            Number of diffusion steps in graph diffusion layers.
            Default is 2.
        residual_alpha (float):
            Residual propagation coefficient.
            Default is 0.05.
        projection_bias (bool):
            Whether the projection layers inside graph diffusion use bias terms.
            Default is True.
    """

    diffusion_steps: int = 2
    residual_alpha: float = 0.05
    projection_bias: bool = True

    def __post_init__(self) -> None:
        """Validates the configuration parameters after initialization."""
        try:
            assert isinstance(self.diffusion_steps, int), "diffusion_steps must be int"
            assert isinstance(self.residual_alpha, float), (
                "residual_alpha must be float"
            )
            assert isinstance(self.projection_bias, bool), (
                "projection_bias must be bool"
            )
        except AssertionError as e:
            raise TypeError(f"Invalid DiffusionConfig parameter: {e}")

        try:
            assert self.diffusion_steps > 0, "diffusion_steps must be > 0"
            assert 0.0 <= self.residual_alpha <= 1.0, "residual_alpha must be in [0, 1]"
        except AssertionError as e:
            raise ValueError(f"Invalid DiffusionConfig parameter: {e}")
