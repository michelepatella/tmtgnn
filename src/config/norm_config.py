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

    def __post_init__(self) -> None:
        """Validates the configuration parameters after initialization."""
        assert isinstance(self.eps, float), "eps must be float"
        assert self.eps > 0.0, "eps must be > 0"
        assert self.eps < 1.0, "eps should be < 1.0"

        assert isinstance(self.affine, bool), "affine must be bool"
