from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for Transformer-based temporal modeling.

    Attributes:
        num_heads (int):
            Number of attention heads used in Transformer layers.
            Default is 4.
        num_layers (int):
            Number of internal layers inside each Transformer block.
            Default is 2.
        dropout (float):
            Dropout rate used in Transformer layers.
            Default is 0.3.
    """

    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.3

    def __post_init__(self) -> None:
        """Validates the configuration parameters after initialization."""
        assert isinstance(self.num_heads, int), "num_heads must be int"
        assert self.num_heads > 0, "num_heads must be > 0"

        assert isinstance(self.num_layers, int), "num_layers must be int"
        assert self.num_layers > 0, "num_layers must be > 0"

        assert isinstance(self.dropout, float), "dropout must be float"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"
