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
