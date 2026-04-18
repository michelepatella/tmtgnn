"""tmtgnn/config/transformer_config.py

Transformer configuration.

Provides the `TransformerConfig` class, which implements configuration
for Transformer-based temporal modeling.
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for Transformer-based temporal modeling.

    Attributes:
        mode (str):
            Attention mode for self-attention:
            - "temporal": Self-attention over time dimension per node
            - "node": Self-attention over node dimension
            Default is "temporal".
        num_heads (int):
            Number of attention heads used in Transformer layers.
            Default is 4.
        num_layers (int):
            Number of internal layers inside each Transformer block.
            Default is 2.
        dropout (float):
            Dropout rate used in Transformer layers.
            Default is 0.3.
        max_sequence_length (int):
            Maximum sequence length for positional encoding.
            Default is 5000.
    """

    mode: str = "temporal"
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.3
    max_sequence_length: int = 5000

    def __post_init__(self) -> None:
        """Validates the configuration parameters after initialization.

        Raises:
            TypeError:
                If any parameter has incorrect type.
            ValueError:
                If any parameter value violates constraints.
        """
        try:
            assert isinstance(self.mode, str), "mode must be str"
            assert isinstance(self.num_heads, int), "num_heads must be int"
            assert isinstance(self.num_layers, int), "num_layers must be int"
            assert isinstance(self.dropout, float), "dropout must be float"
            assert isinstance(self.max_sequence_length, int), (
                "max_sequence_length must be int"
            )
        except AssertionError as e:
            raise TypeError(f"Invalid TransformerConfig parameter: {e}")

        try:
            assert self.mode in ["temporal", "node"], (
                "mode must be 'temporal' or 'node'"
            )
            assert self.num_heads > 0, "num_heads must be > 0"
            assert self.num_layers > 0, "num_layers must be > 0"
            assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"
            assert self.max_sequence_length > 0, "max_sequence_length must be > 0"
        except AssertionError as e:
            raise ValueError(f"Invalid TransformerConfig parameter: {e}")
