"""src/layers/normalization/layer_norm.py

Layer normalization module.

Provides the `LayerNorm` class, which applies layer normalization
while supporting elementwise affine transformation and conditioning 
the affine parameters on node indices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers


class LayerNorm(nn.Module):
    """Layer normalization with node-aware parameter selection.

    Applies layer normalization over input features, with optional
    node-specific affine parameters. Instead of using global weights
    and biases, this implementation conditions the affine transformation
    on individual node indices, enabling per-node normalization behavior.

    Attributes:
        eps (float): 
            Small value added for numerical stability during normalization.
        elementwise_affine (bool): 
            Whether to use learnable per-element affine parameters.
        weight (nn.Parameter): 
            Learnable weight parameters for affine transformation, indexed by node.
        bias (nn.Parameter): 
            Learnable bias parameters for affine transformation, indexed by node.
    """

    def __init__(
        self,
        normalized_shape: int | tuple,
        eps: float,
        elementwise_affine: bool
    ) -> None:
        """Initialize LayerNorm module.

        Args:
            normalized_shape (int | tuple):
                Shape of the normalized dimension(s). Can be a single
                integer or tuple specifying which dimensions to normalize.
            eps (float):
                Small epsilon value added for numerical stability
                during normalization.
            elementwise_affine (bool):
                Whether to use learnable per-element affine parameters.
        """
        super().__init__()

        # Normalize shape to tuple format for consistent 
        # parameter initialization
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # Create learnable weight and bias parameters for affine 
            # transformation (these will be indexed per node in the forward pass)
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))

            # Initialize weight to ones and bias to zeros for identity mapping 
            # at start, allowing the model to learn offsets from this baseline
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            # No affine transformation, register as None parameters
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Compute node-aware layer normalization.

        Applies layer normalization over input features with optional
        node-dependent affine parameters. The key difference from standard
        LayerNorm is that affine parameters (weight and bias) are indexed
        per node, enabling heterogeneous normalization scaling across nodes.

        Args:
            x (torch.Tensor):
                Input tensor of shape (b, c, n, l), where:
                    - b: batch size
                    - c: number of channels
                    - n: number of nodes
                    - l: sequence length

            idx (torch.Tensor):
                Node index tensor of shape (n,), used to select
                node-specific affine parameters for each node's normalization.

        Returns:
            torch.Tensor:
                Normalized tensor of shape (b, c, n, l), where:
                    - b: batch size (same as input)
                    - c: number of channels (same as input)
                    - n: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        if self.elementwise_affine:
            # Index affine parameters for each node to enable per-node
            # normalization scaling and shifting via idx, making different
            # nodes scale and shift features by different amounts
            weight = self.weight[:, idx, :]
            bias = self.bias[:, idx, :]

            # Apply layer normalization with node-specific weight and bias parameters,
            # using eps for numerical stability during normalization
            return F.layer_norm(
                x,
                tuple(x.shape[1:]),
                weight,
                bias,
                self.eps,
            )

        # Apply standard layer normalization without affine transformation,
        # only normalizing the values to zero mean and unit variance
        return F.layer_norm(
            x,
            tuple(x.shape[1:]),
            self.weight,
            self.bias,
            self.eps,
        )
