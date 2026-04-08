"""src/normalization/layer_norm.py

Node-aware layer normalization module.

Provides the `LayerNorm` class, which applies layer normalization
with optional node-dependent parameter selection. The layer supports
elementwise affine transformation and can condition normalization
on node indices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers


class LayerNorm(nn.Module):
    """Node-aware layer normalization.

    Applies layer normalization over input features, with optional
    node-specific affine parameters selected via an index tensor.

    Notes:
        - This implementation extends standard LayerNorm with
          node-conditioned parameters
        - It is designed for graph-based models with per-node variation
    """

    def __init__(
        self,
        normalized_shape: int | tuple,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        """Initialize LayerNorm module.

        Args:
            normalized_shape (int | tuple):
                Shape of the normalized dimension(s).
            eps (float):
                Small value added for numerical stability. Default is 1e-5.
            elementwise_affine (bool):
                Whether to use learnable affine parameters. Default is True.
        """
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))

            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Compute node-aware layer normalization.

        Applies layer normalization over input features with optional
        node-dependent affine parameters selected via an index tensor.

        Args:
            x (torch.Tensor):
                Input tensor of shape (n, c, v, l), where:
                    - n: batch size
                    - c: number of channels
                    - v: number of nodes
                    - l: sequence length

            idx (torch.Tensor):
                Node index tensor of shape (v,), used to select
                node-specific affine parameters for normalization.

        Returns:
            torch.Tensor:
                Normalized tensor of shape (n, c, v, l), where:
                    - n: batch size (same as input)
                    - c: number of channels (same as input)
                    - v: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        if self.elementwise_affine:
            weight = self.weight[:, idx, :]
            bias = self.bias[:, idx, :]

            return F.layer_norm(
                x,
                tuple(x.shape[1:]),
                weight,
                bias,
                self.eps,
            )

        return F.layer_norm(
            x,
            tuple(x.shape[1:]),
            self.weight,
            self.bias,
            self.eps,
        )
