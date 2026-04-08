"""src/utils/channel_projection.py

Channel projection layer.

Provides the `ChannelProjection` class, which applies a learnable
1x1 convolution to project feature channels into a target space.
The layer operates independently on each node and time step.
"""

import torch
import torch.nn as nn


class ChannelProjection(nn.Module):
    """Channel projection layer.

    Applies a pointwise 1x1 convolution over the channel dimension
    to map input features into a target feature space.

    Notes:
        - This layer is fully learnable
        - It operates independently on each node and time step
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        """Initialize ChannelProjection layer.

        Args:
            in_channels (int):
                Number of input feature channels.
            out_channels (int):
                Number of output feature channels.
            bias (bool):
                Whether to include a learnable bias term. Default is True.
        """
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel projection.

        Applies a learnable 1x1 convolution over the channel dimension
        to project input features into a target feature space.

        Args:
            x (torch.Tensor):
                Input feature map of shape (n, c, v, l), where:
                    - n: batch size
                    - c: number of input channels
                    - v: number of nodes
                    - l: sequence length

        Returns:
            torch.Tensor:
                Output feature map of shape (n, c_out, v, l), where:
                    - n: batch size (same as input)
                    - c_out: number of output channels
                    - v: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        return self.projection(x)
