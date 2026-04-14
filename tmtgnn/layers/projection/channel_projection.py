"""tmtgnn/layers/projection/channel_projection.py

Channel projection layer.

Provides the `ChannelProjection` class, which applies a learnable
1x1 convolution to project feature channels into a target space.
The layer operates independently on each node and time step.
"""

import torch
import torch.nn as nn


class ChannelProjection(nn.Module):
    """Channel projection layer with learnable 1x1 convolution.

    Applies a pointwise 1x1 convolution (depthwise fully connected layer)
    over the channel dimension to map input features into a target feature space.
    The projection operates independently on each spatial location (node and time step),
    enabling flexible feature dimension transformation.

    Attributes:
        projection (nn.Conv2d):
            1x1 convolution module for channel-wise feature transformation.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool) -> None:
        """Initialize ChannelProjection layer.

        Args:
            in_channels (int):
                Number of input feature channels to project from.
            out_channels (int):
                Number of output feature channels to project to.
            bias (bool):
                Whether to include a learnable bias term.
        """
        super().__init__()

        # Create 1x1 convolution for pointwise channel projection
        # (kernel_size=1 means no spatial mixing, only channel transformation)
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel projection via 1x1 convolution.

        Applies a learnable 1x1 convolution to transform input features
        from input channel space to output channel space. The transformation
        is applied uniformly across all spatial locations (nodes and time steps).

        Args:
            x (torch.Tensor):
                Input feature map of shape (b, c, n, l), where:
                    - b: batch size
                    - c: number of input channels
                    - n: number of nodes
                    - l: sequence length

        Returns:
            torch.Tensor:
                Output feature map of shape (b, c_out, n, l), where:
                    - b: batch size (same as input)
                    - c_out: number of output channels
                    - n: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        # Apply 1x1 convolution for pointwise channel transformation,
        # preserving spatial structure (nodes and time steps) while
        # transforming feature dimensionality
        return self.projection(x)
