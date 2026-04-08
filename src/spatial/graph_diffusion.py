"""src/models/spatial/graph_diffusion.py

Graph diffusion layer.

Provides the `GraphDiffusion` class, which implements multi-step
message passing over a graph with skip-connected input injection of input features.
The layer aggregates node representations across multiple diffusion
steps and projects them into a target feature space.
"""

import torch
import torch.nn as nn
from spatial.graph_conv import GraphConv
from utils.channel_projection import ChannelProjection


class GraphDiffusion(nn.Module):
    """Graph diffusion layer.

    Performs iterative message passing over a graph using a fixed
    adjacency matrix. At each step, node features are updated via
    a combination of the original input and diffused representations.
    The intermediate representations from all diffusion steps are
    concatenated and projected to the desired output dimensionality.

    Notes:
        - The adjacency matrix is expected to be predefined and fixed
        - Diffusion depth controls the receptive field over the graph
        - Skip-connected input injection helps prevent over-smoothing
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        diffusion_steps: int,
        residual_alpha: float,
    ) -> None:
        """Initialize GraphDiffusion layer.

        Args:
            in_channels (int):
                Number of input feature channels.
            out_channels (int):
                Number of output feature channels.
            diffusion_steps (int):
                Number of diffusion steps (graph propagation depth).
            residual_alpha (float):
                Mixing coefficient between input features and
                diffused features. Must be in [0, 1].
        """
        super().__init__()

        self.graph_conv = GraphConv()
        self.channel_projection = ChannelProjection(
            (diffusion_steps + 1) * in_channels, out_channels
        )
        self.diffusion_steps = diffusion_steps
        self.residual_alpha = residual_alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Compute graph diffusion.

        Performs iterative message passing over a graph with residual
        mixing of the original input features. The hidden states from
        each diffusion step are concatenated and projected to the output space.

        Args:
            x (torch.Tensor):
                Input feature map of shape (n, c, v, l), where:
                    - n: batch size
                    - c: number of channels
                    - v: number of nodes
                    - l: sequence length
            adj (torch.Tensor):
                Adjacency matrix of shape (v, v), representing
                fixed graph connectivity, where:
                    - v: number of source nodes

        Returns:
            torch.Tensor:
                Output feature map of shape (n, c_out, v, l), where:
                    - n: batch size (same as input)
                    - c_out: number of output channels after projection
                    - v: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        adj = adj + torch.eye(adj.size(0), device=x.device)

        degree = adj.sum(dim=1)
        normalized_adj = adj / degree.view(-1, 1)

        hidden = x
        diffusion_states = [hidden]

        for _ in range(self.diffusion_steps):
            hidden = self.residual_alpha * x + (
                1 - self.residual_alpha
            ) * self.graph_conv(hidden, normalized_adj)
            diffusion_states.append(hidden)

        output = torch.cat(diffusion_states, dim=1)
        output = self.channel_projection(output)

        return output
