"""src/modules/spatial/graph_diffusion.py

Graph diffusion layer.

Provides the `GraphDiffusion` class, which implements multi-step
message passing over a graph with skip-connected input injection of input features.
The layer aggregates node representations across multiple diffusion
steps and projects them into a target feature space.
"""

import torch
import torch.nn as nn
from .graph_conv import GraphConv
from layers import ChannelProjection


class GraphDiffusion(nn.Module):
    """Graph diffusion layer with iterative message passing.

    Performs multi-step iterative message passing over a fixed graph structure.
    At each step, node features are updated via residual mixing of original inputs
    and diffused representations, expanding the receptive field progressively.
    All intermediate states from each diffusion step are concatenated and then
    projected to the desired output dimensionality via learned channel transformation.

    Attributes:
        graph_conv (GraphConv):
            Spatial message passing module for neighbor aggregation.
        projection (ChannelProjection):
            Learnable 1x1 convolution for projecting concatenated diffusion states.
        diffusion_steps (int):
            Number of iterative propagation steps.
        residual_alpha (float):
            Residual mixing coefficient [0, 1] balancing input vs aggregated features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        diffusion_steps: int,
        residual_alpha: float,
        projection_bias: bool,
    ) -> None:
        """Initialize GraphDiffusion layer.

        Args:
            in_channels (int):
                Number of input feature channels before diffusion.
            out_channels (int):
                Number of output feature channels after projection.
            diffusion_steps (int):
                Number of iterative diffusion steps controlling receptive field depth.
            residual_alpha (float):
                Residual mixing coefficient in [0, 1] controlling balance
                between original input (alpha) and aggregated features (1-alpha).
            projection_bias (bool):
                Whether to include learnable bias in output projection layer.
        """
        super().__init__()

        self.graph_conv = GraphConv()
        self.projection = ChannelProjection(
            in_channels=(diffusion_steps + 1) * in_channels,
            out_channels=out_channels,
            bias=projection_bias,
        )
        self.diffusion_steps = diffusion_steps
        self.residual_alpha = residual_alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Compute multi-step graph diffusion with learned projection.

        Performs iterative message passing where at each step node features are
        updated via residual mixing of original input and neighbor-aggregated features.
        This expands the receptive field progressively. All intermediate states are
        concatenated across steps and then projected to output dimensionality.

        Args:
            x (torch.Tensor):
                Input feature map of shape (b, c, n, l), where:
                    - b: batch size
                    - c: number of input channels
                    - n: number of nodes
                    - l: sequence length
            adj (torch.Tensor):
                Adjacency matrix of shape (n, n) or (b, n, n), representing
                fixed graph connectivity, where:
                    - b: batch size (optional)
                    - n: number of nodes

        Returns:
            torch.Tensor:
                Output feature map of shape (b, c_out, n, l), where:
                    - b: batch size (same as input)
                    - c_out: number of output channels after projection
                    - n: number of nodes (same as input)
                    - l: sequence length (same as input)
        """
        # Normalize adjacency to batch format (b, n, n) if needed
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)

        _, N, _ = adj.shape

        # Add self-loops to preserve node's own features during aggregation,
        # ensuring information from the node itself is retained through propagation
        eye = torch.eye(N, device=x.device).unsqueeze(0)
        adj = adj + eye

        # Normalize adjacency by row degree for scale-invariant message passing,
        # preventing feature magnitudes from exploding with graph propagation
        degree = adj.sum(dim=-1, keepdim=True)
        normalized_adj = adj / degree

        # Initialize diffusion with input and collect all intermediate states
        # to capture multi-hop neighborhood information at different scales
        hidden = x
        diffusion_states = [hidden]

        # Iteratively propagate features over the graph, mixing residually
        # to balance local input features with aggregated neighbor influences
        for _ in range(self.diffusion_steps):
            # Aggregate features from neighbors via message passing
            agg = self.graph_conv(hidden, normalized_adj)
            
            # Mix original input with aggregated features using residual_alpha
            # to prevent feature smoothing while spreading information
            hidden = self.residual_alpha * x + (1 - self.residual_alpha) * agg
            diffusion_states.append(hidden)

        # Concatenate all K+1 states (input + K diffusion outputs) along 
        # channel dimension, creating multi-scale neighborhood representations
        # for the projection layer
        output = torch.cat(diffusion_states, dim=1)
        
        # Project concatenated multi-step representations to target output dimensionality
        # via learnable 1x1 convolution for channel-wise feature selection
        output = self.projection(output)

        return output
