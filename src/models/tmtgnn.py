"""src/models/tmtgnn.py

T-MTGNN model.

Provides the `TMTGNN` class, which implements a spatio-temporal graph
neural network combining Transformer-based temporal modeling with
graph diffusion layers and an adaptive graph structure learner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from graph.graph_structure_learner import GraphStructureLearner
from modules.temporal.transformer import Transformer
from modules.spatial.graph_diffusion import GraphDiffusion
from layers.normalization.layer_norm import LayerNorm


class TMTGNN(nn.Module):
    """T-MTGNN model.

    Spatio-temporal graph neural network that combines Transformer-based
    temporal modeling with graph diffusion layers and an adaptive graph
    structure learner, enabling joint temporal and spatial dependency
    modeling over graph-structured data.

    Notes:
        - Temporal dependencies are modeled via Transformer encoder blocks
        - Spatial dependencies are modeled via graph diffusion operators
        - Graph structure is dynamically inferred at each forward pass
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        seq_length: int,
        out_channels: int,
        device: torch.device,
        subgraph_size: int = 20,
        node_dim: int = 40,
        hidden_dim: int = 32,
        skip_dim: int = 64,
        head_dim: int = 128,
        num_layers: int = 3,
        gcn_depth: int = 2,
        residual_alpha: float = 0.05,
        dropout: float = 0.3,
        graph_alpha: float = 3.0,
        layer_norm_affine: bool = True,
        graph_noise_scale: float = 0.01,
        node_features: torch.Tensor | None = None,
        num_heads: int = 4,
        transformer_layers: int = 2,
        layer_norm_eps: float = 1e-5,
        projection_bias: bool = True,
    ) -> None:
        """Initialize TMTGNN.

        Args:
            num_nodes (int):
                Number of nodes in the graph.
            in_channels (int):
                Number of input channels.
            seq_length (int):
                Input sequence length.
            out_channels (int):
                Number of output channels.
            device (torch.device):
                Computation device.
            subgraph_size (int):
                Number of outgoing edges per node in learned graph.
                Default is 20.
            node_dim (int):
                Node embedding dimension used in graph learning.
                Default is 40.
            hidden_dim (int):
                Hidden feature dimension.
                Default is 32.
            skip_dim (int):
                Skip connection feature dimension.
                Default is 64.
            head_dim (int):
                Head dimension before output projection.
                Default is 128.
            num_layers (int):
                Number of spatio-temporal blocks.
                Default is 3.
            gcn_depth (int):
                Number of diffusion steps in graph diffusion layers.
                Default is 2.
            residual_alpha (float):
                Residual propagation coefficient.
                Default is 0.05.
            dropout (float):
                Dropout rate.
                Default is 0.3.
            graph_alpha (float):
                Scaling factor for graph learning non-linearity.
                Default is 3.0.
            layer_norm_affine (bool):
                Whether LayerNorm uses learnable affine parameters.
                Default is True.
            graph_noise_scale (float):
                Noise scale used during adjacency construction.
                Default is 0.01.
            node_features (torch.Tensor | None):
                External node feature matrix used to condition graph construction.
                Default is None.
            num_heads (int):
                Number of attention heads used in Transformer layers.
                Default is 4.
            transformer_layers (int):
                Number of internal layers inside each Transformer block.
                Default is 2.
            layer_norm_eps (float):
                Numerical stability epsilon used in the normalization layer.
                Default is 1e-5.
            projection_bias (bool):
                Whether the projection layers inside graph diffusion use bias terms.
                Default is True.
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.dropout = dropout

        # =========================================================
        # Graph Structure Learning
        # =========================================================
        self.graph_learner = GraphStructureLearner(
            num_nodes=num_nodes,
            top_k=subgraph_size,
            hidden_dim=node_dim,
            alpha=graph_alpha,
            noise_scale=graph_noise_scale,
            node_features=node_features,
        )

        # =========================================================
        # Input Projection
        # =========================================================
        self.input_projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
        )

        # =========================================================
        # Spatio-Temporal Blocks
        # =========================================================
        self.temporal_layers = nn.ModuleList()
        self.diffusion_forward = nn.ModuleList()
        self.diffusion_backward = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.normalization_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.temporal_layers.append(
                Transformer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_head=num_heads,
                    num_layers=transformer_layers,
                    dropout=dropout,
                )
            )
            self.diffusion_forward.append(
                GraphDiffusion(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    diffusion_steps=gcn_depth,
                    residual_alpha=residual_alpha,
                    projection_bias=projection_bias,
                )
            )
            self.diffusion_backward.append(
                GraphDiffusion(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    diffusion_steps=gcn_depth,
                    residual_alpha=residual_alpha,
                    projection_bias=projection_bias,
                )
            )
            self.skip_projections.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=skip_dim,
                    kernel_size=(1, seq_length),
                )
            )
            self.normalization_layers.append(
                LayerNorm(
                    (hidden_dim, num_nodes, seq_length),
                    eps=layer_norm_eps,
                    elementwise_affine=layer_norm_affine,
                )
            )

        # =========================================================
        # Output Head
        # =========================================================
        self.head_1 = nn.Conv2d(
            in_channels=skip_dim,
            out_channels=head_dim,
            kernel_size=(1, 1),
        )
        self.head_2 = nn.Conv2d(
            in_channels=head_dim,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )
        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Compute forward pass of TMTGNN.

        Performs spatio-temporal modeling in the following pipeline:
        (1) Learns a dynamic graph from node embeddings
        (2) Applies temporal attention via Transformer layers
        (3) Performs bidirectional graph diffusion for spatial propagation
        (4) Aggregates multi-layer skip connections
        (5) Produces final predictions via output projection

        Args:
            x (torch.Tensor):
                Input tensor of shape (n, c, v, l), where:
                    - n: batch size
                    - c: input channels
                    - v: number of nodes
                    - l: sequence length
            idx (torch.Tensor | None):
                Node index tensor of shape (v,) where v is the number of nodes,
                used for graph construction. Default is None.

        Returns:
            torch.Tensor:
                Output tensor of shape (n, c_out, v, l), where:
                    - n: batch size
                    - c_out: output channels
                    - v: number of nodes
                    - l: sequence length
        """
        node_idx = idx if idx is not None else self.idx
        adj = self.graph_learner(node_idx)

        x = self.input_projection(x)
        skip = 0

        for i in range(self.num_layers):
            residual = x

            x = self.temporal_layers[i](x)

            x = self.diffusion_forward[i](x, adj) + self.diffusion_backward[i](
                x, adj.t()
            )
            x = F.dropout(x, self.dropout, training=self.training)

            skip += self.skip_projections[i](x)

            x = x + residual
            x = self.normalization_layers[i](x, node_idx)

        x = F.relu(skip)
        x = F.relu(self.head_1(x))
        x = self.head_2(x)

        return x
