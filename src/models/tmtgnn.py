"""src/models/tmtgnn.py

T-MTGNN model.

Provides the `TMTGNN` class, which implements a spatio-temporal graph
neural network combining Transformer-based temporal modeling with
graph diffusion layers and an adaptive graph structure learner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from graph import GraphStructureLearner
from modules import Transformer
from modules import GraphDiffusion
from layers import LayerNorm
from config import DiffusionConfig
from config import GraphConfig
from config import NormConfig
from config import TMTGNNConfig
from config import TransformerConfig


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
        diffusion_config: DiffusionConfig | None = None,
        graph_config: GraphConfig | None = None,
        norm_config: NormConfig | None = None,
        tmtgnn_config: TMTGNNConfig | None = None,
        transformer_config: TransformerConfig | None = None,
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
            diffusion_config (DiffusionConfig | None):
                Configuration for graph diffusion layers.
                Default is None.
            graph_config (GraphConfig | None):
                Configuration for graph structure learning.
                Default is None.
            norm_config (NormConfig | None):
                Configuration for normalization layers.
                Default is None.
            tmtgnn_config (TMTGNNConfig | None):
                Configuration for TMTGNN model hyperparameters.
                Default is None.
            transformer_config (TransformerConfig | None):
                Configuration for Transformer temporal modeling.
                Default is None.

        """
        super().__init__()

        # =========================================================
        # Setup
        # =========================================================
        diffusion_config = (
            copy.deepcopy(diffusion_config) if diffusion_config else DiffusionConfig()
        )
        graph_config = copy.deepcopy(graph_config) if graph_config else GraphConfig()
        norm_config = copy.deepcopy(norm_config) if norm_config else NormConfig()
        tmtgnn_config = (
            copy.deepcopy(tmtgnn_config) if tmtgnn_config else TMTGNNConfig()
        )
        transformer_config = (
            copy.deepcopy(transformer_config)
            if transformer_config
            else TransformerConfig()
        )

        self.num_layers = tmtgnn_config.num_layers
        self.dropout = tmtgnn_config.dropout
        self.num_forecast_steps = tmtgnn_config.num_forecast_steps
        self.node_repr_prev = None
        self.ema_alpha = graph_config.ema_alpha
        self.node_emb_layer = nn.Embedding(num_nodes, tmtgnn_config.hidden_dim)
        
        # =========================================================
        # Configuration Validations
        # =========================================================
        assert tmtgnn_config.hidden_dim % transformer_config.num_heads == 0, (
            "hidden_dim must be divisible by num_heads"
        )
        assert tmtgnn_config.hidden_dim >= transformer_config.num_heads, (
            "hidden_dim must be >= num_heads"
        )
        assert 0 < graph_config.top_k < num_nodes, "top_k must be in (0, num_nodes)"

        assert num_nodes > 0, "num_nodes must be > 0"
        assert isinstance(num_nodes, int), "num_nodes must be an int"

        assert in_channels > 0, "in_channels must be > 0"
        assert isinstance(in_channels, int), "in_channels must be an int"

        assert seq_length > 0, "seq_length must be > 0"
        assert isinstance(seq_length, int), "seq_length must be an int"

        assert out_channels > 0, "out_channels must be > 0"
        assert isinstance(out_channels, int), "out_channels must be an int"

        assert isinstance(device, torch.device), "device must be torch.device"

        # =========================================================
        # Graph Structure Learning
        # =========================================================
        self.graph_learner = GraphStructureLearner(
            top_k=graph_config.top_k,
            hidden_dim=tmtgnn_config.hidden_dim,
            sigmoid_alpha=graph_config.sigmoid_alpha,
            noise_scale=graph_config.noise_scale,
        )

        # =========================================================
        # Input Projection
        # =========================================================
        self.input_projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=tmtgnn_config.hidden_dim,
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

        for _ in range(self.num_layers):
            self.temporal_layers.append(
                Transformer(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.hidden_dim,
                    num_head=transformer_config.num_heads,
                    num_layers=transformer_config.num_layers,
                    dropout=transformer_config.dropout,
                    max_sequence_length=transformer_config.max_sequence_length,
                )
            )
            self.diffusion_forward.append(
                GraphDiffusion(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.hidden_dim,
                    diffusion_steps=diffusion_config.diffusion_steps,
                    residual_alpha=diffusion_config.residual_alpha,
                    projection_bias=diffusion_config.projection_bias,
                )
            )
            self.diffusion_backward.append(
                GraphDiffusion(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.hidden_dim,
                    diffusion_steps=diffusion_config.diffusion_steps,
                    residual_alpha=diffusion_config.residual_alpha,
                    projection_bias=diffusion_config.projection_bias,
                )
            )
            self.skip_projections.append(
                nn.Conv2d(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.skip_dim,
                    kernel_size=(1, 1)
                )
            )
            self.normalization_layers.append(
                LayerNorm(
                    (tmtgnn_config.hidden_dim, num_nodes, seq_length),
                    eps=norm_config.eps,
                    elementwise_affine=norm_config.elementwise_affine,
                )
            )

        # =========================================================
        # Output Head
        # =========================================================
        self.head_1 = nn.Conv2d(
            in_channels=tmtgnn_config.skip_dim,
            out_channels=tmtgnn_config.head_dim,
            kernel_size=(1, 1),
        )
        self.head_2 = nn.Conv2d(
            in_channels=tmtgnn_config.head_dim,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )
        
        self.register_buffer("idx", torch.arange(num_nodes, device=device))

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Compute forward pass of TMTGNN.

        Performs spatio-temporal modeling in the following pipeline:
        (1) Projects input to hidden dimension
        (2) For each spatio-temporal block:
            a. Applies temporal Transformer to enrich temporal representations
            b. First layer learns graph structure from temporally-enriched node embeddings
            c. Applies bidirectional graph diffusion for spatial propagation
            d. Aggregates multi-layer skip connections
            e. Applies residual connections and normalization
        (3) Produces final predictions via output projection

        The key insight is that temporal encoding happens before graph learning,
        so the graph is constructed from nodes that have seen the full temporal history.

        Args:
            x (torch.Tensor):
                Input tensor of shape (b, c, v, l), where:
                    - b: batch size
                    - c: input channels
                    - v: number of nodes
                    - l: sequence length
            idx (torch.Tensor | None):
                Node index tensor of shape (v,) where v is the number of nodes,
                used for graph construction. Default is None.

        Returns:
            torch.Tensor:
                Output tensor for predictions:
                    Single-step (num_forecast_steps == 1):
                        - (b, c_out, v) if out_channels > 1
                        - (b, v) if out_channels == 1
                    Multi-horizon (num_forecast_steps > 1):
                        - (b, num_forecast_steps, v, c_out) if out_channels > 1
                        - (b, num_forecast_steps, v) if out_channels == 1
        """                
        node_idx = idx if idx is not None else self.idx

        x = self.input_projection(x)

        skip = None
        adj = None

        for i in range(self.num_layers):
            residual = x
            x = self.temporal_layers[i](x)

            if i == 0:
                h = x.mean(dim=-1)
                h = h.permute(0, 2, 1)

                node_emb = self.node_emb_layer(node_idx)
                node_emb = node_emb.unsqueeze(0).expand(h.size(0), -1, -1)

                node_repr = h + node_emb

                node_repr_mean = node_repr.mean(dim=0)
                if self.node_repr_prev is not None:
                    node_repr_mean = (
                        self.ema_alpha * node_repr_mean +
                        (1 - self.ema_alpha) * self.node_repr_prev
                    )

                self.node_repr_prev = node_repr_mean.detach()
                node_repr = node_repr + node_repr_mean.unsqueeze(0)

                adj = torch.stack(
                    [self.graph_learner(nr) for nr in node_repr],
                    dim=0
                )

            x = self.diffusion_forward[i](x, adj) + self.diffusion_backward[i](
                x, adj.transpose(-1, -2)
            )
            x = F.dropout(x, self.dropout, training=self.training)

            proj = self.skip_projections[i](x) / self.num_layers
            skip = proj if skip is None else skip + proj

            x = x + residual
            x = self.normalization_layers[i](x, node_idx)

        x = self.head_1(skip)
        x = F.relu(x)
        x = self.head_2(x)
        
        if self.num_forecast_steps == 1:
            x = x[:, :, :, -1]
            if x.size(1) == 1:
                x = x[:, 0]
        else:
            x = x[:, :, :, -self.num_forecast_steps:]
            x = x.permute(0, 3, 2, 1).contiguous()
            if x.size(3) == 1:
                x = x[:, :, :, 0]

        return x
