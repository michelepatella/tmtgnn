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
    """Transformer-based Multivariate Temporal Graph Neural Network.

    Spatio-temporal graph neural network combining Transformer-based temporal
    modeling with graph diffusion layers and adaptive graph structure learning.
    Enables joint temporal and spatial dependency modeling over graph-structured
    time series data by first enriching temporal representations, then learning
    data-adaptive graph structure, and finally propagating information via diffusion.

    Attributes:
        graph_learner (GraphStructureLearner):
            Learns adaptive graph structure from temporally-enriched node 
            representations.
        input_projection (nn.Conv2d):
            Projects input channels to hidden dimension via 1x1 convolution.
        temporal_layers (nn.ModuleList):
            Stacked Transformer encoders for temporal self-attention per node.
        diffusion_forward (nn.ModuleList):
            Graph diffusion modules for forward spatial propagation.
        diffusion_backward (nn.ModuleList):
            Graph diffusion modules for backward spatial propagation.
        skip_projections (nn.ModuleList):
            1x1 convolutions projecting block outputs to skip dimension.
        normalization_layers (nn.ModuleList):
            Node-aware layer normalization applied after each block.
        node_emb_layer (nn.Embedding):
            Learnable node embeddings encoding node identity/position.
        head_1 (nn.Conv2d):
            Output MLP head for final prediction transformation.
        head_2 (nn.Conv2d):
            Output MLP head for final prediction transformation.
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
                Number of input feature channels per node and time step.
            seq_length (int):
                Input temporal sequence length.
            out_channels (int):
                Number of output feature channels (prediction dimensionality).
            device (torch.device):
                Computation device for model parameters and buffers.
            diffusion_config (DiffusionConfig | None):
                Configuration for graph diffusion layers. If None, uses default.
            graph_config (GraphConfig | None):
                Configuration for graph structure learning. If None, uses default.
            norm_config (NormConfig | None):
                Configuration for normalization layers. If None, uses default.
            tmtgnn_config (TMTGNNConfig | None):
                Configuration for TMTGNN model hyper-parameters. If None, uses default.
            transformer_config (TransformerConfig | None):
                Configuration for Transformer temporal modeling. If None, uses default.
        """
        super().__init__()

        # Resolve configurations (use provided config or create default instances)
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

        # Validate parameters
        self.__post_init__(
            num_nodes,
            in_channels,
            seq_length,
            out_channels,
            device,
            graph_config,
            tmtgnn_config,
            transformer_config,
        )

        self.num_layers = tmtgnn_config.num_layers
        self.dropout = tmtgnn_config.dropout
        self.num_forecast_steps = tmtgnn_config.num_forecast_steps
        self.node_repr_prev = None
        self.ema_alpha = graph_config.ema_alpha
        
        # Learnable node embeddings to encode node identity and structural 
        # information, helping the model distinguish between different nodes
        # in the graph
        self.node_emb_layer = nn.Embedding(num_nodes, tmtgnn_config.hidden_dim)

        # Learner infers adaptive graph structure from node embeddings,
        # enabling the model to discover data-driven relationships
        self.graph_learner = GraphStructureLearner(
            top_k=graph_config.top_k,
            hidden_dim=tmtgnn_config.hidden_dim,
            sigmoid_alpha=graph_config.sigmoid_alpha,
            noise_scale=graph_config.noise_scale,
        )

        # Project input to hidden dimension for consistent feature space
        # across all layers via learnable 1x1 convolution
        self.input_projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=tmtgnn_config.hidden_dim,
            kernel_size=(1, 1),
        )

        # Create L stacked spatio-temporal blocks, each combining
        # temporal Transformer and spatial graph diffusion
        self.temporal_layers = nn.ModuleList()
        self.diffusion_forward = nn.ModuleList()
        self.diffusion_backward = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.normalization_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            # Temporal modeling: Transformer encoder for per-node self-attention
            # over temporal dimension, learning temporal dependencies
            self.temporal_layers.append(
                Transformer(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.hidden_dim,
                    num_heads=transformer_config.num_heads,
                    num_layers=transformer_config.num_layers,
                    dropout=transformer_config.dropout,
                    max_sequence_length=transformer_config.max_sequence_length,
                )
            )

            # Spatial modeling (forward): diffuse information along learned 
            # graph edges
            self.diffusion_forward.append(
                GraphDiffusion(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.hidden_dim,
                    diffusion_steps=diffusion_config.diffusion_steps,
                    residual_alpha=diffusion_config.residual_alpha,
                    projection_bias=diffusion_config.projection_bias,
                )
            )

            # Spatial modeling (backward): diffuse information along transposed 
            # graph edges for bidirectional information exchange
            self.diffusion_backward.append(
                GraphDiffusion(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.hidden_dim,
                    diffusion_steps=diffusion_config.diffusion_steps,
                    residual_alpha=diffusion_config.residual_alpha,
                    projection_bias=diffusion_config.projection_bias,
                )
            )

            # Skip connections: project block outputs to skip dimension,
            # aggregating multi-layer representations for final prediction
            self.skip_projections.append(
                nn.Conv2d(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.skip_dim,
                    kernel_size=(1, 1),
                )
            )

            # Node-aware normalization: scales and shifts per-node features
            # differently, enabling heterogeneous normalization across nodes
            self.normalization_layers.append(
                LayerNorm(
                    normalized_shape=(tmtgnn_config.hidden_dim, num_nodes, seq_length),
                    eps=norm_config.eps,
                    elementwise_affine=norm_config.elementwise_affine,
                )
            )

        # Output heads for final prediction transformation
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

    @staticmethod
    def __post_init__(
        num_nodes: int,
        in_channels: int,
        seq_length: int,
        out_channels: int,
        device: torch.device,
        graph_config: GraphConfig,
        tmtgnn_config: TMTGNNConfig,
        transformer_config: TransformerConfig,
    ) -> None:
        """Validates initialization parameters before module construction.
        
        Args:
            num_nodes (int):
                Number of nodes in the graph.
            in_channels (int):
                Number of input feature channels per node and time step.
            seq_length (int):
                Input temporal sequence length.
            out_channels (int):
                Number of output feature channels (prediction dimensionality).
            device (torch.device):
                Computation device for model parameters and buffers.
            graph_config (GraphConfig):
                Configuration for graph structure learning including top-k 
                sparsification, sigmoid scaling, noise regularization, and EMA smoothing.
            tmtgnn_config (TMTGNNConfig):
                Configuration for TMTGNN model hyper-parameters including hidden
                dimension, number of layers, skip dimension, head dimension, and dropout.
            transformer_config (TransformerConfig):
                Configuration for Transformer temporal modeling including number of heads,
                encoder layers, dropout, and maximum sequence length for positional encoding.
        
        Raises:
            TypeError:
                If any parameter has incorrect type.
            ValueError:
                If any parameter value violates constraints.
        """
        try:
            assert isinstance(num_nodes, int), "num_nodes must be int"
            assert isinstance(in_channels, int), "in_channels must be int"
            assert isinstance(seq_length, int), "seq_length must be int"
            assert isinstance(out_channels, int), "out_channels must be int"
            assert isinstance(device, torch.device), (
                "device must be torch.device"
            )
        except AssertionError as e:
            raise TypeError(f"Invalid TMTGNN parameter: {e}")

        try:
            assert num_nodes > 0, "num_nodes must be > 0"
            assert in_channels > 0, "in_channels must be > 0"
            assert seq_length > 0, "seq_length must be > 0"
            assert out_channels > 0, "out_channels must be > 0"
            assert (
                tmtgnn_config.hidden_dim % transformer_config.num_heads == 0
            ), "hidden_dim must be divisible by num_heads"
            assert (
                tmtgnn_config.hidden_dim >= transformer_config.num_heads
            ), "hidden_dim must be >= num_heads"
            assert (
                0 < graph_config.top_k < num_nodes
            ), "top_k must be in (0, num_nodes)"
        except AssertionError as e:
            raise ValueError(f"Invalid TMTGNN parameter: {e}")

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Compute forward pass of TMTGNN with adaptive graph structure.

        Performs spatio-temporal modeling via the following pipeline:
        (1) Project input to hidden dimension for feature space alignment
        (2) For each spatio-temporal block:
            a. Apply temporal Transformer to enrich temporal representations
            b. First layer learns adaptive graph structure from temporal features
            c. Apply bidirectional graph diffusion for spatial information propagation
            d. Accumulate skip connections for multi-layer representation aggregation
            e. Apply residual connections and node-aware normalization
        (3) Aggregate skip connections and project to output space
        (4) Extract predictions for specified forecast horizon

        The key design principle: temporal enrichment enables better graph structure
        learning, as node representations with seen full temporal history can capture
        meaningful dependencies between nodes for structure inference.

        Args:
            x (torch.Tensor):
                Input tensor of shape (b, c, n, l), where:
                    - b: batch size
                    - c: input feature channels
                    - n: number of nodes
                    - l: sequence length (temporal dimension)
            idx (torch.Tensor | None):
                Optional node index tensor of shape (n,) for indexing node-specific
                parameters in normalization layers. If None, uses registered buffer.

        Returns:
            torch.Tensor:
                Output predictions of shape (b, c_out, n) for single-step forecast,
                or (b, num_forecast_steps, n, c_out) for multi-horizon, with
                dimensions squeezed if c_out == 1.
        """
        # Use provided node indices or fall back to registered buffer
        node_idx = idx if idx is not None else self.idx

        # Project input features to hidden dimension for consistent feature space
        # across all layers, enabling better gradient flow and learning
        x = self.input_projection(x)

        # Initialize skip connection accumulator and adjacency matrix
        skip = None
        adj = None

        # Process through stacked spatio-temporal blocks
        for i in range(self.num_layers):
            # Store input for residual connection
            residual = x
            
            # Apply temporal Transformer to enrich temporal representations,
            # enabling better capture of long-range temporal dependencies
            x = self.temporal_layers[i](x)

            # Learn graph structure only in first layer from temporally-enriched features,
            # ensuring nodes have seen full temporal history for structure inference
            if i == 0:
                # Extract temporal mean to get node-level representation,
                # averaging over time to form static node embeddings
                h = x.mean(dim=-1)
                h = h.permute(0, 2, 1)

                # Add learnable node embeddings encoding node identity information,
                # helping distinguish nodes and providing positional context
                node_emb = self.node_emb_layer(node_idx)
                node_emb = node_emb.unsqueeze(0).expand(h.size(0), -1, -1)

                # Combine temporal features with structural node embeddings
                node_repr = h + node_emb

                # Apply EMA smoothing to stabilize graph structure across batches,
                # preventing drastic changes in learned adjacency
                node_repr_mean = node_repr.mean(dim=0)
                if self.node_repr_prev is not None:
                    node_repr_mean = (
                        self.ema_alpha * node_repr_mean
                        + (1 - self.ema_alpha) * self.node_repr_prev
                    )

                self.node_repr_prev = node_repr_mean.detach()
                node_repr = node_repr + node_repr_mean.unsqueeze(0)

                # Learn sparse adaptive graph structure for each sample in batch,
                # enabling data-driven discovery of meaningful node relationships
                adj = torch.stack([self.graph_learner(nr) for nr in node_repr], dim=0)

            # Apply bidirectional graph diffusion for spatial information propagation.
            # Forward diffusion propagates along learned edges, backward on transposed edges
            x = self.diffusion_forward[i](x, adj) + self.diffusion_backward[i](
                x, adj.transpose(-1, -2)
            )
            
            # Apply dropout for regularization and preventing over-fitting
            x = F.dropout(x, self.dropout, training=self.training)

            # Project to skip dimension and accumulate across layers,
            # normalizing contribution of each layer to final prediction
            proj = self.skip_projections[i](x) / self.num_layers
            skip = proj if skip is None else skip + proj

            # Apply residual connection for improved gradient flow
            x = x + residual
            
            # Apply node-aware normalization: scales and shifts per-node features 
            # differently, enabling heterogeneous normalization that respects 
            # node-specific properties
            x = self.normalization_layers[i](x, node_idx)

        # Project aggregated skip connections through output head
        x = self.head_1(skip)
        x = F.relu(x)
        x = self.head_2(x)

        # Extract predictions for specified forecast horizon
        if self.num_forecast_steps == 1:
            # Single-step forecast: take last time step and squeeze if needed
            x = x[:, :, :, -1]
            if x.size(1) == 1:
                x = x[:, 0]
        else:
            # Multi-horizon forecast: extract last K time steps and reorder dimensions
            x = x[:, :, :, -self.num_forecast_steps :]
            x = x.permute(0, 3, 2, 1).contiguous()
            if x.size(3) == 1:
                x = x[:, :, :, 0]

        return x
