"""tmtgnn/models/tmtgnn.py

T-MTGNN.

Provides the `TMTGNN` class, a flexible spatio-temporal graph neural network
combining multi-mode Transformer layers with graph diffusion and optional
adaptive graph structure learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ..graph import GraphStructureLearner
from ..modules import Transformer
from ..modules import GraphDiffusion
from ..layers import LayerNorm
from ..config import DiffusionConfig
from ..config import GraphConfig
from ..config import NormConfig
from ..config import TMTGNNConfig
from ..config import TransformerConfig


class TMTGNN(nn.Module):
    """T-MTGNN.

    Combines multi-mode Transformer-based representations with graph diffusion and
    optional adaptive graph learning, enabling diverse graph-structured forecasting.

    Attributes:
        num_layers (int):
            Number of stacked spatio-temporal blocks.
        dropout (float):
            Dropout rate applied to model layers for regularization.
        graph_learning_enabled (bool):
            Whether to learn graph structure from data.
        node_repr_prev (torch.Tensor | None):
            Previous node representations for exponential moving average (EMA) smoothing
            of graph structure stability across batches.
        ema_alpha (float):
            Exponential moving average factor for smoothing node representations
            and stabilizing learned graph structure.
        node_emb_layer (nn.Embedding):
            Learnable node embeddings encoding node identity.
        graph_learner (GraphStructureLearner | None):
            Optional graph structure learner. None if learning disabled.
        input_projection (nn.Conv2d):
            Initial channel projection to hidden dimension.
        temporal_layers (nn.ModuleList):
            Multi-mode Transformer encoders for representation enrichment.
        diffusion_forward (nn.ModuleList):
            Graph diffusion module (forward).
        diffusion_backward (nn.ModuleList):
            Graph diffusion module (backward).
        skip_projections (nn.ModuleList):
            Skip connection projections for multi-layer aggregation.
        normalization_layers (nn.ModuleList):
            Node-aware layer normalization applied after each block.
        head_1 (nn.Conv2d):
            Output MLP head for final prediction transformation.
        head_2 (nn.Conv2d):
            Output MLP head for final prediction transformation.
        idx (torch.Tensor):
            Registered buffer of node indices used for node-aware normalization
            in normalization layers.
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
                Total number of nodes in the graph.
            in_channels (int):
                Number of input feature channels per node per timestep.
            seq_length (int):
                Input temporal sequence length (timesteps).
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
                Configuration for TMTGNN model.
                If None, uses default.
            transformer_config (TransformerConfig | None):
                Configuration for Transformer. If None, uses default.
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
        self.graph_learning_enabled = graph_config.learning_enabled

        self.node_repr_prev = None
        self.ema_alpha = graph_config.ema_alpha

        # Learnable node embeddings to encode node identity and structural
        # information, helping the model distinguish between different nodes
        # in the graph
        self.node_emb_layer = nn.Embedding(num_nodes, tmtgnn_config.hidden_dim)

        # Optional graph structure learner
        self.graph_learner = None
        if self.graph_learning_enabled:
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
        # flexible Transformer and spatial graph diffusion
        self.temporal_layers = nn.ModuleList()
        self.diffusion_forward = nn.ModuleList()
        self.diffusion_backward = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.normalization_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            # Multi-mode Transformer for representation enrichment
            self.temporal_layers.append(
                Transformer(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.hidden_dim,
                    num_heads=transformer_config.num_heads,
                    num_layers=transformer_config.num_layers,
                    dropout=transformer_config.dropout,
                    max_sequence_length=transformer_config.max_sequence_length,
                    mode=transformer_config.mode,
                )
            )

            # Spatial modeling (forward): diffuse information along graph edges
            self.diffusion_forward.append(
                GraphDiffusion(
                    in_channels=tmtgnn_config.hidden_dim,
                    out_channels=tmtgnn_config.hidden_dim,
                    diffusion_steps=diffusion_config.diffusion_steps,
                    residual_alpha=diffusion_config.residual_alpha,
                    projection_bias=diffusion_config.projection_bias,
                )
            )

            # Spatial modeling (backward): diffuse along transposed edges
            # for bidirectional information exchange
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
                dimension, number of layers, skip dimension, head dimension, dropout,
                and graph learning enable flag.
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
            assert isinstance(device, torch.device), "device must be torch.device"
        except AssertionError as e:
            raise TypeError(f"Invalid TMTGNN parameter: {e}")

        try:
            assert num_nodes > 0, "num_nodes must be > 0"
            assert in_channels > 0, "in_channels must be > 0"
            assert seq_length > 0, "seq_length must be > 0"
            assert out_channels > 0, "out_channels must be > 0"
            assert tmtgnn_config.hidden_dim % transformer_config.num_heads == 0, (
                "hidden_dim must be divisible by num_heads"
            )
            assert tmtgnn_config.hidden_dim >= transformer_config.num_heads, (
                "hidden_dim must be >= num_heads"
            )
            if graph_config.learning_enabled:
                assert 0 < graph_config.top_k < num_nodes, (
                    "top_k must be in (0, num_nodes)"
                )
        except AssertionError as e:
            raise ValueError(f"Invalid TMTGNN parameter: {e}")

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
        adj_base: torch.Tensor | None = None,
        idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute forward pass with flexible graph handling.

        Supports three graph modes:
        - Graph learning mode (with base graph):
           - Starts with fixed base graph
           - Learns cross-graph structure via GraphStructureLearner
           - Combines both
           - Requires adj_base parameter

        - Graph learning mode (without base graph):
           - Learns adjacency matrix from temporally-enriched node representations
           - Starts from scratch (no fixed structure)
           - First block only: graph learned once and reused

        - Predefined graph mode:
           - Uses adjacency matrix provided via adj parameter
           - Useful for large spatio-temporal graphs
           - adj required as input, cannot be None

        Pipeline:
        (1) Project input to hidden dimension for feature space alignment
        (2) For each spatio-temporal block:
            a. Apply Transformer to enrich temporal/node representations
            b. In block 0: learn graph (if enabled) or validate provided graph
            c. Apply bidirectional graph diffusion for spatial propagation
            d. Accumulate skip connections for multi-layer aggregation
            e. Apply residual connections and node-aware normalization
        (3) Aggregate skip connections and project to output space
        (4) Extract next-step predictions

        Args:
            x (torch.Tensor):
                Input tensor of shape (b, c, n, l), where:
                    - b: batch size
                    - c: input feature channels
                    - n: number of nodes
                    - l: sequence length (temporal dimension)
            adj (torch.Tensor | None):
                Predefined adjacency matrix of shape (n, n) or (b, n, n).
                - Required if graph_learning_enabled=False (no adj_base)
                - Ignored if graph_learning_enabled=True and adj_base provided
                - Default: None
            adj_base (torch.Tensor | None):
                Base/fixed adjacency matrix for hybrid learning, shape (n, n) or (b, n, n).
                - Used with graph_learning_enabled=True to combine fixed + learned edges
                - Default: None (learn all edges from scratch)
            idx (torch.Tensor | None):
                Optional node index tensor of shape (n,) for indexing node-specific
                parameters in normalization layers. If None, uses registered buffer.

        Returns:
            torch.Tensor:
                Output predictions of shape (b, c_out, n)

        Raises:
            ValueError:
                If graph_learning_enabled=False and both adj and adj_base are None.
        """
        # Validate graph input when learning is disabled
        if not self.graph_learning_enabled and adj is None and adj_base is None:
            raise ValueError(
                "graph_learning_enabled=False requires either adj or adj_base to be provided."
            )

        # Use provided node indices or fall back to registered buffer
        node_idx = idx if idx is not None else self.idx

        # Project input features to hidden dimension for consistent feature space
        # across all layers, enabling better gradient flow and learning
        x = self.input_projection(x)

        # Initialize skip connection accumulator and adjacency matrix
        skip = None
        learned_adj = None

        # Process through stacked spatio-temporal blocks
        for i in range(self.num_layers):
            # Store input for residual connection
            residual = x

            # Apply multi-mode Transformer to enrich representations.
            x = self.temporal_layers[i](x)

            # Handle graph structure in first block only
            if i == 0:
                if self.graph_learning_enabled:
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
                    learned_adj = torch.stack(
                        [self.graph_learner(nr) for nr in node_repr], dim=0
                    )

                    # Combine base graph + learned edges
                    if adj_base is not None:
                        # Ensure base graph is in batch format
                        base_adj = adj_base
                        if base_adj.dim() == 2:
                            base_adj = base_adj.unsqueeze(0).expand(x.size(0), -1, -1)

                        # Combine fixed edges + learned edges, both contributing
                        # to information flow (but learned edges are dynamic)
                        active_adj = base_adj + learned_adj
                    else:
                        # Pure learning mode (only learned edges)
                        active_adj = learned_adj
                else:
                    # Use predefined adjacency matrix
                    active_adj = adj if adj is not None else adj_base

                    # Ensure proper shape: expand to batch dimension if needed
                    if active_adj.dim() == 2:
                        active_adj = active_adj.unsqueeze(0).expand(x.size(0), -1, -1)
            else:
                # Use adjacency from first block in subsequent blocks
                if self.graph_learning_enabled:
                    # In subsequent layers, reuse the same combined graph
                    if adj_base is not None:
                        base_adj = adj_base
                        if base_adj.dim() == 2:
                            base_adj = base_adj.unsqueeze(0).expand(x.size(0), -1, -1)
                        active_adj = base_adj + learned_adj
                    else:
                        active_adj = learned_adj
                else:
                    active_adj = adj if adj is not None else adj_base

            # Apply bidirectional graph diffusion for spatial information propagation
            x = self.diffusion_forward[i](x, active_adj) + self.diffusion_backward[i](
                x, active_adj.transpose(-1, -2)
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
            # node-specific properties and characteristics
            x = self.normalization_layers[i](x, node_idx)

        # Project aggregated skip connections through output head
        x = self.head_1(skip)
        x = F.relu(x)
        x = self.head_2(x)

        # Single-step forecasting
        x = x[:, :, :, -1]

        if x.size(1) == 1:
            x = x[:, 0]

        return x
