<a id="readme-top"></a>

<br/>
<div align="center">
  <h1 align="center">T-MTGNN</h1>
  <p align="center">
    PyTorch implementation of a deep learning model combining <br>
    Transformers and Graph Neural Networks for multivariate time series forecasting.
  </p>
  <p align="center">
    <a href="https://github.com/michelepatella/crypto-closing-price-forecasting">
      <img src="https://img.shields.io/badge/GitHub-Demo-black?style=for-the-badge&logo=github" />
    </a>
  </p>
</div>

<br/>
<br/>
  
<details>
  <summary><strong>Table of Contents</strong></summary>
  ────────────
  <ul>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li>
          <a href="#architecture">Architecture</a>
          <ul>
            <li><a href="#input-projection">Input Projection</a></li>
            <li>
              <a href="#spatio-temporal-blocks">Spatio-Temporal Blocks</a>
              <ul>
                <li><a href="#temporal-module">Temporal Module</a></li>
                <li><a href="#spatial-module">Spatial Module</a></li>
                <li><a href="#dropout">Dropout</a></li>
                <li><a href="#skip--residual-connections">Skip & Residual Connections</a></li>
                <li><a href="#normalization">Normalization</a></li>
              </ul>
            </li>
            <li><a href="#output-head">Output Head</a></li>
          </ul>
        </li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ul>
</details>

<br/>

## About The Project

T-MTGNN is a spatio-temporal graph neural network designed for multivariate time series forecasting, combining Transformer-based 
temporal modeling, graph diffusion mechanisms, and an optional adaptive graph structure learning module.

### Architecture

#### Input Projection

A 1×1 convolution that projects input features into a shared latent space, ensuring uniform dimensionality across subsequent modules.

| Tensor | Shape |
| :--- | :--- |
| **Input** | `(batch_size, in_channels, num_nodes, seq_length)` |
| **Output** | `(batch_size, hidden_dim, num_nodes, seq_length)` |

#### Spatio-Temporal Blocks

Each stacked block consists of:

##### Temporal Module

A stacked Transformer encoder with multi-head self-attention, dropout (for regularization between encoder layers), 
sinusoidal positional encoding (for temporal/node order information, before attention), and a linear projection 
layer (for matching input and hidden dimensions).

It operates in two modes:
  
- Temporal mode:
  - Self-attention is applied along the temporal dimension independently per node
  - Internal `(batch_size x num_nodes, seq_length, hidden_dim)` reshaping so that each node analyzes its sequence
  - A causal, lower triangular mask is applied, preventing attention to future positions

- Node mode:
  - Self-attention is applied along the node dimension independently per timestep
  - Internal `(batch_size x seq_length, num_nodes, hidden_dim)` reshaping so that each node analyzes global spatial correlations
  - No causal masking is applied, allowing full spatial connectivity

Overall, the Transformer builds hierarchical representations that enable rich temporal and node-level dependency modeling.

| Tensor | Shape |
| :--- | :--- |
| **Input** | `(batch_size, hidden_dim, num_nodes, seq_length)` |
| **Output** | `(batch_size, hidden_dim, num_nodes, seq_length)` |

##### Spatial Module

Graph diffusion modules performing scale-invariant iterative message passing over a graph with skip-connected feature injection.

At each step, features are propagated over the graph and a residual mixing of the original input with diffused neighbor 
representations is applied, expanding the receptive field progressively.

Feature aggregation is performed via a weighted sum of source node features across all channels and timesteps, based on the edge
weights from the adjacency matrix. This allows each sample in the batch to utilize a different dynamically generated graph structure.

Node representations across multiple diffusion steps are aggregated and projected into a target feature space via a learnable
1×1 convolution.

Overall, the graph diffusion module enriches data with spatial information.

> **Note**: The model applies bidirectional message passing, summing up forward and backward steps for bidirectional information flow,
capturing asymmetric spatial interactions.

| Tensor | Shape |
| :--- | :--- |
| **Input** | `(batch_size, hidden_dim, num_nodes, seq_length)` |
| **Output** | `(batch_size, hidden_dim, num_nodes, seq_length)` |

##### Dropout

A regularization layer applied directly to the hidden states immediately after the bidirectional spatial graph diffusion process. 

During training, elements of the intermediate feature maps are randomly zeroed out, while during inference, the layer acts as 
an identity function. This prevents feature co-adaptation across the spatio-temporal layers, breaks redundant dependencies 
between channels, and mitigates overfitting.

| Tensor | Shape |
| :--- | :--- |
| **Input** | `(batch_size, hidden_dim, num_nodes, seq_length)` |
| **Output** | `(batch_size, hidden_dim, num_nodes, seq_length)` |

##### Skip & Residual Connections

Projection of block outputs to the skip dimension via 1×1 convolution. These projected feature maps are scaled by the number of
layers and accumulated across all blocks to construct the final representation for prediction.

This preserves multi-layer representations and mitigates vanishing gradients.

| Tensor | Shape |
| :--- | :--- |
| **Input** | `(batch_size, hidden_dim, num_nodes, seq_length)` |
| **Output** | `(batch_size, skip_dim, num_nodes, seq_length)` |

##### Normalization

Node-aware normalization which scales and shifts per-node features differently via node-dependent affine parameters, enabling
the model to handle heterogeneous feature distributions across different graph nodes. Features are normalized collectively across 
the hidden, node, and temporal dimensions for each batch element.

If enabled, the model indexes independent scale and bias parameters for each node via node index tensor, allowing customized
feature scaling based on specific node dynamics.

| Tensor | Shape |
| :--- | :--- |
| **Input** | `(batch_size, hidden_dim, num_nodes, seq_length)` |
| **Output** | `(batch_size, hidden_dim, num_nodes, seq_length)` |

#### Output Head

Processing of the accumulated multi-layer representations from the skip connections through a two-stage projection network to
generate the final single-step predictions. 

It consists of two stacked 1×1 convolutions interspersed with a non-linear activation. The first layer (`head_1`) projects the 
aggregated features from `skip_dim` to an intermediate hidden space (`head_dim`), followed by a ReLU activation to inject 
non-linearity. The second layer (`head_2`) subsequently maps these intermediate features to the target `out_channels`. 

The model slices the output tensor along the temporal axis, retaining only the last timestep. This performs a temporal 
reduction to output a single-step prediction. If the final output channel dimension equals 1, it is automatically squeezed to 
deliver a clean spatial representation matrix.

| Step / Layer | Input Shape | Output Shape | Operation Detail |
| :--- | :--- | :--- | :--- |
| **`head_1` (+ ReLU)** | `(batch_size, skip_dim, num_nodes, seq_length)` | `(batch_size, head_dim, num_nodes, seq_length)` | Reduces skip dimensionality to intermediate hidden space. |
| **`head_2`** | `(batch_size, head_dim, num_nodes, seq_length)` | `(batch_size, out_channels, num_nodes, seq_length)` | Maps intermediate features to final target channels. |
| **Temporal Reduction** | `(batch_size, out_channels, num_nodes, seq_length)` | `(batch_size, out_channels, num_nodes)` | Extracts the last timestep for single-step forecasting. |
| **Squeeze (Conditional)** | `(batch_size, out_channels, num_nodes)` | `(batch_size, num_nodes)` | Removes channel dimension if `out_channels == 1`. |

> [!NOTE]
> **Graph Structure Learning**
> 
> When enabled, this component dynamically computes a sparse, asymmetric adjacency matrix by integrating
> temporal features with structural node identity. The process operates in three key phases:
> 
> 1. Node Representation: In the first spatio-temporal block, temporal features are averaged over time and combined with learnable
>    embeddings that encode node identity. To ensure structural stability across training batches, an Exponential Moving Average (EMA)
>    smoothing is applied to these joint representations.
> 2. Structure Computation: The smoothed features are projected into distinct source and destination spaces using linear encoders
>    with `tanh` activations. Asymmetric interaction scores are then computed via a directional matrix product.
>    These scores are mapped to [0, 1] using a scaled sigmoid activation. During training, a stochastic
>    random perturbation is injected to regularize edge discovery.
> 3. Top-k Sparsification: To enforce graph locality and eliminate weak or redundant connections, a binary mask is generated to
>    retain only the `k` highest-scoring outgoing edges per node, zeroing out the rest.
> 
> The resulting matrix is fixed at the end of the first block and reused across all subsequent layers. It operates either
> in pure adaptive mode or hybrid mode.
<br>

<p align="right"><a href="#readme-top">Top ↑</a></p>

### Built With

[![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)  

<p align="right"><a href="#readme-top">Top ↑</a></p>

## Getting Started

### Prerequisites

**Python**  
Required version: >=3.13  

> [!WARNING]
> Compatibility with earlier or later Python versions has not been tested.  

### Installation

Install the package directly from GitHub:
```sh
pip install git+https://github.com/michelepatella/tmtgnn.git
```

Alternatively, for development setup:
```sh
git clone https://github.com/michelepatella/tmtgnn.git
cd tmtgnn
pip install -e .
```

<p align="right"><a href="#readme-top">Top ↑</a></p>

## Usage

Define model parameters, including:
| Parameter | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`num_nodes`** | `int` | *Required* | `> 0` | Total number of nodes in the graph. |
| **`in_channels`** | `int` | *Required* | `> 0` | Number of input feature channels per node per timestep. |
| **`seq_length`** | `int` | *Required* | `> 0` | Input temporal sequence length (timesteps). |
| **`out_channels`** | `int` | *Required* | `> 0` | Number of output feature channels (prediction dimensionality). |
| **`device`** | `torch.device` | *Required* | Must be `torch.device` | Computation device for model parameters and buffers. |
| **`diffusion_config`** | `DiffusionConfig`  | `None` | - | Configuration object for graph diffusion layers. |
| **`graph_config`** | `GraphConfig` | `None` | - | Configuration object for graph structure learning. |
| **`norm_config`** | `NormConfig` | `None` | - | Configuration object for normalization layers. |
| **`tmtgnn_config`** | `TMTGNNConfig` | `None` | - | Configuration object for hyper-parameters specific to the TMTGNN architecture. |
| **`transformer_config`**| `TransformerConfig` | `None` | - | Configuration object for the internal Transformer temporal modeling. |

Specifically, model configurations include:
| Configuration | Parameter | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DiffusionConfig** | `diffusion_steps` | `int` | `2` | `> 0` | Number of diffusion steps in graph diffusion layers. |
| | `residual_alpha` | `float` | `0.05` | `[0.0, 1.0]` | Residual propagation coefficient. |
| | `projection_bias` | `bool` | `True` | - | Whether the projection layers inside graph diffusion use bias terms. |
| **GraphConfig** | `learning_enabled` | `bool` | `True` | - | Whether to learn graph structure adaptively from data. |
| | `top_k` | `int` | `20` | `> 0`<br>➔ `top_k < num_nodes` (if `learning_enabled=True`) | Number of outgoing edges per node in learned graph. |
| | `sigmoid_alpha` | `float` | `3.0` | `> 0.0` | Scaling factor for sigmoid non-linearity sharpness. |
| | `noise_scale` | `float` | `0.01` | `>= 0.0` | Noise scale used during adjacency construction. |
| | `ema_alpha` | `float` | `0.8` | `[0.0, 1.0]` | Exponential moving average factor. |
| **NormConfig** | `eps` | `float` | `1e-5` | `(0.0, 1.0)` | Numerical stability epsilon used in the normalization layer. |
| | `elementwise_affine` | `bool` | `True` | - | Whether the normalization layer uses learnable affine parameters. |
| **TMTGNNConfig** | `hidden_dim` | `int` | `32` | `> 0`<br>➔ `hidden_dim >= num_heads`<br>➔ `hidden_dim % num_heads == 0` | Hidden feature dimension used throughout the model. |
| | `skip_dim` | `int` | `64` | `>= hidden_dim` | Skip connection feature dimension used for multi-layer aggregation. |
| | `head_dim` | `int` | `128` | `>= hidden_dim` | Head dimension before output projection in the final layers. |
| | `num_layers` | `int` | `3` | `> 0` | Number of spatio-temporal blocks in the model. |
| | `dropout` | `float` | `0.3` | `[0.0, 1.0]` | Dropout rate applied to the model's layers. |
| **TransformerConfig** | `mode` | `str` | `"temporal"` | `"temporal"`, `"node"` | Attention mode for self-attention (over time dimension per node or over node dimension). |
| | `num_heads` | `int` | `4` | `> 0` | Number of attention heads used in Transformer layers. |
| | `num_layers` | `int` | `2` | `> 0` | Number of internal layers inside each Transformer block. |
| | `dropout` | `float` | `0.3` | `[0.0, 1.0]` | Dropout rate used in Transformer layers. |
| | `max_sequence_length`| `int` | `5000` | `> 0` | Maximum sequence length for positional encoding. |

For example:
```python
from tmtgnn.config import (
    DiffusionConfig,
    GraphConfig,
    NormConfig,
    TMTGNNConfig,
    TransformerConfig,
)
from tmtgnn import TMTGNN


# Define required model parameters
num_nodes = 50
in_channels = 5
seq_length = 100
out_channels = 1
device = "cpu"

# Define optional model parameters
tmtgnn_config = TMTGNNConfig(
    hidden_dim=32,
    num_layers=3,
    skip_dim=64,
    head_dim=32,
    dropout=0.3,
)

transformer_config = TransformerConfig(
    num_heads=4,
    num_layers=2,
    dropout=0.3,
    max_sequence_length=100,
    mode="temporal",
)

graph_config = GraphConfig(
    learning_enabled=True,
    top_k=10,
    ema_alpha=0.99,
    sigmoid_alpha=1.0,
    noise_scale=0.01,
)

diffusion_config = DiffusionConfig(
    diffusion_steps=3,
    residual_alpha=0.5,
    projection_bias=True,
)

norm_config = NormConfig(
    eps=1e-5,
    elementwise_affine=True,
)

# Define T-MTGNN model
model = TMTGNN(
    num_nodes=num_nodes,
    in_channels=in_channels,
    seq_length=seq_length,
    out_channels=out_channels,
    device=device,
    diffusion_config=diffusion_config,
    graph_config=graph_config,
    norm_config=norm_config,
    tmtgnn_config=tmtgnn_config,
    transformer_config=transformer_config,
)
```

After having defined the model, train it. For example:
```python
import torch
from torch import nn


# Define hyperparameters
lr = 1e-3
num_epochs = 50
batch_size = 512

# Move model to device
model.to(device)

# Set model to training mode
model.train()

# Define optimizer and criterion for training
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Define 10k samples of dummy data
X = torch.randn(10000, in_channels, num_nodes, seq_length)
y = torch.randn(10000, num_nodes)
A = torch.randn(num_nodes, num_nodes)

# Create dataset and dataloader
dataset = torch.utils.data.TensorDataset(X, y, A)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Training loop
for epoch in range(num_epochs):
    # Single-epoch training
    for x, y_batch, adj in dataloader:
        # Move data to device
        x = x.to(device)
        y_batch = y_batch.to(device)
        adj = adj.to(device)

        optimizer.zero_grad()
    
        # Forward pass
        y_pred = model(x, adj=adj)
    
        # Calculate loss
        loss = criterion(y_pred, y_batch)
    
        # Backward pass
        loss.backward()
        optimizer.step()
```

Finally, evaluate your model. For example:
```python
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# Set model to evaluation mode
model.eval()

# Define 10k samples of dummy data
X = torch.randn(10000, in_channels, num_nodes, seq_length)
y = torch.randn(10000, num_nodes)
A = torch.randn(num_nodes, num_nodes)

# Create dataset and dataloader
dataset = torch.utils.data.TensorDataset(X, y, A)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

all_preds = []
all_targets = []

with torch.no_grad():
    for x, y_batch, adj in dataloader:
        # Move data to device
        x = x.to(device)
        y_batch = y_batch.to(device)
        adj = adj.to(device)

        # Forward pass
        y_pred = model(x, adj=adj)

        all_preds.append(y_pred.cpu())
        all_targets.append(y_batch.cpu())

# Concatenate results
y_pred = torch.cat(all_preds, dim=0).numpy()
y_true = torch.cat(all_targets, dim=0).numpy()

# Compute metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)

# Display metrics
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
```

<p align="right"><a href="#readme-top">Top ↑</a></p>

## License

Distributed under the [MIT License](https://github.com/michelepatella/tmtgnn/blob/main/LICENSE).

<p align="right"><a href="#readme-top">Top ↑</a></p>

## Acknowledgments

This model architecture is inspired by MTGNN ([Wu et al., 2020](https://arxiv.org/abs/2005.11650)). Parts of this codebase are adapted from their [official repository](https://github.com/nnzhan/MTGNN).

<p align="right"><a href="#readme-top">Top ↑</a></p>
