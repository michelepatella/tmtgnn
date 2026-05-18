<a id="readme-top"></a>

<br/>
<div align="center">
  <h1 align="center">T-MTGNN</h1>
  <p align="center">
    PyTorch implementation of a deep learning model combining <br>
    Transformers and Graph Neural Networks for multivariate time series forecasting.
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

<p align="right"><a href="#readme-top">Top ↑</a></p>

## License

Distributed under the [MIT License](https://github.com/michelepatella/tmtgnn/blob/main/LICENSE).

<p align="right"><a href="#readme-top">Top ↑</a></p>

## Acknowledgments

This model architecture is inspired by MTGNN ([Wu et al., 2020](https://arxiv.org/abs/2005.11650)). Parts of this codebase are adapted from their [official repository](https://github.com/nnzhan/MTGNN).

<p align="right"><a href="#readme-top">Top ↑</a></p>
