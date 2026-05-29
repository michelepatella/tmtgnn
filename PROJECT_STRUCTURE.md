```text
tmtgnn/
├── __init__.py                          # Package initialization exporting TMTGNN model
├── config/
│   ├── __init__.py                      # Configuration module exports
│   ├── tmtgnn_config.py                 # Model configuration
│   ├── transformer_config.py            # Transformer configuration
│   ├── graph_config.py                  # Graph structure learning configuration
│   ├── diffusion_config.py              # Graph diffusion configuration
│   └── norm_config.py                   # Normalization layer configuration
├── graph/
│   ├── __init__.py                      # Graph module exports
│   └── graph_structure_learner.py       # Adaptive graph structure learning from data
├── layers/
│   ├── __init__.py                      # Layers module exports
│   ├── normalization/
│   │   ├── __init__.py                  # Normalization submodule exports
│   │   └── layer_norm.py                # Custom layer normalization
│   └── projection/
│       ├── __init__.py                  # Projection submodule exports
│       └── channel_projection.py        # Feature dimension projection layer
├── models/
│   ├── __init__.py                      # Models module exports
│   └── tmtgnn.py                        # T-MTGNN model
└── modules/
    ├── __init__.py                      # Modules exports
    ├── spatial/
    │   ├── __init__.py                  # Spatial submodule exports
    │   ├── graph_conv.py                # Graph convolution for message passing
    │   └── graph_diffusion.py           # Graph diffusion for spatial information propagation
    └── temporal/
        ├── __init__.py                  # Temporal submodule exports
        ├── transformer.py               # Transformer for temporal feature extraction
        └── positional_encoding.py       # Sinusoidal positional encoding
```
