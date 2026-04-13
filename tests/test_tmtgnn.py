"""tests/test_tmtgnn.py

Tests for the TMTGNN end-to-end model.
"""

import pytest
import torch
from models import TMTGNN
from config import (
    DiffusionConfig,
    GraphConfig,
    NormConfig,
    TMTGNNConfig,
    TransformerConfig,
)


# =========================================================
# Helpers / Fixtures
# =========================================================

def make_model(
    num_nodes=25,
    in_channels=2,
    seq_length=12,
    out_channels=1,
    hidden_dim=8,
    num_layers=1,
    skip_dim=16,
    head_dim=16,
    top_k=5,
    num_heads=4,
    node_features=None,
):
    device = torch.device("cpu")
    tmtgnn_cfg = TMTGNNConfig(
        hidden_dim=hidden_dim,
        skip_dim=skip_dim,
        head_dim=head_dim,
        num_layers=num_layers,
        dropout=0.0,
    )
    transformer_cfg = TransformerConfig(num_heads=num_heads, num_layers=1, dropout=0.0)
    diffusion_cfg = DiffusionConfig(gcn_depth=2, residual_alpha=0.1)
    graph_cfg = GraphConfig(
        top_k=top_k,
        node_features=node_features,
    )
    norm_cfg = NormConfig()
    return TMTGNN(
        num_nodes=num_nodes,
        in_channels=in_channels,
        seq_length=seq_length,
        out_channels=out_channels,
        device=device,
        diffusion_config=diffusion_cfg,
        graph_config=graph_cfg,
        norm_config=norm_cfg,
        tmtgnn_config=tmtgnn_cfg,
        transformer_config=transformer_cfg,
    )


def make_input(batch=2, in_channels=2, num_nodes=25, seq_length=12):
    return torch.randn(batch, in_channels, num_nodes, seq_length)


# =========================================================
# Initialization tests
# =========================================================

class TestTMTGNNInit:
    def test_default_configs(self):
        device = torch.device("cpu")
        model = TMTGNN(
            num_nodes=25,
            in_channels=2,
            seq_length=12,
            out_channels=1,
            device=device,
            graph_config=GraphConfig(top_k=5),
        )
        assert model.num_nodes == 25
        assert model.seq_length == 12

    def test_num_layers_stored(self):
        model = make_model(num_layers=2)
        assert model.num_layers == 2

    def test_module_lists_have_correct_length(self):
        n_layers = 3
        model = make_model(num_layers=n_layers)
        assert len(model.temporal_layers) == n_layers
        assert len(model.diffusion_forward) == n_layers
        assert len(model.diffusion_backward) == n_layers
        assert len(model.skip_projections) == n_layers
        assert len(model.normalization_layers) == n_layers

    def test_node_embedding_shape(self):
        model = make_model(num_nodes=25, hidden_dim=8)
        assert model.node_embedding.num_embeddings == 25
        assert model.node_embedding.embedding_dim == 8

    def test_no_node_feat_proj_without_node_features(self):
        model = make_model()
        assert model.node_feat_proj is None

    def test_node_feat_proj_with_node_features(self):
        feat = torch.randn(25, 4)
        model = make_model(node_features=feat)
        assert model.node_feat_proj is not None

    def test_idx_buffer_registered(self):
        model = make_model(num_nodes=25)
        assert hasattr(model, "idx")
        assert model.idx.shape == (25,)

    def test_invalid_num_nodes_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNN(
                num_nodes=0,
                in_channels=2,
                seq_length=12,
                out_channels=1,
                device=torch.device("cpu"),
                graph_config=GraphConfig(top_k=5),
            )

    def test_invalid_num_nodes_float_raises(self):
        with pytest.raises((AssertionError, TypeError)):
            TMTGNN(
                num_nodes=5.0,
                in_channels=2,
                seq_length=12,
                out_channels=1,
                device=torch.device("cpu"),
                graph_config=GraphConfig(top_k=5),
            )

    def test_invalid_in_channels_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNN(
                num_nodes=25,
                in_channels=0,
                seq_length=12,
                out_channels=1,
                device=torch.device("cpu"),
                graph_config=GraphConfig(top_k=5),
            )

    def test_invalid_seq_length_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNN(
                num_nodes=25,
                in_channels=2,
                seq_length=0,
                out_channels=1,
                device=torch.device("cpu"),
                graph_config=GraphConfig(top_k=5),
            )

    def test_invalid_out_channels_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNN(
                num_nodes=25,
                in_channels=2,
                seq_length=12,
                out_channels=0,
                device=torch.device("cpu"),
                graph_config=GraphConfig(top_k=5),
            )

    def test_invalid_device_type_raises(self):
        with pytest.raises(AssertionError):
            TMTGNN(
                num_nodes=25,
                in_channels=2,
                seq_length=12,
                out_channels=1,
                device="cpu",
                graph_config=GraphConfig(top_k=5),
            )

    def test_top_k_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNN(
                num_nodes=25,
                in_channels=2,
                seq_length=12,
                out_channels=1,
                device=torch.device("cpu"),
                graph_config=GraphConfig(top_k=25),  # top_k must be < num_nodes (top_k == num_nodes fails)
            )

    def test_hidden_dim_not_divisible_by_num_heads_raises(self):
        with pytest.raises(AssertionError):
            make_model(hidden_dim=10, num_heads=4)  # 10 % 4 != 0

    def test_configs_are_deep_copied(self):
        diff_cfg = DiffusionConfig()
        model = make_model()
        # Configs passed in should not be mutated
        assert diff_cfg.gcn_depth == 2


# =========================================================
# Forward pass tests
# =========================================================

class TestTMTGNNForward:
    def test_output_shape_single_out_channel(self):
        model = make_model(num_nodes=25, in_channels=2, seq_length=12, out_channels=1)
        model.eval()
        x = make_input(batch=2, in_channels=2, num_nodes=25, seq_length=12)
        out = model(x)
        assert out.shape == (2, 25)  # squeezed because out_channels == 1

    def test_output_shape_multi_out_channel(self):
        model = make_model(num_nodes=25, in_channels=2, seq_length=12, out_channels=3)
        model.eval()
        x = make_input(batch=2, in_channels=2, num_nodes=25, seq_length=12)
        out = model(x)
        assert out.shape == (2, 3, 25)

    def test_output_shape_batch_size_one(self):
        model = make_model(num_nodes=25, in_channels=2, seq_length=12, out_channels=1)
        model.eval()
        x = make_input(batch=1, in_channels=2, num_nodes=25, seq_length=12)
        out = model(x)
        assert out.shape == (1, 25)

    def test_output_shape_with_explicit_idx(self):
        model = make_model(num_nodes=25, in_channels=2, seq_length=12, out_channels=1)
        model.eval()
        x = make_input(batch=2, in_channels=2, num_nodes=25, seq_length=12)
        idx = torch.arange(25)
        out = model(x, idx=idx)
        assert out.shape == (2, 25)

    def test_forward_deterministic_in_eval(self):
        model = make_model()
        model.eval()
        x = make_input()
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_forward_returns_float_tensor(self):
        model = make_model()
        model.eval()
        x = make_input()
        out = model(x)
        assert out.dtype == torch.float32

    def test_gradients_flow_to_input(self):
        model = make_model()
        x = torch.randn(2, 2, 25, 12, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_forward_train_mode(self):
        model = make_model()
        model.train()
        x = make_input()
        out = model(x)
        assert out.shape == (2, 25)

    def test_ema_state_accumulated_over_calls(self):
        model = make_model()
        model.eval()
        assert model.node_repr_prev is None
        x = make_input()
        model(x)
        assert model.node_repr_prev is not None

    def test_ema_state_detached_from_graph(self):
        model = make_model()
        model.eval()
        x = make_input()
        model(x)
        assert not model.node_repr_prev.requires_grad

    def test_second_forward_uses_ema(self):
        model = make_model()
        model.eval()
        x1 = make_input()
        model(x1)
        prev = model.node_repr_prev.clone()
        # Use a very different input so EMA blends two different representations
        x2 = make_input() * 100.0
        model(x2)
        # node_repr_prev should change because the second input is different
        assert not torch.allclose(prev, model.node_repr_prev)

    def test_output_with_node_features(self):
        feat = torch.randn(25, 4)
        model = make_model(node_features=feat)
        model.eval()
        x = make_input()
        out = model(x)
        assert out.shape == (2, 25)

    def test_multiple_layers(self):
        model = make_model(num_layers=3)
        model.eval()
        x = make_input()
        out = model(x)
        assert out.shape == (2, 25)

    def test_different_seq_lengths(self):
        for seq_len in [6, 12, 24]:
            model = make_model(seq_length=seq_len)
            model.eval()
            x = make_input(seq_length=seq_len)
            out = model(x)
            assert out.shape == (2, 25)

    def test_different_num_nodes(self):
        for n in [10, 30, 50]:
            model = make_model(num_nodes=n, top_k=5)
            model.eval()
            x = make_input(num_nodes=n)
            out = model(x)
            assert out.shape == (2, n)

    def test_parameters_update_during_training(self):
        model = make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = make_input()

        params_before = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
        }

        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        params_after = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
        }

        changed = any(
            not torch.allclose(params_before[n], params_after[n])
            for n in params_before
        )
        assert changed, "No parameters were updated during training step"

    def test_is_nn_module(self):
        model = make_model()
        assert isinstance(model, torch.nn.Module)
