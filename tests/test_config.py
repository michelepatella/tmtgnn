"""tests/test_config.py

Tests for all configuration dataclasses:
    - DiffusionConfig
    - GraphConfig
    - NormConfig
    - TMTGNNConfig
    - TransformerConfig
"""

import pytest
import torch
from config import DiffusionConfig, GraphConfig, NormConfig, TMTGNNConfig, TransformerConfig


# =========================================================
# DiffusionConfig
# =========================================================

class TestDiffusionConfig:
    def test_defaults(self):
        cfg = DiffusionConfig()
        assert cfg.gcn_depth == 2
        assert cfg.residual_alpha == 0.05
        assert cfg.projection_bias is True

    def test_custom_values(self):
        cfg = DiffusionConfig(gcn_depth=4, residual_alpha=0.1, projection_bias=False)
        assert cfg.gcn_depth == 4
        assert cfg.residual_alpha == 0.1
        assert cfg.projection_bias is False

    def test_gcn_depth_zero_raises(self):
        with pytest.raises(AssertionError):
            DiffusionConfig(gcn_depth=0)

    def test_gcn_depth_negative_raises(self):
        with pytest.raises(AssertionError):
            DiffusionConfig(gcn_depth=-1)

    def test_gcn_depth_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            DiffusionConfig(gcn_depth=2.0)

    def test_residual_alpha_below_zero_raises(self):
        with pytest.raises(AssertionError):
            DiffusionConfig(residual_alpha=-0.1)

    def test_residual_alpha_above_one_raises(self):
        with pytest.raises(AssertionError):
            DiffusionConfig(residual_alpha=1.1)

    def test_residual_alpha_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            DiffusionConfig(residual_alpha=0)

    def test_projection_bias_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            DiffusionConfig(projection_bias=1)

    def test_residual_alpha_boundary_zero(self):
        cfg = DiffusionConfig(residual_alpha=0.0)
        assert cfg.residual_alpha == 0.0

    def test_residual_alpha_boundary_one(self):
        cfg = DiffusionConfig(residual_alpha=1.0)
        assert cfg.residual_alpha == 1.0


# =========================================================
# GraphConfig
# =========================================================

class TestGraphConfig:
    def test_defaults(self):
        cfg = GraphConfig()
        assert cfg.top_k == 20
        assert cfg.alpha == 3.0
        assert cfg.noise_scale == 0.01
        assert cfg.node_features is None
        assert cfg.ema_alpha == 0.8

    def test_custom_values(self):
        cfg = GraphConfig(top_k=5, alpha=1.0, noise_scale=0.0, ema_alpha=0.5)
        assert cfg.top_k == 5
        assert cfg.alpha == 1.0
        assert cfg.noise_scale == 0.0
        assert cfg.ema_alpha == 0.5

    def test_top_k_zero_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(top_k=-1)

    def test_top_k_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(top_k=5.0)

    def test_alpha_zero_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(alpha=0.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(alpha=-1.0)

    def test_alpha_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(alpha=3)

    def test_noise_scale_negative_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(noise_scale=-0.01)

    def test_noise_scale_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(noise_scale=0)

    def test_node_features_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(node_features=[[1.0, 2.0], [3.0, 4.0]])

    def test_node_features_wrong_dim_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(node_features=torch.randn(10))

    def test_node_features_valid_tensor(self):
        feat = torch.randn(10, 4)
        cfg = GraphConfig(node_features=feat)
        assert cfg.node_features is feat

    def test_ema_alpha_below_zero_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(ema_alpha=-0.1)

    def test_ema_alpha_above_one_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(ema_alpha=1.1)

    def test_ema_alpha_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            GraphConfig(ema_alpha=1)

    def test_ema_alpha_boundary(self):
        cfg_zero = GraphConfig(ema_alpha=0.0)
        cfg_one = GraphConfig(ema_alpha=1.0)
        assert cfg_zero.ema_alpha == 0.0
        assert cfg_one.ema_alpha == 1.0


# =========================================================
# NormConfig
# =========================================================

class TestNormConfig:
    def test_defaults(self):
        cfg = NormConfig()
        assert cfg.eps == 1e-5
        assert cfg.affine is True

    def test_custom_values(self):
        cfg = NormConfig(eps=1e-4, affine=False)
        assert cfg.eps == 1e-4
        assert cfg.affine is False

    def test_eps_zero_raises(self):
        with pytest.raises(AssertionError):
            NormConfig(eps=0.0)

    def test_eps_negative_raises(self):
        with pytest.raises(AssertionError):
            NormConfig(eps=-1e-5)

    def test_eps_one_or_above_raises(self):
        with pytest.raises(AssertionError):
            NormConfig(eps=1.0)

    def test_eps_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            NormConfig(eps=1)

    def test_affine_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            NormConfig(affine=1)


# =========================================================
# TMTGNNConfig
# =========================================================

class TestTMTGNNConfig:
    def test_defaults(self):
        cfg = TMTGNNConfig()
        assert cfg.hidden_dim == 32
        assert cfg.skip_dim == 64
        assert cfg.head_dim == 128
        assert cfg.num_layers == 3
        assert cfg.dropout == 0.3

    def test_custom_values(self):
        cfg = TMTGNNConfig(hidden_dim=16, skip_dim=32, head_dim=64, num_layers=2, dropout=0.0)
        assert cfg.hidden_dim == 16
        assert cfg.skip_dim == 32
        assert cfg.head_dim == 64
        assert cfg.num_layers == 2
        assert cfg.dropout == 0.0

    def test_hidden_dim_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(hidden_dim=0)

    def test_hidden_dim_negative_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(hidden_dim=-1)

    def test_hidden_dim_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(hidden_dim=32.0)

    def test_skip_dim_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(skip_dim=0)

    def test_skip_dim_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(skip_dim=64.0)

    def test_skip_dim_less_than_hidden_dim_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(hidden_dim=32, skip_dim=16)

    def test_head_dim_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(head_dim=0)

    def test_head_dim_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(head_dim=128.0)

    def test_head_dim_less_than_hidden_dim_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(hidden_dim=32, head_dim=16)

    def test_num_layers_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(num_layers=0)

    def test_num_layers_negative_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(num_layers=-1)

    def test_num_layers_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(num_layers=3.0)

    def test_dropout_below_zero_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(dropout=-0.1)

    def test_dropout_above_one_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(dropout=1.1)

    def test_dropout_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            TMTGNNConfig(dropout=0)

    def test_dropout_boundary(self):
        cfg_zero = TMTGNNConfig(dropout=0.0)
        cfg_one = TMTGNNConfig(dropout=1.0)
        assert cfg_zero.dropout == 0.0
        assert cfg_one.dropout == 1.0


# =========================================================
# TransformerConfig
# =========================================================

class TestTransformerConfig:
    def test_defaults(self):
        cfg = TransformerConfig()
        assert cfg.num_heads == 4
        assert cfg.num_layers == 2
        assert cfg.dropout == 0.3

    def test_custom_values(self):
        cfg = TransformerConfig(num_heads=8, num_layers=4, dropout=0.1)
        assert cfg.num_heads == 8
        assert cfg.num_layers == 4
        assert cfg.dropout == 0.1

    def test_num_heads_zero_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(num_heads=0)

    def test_num_heads_negative_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(num_heads=-1)

    def test_num_heads_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(num_heads=4.0)

    def test_num_layers_zero_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(num_layers=0)

    def test_num_layers_negative_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(num_layers=-1)

    def test_num_layers_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(num_layers=2.0)

    def test_dropout_below_zero_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(dropout=-0.1)

    def test_dropout_above_one_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(dropout=1.1)

    def test_dropout_wrong_type_raises(self):
        with pytest.raises(AssertionError):
            TransformerConfig(dropout=0)

    def test_dropout_boundary(self):
        cfg_zero = TransformerConfig(dropout=0.0)
        cfg_one = TransformerConfig(dropout=1.0)
        assert cfg_zero.dropout == 0.0
        assert cfg_one.dropout == 1.0
