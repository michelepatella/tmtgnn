"""tests/test_layers.py

Tests for layer modules:
    - ChannelProjection
    - LayerNorm
"""

import pytest
import torch
from layers import ChannelProjection, LayerNorm


# =========================================================
# ChannelProjection
# =========================================================

class TestChannelProjection:
    def test_init_default_bias(self):
        layer = ChannelProjection(in_channels=8, out_channels=16)
        assert layer.projection.bias is not None

    def test_init_no_bias(self):
        layer = ChannelProjection(in_channels=8, out_channels=16, bias=False)
        assert layer.projection.bias is None

    def test_output_shape(self):
        layer = ChannelProjection(in_channels=8, out_channels=16)
        x = torch.randn(2, 8, 10, 12)
        out = layer(x)
        assert out.shape == (2, 16, 10, 12)

    def test_output_shape_same_channels(self):
        layer = ChannelProjection(in_channels=8, out_channels=8)
        x = torch.randn(3, 8, 5, 7)
        out = layer(x)
        assert out.shape == (3, 8, 5, 7)

    def test_output_shape_reduce_channels(self):
        layer = ChannelProjection(in_channels=32, out_channels=4)
        x = torch.randn(1, 32, 6, 6)
        out = layer(x)
        assert out.shape == (1, 4, 6, 6)

    def test_output_is_float_tensor(self):
        layer = ChannelProjection(in_channels=4, out_channels=8)
        x = torch.randn(2, 4, 3, 3)
        out = layer(x)
        assert out.dtype == torch.float32

    def test_1x1_conv_kernel(self):
        layer = ChannelProjection(in_channels=4, out_channels=8)
        assert layer.projection.kernel_size == (1, 1)
        assert layer.projection.stride == (1, 1)
        assert layer.projection.padding == (0, 0)

    def test_gradients_flow(self):
        layer = ChannelProjection(in_channels=4, out_channels=8)
        x = torch.randn(2, 4, 3, 3, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_batch_size_one(self):
        layer = ChannelProjection(in_channels=4, out_channels=8)
        x = torch.randn(1, 4, 5, 5)
        out = layer(x)
        assert out.shape == (1, 8, 5, 5)

    def test_single_node_single_timestep(self):
        layer = ChannelProjection(in_channels=4, out_channels=8)
        x = torch.randn(2, 4, 1, 1)
        out = layer(x)
        assert out.shape == (2, 8, 1, 1)


# =========================================================
# LayerNorm
# =========================================================

class TestLayerNorm:
    def _make_idx(self, v):
        return torch.arange(v)

    def test_init_int_shape(self):
        norm = LayerNorm(normalized_shape=16)
        assert norm.normalized_shape == (16,)

    def test_init_tuple_shape(self):
        norm = LayerNorm(normalized_shape=(8, 10, 12))
        assert norm.normalized_shape == (8, 10, 12)

    def test_init_elementwise_affine_true_has_params(self):
        norm = LayerNorm(normalized_shape=(8, 5, 12))
        assert norm.weight is not None
        assert norm.bias is not None

    def test_init_elementwise_affine_false_no_params(self):
        norm = LayerNorm(normalized_shape=(8, 5, 12), elementwise_affine=False)
        assert norm.weight is None
        assert norm.bias is None

    def test_weight_initialized_to_ones(self):
        norm = LayerNorm(normalized_shape=(8, 5, 12))
        assert torch.all(norm.weight == 1.0)

    def test_bias_initialized_to_zeros(self):
        norm = LayerNorm(normalized_shape=(8, 5, 12))
        assert torch.all(norm.bias == 0.0)

    def test_output_shape_with_affine(self):
        b, c, v, l = 2, 8, 5, 12
        norm = LayerNorm(normalized_shape=(c, v, l))
        x = torch.randn(b, c, v, l)
        idx = self._make_idx(v)
        out = norm(x, idx)
        assert out.shape == (b, c, v, l)

    def test_output_shape_without_affine(self):
        b, c, v, l = 2, 8, 5, 12
        norm = LayerNorm(normalized_shape=(c, v, l), elementwise_affine=False)
        x = torch.randn(b, c, v, l)
        idx = self._make_idx(v)
        out = norm(x, idx)
        assert out.shape == (b, c, v, l)

    def test_output_normalized_with_affine(self):
        b, c, v, l = 4, 8, 5, 12
        norm = LayerNorm(normalized_shape=(c, v, l))
        x = torch.randn(b, c, v, l) * 10 + 5
        idx = self._make_idx(v)
        out = norm(x, idx)
        # With default weight=1 and bias=0, output should have ~zero mean and ~unit variance
        assert out.mean().abs() < 0.1
        assert (out.std() - 1.0).abs() < 0.2

    def test_output_normalized_without_affine(self):
        b, c, v, l = 4, 8, 5, 12
        norm = LayerNorm(normalized_shape=(c, v, l), elementwise_affine=False)
        x = torch.randn(b, c, v, l) * 10 + 5
        idx = self._make_idx(v)
        out = norm(x, idx)
        assert out.mean().abs() < 0.1

    def test_node_index_subset(self):
        c, v_total, l = 8, 10, 12
        norm = LayerNorm(normalized_shape=(c, v_total, l))
        v_sub = 5
        x = torch.randn(2, c, v_sub, l)
        idx = torch.arange(v_sub)
        out = norm(x, idx)
        assert out.shape == (2, c, v_sub, l)

    def test_gradients_flow_with_affine(self):
        b, c, v, l = 2, 8, 5, 12
        norm = LayerNorm(normalized_shape=(c, v, l))
        x = torch.randn(b, c, v, l, requires_grad=True)
        idx = self._make_idx(v)
        out = norm(x, idx)
        out.sum().backward()
        assert x.grad is not None

    def test_gradients_flow_without_affine(self):
        b, c, v, l = 2, 8, 5, 12
        norm = LayerNorm(normalized_shape=(c, v, l), elementwise_affine=False)
        x = torch.randn(b, c, v, l, requires_grad=True)
        idx = self._make_idx(v)
        out = norm(x, idx)
        out.sum().backward()
        assert x.grad is not None

    def test_eps_custom_value(self):
        norm = LayerNorm(normalized_shape=(8, 5, 12), eps=1e-3)
        assert norm.eps == 1e-3

    def test_batch_size_one(self):
        c, v, l = 8, 5, 12
        norm = LayerNorm(normalized_shape=(c, v, l))
        x = torch.randn(1, c, v, l)
        idx = self._make_idx(v)
        out = norm(x, idx)
        assert out.shape == (1, c, v, l)
