"""tests/test_modules.py

Tests for spatial and temporal modules:
    - GraphConv
    - GraphDiffusion
    - Transformer
"""

import pytest
import torch
from modules import GraphConv, GraphDiffusion, Transformer


# =========================================================
# GraphConv
# =========================================================

class TestGraphConv:
    def test_no_learnable_params(self):
        conv = GraphConv()
        assert sum(p.numel() for p in conv.parameters()) == 0

    def test_output_shape_2d_adj(self):
        conv = GraphConv()
        b, c, v, l = 2, 8, 6, 12
        x = torch.randn(b, c, v, l)
        adj = torch.randn(v, v)
        out = conv(x, adj)
        assert out.shape == (b, c, v, l)

    def test_output_shape_3d_adj(self):
        conv = GraphConv()
        b, c, v, l = 2, 8, 6, 12
        x = torch.randn(b, c, v, l)
        adj = torch.randn(b, v, v)
        out = conv(x, adj)
        assert out.shape == (b, c, v, l)

    def test_2d_adj_gets_unsqueezed(self):
        conv = GraphConv()
        b, c, v, l = 3, 4, 5, 7
        x = torch.randn(b, c, v, l)
        adj_2d = torch.randn(v, v)
        adj_3d = adj_2d.unsqueeze(0)
        out_2d = conv(x, adj_2d)
        out_3d = conv(x, adj_3d)
        assert torch.allclose(out_2d, out_3d)

    def test_output_is_contiguous(self):
        conv = GraphConv()
        x = torch.randn(2, 4, 5, 6)
        adj = torch.randn(5, 5)
        out = conv(x, adj)
        assert out.is_contiguous()

    def test_identity_adj_returns_input(self):
        conv = GraphConv()
        b, c, v, l = 2, 4, 5, 6
        x = torch.randn(b, c, v, l)
        adj = torch.eye(v)
        out = conv(x, adj)
        assert torch.allclose(out, x, atol=1e-5)

    def test_zero_adj_returns_zero(self):
        conv = GraphConv()
        b, c, v, l = 2, 4, 5, 6
        x = torch.randn(b, c, v, l)
        adj = torch.zeros(v, v)
        out = conv(x, adj)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_different_source_and_target_nodes(self):
        conv = GraphConv()
        b, c, v, w, l = 2, 4, 6, 8, 5
        x = torch.randn(b, c, v, l)
        adj = torch.randn(b, v, w)
        out = conv(x, adj)
        assert out.shape == (b, c, w, l)

    def test_gradients_flow(self):
        conv = GraphConv()
        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        adj = torch.randn(5, 5)
        out = conv(x, adj)
        out.sum().backward()
        assert x.grad is not None

    def test_batch_specific_adj_uses_different_adj_per_sample(self):
        conv = GraphConv()
        b, c, v, l = 2, 4, 3, 5
        x = torch.randn(b, c, v, l)
        # Use very different adj per batch sample
        adj = torch.zeros(b, v, v)
        adj[0] = torch.eye(v)
        adj[1] = torch.ones(v, v)
        out = conv(x, adj)
        # Sample 0 should be close to x[0], sample 1 should be different
        assert not torch.allclose(out[0], out[1])


# =========================================================
# GraphDiffusion
# =========================================================

class TestGraphDiffusion:
    def test_output_shape(self):
        b, c, v, l = 2, 8, 6, 12
        diffusion = GraphDiffusion(
            in_channels=c, out_channels=c, diffusion_steps=2, residual_alpha=0.1
        )
        x = torch.randn(b, c, v, l)
        adj = torch.randn(v, v).abs()
        out = diffusion(x, adj)
        assert out.shape == (b, c, v, l)

    def test_output_shape_different_out_channels(self):
        b, c, c_out, v, l = 2, 8, 16, 6, 12
        diffusion = GraphDiffusion(
            in_channels=c, out_channels=c_out, diffusion_steps=2, residual_alpha=0.1
        )
        x = torch.randn(b, c, v, l)
        adj = torch.randn(v, v).abs()
        out = diffusion(x, adj)
        assert out.shape == (b, c_out, v, l)

    def test_single_diffusion_step(self):
        b, c, v, l = 2, 8, 6, 12
        diffusion = GraphDiffusion(
            in_channels=c, out_channels=c, diffusion_steps=1, residual_alpha=0.1
        )
        x = torch.randn(b, c, v, l)
        adj = torch.randn(v, v).abs()
        out = diffusion(x, adj)
        assert out.shape == (b, c, v, l)

    def test_many_diffusion_steps(self):
        b, c, v, l = 2, 4, 5, 8
        diffusion = GraphDiffusion(
            in_channels=c, out_channels=c, diffusion_steps=5, residual_alpha=0.2
        )
        x = torch.randn(b, c, v, l)
        adj = torch.randn(v, v).abs()
        out = diffusion(x, adj)
        assert out.shape == (b, c, v, l)

    def test_3d_adj_accepted(self):
        b, c, v, l = 2, 8, 6, 12
        diffusion = GraphDiffusion(
            in_channels=c, out_channels=c, diffusion_steps=2, residual_alpha=0.1
        )
        x = torch.randn(b, c, v, l)
        adj = torch.randn(b, v, v).abs()
        out = diffusion(x, adj)
        assert out.shape == (b, c, v, l)

    def test_no_bias(self):
        diffusion = GraphDiffusion(
            in_channels=8, out_channels=8, diffusion_steps=2,
            residual_alpha=0.1, projection_bias=False
        )
        assert diffusion.projection.projection.bias is None

    def test_with_bias(self):
        diffusion = GraphDiffusion(
            in_channels=8, out_channels=8, diffusion_steps=2,
            residual_alpha=0.1, projection_bias=True
        )
        assert diffusion.projection.projection.bias is not None

    def test_residual_alpha_one_returns_input_to_diffusion(self):
        # With alpha=1.0, hidden = 1.0*x + 0.0*agg = x at each step
        b, c, v, l = 2, 4, 5, 6
        diffusion = GraphDiffusion(
            in_channels=c, out_channels=c, diffusion_steps=2, residual_alpha=1.0
        )
        x = torch.randn(b, c, v, l)
        adj = torch.randn(v, v).abs()
        out = diffusion(x, adj)
        # All diffusion states are copies of x, concatenated then projected
        # The result is a linear projection of (diffusion_steps+1) copies of x
        assert out.shape == (b, c, v, l)

    def test_gradients_flow(self):
        diffusion = GraphDiffusion(
            in_channels=8, out_channels=8, diffusion_steps=2, residual_alpha=0.1
        )
        x = torch.randn(2, 8, 6, 12, requires_grad=True)
        adj = torch.randn(6, 6).abs()
        out = diffusion(x, adj)
        out.sum().backward()
        assert x.grad is not None

    def test_output_is_contiguous(self):
        diffusion = GraphDiffusion(
            in_channels=8, out_channels=8, diffusion_steps=2, residual_alpha=0.1
        )
        x = torch.randn(2, 8, 6, 12)
        adj = torch.randn(6, 6).abs()
        out = diffusion(x, adj)
        assert out.is_contiguous()


# =========================================================
# Transformer
# =========================================================

class TestTransformer:
    def test_output_shape_same_channels(self):
        b, c, v, l = 2, 8, 6, 12
        transformer = Transformer(in_channels=c, out_channels=c, num_head=4, num_layers=2)
        x = torch.randn(b, c, v, l)
        out = transformer(x)
        assert out.shape == (b, c, v, l)

    def test_output_shape_different_channels(self):
        b, c_in, c_out, v, l = 2, 8, 16, 6, 12
        transformer = Transformer(
            in_channels=c_in, out_channels=c_out, num_head=4, num_layers=2
        )
        x = torch.randn(b, c_in, v, l)
        out = transformer(x)
        assert out.shape == (b, c_out, v, l)

    def test_identity_projection_when_same_channels(self):
        transformer = Transformer(in_channels=8, out_channels=8)
        assert isinstance(transformer.projection, torch.nn.Identity)

    def test_linear_projection_when_different_channels(self):
        transformer = Transformer(in_channels=8, out_channels=16)
        assert isinstance(transformer.projection, torch.nn.Linear)

    def test_single_layer(self):
        b, c, v, l = 2, 8, 4, 10
        transformer = Transformer(in_channels=c, out_channels=c, num_head=4, num_layers=1)
        x = torch.randn(b, c, v, l)
        out = transformer(x)
        assert out.shape == (b, c, v, l)

    def test_multiple_layers(self):
        b, c, v, l = 2, 8, 4, 10
        transformer = Transformer(in_channels=c, out_channels=c, num_head=4, num_layers=4)
        x = torch.randn(b, c, v, l)
        out = transformer(x)
        assert out.shape == (b, c, v, l)

    def test_output_is_contiguous(self):
        transformer = Transformer(in_channels=8, out_channels=8, num_head=4, num_layers=2)
        x = torch.randn(2, 8, 5, 10)
        out = transformer(x)
        assert out.is_contiguous()

    def test_gradients_flow(self):
        transformer = Transformer(in_channels=8, out_channels=8, num_head=4, num_layers=2)
        x = torch.randn(2, 8, 6, 12, requires_grad=True)
        out = transformer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_batch_size_one(self):
        transformer = Transformer(in_channels=8, out_channels=8, num_head=4, num_layers=2)
        x = torch.randn(1, 8, 5, 10)
        out = transformer(x)
        assert out.shape == (1, 8, 5, 10)

    def test_single_node(self):
        transformer = Transformer(in_channels=8, out_channels=8, num_head=4, num_layers=2)
        x = torch.randn(2, 8, 1, 10)
        out = transformer(x)
        assert out.shape == (2, 8, 1, 10)

    def test_single_timestep(self):
        transformer = Transformer(in_channels=8, out_channels=8, num_head=4, num_layers=2)
        x = torch.randn(2, 8, 5, 1)
        out = transformer(x)
        assert out.shape == (2, 8, 5, 1)

    def test_num_heads_stored(self):
        transformer = Transformer(in_channels=8, out_channels=8, num_head=2, num_layers=2)
        assert transformer.transformer.layers[0].self_attn.num_heads == 2

    def test_different_nodes_processed_independently(self):
        # Since there is no cross-node interaction in Transformer,
        # perturbing one node should only affect that node's output
        transformer = Transformer(in_channels=8, out_channels=8, num_head=4, num_layers=2)
        transformer.eval()
        x = torch.randn(1, 8, 5, 10)
        x_perturbed = x.clone()
        x_perturbed[0, :, 0, :] += 100.0  # Perturb only node 0

        out = transformer(x)
        out_perturbed = transformer(x_perturbed)

        # Nodes 1..4 should be unaffected
        assert torch.allclose(out[0, :, 1:, :], out_perturbed[0, :, 1:, :], atol=1e-5)
        # Node 0 should be affected
        assert not torch.allclose(out[0, :, 0, :], out_perturbed[0, :, 0, :])
