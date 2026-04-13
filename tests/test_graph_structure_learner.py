"""tests/test_graph_structure_learner.py

Tests for GraphStructureLearner module.
"""

import pytest
import torch
from graph import GraphStructureLearner


# =========================================================
# Fixtures
# =========================================================

@pytest.fixture
def learner():
    return GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16)


@pytest.fixture
def node_repr(learner):
    return torch.randn(learner.num_nodes, learner.hidden_dim)


# =========================================================
# Tests
# =========================================================

class TestGraphStructureLearnerInit:
    def test_stores_num_nodes(self):
        gsl = GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16)
        assert gsl.num_nodes == 10

    def test_stores_top_k(self):
        gsl = GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16)
        assert gsl.top_k == 3

    def test_stores_hidden_dim(self):
        gsl = GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16)
        assert gsl.hidden_dim == 16

    def test_default_alpha(self):
        gsl = GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16)
        assert gsl.alpha == 3.0

    def test_default_noise_scale(self):
        gsl = GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16)
        assert gsl.noise_scale == 0.01

    def test_no_node_features_by_default(self):
        gsl = GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16)
        assert gsl.node_features is None

    def test_node_features_stored_as_buffer(self):
        feat = torch.randn(10, 8)
        gsl = GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16, node_features=feat)
        assert gsl.node_features is not None
        assert gsl.node_features.shape == (10, 8)

    def test_src_dst_encoders_exist(self):
        gsl = GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16)
        assert isinstance(gsl.src_encoder, torch.nn.Linear)
        assert isinstance(gsl.dst_encoder, torch.nn.Linear)

    def test_encoder_dimensions(self):
        gsl = GraphStructureLearner(num_nodes=10, top_k=3, hidden_dim=16)
        assert gsl.src_encoder.in_features == 16
        assert gsl.src_encoder.out_features == 16
        assert gsl.dst_encoder.in_features == 16
        assert gsl.dst_encoder.out_features == 16

    def test_custom_alpha_and_noise_scale(self):
        gsl = GraphStructureLearner(
            num_nodes=10, top_k=3, hidden_dim=16, alpha=5.0, noise_scale=0.1
        )
        assert gsl.alpha == 5.0
        assert gsl.noise_scale == 0.1


class TestGraphStructureLearnerForward:
    def test_output_shape(self, learner, node_repr):
        adj = learner(node_repr)
        assert adj.shape == (learner.num_nodes, learner.num_nodes)

    def test_output_is_contiguous(self, learner, node_repr):
        adj = learner(node_repr)
        assert adj.is_contiguous()

    def test_output_values_non_negative(self, learner, node_repr):
        learner.eval()
        adj = learner(node_repr)
        assert (adj >= 0.0).all()

    def test_output_values_at_most_one(self, learner, node_repr):
        learner.eval()
        adj = learner(node_repr)
        assert (adj <= 1.0).all()

    def test_top_k_sparsity_eval(self):
        n, k = 10, 3
        gsl = GraphStructureLearner(num_nodes=n, top_k=k, hidden_dim=16)
        gsl.eval()
        node_repr = torch.randn(n, 16)
        adj = gsl(node_repr)
        # Each row should have at most top_k non-zero entries
        nnz_per_row = (adj > 0).sum(dim=1)
        assert (nnz_per_row <= k).all()

    def test_top_k_sparsity_train(self):
        n, k = 10, 3
        gsl = GraphStructureLearner(num_nodes=n, top_k=k, hidden_dim=16)
        gsl.train()
        node_repr = torch.randn(n, 16)
        adj = gsl(node_repr)
        nnz_per_row = (adj > 0).sum(dim=1)
        assert (nnz_per_row <= k).all()

    def test_exact_top_k_entries_per_row_eval(self):
        n, k = 8, 4
        gsl = GraphStructureLearner(num_nodes=n, top_k=k, hidden_dim=32)
        gsl.eval()
        torch.manual_seed(0)
        node_repr = torch.randn(n, 32)
        adj = gsl(node_repr)
        # Each row should have exactly top_k non-zero entries in eval mode
        nnz_per_row = (adj > 0).sum(dim=1)
        assert (nnz_per_row == k).all()

    def test_adjacency_is_asymmetric(self):
        n = 8
        gsl = GraphStructureLearner(num_nodes=n, top_k=3, hidden_dim=16)
        gsl.eval()
        torch.manual_seed(42)
        node_repr = torch.randn(n, 16)
        adj = gsl(node_repr)
        # Asymmetry: adj and adj.T should not be identical for a directed graph
        assert not torch.allclose(adj, adj.t())

    def test_no_noise_in_eval_mode(self):
        n = 6
        gsl = GraphStructureLearner(num_nodes=n, top_k=3, hidden_dim=16, noise_scale=0.5)
        gsl.eval()
        node_repr = torch.randn(n, 16)
        adj1 = gsl(node_repr)
        adj2 = gsl(node_repr)
        assert torch.allclose(adj1, adj2)

    def test_noise_can_affect_train_mode(self):
        n = 10
        gsl = GraphStructureLearner(num_nodes=n, top_k=3, hidden_dim=16, noise_scale=10.0)
        gsl.train()
        node_repr = torch.randn(n, 16)
        # With large noise, two forward passes in training may differ (different top-k selection)
        results = set()
        for _ in range(20):
            adj = gsl(node_repr)
            results.add(adj.sum().item())
        # Should have seen at least two distinct results due to randomness
        assert len(results) > 1

    def test_top_k_capped_at_num_nodes(self):
        # top_k larger than num_nodes should be capped
        n = 4
        gsl = GraphStructureLearner(num_nodes=n, top_k=100, hidden_dim=8)
        gsl.eval()
        node_repr = torch.randn(n, 8)
        adj = gsl(node_repr)
        assert adj.shape == (n, n)

    def test_gradients_flow_through_adjacency(self, learner):
        node_repr = torch.randn(learner.num_nodes, learner.hidden_dim, requires_grad=True)
        adj = learner(node_repr)
        adj.sum().backward()
        assert node_repr.grad is not None

    def test_different_node_repr_different_adj(self, learner):
        learner.eval()
        repr1 = torch.randn(learner.num_nodes, learner.hidden_dim)
        repr2 = torch.randn(learner.num_nodes, learner.hidden_dim)
        adj1 = learner(repr1)
        adj2 = learner(repr2)
        assert not torch.allclose(adj1, adj2)

    def test_zero_noise_scale_deterministic(self):
        n = 6
        gsl = GraphStructureLearner(num_nodes=n, top_k=3, hidden_dim=16, noise_scale=0.0)
        gsl.train()
        node_repr = torch.randn(n, 16)
        adj1 = gsl(node_repr)
        adj2 = gsl(node_repr)
        assert torch.allclose(adj1, adj2)
