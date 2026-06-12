import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAT_PATH = ROOT / "models" / "fedostc_gat.py"

FORBIDDEN_GAT_STRINGS = [
    "GATConv",
    "torch_geometric",
    "client_oa",
    "selected",
    "eval_drift",
    "kl_threshold",
    "x_attr",
    "y_attr",
]


def _torch_or_skip():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - only used on missing optional dep
        raise unittest.SkipTest(f"PyTorch is not importable: {exc}") from exc
    return torch


def _load_gat_class():
    spec = importlib.util.spec_from_file_location("fedostc_gat_under_test", GAT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FedOSTCGAT


def _manual_eq_8_to_10(torch, hidden, edge_pairs):
    import torch.nn.functional as F

    node_count = hidden.size(1)
    edges = {(int(source), int(target)) for source, target in edge_pairs}
    edges.update((node, node) for node in range(node_count))

    refined = []
    for target in range(node_count):
        sources = sorted(source for source, dst in edges if dst == target)
        source_hidden = hidden[:, sources, :]
        target_hidden = hidden[:, target : target + 1, :].expand_as(source_hidden)
        projected = torch.cat((target_hidden, source_hidden), dim=-1).mean(dim=-1)
        scores = F.leaky_relu(projected, negative_slope=0.2)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        refined.append(torch.sigmoid((weights * source_hidden).sum(dim=1)))
    return torch.stack(refined, dim=1)


class FedOSTCPhase2GATTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = _torch_or_skip()
        cls.FedOSTCGAT = _load_gat_class()

    def test_forward_returns_batch_node_hidden_shape(self):
        gat = self.FedOSTCGAT()
        hidden = self.torch.randn(4, 5, 6)
        edge_index = self.torch.tensor([[0, 2, 3], [1, 1, 4]])

        out = gat(hidden, edge_index=edge_index)

        self.assertEqual(list(out.shape), [4, 5, 6])

    def test_directed_target_local_softmax_matches_manual_equations(self):
        gat = self.FedOSTCGAT()
        hidden = self.torch.tensor(
            [
                [
                    [0.2, -0.4],
                    [0.5, 0.1],
                    [-0.3, 0.8],
                ]
            ],
            dtype=self.torch.float32,
        )
        edge_pairs = [(0, 1), (2, 1), (1, 0)]
        edge_index = self.torch.tensor(edge_pairs, dtype=self.torch.long).t()

        expected = _manual_eq_8_to_10(self.torch, hidden, edge_pairs)
        actual = gat(hidden, edge_index=edge_index)

        self.assertTrue(self.torch.allclose(actual, expected, atol=1e-7))

    def test_adj_mx_and_equivalent_edge_index_match(self):
        gat = self.FedOSTCGAT()
        hidden = self.torch.randn(2, 4, 3)
        edge_index = self.torch.tensor([[0, 2, 3], [1, 1, 0]])
        adj_mx = self.torch.zeros(4, 4)
        adj_mx[edge_index[0], edge_index[1]] = 1.0

        from_edges = gat(hidden, edge_index=edge_index)
        from_adj = gat(hidden, adj_mx=adj_mx)

        self.assertTrue(self.torch.allclose(from_edges, from_adj, atol=1e-7))

    def test_self_loop_covers_isolated_node(self):
        gat = self.FedOSTCGAT()
        hidden = self.torch.randn(2, 3, 4)
        edge_index = self.torch.tensor([[0], [1]])

        out = gat(hidden, edge_index=edge_index)

        self.assertTrue(self.torch.allclose(out[:, 2], self.torch.sigmoid(hidden[:, 2])))

    def test_gradient_flows_to_hidden_input(self):
        gat = self.FedOSTCGAT()
        hidden = self.torch.randn(2, 3, 4, requires_grad=True)
        edge_index = self.torch.tensor([[0, 1, 2], [1, 2, 0]])

        loss = gat(hidden, edge_index=edge_index).sum()
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertGreater(hidden.grad.abs().sum().item(), 0.0)

    def test_has_no_parameter_or_buffer_state(self):
        gat = self.FedOSTCGAT()

        self.assertEqual(list(gat.parameters()), [])
        self.assertEqual(dict(gat.state_dict()), {})

    def test_phase2_gat_source_avoids_wrong_paths(self):
        source = GAT_PATH.read_text(encoding="utf-8")

        for needle in FORBIDDEN_GAT_STRINGS:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, source)


if __name__ == "__main__":
    unittest.main()
