import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FL_SERVER = ROOT / "fl-server"
if str(FL_SERVER) not in sys.path:
    sys.path.insert(0, str(FL_SERVER))

from Refol import REFOL


class ParameterizedGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x, edge_index):
        del edge_index
        return x * self.weight


class REFOLAggregationMemoryTests(unittest.TestCase):
    def test_aggregation_does_not_retain_autograd_graph_in_global_model(self):
        server = REFOL(SimpleNamespace())
        server.device = torch.device("cpu")
        server.gcn = ParameterizedGCN()
        server.data = {"edge_index": np.empty((2, 0), dtype=np.int64)}
        server.global_model = {
            "weight": torch.tensor([10.0, 20.0]),
            "bias": torch.tensor([30.0]),
        }
        local_states = [
            {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([3.0])},
            {"weight": torch.tensor([4.0, 5.0]), "bias": torch.tensor([6.0])},
        ]

        server.aggregate(local_states, [0, 1], round=1)

        self.assertEqual(len(local_states), 2, "aggregate must not mutate caller local_states")
        for tensor in server.global_model.values():
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
            self.assertEqual(tensor.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
