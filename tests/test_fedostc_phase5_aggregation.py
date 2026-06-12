import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
FL_SERVER = ROOT / "fl-server"
if str(FL_SERVER) not in sys.path:
    sys.path.insert(0, str(FL_SERVER))

from Fedostc import FedOSTC, average_aggregate, compute_model_distances, compute_rho, period_aggregate


class FedOSTCPhase5AggregationTests(unittest.TestCase):
    def state(self, value):
        return {
            "weight": torch.tensor([float(value), float(value + 2)]),
            "counter": torch.tensor([int(value)], dtype=torch.int64),
        }

    def test_eq13_uniform_average_preserves_non_floating_state(self):
        states = [self.state(1), self.state(3), self.state(5)]

        result = average_aggregate(states)

        self.assertTrue(torch.allclose(result["weight"], torch.tensor([3.0, 5.0])))
        self.assertEqual(result["counter"].dtype, torch.int64)
        self.assertTrue(torch.equal(result["counter"], states[0]["counter"]))
        self.assertIsNot(result["counter"], states[0]["counter"])

    def test_eq14_distance_and_rho_softmax(self):
        states = [self.state(1), self.state(3)]
        fresh = average_aggregate(states)

        distances = compute_model_distances(states, fresh)
        rho = compute_rho(distances)

        self.assertTrue(torch.allclose(distances, torch.tensor([2 ** 0.5, 2 ** 0.5], dtype=torch.float64)))
        self.assertTrue(torch.allclose(rho, torch.softmax(-distances, dim=0)))

    def test_eq15_period_weighted_aggregate(self):
        states = [self.state(1), self.state(3)]

        result = period_aggregate(states, torch.tensor([0.25, 0.75]))

        expected = states[0]["weight"] * 0.25 + states[1]["weight"] * 0.75
        self.assertTrue(torch.allclose(result["weight"], expected))
        self.assertTrue(torch.equal(result["counter"], states[0]["counter"]))

    def test_period_boundary_and_lookup_offset(self):
        server = FedOSTC(SimpleNamespace())
        server.num_clients = 2
        server.period_steps = 2
        states = [self.state(1), self.state(5)]

        tau2 = server.aggregate_for_tau(states, tau=2)
        self.assertTrue(torch.allclose(tau2["weight"], average_aggregate(states)["weight"]))
        server.rho_history[2] = torch.tensor([1.0, 0.0], dtype=torch.float64)

        tau3 = server.aggregate_for_tau(states, tau=3)

        self.assertTrue(torch.allclose(tau3["weight"], states[0]["weight"]))

    def test_missing_client_state_or_rho_fails_fast(self):
        server = FedOSTC(SimpleNamespace())
        server.num_clients = 2
        server.period_steps = 1

        with self.assertRaises(ValueError):
            server.aggregate_for_tau([self.state(1)], tau=1)

        server.period_steps = 0
        with self.assertRaises(KeyError):
            server.aggregate_for_tau([self.state(1), self.state(2)], tau=1)

        with self.assertRaises(ValueError):
            period_aggregate([self.state(1), self.state(2)], torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
