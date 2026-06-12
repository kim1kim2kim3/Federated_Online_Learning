import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
FL_SERVER = ROOT / "fl-server"
if str(FL_SERVER) not in sys.path:
    sys.path.insert(0, str(FL_SERVER))

import Fedavg
from Fedavg import FedAvg
from client_fedavg import FedAvgClient


class IdentityScaler:
    def inverse_transform(self, data):
        return data


class RecordingClient:
    def __init__(self, client_id, calls):
        self.client_id = client_id
        self.calls = calls
        self.current_dataset = None
        self.delayed_dataset = None

    def run_round(self, *, global_state, current_dataset, delayed_dataset):
        history_len = 0 if delayed_dataset is None else len(delayed_dataset)
        self.calls.append(
            {
                "client_id": self.client_id,
                "global_weight": global_state["weight"].clone(),
                "has_current": current_dataset is not None,
                "trained": delayed_dataset is not None,
                "history_len": history_len,
            }
        )
        state_value = float(self.client_id + (10 * history_len if history_len else 0))
        return {
            "state_dict": {"weight": torch.tensor([state_value])},
            "log": {
                "num_samples": 1,
                "loss": torch.tensor(0.0),
                "mse": torch.tensor(float(self.client_id + 1)),
                "rmse": torch.tensor(float(self.client_id + 1)),
                "mae": torch.tensor(float(self.client_id + 1)),
            },
        }


class FedAvgCleanTests(unittest.TestCase):
    def config(self, **kwargs):
        values = dict(
            agg_model="fedavg",
            dataset="toy",
            adj_mx="toy.pkl",
            num_clients=3,
            pred_steps=1,
            seed=123,
            delay=1,
            lr=0.01,
            epoch=1,
            batch_size=1,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            rounds=3,
            fedavg_client_fraction=1.0,
        )
        values.update(kwargs)
        return SimpleNamespace(**values)

    def make_data(self, rounds=4, clients=3):
        shape = (rounds, 1, clients, 1)
        return {
            "x": torch.zeros(shape),
            "y": torch.ones(shape),
            "x_attr": torch.zeros(shape),
            "y_attr": torch.zeros(shape),
            "feature_scaler": IdentityScaler(),
        }

    def make_server(self, delay=1, clients=3, rounds=4, **kwargs):
        calls = []
        server = FedAvg(self.config(delay=delay, num_clients=clients, **kwargs))
        server.num_clients = clients
        server.delay = delay
        server.train_per_num_samples = 1
        server.data = self.make_data(rounds=rounds, clients=clients)
        server.clients = [RecordingClient(i, calls) for i in range(clients)]
        server.global_model = {"weight": torch.tensor([100.0])}
        return server, calls

    def test_boot_uses_fedavg_client_and_initializes_global_model(self):
        data = self.make_data(rounds=2, clients=2)
        cfg = self.config(num_clients=2)

        with patch.object(Fedavg, "load_dataset", return_value=(data, [0, 1])):
            server = FedAvg(cfg)
            server.boot()

        self.assertTrue(all(isinstance(client, FedAvgClient) for client in server.clients))
        self.assertEqual(len(server.clients), 2)
        expected = server.clients[0].model.state_dict()
        self.assertEqual(set(server.global_model), set(expected))
        for name, tensor in server.global_model.items():
            self.assertEqual(tensor.device.type, "cpu")
            self.assertTrue(torch.equal(tensor, expected[name].detach().cpu()))

    def test_delay_then_all_clients_participate_without_sampling_or_drift_selection(self):
        server, calls = self.make_server(delay=1, clients=3, rounds=3)

        first_log = server.train_round(1)["log"]

        self.assertEqual(first_log["num_samples"], 3)
        self.assertEqual([call["trained"] for call in calls], [False, False, False])
        self.assertTrue(torch.equal(server.global_model["weight"], torch.tensor([100.0])))

        calls.clear()
        second_log = server.train_round(2)["log"]

        self.assertEqual(second_log["num_samples"], 3)
        self.assertEqual([call["client_id"] for call in calls], [0, 1, 2])
        self.assertEqual([call["trained"] for call in calls], [True, True, True])
        self.assertEqual([call["history_len"] for call in calls], [1, 1, 1])
        self.assertTrue(all(torch.equal(call["global_weight"], torch.tensor([100.0])) for call in calls))
        self.assertTrue(torch.equal(server.global_model["weight"], torch.tensor([11.0])))

        calls.clear()
        third_log = server.train_round(3)["log"]

        self.assertEqual(third_log["num_samples"], 3)
        self.assertEqual([call["trained"] for call in calls], [True, True, True])
        self.assertEqual([call["history_len"] for call in calls], [2, 2, 2])
        self.assertEqual(server.fedavg_history_indices, {0: [0, 1], 1: [0, 1], 2: [0, 1]})
        self.assertTrue(torch.equal(server.global_model["weight"], torch.tensor([21.0])))

    def test_select_clients_returns_all_history_eligible_clients(self):
        server, _ = self.make_server(delay=1, clients=3, rounds=3)
        server.update_train_data(1, server.clients)
        self.assertEqual(server.select_clients(), [])
        self.assertEqual(server.fedavg_history_indices, {0: [], 1: [], 2: []})

        server.update_train_data(2, server.clients)
        self.assertEqual(server.select_clients(), [0, 1, 2])
        self.assertEqual(server.fedavg_history_indices, {0: [0], 1: [0], 2: [0]})

    def test_client_fraction_samples_delayed_training_clients_deterministically(self):
        server_a, calls_a = self.make_server(
            delay=1,
            clients=5,
            rounds=3,
            seed=7,
            fedavg_client_fraction=0.4,
        )
        server_b, calls_b = self.make_server(
            delay=1,
            clients=5,
            rounds=3,
            seed=7,
            fedavg_client_fraction=0.4,
        )

        log_a = server_a.train_round(2)["log"]
        log_b = server_b.train_round(2)["log"]

        selected_a = [call["client_id"] for call in calls_a if call["trained"]]
        selected_b = [call["client_id"] for call in calls_b if call["trained"]]
        self.assertEqual(log_a["num_samples"], 5)
        self.assertEqual(log_b["num_samples"], 5)
        self.assertEqual(len(calls_a), 5)
        self.assertEqual(len(selected_a), 2)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(server_a.last_selected_client_ids, selected_a)
        self.assertTrue(set(selected_a).issubset({0, 1, 2, 3, 4}))
        self.assertTrue(all(torch.equal(call["global_weight"], torch.tensor([100.0])) for call in calls_a))
        self.assertEqual(
            [call["history_len"] for call in calls_a if call["trained"]],
            [1, 1],
        )

    def test_client_fraction_preserves_unselected_history_until_next_participation(self):
        server, calls = self.make_server(
            delay=1,
            clients=3,
            rounds=4,
            seed=3,
            fedavg_client_fraction=1 / 3,
        )

        server.train_round(2)
        first_trained = [call["client_id"] for call in calls if call["trained"]]
        self.assertEqual(first_trained, [2])
        self.assertEqual(server.fedavg_history_indices[1], [0])

        calls.clear()
        server.train_round(3)
        second_trained = [call for call in calls if call["trained"]]

        self.assertEqual([call["client_id"] for call in second_trained], [1])
        self.assertEqual(second_trained[0]["history_len"], 2)
        self.assertEqual(server.fedavg_history_indices[1], [0, 1])

    def test_selected_updates_are_weighted_by_full_history_length(self):
        server, _ = self.make_server(delay=1, clients=2, rounds=4)
        server.fedavg_history_indices = {0: [0], 1: [0, 1, 2]}
        server.update_train_data(1, server.clients)

        _, local_states = server.local_execute([0, 1])

        self.assertEqual([num_samples for _, num_samples in local_states], [1, 3])
        server.aggregate(local_states, rround=4)
        self.assertTrue(
            torch.allclose(
                server.global_model["weight"],
                torch.tensor([(10.0 * 1.0 + 31.0 * 3.0) / 4.0]),
            )
        )

    def test_client_fraction_rejects_invalid_values(self):
        for fraction in (0.0, -0.1, 1.1):
            with self.subTest(fraction=fraction):
                server, _ = self.make_server(
                    delay=1,
                    clients=3,
                    fedavg_client_fraction=fraction,
                )
                server.update_train_data(2, server.clients)
                with self.assertRaises(ValueError):
                    server.select_clients(2)

    def test_aggregate_weighted_average_preserves_non_floating_buffers(self):
        server, _ = self.make_server()
        states = [
            (
                {
                    "weight": torch.tensor([1.0, 3.0]),
                    "counter": torch.tensor([7], dtype=torch.int64),
                },
                1,
            ),
            (
                {
                    "weight": torch.tensor([5.0, 7.0]),
                    "counter": torch.tensor([9], dtype=torch.int64),
                },
                3,
            ),
        ]

        server.aggregate(states, rround=2)

        self.assertTrue(torch.allclose(server.global_model["weight"], torch.tensor([4.0, 6.0])))
        self.assertEqual(server.global_model["counter"].dtype, torch.int64)
        self.assertTrue(torch.equal(server.global_model["counter"], torch.tensor([7])))

    def test_fedavg_sources_do_not_use_refol_client_or_drift_state(self):
        fedavg_source = (FL_SERVER / "Fedavg.py").read_text(encoding="utf-8")
        client_source = (ROOT / "client_fedavg.py").read_text(encoding="utf-8")
        combined = fedavg_source + "\n" + client_source

        forbidden = [
            "client_oa",
            "Refol",
            "eval_drift",
            "h_client_dataset",
            "kl_threshold",
            ".selected",
            "selected =",
        ]
        for needle in forbidden:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, combined)


if __name__ == "__main__":
    unittest.main()
