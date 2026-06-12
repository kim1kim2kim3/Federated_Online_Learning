import math
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
FL_SERVER = ROOT / "fl-server"
if str(FL_SERVER) not in sys.path:
    sys.path.insert(0, str(FL_SERVER))

import Fedostc
from base_server import BaseFLServer
from Fedostc import FedOSTC
from client_fedostc import FedOSTCClient
from models.fedostc_gat import FedOSTCGAT


class IdentityScaler:
    def inverse_transform(self, data):
        return data


class RecordingGAT:
    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def __call__(self, hidden, graph=None):
        self.calls.append((hidden.detach().clone(), graph))
        return hidden


class RecordingClient:
    def __init__(self, client_id, calls):
        self.client_id = client_id
        self.calls = calls
        self.current_dataset = None
        self.delayed_dataset = None

    def load_model_state(self, state_dict):
        self.calls.append(("load", self.client_id, state_dict["weight"].clone()))

    def encode_current(self, dataset):
        self.calls.append(("encode_current", self.client_id, dataset[0][0].detach().clone()))
        return torch.tensor([[[float(self.client_id + 1)]]])

    def evaluate_current(self, dataset, h, h_prime):
        self.calls.append(("evaluate_current", self.client_id, h.detach().clone(), h_prime.detach().clone()))
        return torch.full((1, 1, 1, 1), float(self.client_id + 1))

    def encode_delayed(self, dataset):
        x, y = dataset[0]
        self.calls.append(("encode_delayed", self.client_id, x.detach().clone(), y.detach().clone()))
        return torch.tensor([[[float(self.client_id + 2)]]])

    def train_delayed(self, dataset, h_prime, update_model_state):
        x, y = dataset[0]
        self.calls.append(
            (
                "train_delayed",
                self.client_id,
                update_model_state["weight"].clone(),
                x.detach().clone(),
                y.detach().clone(),
                h_prime.detach().clone(),
            )
        )
        state = {"weight": update_model_state["weight"].clone() + self.client_id + 1.0}
        return {"state_dict": state, "log": {"num_samples": 1, "loss": 0.0}}


def iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_tensors(child)


class FedOSTCPhase4ServerTests(unittest.TestCase):
    def config(self, **kwargs):
        values = dict(
            agg_model="fedostc",
            dataset="toy",
            adj_mx="toy.pkl",
            num_clients=2,
            pred_steps=1,
            seed=123,
            delay=1,
            period_steps=2,
            lr=0.01,
            epoch=1,
            batch_size=1,
            num_layers=1,
            dropout=0.0,
            rounds=3,
        )
        values.update(kwargs)
        return SimpleNamespace(**values)

    def make_data(self, rounds=5, clients=2):
        x = torch.arange(rounds * clients, dtype=torch.float32).view(rounds, 1, clients, 1)
        y = (100 + torch.arange(rounds * clients, dtype=torch.float32)).view(rounds, 1, clients, 1)
        return {
            "x": x,
            "y": y,
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "feature_scaler": IdentityScaler(),
        }

    def make_server(self, delay=1, period_steps=10, clients=2, rounds=5):
        calls = []
        server = FedOSTC(self.config(delay=delay, period_steps=period_steps, num_clients=clients))
        server.num_clients = clients
        server.delay = delay
        server.period_steps = period_steps
        server.data = self.make_data(rounds=rounds, clients=clients)
        server.clients = [RecordingClient(i, calls) for i in range(clients)]
        server.gat = RecordingGAT()
        server.global_model = {"weight": torch.tensor([10.0])}
        server.rho_history = {}
        server.prediction_history = {}
        server.w_pred = {}
        server.w_update = {}
        server.train_per_num_samples = 1
        return server, calls

    def test_boot_initializes_clients_gat_histories_and_client0_global(self):
        data = self.make_data(rounds=4, clients=3)
        cfg = self.config(num_clients=3, fedostc_encoder_hidden_size=4, fedostc_decoder_hidden_size=8)

        with patch.object(Fedostc, "load_dataset", return_value=(data, [0, 1, 2])):
            server = FedOSTC(cfg)
            server.boot()

        self.assertEqual(len(server.clients), 3)
        self.assertTrue(all(isinstance(client, FedOSTCClient) for client in server.clients))
        self.assertIsInstance(server.gat, FedOSTCGAT)
        self.assertEqual(server.rho_history, {})
        self.assertEqual(server.prediction_history, {})
        self.assertEqual(server.w_pred, {})
        self.assertEqual(server.w_update, {})
        expected = server.clients[0].model.state_dict()
        self.assertEqual(set(server.global_model), set(expected))
        for name in expected:
            self.assertTrue(torch.equal(server.global_model[name], expected[name].detach().cpu()))
            self.assertIsNot(server.global_model[name], expected[name])

    def test_current_round_metric_is_finite_before_delayed_feedback_and_precedes_training(self):
        server, calls = self.make_server(delay=1, clients=2, rounds=3)

        log = server.train_round(1)["log"]

        self.assertEqual(log["num_samples"], 2)
        self.assertFalse(math.isnan(log["rmse"].item()))
        self.assertAlmostEqual(log["rmse"].item(), 99.0)
        self.assertAlmostEqual(log["mae"].item(), 99.0)
        self.assertIn(1, server.prediction_history)
        self.assertFalse(any(call[0] == "train_delayed" for call in calls))

        server.train_round(2)

        first_train = min(index for index, call in enumerate(calls) if call[0] == "train_delayed")
        last_eval = max(index for index, call in enumerate(calls) if call[0] == "evaluate_current")
        self.assertLess(last_eval, first_train)

    def test_prediction_only_path_reads_no_current_label(self):
        server, calls = self.make_server(delay=1, clients=2, rounds=3)
        server.update_train_data(1, server.clients)
        labels = server.data.pop("y")

        predictions = server.predict_current(1)

        self.assertEqual(set(predictions), {0, 1})
        self.assertIn(1, server.prediction_history)
        self.assertFalse(any(call[0] == "train_delayed" for call in calls))
        server.data["y"] = labels

    def test_no_label_round_keeps_global_model_and_skips_delayed_training(self):
        server, calls = self.make_server(delay=2, clients=2)
        before = deepcopy(server.global_model)

        log = server.train_round(1)["log"]

        self.assertEqual(log["num_samples"], 2)
        self.assertFalse(math.isnan(log["rmse"].item()))
        self.assertFalse(any(call[0] == "train_delayed" for call in calls))
        self.assertTrue(torch.equal(server.global_model["weight"], before["weight"]))

    def test_delayed_feedback_round_returns_current_prediction_metric_not_tau_metric(self):
        server, _ = self.make_server(delay=1, clients=2, rounds=4)
        server.prediction_history[2] = {
            0: torch.zeros(1, 1, 1, 1),
            1: torch.zeros(1, 1, 1, 1),
        }

        log = server.train_round(3)["log"]

        self.assertAlmostEqual(log["rmse"].item(), 103.0)
        self.assertAlmostEqual(log["mae"].item(), 103.0)

    def test_prediction_metric_matches_refol_base_local_log_aggregation(self):
        server, _ = self.make_server(delay=1, clients=2, rounds=2)
        server.data["y"][0, :, 0:1, :] = 1.0
        server.data["y"][0, :, 1:2, :] = 3.0
        server.prediction_history[1] = {
            0: torch.zeros(1, 1, 1, 1),
            1: torch.zeros(1, 1, 1, 1),
        }
        local_logs = []
        for client_id in (0, 1):
            target = server.data["y"][0:1, :, client_id:client_id + 1, :]
            local_log = Fedostc.metric_values(
                server.prediction_history[1][client_id],
                target,
                server.data["feature_scaler"],
            )
            local_log["num_samples"] = int(target.size(0))
            local_logs.append(local_log)
        expected = BaseFLServer(self.config()).aggregate_local_logs(local_logs)

        log = server.evaluate_prediction_round(1)

        self.assertAlmostEqual(log["mse"].item(), expected["mse"].item())
        self.assertAlmostEqual(log["rmse"].item(), expected["rmse"].item())
        self.assertAlmostEqual(log["mae"].item(), expected["mae"].item())
        self.assertAlmostEqual(log["rmse"].item(), 2.0)
        self.assertNotAlmostEqual(log["rmse"].item(), math.sqrt(5.0))

    def test_delayed_feedback_uses_tau_and_all_clients_from_update_model(self):
        server, calls = self.make_server(delay=1, clients=3, rounds=4)
        server.prediction_history[2] = {
            0: torch.zeros(1, 1, 1, 1),
            1: torch.zeros(1, 1, 1, 1),
            2: torch.zeros(1, 1, 1, 1),
        }

        server.train_round(3)

        train_calls = [call for call in calls if call[0] == "train_delayed"]
        self.assertEqual([call[1] for call in train_calls], [0, 1, 2])
        for call in train_calls:
            client_id = call[1]
            self.assertTrue(torch.equal(call[2], torch.tensor([10.0])))
            self.assertTrue(torch.equal(call[3], server.data["x"][1, :, client_id:client_id + 1, :]))
            self.assertTrue(torch.equal(call[4], server.data["y"][1, :, client_id:client_id + 1, :]))
        self.assertTrue(torch.equal(server.w_update[2]["weight"], torch.tensor([10.0])))

    def test_feedback_update_returns_update_stats_without_prediction_metrics(self):
        server, _ = self.make_server(delay=1, clients=2, rounds=3)
        server.prediction_history[1] = {
            0: torch.zeros(1, 1, 1, 1),
            1: torch.zeros(1, 1, 1, 1),
        }

        update_log = server.update_from_feedback(1)

        self.assertEqual(update_log, {"num_samples": 2})
        self.assertNotIn("rmse", update_log)
        self.assertNotIn("mae", update_log)

    def test_prediction_provenance_never_initializes_delayed_training(self):
        server, calls = self.make_server(delay=1, clients=2)
        server.w_pred[1] = {"weight": torch.tensor([99.0])}
        server.prediction_history[1] = {
            0: torch.zeros(1, 1, 1, 1),
            1: torch.zeros(1, 1, 1, 1),
        }

        server.train_round(2)

        train_states = [call[2] for call in calls if call[0] == "train_delayed"]
        self.assertEqual(len(train_states), 2)
        for state in train_states:
            self.assertTrue(torch.equal(state, torch.tensor([10.0])))

    def test_flush_processes_remaining_feedback_without_current_prediction(self):
        server, calls = self.make_server(delay=2, period_steps=10, clients=2, rounds=4)
        for tau in (3, 4):
            server.prediction_history[tau] = {
                0: torch.zeros(1, 1, 1, 1),
                1: torch.zeros(1, 1, 1, 1),
            }
        server.predict_current = lambda rround: (_ for _ in ()).throw(AssertionError("no new prediction"))

        server.flush(4)

        train_calls = [call for call in calls if call[0] == "train_delayed"]
        self.assertEqual(len(train_calls), 4)
        self.assertEqual([call[3].item() for call in train_calls], [4.0, 5.0, 6.0, 7.0])

    def test_run_writes_one_metric_row_per_prediction_round_and_flush_adds_none(self):
        server, _ = self.make_server(delay=1, clients=2, rounds=4)
        rows = []
        closed = []

        class RecordingWriter:
            def __init__(self, dataset, pred_steps, agg_model):
                self.dataset = dataset
                self.pred_steps = pred_steps
                self.agg_model = agg_model

            def write_round(self, rround, rmse, mae):
                rows.append((rround, rmse, mae))

            def close(self):
                closed.append(True)

        flush_calls = []
        server._resolve_rounds = lambda: 3
        server.train_round = lambda rround: {
            "log": {
                "rmse": torch.tensor(float(rround)),
                "mae": torch.tensor(float(rround) + 0.5),
            }
        }
        server.flush = lambda rounds: flush_calls.append(rounds)

        with patch.object(Fedostc, "RoundMetricWriter", RecordingWriter):
            server.run()

        self.assertEqual(rows, [(1, 1.0, 1.5), (2, 2.0, 2.5), (3, 3.0, 3.5)])
        self.assertEqual(flush_calls, [3])
        self.assertEqual(closed, [True])

    def test_fedostc_histories_stay_bounded_after_many_rounds_and_flush(self):
        delay = 2
        period_steps = 3
        server, _ = self.make_server(delay=delay, period_steps=period_steps, clients=2, rounds=9)

        for rround in range(1, 8):
            server.train_round(rround)
            self.assertLessEqual(len(server.prediction_history), delay)
            self.assertLessEqual(len(server.w_pred), 1)
            self.assertLessEqual(len(server.w_update), 1)
            self.assertLessEqual(len(server.rho_history), period_steps)

        self.assertEqual(set(server.prediction_history), {6, 7})

        server.flush(7)

        self.assertEqual(server.prediction_history, {})
        self.assertLessEqual(len(server.w_pred), 1)
        self.assertLessEqual(len(server.w_update), 1)
        self.assertLessEqual(len(server.rho_history), period_steps)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-only retention regression")
    def test_long_lived_fedostc_histories_do_not_retain_cuda_tensors(self):
        server, _ = self.make_server(delay=1, period_steps=2, clients=2, rounds=4)
        server.global_model = {"weight": torch.tensor([10.0], device="cuda")}

        server.train_round(1)
        server.train_round(2)

        long_lived = {
            "global_model": server.global_model,
            "w_pred": server.w_pred,
            "w_update": server.w_update,
            "prediction_history": server.prediction_history,
            "rho_history": server.rho_history,
        }
        retained_cuda = [
            name
            for name, value in long_lived.items()
            if any(tensor.is_cuda for tensor in iter_tensors(value))
        ]

        self.assertEqual(retained_cuda, [])


if __name__ == "__main__":
    unittest.main()
