import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "client_fedostc.py"
MODEL_PATH = ROOT / "models" / "fedostc_model.py"

FORBIDDEN_CLIENT_STRINGS = [
    "client_oa",
    "selected",
    "eval_drift",
    "kl_threshold",
    "x_attr",
    "y_attr",
    "Adam",
    "momentum",
    "scheduler",
]


def _torch_or_skip():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - only used on missing optional dep
        raise unittest.SkipTest(f"PyTorch is not importable: {exc}") from exc
    return torch


def _load_class(path, module_name, class_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


class FedOSTCPhase3ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = _torch_or_skip()
        cls.FedOSTCClient = _load_class(
            CLIENT_PATH, "client_fedostc_under_test", "FedOSTCClient"
        )
        cls.FedOSTCClientModel = _load_class(
            MODEL_PATH, "fedostc_model_for_client_test", "FedOSTCClientModel"
        )

    def make_model(self):
        return self.FedOSTCClientModel(
            input_dim=1,
            output_dim=1,
            pred_steps=2,
            encoder_hidden_size=4,
            decoder_hidden_size=8,
            gru_num_layers=1,
            dropout=0.0,
        )

    def make_client(self, local_epochs=1, lr=0.05, model=None):
        self.torch.manual_seed(20260606)
        return self.FedOSTCClient(
            model=model or self.make_model(),
            lr=lr,
            local_epochs=local_epochs,
            batch_size=4,
        )

    def make_batch(self, target_shift=0.0):
        self.torch.manual_seed(11)
        return {
            "x": self.torch.randn(2, 3, 3, 1),
            "y": self.torch.randn(2, 2, 3, 1) + target_shift,
        }

    @staticmethod
    def changed_keys(before, after):
        return [
            key
            for key in before
            if (before[key] - after[key]).abs().max().item() > 1e-9
        ]

    def test_current_evaluation_returns_prediction_without_current_label_access(self):
        client = self.make_client()
        batch = self.make_batch()
        h = client.encode_current(batch)
        h_prime = self.torch.randn_like(h)

        with_label = client.evaluate_current(batch, h, h_prime)
        changed_label = {"x": batch["x"], "y": batch["y"] + 1000.0}
        without_label = {"x": batch["x"]}
        changed_prediction = client.evaluate_current(changed_label, h, h_prime)
        no_label_prediction = client.evaluate_current(without_label, h, h_prime)

        self.assertTrue(self.torch.is_tensor(with_label))
        self.assertEqual(list(with_label.shape), [2, 2, 3, 1])
        self.assertTrue(self.torch.equal(with_label, changed_prediction))
        self.assertTrue(self.torch.equal(with_label, no_label_prediction))

    def test_delayed_training_uses_delayed_labels_and_changes_weights(self):
        client_a = self.make_client()
        initial = deepcopy(client_a.model.state_dict())
        batch_a = self.make_batch(target_shift=0.0)
        h_prime = self.torch.zeros_like(client_a.encode_delayed(batch_a))

        result_a = client_a.train_delayed(batch_a, h_prime, initial)
        after_a = deepcopy(client_a.model.state_dict())

        client_b = self.make_client()
        batch_b = {"x": batch_a["x"].clone(), "y": batch_a["y"] + 5.0}
        result_b = client_b.train_delayed(batch_b, h_prime.clone(), initial)
        after_b = deepcopy(client_b.model.state_dict())

        self.assertGreater(len(self.changed_keys(initial, after_a)), 0)
        self.assertGreater(
            sum((after_a[key] - after_b[key]).abs().sum().item() for key in after_a),
            1e-7,
        )
        self.assertIn("state_dict", result_a)
        self.assertEqual(result_a["log"]["num_samples"], 2)
        self.assertNotEqual(result_a["log"]["loss"], result_b["log"]["loss"])
        self.assertFalse(any(tensor.is_cuda for tensor in result_a["state_dict"].values()))
        self.assertFalse(any(tensor.is_cuda for tensor in client_a.state_dict.values()))

    def test_encoder_updates_while_supplied_refined_hidden_is_detached(self):
        client = self.make_client(local_epochs=1)
        batch = self.make_batch()
        initial = deepcopy(client.model.state_dict())
        h0 = client.encode_delayed(batch)
        h_prime = self.torch.randn_like(h0, requires_grad=True)

        client.train_delayed(batch, h_prime, initial)
        after = client.model.state_dict()
        changed_encoder_keys = [
            key
            for key in after
            if key.startswith("encoder.")
            and (initial[key] - after[key]).abs().max().item() > 1e-9
        ]

        self.assertGreater(len(changed_encoder_keys), 0)
        self.assertIsNone(h_prime.grad)

    def test_recomputes_local_hidden_each_epoch_and_reuses_supplied_refinement(self):
        class RecordingModel(self.FedOSTCClientModel):
            def __init__(self):
                super().__init__(
                    input_dim=1,
                    output_dim=1,
                    pred_steps=2,
                    encoder_hidden_size=4,
                    decoder_hidden_size=8,
                    gru_num_layers=1,
                    dropout=0.0,
                )
                self.encode_calls = 0
                self.refined_ptrs = []
                self.refined_requires_grad = []

            def encode(self, data):
                self.encode_calls += 1
                return super().encode(data)

            def decode(self, data, h, h_prime):
                self.refined_ptrs.append(h_prime.data_ptr())
                self.refined_requires_grad.append(h_prime.requires_grad)
                return super().decode(data, h, h_prime)

        self.torch.manual_seed(20260606)
        model = RecordingModel()
        client = self.make_client(local_epochs=3, lr=0.01, model=model)
        batch = self.make_batch()
        state = deepcopy(client.model.state_dict())
        h_prime = self.torch.randn(2, 3, 4, requires_grad=True)

        client.train_delayed(batch, h_prime, state)

        self.assertEqual(model.encode_calls, 3)
        self.assertEqual(len(model.refined_ptrs), 3)
        self.assertEqual(len(set(model.refined_ptrs)), 1)
        self.assertEqual(model.refined_requires_grad, [False, False, False])

    def test_phase3_client_source_avoids_refol_and_non_ogd_paths(self):
        source = CLIENT_PATH.read_text(encoding="utf-8")

        for needle in FORBIDDEN_CLIENT_STRINGS:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, source)


if __name__ == "__main__":
    unittest.main()
