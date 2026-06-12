import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fedostc_model.py"

FORBIDDEN_MODEL_STRINGS = [
    "GATConv",
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


def _load_model_class():
    spec = importlib.util.spec_from_file_location("fedostc_model_under_test", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FedOSTCClientModel


class FedOSTCPhase1ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = _torch_or_skip()
        cls.FedOSTCClientModel = _load_model_class()

    def make_case(self, output_dim=2, pred_steps=5):
        self.torch.manual_seed(20260605)
        model = self.FedOSTCClientModel(
            input_dim=1,
            output_dim=output_dim,
            pred_steps=pred_steps,
            encoder_hidden_size=64,
            decoder_hidden_size=128,
            gru_num_layers=1,
            dropout=0.0,
        )
        model.eval()
        data = {"x": self.torch.randn(2, 4, 3, 1)}
        return model, data

    def test_encode_returns_batch_node_hidden_shape(self):
        model, data = self.make_case()

        with self.torch.no_grad():
            h = model.encode(data)

        self.assertEqual(list(h.shape), [2, 3, 64])

    def test_decode_returns_batch_horizon_node_output_shape(self):
        model, data = self.make_case(output_dim=2, pred_steps=5)
        h = self.torch.randn(2, 3, 64)
        h_prime = self.torch.randn(2, 3, 64)

        with self.torch.no_grad():
            out = model.decode(data, h, h_prime)

        self.assertEqual(list(out.shape), [2, 5, 3, 2])

    def test_decode_uses_original_and_refined_hidden_states(self):
        model, data = self.make_case(output_dim=1, pred_steps=4)
        h = self.torch.randn(2, 3, 64)
        h_prime = self.torch.randn(2, 3, 64)

        with self.torch.no_grad():
            base = model.decode(data, h, h_prime)
            changed_h = model.decode(data, h + 0.5, h_prime)
            changed_h_prime = model.decode(data, h, h_prime - 0.5)

        self.assertGreater((base - changed_h).abs().max().item(), 1e-7)
        self.assertGreater((base - changed_h_prime).abs().max().item(), 1e-7)

    def test_forward_requires_both_hidden_inputs(self):
        model, data = self.make_case()
        h = self.torch.randn(2, 3, 64)
        h_prime = self.torch.randn(2, 3, 64)

        with self.assertRaises(ValueError):
            model(data)
        with self.assertRaises(ValueError):
            model(data, h=h)
        with self.assertRaises(ValueError):
            model(data, h_prime=h_prime)

    def test_forward_delegates_to_decode_when_hidden_inputs_exist(self):
        model, data = self.make_case(output_dim=2, pred_steps=5)
        h = self.torch.randn(2, 3, 64)
        h_prime = self.torch.randn(2, 3, 64)

        with self.torch.no_grad():
            expected = model.decode(data, h, h_prime)
            actual = model(data, h=h, h_prime=h_prime)

        self.assertEqual(list(actual.shape), [2, 5, 3, 2])
        self.assertTrue(self.torch.allclose(actual, expected))

    def test_phase1_model_source_avoids_refol_selection_and_covariate_paths(self):
        source = MODEL_PATH.read_text(encoding="utf-8")

        for needle in FORBIDDEN_MODEL_STRINGS:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, source)


if __name__ == "__main__":
    unittest.main()
