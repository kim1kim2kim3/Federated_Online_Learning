import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_process_increment_module():
    spec = importlib.util.spec_from_file_location(
        "process_increment_under_test", ROOT / "utils" / "process_increment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RawScaleDatasetTests(unittest.TestCase):
    def setUp(self):
        self.process_increment = load_process_increment_module()

    def test_load_dataset_keeps_feature_and_label_values_on_raw_scale(self):
        node_count = 207
        raw_x = np.zeros((2, 2, node_count, 2), dtype=np.float32)
        raw_y = np.zeros((2, 1, node_count, 2), dtype=np.float32)
        raw_x[..., 0] = 100.0 + np.arange(2 * 2 * node_count).reshape(2, 2, node_count)
        raw_x[..., 1] = 10.0 + np.arange(2 * 2 * node_count).reshape(2, 2, node_count)
        raw_y[..., 0] = 10000.0 + np.arange(2 * 1 * node_count).reshape(2, 1, node_count)
        raw_y[..., 1] = 20.0 + np.arange(2 * 1 * node_count).reshape(2, 1, node_count)
        adjacency = np.eye(node_count, dtype=np.float32)

        with (
            patch.object(
                self.process_increment,
                "load_pickle",
                return_value=(None, None, adjacency),
            ),
            patch.object(
                self.process_increment.np,
                "load",
                return_value={"x": raw_x, "y": raw_y},
            ),
        ):
            data, selected_nodes = self.process_increment.load_dataset(
                "METR-LA", "adj.pkl", num_clients=3, pred_len=12
            )

        np.testing.assert_allclose(
            data["x"].numpy(), raw_x[:, :, selected_nodes, 0:1]
        )
        np.testing.assert_allclose(
            data["y"].numpy(), raw_y[:, :, selected_nodes, 0:1]
        )
        np.testing.assert_allclose(
            data["x_attr"].numpy(), raw_x[:, :, selected_nodes, 1:2]
        )
        np.testing.assert_allclose(
            data["y_attr"].numpy(), raw_y[:, :, selected_nodes, 1:2]
        )

    def test_identity_scaler_returns_inputs_unchanged(self):
        scaler = self.process_increment.IdentityScaler()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        array = np.array([4.0, 5.0, 6.0])

        self.assertIs(scaler.inverse_transform(tensor), tensor)
        self.assertIs(scaler.transform(array), array)

    def test_unscaled_metrics_with_identity_scaler_uses_raw_values(self):
        scaler = self.process_increment.IdentityScaler()
        y_pred = torch.tensor([2.0, 4.0])
        y = torch.tensor([1.0, 1.0])

        metrics = self.process_increment.unscaled_metrics(y_pred, y, scaler)

        self.assertAlmostEqual(metrics["mse"].item(), 5.0)
        self.assertAlmostEqual(metrics["rmse"].item(), 5.0 ** 0.5)
        self.assertAlmostEqual(metrics["mae"].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
