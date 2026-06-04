import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CASE_SCRIPT = r"""
import importlib.util
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
pred_steps = int(sys.argv[2])
mutation = sys.argv[3]

spec = importlib.util.spec_from_file_location(
    "fl_model_under_test", root / "models" / "fl_model.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GRU = module.GRU

torch.manual_seed(20240604)
batch_size, input_steps, node_count = 2, 4, 3
target_dim, attr_dim = 1, 2

model = GRU(
    input_size=target_dim + attr_dim,
    hidden_size=8,
    output_size=target_dim,
    dropout=0.0,
    gru_num_layers=1,
)
model.eval()

data = {
    "x": torch.randn(batch_size, input_steps, node_count, target_dim),
    "x_attr": torch.randn(batch_size, input_steps, node_count, attr_dim),
    "y": torch.randn(batch_size, pred_steps, node_count, target_dim),
    "y_attr": torch.randn(batch_size, pred_steps, node_count, attr_dim),
}

changed = {key: value.clone() for key, value in data.items()}
if mutation == "all_y":
    changed["y"] = changed["y"] + 1000.0
elif mutation == "prefix_y":
    changed["y"][:, :-1] = changed["y"][:, :-1] + 1000.0
elif mutation == "all_y_attr":
    changed["y_attr"] = changed["y_attr"] + 5.0
else:
    raise ValueError(f"unknown mutation: {mutation}")

with torch.no_grad():
    base = model(data)
    perturbed = model(changed)

diff = (base - perturbed).abs().max().item()
print(json.dumps({"diff": diff, "shape": list(base.shape)}))
"""


def _torch_available_for_subprocess():
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stderr.strip()


class GRUNoLabelLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        available, error = _torch_available_for_subprocess()
        if not available:
            raise unittest.SkipTest(
                f"real PyTorch is not importable by {sys.executable}: {error}"
            )

    def run_gru_case(self, pred_steps, mutation):
        result = subprocess.run(
            [sys.executable, "-c", CASE_SCRIPT, str(ROOT), str(pred_steps), mutation],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                "GRU subprocess failed\n"
                f"returncode={result.returncode}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )
        return json.loads(result.stdout)

    def assert_y_change_does_not_change_output(self, pred_steps, mutation):
        result = self.run_gru_case(pred_steps, mutation)
        self.assertEqual(
            result["diff"],
            0.0,
            f"GRU output changed when future target y changed: {result}",
        )
        self.assertEqual(result["shape"], [2, pred_steps, 3, 1])

    def test_single_horizon_does_not_read_future_y(self):
        self.assert_y_change_does_not_change_output(pred_steps=1, mutation="all_y")

    def test_two_horizon_does_not_read_previous_future_y(self):
        self.assert_y_change_does_not_change_output(pred_steps=2, mutation="prefix_y")

    def test_twelve_horizon_does_not_read_any_future_y(self):
        self.assert_y_change_does_not_change_output(pred_steps=12, mutation="all_y")

    def test_future_time_attributes_are_still_decoder_inputs(self):
        result = self.run_gru_case(pred_steps=12, mutation="all_y_attr")
        self.assertGreater(
            result["diff"],
            1e-7,
            f"GRU output did not respond to known future attributes: {result}",
        )
        self.assertEqual(result["shape"], [2, 12, 3, 1])


if __name__ == "__main__":
    unittest.main()
