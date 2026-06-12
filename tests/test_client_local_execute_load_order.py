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
import types
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])


class Scalar:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __add__(self, other):
        return Scalar(self.value + (other.value if isinstance(other, Scalar) else other))

    __radd__ = __add__

    def __mul__(self, other):
        return Scalar(self.value * (other.value if isinstance(other, Scalar) else other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return Scalar(self.value / (other.value if isinstance(other, Scalar) else other))

    def detach(self):
        return self

    def cpu(self):
        return self

    def backward(self):
        pass


class Tensor:
    shape = (2, 1, 1, 1)

    def to(self, device):
        return self


class Loss:
    def __call__(self, pred, target):
        return Scalar(1.0)


class Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def install_stubs():
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    torch.no_grad = Context
    torch.enable_grad = Context
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")

    class TensorDataset:
        def __init__(self, *tensors):
            self.tensors = tensors

        def __iter__(self):
            return iter(zip(*self.tensors))

    torch_utils_data.TensorDataset = TensorDataset
    torch_utils.data = torch_utils_data
    torch.utils = torch_utils

    nn = types.ModuleType("torch.nn")
    nn.MSELoss = nn.L1Loss = nn.SmoothL1Loss = Loss
    torch.nn = nn

    torch_geometric = types.ModuleType("torch_geometric")
    pyg_data = types.ModuleType("torch_geometric.data")

    class DataLoader:
        def __init__(self, dataset, batch_size=None):
            self.dataset = dataset
            self.batch_size = batch_size

        def __iter__(self):
            return iter(self.dataset)

    pyg_data.DataLoader = DataLoader
    torch_geometric.data = pyg_data

    utils_pkg = types.ModuleType("utils")
    process_increment = types.ModuleType("utils.process_increment")
    process_increment.unscaled_metrics = lambda *args, **kwargs: {"rmse": Scalar(2.0)}
    utils_pkg.process_increment = process_increment

    models_pkg = types.ModuleType("models")
    fl_model = types.ModuleType("models.fl_model")
    fl_model.GRU = object
    models_pkg.fl_model = fl_model

    scipy = types.ModuleType("scipy")
    scipy_stats = types.ModuleType("scipy.stats")
    scipy_stats.entropy = lambda *args, **kwargs: 0.0
    scipy.stats = scipy_stats

    sys.modules.update({
        "torch": torch,
        "torch.nn": nn,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "torch_geometric": torch_geometric,
        "torch_geometric.data": pyg_data,
        "utils": utils_pkg,
        "utils.process_increment": process_increment,
        "models": models_pkg,
        "models.fl_model": fl_model,
        "scipy": scipy,
        "scipy.stats": scipy_stats,
    })


class Model:
    def __init__(self, events):
        self.events = events
        self.mode = "init"

    def load_state_dict(self, state):
        self.events.append(f"load:{state['version']}")

    def to(self, device):
        self.events.append(f"to:{device}")
        return self

    def eval(self):
        self.mode = "eval"
        self.events.append("eval")

    def train(self):
        self.mode = "train"
        self.events.append("train")

    def __call__(self, data):
        self.events.append(f"call:{self.mode}")
        return Tensor()

    def state_dict(self):
        self.events.append("state_dict")
        return {"model": self.mode}


def run_case(state_dict_to_load, should_train, has_delayed):
    install_stubs()
    spec = importlib.util.spec_from_file_location("client_oa_under_test", root / "client_oa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    events = []
    client = module.Client.__new__(module.Client)
    client.client_id = 7
    client.model = Model(events)
    client.device = "cpu"
    client.batch_size = 4
    client.args = SimpleNamespace(loss_func="mse", epoch=0)
    client.feature_scaler = object()
    batch = (Tensor(), Tensor(), Tensor(), Tensor())
    client.eval_dataset_current = [batch]
    client.train_dataset_delayed = [batch] if has_delayed else None

    client.local_execute(
        state_dict_to_load=state_dict_to_load,
        should_train=should_train,
    )
    return events


print(json.dumps({
    "selected_with_delayed": run_case({"version": "global"}, True, True),
    "non_selected_with_delayed": run_case(None, False, True),
    "selected_without_delayed": run_case({"version": "global"}, True, False),
}))
"""


class ClientLocalExecuteLoadOrderTests(unittest.TestCase):
    def run_cases(self):
        result = subprocess.run(
            [sys.executable, "-c", CASE_SCRIPT, str(ROOT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                "client_oa local_execute subprocess failed\n"
                f"returncode={result.returncode}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_selected_client_loads_global_before_evaluation_and_then_trains(self):
        events = self.run_cases()["selected_with_delayed"]

        self.assertIn("load:global", events)
        self.assertIn("eval", events)
        self.assertIn("call:eval", events)
        self.assertIn("train", events)
        self.assertLess(events.index("load:global"), events.index("eval"))
        self.assertLess(events.index("load:global"), events.index("call:eval"))
        self.assertLess(events.index("call:eval"), events.index("train"))
        self.assertEqual(events.count("load:global"), 1)

    def test_non_selected_client_does_not_load_global_model(self):
        events = self.run_cases()["non_selected_with_delayed"]

        self.assertNotIn("load:global", events)
        self.assertIn("call:eval", events)
        self.assertNotIn("train", events)

    def test_selected_client_without_delayed_data_evaluates_global_but_does_not_train(self):
        events = self.run_cases()["selected_without_delayed"]

        self.assertIn("load:global", events)
        self.assertLess(events.index("load:global"), events.index("call:eval"))
        self.assertNotIn("train", events)


if __name__ == "__main__":
    unittest.main()
