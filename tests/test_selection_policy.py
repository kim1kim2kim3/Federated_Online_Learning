import inspect
import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FL_SERVER = ROOT / "fl-server"
if str(FL_SERVER) not in sys.path:
    sys.path.insert(0, str(FL_SERVER))


def install_dependency_stubs():
    """Install minimal stubs so server modules can be imported without ML deps."""
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    torch.tensor = lambda value: SimpleNamespace(float=lambda: value)

    class Adam:
        def __init__(self, params, lr):
            self.params = list(params)
            self.lr = lr

    torch.optim = SimpleNamespace(Adam=Adam)

    nn = types.ModuleType("torch.nn")
    nn.MSELoss = nn.L1Loss = nn.SmoothL1Loss = lambda: None
    torch.nn = nn

    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class TensorDataset:
        def __init__(self, *tensors):
            self.tensors = tensors

        def __len__(self):
            return len(self.tensors[0]) if self.tensors else 0

    data_mod.TensorDataset = TensorDataset
    data_mod.DataLoader = object
    utils_mod.data = data_mod

    numpy = types.ModuleType("numpy")
    numpy.array = lambda value, *args, **kwargs: value
    numpy.int64 = int
    numpy.isin = lambda *args, **kwargs: []
    numpy.sum = lambda *args, **kwargs: []
    numpy.zeros = lambda *args, **kwargs: []
    numpy.arange = lambda *args, **kwargs: []
    numpy.full = lambda *args, **kwargs: []
    numpy.stack = lambda *args, **kwargs: []
    numpy.hstack = lambda *args, **kwargs: []
    numpy.concatenate = lambda *args, **kwargs: []
    numpy.random = SimpleNamespace(seed=lambda seed: None)

    xlsxwriter = types.ModuleType("xlsxwriter")
    xlsxwriter.Workbook = lambda *args, **kwargs: None

    utils_pkg = types.ModuleType("utils")
    process_increment = types.ModuleType("utils.process_increment")
    process_increment.load_dataset = lambda *args, **kwargs: ({}, [])
    process_increment.unscaled_metrics = lambda *args, **kwargs: {}
    utils_pkg.process_increment = process_increment

    client_oa = types.ModuleType("client_oa")
    client_oa.Client = object

    models_pkg = types.ModuleType("models")
    aggregation_gcn = types.ModuleType("models.AggregationGCN")
    aggregation_gcn.AttGCN = lambda: None
    models_pkg.AggregationGCN = aggregation_gcn

    sys.modules.update({
        "torch": torch,
        "torch.nn": nn,
        "torch.utils": utils_mod,
        "torch.utils.data": data_mod,
        "numpy": numpy,
        "xlsxwriter": xlsxwriter,
        "utils": utils_pkg,
        "utils.process_increment": process_increment,
        "client_oa": client_oa,
        "models": models_pkg,
        "models.AggregationGCN": aggregation_gcn,
    })


def reload_module(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


class FakeArray:
    def __getitem__(self, key):
        return ("slice", key)


class FakeClient:
    def __init__(self, client_id, delayed=None, selected=False):
        self.client_id = client_id
        self.train_dataset_delayed = delayed
        self.selected = selected
        self.eval_calls = 0
        self.local_calls = []
        self.local_result = {
            "state_dict": {"client": client_id},
            "log": {"num_samples": 1, "loss": 0},
        }

    def eval_drift(self):
        self.eval_calls += 1

    def local_execute(self, state_dict_to_load, should_train=False):
        self.local_calls.append((state_dict_to_load, should_train))


class SelectionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_dependency_stubs()

    def test_base_update_and_select_uses_delayed_label_eligibility_only(self):
        base_server = reload_module("base_server")
        server = base_server.BaseFLServer(SimpleNamespace(delay=1))
        server.train_per_num_samples = 1
        server.data = {name: FakeArray() for name in ["x", "y", "x_attr", "y_attr"]}
        server.clients = [FakeClient(i, selected=True) for i in range(3)]

        server.update_train_data(1, server.clients)
        self.assertEqual(server.select_clients(), [])
        self.assertTrue(all(client.train_dataset_delayed is None for client in server.clients))
        self.assertTrue(all(client.eval_calls == 0 for client in server.clients))

        for client in server.clients:
            client.selected = False
        server.update_train_data(2, server.clients)
        self.assertEqual(server.select_clients(), [0, 1, 2])
        self.assertTrue(all(client.train_dataset_delayed is not None for client in server.clients))
        self.assertTrue(all(client.eval_calls == 0 for client in server.clients))

    def test_delay_can_follow_prediction_horizon(self):
        base_server = reload_module("base_server")
        server = base_server.BaseFLServer(SimpleNamespace(delay="pred_steps", pred_steps=3))
        server.train_per_num_samples = 1
        server.data = {name: FakeArray() for name in ["x", "y", "x_attr", "y_attr"]}
        server.clients = [FakeClient(i, selected=True) for i in range(2)]

        server.update_train_data(3, server.clients)
        self.assertTrue(all(client.train_dataset_delayed is None for client in server.clients))

        server.update_train_data(4, server.clients)
        self.assertTrue(all(client.train_dataset_delayed is not None for client in server.clients))

    def test_base_local_execute_passes_explicit_should_train_flag(self):
        base_server = reload_module("base_server")
        server = base_server.BaseFLServer(SimpleNamespace(delay=0))
        server.global_model = {"w": 1}
        server.clients = [FakeClient(i, delayed=object()) for i in range(3)]

        local_logs, agg_state_dict = server.local_execute([1])

        self.assertEqual(len(local_logs), 3)
        self.assertEqual(agg_state_dict, [{"client": 1}])
        self.assertEqual(server.clients[0].local_calls, [(None, False)])
        self.assertEqual(server.clients[1].local_calls, [({"w": 1}, True)])
        self.assertEqual(server.clients[2].local_calls, [(None, False)])

    def test_refol_select_clients_owns_drift_selection(self):
        reload_module("base_server")
        refol = reload_module("Refol")
        server = refol.REFOL.__new__(refol.REFOL)
        server.clients = [
            FakeClient(0, delayed=object(), selected=True),
            FakeClient(1, delayed=object(), selected=False),
            FakeClient(2, delayed=None, selected=True),
        ]

        self.assertEqual(server.select_clients(), [0])
        self.assertEqual([client.eval_calls for client in server.clients], [1, 1, 1])

    def test_client_local_execute_signature_and_training_gate_are_decoupled_from_selected(self):
        source = (ROOT / "client_oa.py").read_text()
        # Use source assertions instead of importing client_oa because this environment lacks torch_geometric.
        self.assertIn("def local_execute(self, state_dict_to_load, should_train=False):", source)
        self.assertIn("if should_train and self.train_dataset_delayed is not None:", source)
        self.assertNotIn("if self.selected and self.train_dataset_delayed is not None:", source)
        self.assertIn("if self.h_client_dataset is None:\n            self.selected = True", source)

if __name__ == "__main__":
    unittest.main()
