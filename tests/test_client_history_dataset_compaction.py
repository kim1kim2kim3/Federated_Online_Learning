import importlib.util
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "client_oa.py"


def _load_client_module():
    try:
        import torch_geometric  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific
        raise unittest.SkipTest(f"torch_geometric is not importable: {exc}") from exc
    spec = importlib.util.spec_from_file_location("client_oa_under_compaction_test", CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClientHistoryDatasetCompactionTests(unittest.TestCase):
    def test_compact_history_dataset_clones_only_visible_tensor_slices(self):
        module = _load_client_module()
        base = torch.arange(1000, dtype=torch.float32)
        visible = base[123:124]
        dataset = TensorDataset(visible)

        compact = module.compact_history_dataset(dataset)

        self.assertIsInstance(compact, TensorDataset)
        self.assertEqual(compact.tensors[0].numel(), 1)
        self.assertEqual(compact.tensors[0].item(), 123.0)
        self.assertNotEqual(
            compact.tensors[0].untyped_storage().data_ptr(),
            base.untyped_storage().data_ptr(),
        )

        base[123] = -1.0
        self.assertEqual(compact.tensors[0].item(), 123.0)


if __name__ == "__main__":
    unittest.main()
