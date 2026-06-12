from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fl_model import GRU
from utils.process_increment import unscaled_metrics


class FedAvgClient:
    """Client used by the clean delayed-label FedAvg baseline.

    This class intentionally contains no REFOL drift-selection state or methods.
    A server round always evaluates the current sample from the supplied global
    model, then optionally adapts on the full persistent local history whose
    labels have arrived.
    """

    def __init__(
        self,
        client_id: int,
        client_dataset: Any,
        feature_scaler: Any,
        input_size: int,
        output_size: int,
        args: Any,
        model: nn.Module | None = None,
    ) -> None:
        self.client_id = int(client_id)
        self.client_dataset = client_dataset
        self.feature_scaler = feature_scaler
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.args = args
        self.lr = float(args.lr)
        self.batch_size = int(args.batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or GRU(
            self.input_size,
            int(args.hidden_size),
            self.output_size,
            float(args.dropout),
            int(args.num_layers),
        )
        self.current_dataset = None
        self.delayed_dataset = None
        self.local_result: dict[str, Any] | None = None
        self.state_dict = self._snapshot_model_state()

    def run_round(
        self,
        *,
        global_state: Mapping[str, torch.Tensor],
        current_dataset: Any,
        delayed_dataset: Any | None,
    ) -> dict[str, Any]:
        if global_state is None:
            raise ValueError("FedAvgClient requires a global_state every round")
        self.model.load_state_dict(global_state)
        self.current_dataset = current_dataset
        self.delayed_dataset = delayed_dataset

        log = self.evaluate_current(current_dataset)
        if delayed_dataset is not None:
            self.train_delayed(delayed_dataset)

        self.state_dict = self._snapshot_model_state()
        self.local_result = {"state_dict": self.state_dict, "log": log}
        return self.local_result

    def evaluate_current(self, dataset: Any) -> dict[str, Any]:
        self.model.to(self.device)
        self.model.eval()
        epoch_log = defaultdict(lambda: 0.0)
        num_samples = 0
        criterion = self._criterion()

        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=self.batch_size):
                x, y, x_attr, y_attr = self._batch_to_device(batch)
                y_pred = self.model({"x": x, "x_attr": x_attr, "y": y, "y_attr": y_attr})
                loss = criterion(y_pred, y)

                batch_size = int(x.shape[0])
                num_samples += batch_size
                epoch_log["loss"] += loss.detach() * batch_size
                metrics = unscaled_metrics(y_pred, y, self.feature_scaler)
                for key, value in metrics.items():
                    epoch_log[key] += value * batch_size

        if num_samples <= 0:
            raise ValueError("current_dataset must contain at least one sample")
        for key in list(epoch_log):
            epoch_log[key] = (epoch_log[key] / num_samples).detach().cpu()
        epoch_log["num_samples"] = num_samples
        return dict(epoch_log)

    def train_delayed(self, dataset: Any) -> None:
        self.model.to(self.device)
        self.model.train()
        criterion = self._criterion()

        for _ in range(int(self.args.epoch)):
            for batch in DataLoader(dataset, batch_size=self.batch_size):
                x, y, x_attr, y_attr = self._batch_to_device(batch)
                self.model.zero_grad(set_to_none=True)
                y_pred = self.model({"x": x, "x_attr": x_attr, "y": y, "y_attr": y_attr})
                loss = criterion(y_pred, y)
                loss.backward()
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.add_(param.grad, alpha=-self.lr)
                self.model.zero_grad(set_to_none=True)

    def _batch_to_device(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, x_attr, y_attr = batch
        return (
            x.to(self.device),
            y.to(self.device),
            x_attr.to(self.device),
            y_attr.to(self.device),
        )

    def _criterion(self) -> nn.Module:
        loss_func_name = getattr(self.args, "loss_func", "mse")
        if loss_func_name == "mae":
            return nn.L1Loss()
        if loss_func_name == "smae":
            return nn.SmoothL1Loss()
        return nn.MSELoss()

    def _snapshot_model_state(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }


__all__ = ["FedAvgClient"]
