from __future__ import annotations

import csv
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import TensorDataset

from base_server import BaseFLServer
from client_fedostc import FedOSTCClient
from models.fedostc_gat import FedOSTCGAT
from utils.process_increment import load_dataset


def clone_state(
    state: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str | None = "cpu",
) -> dict[str, torch.Tensor]:
    """Detach and clone a model state for long-lived storage.

    FedOSTC keeps several state dictionaries around for delayed feedback and
    debugging/provenance.  Keeping those snapshots on CUDA makes GPU memory grow
    linearly with the number of rounds, so the default is an immutable CPU copy.
    Callers that need a same-device short-lived copy can pass ``device=None``.
    """
    target = torch.device(device) if device is not None else None
    return {
        name: (
            tensor.detach().to(target, copy=True)
            if target is not None
            else tensor.detach().clone()
        )
        for name, tensor in state.items()
    }


def metric_values(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    scaler: Any | None,
) -> dict[str, torch.Tensor]:
    pred = y_pred.detach().cpu()
    target = y.detach().cpu()
    if scaler is not None:
        pred = scaler.inverse_transform(pred)
        target = scaler.inverse_transform(target)
    mse = ((pred - target) ** 2).mean()
    return {
        "mse": mse.detach(),
        "rmse": torch.sqrt(mse).detach(),
        "mae": torch.abs(pred - target).mean().detach(),
    }


class FedOSTC(BaseFLServer):
    """Delayed-label FedOSTC server runtime for the exact-path client/GAT."""

    def boot(self) -> None:
        print("Booting {} fl-server...".format(self.config.agg_model))
        self.num_clients = int(self.config.num_clients)
        print("Total clients: {}".format(self.num_clients))

        data, _ = load_dataset(
            name=self.config.dataset,
            adj_mx_name=self.config.adj_mx,
            num_clients=self.num_clients,
            pred_len=self.config.pred_steps,
        )
        self.data = data
        self.input_dim = int(data["x"].shape[-1])
        self.output_dim = int(data["y"].shape[-1])
        self.max_epoch = int(data["x"].shape[0])
        self.train_per_num_samples = 1
        self.delay = self._resolve_label_delay()
        self.period_steps = int(getattr(self.config, "period_steps", 288))

        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))

        self.clients = [
            FedOSTCClient(
                client_id=client_i,
                client_dataset=None,
                feature_scaler=self.data["feature_scaler"],
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                args=self.config,
                device=self.device,
            )
            for client_i in range(self.num_clients)
        ]
        self.gat = FedOSTCGAT().to(self.device)
        self.rho_history: dict[int, torch.Tensor] = {}
        self.prediction_history: dict[int, dict[int, torch.Tensor]] = {}
        self.w_pred: dict[int, dict[str, torch.Tensor]] = {}
        self.w_update: dict[int, dict[str, torch.Tensor]] = {}
        self.global_model = clone_state(self.clients[0].model.state_dict())

    def run(self) -> None:
        rounds = self._resolve_rounds()
        writer = RoundMetricWriter(self.config.dataset, self.config.pred_steps, self.config.agg_model)

        for rround in range(1, rounds + 1):
            print("**** Round {}/{} ****".format(rround, rounds))
            train_log = self.train_round(rround)
            log = train_log["log"]
            writer.write_round(
                rround,
                self._metric_scalar(log["rmse"]),
                self._metric_scalar(log["mae"]),
            )
            print("prediction rmse is: {} ".format(self._metric_scalar(log["rmse"])))

        self.flush(rounds)
        writer.close()

    def train_round(self, rround: int) -> dict[str, Any]:
        self.update_train_data(rround, self.clients)
        self.predict_current(rround)
        log = self.evaluate_prediction_round(rround)

        if rround > self.delay:
            tau = rround - self.delay
            self.update_from_feedback(tau)
        return {"loss": torch.tensor(0).float(), "progress_bar": log, "log": log}

    def update_train_data(self, rround: int, clients: list[FedOSTCClient]) -> None:
        for client in clients:
            client_id = int(client.client_id)
            client.current_dataset = TensorDataset(self._current_x(rround, client_id))
            if rround > self.delay:
                tau = rround - self.delay
                client.delayed_dataset = TensorDataset(
                    self._current_x(tau, client_id),
                    self._current_y(tau, client_id),
                )
            else:
                client.delayed_dataset = None

    def predict_current(self, rround: int) -> dict[int, torch.Tensor]:
        current_state = clone_state(self.global_model)
        self._record_state_snapshot(self.w_pred, int(rround), current_state)
        for client in self.clients:
            client.load_model_state(current_state)

        hidden = self.collect_hidden("current_dataset", encode_method="encode_current")
        refined = self.server_gat(hidden)
        predictions: dict[int, torch.Tensor] = {}
        for index, client in enumerate(self.clients):
            predictions[int(client.client_id)] = client.evaluate_current(
                client.current_dataset,
                hidden[:, index : index + 1, :],
                refined[:, index : index + 1, :],
            ).detach().cpu()
        self.prediction_history[int(rround)] = predictions
        self._evict_oldest(self.prediction_history, self._prediction_history_keep())
        del hidden, refined
        return predictions

    def evaluate_prediction_round(self, rround: int) -> dict[str, Any]:
        rround = int(rround)
        if rround not in self.prediction_history:
            raise KeyError("missing prediction for evaluation round {}".format(rround))
        predictions = self.prediction_history[rround]
        if len(predictions) != self.num_clients:
            raise ValueError("evaluation round {} does not have all client predictions".format(rround))

        local_logs = []
        for client in self.clients:
            client_id = int(client.client_id)
            if client_id not in predictions:
                raise ValueError("evaluation round {} is missing a client prediction".format(rround))
            target = self._current_y(rround, client_id).detach().cpu()
            local_log = metric_values(
                predictions[client_id].detach().cpu(),
                target,
                self.data["feature_scaler"],
            )
            local_log["num_samples"] = int(target.size(0))
            local_logs.append(local_log)

        return self.aggregate_local_logs(local_logs)

    def update_from_feedback(self, tau: int) -> dict[str, Any]:
        tau = int(tau)
        if tau not in self.prediction_history:
            raise KeyError("missing prediction for feedback sample {}".format(tau))
        predictions = self.prediction_history[tau]
        if len(predictions) != self.num_clients:
            raise ValueError("feedback sample {} does not have all client predictions".format(tau))

        update_model_state = clone_state(self.global_model)
        self._record_state_snapshot(self.w_update, tau, update_model_state)
        for client in self.clients:
            client.load_model_state(update_model_state)
            if getattr(client, "delayed_dataset", None) is None:
                client.delayed_dataset = TensorDataset(
                    self._current_x(tau, int(client.client_id)),
                    self._current_y(tau, int(client.client_id)),
                )

        delayed_hidden = self.collect_hidden("delayed_dataset", encode_method="encode_delayed")
        delayed_refined = self.server_gat(delayed_hidden)

        local_states = []
        total_samples = 0
        for index, client in enumerate(self.clients):
            client_id = int(client.client_id)
            if client_id not in predictions:
                raise ValueError("feedback sample {} is missing a client prediction".format(tau))
            delayed_batch = client.delayed_dataset
            if delayed_batch is None or len(delayed_batch) == 0:
                raise ValueError("feedback sample {} is missing delayed data".format(tau))
            result = client.train_delayed(
                delayed_batch,
                delayed_refined[:, index : index + 1, :],
                update_model_state,
            )
            local_states.append(clone_state(result["state_dict"]))
            total_samples += int(result["log"]["num_samples"])

        self.global_model = self.aggregate_for_tau(local_states, tau)
        self._evict_feedback_history(tau)
        del delayed_hidden, delayed_refined, local_states
        return {"num_samples": total_samples}

    def collect_hidden(self, dataset_attr: str, *, encode_method: str) -> torch.Tensor:
        parts = []
        for client in self.clients:
            dataset = getattr(client, dataset_attr)
            if dataset is None or len(dataset) == 0:
                raise ValueError("{} is missing for client {}".format(dataset_attr, client.client_id))
            hidden = getattr(client, encode_method)(dataset).to(self.device)
            parts.append(hidden)
        return torch.cat(parts, dim=1)

    def server_gat(self, hidden: torch.Tensor) -> torch.Tensor:
        graph = {}
        if "edge_index" in self.data:
            graph["edge_index"] = self.data["edge_index"]
        if "adj_mx" in self.data:
            graph["adj_mx"] = self.data["adj_mx"]
        return self.gat(hidden.to(self.device), graph=graph)

    def flush(self, rounds: int) -> None:
        start_tau = max(1, int(rounds) - self.delay + 1)
        for tau in range(start_tau, int(rounds) + 1):
            if tau in self.prediction_history:
                for client in self.clients:
                    client.delayed_dataset = TensorDataset(
                        self._current_x(tau, int(client.client_id)),
                        self._current_y(tau, int(client.client_id)),
                    )
                self.update_from_feedback(tau)

    def aggregate_for_tau(
        self,
        local_states: list[Mapping[str, torch.Tensor]],
        tau: int,
    ) -> dict[str, torch.Tensor]:
        if not hasattr(self, "rho_history"):
            self.rho_history = {}
        if len(local_states) != self.num_clients:
            raise ValueError("FedOSTC aggregation requires every client state")
        fresh_global = self.average_aggregate(local_states)
        distances = self.compute_model_distances(local_states, fresh_global)
        self.rho_history[int(tau)] = self.compute_rho(distances)

        if int(tau) <= self.period_steps:
            self._evict_rho_history(int(tau))
            return fresh_global

        rho_key = int(tau) + 1 - self.period_steps
        if rho_key not in self.rho_history:
            raise KeyError("missing rho_history[{}]".format(rho_key))
        result = self.period_aggregate(local_states, self.rho_history[rho_key])
        self._evict_rho_history(int(tau))
        return result

    def average_aggregate(
        self,
        local_states: list[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return average_aggregate(local_states)

    def compute_model_distances(
        self,
        local_states: list[Mapping[str, torch.Tensor]],
        fresh_global: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return compute_model_distances(local_states, fresh_global)

    def compute_rho(self, distances: torch.Tensor) -> torch.Tensor:
        return compute_rho(distances)

    def period_aggregate(
        self,
        local_states: list[Mapping[str, torch.Tensor]],
        rho: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return period_aggregate(local_states, rho)

    def _current_x(self, sample_id: int, client_id: int) -> torch.Tensor:
        return self.data["x"][int(sample_id) - 1 : int(sample_id), :, client_id : client_id + 1, :]

    def _current_y(self, sample_id: int, client_id: int) -> torch.Tensor:
        return self.data["y"][int(sample_id) - 1 : int(sample_id), :, client_id : client_id + 1, :]

    def _state_history_keep(self) -> int:
        return max(0, int(getattr(self.config, "fedostc_state_history_keep", 1)))

    def _prediction_history_keep(self) -> int:
        delay = getattr(self, "delay", None)
        if delay is None:
            delay = self._resolve_label_delay()
        return max(1, int(delay) + 1)

    @staticmethod
    def _evict_oldest(history: dict[int, Any], keep: int) -> None:
        if keep < 0:
            return
        while len(history) > keep:
            del history[min(history)]

    def _record_state_snapshot(
        self,
        history: dict[int, dict[str, torch.Tensor]],
        key: int,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        keep = self._state_history_keep()
        if keep <= 0:
            return
        history[int(key)] = clone_state(state)
        self._evict_oldest(history, keep)

    def _evict_feedback_history(self, tau: int) -> None:
        self.prediction_history.pop(int(tau), None)
        self._evict_oldest(self.prediction_history, self._prediction_history_keep())

    def _evict_rho_history(self, tau: int) -> None:
        if self.period_steps <= 0:
            return
        next_required_key = int(tau) + 2 - int(self.period_steps)
        for key in list(self.rho_history):
            if key < next_required_key:
                del self.rho_history[key]

    @staticmethod
    def no_label_log() -> dict[str, Any]:
        nan = torch.tensor(float("nan")).float()
        return {"num_samples": 0, "mse": nan.clone(), "rmse": nan.clone(), "mae": nan.clone()}

    @staticmethod
    def _metric_scalar(value: Any) -> float:
        if torch.is_tensor(value):
            value = value.item()
        return float(value)


def validate_state_list(local_states: list[Mapping[str, torch.Tensor]]) -> None:
    if not local_states:
        raise ValueError("local_states must not be empty")
    names = set(local_states[0].keys())
    for state in local_states:
        if set(state.keys()) != names:
            raise ValueError("all state dictionaries must have the same keys")
        for name in names:
            if state[name].shape != local_states[0][name].shape:
                raise ValueError("state tensor shape mismatch for {}".format(name))


def average_aggregate(
    local_states: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    validate_state_list(local_states)
    aggregated = {}
    for name, first_tensor in local_states[0].items():
        if torch.is_floating_point(first_tensor):
            stacked = torch.stack(
                [state[name].detach().to(first_tensor.device) for state in local_states],
                dim=0,
            )
            aggregated[name] = stacked.mean(dim=0).to(dtype=first_tensor.dtype).clone()
        else:
            aggregated[name] = first_tensor.detach().clone()
    return aggregated


def compute_model_distances(
    local_states: list[Mapping[str, torch.Tensor]],
    fresh_global: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    validate_state_list(local_states)
    distances = []
    for state in local_states:
        total = torch.tensor(0.0, dtype=torch.float64)
        for name, local_tensor in state.items():
            if torch.is_floating_point(local_tensor):
                diff = local_tensor.detach().cpu().double() - fresh_global[name].detach().cpu().double()
                total = total + torch.sum(diff * diff)
        distances.append(torch.sqrt(total))
    return torch.stack(distances)


def compute_rho(distances: torch.Tensor) -> torch.Tensor:
    distances = torch.as_tensor(distances, dtype=torch.float64)
    if distances.numel() == 0:
        raise ValueError("distances must not be empty")
    return torch.softmax(-distances, dim=0)


def period_aggregate(
    local_states: list[Mapping[str, torch.Tensor]],
    rho: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_state_list(local_states)
    rho = torch.as_tensor(rho, dtype=torch.float64)
    if rho.numel() != len(local_states):
        raise ValueError("rho length must match local_states")
    aggregated = {}
    for name, first_tensor in local_states[0].items():
        if torch.is_floating_point(first_tensor):
            total = torch.zeros_like(first_tensor, dtype=torch.float64)
            for weight, state in zip(rho, local_states):
                total = total + state[name].detach().to(first_tensor.device).double() * weight.to(first_tensor.device)
            aggregated[name] = total.to(dtype=first_tensor.dtype).clone()
        else:
            aggregated[name] = first_tensor.detach().clone()
    return aggregated


class RoundMetricWriter:
    def __init__(self, dataset: str, pred_steps: int, agg_model: str) -> None:
        self.workbook = None
        self.sheet = None
        self.file_handle = None
        self.csv_writer = None
        self.row = 1
        try:
            import xlsxwriter

            self.workbook = xlsxwriter.Workbook(
                "{}_{}.xlsx".format(dataset, pred_steps),
                {"nan_inf_to_errors": True},
            )
            self.sheet = self.workbook.add_worksheet("round")
            self.sheet.write(0, 0, "round_num")
            self.sheet.write(0, 1, "{}_rmse".format(agg_model))
            self.sheet.write(0, 2, "{}_mae".format(agg_model))
        except ModuleNotFoundError:
            self.file_handle = open("{}_{}.csv".format(dataset, pred_steps), "w", newline="")
            self.csv_writer = csv.writer(self.file_handle)
            self.csv_writer.writerow(["round_num", "{}_rmse".format(agg_model), "{}_mae".format(agg_model)])

    def write_round(self, rround: int, rmse: float, mae: float) -> None:
        if self.sheet is not None:
            self.sheet.write(self.row, 0, rround)
            self.sheet.write(self.row, 1, rmse)
            self.sheet.write(self.row, 2, mae)
        else:
            self.csv_writer.writerow([rround, rmse, mae])
        self.row += 1

    def close(self) -> None:
        if self.workbook is not None:
            self.workbook.close()
        if self.file_handle is not None:
            self.file_handle.close()


__all__ = [
    "FedOSTC",
    "average_aggregate",
    "compute_model_distances",
    "compute_rho",
    "period_aggregate",
]
