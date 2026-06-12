from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import TensorDataset

from base_server import BaseFLServer
from client_fedavg import FedAvgClient
from utils.process_increment import load_dataset


def clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
    }


class FedAvg(BaseFLServer):
    """Partial-participation FedAvg under the delayed-label protocol.

    Each client keeps a persistent local history ``P_k`` of samples whose labels
    have arrived.  Every round still evaluates the current sample for every
    client from the current global model, while ``fedavg_client_fraction`` only
    controls which history-bearing clients perform local FedAvg training and
    contribute to aggregation.
    """

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
        self.input_size = int(self.data["x"].shape[-1] + self.data["x_attr"].shape[-1])
        self.output_size = int(self.data["y"].shape[-1])
        self.max_epoch = int(data["x"].shape[0])
        self.train_per_num_samples = 1
        self.delay = self._resolve_label_delay()

        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))

        self.clients = []
        self.fedavg_history_indices: dict[int, list[int]] = {}
        for client_i in range(self.num_clients):
            initial_dataset = TensorDataset(
                data["x"][: self.train_per_num_samples, :, client_i : client_i + 1, :],
                data["y"][: self.train_per_num_samples, :, client_i : client_i + 1, :],
                data["x_attr"][: self.train_per_num_samples, :, client_i : client_i + 1, :],
                data["y_attr"][: self.train_per_num_samples, :, client_i : client_i + 1, :],
            )
            self.clients.append(
                FedAvgClient(
                    client_id=client_i,
                    client_dataset=initial_dataset,
                    feature_scaler=self.data["feature_scaler"],
                    input_size=self.input_size,
                    output_size=self.output_size,
                    args=self.config,
                )
            )
            history: list[int] = []
            self.fedavg_history_indices[client_i] = history
            self.clients[-1].fedavg_history_indices = history

        self.global_model = clone_state(self.clients[0].model.state_dict())

    def train_round(self, rround: int) -> dict[str, Any]:
        self.update_train_data(rround, self.clients)
        selected_client_ids = self.select_clients(rround)
        print("selected clients：", selected_client_ids)
        local_logs, local_states = self.local_execute(selected_client_ids)
        self.aggregate(local_states, rround)
        agg_log = self.aggregate_local_logs(local_logs)
        return {
            "loss": torch.tensor(0).float(),
            "progress_bar": agg_log,
            "log": agg_log,
        }

    def select_clients(self, rround: int | None = None) -> list[int]:
        """Sample history-bearing clients using FedAvg's C fraction.

        The default ``fedavg_client_fraction=1.0`` preserves all-client FedAvg.
        Smaller values implement standard FedAvg partial participation over
        clients whose persistent local dataset ``P_k`` is non-empty.
        Current-round evaluation is still performed for every client; this
        selection controls only local history training and server aggregation.
        """
        histories = self._ensure_history_storage()
        eligible = [
            int(client.client_id)
            for client in self.clients
            if len(histories.get(int(client.client_id), [])) > 0
        ]
        if not eligible:
            self.last_selected_client_ids = []
            return []

        fraction = self._client_fraction()
        if fraction >= 1.0:
            selected_client_ids = eligible
        else:
            sample_size = max(1, math.ceil(len(eligible) * fraction))
            rng = np.random.default_rng(int(self.config.seed) + int(rround or 0))
            selected_client_ids = sorted(
                int(client_id)
                for client_id in rng.choice(eligible, size=sample_size, replace=False)
            )
        self.last_selected_client_ids = selected_client_ids
        return selected_client_ids

    def local_execute(
        self,
        selected_client_ids: list[int],
    ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, torch.Tensor], int]]]:
        local_logs = []
        local_states = []
        selected_id_set = set(selected_client_ids)
        histories = self._ensure_history_storage()
        for client in self.clients:
            client_id = int(client.client_id)
            history_dataset = None
            if client_id in selected_id_set:
                history_indices = histories.get(client_id, [])
                if history_indices:
                    history_dataset = self._make_history_dataset(client_id, history_indices)
            result = client.run_round(
                global_state=self.global_model,
                current_dataset=client.current_dataset,
                delayed_dataset=history_dataset,
            )
            local_logs.append(result["log"])
            if history_dataset is not None:
                local_states.append((clone_state(result["state_dict"]), len(history_dataset)))
        return local_logs, local_states

    def update_train_data(self, rround: int, clients: list[FedAvgClient]) -> None:
        histories = self._ensure_history_storage()
        for client in clients:
            client_id = int(client.client_id)
            client.current_dataset = TensorDataset(
                self.data["x"][
                    int(rround) - 1 : self.train_per_num_samples + int(rround) - 1,
                    :,
                    client_id : client_id + 1,
                    :,
                ],
                self.data["y"][
                    int(rround) - 1 : self.train_per_num_samples + int(rround) - 1,
                    :,
                    client_id : client_id + 1,
                    :,
                ],
                self.data["x_attr"][
                    int(rround) - 1 : self.train_per_num_samples + int(rround) - 1,
                    :,
                    client_id : client_id + 1,
                    :,
                ],
                self.data["y_attr"][
                    int(rround) - 1 : self.train_per_num_samples + int(rround) - 1,
                    :,
                    client_id : client_id + 1,
                    :,
                ],
            )
            client.delayed_dataset = None

            if int(rround) > self.delay:
                train_idx = int(rround) - 1 - self.delay
                histories.setdefault(client_id, []).append(train_idx)

    def aggregate(
        self,
        local_states: list[tuple[Mapping[str, torch.Tensor], int]],
        rround: int,
    ) -> None:
        del rround
        if not local_states:
            return
        total_samples = sum(num_samples for _, num_samples in local_states)
        if total_samples <= 0:
            return

        names = set(local_states[0][0].keys())
        for state, _ in local_states:
            if set(state.keys()) != names:
                raise ValueError("all FedAvg state dictionaries must have the same keys")

        aggregated: dict[str, torch.Tensor] = {}
        first_state = local_states[0][0]
        for name, first_tensor in first_state.items():
            if torch.is_floating_point(first_tensor):
                total = torch.zeros_like(first_tensor, dtype=torch.float64)
                for state, num_samples in local_states:
                    weight = float(num_samples) / float(total_samples)
                    total = total + state[name].detach().cpu().double() * weight
                aggregated[name] = total.to(dtype=first_tensor.dtype).clone()
            else:
                aggregated[name] = first_tensor.detach().cpu().clone()

        self.global_model = aggregated

    def _client_fraction(self) -> float:
        fraction = float(getattr(self.config, "fedavg_client_fraction", 1.0))
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fedavg_client_fraction must satisfy 0 < C <= 1.")
        return fraction

    def _ensure_history_storage(self) -> dict[int, list[int]]:
        histories = getattr(self, "fedavg_history_indices", None)
        if histories is None:
            histories = {}
        elif not isinstance(histories, dict):
            histories = {
                int(client_id): list(history)
                for client_id, history in enumerate(histories)
            }

        for client in getattr(self, "clients", []):
            client_id = int(client.client_id)
            history = histories.get(client_id)
            if history is None:
                existing = getattr(client, "fedavg_history_indices", None)
                history = list(existing) if existing is not None else []
                histories[client_id] = history
            client.fedavg_history_indices = history
        self.fedavg_history_indices = histories
        return histories

    def _make_history_dataset(self, client_id: int, history_indices: list[int]) -> TensorDataset:
        return TensorDataset(
            self.data["x"][
                history_indices,
                :,
                client_id : client_id + 1,
                :,
            ],
            self.data["y"][
                history_indices,
                :,
                client_id : client_id + 1,
                :,
            ],
            self.data["x_attr"][
                history_indices,
                :,
                client_id : client_id + 1,
                :,
            ],
            self.data["y_attr"][
                history_indices,
                :,
                client_id : client_id + 1,
                :,
            ],
        )


__all__ = ["FedAvg", "FedAvgClient", "clone_state"]
