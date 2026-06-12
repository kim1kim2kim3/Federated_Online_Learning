"""Phase3 FedOSTC client prerequisite.

The client owns only the temporal encoder/decoder model and the delayed local
manual gradient step.  Full stream scheduling, all-client coordination, server
runtime, and aggregation remain later phases.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fedostc_model import FedOSTCClientModel


class FedOSTCClient:
    """FedOSTC-specific client wrapper around :class:`FedOSTCClientModel`."""

    def __init__(
        self,
        client_id: int | None = None,
        client_dataset: Any | None = None,
        feature_scaler: Any | None = None,
        input_size: int | None = None,
        output_size: int | None = None,
        args: Any | None = None,
        model: nn.Module | None = None,
        *,
        input_dim: int = 1,
        output_dim: int = 1,
        pred_steps: int | None = None,
        lr: float | None = None,
        local_epochs: int | None = None,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_dataset = client_dataset
        self.feature_scaler = feature_scaler
        self.args = args
        if input_size is not None:
            input_dim = input_size
        if output_size is not None:
            output_dim = output_size
        self.lr = float(lr if lr is not None else getattr(args, "lr", 0.001))
        self.local_epochs = int(
            local_epochs if local_epochs is not None else getattr(args, "epoch", 5)
        )
        self.batch_size = int(
            batch_size if batch_size is not None else getattr(args, "batch_size", 1)
        )
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if pred_steps is None:
            pred_steps = int(getattr(args, "pred_steps", 12))
        if model is None:
            options = dict(model_kwargs or {})
            options.setdefault(
                "encoder_hidden_size",
                int(getattr(args, "fedostc_encoder_hidden_size", 64)),
            )
            options.setdefault(
                "decoder_hidden_size",
                int(getattr(args, "fedostc_decoder_hidden_size", 128)),
            )
            options.setdefault("gru_num_layers", int(getattr(args, "num_layers", 1)))
            options.setdefault("dropout", float(getattr(args, "dropout", 0.0)))
            model = FedOSTCClientModel(
                input_dim=input_dim,
                output_dim=output_dim,
                pred_steps=pred_steps,
                **options,
            )
        self.model = model.to(self.device)
        self.state_dict = self._snapshot_model_state()
        self.last_result: dict[str, Any] | None = None
        self.local_result: dict[str, Any] | None = None

    def load_model_state(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load the global update model before prediction or delayed training."""
        if state_dict is None:
            raise ValueError("state_dict is required")
        self.model.load_state_dict(state_dict)
        self.state_dict = self._snapshot_model_state()

    def encode_current(self, batch_or_dataset: Any) -> torch.Tensor:
        """Encode current speed history and return ``[batch, node, hidden]``."""
        return self._encode_batches(batch_or_dataset)

    def evaluate_current(
        self,
        batch_or_dataset: Any,
        h: torch.Tensor,
        h_prime: torch.Tensor,
    ) -> torch.Tensor:
        """Predict current future speed without reading current labels."""
        self.model.eval()
        outputs: list[torch.Tensor] = []
        offset = 0
        with torch.no_grad():
            for batch in self._iter_batches(batch_or_dataset):
                speed = self._speed(batch)
                part_size = speed.size(0)
                data = {"x": speed.to(self.device)}
                h_part = self._hidden_part(h, offset, part_size)
                h_prime_part = self._hidden_part(h_prime, offset, part_size)
                outputs.append(self.model(data, h=h_part, h_prime=h_prime_part))
                offset += part_size
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

    def encode_delayed(self, batch_or_dataset: Any) -> torch.Tensor:
        """Encode delayed speed history for the server hidden-state update."""
        return self._encode_batches(batch_or_dataset)

    def train_delayed(
        self,
        batch_or_dataset: Any,
        h_prime: torch.Tensor,
        update_model_state: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run delayed-label local OGD from the supplied update model."""
        if update_model_state is None:
            raise ValueError("update_model_state is required")
        self.load_model_state(update_model_state)
        fixed_h_prime = h_prime.detach().to(self.device)

        criterion = self._criterion()
        total_loss = 0.0
        total_samples = 0
        self.model.train()
        for _ in range(self.local_epochs):
            offset = 0
            for batch in self._iter_batches(batch_or_dataset):
                speed = self._speed(batch).to(self.device)
                target = self._target(batch).to(self.device)
                part_size = speed.size(0)
                h_prime_part = self._hidden_part(fixed_h_prime, offset, part_size)

                self.model.zero_grad(set_to_none=True)
                data = {"x": speed}
                h = self.model.encode(data)
                prediction = self.model.decode(data, h, h_prime_part)
                loss = criterion(prediction, target)
                loss.backward()

                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.add_(param.grad, alpha=-self.lr)
                self.model.zero_grad(set_to_none=True)

                total_loss += float(loss.detach().cpu()) * part_size
                total_samples += part_size
                offset += part_size

        self.state_dict = self._snapshot_model_state()
        log = {
            "loss": total_loss / total_samples if total_samples else 0.0,
            "num_samples": total_samples,
        }
        self.last_result = {"state_dict": self.state_dict, "log": log}
        self.local_result = self.last_result
        return self.last_result

    def _encode_batches(self, batch_or_dataset: Any) -> torch.Tensor:
        self.model.eval()
        hidden_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in self._iter_batches(batch_or_dataset):
                speed = self._speed(batch).to(self.device)
                hidden_parts.append(self.model.encode({"x": speed}))
        return torch.cat(hidden_parts, dim=0) if len(hidden_parts) > 1 else hidden_parts[0]

    def _iter_batches(self, batch_or_dataset: Any) -> Iterable[Any]:
        if batch_or_dataset is None:
            raise ValueError("batch_or_dataset is required")
        if self._looks_like_batch(batch_or_dataset):
            return (batch_or_dataset,)
        if hasattr(batch_or_dataset, "__getitem__") and hasattr(batch_or_dataset, "__len__"):
            return DataLoader(batch_or_dataset, batch_size=self.batch_size)
        return batch_or_dataset

    @staticmethod
    def _looks_like_batch(value: Any) -> bool:
        if isinstance(value, Mapping):
            return "x" in value
        if isinstance(value, (tuple, list)) and value:
            return torch.is_tensor(value[0])
        return False

    @staticmethod
    def _with_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor

    def _speed(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, Mapping):
            value = batch["x"]
        else:
            value = batch[0]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return self._with_batch_dim(value)

    def _target(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, Mapping):
            value = batch["y"]
        else:
            if len(batch) < 2:
                raise ValueError("delayed training batch must include labels")
            value = batch[1]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return self._with_batch_dim(value)

    def _hidden_part(self, hidden: torch.Tensor, offset: int, part_size: int) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError("hidden inputs must have shape [batch, node, hidden]")
        end = offset + part_size
        if end > hidden.size(0):
            raise ValueError("hidden batch dimension is shorter than the data batch")
        if offset == 0 and part_size == hidden.size(0):
            return hidden.to(self.device)
        return hidden[offset:end].to(self.device)

    def _criterion(self) -> nn.Module:
        name = getattr(self.args, "loss_func", "mse")
        if name == "mae":
            return nn.L1Loss()
        if name == "smae":
            return nn.SmoothL1Loss()
        return nn.MSELoss()

    def _snapshot_model_state(self) -> dict[str, torch.Tensor]:
        """Store long-lived client state snapshots on CPU, not CUDA."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }


__all__ = ["FedOSTCClient"]
