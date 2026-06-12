"""Phase1 FedOSTC client model.

This module implements only the client-side temporal encoder and decoder.
Spatial hidden-state refinement is provided by a later server component, so
``forward`` requires both the original and refined hidden states.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FedOSTCClientModel(nn.Module):
    """Client-side GRU encoder/decoder for the Phase1 FedOSTC path."""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        pred_steps: int = 12,
        encoder_hidden_size: int = 64,
        decoder_hidden_size: int = 128,
        gru_num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pred_steps <= 0:
            raise ValueError("pred_steps must be positive")
        if encoder_hidden_size * 2 != decoder_hidden_size:
            raise ValueError(
                "Phase1 concat-repeat bridge requires decoder_hidden_size "
                "to equal 2 * encoder_hidden_size"
            )

        recurrent_dropout = dropout if gru_num_layers > 1 else 0.0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_steps = pred_steps
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=encoder_hidden_size,
            num_layers=gru_num_layers,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.GRU(
            input_size=decoder_hidden_size,
            hidden_size=decoder_hidden_size,
            num_layers=gru_num_layers,
            dropout=recurrent_dropout,
        )
        self.output_head = nn.Linear(decoder_hidden_size, output_dim)

    def encode(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode historical traffic speed into ``[batch, node, hidden]``."""
        x = data["x"]
        if x.dim() != 4:
            raise ValueError("data['x'] must have shape [batch, steps, node, feature]")
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got feature dim {x.size(-1)}"
            )

        batch_size, _, node_count, _ = x.shape
        sequence = x.permute(1, 0, 2, 3).reshape(x.size(1), batch_size * node_count, -1)
        _, hidden = self.encoder(sequence)
        last_hidden = hidden[-1]
        return last_hidden.reshape(batch_size, node_count, self.encoder_hidden_size)

    def decode(
        self,
        data: dict[str, torch.Tensor],
        h: torch.Tensor,
        h_prime: torch.Tensor,
    ) -> torch.Tensor:
        """Decode from concatenated original/refined hidden states.

        The Phase1 ambiguity decision is concat-repeat: concatenate ``h`` and
        ``h_prime`` on the feature axis, repeat that fused vector for each
        forecast step, run the decoder GRU, and apply a fully connected head.
        """
        del data
        if h.shape != h_prime.shape:
            raise ValueError("h and h_prime must have identical shapes")
        if h.dim() != 3:
            raise ValueError("h and h_prime must have shape [batch, node, hidden]")
        if h.size(-1) != self.encoder_hidden_size:
            raise ValueError(
                f"expected hidden dim {self.encoder_hidden_size}, got {h.size(-1)}"
            )

        batch_size, node_count, _ = h.shape
        fused = torch.cat((h, h_prime), dim=-1)
        decoder_input = (
            fused.reshape(batch_size * node_count, self.decoder_hidden_size)
            .unsqueeze(0)
            .repeat(self.pred_steps, 1, 1)
        )
        decoder_hidden, _ = self.decoder(decoder_input)
        out = self.output_head(decoder_hidden)
        return out.reshape(
            self.pred_steps,
            batch_size,
            node_count,
            self.output_dim,
        ).permute(1, 0, 2, 3)

    def forward(
        self,
        data: dict[str, torch.Tensor],
        h: torch.Tensor | None = None,
        h_prime: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode only when both Algorithm 2 hidden inputs are supplied."""
        if h is None or h_prime is None:
            raise ValueError(
                "FedOSTC Phase1 forward requires both h and h_prime; "
                "server-side refinement is not implemented in this model"
            )
        return self.decode(data, h, h_prime)
