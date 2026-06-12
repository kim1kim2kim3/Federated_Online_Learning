"""Phase2 FedOSTC spatial hidden-state update.

This module implements the server-side Eq. (8)-(10) operation over client
hidden states.  The projection function from Eq. (8) is unresolved in the
paper, so this Phase2 prerequisite uses the documented fixed assumption:
``a([h_n || h_m]) = mean([h_n || h_m])``.  The module intentionally has no
learned parameter or buffer state.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FedOSTCGAT(nn.Module):
    """FedOSTC Eq. (8)-(10) update for tensors shaped ``[batch, node, hidden]``."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        hidden: torch.Tensor,
        graph: dict[str, Any] | None = None,
        *,
        edge_index: torch.Tensor | None = None,
        adj_mx: torch.Tensor | Any | None = None,
    ) -> torch.Tensor:
        """Return spatially refined hidden states.

        Edges are interpreted as ``source -> target``.  For every target node,
        attention is normalized over its incoming source-neighbor set after
        self-loops are added.
        """
        if hidden.dim() != 3:
            raise ValueError("hidden must have shape [batch, node, hidden]")
        if hidden.size(1) <= 0:
            raise ValueError("hidden must include at least one node")

        node_count = hidden.size(1)
        if graph is not None and not isinstance(graph, dict):
            if edge_index is None and adj_mx is None:
                candidate = torch.as_tensor(graph)
                if (
                    candidate.dim() == 2
                    and candidate.size(0) == 2
                    and candidate.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
                ):
                    edge_index = graph
                else:
                    adj_mx = graph
            graph = None
        edges = self._resolve_edges(
            node_count=node_count,
            device=hidden.device,
            graph=graph,
            edge_index=edge_index,
            adj_mx=adj_mx,
        )

        refined = []
        for target in range(node_count):
            target_sources = edges[0, edges[1] == target]
            source_hidden = hidden[:, target_sources, :]
            target_hidden = hidden[:, target : target + 1, :].expand_as(source_hidden)
            projected = torch.cat((target_hidden, source_hidden), dim=-1).mean(dim=-1)
            scores = F.leaky_relu(projected, negative_slope=0.2)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            refined.append(torch.sigmoid((weights * source_hidden).sum(dim=1)))

        return torch.stack(refined, dim=1)

    @staticmethod
    def _resolve_edges(
        *,
        node_count: int,
        device: torch.device,
        graph: dict[str, Any] | None,
        edge_index: torch.Tensor | None,
        adj_mx: torch.Tensor | Any | None,
    ) -> torch.Tensor:
        if graph is not None:
            if edge_index is None and "edge_index" in graph:
                edge_index = graph["edge_index"]
            if adj_mx is None and "adj_mx" in graph:
                adj_mx = graph["adj_mx"]

        if edge_index is not None:
            edges = torch.as_tensor(edge_index, device=device, dtype=torch.long)
            if edges.dim() != 2 or edges.size(0) != 2:
                raise ValueError("edge_index must have shape [2, edge]")
        elif adj_mx is not None:
            adjacency = torch.as_tensor(adj_mx, device=device)
            if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
                raise ValueError("adj_mx must be a square matrix")
            if adjacency.size(0) != node_count:
                raise ValueError("adj_mx node dimension must match hidden")
            edges = adjacency.ne(0).nonzero(as_tuple=False).t().to(dtype=torch.long)
        else:
            edges = torch.empty((2, 0), device=device, dtype=torch.long)

        if edges.numel() > 0:
            if edges.min().item() < 0 or edges.max().item() >= node_count:
                raise ValueError("edge endpoint is outside hidden node dimension")

        loops = torch.arange(node_count, device=device, dtype=torch.long).repeat(2, 1)
        edges = torch.cat((edges, loops), dim=1)
        return torch.unique(edges, dim=1)


FedOSTCServerGAT = FedOSTCGAT


__all__ = ["FedOSTCGAT", "FedOSTCServerGAT"]
