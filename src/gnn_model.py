from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool


class MusicGNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        gnn_type: str = "graphsage",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()

        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if self.gnn_type == "gat":
                conv = GATConv(dims[i], dims[i + 1], heads=4, concat=False, dropout=dropout)
            else:
                conv = SAGEConv(dims[i], dims[i + 1])
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        for conv, norm in zip(self.convs, self.norms):
            residual = x if x.size(-1) == conv.out_channels else None
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual is not None and residual.shape == x.shape:
                x = x + residual

        pooled = global_mean_pool(x, batch)
        return x, pooled
