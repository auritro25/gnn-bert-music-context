from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_encoder import TextEncoder
from .gnn_model import MusicGNNEncoder


class GraphTextContrastiveModel(nn.Module):
    def __init__(
        self,
        gnn: MusicGNNEncoder,
        text_encoder: TextEncoder,
        hidden_dim: int,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.gnn = gnn
        self.text_encoder = text_encoder
        self.graph_proj = nn.Linear(hidden_dim, projection_dim)
        self.text_proj = nn.Linear(hidden_dim, projection_dim)
        self.temperature = temperature

    def encode_graph(self, graph):
        _, pooled = self.gnn(graph)
        return F.normalize(self.graph_proj(pooled), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        _, cls = self.text_encoder(input_ids, attention_mask)
        return F.normalize(self.text_proj(cls), dim=-1)

    def forward(self, batch):
        g = self.encode_graph(batch["graph"])
        t = self.encode_text(batch["input_ids"], batch["attention_mask"])
        logits = g @ t.T / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_g2t = F.cross_entropy(logits, targets)
        loss_t2g = F.cross_entropy(logits.T, targets)
        loss = 0.5 * (loss_g2t + loss_t2g)
        return {"loss": loss, "graph_embedding": g, "text_embedding": t, "similarity": logits}
