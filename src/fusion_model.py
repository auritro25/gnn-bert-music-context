from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from .bert_encoder import TextEncoder
from .gnn_model import MusicGNNEncoder


class BERTTagClassifier(nn.Module):
    def __init__(self, text_encoder: TextEncoder, hidden_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.text_encoder = text_encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, batch):
        _, cls = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        return {"logits": self.head(cls), "embedding": cls}


class GNNTagClassifier(nn.Module):
    def __init__(self, gnn: MusicGNNEncoder, hidden_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.gnn = gnn
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, batch):
        _, pooled = self.gnn(batch["graph"])
        return {"logits": self.head(pooled), "embedding": pooled}


class EarlyConcatFusion(nn.Module):
    def __init__(
        self,
        gnn: MusicGNNEncoder,
        text_encoder: TextEncoder,
        hidden_dim: int,
        num_labels: int,
        dropout: float,
    ):
        super().__init__()
        self.gnn = gnn
        self.text_encoder = text_encoder
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, num_labels)

    def forward(self, batch):
        _, g = self.gnn(batch["graph"])
        _, t = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        z = self.fusion(torch.cat([g, t], dim=-1))
        return {"logits": self.head(z), "embedding": z}


class CrossAttentionFusion(nn.Module):
    """
    Graph nodes query BERT token embeddings.
    This preserves segment-level graph information before graph-level fusion.
    """

    def __init__(
        self,
        gnn: MusicGNNEncoder,
        text_encoder: TextEncoder,
        hidden_dim: int,
        num_labels: int,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by fusion_heads.")

        self.gnn = gnn
        self.text_encoder = text_encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, num_labels)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, batch):
        node_h, graph_pooled = self.gnn(batch["graph"])
        text_tokens, text_cls = self.text_encoder(batch["input_ids"], batch["attention_mask"])

        dense_nodes, node_mask = to_dense_batch(node_h, batch["graph"].batch)
        text_key_padding = ~batch["attention_mask"].bool()

        attended, attn_weights = self.cross_attn(
            query=dense_nodes,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_key_padding,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = self.norm(dense_nodes + attended)
        cross_pooled = self._masked_mean(attended, node_mask)

        z = self.fusion(torch.cat([graph_pooled, text_cls, cross_pooled], dim=-1))
        return {
            "logits": self.head(z),
            "embedding": z,
            "attention": attn_weights,
            "node_mask": node_mask,
        }
