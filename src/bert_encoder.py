from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


def _find_transformer_layers(model):
    # BERT/RoBERTa-like
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return list(model.encoder.layer)
    # DistilBERT-like
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return list(model.transformer.layer)
    # Wrapped base model, e.g. model.bert.encoder.layer
    for name in ["bert", "roberta", "distilbert"]:
        base = getattr(model, name, None)
        if base is not None:
            layers = _find_transformer_layers(base)
            if layers:
                return layers
    return []


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        freeze_encoder: bool = False,
        unfreeze_last_n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden = int(self.transformer.config.hidden_size)

        if freeze_encoder:
            for p in self.transformer.parameters():
                p.requires_grad = False
        elif unfreeze_last_n_layers >= 0:
            # Freeze all first, then selectively unfreeze last N blocks + final norms/pooler.
            for p in self.transformer.parameters():
                p.requires_grad = False
            layers = _find_transformer_layers(self.transformer)
            if layers:
                for layer in layers[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                # Keep embeddings frozen to reduce memory/overfitting on small datasets.
                for name, p in self.transformer.named_parameters():
                    if "pooler" in name or name.endswith("LayerNorm.weight") or name.endswith("LayerNorm.bias"):
                        p.requires_grad = True
            else:
                # Unknown architecture: safer to fine-tune all rather than silently train no text parameters.
                for p in self.transformer.parameters():
                    p.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(hidden, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_hidden = out.last_hidden_state
        cls_hidden = token_hidden[:, 0]
        return self.proj(token_hidden), self.proj(cls_hidden)
