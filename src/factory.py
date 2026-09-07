from __future__ import annotations

from .bert_encoder import TextEncoder
from .fusion_model import (
    BERTTagClassifier,
    CrossAttentionFusion,
    EarlyConcatFusion,
    GNNTagClassifier,
)
from .gnn_model import MusicGNNEncoder


def node_feature_dim(cfg: dict) -> int:
    acfg = cfg["audio"]
    return int(acfg["n_mfcc"]) * 2 + int(acfg["n_chroma"])


def build_components(cfg: dict):
    hidden = int(cfg["model"]["hidden_dim"])
    dropout = float(cfg["model"]["dropout"])

    gnn = MusicGNNEncoder(
        in_dim=node_feature_dim(cfg),
        hidden_dim=hidden,
        num_layers=int(cfg["model"]["gnn_layers"]),
        gnn_type=cfg["model"]["gnn_type"],
        dropout=dropout,
    )

    text = TextEncoder(
        model_name=cfg["text"]["model_name"],
        output_dim=hidden,
        freeze_encoder=bool(cfg["text"].get("freeze_encoder", False)),
        unfreeze_last_n_layers=int(cfg["text"].get("unfreeze_last_n_layers", 2)),
        dropout=dropout,
    )
    return gnn, text


def build_supervised_model(cfg: dict, model_type: str, num_labels: int):
    model_type = model_type.lower()
    hidden = int(cfg["model"]["hidden_dim"])
    dropout = float(cfg["model"]["dropout"])

    if model_type == "bert":
        _, text = build_components(cfg)
        return BERTTagClassifier(text, hidden, num_labels, dropout)

    if model_type == "gnn":
        gnn, _ = build_components(cfg)
        return GNNTagClassifier(gnn, hidden, num_labels, dropout)

    if model_type == "concat":
        gnn, text = build_components(cfg)
        return EarlyConcatFusion(gnn, text, hidden, num_labels, dropout)

    if model_type == "fusion":
        gnn, text = build_components(cfg)
        return CrossAttentionFusion(
            gnn,
            text,
            hidden,
            num_labels,
            num_heads=int(cfg["model"]["fusion_heads"]),
            dropout=dropout,
        )

    raise ValueError("model_type must be one of: bert, gnn, concat, fusion")
