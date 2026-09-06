from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from .audio_features import AudioConfig, extract_node_features, load_audio_fixed
from .factory import build_supervised_model
from .graph_builder import GraphConfig, build_segment_graph
from .labels import load_label_vocab, mask_label_terms
from .utils import load_config, resolve_device


class MusicContextPredictor:
    def __init__(self, config_path: str, checkpoint_path: str, model_type: str = "fusion"):
        self.cfg = load_config(config_path)
        self.device = resolve_device(self.cfg.get("device", "cuda:0"))
        self.labels = load_label_vocab(self.cfg["data"]["label_vocab"])

        self.model = build_supervised_model(self.cfg, model_type, len(self.labels)).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["text"]["model_name"], use_fast=True)

    def predict(self, audio_path: str, text: str, threshold: float = 0.5, top_k: int = 10):
        acfg = AudioConfig(**self.cfg["audio"])
        gcfg = GraphConfig(**self.cfg["graph"])

        y = load_audio_fixed(audio_path, acfg)
        graph = build_segment_graph(extract_node_features(y, acfg), gcfg)
        graph_batch = Batch.from_data_list([graph]).to(self.device)

        if self.cfg["data"].get("mask_label_terms_in_text", False):
            text = mask_label_terms(text, self.labels)

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=int(self.cfg["text"]["max_length"]),
            return_tensors="pt",
        )

        batch = {
            "graph": graph_batch,
            "input_ids": tokens["input_ids"].to(self.device),
            "attention_mask": tokens["attention_mask"].to(self.device),
        }

        with torch.no_grad():
            out = self.model(batch)
            prob = torch.sigmoid(out["logits"])[0].cpu()

        order = torch.argsort(prob, descending=True)
        ranked = [(self.labels[i], float(prob[i])) for i in order[:top_k]]
        selected = [(label, score) for label, score in ranked if score >= threshold]
        return {"selected": selected, "top_k": ranked}
