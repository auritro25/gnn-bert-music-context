from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from .labels import encode_multihot, load_label_vocab, mask_label_terms


def _safe_id(x: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(x))


class PairedGraphTextDataset(Dataset):
    def __init__(self, cfg: dict, split: str):
        dcfg = cfg["data"]
        tcfg = cfg["text"]

        df = pd.read_csv(dcfg["metadata_csv"])
        split_col = dcfg["split_col"]
        self.df = df[df[split_col].astype(str).str.lower() == split.lower()].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split}")

        self.dcfg = dcfg
        self.graph_dir = Path(dcfg["graph_dir"])
        self.vocab = load_label_vocab(dcfg["label_vocab"])
        self.tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"], use_fast=True)
        self.max_length = int(tcfg["max_length"])
        self.mask_terms = bool(dcfg.get("mask_label_terms_in_text", False))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_id = _safe_id(row[self.dcfg["track_id_col"]])
        graph_path = self.graph_dir / f"{track_id}.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing graph: {graph_path}. Run src.preprocess first.")

        # PyG Data is a trusted local object produced by this repository.
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)

        text = str(row[self.dcfg["text_col"]])
        if self.mask_terms:
            text = mask_label_terms(text, self.vocab)

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        y = encode_multihot(row[self.dcfg["labels_col"]], self.vocab)

        return {
            "track_id": track_id,
            "graph": graph,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.float32),
            "text": text,
        }


def graph_text_collate(items):
    return {
        "track_id": [x["track_id"] for x in items],
        "graph": Batch.from_data_list([x["graph"] for x in items]),
        "input_ids": torch.stack([x["input_ids"] for x in items]),
        "attention_mask": torch.stack([x["attention_mask"] for x in items]),
        "labels": torch.stack([x["labels"] for x in items]),
        "text": [x["text"] for x in items],
    }


class MelDataset(Dataset):
    def __init__(self, cfg: dict, split: str):
        dcfg = cfg["data"]
        df = pd.read_csv(dcfg["metadata_csv"])
        self.df = df[df[dcfg["split_col"]].astype(str).str.lower() == split.lower()].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split}")

        self.dcfg = dcfg
        self.mel_dir = Path(dcfg["mel_dir"])
        self.vocab = load_label_vocab(dcfg["label_vocab"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_id = _safe_id(row[self.dcfg["track_id_col"]])
        mel_path = self.mel_dir / f"{track_id}.npy"
        if not mel_path.exists():
            raise FileNotFoundError(f"Missing mel file: {mel_path}. Run src.preprocess first.")

        mel = np.load(mel_path).astype(np.float32)
        y = encode_multihot(row[self.dcfg["labels_col"]], self.vocab)
        return {
            "track_id": track_id,
            "mel": torch.tensor(mel[None, ...], dtype=torch.float32),
            "labels": torch.tensor(y, dtype=torch.float32),
        }
