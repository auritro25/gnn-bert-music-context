from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_label_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, float) and np.isnan(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]

    return [x.strip() for x in text.split(",") if x.strip()]


def build_label_vocab(
    metadata_csv: str | Path,
    labels_col: str,
    split_col: str,
    top_k: int = 50,
    min_count: int = 2,
) -> list[str]:
    df = pd.read_csv(metadata_csv)
    train_df = df[df[split_col].astype(str).str.lower() == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("No training rows found. The label vocabulary must be built from train only.")

    counts = Counter()
    for value in train_df[labels_col].tolist():
        counts.update(parse_label_list(value))

    labels = [label for label, count in counts.most_common() if count >= min_count]
    return labels[:top_k]


def save_label_vocab(labels: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, indent=2, ensure_ascii=False)


def load_label_vocab(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["labels"]


def encode_multihot(raw_labels, vocab: list[str]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(vocab)}
    y = np.zeros(len(vocab), dtype=np.float32)
    for label in parse_label_list(raw_labels):
        if label in idx:
            y[idx[label]] = 1.0
    return y


def mask_label_terms(text: str, vocab: list[str]) -> str:
    """Mask exact target tag phrases to reduce trivial text->label leakage."""
    out = str(text)
    for label in sorted(vocab, key=len, reverse=True):
        label = label.strip()
        if len(label) < 3:
            continue
        pattern = re.compile(r"(?i)(?<!\w)" + re.escape(label) + r"(?!\w)")
        out = pattern.sub("[MASK]", out)
    return out
