from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .audio_features import AudioConfig, extract_logmel, extract_node_features, load_audio_fixed
from .graph_builder import GraphConfig, build_segment_graph
from .labels import build_label_vocab, save_label_vocab
from .utils import ensure_dir, load_config, set_seed


def resolve_audio_path(audio_root: str | Path, value: str) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else Path(audio_root) / p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Debug: preprocess only first N rows.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    dcfg = cfg["data"]

    graph_dir = ensure_dir(dcfg["graph_dir"])
    mel_dir = ensure_dir(dcfg["mel_dir"])

    vocab = build_label_vocab(
        metadata_csv=dcfg["metadata_csv"],
        labels_col=dcfg["labels_col"],
        split_col=dcfg["split_col"],
        top_k=int(dcfg["top_k_tags"]),
        min_count=int(dcfg["min_tag_count"]),
    )
    if not vocab:
        raise RuntimeError("Training split produced an empty label vocabulary.")
    save_label_vocab(vocab, dcfg["label_vocab"])
    print(f"Saved {len(vocab)} train-only labels -> {dcfg['label_vocab']}")

    acfg = AudioConfig(**cfg["audio"])
    gcfg = GraphConfig(**cfg["graph"])
    df = pd.read_csv(dcfg["metadata_csv"])
    if args.limit:
        df = df.head(args.limit)

    failures = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        track_id = str(row[dcfg["track_id_col"]])
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in track_id)
        audio_path = resolve_audio_path(dcfg["audio_root"], row[dcfg["audio_col"]])

        try:
            y = load_audio_fixed(audio_path, acfg)
            node_features = extract_node_features(y, acfg)
            graph = build_segment_graph(node_features, gcfg)
            graph.track_id = safe_id

            torch.save(graph, graph_dir / f"{safe_id}.pt")
            np.save(mel_dir / f"{safe_id}.npy", extract_logmel(y, acfg))
        except Exception as e:
            failures.append((track_id, str(audio_path), repr(e)))

    print(f"Done. Successful={len(df)-len(failures)}, failed={len(failures)}")
    if failures:
        fail_path = Path("results") / "preprocess_failures.csv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures, columns=["track_id", "audio_path", "error"]).to_csv(fail_path, index=False)
        print(f"Failures saved to {fail_path}")


if __name__ == "__main__":
    main()
