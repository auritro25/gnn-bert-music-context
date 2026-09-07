from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_networkx
from transformers import AutoTokenizer

from .datasets import PairedGraphTextDataset, graph_text_collate
from .factory import build_supervised_model
from .labels import load_label_vocab
from .utils import ensure_dir, load_config, resolve_device


def move_batch(batch, device):
    out = dict(batch)
    out["graph"] = batch["graph"].to(device)
    out["input_ids"] = batch["input_ids"].to(device)
    out["attention_mask"] = batch["attention_mask"].to(device)
    out["labels"] = batch["labels"].to(device)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="results/fusion/best_model.pt")
    parser.add_argument("--num-cases", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "cuda:0"))
    labels = load_label_vocab(cfg["data"]["label_vocab"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model_name"], use_fast=True)

    ds = PairedGraphTextDataset(cfg, "test")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=graph_text_collate)

    model = build_supervised_model(cfg, "fusion", len(labels)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = ensure_dir(Path(cfg["train"]["output_dir"]) / "fusion" / "case_studies")
    summaries = []

    with torch.no_grad():
        for case_idx, raw in enumerate(loader):
            if case_idx >= args.num_cases:
                break

            batch = move_batch(raw, device)
            out = model(batch)
            probs = torch.sigmoid(out["logits"])[0].cpu().numpy()
            top_pred_idx = np.argsort(probs)[::-1][:5]
            predictions = [{"label": labels[i], "probability": float(probs[i])} for i in top_pred_idx]

            target = batch["labels"][0].cpu().numpy()
            true_labels = [labels[i] for i in np.flatnonzero(target > 0.5)]

            # [B, heads, graph_nodes, text_tokens] -> [graph_nodes, text_tokens]
            attn = out["attention"][0].mean(dim=0).cpu().numpy()
            valid_tokens = int(batch["attention_mask"][0].sum().item())
            valid_nodes = int(out["node_mask"][0].sum().item())
            attn = attn[:valid_nodes, :valid_tokens]

            token_ids = batch["input_ids"][0, :valid_tokens].cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # Remove special/padding-like positions for readable alignment.
            keep = [
                i for i, tok in enumerate(tokens)
                if tok not in {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
            ]
            if keep:
                attn_show = attn[:, keep]
                tokens_show = [tokens[i] for i in keep]
            else:
                attn_show, tokens_show = attn, tokens

            # Heatmap: musical segment -> caption tokens.
            plt.figure(figsize=(max(9, len(tokens_show) * 0.28), max(4, valid_nodes * 0.55)))
            plt.imshow(attn_show, aspect="auto")
            plt.xticks(range(len(tokens_show)), tokens_show, rotation=90, fontsize=8)
            plt.yticks(range(valid_nodes), [f"Segment {i}" for i in range(valid_nodes)])
            plt.xlabel("BERT tokens")
            plt.ylabel("Music graph nodes")
            plt.title(f"Cross-attention alignment — {raw['track_id'][0]}")
            plt.tight_layout()
            heatmap_path = out_dir / f"case_{case_idx+1}_cross_attention.png"
            plt.savefig(heatmap_path, dpi=220)
            plt.close()

            # Give each graph node the token receiving its maximum attention.
            top_token_for_node = []
            if attn_show.size:
                for node_i in range(valid_nodes):
                    j = int(np.argmax(attn_show[node_i]))
                    top_token_for_node.append(tokens_show[j])
            else:
                top_token_for_node = ["?"] * valid_nodes

            data = raw["graph"].to_data_list()[0]
            G = to_networkx(data, to_undirected=True)
            node_labels = {i: f"{i}: {top_token_for_node[i]}" for i in range(valid_nodes)}
            plt.figure(figsize=(7, 5))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=1000, font_size=8)
            plt.title("Music graph with strongest aligned caption token")
            plt.tight_layout()
            graph_path = out_dir / f"case_{case_idx+1}_graph_alignment.png"
            plt.savefig(graph_path, dpi=220)
            plt.close()

            summary = {
                "track_id": raw["track_id"][0],
                "caption": raw["text"][0],
                "true_labels": true_labels,
                "top_predictions": predictions,
                "segment_to_token_alignment": {
                    f"segment_{i}": top_token_for_node[i] for i in range(valid_nodes)
                },
                "heatmap_file": heatmap_path.name,
                "graph_file": graph_path.name,
            }
            summaries.append(summary)

    with open(out_dir / "case_studies.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(summaries)} case studies to {out_dir}")


if __name__ == "__main__":
    main()
